//! Actual deployed S3 codec comparisons; synthetic fixtures and weights stay external.
use chatterbox_rs::{npy::F32Npy, s3_tokenizer::S3Tokenizer};
use serde_json::Value;
use std::{path::Path, process::Command};

fn read(root: &Path, asset: &Value) -> F32Npy {
    F32Npy::from_bytes(&std::fs::read(root.join(asset["file"].as_str().unwrap())).unwrap()).unwrap()
}
fn compare(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0_f32;
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            a.is_finite() && (a - b).abs() <= atol + rtol * b.abs(),
            "{a} != {b}"
        );
        max = max.max((a - b).abs());
    }
    max
}

#[test]
#[ignore = "requires isolated deployed S3 tokenizer export"]
fn s3_tokenizer_replay_deployed_parity() {
    let directory = std::env::var("CHATTERBOX_S3_TOKENIZER_ASSETS").expect("external assets");
    let root = Path::new(&directory);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let digest = oracle["manifest_sha256"].as_str().unwrap();
    let model = S3Tokenizer::open(&root.join("s3-tokenizer.json"), digest).unwrap();
    let mut first = None;
    for case in oracle["cases"].as_array().unwrap() {
        let audio = read(root, &case["audio"]);
        let features = model.features(&audio).unwrap();
        let mels = read(root, &case["mels"]);
        assert_eq!(features.shape, mels.shape);
        // A float64 FFT of the same PCM/window differs from the deployed FP32
        // FFT by 5.57e-5 after log scaling in quiet bins of the long fixture.
        // Allow that measured FFT roundoff; hidden/logit bounds remain fixed
        // and the final quantized tokens must still agree exactly.
        let mel_error = compare(&features.values, &mels.values, 8e-5, 2e-5);
        if case.get("hidden").is_some() {
            let encoded = model.encode_segment(&mels).unwrap();
            let hidden = read(root, &case["hidden"]);
            let logits = read(root, &case["logits"]);
            assert_eq!(encoded.hidden.shape, hidden.shape);
            compare(&encoded.hidden.values, &hidden.values, 3e-4, 3e-4);
            compare(&encoded.logits.values, &logits.values, 3e-4, 3e-4);
        }
        let tokens = model.encode_mels(&features).unwrap();
        let expected: Vec<u32> = serde_json::from_value(case["codes"].clone()).unwrap();
        assert_eq!(tokens, expected, "{}", case["name"]);
        let cli = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args([
                "s3-tokenize",
                root.join("s3-tokenizer.json").to_str().unwrap(),
                digest,
                root.join(case["audio"]["file"].as_str().unwrap())
                    .to_str()
                    .unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            cli.status.success(),
            "{}",
            String::from_utf8_lossy(&cli.stderr)
        );
        let receipt: Value = serde_json::from_slice(&cli.stdout).unwrap();
        assert_eq!(receipt["status"], "s3_tokenized");
        assert_eq!(receipt["tokens"], case["codes"]);
        eprintln!(
            "{}: mel max {mel_error}; {} exact tokens",
            case["name"],
            tokens.len()
        );
        if first.is_none() {
            first = Some((audio, tokens));
        }
    }
    let (audio, tokens) = first.unwrap();
    assert_eq!(model.encode_pcm16k(&audio).unwrap(), tokens);
    for length in [0, 1, 200] {
        assert!(model
            .encode_pcm16k(&F32Npy {
                shape: vec![length],
                values: vec![0.0; length]
            })
            .is_err());
    }
    for bad in [
        F32Npy {
            shape: vec![201],
            values: vec![0.0; 200],
        },
        F32Npy {
            shape: vec![1, 201],
            values: vec![0.0; 201],
        },
        F32Npy {
            shape: vec![201],
            values: vec![f32::MAX; 201],
        },
        F32Npy {
            shape: vec![201],
            values: vec![f32::NAN; 201],
        },
    ] {
        assert!(model.encode_pcm16k(&bad).is_err());
    }
    for bad in [
        F32Npy {
            shape: vec![1, 128, 0],
            values: vec![],
        },
        F32Npy {
            shape: vec![1, 127, 1],
            values: vec![0.0; 127],
        },
        F32Npy {
            shape: vec![1, 128, 1],
            values: vec![f32::INFINITY; 128],
        },
    ] {
        assert!(model.encode_mels(&bad).is_err());
    }
}
