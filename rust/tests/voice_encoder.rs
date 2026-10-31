//! Independent deployed VoiceEncoder proof. Exported checkpoint/fixtures stay outside Git.
use chatterbox_rs::{npy::F32Npy, voice_encoder::T3VoiceEncoder};
use serde_json::Value;
use sha2::{Digest, Sha256};
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
#[ignore = "requires isolated deployed VoiceEncoder export"]
fn voice_encoder_replay_deployed_parity() {
    let directory = std::env::var("CHATTERBOX_VOICE_ENCODER_ASSETS").expect("external assets");
    let root = Path::new(&directory);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let digest = oracle["manifest_sha256"].as_str().unwrap();
    let model = T3VoiceEncoder::open(&root.join("voice-encoder.json"), digest).unwrap();
    let mut first = None;
    for case in oracle["cases"].as_array().unwrap() {
        let audio = read(root, &case["audio"]);
        let features = model.features(&audio).unwrap();
        assert_eq!(serde_json::json!(features.trim), case["trim"]);
        let mels = read(root, &case["mels"]);
        assert_eq!(features.mels.shape, mels.shape);
        let mel_error = compare(&features.mels.values, &mels.values, 2e-5, 2e-5);
        // Both supplied oracle mels and native PCM features must reach the same embedding.
        let expected_partials = read(root, &case["partial_embeddings"]);
        let expected = read(root, &case["embedding"]);
        let encoded_mels = model.encode_mels(&mels).unwrap();
        compare(
            &encoded_mels.partial_embeddings.values,
            &expected_partials.values,
            3e-5,
            3e-4,
        );
        compare(&encoded_mels.embedding.values, &expected.values, 3e-5, 3e-4);
        let actual = model.encode_pcm16k(&audio).unwrap();
        assert_eq!(
            actual.partial_embeddings.shape[0],
            case["partials"].as_u64().unwrap() as usize
        );
        let error = compare(&actual.embedding.values, &expected.values, 3e-5, 3e-4);
        eprintln!(
            "{} mel max {mel_error}, embedding max {error}",
            case["name"]
        );
        let cli = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args([
                "voice-encode",
                root.join("voice-encoder.json").to_str().unwrap(),
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
        assert_eq!(receipt["status"], "t3_speaker_encoded");
        let values: Vec<f32> = serde_json::from_value(receipt["embedding"].clone()).unwrap();
        assert_eq!(values, actual.embedding.values);
        if first.is_none() {
            first = Some((audio, values));
        }
    }
    // Invalid public tensor fields are rejected before indexing or native matrix calls.
    for input in [
        F32Npy {
            shape: vec![0],
            values: vec![],
        },
        F32Npy {
            shape: vec![2],
            values: vec![0.0],
        },
        F32Npy {
            shape: vec![1, 1],
            values: vec![0.0],
        },
        F32Npy {
            shape: vec![1],
            values: vec![f32::NAN],
        },
        F32Npy {
            shape: vec![1],
            values: vec![f32::MAX],
        },
    ] {
        assert!(model.encode_pcm16k(&input).is_err());
    }
    for input in [
        F32Npy {
            shape: vec![0, 40],
            values: vec![],
        },
        F32Npy {
            shape: vec![1, 40],
            values: vec![0.0; 39],
        },
        F32Npy {
            shape: vec![1, 40],
            values: vec![f32::INFINITY; 40],
        },
        F32Npy {
            shape: vec![1, 40],
            values: vec![-1.0; 40],
        },
    ] {
        assert!(model.encode_mels(&input).is_err());
    }
    let (audio, expected) = first.unwrap();
    assert_eq!(
        model.encode_pcm16k(&audio).unwrap().embedding.values,
        expected
    );
    assert!(T3VoiceEncoder::open(&root.join("voice-encoder.json"), &"0".repeat(64)).is_err());
}

#[test]
#[ignore = "requires isolated deployed VoiceEncoder export"]
fn voice_encoder_asset_admission() {
    let directory = std::env::var("CHATTERBOX_VOICE_ENCODER_ASSETS").expect("external assets");
    let root = Path::new(&directory);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(root.join("voice-encoder.json")).unwrap()).unwrap();
    let temporary = std::env::temp_dir().join(format!(
        "voice-admission-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&temporary).unwrap();
    // Independent names point at immutable source bytes. No model asset is changed.
    for asset in manifest["tensors"].as_object().unwrap().values() {
        let file = asset["file"].as_str().unwrap();
        std::fs::copy(root.join(file), temporary.join(file)).unwrap();
    }
    let reject = |value: &Value, reason: &str| {
        let bytes = serde_json::to_vec(value).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let path = temporary.join("probe.json");
        std::fs::write(&path, bytes).unwrap();
        let error = match T3VoiceEncoder::open(&path, &digest) {
            Ok(_) => panic!("admitted invalid voice assets"),
            Err(error) => error,
        };
        assert!(error.contains(reason), "{error}");
    };
    for (field, value) in [
        ("schema_version", serde_json::json!(2)),
        ("encoder_kind", serde_json::json!("other")),
        ("source_sha256", serde_json::json!("invalid")),
    ] {
        let mut bad = manifest.clone();
        bad[field] = value;
        reject(&bad, "Unsupported");
    }
    let mut bad = manifest.clone();
    bad["tensors"].as_object_mut().unwrap().remove("proj.bias");
    reject(&bad, "incomplete");
    let mut bad = manifest.clone();
    bad["tensors"]["extra"] = manifest["tensors"]["proj.bias"].clone();
    reject(&bad, "incomplete");
    let first = "lstm.bias_hh_l0";
    let mut bad = manifest.clone();
    bad["tensors"][first] = manifest["tensors"]["mel_basis"].clone();
    reject(&bad, "shape mismatch");
    let mut bad = manifest.clone();
    bad["tensors"][first]["sha256"] = serde_json::json!("0".repeat(64));
    reject(&bad, "checksum");
    let mut bad = manifest.clone();
    bad["tensors"][first]["file"] = serde_json::json!("../outside.npy");
    reject(&bad, "local filename");
    let mut bytes =
        std::fs::read(root.join(manifest["tensors"][first]["file"].as_str().unwrap())).unwrap();
    let end = bytes.len();
    bytes[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(temporary.join("nonfinite.npy"), &bytes).unwrap();
    let mut bad = manifest.clone();
    bad["tensors"][first] = serde_json::json!({"file":"nonfinite.npy",
        "sha256":format!("{:x}",Sha256::digest(&bytes))});
    reject(&bad, "Nonfinite");
    std::fs::remove_dir_all(temporary).unwrap();
}
