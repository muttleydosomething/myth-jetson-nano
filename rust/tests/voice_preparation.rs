//! Independent deployed-method gates for S3 analysis and joined native voices.
use chatterbox_rs::{npy::F32Npy, s3gen::S3ReferenceAnalysis, voice_preparation::VoicePreparation};
use serde_json::Value;
use std::{path::Path, process::Command};

#[test]
fn voice_preparation_reference_admission() {
    use chatterbox_rs::s3gen::S3GenReference;
    let make = |shape: Vec<usize>, values: Vec<f32>| F32Npy { shape, values };
    assert!(S3GenReference::from_parts(
        "a".repeat(64),
        vec![1, 2],
        make(vec![1, 5, 80], vec![0.1; 400]),
        None,
        make(vec![1, 192], vec![0.2; 192])
    )
    .is_ok());
    for (shape, values) in [
        (vec![1, 4, 80], vec![0.1; 400]),
        (vec![1, 5, 80], vec![f32::NAN; 400]),
        (vec![1, usize::MAX, 80], vec![]),
    ] {
        assert!(S3GenReference::from_parts(
            "a".repeat(64),
            vec![1, 2],
            make(shape, values),
            None,
            make(vec![1, 192], vec![0.2; 192])
        )
        .is_err());
    }
}

// Independent row markers prove that a newly encoded voice reaches existing
// T3 prompt assembly and S3 flow without preparing any reference a second time.
fn connection(voice: &mut chatterbox_rs::voice_preparation::PreparedVoice) {
    use chatterbox_rs::{conditioning::PromptAssets, npy::Npy, s3gen::S3GenFlowAssets};
    use half::f16;
    use sha2::{Digest, Sha256};
    let dir = std::env::var("CHATTERBOX_VOICE_PROMPT_ASSETS").unwrap();
    let root = Path::new(&dir);
    let path = root.join("retained-assets.json");
    let sha = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
    let prompt = || PromptAssets::open(&path, &sha).unwrap();
    let mut weight = Npy {
        shape: vec![1024, 256],
        halves: vec![0; 1024 * 256],
    };
    weight.halves[0] = f16::ONE.to_bits();
    weight.halves[257] = f16::ONE.to_bits();
    let bias = Npy {
        shape: vec![1024],
        halves: vec![0; 1024],
    };
    let prepared = voice.condition_prompt(prompt(), &weight, &bias).unwrap();
    let text = prepared.tokenize("Ready.").unwrap();
    let assembled = prepared.assemble(&text).unwrap();
    assert_eq!(
        assembled.shape[1],
        1 + voice.t3_tokens.len() + text.input_ids.len() + 1
    );
    assert_eq!(
        assembled.halves[0],
        f16::from_f32(voice.t3_speaker.values[0]).to_bits()
    );
    assert_eq!(
        assembled.halves[1],
        f16::from_f32(voice.t3_speaker.values[1]).to_bits()
    );
    // The remaining speaker row must be zero, not the retained voice prefix.
    assert!(assembled.halves[2..1024].iter().all(|&v| v == 0));
    voice.t3_speaker.shape = vec![256];
    assert!(voice.condition_prompt(prompt(), &weight, &bias).is_err());
    voice.t3_speaker.shape = vec![1, 256];
    let dir = std::env::var("CHATTERBOX_VOICE_FLOW_ASSETS").unwrap();
    let root = Path::new(&dir);
    let path = root.join("flow.json");
    let sha = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
    let flow = S3GenFlowAssets::open(&path, &sha).unwrap();
    let state = flow.prepare(&voice.reference).unwrap();
    let input = state.assemble(&[1, 2, 3]).unwrap();
    assert_eq!(
        &input.token_ids()[..voice.reference.prompt_token_len()],
        voice.reference.prompt_tokens()
    );
    assert_eq!(
        &input.token_ids()[voice.reference.prompt_token_len()..],
        &[1, 2, 3]
    );
    assert_eq!(
        state.reference().prompt_feat().values,
        voice.reference.prompt_feat().values
    );
}

fn array(root: &Path, value: &Value) -> F32Npy {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(root.join(value["file"].as_str().unwrap())).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        value["sha256"].as_str().unwrap()
    );
    F32Npy::from_bytes(&bytes).unwrap()
}
fn compare(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0f32;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= absolute + relative * b.abs(),
            "index {i}: {a} != {b}, error {}",
            (a - b).abs()
        );
        maximum = maximum.max((a - b).abs());
    }
    maximum
}
#[test]
#[ignore = "requires deployed S3 analysis capture"]
fn voice_preparation_analysis_parity() {
    let directory = std::env::var("CHATTERBOX_VOICE_PREPARATION_ASSETS").unwrap();
    let root = Path::new(&directory);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let analysis = S3ReferenceAnalysis::open(
        &root.join("analysis.json"),
        oracle["analysis_manifest_sha256"].as_str().unwrap(),
    )
    .unwrap();
    let mut first = None;
    for case in oracle["analysis_cases"].as_array().unwrap() {
        let pcm = array(root, &case["pcm24"]);
        let resampled = analysis.resample(&pcm).unwrap();
        let mels = analysis.mels(&pcm).unwrap();
        let expected_mels = array(root, &case["mels"]);
        assert_eq!(mels.shape, expected_mels.shape);
        let a = compare(
            &resampled.values,
            &array(root, &case["pcm16"]).values,
            2e-6,
            2e-5,
        );
        // Independent double-FFT diagnosis matches native within 1e-6; the
        // deployed FP32 FFT differs by up to 8.98e-4 near the 1e-5 mel floor.
        // Only log-mel tolerance changes; exact IDs/vector gates stay intact.
        let b = compare(&mels.values, &expected_mels.values, 1e-3, 2e-5);
        eprintln!("analysis {} resample={a} mels={b}", case["name"]);
        if first.is_none() {
            first = Some((pcm, resampled.values, mels.values));
        }
    }
    let (pcm, samples, mels) = first.unwrap();
    assert_eq!(analysis.resample(&pcm).unwrap().values, samples);
    assert_eq!(analysis.mels(&pcm).unwrap().values, mels);
    for pcm in [
        F32Npy {
            shape: vec![0],
            values: vec![],
        },
        F32Npy {
            shape: vec![2],
            values: vec![0.0],
        },
        F32Npy {
            shape: vec![721],
            values: vec![f32::NAN; 721],
        },
    ] {
        assert!(analysis.mels(&pcm).is_err());
        assert!(analysis.resample(&pcm).is_err());
    }
    assert!(analysis
        .mels(&F32Npy {
            shape: vec![720],
            values: vec![0.0; 720]
        })
        .is_err());
    assert!(analysis
        .mels(&F32Npy {
            shape: vec![721],
            values: vec![f32::MAX; 721]
        })
        .is_err());
    assert!(S3ReferenceAnalysis::open(&root.join("analysis.json"), &"0".repeat(64)).is_err());
}
#[test]
#[ignore = "requires deployed full voice capture and native encoder exports"]
fn voice_preparation_deployed_parity() {
    let directory = std::env::var("CHATTERBOX_VOICE_PREPARATION_ASSETS").unwrap();
    let root = Path::new(&directory);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let manifest = root.join("preparation.json");
    let digest = oracle["preparation_sha256"].as_str().unwrap();
    let engine = VoicePreparation::open(&manifest, digest).unwrap();
    let mut first = None;
    for case in oracle["voice_cases"].as_array().unwrap() {
        let recording = root.join(case["audio"].as_str().unwrap());
        let mut result = engine.prepare(&recording, true).unwrap();
        let speaker = compare(
            &result.t3_speaker.values,
            &array(root, &case["t3_speaker"]).values,
            3e-5,
            3e-4,
        );
        assert_eq!(
            result.t3_tokens,
            serde_json::from_value::<Vec<u32>>(case["t3_tokens"].clone()).unwrap()
        );
        assert_eq!(
            result.reference.prompt_tokens(),
            serde_json::from_value::<Vec<i64>>(case["s3_tokens"].clone()).unwrap()
        );
        let mels = array(root, &case["s3_mels"]);
        assert_eq!(result.reference.prompt_feat().shape, mels.shape);
        let mel = compare(
            &result.reference.prompt_feat().values,
            &mels.values,
            1e-3,
            2e-5,
        );
        let embedding = compare(
            &result.reference.embedding().values,
            &array(root, &case["s3_embedding"]).values,
            3e-4,
            3e-4,
        );
        assert_eq!(result.reference.prompt_feat_len(), None);
        assert!(result.t3_tokens.len() <= 375);
        assert_eq!(
            result.reference.prompt_token_len(),
            result.reference.prompt_feat().shape[1] / 2
        );
        eprintln!(
            "voice {} speaker={speaker} mel={mel} embedding={embedding}",
            case["name"]
        );
        // Exercise CLI/library identity once; other cases exercise the same routine.
        if first.is_none() {
            connection(&mut result);
            let cli = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
                .args([
                    "prepare-voice",
                    manifest.to_str().unwrap(),
                    digest,
                    recording.to_str().unwrap(),
                ])
                .output()
                .unwrap();
            assert!(
                cli.status.success(),
                "{}",
                String::from_utf8_lossy(&cli.stderr)
            );
            let receipt: Value = serde_json::from_slice(&cli.stdout).unwrap();
            // JSON is a decimal transport; decode back to the declared FP32
            // contract before comparing bits. Value compares parser FP64 tails.
            for (key, expected) in [
                ("t3_speaker", &result.t3_speaker.values),
                ("prompt_feat", &result.reference.prompt_feat().values),
                ("embedding", &result.reference.embedding().values),
            ] {
                let actual: Vec<f32> =
                    serde_json::from_value(receipt["voice"][key].clone()).unwrap();
                assert!(
                    actual == *expected,
                    "CLI/library FP32 tensors differ: {key}"
                );
            }
            let expected = serde_json::to_value(result.receipt()).unwrap();
            for key in [
                "t3_tokens",
                "prompt_tokens",
                "prompt_feat_shape",
                "reference",
            ] {
                assert_eq!(
                    receipt["voice"][key], expected[key],
                    "CLI/library metadata differs: {key}"
                );
            }
            assert!(receipt["finished_unix_ms"].as_u64() >= receipt["started_unix_ms"].as_u64());
            first = Some((recording, serde_json::to_value(result.receipt()).unwrap()));
        }
    }
    let (recording, receipt) = first.unwrap();
    assert!(
        serde_json::to_value(engine.prepare(&recording, true).unwrap().receipt()).unwrap()
            == receipt,
        "Native voice tensors changed on replay"
    );
    assert!(VoicePreparation::open(&manifest, &"0".repeat(64)).is_err());
    assert!(engine.prepare(&root.join("absent.wav"), true).is_err());
}
