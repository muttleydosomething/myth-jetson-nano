//! Deployed CAMPPlus comparisons, with independent synthetic PCM and checkpoints.
use chatterbox_rs::{campplus::CampPlus, npy::F32Npy};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{path::Path, process::Command};

fn read(root: &Path, asset: &Value) -> F32Npy {
    F32Npy::from_bytes(&std::fs::read(root.join(asset["file"].as_str().unwrap())).unwrap()).unwrap()
}
fn compare(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0.0_f32;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= atol + rtol * b.abs(),
            "at {index}: {a} != {b}, error {}",
            (a - b).abs()
        );
        max = max.max((a - b).abs());
    }
    max
}
#[test]
#[ignore = "requires independent deployed CAMPPlus export"]
fn campplus_replay_deployed_parity() {
    let directory = std::env::var("CHATTERBOX_CAMPPLUS_ASSETS").expect("external assets");
    let root = Path::new(&directory);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let digest = oracle["manifest_sha256"].as_str().unwrap();
    let model = CampPlus::open(&root.join("campplus.json"), digest).unwrap();
    let mut first = None;
    for case in oracle["cases"].as_array().unwrap() {
        let audio = read(root, &case["audio"]);
        let expected_features = read(root, &case["features"]);
        let features = model.features(&audio).unwrap();
        assert_eq!(features.shape, expected_features.shape);
        // Independent diagnosis measured up to 2.193450927734375e-4 in log
        // features from FP32 DC-reduction and FFT rounding in quiet bins.
        // Only the feature gate changes; the downstream embedding gate stays fixed.
        let feature_error = compare(&features.values, &expected_features.values, 3e-4, 2e-5);
        let expected = read(root, &case["embedding"]);
        // Isolate the network comparison from the independent PCM feature calculation.
        let encoded_features = model.encode_features(&expected_features).unwrap();
        compare(&encoded_features.values, &expected.values, 3e-4, 3e-4);
        let embedding = model.encode_features(&features).unwrap();
        let error = compare(&embedding.values, &expected.values, 3e-4, 3e-4);
        eprintln!(
            "{} features max {feature_error}, embedding max {error}",
            case["name"]
        );
        let cli = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args([
                "campplus-encode",
                root.join("campplus.json").to_str().unwrap(),
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
        assert_eq!(receipt["status"], "campplus_encoded");
        assert_eq!(receipt["mel_frames"], case["frames"]);
        let cli_values: Vec<f32> = serde_json::from_value(receipt["embedding"].clone()).unwrap();
        assert_eq!(cli_values, embedding.values);
        if first.is_none() {
            first = Some((audio, embedding.values));
        }
    }
    for input in [
        F32Npy {
            shape: vec![0],
            values: vec![],
        },
        F32Npy {
            shape: vec![399],
            values: vec![0.0; 399],
        },
        F32Npy {
            shape: vec![400],
            values: vec![0.0; 400],
        },
        F32Npy {
            shape: vec![560],
            values: vec![0.0; 560],
        },
        F32Npy {
            shape: vec![720],
            values: vec![0.0; 719],
        },
        F32Npy {
            shape: vec![1, 720],
            values: vec![0.0; 720],
        },
        F32Npy {
            shape: vec![720],
            values: vec![f32::NAN; 720],
        },
        F32Npy {
            shape: vec![720],
            values: vec![f32::MAX; 720],
        },
    ] {
        assert!(model.encode_pcm16k(&input).is_err());
    }
    for input in [
        F32Npy {
            shape: vec![3, 80],
            values: vec![f32::INFINITY; 240],
        },
        F32Npy {
            shape: vec![3, 80],
            values: vec![f32::MAX; 240],
        },
    ] {
        assert!(model.encode_features(&input).is_err());
    }
    let (audio, expected) = first.unwrap();
    assert_eq!(model.encode_pcm16k(&audio).unwrap().values, expected);
    assert!(CampPlus::open(&root.join("campplus.json"), &"0".repeat(64)).is_err());
}

#[test]
#[ignore = "requires independent deployed CAMPPlus export"]
fn campplus_asset_admission() {
    let directory = std::env::var("CHATTERBOX_CAMPPLUS_ASSETS").expect("external assets");
    let root = Path::new(&directory);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(root.join("campplus.json")).unwrap()).unwrap();
    let temporary = std::env::temp_dir().join(format!(
        "campplus-admission-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&temporary).unwrap();
    for asset in manifest["tensors"].as_object().unwrap().values() {
        let file = asset["file"].as_str().unwrap();
        std::fs::copy(root.join(file), temporary.join(file)).unwrap();
    }
    let reject = |value: &Value, reason: &str| {
        let bytes = serde_json::to_vec(value).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let path = temporary.join("probe.json");
        std::fs::write(&path, bytes).unwrap();
        let error = match CampPlus::open(&path, &digest) {
            Ok(_) => panic!("admitted invalid CAMPPlus assets"),
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
    bad["tensors"].as_object_mut().unwrap().remove("window");
    reject(&bad, "incomplete");
    let mut bad = manifest.clone();
    bad["tensors"]["extra"] = manifest["tensors"]["window"].clone();
    reject(&bad, "incomplete");
    let first = "head.bn1.bias";
    let mut bad = manifest.clone();
    bad["tensors"][first] = manifest["tensors"]["window"].clone();
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
    bad["tensors"][first] =
        serde_json::json!({"file":"nonfinite.npy","sha256":format!("{:x}",Sha256::digest(&bytes))});
    reject(&bad, "Nonfinite");
    // Negative running variances are invalid even if correctly checksummed.
    let name = "head.bn1.running_var";
    let mut bytes =
        std::fs::read(root.join(manifest["tensors"][name]["file"].as_str().unwrap())).unwrap();
    let end = bytes.len();
    bytes[end - 4..].copy_from_slice(&(-1_f32).to_le_bytes());
    std::fs::write(temporary.join("negative.npy"), &bytes).unwrap();
    let mut bad = manifest.clone();
    bad["tensors"][name] =
        serde_json::json!({"file":"negative.npy","sha256":format!("{:x}",Sha256::digest(&bytes))});
    reject(&bad, "nonnegative");
    std::fs::remove_dir_all(temporary).unwrap();
}
