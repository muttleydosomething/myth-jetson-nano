//! Independent deployed-encoder comparisons; assets never enter Git.
use chatterbox_rs::{conformer::S3GenConformer, npy::F32Npy};
use serde_json::Value;
use std::{path::Path, process::Command};

fn compare(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0.0_f32;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-4 + 2e-4 * b.abs(),
            "{a} != {b}"
        );
        max = max.max((a - b).abs());
    }
    max
}

#[test]
#[ignore = "requires CPU-only deployed Conformer export"]
fn conformer_replay_deployed_parity() {
    let root = std::env::var("CHATTERBOX_CONFORMER_ASSETS").expect("external oracle assets");
    let root = Path::new(&root);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let model = S3GenConformer::open(
        &root.join("conformer.json"),
        oracle["manifest_sha256"].as_str().unwrap(),
    )
    .unwrap();
    let mut first = None;
    for case in oracle["cases"].as_array().unwrap() {
        let input =
            F32Npy::from_bytes(&std::fs::read(root.join(case["input"].as_str().unwrap())).unwrap())
                .unwrap();
        let expected = F32Npy::from_bytes(
            &std::fs::read(root.join(case["output"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let actual = model.encode(&input).unwrap();
        assert_eq!(actual.features.shape, expected.shape);
        assert_eq!(actual.mask.shape, [1, 1, input.shape[1] * 2]);
        assert!(actual.mask.values.iter().all(|&v| v == 1.0));
        eprintln!(
            "{} max absolute error {}",
            case["name"],
            compare(&actual.features.values, &expected.values)
        );
        let cli = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args([
                "s3-encode",
                root.join("conformer.json").to_str().unwrap(),
                oracle["manifest_sha256"].as_str().unwrap(),
                root.join(case["input"].as_str().unwrap()).to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            cli.status.success(),
            "{}",
            String::from_utf8_lossy(&cli.stderr)
        );
        let receipt: Value = serde_json::from_slice(&cli.stdout).unwrap();
        assert_eq!(receipt["status"], "conformer_encoded");
        let values: Vec<f32> = serde_json::from_value(receipt["features"].clone()).unwrap();
        assert_eq!(values, actual.features.values);
        assert_eq!(receipt["reference_preparations"], 0);
        if first.is_none() {
            first = Some((input, actual.features.values));
        }
    }
    // Reuse after all other lengths: no request cache is retained by the model.
    let (input, values) = first.unwrap();
    assert_eq!(model.encode(&input).unwrap().features.values, values);
    assert!(model
        .encode(&F32Npy {
            shape: vec![1, 0, 512],
            values: vec![]
        })
        .is_err());
    assert!(model
        .encode(&F32Npy {
            shape: vec![2, 1, 512],
            values: vec![0.0; 1024]
        })
        .is_err());
    assert!(model
        .encode(&F32Npy {
            shape: vec![1, 1, 512],
            values: vec![f32::NAN; 512]
        })
        .is_err());
    assert!(S3GenConformer::open(&root.join("conformer.json"), &"0".repeat(64)).is_err());
}

#[test]
#[ignore = "requires deployed strict-FP32 Conformer graph and retained oracle"]
fn conformer_graph_deployed_parity() {
    use std::time::Instant;
    let root = std::env::var("CHATTERBOX_CONFORMER_ASSETS").unwrap();
    let root = Path::new(&root);
    let path = std::env::var("CHATTERBOX_CONFORMER_GRAPH").unwrap();
    let path = Path::new(&path);
    let bytes = std::fs::read(path).unwrap();
    use sha2::{Digest, Sha256};
    let pin = format!("{:x}", Sha256::digest(&bytes));
    let model = S3GenConformer::open(path, &pin).unwrap();
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let mut retained = None;
    for case in oracle["cases"].as_array().unwrap() {
        let input =
            F32Npy::from_bytes(&std::fs::read(root.join(case["input"].as_str().unwrap())).unwrap())
                .unwrap();
        let expected = F32Npy::from_bytes(
            &std::fs::read(root.join(case["output"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let actual = model.encode(&input).unwrap();
        assert_eq!(actual.features.shape, expected.shape);
        assert_eq!(actual.mask.shape, [1, 1, 2 * input.shape[1]]);
        assert!(actual.mask.values.iter().all(|&v| v == 1.));
        eprintln!(
            "{} max absolute error {}",
            case["name"],
            compare(&actual.features.values, &expected.values)
        );
        if retained.is_none() {
            retained = Some((input, actual.features.values));
        }
    }
    let (input, expected) = retained.unwrap();
    let mut times = Vec::new();
    for _ in 0..5 {
        let began = Instant::now();
        let actual = model.encode(&input).unwrap();
        times.push(began.elapsed().as_secs_f64() * 1000.);
        assert_eq!(actual.features.values, expected);
    }
    // Different odd/capacity lengths share the exact context and workspaces.
    for frames in [3, 7, 850, 1] {
        let probe = F32Npy {
            shape: vec![1, frames, 512],
            values: (0..frames * 512)
                .map(|i| input.values[i % input.values.len()])
                .collect(),
        };
        let actual = model.encode(&probe).unwrap();
        assert_eq!(actual.features.shape, [1, frames * 2, 512]);
        assert!(actual.features.values.iter().all(|v| v.is_finite()));
        assert_eq!(actual.mask.values, vec![1.; 2 * frames]);
    }
    assert_eq!(model.encode(&input).unwrap().features.values, expected);
    for probe in [
        F32Npy {
            shape: vec![1, 0, 512],
            values: vec![],
        },
        F32Npy {
            shape: vec![1, 851, 512],
            values: vec![0.; 851 * 512],
        },
        F32Npy {
            shape: vec![2, 1, 512],
            values: vec![0.; 1024],
        },
        F32Npy {
            shape: vec![1, 1, 512],
            values: vec![f32::NAN; 512],
        },
    ] {
        assert!(model.encode(&probe).is_err());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "resident Conformer five warm ms {:?}; median {}",
        times, times[2]
    );
}
