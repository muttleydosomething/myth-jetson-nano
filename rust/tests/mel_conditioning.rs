//! Independent deployed decoder-call boundary and retained-state replay.
use chatterbox_rs::{
    conformer::EncoderOutput,
    npy::F32Npy,
    s3gen::{S3GenFlowAssets, S3GenMelAssets, S3GenReference},
};
use serde_json::Value;
use std::{path::Path, process::Command};
fn read(root: &Path, name: &str) -> F32Npy {
    F32Npy::from_bytes(&std::fs::read(root.join(name)).unwrap()).unwrap()
}
fn compare(actual: &F32Npy, expected: &F32Npy, exact: bool) -> f32 {
    assert_eq!(actual.shape, expected.shape);
    assert_eq!(actual.values.len(), expected.values.len());
    let mut maximum = 0.0_f32;
    for (&a, &b) in actual.values.iter().zip(&expected.values) {
        assert!(a.is_finite());
        if exact {
            assert_eq!(a.to_bits(), b.to_bits());
        } else {
            assert!((a - b).abs() <= 2e-6 + 1e-5 * b.abs(), "{a} != {b}");
        }
        maximum = maximum.max((a - b).abs());
    }
    maximum
}
#[test]
#[ignore = "requires isolated deployed post-encoder export"]
fn mel_conditioning_replay_deployed_parity() {
    let root = std::env::var("CHATTERBOX_MEL_ASSETS").expect("external oracle assets");
    let root = Path::new(&root);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let model = S3GenMelAssets::open(
        &root.join("mel.json"),
        oracle["manifest_sha256"].as_str().unwrap(),
    )
    .unwrap();
    let flow = S3GenFlowAssets::open(
        &root.join("flow.json"),
        oracle["flow_sha256"].as_str().unwrap(),
    )
    .unwrap();
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(
        oracle["upstream_mask_correction"], true,
        "Non-final execution requires the explicitly corrected deployed oracle"
    );
    // Retain both voices and their prepared speaker projections for the whole
    // sequence, including replay after the intervening shorter requests.
    let open_reference = |case: &Value| {
        S3GenReference::open(
            &root.join(case["reference"].as_str().unwrap()),
            case["reference_sha256"].as_str().unwrap(),
        )
        .unwrap()
    };
    let retained = open_reference(&cases[0]);
    let even = open_reference(cases.last().unwrap());
    let retained = flow.prepare(&retained).unwrap();
    let even = flow.prepare(&even).unwrap();
    let mut first = None;
    for case in cases {
        let prepared = if case["reference"] == "explicit-length-s3gen.json" {
            &even
        } else {
            &retained
        };
        let ids: Vec<i64> = serde_json::from_slice(
            &std::fs::read(root.join(case["generated"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let input = prepared.assemble(&ids).unwrap();
        let features = read(root, case["input"].as_str().unwrap());
        let frames = features.shape[1];
        let encoded = EncoderOutput {
            features,
            mask: F32Npy {
                shape: vec![1, 1, frames],
                values: vec![1.0; frames],
            },
        };
        let finalize = case["finalize"].as_bool().unwrap();
        let output = model.condition(&input, &encoded, finalize).unwrap();
        assert_eq!(
            output.reference_frames,
            case["reference_frames"].as_u64().unwrap() as usize
        );
        assert_eq!(
            output.generated_frames,
            case["generated_frames"].as_u64().unwrap() as usize
        );
        for (name, tensor, exact) in [
            ("mu", &output.mu, false),
            ("cond", &output.cond, true),
            ("mask", &output.mask, true),
            ("spks", &output.speaker, false),
        ] {
            let error = compare(
                tensor,
                &read(root, case["tensors"][name]["file"].as_str().unwrap()),
                exact,
            );
            eprintln!("{} {name} max absolute error {error}", case["name"]);
        }
        let args = vec![
            "s3-mel".to_string(),
            root.join("mel.json").display().to_string(),
            oracle["manifest_sha256"].as_str().unwrap().to_string(),
            root.join(case["reference"].as_str().unwrap())
                .display()
                .to_string(),
            case["reference_sha256"].as_str().unwrap().to_string(),
            root.join("flow.json").display().to_string(),
            oracle["flow_sha256"].as_str().unwrap().to_string(),
            root.join(case["generated"].as_str().unwrap())
                .display()
                .to_string(),
            root.join(case["input"].as_str().unwrap())
                .display()
                .to_string(),
            finalize.to_string(),
        ];
        let cli = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args(&args)
            .output()
            .unwrap();
        assert!(
            cli.status.success(),
            "{}",
            String::from_utf8_lossy(&cli.stderr)
        );
        let receipt: Value = serde_json::from_slice(&cli.stdout).unwrap();
        assert_eq!(receipt["status"], "mel_conditioned");
        assert_eq!(receipt["reference_preparations"], 0);
        for (name, tensor) in [
            ("mu", &output.mu),
            ("cond", &output.cond),
            ("mask", &output.mask),
            ("speaker", &output.speaker),
        ] {
            let values: Vec<f32> = serde_json::from_value(receipt[name]["values"].clone()).unwrap();
            assert_eq!(values, tensor.values);
            assert_eq!(receipt[name]["shape"], serde_json::json!(tensor.shape));
        }
        // Alternate finalize mode on the same immutable assets, then replay.
        if ids.len() > 4 {
            model.condition(&input, &encoded, !finalize).unwrap();
        }
        assert_eq!(
            model
                .condition(&input, &encoded, finalize)
                .unwrap()
                .mu
                .values,
            output.mu.values
        );
        if first.is_none() {
            first = Some((ids.clone(), output.mu.values));
        }
        // Short/truncated, padded and nonfinite external encoder boundaries fail.
        let mut bad = EncoderOutput {
            features: F32Npy {
                shape: encoded.features.shape.clone(),
                values: encoded.features.values.clone(),
            },
            mask: F32Npy {
                shape: encoded.mask.shape.clone(),
                values: encoded.mask.values.clone(),
            },
        };
        bad.mask.values[0] = 0.0;
        assert!(model.condition(&input, &bad, finalize).is_err());
        bad.mask.values[0] = 1.0;
        bad.features.shape[1] -= 1;
        assert!(model.condition(&input, &bad, finalize).is_err());
        bad.features.shape[1] += 1;
        bad.features.values.pop();
        assert!(model.condition(&input, &bad, finalize).is_err());
        bad.features.values.push(0.0);
        bad.features.values[0] = f32::NAN;
        assert!(model.condition(&input, &bad, finalize).is_err());
        if ids.len() == 1 {
            assert!(model.condition(&input, &encoded, false).is_err());
        }
    }
    let (ids, values) = first.unwrap();
    let input = retained.assemble(&ids).unwrap();
    let features = read(root, cases[0]["input"].as_str().unwrap());
    let frames = features.shape[1];
    let encoded = EncoderOutput {
        features,
        mask: F32Npy {
            shape: vec![1, 1, frames],
            values: vec![1.0; frames],
        },
    };
    assert_eq!(
        model.condition(&input, &encoded, true).unwrap().mu.values,
        values
    );
    assert!(S3GenMelAssets::open(&root.join("mel.json"), &"0".repeat(64)).is_err());
}
// Synthetic portable-manifest admission requires no external model assets.
#[test]
fn mel_assets_reject_malformed_exports() {
    use sha2::{Digest, Sha256};
    use std::time::{SystemTime, UNIX_EPOCH};
    let root = std::env::temp_dir().join(format!(
        "chatterbox-mel-test-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    fn npy(shape: &[usize], value: f32) -> Vec<u8> {
        let dimensions = shape
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dimensions},), }}");
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for _ in 0..shape.iter().product::<usize>() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
    let asset = |name: &str, bytes: Vec<u8>| {
        std::fs::write(root.join(name), &bytes).unwrap();
        serde_json::json!({"file":name,"sha256":format!("{:x}",Sha256::digest(&bytes))})
    };
    let good = serde_json::json!({"schema_version":1,"mel_kind":"s3gen_post_encoder_fp32",
        "source_sha256":"a".repeat(64),"weight":asset("weight.npy",npy(&[80,512],0.0)),
        "bias":asset("bias.npy",npy(&[80],0.0))});
    let open = |manifest: &Value| {
        let bytes = serde_json::to_vec(manifest).unwrap();
        std::fs::write(root.join("mel.json"), &bytes).unwrap();
        S3GenMelAssets::open(
            &root.join("mel.json"),
            &format!("{:x}", Sha256::digest(&bytes)),
        )
    };
    assert!(open(&good).is_ok());
    for (key, value) in [
        ("schema_version", serde_json::json!(2)),
        ("mel_kind", serde_json::json!("other")),
        ("source_sha256", serde_json::json!("broken")),
        ("extra", serde_json::json!(0)),
    ] {
        let mut bad = good.clone();
        bad[key] = value;
        assert!(open(&bad).is_err());
    }
    let mut bad = good.clone();
    bad.as_object_mut().unwrap().remove("bias");
    assert!(open(&bad).is_err());
    for path in ["../weight.npy", "/weight.npy", "missing.npy"] {
        let mut bad = good.clone();
        bad["weight"]["file"] = serde_json::json!(path);
        assert!(open(&bad).is_err());
    }
    let mut bad = good.clone();
    bad["weight"]["sha256"] = serde_json::json!("0".repeat(64));
    assert!(open(&bad).is_err());
    let mut bad = good.clone();
    bad["bias"] = asset("wrong.npy", npy(&[1, 80], 0.0));
    assert!(open(&bad).is_err());
    bad["bias"] = asset("nan.npy", npy(&[80], f32::NAN));
    assert!(open(&bad).is_err());
    // Runtime source reads no assets after admission; corruption cannot change
    // an already-admitted model. The independent replay covers the same property.
    std::fs::remove_dir_all(root).unwrap();
}
