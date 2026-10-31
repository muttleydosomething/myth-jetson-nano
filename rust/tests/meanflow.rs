//! Independent meanflow estimator/Euler and immutable-model replay.
use chatterbox_rs::{meanflow::S3GenMeanflow, npy::F32Npy, s3gen::S3GenMelInput};
use serde_json::Value;
use std::{path::Path, process::Command};
fn read(root: &Path, name: &str) -> F32Npy {
    F32Npy::from_bytes(&std::fs::read(root.join(name)).unwrap()).unwrap()
}
fn copy(x: &F32Npy) -> F32Npy {
    F32Npy {
        shape: x.shape.clone(),
        values: x.values.clone(),
    }
}
fn compare(a: &F32Npy, b: &F32Npy) -> f32 {
    assert_eq!(a.shape, b.shape);
    assert_eq!(a.values.len(), b.values.len());
    let mut maximum = 0f32;
    for (&a, &b) in a.values.iter().zip(&b.values) {
        assert!(
            a.is_finite() && (a - b).abs() <= 3e-4 + 3e-4 * b.abs(),
            "{a} != {b}"
        );
        maximum = maximum.max((a - b).abs());
    }
    maximum
}
#[test]
#[ignore = "requires independently captured deployed meanflow assets"]
fn meanflow_replay_deployed_parity() {
    let root = std::env::var("CHATTERBOX_MEANFLOW_ASSETS").expect("oracle assets");
    let root = Path::new(&root);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    // Reuse the independent oracle with either admitted implementation. The
    // graph manifest is separately pinned; expected arrays remain unchanged.
    let graph = std::env::var("CHATTERBOX_MEANFLOW_GRAPH").ok();
    let model_path = graph
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.join("meanflow.json"));
    let graph_sha = std::env::var("CHATTERBOX_MEANFLOW_GRAPH_SHA256").ok();
    let sha = graph_sha
        .as_deref()
        .unwrap_or_else(|| oracle["manifest_sha256"].as_str().unwrap());
    let model = S3GenMeanflow::open(&model_path, sha).unwrap();
    let cases = oracle["cases"].as_array().unwrap();
    let mut first = None;
    let mut tiny = None;
    for case in cases {
        let array = |key: &str| read(root, case["arrays"][key]["file"].as_str().unwrap());
        let mu = array("mu");
        let frames = mu.shape[2];
        let reference_frames = case["reference_frames"].as_u64().unwrap() as usize;
        let input = S3GenMelInput {
            mu,
            mask: array("mask"),
            speaker: array("spks"),
            cond: array("cond"),
            reference_frames,
            generated_frames: frames - reference_frames,
        };
        let noise = array("noise");
        let steps = case["steps"].as_u64().unwrap() as usize;
        let (out, compute) = model.run(&input, &noise, steps).unwrap();
        if graph.is_some() {
            assert_eq!(compute.backend, "fp32-tensorrt");
            assert_eq!(
                compute.host_to_device_bytes,
                ((241 * frames + 80) * 4 + steps * 8) as u64
            );
            assert_eq!(compute.device_to_host_bytes, (80 * frames * 4) as u64);
            assert_eq!(compute.gemm_calls, 0);
            eprintln!(
                "graph {} {}",
                case["name"],
                serde_json::to_string(&compute).unwrap()
            );
        }
        let error = compare(&out.full, &array("output"));
        compare(&out.generated, &array("generated"));
        eprintln!("{} full mel maximum absolute error {error}", case["name"]);
        if case["name"] == "single-frame" {
            tiny = Some(out.full.values.clone());
        }
        // Independently compare each derivative on the oracle's intermediate x.
        let mut x = copy(&noise);
        for (i, derivative) in case["derivatives"].as_array().unwrap().iter().enumerate() {
            let t = i as f32 / steps as f32;
            let r = (i + 1) as f32 / steps as f32;
            let expected = read(root, derivative["file"].as_str().unwrap());
            compare(&model.estimate(&input, &x, t, r).unwrap(), &expected);
            for (x, dx) in x.values.iter_mut().zip(&expected.values) {
                *x += (r - t) * dx;
            }
        }
        if first.is_none() {
            first = Some((input, noise, out.full.values));
            continue;
        }
        // Malformed/rejected calls must not poison retained model state.
        let mut bad = copy(&noise);
        bad.values.pop();
        assert!(model.decode(&input, &bad, steps).is_err());
        bad = copy(&noise);
        bad.values[0] = f32::NAN;
        assert!(model.decode(&input, &bad, steps).is_err());
        assert!(model.decode(&input, &noise, 0).is_err());
        let mut invalid = S3GenMelInput {
            mu: copy(&input.mu),
            cond: copy(&input.cond),
            speaker: copy(&input.speaker),
            mask: copy(&input.mask),
            reference_frames: input.reference_frames,
            generated_frames: input.generated_frames,
        };
        invalid.mask.shape[2] += 6;
        invalid.mask.values.extend([1.; 6]);
        assert!(model.decode(&invalid, &noise, steps).is_err());
        invalid.mask = copy(&input.mask);
        invalid.mask.values[0] = 0.;
        assert!(model.decode(&invalid, &noise, steps).is_err());
        invalid.mask = copy(&input.mask);
        invalid.generated_frames += 1;
        assert!(model.decode(&invalid, &noise, steps).is_err());
        assert!(model.estimate(&input, &noise, 0., f32::NAN).is_err());
    }
    let (input, noise, expected) = first.unwrap();
    assert_eq!(
        model.decode(&input, &noise, 2).unwrap().full.values,
        expected
    );
    // One real CLI invocation checks the proof route, timing and library identity.
    let case = cases
        .iter()
        .find(|c| c["name"] == "single-frame")
        .unwrap_or(&cases[0]);
    let mut args = vec![
        "s3-meanflow".to_string(),
        model_path.display().to_string(),
        sha.to_string(),
    ];
    for key in ["mu", "mask", "spks", "cond", "noise"] {
        args.push(
            root.join(case["arrays"][key]["file"].as_str().unwrap())
                .display()
                .to_string(),
        );
    }
    args.push(case["steps"].to_string());
    args.push(case["reference_frames"].to_string());
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
    assert_eq!(receipt["status"], "meanflow_mel_generated");
    assert_eq!(receipt["reference_preparations"], 0);
    let values: Vec<f32> = serde_json::from_value(receipt["full"]["values"].clone()).unwrap();
    if let Some(tiny) = tiny {
        assert_eq!(values, tiny);
    } else {
        // Non-final-only captures exercise the real corrected mask boundary.
        // CLI and library still consume identical fixed noise and model pins.
        assert_eq!(values, expected);
    }
    assert!(
        receipt["started_unix_ms"].as_u64().unwrap()
            <= receipt["finished_unix_ms"].as_u64().unwrap()
    );
    assert!(receipt["duration_ms"].is_u64());
    compare(
        &F32Npy {
            shape: read(root, case["arrays"]["output"]["file"].as_str().unwrap()).shape,
            values,
        },
        &read(root, case["arrays"]["output"]["file"].as_str().unwrap()),
    );
    assert!(S3GenMeanflow::open(&model_path, &"0".repeat(64)).is_err());
}

#[test]
#[ignore = "requires retained deployed meanflow assets"]
fn native_random_meanflow_consumption() {
    use chatterbox_rs::sampling::s3gen_meanflow_noise;
    let root = std::env::var("CHATTERBOX_MEANFLOW_ASSETS").expect("oracle assets");
    let root = Path::new(&root);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let sha = oracle["manifest_sha256"].as_str().unwrap();
    let model = S3GenMeanflow::open(&root.join("meanflow.json"), sha).unwrap();
    let case = &oracle["cases"][3];
    let array = |key: &str| read(root, case["arrays"][key]["file"].as_str().unwrap());
    let mu = array("mu");
    let frames = mu.shape[2];
    let reference_frames = case["reference_frames"].as_u64().unwrap() as usize;
    let input = S3GenMelInput {
        mu,
        mask: array("mask"),
        speaker: array("spks"),
        cond: array("cond"),
        reference_frames,
        generated_frames: frames - reference_frames,
    };
    let noise = s3gen_meanflow_noise(frames, 42).unwrap();
    let steps = case["steps"].as_u64().unwrap() as usize;
    let first = model.decode(&input, &noise, steps).unwrap();
    model
        .decode(&input, &s3gen_meanflow_noise(frames, 43).unwrap(), steps)
        .unwrap();
    assert_eq!(
        first.full.values,
        model.decode(&input, &noise, steps).unwrap().full.values
    );
    let mut args = vec![
        "s3-meanflow".to_string(),
        root.join("meanflow.json").display().to_string(),
        sha.into(),
    ];
    for key in ["mu", "mask", "spks", "cond"] {
        args.push(
            root.join(case["arrays"][key]["file"].as_str().unwrap())
                .display()
                .to_string(),
        );
    }
    args.extend([
        "--seed".into(),
        "42".into(),
        steps.to_string(),
        reference_frames.to_string(),
    ]);
    let run = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let receipt: Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(receipt["random_seed"], 42);
    let actual: Vec<f32> = serde_json::from_value(receipt["full"]["values"].clone()).unwrap();
    assert_eq!(first.full.values, actual);
}
