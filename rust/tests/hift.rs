//! Independent deployed HiFT waveform parity and retained-model replay.
use chatterbox_rs::{hift::S3GenHiFT, npy::F32Npy};
use serde_json::Value;
use std::{path::Path, process::Command};
fn read(root: &Path, asset: &Value) -> F32Npy {
    F32Npy::from_bytes(&std::fs::read(root.join(asset["file"].as_str().unwrap())).unwrap()).unwrap()
}
fn copy(a: &F32Npy) -> F32Npy {
    F32Npy {
        shape: a.shape.clone(),
        values: a.values.clone(),
    }
}
fn compare(a: &F32Npy, b: &F32Npy) -> f32 {
    assert_eq!(a.shape, b.shape);
    assert_eq!(a.values.len(), b.values.len());
    let mut max = 0f32;
    for (&a, &b) in a.values.iter().zip(&b.values) {
        assert!(
            a.is_finite() && (a - b).abs() <= 3e-4 + 3e-4 * b.abs(),
            "{a} != {b}"
        );
        max = max.max((a - b).abs());
    }
    max
}
#[test]
#[ignore = "requires Borg strict-FP32 HiFT graph and independent oracle assets"]
fn hift_graph_deployed_parity() {
    use sha2::{Digest, Sha256};
    let path = std::env::var("CHATTERBOX_HIFT_GRAPH").expect("graph manifest");
    let pin = format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap()));
    let model = S3GenHiFT::open(Path::new(&path), &pin).unwrap();
    let root = std::env::var("CHATTERBOX_HIFT_ASSETS").expect("oracle assets");
    let root = Path::new(&root);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let mut retained = None;
    // Includes one/two/odd frames, different mel content and long-to-short
    // context reuse. All comparisons keep the established numerical tolerance.
    for case in oracle["cases"].as_array().unwrap() {
        let a = |name: &str| read(root, &case["arrays"][name]);
        let (mel, normal, phases) = (a("mel"), a("normal"), a("phases"));
        let out = model.synthesize(&mel, &normal, &phases).unwrap();
        compare(&out.pitch, &a("f0"));
        compare(&out.source, &a("source"));
        let error = compare(&out.waveform, &a("waveform"));
        eprintln!(
            "{}",
            serde_json::json!({"case":case["name"],"frames":mel.shape[2],"waveform_max_absolute":error})
        );
        if case["name"] == "retained-final" {
            retained = Some((mel, normal, phases));
        }
    }
    let (mel, normal, phases) = retained.unwrap();
    let first = model.synthesize(&mel, &normal, &phases).unwrap();
    let mut timings = Vec::new();
    for _ in 0..5 {
        let timer = std::time::Instant::now();
        let next = model.synthesize(&mel, &normal, &phases).unwrap();
        timings.push(timer.elapsed().as_secs_f64() * 1000.);
        assert_eq!(next.waveform.values, first.waveform.values);
    }
    eprintln!(
        "{}",
        serde_json::json!({"fixed_frames":mel.shape[2],"warm_ms":timings})
    );
    let cache_root = std::env::var("CHATTERBOX_HIFT_CACHE_ASSETS").expect("cache oracle assets");
    let cache_root = Path::new(&cache_root);
    let cached: Value =
        serde_json::from_slice(&std::fs::read(cache_root.join("oracle.json")).unwrap()).unwrap();
    for case in cached["cache_cases"].as_array().unwrap() {
        let a = |name: &str| read(cache_root, &case["arrays"][name]);
        let cache = if case["cache_samples"] == 0 {
            F32Npy {
                shape: vec![1, 1, 0],
                values: Vec::new(),
            }
        } else {
            a("cache")
        };
        let out = model
            .synthesize_cached(&a("mel"), &a("normal"), &a("phases"), &cache)
            .unwrap();
        compare(&out.source, &a("source"));
        let error = compare(&out.waveform, &a("waveform"));
        assert_eq!(out.source.values[..cache.values.len()], cache.values);
        eprintln!(
            "{}",
            serde_json::json!({"case":case["name"],"waveform_max_absolute":error})
        );
    }
    // Exercise the full admitted capacity, then return to the original small
    // request. This catches profile and workspace-growth regressions directly.
    let manifest: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let max_frames = manifest["max_frames"].as_u64().unwrap() as usize;
    let full = model
        .synthesize(
            &F32Npy {
                shape: vec![1, 80, max_frames],
                values: vec![0.; 80 * max_frames],
            },
            &F32Npy {
                shape: vec![1, 9, 480 * max_frames],
                values: vec![0.; 9 * 480 * max_frames],
            },
            &F32Npy {
                shape: vec![1, 9, 1],
                values: vec![0.; 9],
            },
        )
        .unwrap();
    assert_eq!(full.waveform.shape, vec![1, 480 * max_frames]);
    assert!(full
        .waveform
        .values
        .iter()
        .all(|v| v.is_finite() && v.abs() <= 0.99));
    eprintln!(
        "{}",
        serde_json::json!({"maximum_frames_exercised":max_frames})
    );
    assert_eq!(
        model
            .synthesize(&mel, &normal, &phases)
            .unwrap()
            .waveform
            .values,
        first.waveform.values
    );
}
#[test]
#[ignore = "requires independently captured deployed HiFT cache assets"]
fn hift_cached_source_deployed_parity() {
    let root = std::env::var("CHATTERBOX_HIFT_CACHE_ASSETS").expect("cache oracle assets");
    let root = Path::new(&root);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let model = S3GenHiFT::open(
        &root.join("hift.json"),
        oracle["manifest_sha256"].as_str().unwrap(),
    )
    .unwrap();
    let cases = oracle["cache_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    let first = &cases[0]["arrays"];
    let (mel, normal, phases) = (
        read(root, &first["mel"]),
        read(root, &first["normal"]),
        read(root, &first["phases"]),
    );
    let original = model.synthesize(&mel, &normal, &phases).unwrap();
    for case in cases {
        let a = &case["arrays"];
        // General tensor admission rejects zero-sized NumPy arrays. The empty
        // cache is an algorithmic sentinel, supplied in memory just as callers
        // supply the default empty prefix to deployed inference.
        let cache = if case["cache_samples"] == 0 {
            F32Npy {
                shape: vec![1, 1, 0],
                values: Vec::new(),
            }
        } else {
            read(root, &a["cache"])
        };
        let output = model
            .synthesize_cached(&mel, &normal, &phases, &cache)
            .unwrap();
        let error = compare(&output.waveform, &read(root, &a["waveform"]));
        compare(&output.source, &read(root, &a["source"]));
        assert_eq!(&output.source.values[..cache.values.len()], &cache.values);
        assert_eq!(output.pitch.values, original.pitch.values);
        assert_eq!(
            &output.source.values[cache.values.len()..],
            &original.source.values[cache.values.len()..]
        );
        if cache.values.is_empty() {
            assert_eq!(output.waveform.values, original.waveform.values);
        }
        eprintln!("{} waveform maximum absolute error {error}", case["name"]);
    }
    // No cache enters immutable model state: uncached replay after every prefix
    // reproduces the first output exactly, including the generated excitation.
    let replay = model.synthesize(&mel, &normal, &phases).unwrap();
    assert_eq!(replay.source.values, original.source.values);
    assert_eq!(replay.waveform.values, original.waveform.values);
    for cache in [
        F32Npy {
            shape: vec![1, 2, 1],
            values: vec![0.; 2],
        },
        F32Npy {
            shape: vec![1, 1, 1],
            values: vec![],
        },
        F32Npy {
            shape: vec![1, 1, 1],
            values: vec![f32::NAN],
        },
        F32Npy {
            shape: vec![1, 1, mel.shape[2] * 480 + 1],
            values: vec![0.; mel.shape[2] * 480 + 1],
        },
    ] {
        assert!(model
            .synthesize_cached(&mel, &normal, &phases, &cache)
            .is_err());
    }
    assert_eq!(
        model
            .synthesize(&mel, &normal, &phases)
            .unwrap()
            .waveform
            .values,
        original.waveform.values
    );
}
#[test]
#[ignore = "requires independently captured deployed HiFT assets"]
fn hift_replay_deployed_parity() {
    let root = std::env::var("CHATTERBOX_HIFT_ASSETS").expect("oracle assets");
    let root = Path::new(&root);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let sha = oracle["manifest_sha256"].as_str().unwrap();
    let model = S3GenHiFT::open(&root.join("hift.json"), sha).unwrap();
    let mut first = None;
    let mut tiny = None;
    for case in oracle["cases"].as_array().unwrap() {
        let a = |name: &str| read(root, &case["arrays"][name]);
        let (mel, noise, phases) = (a("mel"), a("normal"), a("phases"));
        let output = model.synthesize(&mel, &noise, &phases).unwrap();
        compare(&output.pitch, &a("f0"));
        compare(&output.source, &a("source"));
        let error = compare(&output.waveform, &a("waveform"));
        // Fixed deployed source isolates vocoder arithmetic from pitch/source.
        compare(&model.decode(&mel, &a("source")).unwrap(), &a("waveform"));
        compare(
            &model.source_spectrum(&a("source")).unwrap(),
            &a("source_stft"),
        );
        eprintln!("{} waveform maximum absolute error {error}", case["name"]);
        assert_eq!(
            output.waveform.shape,
            vec![1, case["frames"].as_u64().unwrap() as usize * 480]
        );
        assert!(output.waveform.values.iter().all(|x| x.abs() <= 0.99));
        if case["name"] == "single-frame" {
            tiny = Some(output.waveform.values.clone());
        }
        let mut bad = copy(&mel);
        bad.values.pop();
        assert!(model.synthesize(&bad, &noise, &phases).is_err());
        let mut bad = copy(&noise);
        bad.values[0] = f32::NAN;
        assert!(model.synthesize(&mel, &bad, &phases).is_err());
        let mut bad = copy(&phases);
        bad.shape = vec![9];
        assert!(model.synthesize(&mel, &noise, &bad).is_err());
        let mut bad = copy(&mel);
        bad.shape[0] = 2;
        assert!(model.predict_pitch(&bad).is_err());
        assert!(model
            .decode(
                &mel,
                &F32Npy {
                    shape: vec![1, 1, 0],
                    values: vec![]
                }
            )
            .is_err());
        if first.is_none() {
            first = Some((mel, noise, phases, output.waveform.values));
        }
    }
    for case in oracle["source_cases"].as_array().unwrap() {
        let a = |name: &str| read(root, &case["arrays"][name]);
        compare(
            &model.excite(&a("f0"), &a("normal"), &a("phases")).unwrap(),
            &a("source"),
        );
    }
    let (mel, noise, phases, waveform) = first.unwrap();
    assert_eq!(
        model
            .synthesize(&mel, &noise, &phases)
            .unwrap()
            .waveform
            .values,
        waveform
    );
    let case = &oracle["cases"][2];
    let mut args = vec![
        "s3-hift".to_string(),
        root.join("hift.json").display().to_string(),
        sha.to_string(),
    ];
    for key in ["mel", "normal", "phases"] {
        args.push(
            root.join(case["arrays"][key]["file"].as_str().unwrap())
                .display()
                .to_string(),
        );
    }
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
    assert_eq!(receipt["status"], "hift_waveform_generated");
    assert_eq!(receipt["reference_preparations"], 0);
    assert_eq!(receipt["sample_rate"], 24000);
    let actual: Vec<f32> = serde_json::from_value(receipt["waveform"]["values"].clone()).unwrap();
    assert_eq!(actual, tiny.unwrap());
    assert!(
        receipt["started_unix_ms"].as_u64().unwrap()
            <= receipt["finished_unix_ms"].as_u64().unwrap()
    );
    assert!(receipt["duration_ms"].is_u64());
    assert!(S3GenHiFT::open(&root.join("hift.json"), &"0".repeat(64)).is_err());
}

#[test]
#[ignore = "requires retained deployed HiFT assets"]
fn native_random_hift_consumption() {
    use chatterbox_rs::sampling::s3gen_hift_random;
    let root = std::env::var("CHATTERBOX_HIFT_ASSETS").expect("oracle assets");
    let root = Path::new(&root);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let sha = oracle["manifest_sha256"].as_str().unwrap();
    let model = S3GenHiFT::open(&root.join("hift.json"), sha).unwrap();
    let case = &oracle["cases"][3]; // Two frames also admit the final trim/fade.
    let mel = read(root, &case["arrays"]["mel"]);
    let (normal, phases) = s3gen_hift_random(mel.shape[2], 42).unwrap();
    let first = model.synthesize(&mel, &normal, &phases).unwrap().waveform;
    let (other_noise, other_phases) = s3gen_hift_random(mel.shape[2], 43).unwrap();
    let other = model
        .synthesize(&mel, &other_noise, &other_phases)
        .unwrap()
        .waveform;
    assert_ne!(first.values, other.values);
    assert_eq!(
        first.values,
        model
            .synthesize(&mel, &normal, &phases)
            .unwrap()
            .waveform
            .values
    );
    let run = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
        .args([
            "s3-hift",
            root.join("hift.json").to_str().unwrap(),
            sha,
            root.join(case["arrays"]["mel"]["file"].as_str().unwrap())
                .to_str()
                .unwrap(),
            "--seed",
            "42",
        ])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let receipt: Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(receipt["random_seed"], 42);
    let actual: Vec<f32> = serde_json::from_value(receipt["waveform"]["values"].clone()).unwrap();
    assert_eq!(first.values, actual);
    assert!(chatterbox_rs::hift::finish_waveform(&first).is_ok());
}
