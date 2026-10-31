//! Independent deployed librosa / norm_loudness proof; fixtures remain outside Git.
use chatterbox_rs::{audio_preparation::AudioPreparation, npy::F32Npy};
use serde_json::Value;
use std::{path::Path, process::Command};

fn compare(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0f32;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-6 + 2e-5 * b.abs(),
            "sample {i}: {a} != {b}, error {}",
            (a - b).abs()
        );
        max = max.max((a - b).abs());
    }
    max
}
fn array(root: &Path, value: &Value) -> Vec<f32> {
    F32Npy::from_bytes(&std::fs::read(root.join(value["file"].as_str().unwrap())).unwrap())
        .unwrap()
        .values
}
#[test]
#[ignore = "requires independent deployed audio preparation capture and native audio libraries"]
fn audio_preparation_deployed_parity() {
    let directory = std::env::var("CHATTERBOX_AUDIO_PREPARATION_ASSETS").unwrap();
    let root = Path::new(&directory);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    let engine = AudioPreparation::open().unwrap();
    let mut first = None;
    for case in oracle["cases"].as_array().unwrap() {
        let path = root.join(case["audio"].as_str().unwrap());
        let decoded = engine.decode(&path).unwrap();
        assert_eq!(decoded.sample_rate, case["rate"].as_u64().unwrap() as u32);
        assert_eq!(
            decoded.channels,
            case["channels"].as_u64().unwrap() as usize
        );
        compare(&decoded.mono, &array(root, &case["decoded"]));
        let enabled = case["normalise"].as_bool().unwrap();
        let result = engine.prepare(&path, enabled).unwrap();
        let error24 = compare(&result.pcm24, &array(root, &case["pcm24"]));
        let error16 = compare(&result.pcm16, &array(root, &case["pcm16"]));
        if enabled {
            if let Some(expected) = case["loudness"].as_f64() {
                assert!((result.loudness_lufs.unwrap() - expected).abs() < 1e-5);
            } else {
                assert!(result.loudness_lufs.is_none());
            }
        }
        assert_eq!(
            result.decoder_reference().len(),
            result.pcm24.len().min(240000)
        );
        assert_eq!(result.t3_reference().len(), result.pcm16.len().min(240000));
        let mut command = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"));
        command.args(["prepare-audio", path.to_str().unwrap()]);
        if !enabled {
            command.arg("--no-normalise");
        }
        let cli = command.output().unwrap();
        assert!(
            cli.status.success(),
            "{}",
            String::from_utf8_lossy(&cli.stderr)
        );
        let receipt: Value = serde_json::from_slice(&cli.stdout).unwrap();
        let values: Vec<f32> = serde_json::from_value(receipt["pcm16"].clone()).unwrap();
        assert_eq!(values, result.pcm16);
        let values: Vec<f32> = serde_json::from_value(receipt["pcm24"].clone()).unwrap();
        assert_eq!(values, result.pcm24);
        assert!(receipt["finished_unix_ms"].as_u64() >= receipt["started_unix_ms"].as_u64());
        eprintln!("{} max24={error24} max16={error16}", case["name"]);
        if first.is_none() {
            first = Some((path, result.pcm24, result.pcm16));
        }
    }
    let (path, a, b) = first.unwrap();
    let replay = engine.prepare(&path, true).unwrap();
    assert_eq!(a, replay.pcm24);
    assert_eq!(b, replay.pcm16);
    for name in [
        "exact-five.wav",
        "short.wav",
        "nonfinite.wav",
        "invalid.wav",
        "absent.wav",
    ] {
        assert!(engine.prepare(&root.join(name), true).is_err(), "{name}");
    }
}
