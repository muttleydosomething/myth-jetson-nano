//! Final waveform boundary: independent deployed wrapper parity, no model reload.
use chatterbox_rs::{hift::finish_waveform, npy::F32Npy};
use serde_json::Value;
use std::{path::Path, process::Command};

fn read(root: &Path, name: &str) -> F32Npy {
    F32Npy::from_bytes(&std::fs::read(root.join(name)).unwrap()).unwrap()
}

#[test]
fn output_finish_preserves_length_tail_and_input() {
    let input = F32Npy {
        shape: vec![1, 1440],
        values: (0..1440)
            .map(|i| if i % 2 == 0 { 0.75 } else { -0.75 })
            .collect(),
    };
    let original = input.values.clone();
    let output = finish_waveform(&input).unwrap();
    assert_eq!(output.shape, input.shape);
    assert_eq!(output.values.len(), input.values.len());
    assert!(output.values[..480].iter().all(|v| *v == 0.0));
    assert_eq!(output.values[480], 0.0);
    assert_eq!(output.values[959], input.values[959]);
    assert_eq!(output.values[960..], input.values[960..]);
    assert_eq!(input.values, original);
    // Alternating sign makes polarity reversal and erroneous absolute value visible.
    for i in 481..959 {
        assert_eq!(
            output.values[i].is_sign_negative(),
            input.values[i].is_sign_negative()
        );
        assert!(output.values[i].abs() <= input.values[i].abs());
        assert!(output.values[i].abs() >= output.values[i - 1].abs());
    }
    assert_eq!(finish_waveform(&input).unwrap().values, output.values);
}

#[test]
fn output_finish_rejects_incomplete_short_nonfinite_or_batched_input() {
    for shape in [
        vec![960],
        vec![2, 960],
        vec![1, 1, 960],
        vec![1, 959],
        vec![1, 0],
    ] {
        let input = F32Npy {
            values: vec![1.0; shape.iter().product()],
            shape,
        };
        assert!(finish_waveform(&input).is_err());
    }
    let mut input = F32Npy {
        shape: vec![1, 960],
        values: vec![1.0; 960],
    };
    assert!(finish_waveform(&input).is_ok()); // Exact fade-window boundary.
    input.values.pop();
    assert!(finish_waveform(&input).is_err());
    input.values.push(f32::NAN);
    assert!(finish_waveform(&input).is_err());
    input.values[959] = f32::INFINITY;
    assert!(finish_waveform(&input).is_err());
    input.values[959] = f32::NEG_INFINITY;
    assert!(finish_waveform(&input).is_err());
    input.values[959] = 1.0;
    input.values[0] = f32::NAN; // Invalid content cannot be hidden by the silent prefix.
    assert!(finish_waveform(&input).is_err());
}

#[test]
#[ignore = "requires independently executed deployed S3Token2Wav output oracle"]
fn output_finish_deployed_wrapper_parity() {
    let root = std::env::var("CHATTERBOX_OUTPUT_ASSETS").expect("oracle assets");
    let root = Path::new(&root);
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
    assert_eq!(oracle["reference_preparations"], 0);
    assert_eq!(oracle["watermark_applied"], false);
    let mut first = None;
    for case in oracle["cases"].as_array().unwrap() {
        let input = read(root, case["input"].as_str().unwrap());
        let original = input.values.clone();
        let output = finish_waveform(&input).unwrap();
        let expected = read(root, case["output"].as_str().unwrap());
        assert_eq!(output.shape, expected.shape);
        let mut max_error = 0f32;
        for (&actual, &expected) in output.values.iter().zip(&expected.values) {
            max_error = max_error.max((actual - expected).abs());
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 3e-7,
                "{}: {actual} != {expected}",
                case["name"]
            );
        }
        assert_eq!(input.values, original);
        assert!(output.values[960..]
            .iter()
            .zip(&input.values[960..])
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        eprintln!("{} maximum absolute error {max_error}", case["name"]);
        let run = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args([
                "s3-finish",
                root.join(case["input"].as_str().unwrap()).to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let receipt: Value = serde_json::from_slice(&run.stdout).unwrap();
        assert_eq!(receipt["status"], "waveform_finished");
        assert_eq!(receipt["sample_rate"], 24000);
        assert_eq!(receipt["trim_fade_applied"], true);
        assert_eq!(receipt["watermark_applied"], false);
        assert_eq!(receipt["reference_preparations"], 0);
        assert_eq!(
            receipt["waveform"]["shape"],
            serde_json::json!(output.shape)
        );
        let values: Vec<f32> =
            serde_json::from_value(receipt["waveform"]["values"].clone()).unwrap();
        assert_eq!(values, output.values);
        assert!(
            receipt["started_unix_ms"].as_u64().unwrap()
                <= receipt["finished_unix_ms"].as_u64().unwrap()
        );
        assert!(receipt["duration_ms"].is_u64());
        if first.is_none() {
            first = Some((input, output.values));
        }
    }
    let (input, expected) = first.unwrap();
    assert_eq!(finish_waveform(&input).unwrap().values, expected);
    for name in ["single-frame.npy", "short-window.npy"] {
        assert!(finish_waveform(&read(root, name)).is_err());
        let run = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args(["s3-finish", root.join(name).to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!run.status.success());
        assert!(String::from_utf8_lossy(&run.stderr).contains("960"));
    }
}
