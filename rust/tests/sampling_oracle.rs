use chatterbox_rs::sampling::{softmax, Sampling};
use serde::Deserialize;
#[derive(Deserialize)]
struct Oracle {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    logits: Vec<f32>,
    history: Vec<u32>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    processed_logits: Vec<Option<f32>>,
    probabilities: Vec<f32>,
}
#[test]
fn authentic_hugging_face_float32_oracle() {
    let oracle: Oracle = serde_json::from_str(include_str!("sampling-oracle.json")).unwrap();
    assert_eq!(oracle.cases.len(), 8);
    for case in oracle.cases {
        let config = Sampling {
            temperature: case.temperature,
            top_k: case.top_k,
            top_p: case.top_p,
            repetition_penalty: case.repetition_penalty,
        };
        let processed = config.process(&case.logits, &case.history).unwrap();
        let probabilities = softmax(&processed).unwrap();
        for (index, (&actual, expected)) in processed.iter().zip(&case.processed_logits).enumerate()
        {
            match expected {
                None => assert_eq!(actual, f32::NEG_INFINITY, "{} mask {index}", case.name),
                Some(value) => assert!(
                    (actual - value).abs() < 1e-6,
                    "{} logit {index}: {actual} vs {value}",
                    case.name
                ),
            }
        }
        for (index, (&actual, &expected)) in
            probabilities.iter().zip(&case.probabilities).enumerate()
        {
            assert!(
                (actual - expected).abs() < 2e-7,
                "{} probability {index}: {actual} vs {expected}",
                case.name
            );
        }
    }
}

#[test]
fn native_random_contract_and_replay() {
    use chatterbox_rs::sampling::{s3gen_hift_random, s3gen_meanflow_noise};
    let first = s3gen_meanflow_noise(403 + 137, 42).unwrap();
    assert_eq!(first.shape, [1, 80, 540]);
    assert_eq!(first.values.len(), 80 * 540);
    let (normal, phases) = s3gen_hift_random(137, 42).unwrap();
    assert_eq!(normal.shape, [1, 9, 137 * 480]);
    assert_eq!(phases.shape, [1, 9, 1]);
    assert!(phases
        .values
        .iter()
        .all(|&x| x >= -std::f32::consts::PI && x < std::f32::consts::PI));
    assert_ne!(&first.values[..100], &normal.values[..100]);
    // An intervening length/seed cannot advance any other request's stream.
    s3gen_meanflow_noise(1, u64::MAX).unwrap();
    s3gen_hift_random(2, 0).unwrap();
    assert_eq!(first.values, s3gen_meanflow_noise(540, 42).unwrap().values);
    let again = s3gen_hift_random(137, 42).unwrap();
    assert_eq!(normal.values, again.0.values);
    assert_eq!(phases.values, again.1.values);
    assert_ne!(first.values, s3gen_meanflow_noise(540, 43).unwrap().values);
    assert_ne!(normal.values, s3gen_hift_random(137, 43).unwrap().0.values);
    assert_eq!(phases.values, s3gen_hift_random(1, 42).unwrap().1.values);
    for seed in [0, 1, 42, u64::MAX] {
        let values = s3gen_meanflow_noise(401, seed).unwrap().values;
        assert!(values.iter().all(|v| v.is_finite()));
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
        let square_mean =
            values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / values.len() as f64;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!(
            (square_mean - 1.).abs() < 0.06,
            "second moment {square_mean}"
        );
        // A uniform distribution with similar moments must still fail this gate.
        let tails = values.iter().filter(|v| v.abs() > 2.).count() as f64 / values.len() as f64;
        assert!((0.03..0.06).contains(&tails), "normal tails {tails}");
    }
    for frames in [0, usize::MAX, usize::MAX / 80, usize::MAX / 480] {
        assert!(s3gen_meanflow_noise(frames, 0).is_err());
        assert!(s3gen_hift_random(frames, 0).is_err());
    }
}
