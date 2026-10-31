use chatterbox_rs::{
    generation::{self, Decoder, BOS, EOS, VOCABULARY},
    sampling::{sample, softmax, Sampling},
    Result,
};
use std::collections::VecDeque;

struct Replay {
    sequence: Vec<Vec<f32>>,
    queue: VecDeque<Vec<f32>>,
    resets: usize,
    decoded: Vec<u32>,
    fail_at: Option<usize>,
}
fn one_hot(token: u32) -> Vec<f32> {
    let mut values = vec![f32::NEG_INFINITY; VOCABULARY];
    values[token as usize] = 0.0;
    values
}
impl Replay {
    fn new(tokens: &[u32]) -> Self {
        Self {
            sequence: tokens.iter().map(|&t| one_hot(t)).collect(),
            queue: VecDeque::new(),
            resets: 0,
            decoded: vec![],
            fail_at: None,
        }
    }
}
impl Decoder for Replay {
    fn reset(&mut self) -> Result<()> {
        self.resets += 1;
        self.queue = self.sequence.clone().into();
        self.decoded.clear();
        Ok(())
    }
    fn prefill(&mut self) -> Result<Vec<f32>> {
        self.queue.pop_front().ok_or("Empty replay".into())
    }
    fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        self.decoded.push(token);
        if self.fail_at == Some(self.decoded.len()) {
            return Err("Injected decode failure".into());
        }
        self.queue.pop_front().ok_or("Empty replay".into())
    }
}
fn unfiltered() -> Sampling {
    Sampling {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    }
}

#[test]
fn upstream_initial_eos_does_not_stop_decode_and_final_eos_is_stripped() {
    let mut backend = Replay::new(&[EOS, 7, EOS]);
    let output = generation::generate(&mut backend, &unfiltered(), 10, 42, || false).unwrap();
    assert_eq!(output.tokens, [EOS, 7]);
    assert!(output.reached_eos);
    assert_eq!(backend.resets, 2);
}
#[test]
fn zero_limit_still_draws_prefill() {
    for token in [7, EOS] {
        let mut backend = Replay::new(&[token]);
        let output = generation::generate(&mut backend, &unfiltered(), 0, 42, || false).unwrap();
        assert_eq!(
            output.tokens,
            if token == EOS { vec![] } else { vec![token] }
        );
        assert_eq!(output.reached_eos, token == EOS);
    }
}
#[test]
fn limit_is_one_prefill_plus_requested_decodes() {
    let mut backend = Replay::new(&[2, 3, 4, 5]);
    let output = generation::generate(&mut backend, &unfiltered(), 2, 42, || false).unwrap();
    assert_eq!(output.tokens, [2, 3, 4]);
    assert!(!output.reached_eos);
}
#[test]
fn repeated_requests_and_failure_do_not_retain_cache_or_history() {
    let mut backend = Replay::new(&[2, 3, EOS]);
    let a = generation::generate(&mut backend, &unfiltered(), 10, 7, || false).unwrap();
    let b = generation::generate(&mut backend, &unfiltered(), 10, 7, || false).unwrap();
    assert_eq!(a, b);
    assert_eq!(backend.resets, 4);
    backend.fail_at = Some(1);
    assert!(generation::generate(&mut backend, &unfiltered(), 10, 7, || false).is_err());
    assert_eq!(backend.resets, 6);
    backend.fail_at = None;
    assert_eq!(
        generation::generate(&mut backend, &unfiltered(), 10, 7, || false).unwrap(),
        a
    );
}
#[test]
fn cancellation_resets_the_session() {
    let mut backend = Replay::new(&[2, 3, EOS]);
    let mut polls = 0;
    let result = generation::generate(&mut backend, &unfiltered(), 10, 42, || {
        polls += 1;
        polls == 2
    });
    assert!(result.unwrap_err().contains("cancelled"));
    assert_eq!(backend.resets, 2);
}
#[test]
fn first_processor_history_contains_bos_then_only_generated_tokens() {
    // A penalty on BOS changes the first draw, while generated history replaces
    // BOS on later calls. The same deterministic variate exposes the difference.
    let config = Sampling {
        repetition_penalty: 2.0,
        ..unfiltered()
    };
    let mut logits = vec![-1000.0; VOCABULARY];
    logits[BOS as usize] = 2.0;
    logits[3] = 1.5;
    let processed = config.process(&logits, &[BOS]).unwrap();
    assert_eq!(processed[BOS as usize], 1.0);
    assert_eq!(config.process(&logits, &[3]).unwrap()[BOS as usize], 2.0);
}
#[test]
fn processor_order_keeps_top_k_ties_and_penalizes_each_history_id_once() {
    let config = Sampling {
        temperature: 0.5,
        top_k: 2,
        top_p: 1.0,
        repetition_penalty: 2.0,
    };
    let result = config.process(&[2.0, 1.0, 1.0, -1.0], &[0, 0, 1]).unwrap();
    assert_eq!(result, [2.0, 1.0, 2.0, f32::NEG_INFINITY]);
    let negatives = Sampling {
        repetition_penalty: 2.0,
        ..unfiltered()
    }
    .process(&[-2.0, 2.0], &[0, 1])
    .unwrap();
    assert_eq!(negatives, [-4.0, 1.0]);
}
#[test]
fn nucleus_runs_before_repetition_and_always_keeps_one() {
    let config = Sampling {
        top_p: 0.8,
        repetition_penalty: 100.0,
        ..unfiltered()
    };
    let result = config.process(&[3.0, 2.0, 1.0], &[0]).unwrap();
    assert_eq!(result[0], 0.03);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], f32::NEG_INFINITY);
    let result = Sampling {
        top_p: 0.0,
        ..unfiltered()
    }
    .process(&[1.0, 2.0, 3.0], &[])
    .unwrap();
    assert_eq!(result, [f32::NEG_INFINITY, f32::NEG_INFINITY, 3.0]);
}
#[test]
fn masked_decode_stops_without_fabricating_an_eos() {
    let mut backend = Replay::new(&[2]);
    backend.sequence.push(vec![f32::NEG_INFINITY; VOCABULARY]);
    let output = generation::generate(&mut backend, &unfiltered(), 10, 42, || false).unwrap();
    assert_eq!(output.tokens, [2]);
    assert!(!output.reached_eos);
}
#[test]
fn invalid_distributions_fail_and_cdf_never_selects_masked_ids() {
    assert!(softmax(&[f32::NEG_INFINITY]).is_err());
    assert!(softmax(&[f32::NAN, 1.0]).is_err());
    assert_eq!(sample(&[0.0, 0.5, 0.0, 0.5], 0.0).unwrap(), 1);
    assert_eq!(sample(&[0.0, 0.5, 0.0, 0.5], 0.5).unwrap(), 3);
    assert!(sample(&[1.0], 1.0).is_err());
    assert!(Sampling {
        repetition_penalty: 0.0,
        ..unfiltered()
    }
    .validate()
    .is_err());
}
