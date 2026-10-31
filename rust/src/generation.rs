// Native port of the deployed Turbo loop. Neural computation is delegated to
// a session, while Rust owns history, stop rules, cancellation and cleanup.
use crate::{
    sampling::{sample, Sampling, Uniforms},
    Result,
};

pub const BOS: u32 = 6561;
pub const EOS: u32 = 6562;
pub const VOCABULARY: usize = 6563;

pub trait Decoder {
    /// Discard the previous request's K/V and position, including after failure.
    fn reset(&mut self) -> Result<()>;
    fn prefill(&mut self) -> Result<Vec<f32>>;
    fn decode(&mut self, token: u32) -> Result<Vec<f32>>;
}

#[derive(Debug, PartialEq)]
pub struct Generated {
    pub tokens: Vec<u32>,
    pub reached_eos: bool,
}

/// Preserve upstream's unusual first-token rule: the prefill sample is emitted
/// even when max_gen_len is zero, and initial EOS does not stop the next decode.
/// The final EOS alone is stripped. Callers must explicitly handle a length stop.
pub fn generate<D, F>(
    decoder: &mut D,
    sampling: &Sampling,
    max_gen_len: usize,
    seed: u64,
    mut cancelled: F,
) -> Result<Generated>
where
    D: Decoder,
    F: FnMut() -> bool,
{
    let _trace = crate::trace::span("generation");
    sampling.validate()?;
    decoder.reset()?;
    let result = (|| {
        let mut rng = Uniforms::new(seed);
        let mut draw = |logits: Vec<f32>, history: &[u32]| -> Result<u32> {
            if logits.len() != VOCABULARY {
                return Err(format!(
                    "Expected {VOCABULARY} logits, received {}",
                    logits.len()
                ));
            }
            sample(
                &sampling.probabilities(&logits, history)?,
                rng.next_uniform(),
            )
        };
        if cancelled() {
            return Err("Generation cancelled".into());
        }
        let logits = {
            let _trace = crate::trace::span("token_prefill");
            decoder.prefill()?
        };
        let mut current = {
            let _trace = crate::trace::span("token_sample_initial");
            draw(logits, &[BOS])?
        };
        let mut tokens = vec![current];
        for index in 0..max_gen_len {
            let _trace = crate::trace::iteration("token_iteration", index);
            if cancelled() {
                return Err("Generation cancelled".into());
            }
            let logits = {
                let _trace = crate::trace::span("token_decode");
                decoder.decode(current)?
            };
            let sample_trace = crate::trace::span("token_process_sample");
            let processed = sampling.process(&logits, &tokens)?;
            // The deployed loop stops when every processed decode logit is -inf.
            if processed.iter().all(|x| *x == f32::NEG_INFINITY) {
                break;
            }
            if logits.len() != VOCABULARY {
                return Err(format!(
                    "Expected {VOCABULARY} logits, received {}",
                    logits.len()
                ));
            }
            current = sample(&crate::sampling::softmax(&processed)?, rng.next_uniform())?;
            drop(sample_trace);
            tokens.push(current);
            if current == EOS {
                break;
            }
        }
        let reached_eos = tokens.last() == Some(&EOS);
        if reached_eos {
            tokens.pop();
        }
        Ok(Generated {
            tokens,
            reached_eos,
        })
    })();
    // Cleanup must run for errors and cancellation too; a reset failure cannot
    // be concealed behind a superficially successful token result.
    decoder.reset()?;
    result
}
