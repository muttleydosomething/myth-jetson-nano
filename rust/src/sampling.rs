// Derived from Resemble AI Chatterbox (MIT), T3.inference_turbo, and the
// deployed Hailstorm TensorRT backend. See LICENSE and source-provenance.json.
// Keep this processor order: temperature -> top-k -> top-p -> repetition.
// Moving repetition earlier changes which tokens survive nucleus filtering.
use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 1000,
            top_p: 0.95,
            repetition_penalty: 1.2,
        }
    }
}
impl Sampling {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || !self.top_p.is_finite()
            || !(0.0..=1.0).contains(&self.top_p)
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
        {
            return Err("Invalid sampling configuration".into());
        }
        Ok(())
    }

    pub fn process(&self, logits: &[f32], history: &[u32]) -> Result<Vec<f32>> {
        self.validate()?;
        if logits.is_empty() || logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
            return Err("Logits must be nonempty and contain no NaN or positive infinity".into());
        }
        if history.iter().any(|&id| id as usize >= logits.len()) {
            return Err("History token is outside the logit vocabulary".into());
        }
        let mut result = logits.to_vec();
        // Upstream treats temperature <= 0 as disabling the warper, not argmax.
        if self.temperature > 0.0 && self.temperature != 1.0 {
            for value in &mut result {
                *value /= self.temperature;
            }
        }
        if self.top_k > 0 && self.top_k < result.len() {
            let mut ranked = result.clone();
            ranked.sort_unstable_by(|a, b| b.total_cmp(a));
            let cutoff = ranked[self.top_k - 1];
            // Preserve ties, as Hugging Face's strict-less-than mask does.
            for value in &mut result {
                if *value < cutoff {
                    *value = f32::NEG_INFINITY;
                }
            }
        }
        // A fully masked decode is an explicit stop in the deployed loop.
        // Avoid manufacturing NaNs in softmax before that stop can be observed.
        if self.top_p < 1.0 && result.iter().any(|v| v.is_finite()) {
            let mut order: Vec<usize> = (0..result.len()).collect();
            order.sort_by(|&a, &b| result[a].total_cmp(&result[b]));
            let sorted: Vec<f32> = order.iter().map(|&i| result[i]).collect();
            let probs = softmax(&sorted)?;
            let mut cumulative = 0.0_f32;
            // Ascending HF nucleus removes cumulative <= 1-p. Always retain
            // the final (highest) token, including the p=0 edge case.
            for (rank, (&index, probability)) in order.iter().zip(probs).enumerate() {
                cumulative += probability;
                if rank + 1 < order.len() && cumulative <= 1.0 - self.top_p {
                    result[index] = f32::NEG_INFINITY;
                }
            }
        }
        if self.repetition_penalty != 1.0 {
            // Gather/scatter applies once per distinct token, not per occurrence.
            let mut seen = vec![false; result.len()];
            for &id in history {
                let index = id as usize;
                if !seen[index] {
                    result[index] = if result[index] < 0.0 {
                        result[index] * self.repetition_penalty
                    } else {
                        result[index] / self.repetition_penalty
                    };
                    seen[index] = true;
                }
            }
        }
        if result.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
            return Err("Sampling arithmetic produced nonfinite logits".into());
        }
        Ok(result)
    }

    pub fn probabilities(&self, logits: &[f32], history: &[u32]) -> Result<Vec<f32>> {
        softmax(&self.process(logits, history)?)
    }
}

/// Stable FP32 softmax, including masked -infinity values.
pub fn softmax(logits: &[f32]) -> Result<Vec<f32>> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() || logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
        return Err("No finite probability distribution".into());
    }
    let mut weights: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("No positive probability mass".into());
    }
    for value in &mut weights {
        *value /= sum;
    }
    Ok(weights)
}

/// Explicit inverse-CDF sampling permits shared uniform fixtures across languages.
/// A Rust seed is not claimed to reproduce PyTorch's CUDA random sequence.
pub fn sample(probabilities: &[f32], uniform: f64) -> Result<u32> {
    if !(0.0..1.0).contains(&uniform)
        || probabilities.is_empty()
        || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
    {
        return Err("Invalid probabilities or uniform variate".into());
    }
    let total: f64 = probabilities.iter().map(|&p| f64::from(p)).sum();
    if total <= 0.0 {
        return Err("No positive probability mass".into());
    }
    let target = uniform * total;
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for (index, &p) in probabilities.iter().enumerate() {
        cumulative += f64::from(p);
        if p > 0.0 {
            last_positive = Some(index as u32);
            if target < cumulative {
                return Ok(index as u32);
            }
        }
    }
    last_positive.ok_or_else(|| "No sample available".into())
}

/// Small deterministic PRNG for repeatable native runs; no external RNG service.
pub struct Uniforms(u64);
impl Uniforms {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        x
    }
}

/// Allocate a tensor without introducing an utterance cap. Reject impossible
/// shapes and allocation failures before writing any samples.
fn random_tensor(shape: Vec<usize>) -> Result<crate::npy::F32Npy> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or("Random tensor shape overflows")?;
    if count == 0 || count > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err("Random tensor must have a nonzero representable size".into());
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|e| format!("Random tensor allocation: {e}"))?;
    values.resize(count, 0.);
    Ok(crate::npy::F32Npy { shape, values })
}

// Box-Muller consumes pairs, retaining both results. 1-U is in (0,1], so ln
// never sees zero. This extends the existing native PRNG without altering its
// token-sampling sequence or claiming PyTorch/CUDA seeded-sequence parity.
fn fill_normal(values: &mut [f32], rng: &mut Uniforms) {
    for pair in values.chunks_mut(2) {
        let radius = (-2. * (1. - rng.next_uniform()).ln()).sqrt();
        let angle = std::f64::consts::TAU * rng.next_uniform();
        let (sin, cos) = angle.sin_cos();
        pair[0] = (radius * cos) as f32;
        if pair.len() == 2 {
            pair[1] = (radius * sin) as f32;
        }
    }
}

/// Request-owned N(0,1) input for the complete [1,80,F] meanflow tensor,
/// including reference frames. Callers must not overwrite it with conditioning.
/// Domain separation prevents reuse of HiFT or speech-token random streams.
pub fn s3gen_meanflow_noise(frames: usize, seed: u64) -> Result<crate::npy::F32Npy> {
    let mut noise = random_tensor(vec![1, 80, frames])?;
    let mut rng = Uniforms::new(seed ^ u64::from_le_bytes(*b"meanflow"));
    fill_normal(&mut noise.values, &mut rng);
    Ok(noise)
}

/// Request-owned HiFT excitation [1,9,480F] and uniform radian phases [1,9,1].
/// Separate streams keep phases independent of utterance length. The fundamental
/// phase is supplied too; the existing vocoder deliberately ignores that value.
pub fn s3gen_hift_random(
    frames: usize,
    seed: u64,
) -> Result<(crate::npy::F32Npy, crate::npy::F32Npy)> {
    let samples = frames
        .checked_mul(480)
        .ok_or("HiFT random sample count overflows")?;
    let mut noise = random_tensor(vec![1, 9, samples])?;
    let mut phases = random_tensor(vec![1, 9, 1])?;
    let mut rng = Uniforms::new(seed ^ u64::from_le_bytes(*b"hiftnois"));
    fill_normal(&mut noise.values, &mut rng);
    let mut phase_rng = Uniforms::new(seed ^ u64::from_le_bytes(*b"hiftphas"));
    let pi = std::f32::consts::PI;
    // Rounding a value just below pi to FP32 can otherwise close the open bound.
    let below_pi = f32::from_bits(pi.to_bits() - 1);
    for phase in &mut phases.values {
        *phase = ((2. * phase_rng.next_uniform() - 1.) * std::f64::consts::PI) as f32;
        *phase = phase.min(below_pi);
    }
    Ok((noise, phases))
}

/// Growing channel-major tensors cannot be regenerated from a flat seeded RNG:
/// changing the time stride would change the old coordinates of every channel.
/// Keep independent channel streams, including Box-Muller's unused odd sample.
struct NoiseChannel {
    rng: Uniforms,
    spare: Option<f32>,
    values: Vec<f32>,
}
impl NoiseChannel {
    fn next(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let mut pair = [0.; 2];
        fill_normal(&mut pair, &mut self.rng);
        self.spare = Some(pair[1]);
        pair[0]
    }
}
struct PrefixNoise {
    channels: Vec<NoiseChannel>,
    length: usize,
}
impl PrefixNoise {
    fn new(channels: usize, seed: u64) -> Self {
        // Hash each lane seed. Adding the PRNG increment per channel would
        // merely shift one shared sequence and repeat noise across coordinates.
        let mut seeder = Uniforms::new(seed);
        Self {
            channels: (0..channels)
                .map(|_| NoiseChannel {
                    rng: Uniforms::new(seeder.next_u64()),
                    spare: None,
                    values: Vec::new(),
                })
                .collect(),
            length: 0,
        }
    }
    fn prefix(&mut self, length: usize) -> Result<crate::npy::F32Npy> {
        if length < self.length {
            return Err("Request noise prefix cannot shrink".into());
        }
        // Admit output size and reserve every lane before advancing any RNG.
        let mut output = random_tensor(vec![1, self.channels.len(), length])?;
        for channel in &mut self.channels {
            channel
                .values
                .try_reserve(length - channel.values.len())
                .map_err(|e| format!("Retained request noise allocation: {e}"))?;
        }
        for (index, channel) in self.channels.iter_mut().enumerate() {
            while channel.values.len() < length {
                let value = channel.next();
                channel.values.push(value);
            }
            output.values[index * length..(index + 1) * length].copy_from_slice(&channel.values);
        }
        self.length = length;
        Ok(output)
    }
}

/// One utterance owns its full-prefix meanflow/excitation randomness and phases.
/// Dropping this value after cancellation retires all state. Complete-response
/// callers retain their established random sequence through the functions above.
pub struct RequestNoise {
    meanflow: PrefixNoise,
    hift: PrefixNoise,
    seed: u64,
}
impl RequestNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            meanflow: PrefixNoise::new(80, seed ^ u64::from_le_bytes(*b"meanflow")),
            hift: PrefixNoise::new(9, seed ^ u64::from_le_bytes(*b"hiftnois")),
            seed,
        }
    }
    pub fn meanflow(&mut self, frames: usize) -> Result<crate::npy::F32Npy> {
        self.meanflow.prefix(frames)
    }
    pub fn hift(&mut self, frames: usize) -> Result<(crate::npy::F32Npy, crate::npy::F32Npy)> {
        let samples = frames
            .checked_mul(480)
            .ok_or("HiFT random sample count overflows")?;
        let normal = self.hift.prefix(samples)?;
        // Existing phases are independent of tensor length; share that exact
        // policy without allocating another full excitation tensor.
        let mut phases = random_tensor(vec![1, 9, 1])?;
        let mut rng = Uniforms::new(self.seed ^ u64::from_le_bytes(*b"hiftphas"));
        let pi = std::f32::consts::PI;
        let below_pi = f32::from_bits(pi.to_bits() - 1);
        for phase in &mut phases.values {
            *phase = (((2. * rng.next_uniform() - 1.) * std::f64::consts::PI) as f32).min(below_pi);
        }
        Ok((normal, phases))
    }
}
