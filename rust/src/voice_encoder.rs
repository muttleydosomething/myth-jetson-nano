// VoiceEncoder derived from Resemble AI Chatterbox / Real-Time-Voice-Cloning.
// Ported into Rust by Hailstorm contributors, 2026. SPDX-License-Identifier: MIT
//! Native T3 speaker identity from mono FP32 PCM already at 16 kHz.
//! The pinned default branch trims at 20 dB, computes 40 unscaled power mels,
//! runs overlapping 160-frame partials through three LSTMs, then normalizes.
//! Audio decoding, resampling, loudness normalization and the two S3 encoders
//! belong to the surrounding preparation pipeline, not this encoder.
use crate::{
    conditioning::{admitted_bytes, asset_path, valid_sha, Asset},
    conformer::affine,
    npy::F32Npy,
    Result,
};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

const HIDDEN: usize = 256;
const MELS: usize = 40;
const WINDOW: usize = 160;
// np.round((16000 / 1.3) / 160), the deployed embeds_from_wavs default.
const STEP: usize = 77;
const FFT: usize = 400;
const BINS: usize = FFT / 2 + 1;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    encoder_kind: String,
    source_sha256: String,
    tensors: BTreeMap<String, Asset>,
}

/// Only immutable checkpoint tables and deterministic analysis tables persist.
pub struct T3VoiceEncoder {
    tensors: BTreeMap<String, F32Npy>,
    pub source_sha256: String,
    fourier_real: Vec<f64>,
    fourier_imag: Vec<f64>,
}
pub struct VoiceFeatures {
    pub trim: [usize; 2],
    pub mels: F32Npy,
}
pub struct VoiceEmbedding {
    pub embedding: F32Npy,
    pub partial_embeddings: F32Npy,
}
fn required_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    for layer in 0..3 {
        shapes.insert(
            format!("lstm.weight_ih_l{layer}"),
            vec![4 * HIDDEN, if layer == 0 { MELS } else { HIDDEN }],
        );
        shapes.insert(format!("lstm.weight_hh_l{layer}"), vec![4 * HIDDEN, HIDDEN]);
        for bias in ["ih", "hh"] {
            shapes.insert(format!("lstm.bias_{bias}_l{layer}"), vec![4 * HIDDEN]);
        }
    }
    shapes.insert("proj.weight".into(), vec![HIDDEN, HIDDEN]);
    shapes.insert("proj.bias".into(), vec![HIDDEN]);
    shapes.insert("mel_basis".into(), vec![MELS, BINS]);
    shapes
}

/// Exact deployed coverage rule, with checked arithmetic and no utterance cap.
fn partial_plan(frames: usize) -> Result<(usize, usize)> {
    if frames == 0 {
        return Err("Voice encoder requires at least one mel frame".into());
    }
    let extent = frames.saturating_sub(WINDOW - STEP);
    let mut count = extent / STEP;
    if count == 0 || extent % STEP + WINDOW - STEP >= 128 {
        count = count.checked_add(1).ok_or("Voice partial count overflow")?;
    }
    let target = (count - 1)
        .checked_mul(STEP)
        .and_then(|n| n.checked_add(WINDOW))
        .ok_or("Voice partial length overflow")?;
    Ok((count, target))
}
fn normalize(values: &mut [f32]) -> Result<()> {
    // Keep FP32 at the layer boundary; a wide reduction limits accumulated error.
    let norm = values
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt() as f32;
    if !norm.is_finite() || norm <= 0.0 || values.iter().any(|v| !v.is_finite()) {
        return Err("Voice encoder produced a nonfinite or zero embedding".into());
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}
fn validate_mels(mels: &F32Npy) -> Result<usize> {
    if mels.shape.len() != 2
        || mels.shape[1] != MELS
        || mels.shape[0] == 0
        || mels.shape[0].checked_mul(MELS) != Some(mels.values.len())
        || mels.values.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("Voice encoder requires finite nonnegative [T,40] power mels".into());
    }
    Ok(mels.shape[0])
}

impl T3VoiceEncoder {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(&admitted_bytes(path, expected)?)
            .map_err(|e| format!("Voice encoder manifest: {e}"))?;
        let shapes = required_shapes();
        if manifest.schema_version != 1
            || manifest.encoder_kind != "t3_voice_encoder_fp32"
            || !valid_sha(&manifest.source_sha256)
            || manifest.tensors.keys().ne(shapes.keys())
        {
            return Err("Unsupported or incomplete voice encoder export".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tensors = BTreeMap::new();
        for (name, shape) in shapes {
            let asset = &manifest.tensors[&name];
            let array =
                F32Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)?;
            if array.shape != shape {
                return Err(format!("Voice encoder shape mismatch: {name}"));
            }
            if name == "mel_basis" && array.values.iter().any(|v| *v < 0.0) {
                return Err("Voice mel basis must be nonnegative".into());
            }
            tensors.insert(name, array);
        }
        // A fixed 400-point transform avoids an additional runtime dependency.
        // Tables include the periodic Hann window. Use f64 like librosa's window,
        // then round the complex STFT boundary to FP32 before magnitude/power.
        let mut fourier_real = Vec::with_capacity(BINS * FFT);
        let mut fourier_imag = Vec::with_capacity(BINS * FFT);
        for bin in 0..BINS {
            for sample in 0..FFT {
                let phase = std::f64::consts::TAU * sample as f64 / FFT as f64;
                let window = 0.5 - 0.5 * phase.cos();
                fourier_real.push((phase * bin as f64).cos() * window);
                fourier_imag.push(-(phase * bin as f64).sin() * window);
            }
        }
        Ok(Self {
            tensors,
            source_sha256: manifest.source_sha256,
            fourier_real,
            fourier_imag,
        })
    }

    pub fn features(&self, audio: &F32Npy) -> Result<VoiceFeatures> {
        if audio.shape.len() != 1
            || audio.shape[0] == 0
            || audio.shape[0] != audio.values.len()
            || audio.values.iter().any(|v| !v.is_finite())
        {
            return Err(
                "Voice encoder requires nonempty finite mono FP32 [samples] at 16000 Hz".into(),
            );
        }
        let wav = &audio.values;
        // librosa.effects.trim: centered, zero-padded 2048-sample RMS frames,
        // 512 hop, relative peak reference and amplitude floor 1e-5.
        let mut rms = Vec::with_capacity(1 + wav.len() / 512);
        for frame in 0..=wav.len() / 512 {
            let center = frame * 512;
            let start = center.saturating_sub(1024);
            let end = center.saturating_add(1024).min(wav.len());
            let energy = wav[start..end].iter().map(|&x| x * x).sum::<f32>() / 2048.0;
            if !energy.is_finite() {
                return Err("Voice RMS overflow".into());
            }
            rms.push(energy.sqrt());
        }
        let reference = rms.iter().copied().fold(0.0_f32, f32::max).max(1e-5);
        let active = |&x: &f32| x.max(1e-5) > reference * 0.1;
        let first = rms
            .iter()
            .position(active)
            .ok_or("Voice audio trimmed to empty")?;
        let last = rms.iter().rposition(active).unwrap();
        let trim = [first * 512, ((last + 1) * 512).min(wav.len())];
        let wav = &wav[trim[0]..trim[1]];
        if wav.is_empty() {
            return Err("Voice audio trimmed to empty".into());
        }
        let frames = 1 + wav.len() / 160;
        let size = frames
            .checked_mul(BINS)
            .ok_or("Voice spectrum size overflow")?;
        let mut powers = vec![0.0; size];
        // Reflect-centered analysis also handles a one-sample reference. Keep one
        // frame of workspace; no utterance-sized window or complex matrix.
        let mut samples = [0.0_f64; FFT];
        for frame in 0..frames {
            for (i, sample) in samples.iter_mut().enumerate() {
                let index = (frame * 160) as i128 + i as i128 - (FFT / 2) as i128;
                let reflected = if wav.len() == 1 {
                    0
                } else {
                    let period = 2 * (wav.len() - 1) as i128;
                    let folded = index.rem_euclid(period);
                    if folded < wav.len() as i128 {
                        folded as usize
                    } else {
                        (period - folded) as usize
                    }
                };
                *sample = f64::from(wav[reflected]);
            }
            for bin in 0..BINS {
                let start = bin * FFT;
                let re = samples
                    .iter()
                    .zip(&self.fourier_real[start..start + FFT])
                    .map(|(x, w)| x * w)
                    .sum::<f64>() as f32;
                let im = samples
                    .iter()
                    .zip(&self.fourier_imag[start..start + FFT])
                    .map(|(x, w)| x * w)
                    .sum::<f64>() as f32;
                let magnitude = re.hypot(im);
                powers[frame * BINS + bin] = magnitude * magnitude;
            }
        }
        let values = affine(&powers, &self.tensors["mel_basis"], None);
        let mels = F32Npy {
            shape: vec![frames, MELS],
            values,
        };
        validate_mels(&mels)?;
        Ok(VoiceFeatures { trim, mels })
    }

    /// Pinned single-utterance default (rate 1.3, min coverage 0.8).
    /// Process partials individually to bound recurrent workspace for long audio.
    pub fn encode_mels(&self, mels: &F32Npy) -> Result<VoiceEmbedding> {
        let frames = validate_mels(mels)?;
        let (count, _) = partial_plan(frames)?;
        let capacity = count
            .checked_mul(HIDDEN)
            .ok_or("Voice embedding size overflow")?;
        let mut partial_embeddings = Vec::with_capacity(capacity);
        let mut mean = vec![0.0_f32; HIDDEN];
        for partial in 0..count {
            let start = partial * STEP;
            let mut inputs = vec![0.0_f32; WINDOW * MELS];
            let available = frames.saturating_sub(start).min(WINDOW);
            inputs[..available * MELS]
                .copy_from_slice(&mels.values[start * MELS..(start + available) * MELS]);
            let mut last_hidden = vec![0.0; HIDDEN];
            for layer in 0..3 {
                let p = |key: &str| &self.tensors[&format!("lstm.{key}_l{layer}")];
                let gates = affine(&inputs, p("weight_ih"), Some(p("bias_ih")));
                if gates.iter().any(|v| !v.is_finite()) {
                    return Err("Nonfinite voice LSTM input gates".into());
                }
                // State is reset for each layer/partial/request, never in self.
                let mut hidden = vec![0.0_f32; HIDDEN];
                let mut cell = vec![0.0_f32; HIDDEN];
                let mut outputs = Vec::with_capacity(WINDOW * HIDDEN);
                for time in 0..WINDOW {
                    let recurrent = affine(&hidden, p("weight_hh"), Some(p("bias_hh")));
                    if recurrent.iter().any(|v| !v.is_finite()) {
                        return Err("Nonfinite voice LSTM recurrent gates".into());
                    }
                    let row = &gates[time * 4 * HIDDEN..(time + 1) * 4 * HIDDEN];
                    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
                    // PyTorch ordering is input, forget, cell candidate, output.
                    for j in 0..HIDDEN {
                        let input = sigmoid(row[j] + recurrent[j]);
                        let forget = sigmoid(row[HIDDEN + j] + recurrent[HIDDEN + j]);
                        let candidate = (row[2 * HIDDEN + j] + recurrent[2 * HIDDEN + j]).tanh();
                        let output = sigmoid(row[3 * HIDDEN + j] + recurrent[3 * HIDDEN + j]);
                        cell[j] = forget * cell[j] + input * candidate;
                        hidden[j] = output * cell[j].tanh();
                    }
                    outputs.extend_from_slice(&hidden);
                }
                inputs = outputs;
                last_hidden = hidden;
            }
            let mut embedding = affine(
                &last_hidden,
                &self.tensors["proj.weight"],
                Some(&self.tensors["proj.bias"]),
            );
            if embedding.iter().any(|v| !v.is_finite()) {
                return Err("Nonfinite voice projection".into());
            }
            for v in &mut embedding {
                *v = v.max(0.0);
            }
            normalize(&mut embedding)?;
            for (m, v) in mean.iter_mut().zip(&embedding) {
                *m += v;
            }
            partial_embeddings.extend(embedding);
        }
        for m in &mut mean {
            *m /= count as f32;
        }
        normalize(&mut mean)?;
        Ok(VoiceEmbedding {
            embedding: F32Npy {
                shape: vec![1, HIDDEN],
                values: mean,
            },
            partial_embeddings: F32Npy {
                shape: vec![count, HIDDEN],
                values: partial_embeddings,
            },
        })
    }

    pub fn encode_pcm16k(&self, audio: &F32Npy) -> Result<VoiceEmbedding> {
        self.encode_mels(&self.features(audio)?.mels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_coverage_and_empty_contract() {
        assert!(partial_plan(0).is_err());
        for (frames, expected) in [
            (1, (1, 160)),
            (160, (1, 160)),
            (204, (1, 160)),
            (205, (2, 237)),
            (237, (2, 237)),
            (281, (2, 237)),
            (282, (3, 314)),
        ] {
            assert_eq!(partial_plan(frames).unwrap(), expected);
        }
        assert!(normalize(&mut [0.0; 256]).is_err());
        assert!(normalize(&mut [f32::NAN; 256]).is_err());
    }
}
