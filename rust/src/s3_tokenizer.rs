// Derived from S3Tokenizer v2 (Mddct: Dinghao Zhou), OpenAI Whisper and
// Resemble AI Chatterbox. Ported by Hailstorm contributors, 2026.
// SPDX-License-Identifier: Apache-2.0 AND MIT (see LICENSE).
//! Native FP32 S3 v2/25 Hz codec encoding for one mono 16 kHz utterance.
//! Reuses the existing native matrix/GELU kernels. Weights are immutable;
//! request-owned segments bound attention workspace without limiting audio length.
use crate::{
    conditioning::{admitted_bytes, asset_path, valid_sha, Asset},
    conformer::{affine, norm, product},
    meanflow::gelu,
    npy::F32Npy,
    Result,
};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

const D: usize = 1280;
const HEAD: usize = 64;
const MELS: usize = 128;
const FFT: usize = 400;
const BINS: usize = 201;
const SEGMENT: usize = 3000;
const STRIDE: usize = 2600;
const HALF_OVERLAP_TOKENS: usize = 50;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    encoder_kind: String,
    source_sha256: String,
    tensors: BTreeMap<String, Asset>,
}
pub struct S3Tokenizer {
    tensors: BTreeMap<String, F32Npy>,
    pub source_sha256: String,
    fourier_real: Vec<f64>,
    fourier_imag: Vec<f64>,
}
/// Exposes the continuous boundary so proofs can distinguish arithmetic drift
/// from a discrete FSQ threshold crossing.
pub struct CodecSegment {
    pub hidden: F32Npy,
    pub logits: F32Npy,
    pub tokens: Vec<u32>,
}
fn required_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    for (name, channels) in [("conv1", MELS), ("conv2", D)] {
        shapes.insert(format!("encoder.{name}.weight"), vec![D, channels, 3]);
        shapes.insert(format!("encoder.{name}.bias"), vec![D]);
    }
    for layer in 0..6 {
        let p = format!("encoder.blocks.{layer}");
        for name in ["attn.query", "attn.key", "attn.value", "attn.out"] {
            shapes.insert(format!("{p}.{name}.weight"), vec![D, D]);
            if name != "attn.key" {
                shapes.insert(format!("{p}.{name}.bias"), vec![D]);
            }
        }
        shapes.insert(format!("{p}.attn.fsmn_block.weight"), vec![D, 1, 31]);
        for name in ["attn_ln", "mlp_ln"] {
            for suffix in ["weight", "bias"] {
                shapes.insert(format!("{p}.{name}.{suffix}"), vec![D]);
            }
        }
        for (name, output, input) in [("mlp.0", 4 * D, D), ("mlp.2", D, 4 * D)] {
            shapes.insert(format!("{p}.{name}.weight"), vec![output, input]);
            shapes.insert(format!("{p}.{name}.bias"), vec![output]);
        }
    }
    shapes.insert("quantizer._codebook.project_down.weight".into(), vec![8, D]);
    shapes.insert("quantizer._codebook.project_down.bias".into(), vec![8]);
    shapes.insert("_mel_filters".into(), vec![MELS, BINS]);
    shapes.insert("window".into(), vec![FFT]);
    shapes.insert("rotary_cos".into(), vec![2048, HEAD]);
    shapes.insert("rotary_sin".into(), vec![2048, HEAD]);
    shapes
}
fn finite(values: &[f32], stage: &str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(format!("S3 tokenizer nonfinite {stage}"))
    } else {
        Ok(())
    }
}
fn mel_frames(mels: &F32Npy) -> Result<usize> {
    if mels.shape.len() != 3
        || mels.shape[..2] != [1, MELS]
        || mels.shape[2] == 0
        || mels.shape[2].checked_mul(MELS) != Some(mels.values.len())
    {
        return Err("S3 tokenizer requires finite nonempty [1,128,T] log mels".into());
    }
    finite(&mels.values, "log mels")?;
    Ok(mels.shape[2])
}
/// Torch's ties-to-even FSQ, base-three little-endian digits, 0..6560.
fn quantize(logits: &[f32]) -> Result<Vec<u32>> {
    if logits.len() % 8 != 0 {
        return Err("S3 FSQ requires groups of eight logits".into());
    }
    finite(logits, "FSQ logits")?;
    Ok(logits
        .chunks_exact(8)
        .map(|row| {
            let mut token = 0;
            let mut power = 1;
            for &logit in row {
                let digit = (logit.tanh() * 0.9990000128746033_f32).round_ties_even() as i32 + 1;
                token += digit as u32 * power;
                power *= 3;
            }
            token
        })
        .collect())
}
impl S3Tokenizer {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(&admitted_bytes(path, expected)?)
            .map_err(|e| format!("S3 tokenizer manifest: {e}"))?;
        let shapes = required_shapes();
        if manifest.schema_version != 1
            || manifest.encoder_kind != "s3_tokenizer_v2_25hz_fp32"
            || !valid_sha(&manifest.source_sha256)
            || manifest.tensors.keys().ne(shapes.keys())
        {
            return Err("Unsupported or incomplete S3 tokenizer export".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tensors = BTreeMap::new();
        for (name, shape) in shapes {
            let asset = &manifest.tensors[&name];
            let array =
                F32Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)?;
            if array.shape != shape {
                return Err(format!("S3 tokenizer shape mismatch: {name}"));
            }
            if (name == "_mel_filters" || name == "window") && array.values.iter().any(|v| *v < 0.0)
            {
                return Err(format!("S3 analysis table must be nonnegative: {name}"));
            }
            tensors.insert(name, array);
        }
        // Torch owns an FP32 periodic Hann window. Keep that rounding, then use
        // a wide Fourier accumulation and round real/imaginary outputs to FP32.
        let mut fourier_real = Vec::with_capacity(BINS * FFT);
        let mut fourier_imag = Vec::with_capacity(BINS * FFT);
        for bin in 0..BINS {
            for sample in 0..FFT {
                let phase = std::f64::consts::TAU * (sample * bin) as f64 / FFT as f64;
                fourier_real.push(phase.cos());
                fourier_imag.push(-phase.sin());
            }
        }
        Ok(Self {
            tensors,
            source_sha256: manifest.source_sha256,
            fourier_real,
            fourier_imag,
        })
    }

    /// Same reflect-centered STFT, last-frame removal, Slaney power mels,
    /// log floor and utterance-wide 80 dB power clamp as deployed Chatterbox.
    /// The >200-sample lower bound is Torch reflect padding's real precondition.
    pub fn features(&self, audio: &F32Npy) -> Result<F32Npy> {
        if audio.shape.len() != 1
            || audio.shape[0] <= FFT / 2
            || audio.shape[0] != audio.values.len()
        {
            return Err(
                "S3 tokenizer requires mono FP32 [samples] at 16000 Hz, more than 200 samples"
                    .into(),
            );
        }
        finite(&audio.values, "PCM")?;
        let wav = &audio.values;
        let frames = wav.len() / 160;
        let count = frames
            .checked_mul(BINS)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or("S3 spectrum size overflow")?;
        let mut powers = vec![0.0; count];
        let mut samples = [0.0_f32; FFT];
        for frame in 0..frames {
            let center = frame * 160;
            for (tap, value) in samples.iter_mut().enumerate() {
                let pos = center as isize + tap as isize - (FFT / 2) as isize;
                let reflected = if pos < 0 {
                    -pos
                } else if pos >= wav.len() as isize {
                    2 * (wav.len() as isize - 1) - pos
                } else {
                    pos
                } as usize;
                *value = wav[reflected] * self.tensors["window"].values[tap];
            }
            for bin in 0..BINS {
                let offset = bin * FFT;
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (tap, &sample) in samples.iter().enumerate() {
                    re += f64::from(sample) * self.fourier_real[offset + tap];
                    im += f64::from(sample) * self.fourier_imag[offset + tap];
                }
                // abs() then square matches Torch rather than combining re²+im².
                let magnitude = (re as f32).hypot(im as f32);
                powers[frame * BINS + bin] = magnitude * magnitude;
            }
        }
        finite(&powers, "power spectrum")?;
        let mut values = product(
            &powers,
            &self.tensors["_mel_filters"].values,
            frames,
            BINS,
            MELS,
        );
        finite(&values, "power mels")?;
        for value in &mut values {
            *value = value.max(1e-10).log10();
        }
        let floor = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
        for value in &mut values {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
        // The native matrix kernel is time-major; the public boundary stays
        // [1,128,T], exactly as the upstream tokenizer and its exported oracle.
        let mut channel_major = vec![0.0; values.len()];
        for frame in 0..frames {
            for channel in 0..MELS {
                channel_major[channel * frames + frame] = values[frame * MELS + channel];
            }
        }
        Ok(F32Npy {
            shape: vec![1, MELS, frames],
            values: channel_major,
        })
    }
    fn linear(&self, x: &[f32], p: &str) -> Vec<f32> {
        affine(
            x,
            &self.tensors[&format!("{p}.weight")],
            self.tensors.get(&format!("{p}.bias")),
        )
    }
    fn normalized(&self, x: &[f32], p: &str) -> Vec<f32> {
        norm(
            x,
            &self.tensors[&format!("{p}.weight")],
            &self.tensors[&format!("{p}.bias")],
            1e-5,
        )
    }
    /// Stride-two Conv1d, kernel=3, zero pad=1. Input/output are time-major.
    fn convolution(&self, x: &[f32], channels: usize, p: &str) -> Vec<f32> {
        let frames = x.len() / channels;
        let output_frames = frames.div_ceil(2);
        let mut columns = vec![0.0; output_frames * channels * 3];
        for frame in 0..output_frames {
            for channel in 0..channels {
                for tap in 0..3 {
                    let at = frame as isize * 2 + tap as isize - 1;
                    if at >= 0 && (at as usize) < frames {
                        columns[(frame * channels + channel) * 3 + tap] =
                            x[at as usize * channels + channel];
                    }
                }
            }
        }
        self.linear(&columns, p).into_iter().map(gelu).collect()
    }
    fn rotate(&self, values: &mut [f32]) {
        for (frame, row) in values.chunks_exact_mut(D).enumerate() {
            for head in row.chunks_exact_mut(HEAD) {
                for j in 0..HEAD / 2 {
                    let left = head[j];
                    let right = head[j + HEAD / 2];
                    let cos = self.tensors["rotary_cos"].values[frame * HEAD + j];
                    let sin = self.tensors["rotary_sin"].values[frame * HEAD + j];
                    head[j] = left * cos - right * sin;
                    head[j + HEAD / 2] = right * cos + left * sin;
                }
            }
        }
    }
    fn attention(&self, x: &[f32], p: &str) -> Result<Vec<f32>> {
        let frames = x.len() / D;
        let mut q = self.linear(x, &format!("{p}.query"));
        let mut k = self.linear(x, &format!("{p}.key"));
        let v = self.linear(x, &format!("{p}.value"));
        self.rotate(&mut q);
        self.rotate(&mut k);
        finite(&q, "query")?;
        finite(&k, "key")?;
        finite(&v, "value")?;
        let mut attended = vec![0.0; x.len()];
        // One head's T² workspace at a time. Native matrix calls own their
        // outputs, so there is no alias with checkpoint tensors or other heads.
        let scale = (HEAD as f32).powf(-0.25);
        for head in 0..D / HEAD {
            let mut qh = Vec::with_capacity(frames * HEAD);
            let mut kh = Vec::with_capacity(frames * HEAD);
            let mut vt = vec![0.0; frames * HEAD];
            for frame in 0..frames {
                let start = frame * D + head * HEAD;
                qh.extend(q[start..start + HEAD].iter().map(|&v| v * scale));
                kh.extend(k[start..start + HEAD].iter().map(|&v| v * scale));
                for j in 0..HEAD {
                    vt[j * frames + frame] = v[start + j];
                }
            }
            let mut scores = product(&qh, &kh, frames, HEAD, frames);
            finite(&scores, "attention scores")?;
            for row in scores.chunks_exact_mut(frames) {
                let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for value in row.iter_mut() {
                    *value = (*value - peak).exp();
                }
                let total = row.iter().sum::<f32>();
                for value in row {
                    *value /= total;
                }
            }
            let context = product(&scores, &vt, frames, frames, HEAD);
            for frame in 0..frames {
                attended[frame * D + head * HEAD..frame * D + (head + 1) * HEAD]
                    .copy_from_slice(&context[frame * HEAD..(frame + 1) * HEAD]);
            }
        }
        let mut result = self.linear(&attended, &format!("{p}.out"));
        let weights = &self.tensors[&format!("{p}.fsmn_block.weight")].values;
        // FSMN operates on unrotated V with symmetric 15-frame zero padding.
        for frame in 0..frames {
            for channel in 0..D {
                let mut sum = 0.0_f64;
                for tap in 0..31 {
                    let at = frame as isize + tap as isize - 15;
                    if at >= 0 && (at as usize) < frames {
                        sum += f64::from(v[at as usize * D + channel])
                            * f64::from(weights[channel * 31 + tap]);
                    }
                }
                result[frame * D + channel] += (sum as f32) + v[frame * D + channel];
            }
        }
        finite(&result, "attention output")?;
        Ok(result)
    }
    /// A single true-length segment. Upstream pads long-batch tails to 3000
    /// frames and masks them before both convolutions, attention and FSMN.
    /// Omitting those masked rows yields the same valid outputs, with bounded
    /// workspace and no computation for padding. The independent long probe
    /// compares against the actual padded-batch upstream implementation.
    pub fn encode_segment(&self, mels: &F32Npy) -> Result<CodecSegment> {
        let frames = mel_frames(mels)?;
        if frames > SEGMENT {
            return Err("Use encode_mels for multi-segment S3 audio".into());
        }
        let mut input = vec![0.0; mels.values.len()];
        for frame in 0..frames {
            for channel in 0..MELS {
                input[frame * MELS + channel] = mels.values[channel * frames + frame];
            }
        }
        let first = self.convolution(&input, MELS, "encoder.conv1");
        finite(&first, "first convolution")?;
        let mut x = self.convolution(&first, D, "encoder.conv2");
        finite(&x, "second convolution")?;
        for layer in 0..6 {
            let p = format!("encoder.blocks.{layer}");
            let attention = self.attention(
                &self.normalized(&x, &format!("{p}.attn_ln")),
                &format!("{p}.attn"),
            )?;
            for (x, a) in x.iter_mut().zip(attention) {
                *x += a;
            }
            let hidden = self.linear(
                &self.normalized(&x, &format!("{p}.mlp_ln")),
                &format!("{p}.mlp.0"),
            );
            let hidden: Vec<f32> = hidden.into_iter().map(gelu).collect();
            let residual = self.linear(&hidden, &format!("{p}.mlp.2"));
            for (x, r) in x.iter_mut().zip(residual) {
                *x += r;
            }
            finite(&x, "residual block")?;
        }
        let logits = self.linear(&x, "quantizer._codebook.project_down");
        let tokens = quantize(&logits)?;
        let frames = x.len() / D;
        Ok(CodecSegment {
            hidden: F32Npy {
                shape: vec![1, frames, D],
                values: x,
            },
            logits: F32Npy {
                shape: vec![1, frames, 8],
                values: logits,
            },
            tokens,
        })
    }
    pub fn encode_mels(&self, mels: &F32Npy) -> Result<Vec<u32>> {
        let frames = mel_frames(mels)?;
        if frames <= SEGMENT {
            return Ok(self.encode_segment(mels)?.tokens);
        }
        let mut tokens = Vec::new();
        for start in (0..frames).step_by(STRIDE) {
            let end = start.saturating_add(SEGMENT).min(frames);
            let length = end - start;
            let mut values = Vec::with_capacity(length * MELS);
            for channel in 0..MELS {
                values.extend_from_slice(
                    &mels.values[channel * frames + start..channel * frames + end],
                );
            }
            let segment = self.encode_segment(&F32Npy {
                shape: vec![1, MELS, length],
                values,
            })?;
            // Reproduce Python slice semantics even for a very short last tail.
            let left = if start == 0 { 0 } else { HALF_OVERLAP_TOKENS };
            let right = if start.saturating_add(STRIDE) < frames {
                segment.tokens.len().saturating_sub(HALF_OVERLAP_TOKENS)
            } else {
                segment.tokens.len()
            };
            if left < right {
                tokens.extend_from_slice(&segment.tokens[left..right]);
            }
        }
        Ok(tokens)
    }
    pub fn encode_pcm16k(&self, audio: &F32Npy) -> Result<Vec<u32>> {
        self.encode_mels(&self.features(audio)?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn s3_asset_admission() {
        use sha2::{Digest, Sha256};
        let sha = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        let root = std::env::temp_dir().join(format!(
            "s3-admission-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let reject = |manifest: &serde_json::Value, expected: &str| {
            let bytes = serde_json::to_vec(manifest).unwrap();
            let path = root.join("probe.json");
            std::fs::write(&path, &bytes).unwrap();
            let error = S3Tokenizer::open(&path, &sha(&bytes))
                .err()
                .expect("must reject");
            assert!(error.contains(expected), "{error}");
        };
        let manifest = serde_json::json!({"schema_version":1,
            "encoder_kind":"s3_tokenizer_v2_25hz_fp32", "source_sha256":"a".repeat(64),
            "tensors":{}});
        reject(&manifest, "Unsupported");
        // Every tensor name is present, but the first sorted asset has the wrong
        // shape. Reject it before allocating any checkpoint-sized model tables.
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n";
        let mut array = b"\x93NUMPY\x01\x00".to_vec();
        array.extend_from_slice(&(header.len() as u16).to_le_bytes());
        array.extend_from_slice(header.as_bytes());
        array.extend_from_slice(&0.0_f32.to_le_bytes());
        std::fs::write(root.join("wrong.npy"), &array).unwrap();
        let mut complete = manifest.clone();
        for key in required_shapes().keys() {
            complete["tensors"][key] = serde_json::json!({"file":"wrong.npy","sha256":sha(&array)});
        }
        reject(&complete, "shape mismatch");
        for (field, value) in [
            ("schema_version", serde_json::json!(2)),
            ("encoder_kind", serde_json::json!("wrong")),
            ("source_sha256", serde_json::json!("invalid")),
        ] {
            let mut bad = complete.clone();
            bad[field] = value;
            reject(&bad, "Unsupported");
        }
        let mut bad = complete.clone();
        bad["tensors"].as_object_mut().unwrap().remove("window");
        reject(&bad, "Unsupported");
        let mut bad = complete.clone();
        bad["tensors"]["extra"] = serde_json::json!({"file":"wrong.npy","sha256":sha(&array)});
        reject(&bad, "Unsupported");
        let first = "_mel_filters";
        for (field, value, expected) in [
            ("sha256", "0".repeat(64), "checksum differs"),
            ("file", "../wrong.npy".into(), "local"),
            ("file", "/wrong.npy".into(), "local"),
        ] {
            let mut bad = complete.clone();
            bad["tensors"][first][field] = serde_json::json!(value);
            reject(&bad, expected);
        }
        let mut invalid = array;
        let size = invalid.len();
        invalid[size - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(root.join("wrong.npy"), &invalid).unwrap();
        complete["tensors"][first]["sha256"] = serde_json::json!(sha(&invalid));
        reject(&complete, "Nonfinite");
        // This directory contains only the synthetic test's own probe bytes.
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn s3_fsq_endpoints_and_mel_contract() {
        assert_eq!(quantize(&[-100.0; 8]).unwrap(), vec![0]);
        assert_eq!(quantize(&[0.0; 8]).unwrap(), vec![3280]);
        assert_eq!(quantize(&[100.0; 8]).unwrap(), vec![6560]);
        let mut digits = [-100.0; 8];
        digits[1] = 100.0;
        assert_eq!(quantize(&digits).unwrap(), vec![6]);
        assert!(quantize(&[f32::NAN; 8]).is_err());
        assert!(quantize(&[1.0; 7]).is_err());
        assert!(mel_frames(&F32Npy {
            shape: vec![1, 128, 1],
            values: vec![0.0; 127]
        })
        .is_err());
        assert!(mel_frames(&F32Npy {
            shape: vec![1, 128, 0],
            values: vec![]
        })
        .is_err());
    }
}
