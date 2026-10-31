//! S3Gen's prepared ref_dict boundary, separate from T3's projected prefix.
//! Load retained reference state once. No audio decoding, model construction,
//! waveform inference occurs at reference admission. Flow assets separately
//! normalise/project its x-vector once and assemble the Conformer input.
//! Post-encoder assets trim lookahead and assemble the decoder's mel conditions.
use crate::{
    conditioning::{admitted_bytes, asset_path, valid_sha, Asset},
    npy::F32Npy,
    Result,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

// Post-encoder flow derived from CosyVoice, modified into Rust, 2026.
// Copyright (c) 2024 Alibaba Inc (Xiang Lyu, Zhihao Du).
// SPDX-License-Identifier: Apache-2.0 (see LICENSE).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MelManifest {
    schema_version: u32,
    mel_kind: String,
    source_sha256: String,
    weight: Asset,
    bias: Asset,
}

/// Retained FP32 encoder-to-mel affine tables, separate from waveform weights.
pub struct S3GenMelAssets {
    source_sha256: String,
    weight: F32Npy,
    bias: F32Npy,
}

/// Exact decoder-call boundary. All arrays are contiguous, channel first.
/// Non-final calls remove the same six lookahead frames from features and mask.
/// This repairs the deployed flow's mismatched decoder boundary; it does not
/// establish prefix stability or waveform overlap for streaming synthesis.
pub struct S3GenMelInput {
    pub mu: F32Npy,
    pub cond: F32Npy,
    pub mask: F32Npy,
    pub speaker: F32Npy,
    pub reference_frames: usize,
    pub generated_frames: usize,
}

#[cfg(test)]
mod mel_tests {
    use super::*;

    #[test]
    fn mel_nonfinal_conditioning_aligns_mask_and_rejects_short_prefix() {
        // Tiny private fixtures isolate frame accounting from checkpoint/voice
        // admission. Only token 0 is used, so one embedding row suffices here.
        let array = |shape: &[usize], value| F32Npy {
            shape: shape.to_vec(),
            values: vec![value; shape.iter().product()],
        };
        let flow = S3GenFlowAssets {
            source_sha256: "a".repeat(64),
            token_embedding: array(&[1, 512], 0.0),
            speaker_weight: array(&[80, 192], 0.0),
            speaker_bias: array(&[80], 0.0),
        };
        let mel = S3GenMelAssets {
            source_sha256: "a".repeat(64),
            weight: array(&[80, 512], 0.0),
            bias: array(&[80], 0.25),
        };
        // Odd reference mels consume one additional frame beyond their tokens.
        // A three-token prefix for an even reference leaves exactly zero audio.
        for reference_frames in [3usize, 4] {
            let reference = S3GenReference {
                source_sha256: "b".repeat(64),
                prompt_tokens: vec![0; reference_frames / 2],
                prompt_feat: array(&[1, reference_frames, 80], 0.5),
                prompt_feat_len: None,
                embedding: array(&[1, 192], 1.0),
            };
            let prepared = flow.prepare(&reference).unwrap();
            for prefix in [4usize, 1, 2, 3, 4, 5] {
                let input = prepared.assemble(&vec![0; prefix]).unwrap();
                let frames = input.token_len() * 2;
                let encoded = crate::conformer::EncoderOutput {
                    features: array(&[1, frames, 512], 0.0),
                    mask: array(&[1, 1, frames], 1.0),
                };
                let final_output = mel.condition(&input, &encoded, true).unwrap();
                let partial = mel.condition(&input, &encoded, false);
                if prefix <= 3 {
                    assert!(
                        partial.is_err(),
                        "reference={reference_frames}, prefix={prefix}"
                    );
                    continue;
                }
                let partial = partial.unwrap();
                let kept = frames - 6;
                assert_eq!(partial.mu.shape, [1, 80, kept]);
                assert_eq!(partial.cond.shape, partial.mu.shape);
                assert_eq!(partial.mask.shape, [1, 1, kept]);
                assert_eq!(partial.mask.values, vec![1.0; kept]);
                assert_eq!(partial.reference_frames, reference_frames);
                assert_eq!(partial.generated_frames, kept - reference_frames);
                for c in 0..80 {
                    assert_eq!(
                        partial.mu.values[c * kept..(c + 1) * kept],
                        final_output.mu.values[c * frames..c * frames + kept]
                    );
                    assert_eq!(
                        &partial.cond.values[c * kept..c * kept + reference_frames],
                        vec![0.5; reference_frames]
                    );
                    assert!(
                        partial.cond.values[c * kept + reference_frames..(c + 1) * kept]
                            .iter()
                            .all(|&v| v == 0.0)
                    );
                }
                // An intervening non-final call must not mutate the final path.
                let replay = mel.condition(&input, &encoded, true).unwrap();
                assert_eq!(replay.mu.values, final_output.mu.values);
                assert_eq!(replay.mask.values, encoded.mask.values);
            }
        }
    }
}

impl S3GenMelAssets {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let manifest: MelManifest = serde_json::from_slice(&admitted_bytes(path, expected)?)
            .map_err(|e| format!("S3Gen mel manifest: {e}"))?;
        if manifest.schema_version != 1
            || manifest.mel_kind != "s3gen_post_encoder_fp32"
            || !valid_sha(&manifest.source_sha256)
        {
            return Err("Unsupported S3Gen mel export".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let read = |asset: &Asset, shape: &[usize]| -> Result<F32Npy> {
            let array =
                F32Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)?;
            if array.shape != shape {
                return Err(format!("S3Gen mel table requires shape {shape:?}"));
            }
            Ok(array)
        };
        Ok(Self {
            source_sha256: manifest.source_sha256,
            weight: read(&manifest.weight, &[80, 512])?,
            bias: read(&manifest.bias, &[80])?,
        })
    }

    /// Consume the existing admitted codec/voice boundary and its encoder output.
    /// No mutable voice/request state, noise generation or decoder invocation.
    pub fn condition(
        &self,
        input: &S3GenFlowInput<'_>,
        encoded: &crate::conformer::EncoderOutput,
        finalize: bool,
    ) -> Result<S3GenMelInput> {
        if self.source_sha256 != input.prepared.assets.source_sha256 {
            return Err("S3Gen flow and mel checkpoint checksums differ".into());
        }
        let frames = input
            .token_len()
            .checked_mul(2)
            .ok_or("S3Gen mel length overflow")?;
        let elements = frames
            .checked_mul(512)
            .ok_or("S3Gen encoded size overflow")?;
        // Batch one, unpadded, full encoder output is the only admitted contract.
        // Verify public F32Npy fields before calling the shared affine kernel.
        if encoded.features.shape != [1, frames, 512]
            || encoded.features.values.len() != elements
            || encoded.features.values.iter().any(|v| !v.is_finite())
            || encoded.mask.shape != [1, 1, frames]
            || encoded.mask.values.len() != frames
            || encoded.mask.values.iter().any(|&v| v != 1.0)
        {
            return Err("S3Gen mel requires complete finite unpadded encoder output".into());
        }
        let kept = if finalize {
            frames
        } else {
            frames.saturating_sub(6)
        };
        let reference = input.prepared.reference.prompt_feat();
        let reference_frames = reference.shape[1];
        let generated_frames = kept
            .checked_sub(reference_frames)
            .ok_or("S3Gen lookahead trim leaves fewer frames than the reference mels")?;
        if !finalize && generated_frames == 0 {
            return Err("S3Gen non-final prefix has no generated frames after lookahead".into());
        }
        let output_elements = kept.checked_mul(80).ok_or("S3Gen mel size overflow")?;
        // Same native matrix kernel and affine orientation as the Conformer.
        let projected = crate::conformer::affine(
            &encoded.features.values[..kept * 512],
            &self.weight,
            Some(&self.bias),
        );
        if projected.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite S3Gen mel projection".into());
        }
        let mut mu = vec![0.0; output_elements];
        let mut cond = vec![0.0; output_elements];
        // Transpose [1,F,80] to contiguous [1,80,F]. Copy every retained mel
        // bit, including an odd final frame; generated conditioning stays zero.
        for channel in 0..80 {
            for frame in 0..kept {
                mu[channel * kept + frame] = projected[frame * 80 + channel];
                if frame < reference_frames {
                    cond[channel * kept + frame] = reference.values[frame * 80 + channel];
                }
            }
        }
        Ok(S3GenMelInput {
            mu: F32Npy {
                shape: vec![1, 80, kept],
                values: mu,
            },
            cond: F32Npy {
                shape: vec![1, 80, kept],
                values: cond,
            },
            // The deployed flow derives its mask from untrimmed h_masks and
            // fails in meanflow on non-final input. Match the retained feature
            // span explicitly; final calls still copy the complete original mask.
            mask: F32Npy {
                shape: vec![1, 1, kept],
                values: encoded.mask.values[..kept].to_vec(),
            },
            speaker: F32Npy {
                shape: vec![1, 80],
                values: input.prepared.speaker.values.clone(),
            },
            reference_frames,
            generated_frames,
        })
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    reference_kind: String,
    source_sha256: String,
    prompt_tokens: Asset,
    prompt_token_len: usize,
    prompt_feat: Asset,
    prompt_feat_len: Option<usize>,
    embedding: Asset,
}

/// Immutable, single-voice, unpadded reference state for the future native flow.
/// T3's 256-value speaker tensor is unrelated to this 192-value x-vector.
pub struct S3GenReference {
    source_sha256: String,
    prompt_tokens: Vec<i64>,
    prompt_feat: F32Npy,
    prompt_feat_len: Option<usize>,
    embedding: F32Npy,
}

#[derive(Serialize)]
pub struct TensorReceipt {
    pub shape: Vec<usize>,
    pub dtype: &'static str,
    pub sha256: String,
}

#[derive(Serialize)]
pub struct ReferenceReceipt {
    pub source_sha256: String,
    pub prompt_token: TensorReceipt,
    pub prompt_token_len: TensorReceipt,
    pub prompt_feat: TensorReceipt,
    pub prompt_feat_len: Option<TensorReceipt>,
    pub embedding: TensorReceipt,
}

fn digest(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut hash = Sha256::new();
    // The reference arrays are small; hash in bounded blocks without another
    // full tensor allocation. Byte order is portable across native hosts.
    let mut block = Vec::with_capacity(4096);
    for byte in bytes {
        block.push(byte);
        if block.len() == 4096 {
            hash.update(&block);
            block.clear();
        }
    }
    hash.update(&block);
    format!("{:x}", hash.finalize())
}

fn integer_receipt(values: &[i64], shape: Vec<usize>) -> TensorReceipt {
    TensorReceipt {
        shape,
        dtype: "int64",
        sha256: digest(values.iter().flat_map(|v| v.to_le_bytes())),
    }
}
fn float_receipt(array: &F32Npy) -> TensorReceipt {
    TensorReceipt {
        shape: array.shape.clone(),
        dtype: "float32",
        sha256: digest(array.values.iter().flat_map(|v| v.to_le_bytes())),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FlowManifest {
    schema_version: u32,
    flow_kind: String,
    source_sha256: String,
    token_embedding: Asset,
    speaker_weight: Asset,
    speaker_bias: Asset,
}

/// Only the three FP32 tables used before the Conformer. No model/engine is
/// loaded; dimensions belong to the pinned Turbo S3Gen, separate from T3.
pub struct S3GenFlowAssets {
    source_sha256: String,
    token_embedding: F32Npy,
    speaker_weight: F32Npy,
    speaker_bias: F32Npy,
}

/// A voice borrows immutable assets/reference and computes speaker work once.
/// Reference mels (including an odd tail) remain available for the later flow.
pub struct PreparedS3GenFlow<'a> {
    assets: &'a S3GenFlowAssets,
    reference: &'a S3GenReference,
    normalized: F32Npy,
    speaker: F32Npy,
}

/// Single-request, unpadded batch-one encoder inputs. Mask is FP32 [1,T,1].
/// Finalize/lookahead trimming occurs after the Conformer and is not applied here.
pub struct S3GenFlowInput<'a> {
    prepared: &'a PreparedS3GenFlow<'a>,
    token_ids: Vec<i64>,
    embeddings: F32Npy,
    mask: F32Npy,
}

#[derive(Serialize)]
pub struct FlowInputReceipt {
    pub source_sha256: String,
    pub reference: ReferenceReceipt,
    pub token: TensorReceipt,
    pub token_len: TensorReceipt,
    pub mask: TensorReceipt,
    pub encoder_input: TensorReceipt,
    pub normalized_speaker: TensorReceipt,
    pub speaker: TensorReceipt,
}

impl S3GenFlowAssets {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let manifest: FlowManifest = serde_json::from_slice(&admitted_bytes(path, expected)?)
            .map_err(|e| format!("S3Gen flow manifest: {e}"))?;
        if manifest.schema_version != 1
            || manifest.flow_kind != "s3gen_encoder_input"
            || !valid_sha(&manifest.source_sha256)
        {
            return Err("Unsupported S3Gen flow-input export".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let read = |asset: &Asset, shape: &[usize]| -> Result<F32Npy> {
            let array =
                F32Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)?;
            if array.shape != shape {
                return Err(format!("S3Gen flow table requires shape {shape:?}"));
            }
            Ok(array)
        };
        Ok(Self {
            source_sha256: manifest.source_sha256,
            token_embedding: read(&manifest.token_embedding, &[6561, 512])?,
            speaker_weight: read(&manifest.speaker_weight, &[80, 192])?,
            speaker_bias: read(&manifest.speaker_bias, &[80])?,
        })
    }

    pub fn prepare<'a>(&'a self, reference: &'a S3GenReference) -> Result<PreparedS3GenFlow<'a>> {
        let speaker = &reference.embedding.values;
        // Match F.normalize(p=2, dim=1, eps=1e-12) in model FP32. A scalar
        // reduction need not be bit-identical to Torch's vectorised reduction.
        let norm = speaker.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !norm.is_finite() {
            return Err("Nonfinite S3Gen speaker norm".into());
        }
        let normalized = F32Npy {
            shape: vec![1, 192],
            values: speaker.iter().map(|v| v / norm.max(1e-12)).collect(),
        };
        let mut projected = Vec::with_capacity(80);
        // Wide accumulation and one final rounding avoid error accumulation in
        // this tiny once-per-voice affine transform. Verify against Torch with
        // an explicit FP32 tolerance; no universal bit parity is implied.
        for (row, &bias) in self
            .speaker_weight
            .values
            .chunks_exact(192)
            .zip(&self.speaker_bias.values)
        {
            let value = (row
                .iter()
                .zip(&normalized.values)
                .map(|(&w, &v)| f64::from(w) * f64::from(v))
                .sum::<f64>()
                + f64::from(bias)) as f32;
            if !value.is_finite() {
                return Err("Nonfinite S3Gen speaker projection".into());
            }
            projected.push(value);
        }
        Ok(PreparedS3GenFlow {
            assets: self,
            reference,
            normalized,
            speaker: F32Npy {
                shape: vec![1, 80],
                values: projected,
            },
        })
    }
}

impl PreparedS3GenFlow<'_> {
    pub fn reference(&self) -> &S3GenReference {
        self.reference
    }
    pub fn normalized_speaker(&self) -> &F32Npy {
        &self.normalized
    }
    pub fn speaker(&self) -> &F32Npy {
        &self.speaker
    }

    pub fn assemble(&self, speech_tokens: &[i64]) -> Result<S3GenFlowInput<'_>> {
        if speech_tokens.is_empty() || speech_tokens.iter().any(|&t| !(0..6561).contains(&t)) {
            return Err("S3Gen flow requires nonempty generated codec IDs in 0..6561".into());
        }
        let length = self
            .reference
            .prompt_token_len()
            .checked_add(speech_tokens.len())
            .ok_or("S3Gen token length overflow")?;
        let elements = length
            .checked_mul(512)
            .ok_or("S3Gen embedding size overflow")?;
        let token_ids: Vec<i64> = self
            .reference
            .prompt_tokens
            .iter()
            .chain(speech_tokens)
            .copied()
            .collect();
        let mut values = Vec::with_capacity(elements);
        // Unpadded batch one: every admitted token is valid, hence mask=1.
        // Preserve reference order, then generated tokens, without T3 BOS/EOS.
        for &id in &token_ids {
            values.extend_from_slice(
                &self.assets.token_embedding.values[id as usize * 512..(id as usize + 1) * 512],
            );
        }
        Ok(S3GenFlowInput {
            prepared: self,
            token_ids,
            embeddings: F32Npy {
                shape: vec![1, length, 512],
                values,
            },
            mask: F32Npy {
                shape: vec![1, length, 1],
                values: vec![1.0; length],
            },
        })
    }
}

impl S3GenFlowInput<'_> {
    pub fn token_ids(&self) -> &[i64] {
        &self.token_ids
    }
    pub fn token_len(&self) -> usize {
        self.token_ids.len()
    }
    pub fn embeddings(&self) -> &F32Npy {
        &self.embeddings
    }
    pub fn mask(&self) -> &F32Npy {
        &self.mask
    }
    pub fn receipt(&self) -> FlowInputReceipt {
        FlowInputReceipt {
            source_sha256: self.prepared.assets.source_sha256.clone(),
            reference: self.prepared.reference.receipt(),
            token: integer_receipt(&self.token_ids, vec![1, self.token_len()]),
            token_len: integer_receipt(&[self.token_len() as i64], vec![1]),
            mask: float_receipt(&self.mask),
            encoder_input: float_receipt(&self.embeddings),
            normalized_speaker: float_receipt(&self.prepared.normalized),
            speaker: float_receipt(&self.prepared.speaker),
        }
    }
}

// Audio analysis belongs to the existing S3 reference boundary, not to T3's
// incompatible SoXR/power-mel path. Tables are exported from deployed functions.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisManifest {
    schema_version: u32,
    analysis_kind: String,
    source_sha256: String,
    tensors: std::collections::BTreeMap<String, Asset>,
}

/// Immutable 24kHz reference analysis used only when preparing a new voice.
/// TorchAudio's fixed 3:2 Hann-sinc kernel differs from librosa/SoXR.
pub struct S3ReferenceAnalysis {
    pub source_sha256: String,
    window: Vec<f32>,
    basis: Vec<f32>,
    kernel: Vec<f32>,
    fourier_real: Vec<f64>,
    fourier_imag: Vec<f64>,
}
impl S3ReferenceAnalysis {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let manifest: AnalysisManifest =
            serde_json::from_slice(&admitted_bytes(path, expected)?)
                .map_err(|e| format!("S3 reference analysis manifest: {e}"))?;
        let shapes = std::collections::BTreeMap::from([
            ("mel_basis", vec![80, 961]),
            ("resample_kernel", vec![2, 1, 23]),
            ("window", vec![1920]),
        ]);
        if manifest.schema_version != 1
            || manifest.analysis_kind != "s3gen_reference_analysis_fp32"
            || !valid_sha(&manifest.source_sha256)
            || !manifest
                .tensors
                .keys()
                .map(String::as_str)
                .eq(shapes.keys().copied())
        {
            return Err("Unsupported S3 reference analysis export".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tensors = std::collections::BTreeMap::new();
        for (name, shape) in shapes {
            let asset = &manifest.tensors[name];
            let array =
                F32Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)?;
            if array.shape != shape
                || (name != "resample_kernel" && array.values.iter().any(|&v| v < 0.0))
            {
                return Err(format!("Invalid S3 reference analysis table {name}"));
            }
            tensors.insert(name, array.values);
        }
        // Same retained Fourier-table pattern as the existing native encoders.
        // FP64 accumulation avoids platform-dependent intermediate FFT roundoff.
        let mut fourier_real = Vec::with_capacity(961 * 1920);
        let mut fourier_imag = Vec::with_capacity(961 * 1920);
        for bin in 0..961 {
            for tap in 0..1920 {
                let phase = -std::f64::consts::TAU * bin as f64 * tap as f64 / 1920.0;
                let (sin, cos) = phase.sin_cos();
                fourier_real.push(cos);
                fourier_imag.push(sin);
            }
        }
        Ok(Self {
            source_sha256: manifest.source_sha256,
            window: tensors.remove("window").unwrap(),
            basis: tensors.remove("mel_basis").unwrap(),
            kernel: tensors.remove("resample_kernel").unwrap(),
            fourier_real,
            fourier_imag,
        })
    }
    fn audio<'a>(&self, input: &'a F32Npy) -> Result<&'a [f32]> {
        if input.shape.len() != 1
            || input.values.is_empty()
            || input.shape[0] != input.values.len()
            || input.values.iter().any(|v| !v.is_finite())
            || input.values.len() > isize::MAX as usize - 1920
        {
            return Err(
                "S3 reference requires finite complete mono FP32 [samples] at 24000 Hz".into(),
            );
        }
        Ok(&input.values)
    }
    pub fn resample(&self, input: &F32Npy) -> Result<F32Npy> {
        let audio = self.audio(input)?;
        let length = audio
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .ok_or("S3 reference resample length overflow")?
            / 3;
        let mut values = Vec::with_capacity(length);
        // Zero-padded stride-three convolution; interleave its two output phases.
        // This preserves ceil(2*N/3), including the final partial input block.
        for frame in 0..length.div_ceil(2) {
            for phase in 0..2 {
                if values.len() == length {
                    break;
                }
                let mut sum = 0f32;
                for tap in 0..23 {
                    let index = (frame * 3) as isize + tap as isize - 10;
                    if index >= 0 && (index as usize) < audio.len() {
                        sum += audio[index as usize] * self.kernel[phase * 23 + tap];
                        if !sum.is_finite() {
                            return Err("S3 reference resampling overflow".into());
                        }
                    }
                }
                values.push(sum);
            }
        }
        Ok(F32Npy {
            shape: vec![length],
            values,
        })
    }
    pub fn mels(&self, input: &F32Npy) -> Result<F32Npy> {
        let audio = self.audio(input)?;
        if audio.len() <= 720 {
            return Err("S3 reference reflect padding requires more than 720 samples".into());
        }
        // Explicit reflect pad720, FFT1920/hop480, center=False. No last-frame
        // removal, power squaring, log10 scaling or utterance-wide clamp.
        let frames = audio.len() / 480;
        let count = frames
            .checked_mul(961)
            .ok_or("S3 reference spectrum size overflow")?;
        let mut magnitudes = vec![0.0; count];
        let mut windowed = vec![0f32; 1920];
        for frame in 0..frames {
            for (tap, value) in windowed.iter_mut().enumerate() {
                let index = (frame * 480) as isize + tap as isize - 720;
                let reflected = if index < 0 {
                    -index
                } else if index >= audio.len() as isize {
                    2 * (audio.len() as isize - 1) - index
                } else {
                    index
                } as usize;
                *value = audio[reflected] * self.window[tap];
            }
            for bin in 0..961 {
                let mut real = 0f64;
                let mut imag = 0f64;
                for (tap, &sample) in windowed.iter().enumerate() {
                    real += f64::from(sample) * self.fourier_real[bin * 1920 + tap];
                    imag += f64::from(sample) * self.fourier_imag[bin * 1920 + tap];
                }
                let real = real as f32;
                let imag = imag as f32;
                let magnitude = (real * real + imag * imag + 1e-9).sqrt();
                if !magnitude.is_finite() {
                    return Err("S3 reference spectrum overflow".into());
                }
                magnitudes[frame * 961 + bin] = magnitude;
            }
        }
        let mut values = crate::conformer::product(&magnitudes, &self.basis, frames, 961, 80);
        for value in &mut values {
            if !value.is_finite() {
                return Err("S3 reference mel overflow".into());
            }
            *value = value.max(1e-5).ln();
        }
        Ok(F32Npy {
            shape: vec![1, frames, 80],
            values,
        })
    }
}

impl S3GenReference {
    /// Use the same FP32 reference format already admitted by native synthesis.
    pub(crate) fn export(&self, destination: &Path) -> Result<Asset> {
        use crate::voice_preparation::{float_array, publish_file};
        let manifest = Manifest {
            schema_version: 1,
            reference_kind: "s3gen_prepared".into(),
            source_sha256: self.source_sha256.clone(),
            prompt_tokens: publish_file(
                destination,
                "s3-tokens.json",
                &serde_json::to_vec(&self.prompt_tokens).map_err(|e| e.to_string())?,
            )?,
            prompt_token_len: self.prompt_tokens.len(),
            prompt_feat: publish_file(destination, "mels.npy", &float_array(&self.prompt_feat)?)?,
            prompt_feat_len: self.prompt_feat_len,
            embedding: publish_file(destination, "embedding.npy", &float_array(&self.embedding)?)?,
        };
        let pin = publish_file(
            destination,
            "reference.json",
            &serde_json::to_vec(&manifest).map_err(|e| e.to_string())?,
        )?;
        Self::open(&destination.join(&pin.file), &pin.sha256)?;
        Ok(pin)
    }

    /// Admit freshly encoded tensors through the same boundary as retained voices.
    /// Lengths remain integral and an odd final mel frame remains untouched.
    pub fn from_parts(
        source_sha256: String,
        prompt_tokens: Vec<i64>,
        prompt_feat: F32Npy,
        prompt_feat_len: Option<usize>,
        embedding: F32Npy,
    ) -> Result<Self> {
        let frames = prompt_feat.shape.get(1).copied().unwrap_or(0);
        if !valid_sha(&source_sha256)
            || prompt_tokens.is_empty()
            || prompt_tokens.iter().any(|&t| !(0..6561).contains(&t))
            || prompt_feat.shape != [1, frames, 80]
            || frames.checked_mul(80) != Some(prompt_feat.values.len())
            || frames / 2 != prompt_tokens.len()
            || prompt_feat_len.is_some_and(|n| n != frames)
            || embedding.shape != [1, 192]
            || embedding.values.len() != 192
            || prompt_feat
                .values
                .iter()
                .chain(&embedding.values)
                .any(|v| !v.is_finite())
        {
            return Err("Invalid complete S3Gen reference tensors".into());
        }
        Ok(Self {
            source_sha256,
            prompt_tokens,
            prompt_feat,
            prompt_feat_len,
            embedding,
        })
    }

    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(&admitted_bytes(path, expected)?)
            .map_err(|e| format!("S3Gen reference manifest: {e}"))?;
        if manifest.schema_version != 1
            || manifest.reference_kind != "s3gen_prepared"
            || !valid_sha(&manifest.source_sha256)
        {
            return Err("Unsupported S3Gen prepared reference export".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let read = |asset: &Asset| admitted_bytes(&asset_path(root, asset)?, &asset.sha256);
        let prompt_tokens: Vec<i64> = serde_json::from_slice(&read(&manifest.prompt_tokens)?)
            .map_err(|e| format!("S3Gen reference tokens: {e}"))?;
        if prompt_tokens.is_empty()
            || prompt_tokens.len() != manifest.prompt_token_len
            || prompt_tokens.iter().any(|&t| !(0..6561).contains(&t))
        {
            return Err(
                "S3Gen reference requires nonempty codec tokens and their exact length".into(),
            );
        }
        let prompt_feat = F32Npy::from_bytes(&read(&manifest.prompt_feat)?)?;
        let embedding = F32Npy::from_bytes(&read(&manifest.embedding)?)?;
        // embed_ref retains an odd final mel frame. Floor(mel/2) equals the
        // codec-token count; never truncate the 403rd frame of the retained voice.
        let frames = prompt_feat.shape.get(1).copied().unwrap_or(0);
        if prompt_feat.shape != [1, frames, 80]
            || frames / 2 != prompt_tokens.len()
            || manifest
                .prompt_feat_len
                .is_some_and(|length| length != frames)
        {
            return Err("S3Gen reference mel shape/length must match its codec tokens".into());
        }
        if embedding.shape != [1, 192] {
            return Err("S3Gen speaker embedding must have shape [1,192]".into());
        }
        Self::from_parts(
            manifest.source_sha256,
            prompt_tokens,
            prompt_feat,
            manifest.prompt_feat_len,
            embedding,
        )
    }

    pub fn prompt_tokens(&self) -> &[i64] {
        &self.prompt_tokens
    }
    pub fn prompt_token_len(&self) -> usize {
        self.prompt_tokens.len()
    }
    pub fn prompt_feat(&self) -> &F32Npy {
        &self.prompt_feat
    }
    pub fn prompt_feat_len(&self) -> Option<usize> {
        self.prompt_feat_len
    }
    pub fn embedding(&self) -> &F32Npy {
        &self.embedding
    }

    /// Hash the exact portable ref_dict tensors, not their NumPy envelopes.
    /// Lengths stay integral until the future flow boundary performs any casts.
    pub fn receipt(&self) -> ReferenceReceipt {
        ReferenceReceipt {
            source_sha256: self.source_sha256.clone(),
            prompt_token: integer_receipt(&self.prompt_tokens, vec![1, self.prompt_token_len()]),
            prompt_token_len: integer_receipt(&[self.prompt_token_len() as i64], vec![1]),
            prompt_feat: float_receipt(&self.prompt_feat),
            prompt_feat_len: self
                .prompt_feat_len
                .map(|n| integer_receipt(&[n as i64], vec![1])),
            embedding: float_receipt(&self.embedding),
        }
    }
}
