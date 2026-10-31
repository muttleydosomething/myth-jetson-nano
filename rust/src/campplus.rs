// CAMPPlus derived from FunASR / 3D-Speaker via Resemble AI Chatterbox.
// Kaldi feature analysis derived from torchaudio.compliance.kaldi.
// Ported into Rust by Hailstorm contributors, 2026.
// SPDX-License-Identifier: MIT AND BSD-2-Clause
//! Single-utterance evaluation CAMPPlus: mono 16 kHz PCM -> 192-value x-vector.
//! Kaldi FBANK has different framing/window/floor from both other voice encoders.
//! Tables are immutable; all activations, segment context and statistics are local.
use crate::{
    conditioning::{admitted_bytes, asset_path, valid_sha, Asset},
    conformer::affine,
    npy::F32Npy,
    Result,
};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

const MEL: usize = 80;
const WINDOW: usize = 400;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const HOP: usize = 160;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    encoder_kind: String,
    source_sha256: String,
    tensors: BTreeMap<String, Asset>,
}

/// Evaluation weights and fixed Fourier tables; no mutable recurrent state.
pub struct CampPlus {
    tensors: BTreeMap<String, F32Npy>,
    pub source_sha256: String,
    real: Vec<f64>,
    imag: Vec<f64>,
}

fn bn_shapes(shapes: &mut BTreeMap<String, Vec<usize>>, name: &str, n: usize, affine: bool) {
    for field in ["running_mean", "running_var"] {
        shapes.insert(format!("{name}.{field}"), vec![n]);
    }
    if affine {
        for field in ["weight", "bias"] {
            shapes.insert(format!("{name}.{field}"), vec![n]);
        }
    }
}
fn weight(shapes: &mut BTreeMap<String, Vec<usize>>, name: &str, dims: Vec<usize>, bias: bool) {
    if bias {
        shapes.insert(format!("{name}.bias"), vec![dims[0]]);
    }
    shapes.insert(format!("{name}.weight"), dims);
}
fn required_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    weight(&mut shapes, "head.conv1", vec![32, 1, 3, 3], false);
    weight(&mut shapes, "head.conv2", vec![32, 32, 3, 3], false);
    for name in ["head.bn1", "head.bn2"] {
        bn_shapes(&mut shapes, name, 32, true);
    }
    for stage in 1..=2 {
        for block in 0..2 {
            let prefix = format!("head.layer{stage}.{block}");
            for i in 1..=2 {
                weight(
                    &mut shapes,
                    &format!("{prefix}.conv{i}"),
                    vec![32, 32, 3, 3],
                    false,
                );
                bn_shapes(&mut shapes, &format!("{prefix}.bn{i}"), 32, true);
            }
            if block == 0 {
                weight(
                    &mut shapes,
                    &format!("{prefix}.shortcut.0"),
                    vec![32, 32, 1, 1],
                    false,
                );
                bn_shapes(&mut shapes, &format!("{prefix}.shortcut.1"), 32, true);
            }
        }
    }
    weight(&mut shapes, "xvector.tdnn.linear", vec![128, 320, 5], false);
    bn_shapes(&mut shapes, "xvector.tdnn.nonlinear.batchnorm", 128, true);
    let mut channels = 128;
    for (block, layers) in [(1, 12), (2, 24), (3, 16)] {
        for layer in 1..=layers {
            let prefix = format!("xvector.block{block}.tdnnd{layer}");
            bn_shapes(
                &mut shapes,
                &format!("{prefix}.nonlinear1.batchnorm"),
                channels,
                true,
            );
            weight(
                &mut shapes,
                &format!("{prefix}.linear1"),
                vec![128, channels, 1],
                false,
            );
            bn_shapes(
                &mut shapes,
                &format!("{prefix}.nonlinear2.batchnorm"),
                128,
                true,
            );
            weight(
                &mut shapes,
                &format!("{prefix}.cam_layer.linear_local"),
                vec![32, 128, 3],
                false,
            );
            weight(
                &mut shapes,
                &format!("{prefix}.cam_layer.linear1"),
                vec![64, 128, 1],
                true,
            );
            weight(
                &mut shapes,
                &format!("{prefix}.cam_layer.linear2"),
                vec![32, 64, 1],
                true,
            );
            channels += 32;
        }
        let prefix = format!("xvector.transit{block}");
        bn_shapes(
            &mut shapes,
            &format!("{prefix}.nonlinear.batchnorm"),
            channels,
            true,
        );
        weight(
            &mut shapes,
            &format!("{prefix}.linear"),
            vec![channels / 2, channels, 1],
            false,
        );
        channels /= 2;
    }
    bn_shapes(
        &mut shapes,
        "xvector.out_nonlinear.batchnorm",
        channels,
        true,
    );
    weight(
        &mut shapes,
        "xvector.dense.linear",
        vec![192, channels * 2, 1],
        false,
    );
    bn_shapes(&mut shapes, "xvector.dense.nonlinear.batchnorm", 192, false);
    shapes.insert("mel_basis".into(), vec![MEL, BINS]);
    shapes.insert("window".into(), vec![WINDOW]);
    shapes
}
fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        Err("CAMPPlus produced nonfinite features or activations".into())
    } else {
        Ok(())
    }
}
fn validate_features(features: &F32Npy) -> Result<usize> {
    if features.shape.len() != 2
        || features.shape[1] != MEL
        || features.shape[0] < 3
        || features.shape[0].checked_mul(MEL) != Some(features.values.len())
    {
        return Err("CAMPPlus requires finite [T,80] features with at least three frames".into());
    }
    features.shape[0]
        .checked_mul(80 * 32)
        .ok_or("CAMPPlus workspace length overflow")?;
    finite(&features.values)?;
    Ok(features.shape[0])
}

/// Layout for 2D convolutions: [frequency, time, channels].
struct Map {
    values: Vec<f32>,
    freq: usize,
    time: usize,
    channels: usize,
}

impl CampPlus {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(&admitted_bytes(path, expected)?)
            .map_err(|e| format!("CAMPPlus manifest: {e}"))?;
        let shapes = required_shapes();
        if manifest.schema_version != 1
            || manifest.encoder_kind != "campplus_192_fp32"
            || !valid_sha(&manifest.source_sha256)
            || manifest.tensors.keys().ne(shapes.keys())
        {
            return Err("Unsupported or incomplete CAMPPlus export".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tensors = BTreeMap::new();
        for (name, shape) in shapes {
            let asset = &manifest.tensors[&name];
            let array =
                F32Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)?;
            if array.shape != shape {
                return Err(format!("CAMPPlus shape mismatch: {name}"));
            }
            if (name.ends_with(".running_var") || name == "mel_basis" || name == "window")
                && array.values.iter().any(|v| *v < 0.0)
            {
                return Err(format!("CAMPPlus invalid nonnegative table: {name}"));
            }
            tensors.insert(name, array);
        }
        // FFT is zero-padded to 512; keep FP32 preprocessing outside the wide DFT.
        let mut real = Vec::with_capacity(BINS * WINDOW);
        let mut imag = Vec::with_capacity(BINS * WINDOW);
        for bin in 0..BINS {
            for sample in 0..WINDOW {
                let angle = std::f64::consts::TAU * bin as f64 * sample as f64 / FFT as f64;
                real.push(angle.cos());
                imag.push(-angle.sin());
            }
        }
        Ok(Self {
            tensors,
            source_sha256: manifest.source_sha256,
            real,
            imag,
        })
    }

    /// Kaldi defaults: snipped 25ms frames, DC removal, 0.97 preemphasis, Povey
    /// window, 512 FFT, power mels, epsilon log floor, then column mean removal.
    pub fn features(&self, audio: &F32Npy) -> Result<F32Npy> {
        if audio.shape.len() != 1 || audio.shape[0] < WINDOW || audio.shape[0] != audio.values.len()
        {
            return Err(
                "CAMPPlus requires finite mono FP32 [samples] at 16000 Hz, at least 400 samples"
                    .into(),
            );
        }
        finite(&audio.values)?;
        let frames = 1 + (audio.values.len() - WINDOW) / HOP;
        let count = frames
            .checked_mul(MEL)
            .ok_or("CAMPPlus feature length overflow")?;
        let mut values = Vec::with_capacity(count);
        let window = &self.tensors["window"].values;
        let mut samples = vec![0.0_f32; WINDOW];
        let mut powers = vec![0.0; BINS];
        for frame in 0..frames {
            let input = &audio.values[frame * HOP..frame * HOP + WINDOW];
            let mean = input.iter().sum::<f32>() / WINDOW as f32;
            for i in 0..WINDOW {
                let value = input[i] - mean;
                let previous = input[i.saturating_sub(1)] - mean;
                samples[i] = (value - 0.97_f32 * previous) * window[i];
            }
            finite(&samples)?;
            for (bin, power) in powers.iter_mut().enumerate() {
                let mut re = 0.0;
                let mut im = 0.0;
                for (i, &sample) in samples.iter().enumerate() {
                    re += f64::from(sample) * self.real[bin * WINDOW + i];
                    im += f64::from(sample) * self.imag[bin * WINDOW + i];
                }
                // Match the FP32 complex/magnitude/power boundary.
                let re = re as f32;
                let im = im as f32;
                let magnitude = re.hypot(im);
                *power = magnitude * magnitude;
            }
            finite(&powers)?;
            for filter in self.tensors["mel_basis"].values.chunks_exact(BINS) {
                let energy = filter
                    .iter()
                    .zip(&powers)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>() as f32;
                values.push(energy.max(f32::EPSILON).ln());
            }
        }
        finite(&values)?;
        for channel in 0..MEL {
            let mean = (values
                .chunks_exact(MEL)
                .map(|row| f64::from(row[channel]))
                .sum::<f64>()
                / frames as f64) as f32;
            for row in values.chunks_exact_mut(MEL) {
                row[channel] -= mean;
            }
        }
        finite(&values)?;
        Ok(F32Npy {
            shape: vec![frames, MEL],
            values,
        })
    }

    fn bn(&self, x: &mut [f32], name: &str, channels: usize, relu: bool) -> Result<()> {
        finite(x)?;
        let mean = &self.tensors[&format!("{name}.running_mean")].values;
        let var = &self.tensors[&format!("{name}.running_var")].values;
        let weight = self.tensors.get(&format!("{name}.weight"));
        let bias = self.tensors.get(&format!("{name}.bias"));
        // Evaluation BatchNorm uses retained running variance (epsilon=1e-5).
        let scales: Vec<f32> = (0..channels)
            .map(|c| weight.map_or(1.0, |w| w.values[c]) / (var[c] + 1e-5).sqrt())
            .collect();
        for row in x.chunks_exact_mut(channels) {
            for c in 0..channels {
                let value = (row[c] - mean[c]) * scales[c] + bias.map_or(0.0, |b| b.values[c]);
                if !value.is_finite() {
                    return Err("CAMPPlus BatchNorm overflow".into());
                }
                row[c] = if relu { value.max(0.0) } else { value };
            }
        }
        Ok(())
    }

    fn conv2(&self, x: &Map, name: &str, stride: usize) -> Map {
        let weight = &self.tensors[&format!("{name}.weight")];
        let kernel = weight.shape[2];
        let pad = kernel / 2;
        let freq = x.freq.div_ceil(stride);
        let width = x.channels * kernel * kernel;
        // Tile im2col rows so workspace stays bounded as utterance length grows.
        // Each row is [input_channel, kernel_frequency, kernel_time], matching
        // the checkpoint's contiguous [out,in,kf,kt] weight order.
        let rows = freq * x.time;
        let mut output = Vec::with_capacity(rows * weight.shape[0]);
        for begin in (0..rows).step_by(128) {
            let count = (rows - begin).min(128);
            let mut columns = vec![0.0; count * width];
            for offset in 0..count {
                let f = (begin + offset) / x.time;
                let t = (begin + offset) % x.time;
                let row = &mut columns[offset * width..(offset + 1) * width];
                for c in 0..x.channels {
                    for kf in 0..kernel {
                        for kt in 0..kernel {
                            let sf = (f * stride + kf).wrapping_sub(pad);
                            let st = (t + kt).wrapping_sub(pad);
                            if sf < x.freq && st < x.time {
                                row[(c * kernel + kf) * kernel + kt] =
                                    x.values[(sf * x.time + st) * x.channels + c];
                            }
                        }
                    }
                }
            }
            output.extend(affine(&columns, weight, None));
        }
        Map {
            values: output,
            freq,
            time: x.time,
            channels: weight.shape[0],
        }
    }

    fn conv1(&self, x: &[f32], name: &str, stride: usize, dilation: usize) -> Vec<f32> {
        let weight = &self.tensors[&format!("{name}.weight")];
        let (input, kernel) = (weight.shape[1], weight.shape[2]);
        let frames = x.len() / input;
        let time = frames.div_ceil(stride);
        let bias = self.tensors.get(&format!("{name}.bias"));
        let width = input * kernel;
        if kernel == 1 && stride == 1 {
            return affine(x, weight, bias);
        }
        let pad = (kernel - 1) / 2 * dilation;
        let mut columns = vec![0.0; time * width];
        for t in 0..time {
            for c in 0..input {
                for k in 0..kernel {
                    let source = (t * stride + k * dilation).wrapping_sub(pad);
                    if source < frames {
                        columns[t * width + c * kernel + k] = x[source * input + c];
                    }
                }
            }
        }
        affine(&columns, weight, bias)
    }

    fn head(&self, features: &F32Npy) -> Result<Vec<f32>> {
        let time = features.shape[0];
        let mut values = vec![0.0; time * MEL];
        for t in 0..time {
            for f in 0..MEL {
                values[f * time + t] = features.values[t * MEL + f];
            }
        }
        let mut x = self.conv2(
            &Map {
                values,
                freq: MEL,
                time,
                channels: 1,
            },
            "head.conv1",
            1,
        );
        self.bn(&mut x.values, "head.bn1", 32, true)?;
        for stage in 1..=2 {
            for block in 0..2 {
                let prefix = format!("head.layer{stage}.{block}");
                let stride = if block == 0 { 2 } else { 1 };
                let mut residual = self.conv2(&x, &format!("{prefix}.conv1"), stride);
                self.bn(&mut residual.values, &format!("{prefix}.bn1"), 32, true)?;
                residual = self.conv2(&residual, &format!("{prefix}.conv2"), 1);
                self.bn(&mut residual.values, &format!("{prefix}.bn2"), 32, false)?;
                let shortcut = if block == 0 {
                    let mut v = self.conv2(&x, &format!("{prefix}.shortcut.0"), stride);
                    self.bn(&mut v.values, &format!("{prefix}.shortcut.1"), 32, false)?;
                    v
                } else {
                    x
                };
                for (value, &skip) in residual.values.iter_mut().zip(&shortcut.values) {
                    let sum = *value + skip;
                    if !sum.is_finite() {
                        return Err("CAMPPlus residual overflow".into());
                    }
                    *value = sum.max(0.0);
                }
                x = residual;
            }
        }
        x = self.conv2(&x, "head.conv2", 2);
        self.bn(&mut x.values, "head.bn2", 32, true)?;
        // Torch reshape [B,C,F,T] -> [B,C*F,T], then native time-major rows.
        let mut output = vec![0.0; time * 320];
        for t in 0..time {
            for c in 0..32 {
                for f in 0..10 {
                    output[t * 320 + c * 10 + f] = x.values[(f * time + t) * 32 + c];
                }
            }
        }
        Ok(output)
    }

    /// No L2 normalization: S3Gen applies its own normalization downstream.
    pub fn encode_features(&self, features: &F32Npy) -> Result<F32Npy> {
        validate_features(features)?;
        let mut x = self.conv1(&self.head(features)?, "xvector.tdnn.linear", 2, 1);
        self.bn(&mut x, "xvector.tdnn.nonlinear.batchnorm", 128, true)?;
        let frames = features.shape[0].div_ceil(2);
        let mut channels = 128;
        for (block, layers, dilation) in [(1, 12, 1), (2, 24, 2), (3, 16, 2)] {
            for layer in 1..=layers {
                let prefix = format!("xvector.block{block}.tdnnd{layer}");
                let mut input = x.clone();
                self.bn(
                    &mut input,
                    &format!("{prefix}.nonlinear1.batchnorm"),
                    channels,
                    true,
                )?;
                let mut bottleneck = self.conv1(&input, &format!("{prefix}.linear1"), 1, 1);
                self.bn(
                    &mut bottleneck,
                    &format!("{prefix}.nonlinear2.batchnorm"),
                    128,
                    true,
                )?;
                let mut local = self.conv1(
                    &bottleneck,
                    &format!("{prefix}.cam_layer.linear_local"),
                    1,
                    dilation,
                );
                // Context is constant within each 100-frame segment. Evaluate its
                // two pointwise layers once per segment instead of once per frame.
                finite(&local)?;
                let context = segment_context(&bottleneck, frames, 128);
                finite(&context)?;
                let mut gates = self.conv1(&context, &format!("{prefix}.cam_layer.linear1"), 1, 1);
                finite(&gates)?;
                for v in &mut gates {
                    *v = v.max(0.0);
                }
                gates = self.conv1(&gates, &format!("{prefix}.cam_layer.linear2"), 1, 1);
                finite(&gates)?;
                for v in &mut gates {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
                for t in 0..frames {
                    for c in 0..32 {
                        local[t * 32 + c] *= gates[(t / 100) * 32 + c];
                    }
                }
                let mut combined = Vec::with_capacity(frames * (channels + 32));
                for (old, new) in x.chunks_exact(channels).zip(local.chunks_exact(32)) {
                    combined.extend_from_slice(old);
                    combined.extend_from_slice(new);
                }
                x = combined;
                channels += 32;
            }
            let prefix = format!("xvector.transit{block}");
            self.bn(
                &mut x,
                &format!("{prefix}.nonlinear.batchnorm"),
                channels,
                true,
            )?;
            x = self.conv1(&x, &format!("{prefix}.linear"), 1, 1);
            channels /= 2;
            finite(&x)?;
        }
        self.bn(&mut x, "xvector.out_nonlinear.batchnorm", channels, true)?;
        let stats = statistics(&x, frames, channels)?;
        let mut embedding = self.conv1(&stats, "xvector.dense.linear", 1, 1);
        self.bn(
            &mut embedding,
            "xvector.dense.nonlinear.batchnorm",
            192,
            false,
        )?;
        finite(&embedding)?;
        Ok(F32Npy {
            shape: vec![1, 192],
            values: embedding,
        })
    }

    pub fn encode_pcm16k(&self, audio: &F32Npy) -> Result<F32Npy> {
        self.encode_features(&self.features(audio)?)
    }
}

/// Average pooling with ceil_mode divides the final segment by its actual size.
fn segment_context(x: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut global = vec![0.0; channels];
    for row in x.chunks_exact(channels) {
        for (sum, &v) in global.iter_mut().zip(row) {
            *sum += v;
        }
    }
    for sum in &mut global {
        *sum /= frames as f32;
    }
    let mut context = Vec::with_capacity(frames.div_ceil(100) * channels);
    for segment in x.chunks(100 * channels) {
        let count = segment.len() / channels;
        for (c, &mean) in global.iter().enumerate() {
            let sum: f64 = segment
                .chunks_exact(channels)
                .map(|row| f64::from(row[c]))
                .sum();
            context.push(mean + (sum / count as f64) as f32);
        }
    }
    context
}

/// Torch StatsPool uses unbiased standard deviation, with no epsilon/floor.
fn statistics(x: &[f32], frames: usize, channels: usize) -> Result<Vec<f32>> {
    if frames < 2 {
        return Err("CAMPPlus statistics require two TDNN frames".into());
    }
    let mut stats = vec![0.0; channels * 2];
    for c in 0..channels {
        let mean = x
            .chunks_exact(channels)
            .map(|r| f64::from(r[c]))
            .sum::<f64>()
            / frames as f64;
        stats[c] = mean as f32;
        let variance = x
            .chunks_exact(channels)
            .map(|r| (f64::from(r[c]) - mean).powi(2))
            .sum::<f64>()
            / (frames - 1) as f64;
        stats[channels + c] = variance.sqrt() as f32;
    }
    finite(&stats)?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn campplus_segment_statistics() {
        let x: Vec<f32> = (0..101).map(|n| n as f32).collect();
        let context = segment_context(&x, 101, 1);
        assert_eq!(context, vec![99.5, 150.0]);
        assert!(statistics(&[1.0], 1, 1).is_err());
        assert_eq!(
            statistics(&[1.0, 3.0], 2, 1).unwrap(),
            vec![2.0, 2_f32.sqrt()]
        );
        for bad in [
            F32Npy {
                shape: vec![2, 80],
                values: vec![0.0; 160],
            },
            F32Npy {
                shape: vec![3, 80],
                values: vec![0.0; 239],
            },
            F32Npy {
                shape: vec![3, 80],
                values: vec![f32::NAN; 240],
            },
        ] {
            assert!(validate_features(&bad).is_err());
        }
    }
}
