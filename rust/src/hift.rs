// HiFT/F0 inference derived from CosyVoice, ported into Rust in 2026.
// Copyright (c) 2024 Alibaba Inc (Xiang Lyu, Kai Hu).
// SPDX-License-Identifier: Apache-2.0 (see LICENSE).
//! FP32 native or retained TensorRT HiFT, unpadded requests at 24 kHz.
//! Explicit standard-normal excitation and harmonic phases make requests
//! reproducible without pretending to reproduce Torch's random-number stream.
//! Raw synthesis and the separate final trim/fade boundary share this module.
//! Source-cache prefixes are caller-owned; the immutable model retains no
//! request history. No reference preparation, watermark or service ownership.
use crate::{
    conditioning::{admitted_bytes, asset_path, valid_sha, Asset},
    conformer::product,
    npy::F32Npy,
    Result,
};
use serde::Deserialize;
use std::{cell::RefCell, collections::BTreeMap, path::Path};
const RATE: usize = 24000;
const HOP: usize = 480;
const FFT: usize = 16;
const FFT_HOP: usize = 4;
const BINS: usize = FFT / 2 + 1;
// Bound the im2col workspace independently of utterance length.
const TILE: usize = 128;
// Cap the combined packed input/result at 8 MiB per CUDA call. Larger row
// batches amortise unchanged weight uploads and blocking result transfers;
// CPU packing keeps its existing geometry. Device buffers still have their
// existing stage ownership and are released before the next model is admitted.
const TILE_BYTES: usize = 8 * 1024 * 1024;
fn tile_rows(cuda: bool, input: usize, output: usize) -> usize {
    if cuda {
        (TILE_BYTES / 4 / (input + output)).clamp(1, 1024)
    } else {
        TILE
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    decoder_kind: String,
    source_sha256: String,
    tensors: BTreeMap<String, Asset>,
}
pub struct S3GenHiFT {
    tensors: BTreeMap<String, F32Npy>,
    graph: Option<RefCell<crate::hift_trt::HiftGraph>>,
    pub source_sha256: String,
}
pub struct HiFTOutput {
    pub pitch: F32Npy,
    pub source: F32Npy,
    pub waveform: F32Npy,
}

/// One service-owned graph. Dependency identity uses the existing checksum and
/// inode/size/mtime/ctime policy; source prefixes remain entirely request-owned.
#[derive(Default)]
pub(crate) struct HiftCache {
    retained: Option<(Vec<crate::tensorrt::SourceIdentity>, String, S3GenHiFT)>,
}
impl HiftCache {
    pub(crate) fn clear(&mut self) {
        self.retained = None;
    }
    pub(crate) fn admit(&mut self, path: &Path, expected: &str) -> Result<bool> {
        let bytes = admitted_bytes(path, expected)?;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if header["decoder_kind"] != "s3gen_hift_trt_fp32" {
            self.clear();
            return Ok(false);
        }
        let manifest: crate::hift_trt::Manifest =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let paths = [
            path.to_path_buf(),
            asset_path(root, &manifest.engine)?,
            asset_path(root, &manifest.bridge)?,
        ];
        let sources: Vec<_> = paths
            .iter()
            .map(|p| crate::tensorrt::SourceIdentity::read(p))
            .collect::<Result<_>>()?;
        let hit = self
            .retained
            .as_ref()
            .is_some_and(|(s, pin, _)| *s == sources && pin.eq_ignore_ascii_case(expected));
        if !hit {
            self.clear(); // Retire before replacement, bounding peak ownership.
            let model = S3GenHiFT::open(path, expected)?;
            if sources.iter().any(|s| s.unchanged() != Ok(true)) {
                return Err("HiFT source changed during admission".into());
            }
            self.retained = Some((sources, expected.to_ascii_lowercase(), model));
        }
        let _trace = crate::trace::span(if hit {
            "hift_context_reuse"
        } else {
            "hift_context_admitted"
        });
        Ok(true)
    }
    pub(crate) fn run(
        &mut self,
        mel: &F32Npy,
        normal: &F32Npy,
        phases: &F32Npy,
        source: &F32Npy,
    ) -> Result<(HiFTOutput, crate::gpu::Stats)> {
        let model = &self.retained.as_ref().expect("HiFT graph admitted").2;
        // Reuse the public validation before calling the exclusive graph, and
        // capture its transfer counters without a second inference or context.
        let result = model.run_cached(mel, normal, phases, source);
        if result.is_err() {
            self.clear();
        }
        result
    }
}

/// An ordered PCM fragment. Absolute offsets let the transport prove that no
/// committed sample is repeated or skipped, including the final withheld tail.
pub struct WaveformFragment {
    pub offset_samples: usize,
    pub pcm: F32Npy,
}

/// Request-owned full-prefix vocoder state. The model uses noncausal attention,
/// so old waveform estimates may change: only the undecided tail is crossfaded.
/// This commit policy does not promise equality with whole-utterance synthesis.
pub struct WaveformOverlap {
    overlap: usize,
    source: F32Npy,
    tail: Vec<f32>,
    emitted: usize,
    total: usize,
    finalized: bool,
}
impl WaveformOverlap {
    pub fn new(overlap_samples: usize) -> Result<Self> {
        if overlap_samples == 0 {
            return Err("Waveform overlap must contain samples".into());
        }
        Ok(Self {
            overlap: overlap_samples,
            source: F32Npy {
                shape: vec![1, 1, 0],
                values: Vec::new(),
            },
            tail: Vec::new(),
            emitted: 0,
            total: 0,
            finalized: false,
        })
    }
    /// This cache starts at utterance sample zero. It must never be passed to
    /// a cropped mel window without an explicit coordinate transformation.
    pub fn cached_source(&self) -> &F32Npy {
        &self.source
    }

    pub fn push(&mut self, output: HiFTOutput, finalize: bool) -> Result<WaveformFragment> {
        let total = output
            .waveform
            .shape
            .get(1)
            .copied()
            .ok_or("Missing waveform sample dimension")?;
        checked(&output.waveform, &[1, total])?;
        checked(&output.source, &[1, 1, total])?;
        if self.finalized || total < self.total || (!finalize && total == self.total) {
            return Err("Waveform prefix is stale, shrinking, or already finalized".into());
        }
        if total < RATE / 25 || total % HOP != 0 {
            return Err("Waveform prefix must contain at least two complete mel frames".into());
        }
        checked(&output.pitch, &[1, total / HOP])?;
        if output.source.values[..self.source.values.len()] != self.source.values {
            return Err(
                "HiFT source prefix does not match this request's absolute coordinates".into(),
            );
        }
        // Apply spillover suppression at utterance origin only. Cropping after
        // this pass prevents each fragment acquiring a fresh initial silence.
        let finished = finish_waveform(&output.waveform)?;
        let mut pending = finished.values[self.emitted..].to_vec();
        let blend = self.tail.len();
        for (index, (next, previous)) in pending.iter_mut().zip(&self.tail).enumerate() {
            let weight = if blend == 1 {
                1.
            } else {
                (1. - (std::f32::consts::PI * index as f32 / (blend - 1) as f32).cos()) / 2.
            };
            *next = previous * (1. - weight) + *next * weight;
        }
        let retain = if finalize {
            0
        } else {
            self.overlap.min(pending.len())
        };
        let boundary = pending.len() - retain;
        let tail = pending.split_off(boundary);
        let offset_samples = self.emitted;
        self.emitted += pending.len();
        self.tail = tail;
        self.total = total;
        self.source = output.source;
        self.finalized = finalize;
        Ok(WaveformFragment {
            offset_samples,
            pcm: F32Npy {
                shape: vec![1, pending.len()],
                values: pending,
            },
        })
    }
}

/// Apply S3Token2Wav's inference-only reference-spillover suppression once to
/// a raw 24 kHz mono waveform. "Trim" means silence, not removal: keep every
/// sample and leave the tail bit-exact. Raw HiFT synthesis stays reusable.
pub fn finish_waveform(raw: &F32Npy) -> Result<F32Npy> {
    const SILENT: usize = RATE / 50;
    const WINDOW: usize = 2 * SILENT;
    if raw.shape.len() != 2
        || raw.shape[0] != 1
        || raw.shape[1] < WINDOW
        || raw.values.len() != raw.shape[1]
        || raw.values.iter().any(|v| !v.is_finite())
    {
        // The deployed wrapper also rejects shorter audio (its fixed 960-value
        // window cannot broadcast). Reject explicitly rather than truncate it.
        return Err("Final waveform must be finite and complete with shape [1,N], N >= 960".into());
    }
    // This immutable window belongs to the algorithm, not to a voice/request.
    // Retain it so repeated speech never recomputes trigonometric coefficients.
    static FADE: std::sync::OnceLock<[f32; WINDOW]> = std::sync::OnceLock::new();
    let fade = FADE.get_or_init(|| {
        let mut window = [0.0; WINDOW];
        let pi = std::f32::consts::PI;
        let step = pi / (SILENT - 1) as f32;
        for i in 0..SILENT {
            // Torch linspace fills each half from its nearest endpoint in FP32.
            let angle = if i < SILENT / 2 {
                pi - step * i as f32
            } else {
                step * (SILENT - i - 1) as f32
            };
            window[SILENT + i] = (angle.cos() + 1.0) / 2.0;
        }
        window
    });
    let mut values = raw.values.clone();
    for (sample, &scale) in values[..WINDOW].iter_mut().zip(fade) {
        *sample *= scale;
    }
    Ok(F32Npy {
        shape: raw.shape.clone(),
        values,
    })
}
fn required_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    fn layer(s: &mut BTreeMap<String, Vec<usize>>, p: &str, shape: Vec<usize>, bias: usize) {
        s.insert(format!("{p}.weight"), shape);
        s.insert(format!("{p}.bias"), vec![bias]);
    }
    layer(&mut shapes, "m_source.l_linear", vec![1, 9], 1);
    layer(&mut shapes, "conv_pre", vec![512, 80, 7], 512);
    layer(&mut shapes, "conv_post", vec![18, 64, 7], 18);
    for (i, (input, output, kernel)) in [(512, 256, 16), (256, 128, 11), (128, 64, 7)]
        .into_iter()
        .enumerate()
    {
        layer(
            &mut shapes,
            &format!("ups.{i}"),
            vec![input, output, kernel],
            output,
        );
    }
    for (i, (c, k)) in [(256, 30), (128, 6), (64, 1)].into_iter().enumerate() {
        layer(&mut shapes, &format!("source_downs.{i}"), vec![c, 18, k], c);
    }
    for i in 0..5 {
        layer(
            &mut shapes,
            &format!("f0_predictor.condnet.{}", i * 2),
            vec![512, if i == 0 { 80 } else { 512 }, 3],
            512,
        );
    }
    layer(&mut shapes, "f0_predictor.classifier", vec![1, 512], 1);
    for (p, c, k) in (0..3)
        .map(|i| {
            (
                format!("source_resblocks.{i}"),
                [256, 128, 64][i],
                [7, 7, 11][i],
            )
        })
        .chain((0..9).map(|i| {
            (
                format!("resblocks.{i}"),
                [256, 128, 64][i / 3],
                [3, 7, 11][i % 3],
            )
        }))
    {
        for j in 0..3 {
            for stack in ["convs1", "convs2"] {
                layer(&mut shapes, &format!("{p}.{stack}.{j}"), vec![c, c, k], c);
            }
            for stack in ["activations1", "activations2"] {
                shapes.insert(format!("{p}.{stack}.{j}.alpha"), vec![c]);
            }
        }
    }
    shapes.insert("stft_window".into(), vec![FFT]);
    shapes
}
fn checked(a: &F32Npy, shape: &[usize]) -> Result<()> {
    let n = shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or("HiFT shape overflow")?;
    if a.shape != shape || a.values.len() != n || a.values.iter().any(|v| !v.is_finite()) {
        return Err(format!("HiFT requires finite complete tensor {shape:?}"));
    }
    Ok(())
}
fn mel_frames(mel: &F32Npy) -> Result<usize> {
    let frames = mel.shape.get(2).copied().ok_or("HiFT missing mel frames")?;
    if frames == 0 {
        return Err("HiFT requires at least one mel frame".into());
    }
    checked(mel, &[1, 80, frames])?;
    // This is overflow protection, not an arbitrary utterance-length policy.
    frames
        .checked_mul(HOP)
        .and_then(|n| n.checked_mul(512))
        .filter(|&n| n <= isize::MAX as usize / 4)
        .ok_or("HiFT request dimensions overflow")?;
    Ok(frames)
}
/// Internal layout is time-major, sharing the existing Conformer matrix kernel.
struct Frames {
    c: usize,
    x: Vec<f32>,
}
impl Frames {
    fn len(&self) -> usize {
        self.x.len() / self.c
    }
    fn from_array(a: &F32Npy) -> Self {
        let c = a.shape[a.shape.len() - 2];
        let t = *a.shape.last().unwrap();
        let mut x = vec![0.; c * t];
        for i in 0..t {
            for j in 0..c {
                x[i * c + j] = a.values[j * t + i];
            }
        }
        Self { c, x }
    }
    fn leaky(&mut self, slope: f32) {
        for v in &mut self.x {
            if *v < 0. {
                *v *= slope;
            }
        }
    }
}
/// Tiled im2col covers ordinary, dilated and strided symmetric convolutions.
fn conv(x: &Frames, w: &F32Npy, b: &F32Npy, stride: usize, pad: usize, dilation: usize) -> Frames {
    let tile = tile_rows(crate::gpu::cuda_active(), x.c * w.shape[2], w.shape[0]);
    conv_tiled(x, w, b, stride, pad, dilation, tile)
}
fn conv_tiled(
    x: &Frames,
    w: &F32Npy,
    b: &F32Npy,
    stride: usize,
    pad: usize,
    dilation: usize,
    tile: usize,
) -> Frames {
    let _trace = crate::trace::span("hift_conv");
    let out = w.shape[0];
    let kernel = w.shape[2];
    let width = x.c * kernel;
    let length = (x.len() + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    let mut result = vec![0.; length * out];
    for start in (0..length).step_by(tile) {
        let _tile_trace = crate::trace::iteration("hift_conv_tile", start);
        let pack_trace = crate::trace::span("hift_conv_pack");
        let count = tile.min(length - start);
        let mut columns = vec![0.; count * width];
        for row in 0..count {
            for c in 0..x.c {
                for k in 0..kernel {
                    let at = ((start + row) * stride + k * dilation) as isize - pad as isize;
                    if at >= 0 && (at as usize) < x.len() {
                        columns[row * width + c * kernel + k] = x.x[at as usize * x.c + c];
                    }
                }
            }
        }
        drop(pack_trace);
        let values = product(&columns, &w.values, count, width, out);
        let _scatter_trace = crate::trace::span("hift_conv_scatter");
        for row in 0..count {
            for c in 0..out {
                result[(start + row) * out + c] = values[row * out + c] + b.values[c];
            }
        }
    }
    Frames { c: out, x: result }
}
/// Torch transposed-convolution weights are [Cin,Cout,K]. Compute each input's
/// channel reduction in the shared matrix kernel, then overlap-add its taps.
fn transpose(x: &Frames, w: &F32Npy, b: &F32Npy, stride: usize, pad: usize) -> Frames {
    let tile = tile_rows(crate::gpu::cuda_active(), x.c, w.shape[1] * w.shape[2]);
    transpose_tiled(x, w, b, stride, pad, tile)
}
fn transpose_tiled(
    x: &Frames,
    w: &F32Npy,
    b: &F32Npy,
    stride: usize,
    pad: usize,
    tile: usize,
) -> Frames {
    let _trace = crate::trace::span("hift_transpose");
    let weights_trace = crate::trace::span("hift_transpose_weights");
    let output = w.shape[1];
    let kernel = w.shape[2];
    let length = (x.len() - 1) * stride + kernel - 2 * pad;
    let mut weights = vec![0.; w.values.len()];
    for input in 0..x.c {
        for c in 0..output {
            for k in 0..kernel {
                weights[(c * kernel + k) * x.c + input] =
                    w.values[(input * output + c) * kernel + k];
            }
        }
    }
    drop(weights_trace);
    let mut result = vec![0.; length * output];
    for start in (0..x.len()).step_by(tile) {
        let _tile_trace = crate::trace::iteration("hift_transpose_tile", start);
        let count = tile.min(x.len() - start);
        let contributions = product(
            &x.x[start * x.c..(start + count) * x.c],
            &weights,
            count,
            x.c,
            output * kernel,
        );
        let _scatter_trace = crate::trace::span("hift_transpose_scatter");
        for row in 0..count {
            for c in 0..output {
                for k in 0..kernel {
                    let at = ((start + row) * stride + k) as isize - pad as isize;
                    if at >= 0 && (at as usize) < length {
                        result[at as usize * output + c] +=
                            contributions[(row * output + c) * kernel + k];
                    }
                }
            }
        }
    }
    for row in result.chunks_exact_mut(output) {
        for (v, &b) in row.iter_mut().zip(&b.values) {
            *v += b;
        }
    }
    Frames {
        c: output,
        x: result,
    }
}
fn snake(x: &Frames, alpha: &F32Npy) -> Frames {
    Frames {
        c: x.c,
        x: x.x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let a = alpha.values[i % x.c];
                let sine = (v * a).sin();
                v + (1. / (a + 1e-9)) * sine.powi(2)
            })
            .collect(),
    }
}
fn reflect(index: isize, length: usize) -> usize {
    // All FFT calls have at least 480 samples; only one reflection is needed.
    if index < 0 {
        (-index) as usize
    } else if index >= length as isize {
        (2 * length - 2) as isize as usize - index as usize
    } else {
        index as usize
    }
}
impl S3GenHiFT {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let _trace = crate::trace::span("tensor_model_load");
        let bytes = admitted_bytes(path, expected)?;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if header["decoder_kind"] == "s3gen_hift_trt_fp32" {
            let manifest: crate::hift_trt::Manifest =
                serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            let root = path.parent().unwrap_or_else(|| Path::new("."));
            let graph = crate::hift_trt::HiftGraph::open(root, &manifest)?;
            return Ok(Self {
                tensors: BTreeMap::new(),
                graph: Some(RefCell::new(graph)),
                source_sha256: manifest.source_sha256,
            });
        }
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|e| format!("HiFT manifest: {e}"))?;
        if manifest.schema_version != 1
            || manifest.decoder_kind != "s3gen_hift_fp32"
            || !valid_sha(&manifest.source_sha256)
        {
            return Err("Unsupported S3Gen HiFT export".into());
        }
        let shapes = required_shapes();
        if !manifest.tensors.keys().eq(shapes.keys()) {
            return Err("HiFT tensor names are incomplete or unexpected".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tensors = BTreeMap::new();
        for (name, asset) in &manifest.tensors {
            let _trace = crate::trace::named("tensor_admission", &name);
            let a = F32Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)?;
            checked(&a, &shapes[name])?;
            tensors.insert(name.clone(), a);
        }
        // The periodic Hann is part of the admitted model contract.
        for (i, &v) in tensors["stft_window"].values.iter().enumerate() {
            let expected =
                (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FFT as f64).cos()) as f32;
            if (v - expected).abs() > 1e-7 {
                return Err("HiFT requires periodic Hann window".into());
            }
        }
        Ok(Self {
            tensors,
            graph: None,
            source_sha256: manifest.source_sha256,
        })
    }
    fn tensor(&self, p: &str) -> &F32Npy {
        &self.tensors[p]
    }
    fn conv(&self, x: &Frames, p: &str, stride: usize, pad: usize, dilation: usize) -> Frames {
        let _trace = crate::trace::named("hift_layer", p);
        conv(
            x,
            self.tensor(&format!("{p}.weight")),
            self.tensor(&format!("{p}.bias")),
            stride,
            pad,
            dilation,
        )
    }
    fn residual(&self, mut x: Frames, p: &str) -> Frames {
        let _trace = crate::trace::named("hift_residual", p);
        for (i, dilation) in [1, 3, 5].into_iter().enumerate() {
            let _trace = crate::trace::iteration("hift_residual_iteration", i);
            let mut y = snake(&x, self.tensor(&format!("{p}.activations1.{i}.alpha")));
            let kernel = self.tensor(&format!("{p}.convs1.{i}.weight")).shape[2];
            y = self.conv(
                &y,
                &format!("{p}.convs1.{i}"),
                1,
                dilation * (kernel - 1) / 2,
                dilation,
            );
            y = snake(&y, self.tensor(&format!("{p}.activations2.{i}.alpha")));
            y = self.conv(&y, &format!("{p}.convs2.{i}"), 1, (kernel - 1) / 2, 1);
            for (v, y) in x.x.iter_mut().zip(y.x) {
                *v += y;
            }
        }
        x
    }
    pub fn predict_pitch(&self, mel: &F32Npy) -> Result<F32Npy> {
        if self.graph.is_some() {
            return Err("HiFT graph exposes complete synthesis only".into());
        }
        let _trace = crate::trace::span("hift_pitch");
        let frames = mel_frames(mel)?;
        let mut x = Frames::from_array(mel);
        for i in 0..5 {
            x = self.conv(&x, &format!("f0_predictor.condnet.{}", i * 2), 1, 1, 1);
            for v in &mut x.x {
                if *v < 0. {
                    *v = v.exp() - 1.;
                }
            }
        }
        let w = self.tensor("f0_predictor.classifier.weight");
        let b = self.tensor("f0_predictor.classifier.bias").values[0];
        let values = product(&x.x, &w.values, frames, 512, 1)
            .into_iter()
            .map(|v| (v + b).abs())
            .collect();
        let result = F32Npy {
            shape: vec![1, frames],
            values,
        };
        checked(&result, &[1, frames])?;
        Ok(result)
    }
    /// Generate the used source branch only. The deployed module's second noise
    /// result is discarded by HiFT; no native RNG is consumed for unused work.
    pub fn excite(&self, pitch: &F32Npy, normal: &F32Npy, phases: &F32Npy) -> Result<F32Npy> {
        if self.graph.is_some() {
            return Err("HiFT graph exposes complete synthesis only".into());
        }
        let _trace = crate::trace::span("hift_excitation");
        let frames = pitch
            .shape
            .get(1)
            .copied()
            .ok_or("HiFT missing pitch frames")?;
        if frames == 0 {
            return Err("HiFT empty pitch".into());
        }
        checked(pitch, &[1, frames])?;
        if pitch.values.iter().any(|&v| v < 0.) {
            return Err("HiFT negative pitch".into());
        }
        let samples = frames
            .checked_mul(HOP)
            .ok_or("HiFT excitation length overflow")?;
        checked(normal, &[1, 9, samples])?;
        checked(phases, &[1, 9, 1])?;
        let w = &self.tensor("m_source.l_linear.weight").values;
        let b = self.tensor("m_source.l_linear.bias").values[0];
        let mut harmonics = vec![0.; samples * 9];
        let mut sums = [0f64; 9];
        for i in 0..samples {
            let f0 = pitch.values[i / HOP];
            let uv = if f0 > 10. { 1. } else { 0. };
            let amp = uv * 0.003 + (1. - uv) * 0.1 / 3.;
            for h in 0..9 {
                // Torch cumsum reduces FP32 inputs in a wider accumulator and
                // rounds each emitted prefix to FP32 before fractional modulo.
                let frequency = (f0 * (h + 1) as f32) / RATE as f32;
                sums[h] += frequency as f64;
                let theta = ((sums[h] as f32) % 1.) * std::f32::consts::TAU;
                let phase = if h == 0 { 0. } else { phases.values[h] };
                harmonics[i * 9 + h] =
                    (0.1 * (theta + phase).sin()) * uv + amp * normal.values[h * samples + i];
            }
        }
        // Batch the source linear through the existing matrix kernel.
        let values = product(&harmonics, w, samples, 9, 1)
            .into_iter()
            .map(|v| (v + b).tanh())
            .collect();
        let result = F32Npy {
            shape: vec![1, 1, samples],
            values,
        };
        checked(&result, &[1, 1, samples])?;
        Ok(result)
    }
    /// Centered, reflect-padded real DFT16. The fixed tiny transform avoids a
    /// second FFT library; wide sums keep its FP32 boundary close to Torch FFT.
    pub fn source_spectrum(&self, source: &F32Npy) -> Result<F32Npy> {
        if self.graph.is_some() {
            return Err("HiFT graph exposes complete synthesis only".into());
        }
        let _trace = crate::trace::span("hift_source_spectrum");
        let samples = source
            .shape
            .get(2)
            .copied()
            .ok_or("HiFT missing source samples")?;
        if samples < FFT {
            return Err("HiFT source too short for reflection".into());
        }
        checked(source, &[1, 1, samples])?;
        let frames = samples / FFT_HOP + 1;
        let mut values = vec![0.; 18 * frames];
        let window = &self.tensor("stft_window").values;
        for frame in 0..frames {
            for bin in 0..BINS {
                let mut real = 0.;
                let mut imag = 0.;
                for (tap, &window) in window.iter().enumerate() {
                    let at = reflect(
                        (frame * FFT_HOP + tap) as isize - (FFT / 2) as isize,
                        samples,
                    );
                    let v = (source.values[at] * window) as f64;
                    let angle = std::f64::consts::TAU * (bin * tap) as f64 / FFT as f64;
                    real += v * angle.cos();
                    imag -= v * angle.sin();
                }
                values[bin * frames + frame] = real as f32;
                values[(bin + BINS) * frames + frame] = imag as f32;
            }
        }
        let output = F32Npy {
            shape: vec![1, 18, frames],
            values,
        };
        checked(&output, &[1, 18, frames])?;
        Ok(output)
    }
    fn inverse(&self, logits: &Frames) -> F32Npy {
        let _trace = crate::trace::span("hift_inverse");
        let frames = logits.len();
        let length = (frames - 1) * FFT_HOP + FFT;
        let mut signal = vec![0f64; length];
        let mut envelope = vec![0f64; length];
        let window = &self.tensor("stft_window").values;
        for frame in 0..frames {
            let mut real = [0f64; BINS];
            let mut imag = [0f64; BINS];
            for bin in 0..BINS {
                let magnitude = logits.x[frame * 18 + bin].exp().min(100.);
                let phase = logits.x[frame * 18 + bin + BINS].sin();
                real[bin] = (magnitude * phase.cos()) as f64;
                imag[bin] = (magnitude * phase.sin()) as f64;
            }
            for (tap, &win) in window.iter().enumerate() {
                let mut v = 0.;
                for bin in 0..BINS {
                    let angle = std::f64::consts::TAU * (bin * tap) as f64 / FFT as f64;
                    let factor = if bin == 0 || bin == FFT / 2 { 1. } else { 2. };
                    v += factor * (real[bin] * angle.cos() - imag[bin] * angle.sin());
                }
                let at = frame * FFT_HOP + tap;
                signal[at] += (v / FFT as f64) * win as f64;
                envelope[at] += (win as f64).powi(2);
            }
        }
        // Torch istft(center=True,length=None) removes n_fft/2 at each end.
        let values = (FFT / 2..length - FFT / 2)
            .map(|i| (signal[i] / envelope[i]) as f32)
            .map(|v| v.clamp(-0.99, 0.99))
            .collect();
        F32Npy {
            shape: vec![1, (frames - 1) * FFT_HOP],
            values,
        }
    }
    pub fn decode(&self, mel: &F32Npy, source: &F32Npy) -> Result<F32Npy> {
        if self.graph.is_some() {
            return Err("HiFT graph exposes complete synthesis only".into());
        }
        let _trace = crate::trace::span("hift_decode");
        let frames = mel_frames(mel)?;
        checked(source, &[1, 1, frames * HOP])?;
        let spectrum = Frames::from_array(&self.source_spectrum(source)?);
        let mut x = self.conv(&Frames::from_array(mel), "conv_pre", 1, 3, 1);
        for i in 0..3 {
            let _trace = crate::trace::iteration("hift_upsample_stage", i);
            x.leaky(0.1);
            let p = format!("ups.{i}");
            x = transpose(
                &x,
                self.tensor(&format!("{p}.weight")),
                self.tensor(&format!("{p}.bias")),
                [8, 5, 3][i],
                [4, 3, 2][i],
            );
            if i == 2 {
                // ReflectionPad1d((1,0)): prefix second frame, not first.
                let mut padded = Vec::with_capacity(x.x.len() + x.c);
                padded.extend_from_slice(&x.x[x.c..2 * x.c]);
                padded.extend(x.x);
                x.x = padded;
            }
            let mut si = self.conv(
                &spectrum,
                &format!("source_downs.{i}"),
                [15, 3, 1][i],
                [7, 1, 0][i],
                1,
            );
            si = self.residual(si, &format!("source_resblocks.{i}"));
            if x.x.len() != si.x.len() {
                return Err("HiFT source/upsample length mismatch".into());
            }
            for (v, s) in x.x.iter_mut().zip(si.x) {
                *v += s;
            }
            let mut combined = Vec::new();
            for j in 0..3 {
                let _trace = crate::trace::iteration("hift_residual_branch", j);
                let branch = self.residual(
                    Frames {
                        c: x.c,
                        x: x.x.clone(),
                    },
                    &format!("resblocks.{}", i * 3 + j),
                );
                if j == 0 {
                    combined = branch.x;
                } else {
                    for (v, b) in combined.iter_mut().zip(branch.x) {
                        *v += b;
                    }
                }
            }
            for v in &mut combined {
                *v /= 3.;
            }
            x.x = combined;
        }
        x.leaky(0.01);
        let logits = self.conv(&x, "conv_post", 1, 3, 1);
        if logits.x.iter().any(|v| !v.is_finite()) {
            return Err("HiFT nonfinite spectral logits".into());
        }
        let output = self.inverse(&logits);
        checked(&output, &[1, frames * HOP])?;
        Ok(output)
    }
    pub fn synthesize(&self, mel: &F32Npy, normal: &F32Npy, phases: &F32Npy) -> Result<HiFTOutput> {
        // The complete-response path is exactly cached inference with no prefix.
        self.synthesize_cached(
            mel,
            normal,
            phases,
            &F32Npy {
                shape: vec![1, 1, 0],
                values: Vec::new(),
            },
        )
    }

    /// Match deployed HiFTGenerator.inference: generate fresh excitation, replace
    /// its prefix with the preceding overlap source, then decode and return that
    /// same source. Cache length is measured in samples, with no frame rounding.
    /// The caller owns overlap selection and retirement between requests.
    pub fn synthesize_cached(
        &self,
        mel: &F32Npy,
        normal: &F32Npy,
        phases: &F32Npy,
        cache_source: &F32Npy,
    ) -> Result<HiFTOutput> {
        self.run_cached(mel, normal, phases, cache_source)
            .map(|v| v.0)
    }
    fn run_cached(
        &self,
        mel: &F32Npy,
        normal: &F32Npy,
        phases: &F32Npy,
        cache_source: &F32Npy,
    ) -> Result<(HiFTOutput, crate::gpu::Stats)> {
        let frames = mel_frames(mel)?;
        // Reject malformed overlap and randomness before expensive convolutions;
        // F32Npy's public fields cannot establish a valid contiguous tensor alone.
        let cached = cache_source
            .shape
            .get(2)
            .copied()
            .ok_or("HiFT cache missing sample dimension")?;
        checked(cache_source, &[1, 1, cached])?;
        if cached > frames * HOP {
            return Err("HiFT cached source exceeds generated source".into());
        }
        checked(normal, &[1, 9, frames * HOP])?;
        checked(phases, &[1, 9, 1])?;
        if let Some(graph) = &self.graph {
            return graph.borrow_mut().run(mel, normal, phases, cache_source);
        }
        let pitch = self.predict_pitch(mel)?;
        let mut source = self.excite(&pitch, normal, phases)?;
        source.values[..cached].copy_from_slice(&cache_source.values);
        let waveform = self.decode(mel, &source)?;
        Ok((
            HiFTOutput {
                pitch,
                source,
                waveform,
            },
            crate::gpu::Stats::default(),
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hift_graph_lifecycle() {
        use sha2::{Digest, Sha256};
        // Bridge code is intentionally process-resident. Give this fake ABI
        // its own process so another model's bridge test cannot rebind it.
        if std::env::var_os("MYTH_HIFT_GRAPH_LIFECYCLE_CHILD").is_none() {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "hift::tests::hift_graph_lifecycle",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("MYTH_HIFT_GRAPH_LIFECYCLE_CHILD", "1")
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "child stdout:\n{}\nchild stderr:\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }
        // A fake ABI isolates ownership and admission from CUDA arithmetic.
        // Its output includes every request marker, exposing stale buffers or
        // a retained source prefix without needing actual model weights.
        let root = std::env::temp_dir().join(format!("hift-cache-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let bridge_source = root.join("fake.c");
        let bridge = root.join("fake.so");
        std::fs::write(
            &bridge_source,
            r#"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
static int opens=0, live=0, calls=0;
int cb_hift_metric(int key) { return key==0?opens:key==1?live:calls; }
void* cb_hift_open(const char* path,char* err,size_t cap) {
    ++opens;++live;return malloc(1);
}
int cb_hift_run(void* h,const float* mel,const float* normal,const float* phases,
 const float* cache,size_t frames,size_t cached,float* pitch,float* source,
 float* waveform,uint64_t* counts,char* err,size_t cap) {
    ++calls;
    if(mel[0]==99) {snprintf(err,cap,"injected HiFT failure");return -1;}
    for(size_t i=0;i<frames;++i)pitch[i]=mel[i];
    for(size_t i=0;i<frames*480;++i) {
        source[i]=i<cached?cache[i]:normal[i]+phases[1];
        waveform[i]=source[i]+mel[0];
    }
    counts[0]=(80*frames+9*frames*480+9+cached)*4+4;
    counts[1]=(frames+2*frames*480)*4;
    counts[2]=counts[0]+counts[1];return 0;
}
void cb_hift_close(void* h) {--live;free(h);}
"#,
        )
        .unwrap();
        assert!(std::process::Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&bridge_source)
            .arg("-o")
            .arg(&bridge)
            .status()
            .unwrap()
            .success());
        let library = unsafe { libloading::Library::new(&bridge).unwrap() };
        let metric = unsafe {
            library
                .get::<unsafe extern "C" fn(i32) -> i32>(b"cb_hift_metric\0")
                .unwrap()
        };
        let get = |key| unsafe { metric(key) };
        let digest = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        let engine = root.join("fake.engine");
        std::fs::write(&engine, b"one").unwrap();
        let path = root.join("hift.json");
        let manifest = serde_json::json!({
            "schema_version":1,"decoder_kind":"s3gen_hift_trt_fp32",
            "source_sha256":"0".repeat(64),"max_frames":4,
            "engine":{"file":"fake.engine","sha256":digest(b"one")},
            "bridge":{"file":"fake.so","sha256":digest(&std::fs::read(&bridge).unwrap())}
        });
        let bytes = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let pin = digest(&bytes);
        let array = |shape: &[usize], mark| F32Npy {
            shape: shape.to_vec(),
            values: vec![mark; shape.iter().product()],
        };
        let mut cache = HiftCache::default();
        let mut first = None;
        for (frames, prefix, mark) in [(2, 481, 1.), (4, 1920, 2.), (1, 0, 3.), (3, 1, 4.)] {
            assert!(cache.admit(&path, &pin).unwrap());
            let source = array(&[1, 1, prefix], -mark);
            let (output, stats) = cache
                .run(
                    &array(&[1, 80, frames], mark),
                    &array(&[1, 9, frames * HOP], mark * 2.),
                    &array(&[1, 9, 1], mark * 3.),
                    &source,
                )
                .unwrap();
            assert_eq!(output.pitch.shape, [1, frames]);
            assert_eq!(output.pitch.values, vec![mark; frames]);
            assert_eq!(output.source.shape, [1, 1, frames * HOP]);
            assert_eq!(output.waveform.shape, [1, frames * HOP]);
            assert_eq!(&output.source.values[..prefix], &source.values);
            assert!(output.source.values[prefix..]
                .iter()
                .all(|&v| v == mark * 5.));
            assert!(output.waveform.values[..prefix].iter().all(|&v| v == 0.));
            assert!(output.waveform.values[prefix..]
                .iter()
                .all(|&v| v == mark * 6.));
            assert_eq!(stats.backend, "fp32-tensorrt");
            assert_eq!((get(0), get(1)), (1, 1), "warm calls reuse one owner");
            if first.is_none() {
                first = Some(output);
            }
        }
        // Different lengths and prefixes cannot influence a later replay.
        let replay = cache
            .run(
                &array(&[1, 80, 2], 1.),
                &array(&[1, 9, 2 * HOP], 2.),
                &array(&[1, 9, 1], 3.),
                &array(&[1, 1, 481], -1.),
            )
            .unwrap()
            .0;
        let first = first.unwrap();
        assert_eq!(replay.pitch.values, first.pitch.values);
        assert_eq!(replay.source.values, first.source.values);
        assert_eq!(replay.waveform.values, first.waveform.values);
        // Malformed fields and profile excess are rejected before the ABI.
        // Cache errors deliberately retire the owner even before native work.
        for malformed in 0..4 {
            cache.admit(&path, &pin).unwrap();
            let mut mel = array(&[1, 80, 2], 1.);
            let mut normal = array(&[1, 9, 2 * HOP], 2.);
            let phases = array(&[1, 9, 1], 3.);
            let mut source = array(&[1, 1, 0], 0.);
            match malformed {
                0 => {
                    mel.values.pop().unwrap();
                }
                1 => normal.values[0] = f32::NAN,
                2 => source = array(&[1, 1, 2 * HOP + 1], 0.),
                _ => {
                    mel = array(&[1, 80, 5], 1.);
                    normal = array(&[1, 9, 5 * HOP], 2.);
                }
            }
            let calls = get(2);
            assert!(cache.run(&mel, &normal, &phases, &source).is_err());
            assert_eq!(get(2), calls, "invalid request crossed the native ABI");
            assert_eq!(get(1), 0, "request error must retire the cached owner");
        }
        cache.admit(&path, &pin).unwrap();
        let calls = get(2);
        let error = cache
            .run(
                &array(&[1, 80, 1], 99.),
                &array(&[1, 9, HOP], 0.),
                &array(&[1, 9, 1], 0.),
                &array(&[1, 1, 0], 0.),
            )
            .err()
            .unwrap();
        assert!(error.contains("injected HiFT failure"));
        assert_eq!(get(2), calls + 1);
        assert_eq!(get(1), 0, "native failure must evict the poisoned owner");
        cache.admit(&path, &pin).unwrap();
        let opens = get(0);
        // A same-length rewrite with mtime restored must still invalidate the
        // admitted owner through ctime, then fail its original checksum pin.
        let modified = engine.metadata().unwrap().modified().unwrap();
        std::fs::write(&engine, b"two").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&engine)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        assert_eq!(engine.metadata().unwrap().len(), 3);
        assert_eq!(engine.metadata().unwrap().modified().unwrap(), modified);
        assert!(cache.admit(&path, &pin).unwrap_err().contains("checksum"));
        assert_eq!((get(0), get(1)), (opens, 0));
        std::fs::write(&engine, b"one").unwrap();
        assert!(cache.admit(&path, &pin).unwrap());
        assert_eq!((get(0), get(1)), (opens + 1, 1));
        cache.clear();
        assert_eq!(get(1), 0, "unload releases the graph context and weights");
    }
    #[test]
    fn hift_cached_source_rejects_invalid_before_inference() {
        // An empty model would panic if a rejected overlap reached any kernel.
        let model = S3GenHiFT {
            tensors: BTreeMap::new(),
            graph: None,
            source_sha256: String::new(),
        };
        let mel = F32Npy {
            shape: vec![1, 80, 1],
            values: vec![0.; 80],
        };
        let normal = F32Npy {
            shape: vec![1, 9, HOP],
            values: vec![0.; 9 * HOP],
        };
        let phases = F32Npy {
            shape: vec![1, 9, 1],
            values: vec![0.; 9],
        };
        for cache in [
            F32Npy {
                shape: vec![1],
                values: vec![],
            },
            F32Npy {
                shape: vec![1, 2, 0],
                values: vec![],
            },
            F32Npy {
                shape: vec![1, 1, 1],
                values: vec![],
            },
            F32Npy {
                shape: vec![1, 1, 1],
                values: vec![f32::INFINITY],
            },
            F32Npy {
                shape: vec![1, 1, HOP + 1],
                values: vec![0.; HOP + 1],
            },
        ] {
            assert!(model
                .synthesize_cached(&mel, &normal, &phases, &cache)
                .is_err());
        }
    }
    #[test]
    fn hift_rejects_malformed_assets() {
        use serde_json::{json, Value};
        use sha2::{Digest, Sha256};
        let root = std::env::temp_dir().join(format!("hift-admission-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let tensors: BTreeMap<_, _> = required_shapes()
            .keys()
            .map(|name| {
                (
                    name.clone(),
                    json!({"file":"missing.npy","sha256":"a".repeat(64)}),
                )
            })
            .collect();
        let good = json!({"schema_version":1,"decoder_kind":"s3gen_hift_fp32",
            "source_sha256":"a".repeat(64),"tensors":tensors});
        let open = |manifest: &Value| {
            let bytes = serde_json::to_vec(manifest).unwrap();
            let path = root.join("hift.json");
            std::fs::write(&path, &bytes).unwrap();
            S3GenHiFT::open(&path, &format!("{:x}", Sha256::digest(&bytes)))
                .err()
                .unwrap()
        };
        for (key, value) in [
            ("schema_version", json!(2)),
            ("decoder_kind", json!("other")),
            ("source_sha256", json!("bad")),
            ("extra", json!(0)),
        ] {
            let mut bad = good.clone();
            bad[key] = value;
            assert!(!open(&bad).contains("missing.npy"));
        }
        let first = required_shapes().keys().next().unwrap().clone();
        let mut bad = good.clone();
        bad["tensors"].as_object_mut().unwrap().remove(&first);
        assert!(open(&bad).contains("tensor names"));
        bad = good.clone();
        bad["tensors"]["extra"] = json!({"file":"extra.npy","sha256":"a".repeat(64)});
        assert!(open(&bad).contains("tensor names"));
        for path in ["../escape.npy", "/absolute.npy", "nested/array.npy"] {
            bad = good.clone();
            bad["tensors"][&first]["file"] = json!(path);
            assert!(open(&bad).contains("one local filename"));
        }
        assert!(open(&good).contains("missing.npy"));
        // Reject at the first small tensor before allocating any model weights.
        let mut header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }".to_string();
        while (10 + header.len() + 1) % 16 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0f32.to_le_bytes());
        std::fs::write(root.join("bad.npy"), &bytes).unwrap();
        bad = good.clone();
        bad["tensors"][&first]["file"] = json!("bad.npy");
        assert!(open(&bad).contains("checksum differs"));
        bad["tensors"][&first]["sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
        assert!(open(&bad).contains("finite complete tensor"));
        let n = bytes.len();
        bytes[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(root.join("bad.npy"), &bytes).unwrap();
        bad["tensors"][&first]["sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
        assert!(open(&bad).contains("Nonfinite"));
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn hift_spectrum_rejects_finite_overflow() {
        let window = F32Npy {
            shape: vec![16],
            values: (0..16)
                .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 16.).cos()) as f32)
                .collect(),
        };
        let model = S3GenHiFT {
            tensors: BTreeMap::from([("stft_window".into(), window)]),
            graph: None,
            source_sha256: "a".repeat(64),
        };
        let source = a(&[1, 1, 16], &[2.; 16]);
        let spec = model.source_spectrum(&source).unwrap();
        assert_eq!(spec.shape, [1, 18, 5]);
        assert!(spec.values[..5].iter().all(|&v| (v - 16.).abs() < 1e-6));
        assert!(model
            .source_spectrum(&a(&[1, 1, 16], &[f32::MAX; 16]))
            .is_err());
    }
    fn a(shape: &[usize], values: &[f32]) -> F32Npy {
        F32Npy {
            shape: shape.to_vec(),
            values: values.to_vec(),
        }
    }
    #[test]
    fn hift_tiling_preserves_geometry() {
        // Dyadic inputs make scalar accumulation exact here. Cross both the
        // old 128-row and new 1024-row boundaries, including partial last tiles.
        let x = Frames {
            c: 2,
            x: (0..1109 * 2).map(|i| (i % 7) as f32 * 0.125).collect(),
        };
        let b = a(&[3], &[0.25, -0.5, 0.125]);
        let w = a(
            &[3, 2, 3],
            &(0..18).map(|i| (i % 5) as f32 * 0.25).collect::<Vec<_>>(),
        );
        for (stride, pad, dilation) in [(1, 1, 1), (2, 2, 2), (3, 0, 1)] {
            let length = (x.len() + 2 * pad - dilation * 2 - 1) / stride + 1;
            let mut expected = vec![0.; length * 3];
            for t in 0..length {
                for out in 0..3 {
                    let mut sum = 0.;
                    for c in 0..2 {
                        for k in 0..3 {
                            let at = (t * stride + k * dilation) as isize - pad as isize;
                            if at >= 0 && (at as usize) < x.len() {
                                sum += x.x[at as usize * 2 + c] * w.values[(out * 2 + c) * 3 + k];
                            }
                        }
                    }
                    expected[t * 3 + out] = sum + b.values[out];
                }
            }
            for tile in [128, 455, 1024] {
                assert_eq!(
                    conv_tiled(&x, &w, &b, stride, pad, dilation, tile).x,
                    expected
                );
            }
        }
        let w = a(
            &[2, 3, 4],
            &(0..24).map(|i| (i % 5) as f32 * 0.25).collect::<Vec<_>>(),
        );
        for (stride, pad) in [(1, 1), (2, 1), (3, 0)] {
            let length = (x.len() - 1) * stride + 4 - 2 * pad;
            let mut expected = vec![0.; length * 3];
            // Preserve the actual reduction contract: sum channels for a tap,
            // then overlap-add in ascending input-row and tap order.
            for t in 0..x.len() {
                for out in 0..3 {
                    for k in 0..4 {
                        let at = (t * stride + k) as isize - pad as isize;
                        if at >= 0 && (at as usize) < length {
                            let mut sum = 0.;
                            for c in 0..2 {
                                sum += x.x[t * 2 + c] * w.values[(c * 3 + out) * 4 + k];
                            }
                            expected[at as usize * 3 + out] += sum;
                        }
                    }
                }
            }
            for row in expected.chunks_exact_mut(3) {
                for (v, bias) in row.iter_mut().zip(&b.values) {
                    *v += bias;
                }
            }
            for tile in [128, 455, 1024] {
                assert_eq!(transpose_tiled(&x, &w, &b, stride, pad, tile).x, expected);
            }
        }
        assert_eq!(tile_rows(false, 3584, 512), 128);
        for (input, output) in [(3584, 512), (512, 4096), (64, 64)] {
            let rows = tile_rows(true, input, output);
            assert!(rows > 128 && rows <= 1024);
            assert!(rows * (input + output) * 4 <= TILE_BYTES);
        }
    }
    #[test]
    fn hift_convolution_and_transpose_layout() {
        let x = Frames {
            c: 1,
            x: vec![1., 2., 3.],
        };
        let w = a(&[1, 1, 3], &[1., 2., 3.]);
        let b = a(&[1], &[0.]);
        assert_eq!(conv(&x, &w, &b, 1, 1, 1).x, vec![8., 14., 8.]);
        assert_eq!(conv(&x, &w, &b, 1, 2, 2).x, vec![11., 4., 7.]);
        let w = a(&[1, 1, 4], &[1., 2., 3., 4.]);
        assert_eq!(transpose(&x, &w, &b, 2, 1).x, vec![2., 5., 8., 9., 14., 9.]);
        let x = Frames {
            c: 2,
            x: vec![1., 2., 3., 4.],
        };
        let w = a(&[2, 1, 2], &[1., 2., 3., 4.]);
        assert_eq!(transpose(&x, &w, &b, 2, 0).x, vec![7., 10., 15., 22.]);
    }
    #[test]
    fn hift_request_contract_and_reflection() {
        assert_eq!(reflect(-1, 480), 1);
        assert_eq!(reflect(480, 480), 478);
        assert_eq!(required_shapes().len(), 247);
        assert!(mel_frames(&a(&[1, 80, 0], &[])).is_err());
        assert!(mel_frames(&a(&[1, 80, 1], &[0.])).is_err());
        assert!(checked(&a(&[1], &[f32::NAN]), &[1]).is_err());
        assert!(checked(&a(&[1], &[0.]), &[usize::MAX, 2]).is_err());
    }
}
