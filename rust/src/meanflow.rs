// Meanflow/ConditionalDecoder derived from CosyVoice and Matcha, ported 2026.
// Copyright (c) 2024 Alibaba Inc (Xiang Lyu, Zhihao Du).
// SPDX-License-Identifier: Apache-2.0 (see LICENSE).
//! Immutable FP32 CPU estimator and Euler integration, batch one/unpadded/final.
//! Noise is explicit: callers can retain/replay it without claiming Torch RNG
//! parity. No CFG, cosine schedule, reference encoding, HiFT or waveform output.
use crate::{
    conditioning::{admitted_bytes, asset_path, valid_sha, Asset},
    conformer::{affine, product},
    npy::F32Npy,
    s3gen::S3GenMelInput,
    Result,
};
use serde::Deserialize;
use std::{cell::RefCell, collections::BTreeMap, path::Path};
const C: usize = 256;
const INNER: usize = 512;
const HEAD: usize = 64;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    decoder_kind: String,
    source_sha256: String,
    tensors: BTreeMap<String, Asset>,
}
/// All admitted tensors remain immutable; every request owns its intermediates.
pub struct S3GenMeanflow {
    tensors: BTreeMap<String, F32Npy>,
    graph: Option<RefCell<crate::meanflow_trt::MeanflowGraph>>,
    pub source_sha256: String,
}
pub struct MeanflowOutput {
    /// Includes reference frames, matching the CFM decoder's return boundary.
    pub full: F32Npy,
    /// Reference removed, matching CausalMaskedDiffWithXvec.inference's crop.
    pub generated: F32Npy,
}

/// One service-owned graph and its admitted dependency identities. CPU-only
/// CLI/models keep their original sequential lifetime. Every inference error
/// evicts the graph, because a failed native context may be poisoned.
#[derive(Default)]
pub(crate) struct MeanflowCache {
    retained: Option<(Vec<crate::tensorrt::SourceIdentity>, String, S3GenMeanflow)>,
}
impl MeanflowCache {
    pub(crate) fn clear(&mut self) {
        self.retained = None;
    }
    pub(crate) fn admit(&mut self, path: &Path, expected: &str) -> Result<bool> {
        // The small manifest remains checksummed on every entry. It declares
        // all dependencies; a manifest/pin/path change therefore cannot reuse
        // the previous graph by accident.
        let bytes = admitted_bytes(path, expected)?;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if header["decoder_kind"] != "s3gen_meanflow_trt_fp32" {
            self.clear();
            return Ok(false);
        }
        let manifest: crate::meanflow_trt::Manifest =
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
            self.clear(); // Never overlap old and replacement graph allocations.
            let model = S3GenMeanflow::open(path, expected)?;
            if sources.iter().any(|s| s.unchanged() != Ok(true)) {
                return Err("Meanflow source changed during admission".into());
            }
            self.retained = Some((sources, expected.to_ascii_lowercase(), model));
        }
        let _trace = crate::trace::span(if hit {
            "meanflow_context_reuse"
        } else {
            "meanflow_context_admitted"
        });
        Ok(true)
    }
    pub(crate) fn run(
        &mut self,
        input: &S3GenMelInput,
        noise: &F32Npy,
        steps: usize,
    ) -> Result<(MeanflowOutput, crate::gpu::Stats)> {
        let result = self
            .retained
            .as_ref()
            .expect("graph admitted")
            .2
            .run(input, noise, steps);
        if result.is_err() {
            self.clear();
        }
        result
    }
}
fn required_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    let mut linear = |name: String, out: usize, input: usize, kernel: Option<usize>, bias: bool| {
        let mut shape = vec![out, input];
        if let Some(k) = kernel {
            shape.push(k);
        }
        shapes.insert(format!("{name}.weight"), shape);
        if bias {
            shapes.insert(format!("{name}.bias"), vec![out]);
        }
    };
    for (name, out, input) in [
        ("time_mlp.linear_1", 1024, 320),
        ("time_mlp.linear_2", 1024, 1024),
        ("time_embed_mixer", 1024, 2048),
    ] {
        linear(name.into(), out, input, None, name != "time_embed_mixer");
    }
    for (stack, blocks) in [("down_blocks", 1), ("mid_blocks", 12), ("up_blocks", 1)] {
        for i in 0..blocks {
            let p = format!("{stack}.{i}");
            let input = match stack {
                "down_blocks" => 320,
                "up_blocks" => 512,
                _ => C,
            };
            for (block, input) in [("block1", input), ("block2", C)] {
                linear(format!("{p}.0.{block}.block.0"), C, input, Some(3), true);
                linear(format!("{p}.0.{block}.block.2"), C, 1, None, true);
            }
            linear(format!("{p}.0.mlp.1"), C, 1024, None, true);
            linear(format!("{p}.0.res_conv"), C, input, Some(1), true);
            for j in 0..4 {
                let q = format!("{p}.1.{j}");
                for norm in ["norm1", "norm3"] {
                    linear(format!("{q}.{norm}"), C, 1, None, true);
                }
                for name in ["to_q", "to_k", "to_v"] {
                    linear(format!("{q}.attn1.{name}"), INNER, C, None, false);
                }
                linear(format!("{q}.attn1.to_out.0"), C, INNER, None, true);
                linear(format!("{q}.ff.net.0.proj"), 1024, C, None, true);
                linear(format!("{q}.ff.net.2"), C, 1024, None, true);
            }
            if stack != "mid_blocks" {
                linear(format!("{p}.2"), C, C, Some(3), true);
            }
        }
    }
    linear("final_block.block.0".into(), C, C, Some(3), true);
    linear("final_block.block.2".into(), C, 1, None, true);
    linear("final_proj".into(), 80, C, Some(1), true);
    // Norm weights are vectors, not linear [C,1] tables.
    for (name, shape) in &mut shapes {
        if name.ends_with(".weight") && shape == &[C, 1] {
            *shape = vec![C];
        }
    }
    shapes.insert("time_frequencies".into(), vec![160]);
    shapes
}
fn checked_array(array: &F32Npy, shape: &[usize]) -> Result<()> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or("Meanflow shape overflow")?;
    if array.shape != shape
        || array.values.len() != count
        || array.values.iter().any(|v| !v.is_finite())
    {
        return Err(format!(
            "Meanflow requires finite complete tensor {shape:?}"
        ));
    }
    Ok(())
}
fn validate(input: &S3GenMelInput, noise: &F32Npy) -> Result<usize> {
    let frames = *input
        .mu
        .shape
        .get(2)
        .ok_or("Meanflow missing frame dimension")?;
    if frames == 0 {
        return Err("Meanflow requires at least one frame".into());
    }
    for a in [&input.mu, &input.cond, noise] {
        checked_array(a, &[1, 80, frames])?;
    }
    checked_array(&input.mask, &[1, 1, frames])?;
    checked_array(&input.speaker, &[1, 80])?;
    if input.mask.values.iter().any(|&v| v != 1.) {
        return Err("Meanflow admits unpadded final requests only".into());
    }
    if input.reference_frames.checked_add(input.generated_frames) != Some(frames) {
        return Err("Meanflow reference/generated frame accounting mismatch".into());
    }
    // Check attention products before any allocation/kernel, including forged
    // public F32Npy fields. Physical allocation limits are not a request policy.
    frames
        .checked_mul(frames)
        .and_then(|n| n.checked_mul(4))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or("Meanflow attention size overflow")?;
    Ok(frames)
}
fn mish(v: f32) -> f32 {
    let softplus = if v > 20. { v } else { (v.exp() + 1.).ln() };
    v * softplus.tanh()
}
fn silu(v: f32) -> f32 {
    v / (1. + (-v).exp())
}
// Torch's deployed GELU uses erf, not the tanh approximation. libm is present on
// both supported Linux targets; this adds no interpreter or tensor dependency.
#[link(name = "m")]
extern "C" {
    fn erff(x: f32) -> f32;
}
pub(crate) fn gelu(v: f32) -> f32 {
    // SAFETY: C libm erff is a pure scalar function with no pointer arguments.
    0.5 * v * (1. + unsafe { erff(v * std::f32::consts::FRAC_1_SQRT_2) })
}
fn add(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (x, y) in x.iter_mut().zip(y) {
        *x += y;
    }
}
impl S3GenMeanflow {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let bytes = admitted_bytes(path, expected)?;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("Meanflow manifest: {e}"))?;
        if header["decoder_kind"] == "s3gen_meanflow_trt_fp32" {
            let manifest: crate::meanflow_trt::Manifest = serde_json::from_slice(&bytes)
                .map_err(|e| format!("Meanflow graph manifest: {e}"))?;
            let root = path.parent().unwrap_or_else(|| Path::new("."));
            let graph = crate::meanflow_trt::MeanflowGraph::open(root, &manifest)?;
            return Ok(Self {
                tensors: BTreeMap::new(),
                graph: Some(RefCell::new(graph)),
                source_sha256: manifest.source_sha256,
            });
        }
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|e| format!("Meanflow manifest: {e}"))?;
        if manifest.schema_version != 1
            || manifest.decoder_kind != "s3gen_meanflow_fp32"
            || !valid_sha(&manifest.source_sha256)
        {
            return Err("Unsupported S3Gen meanflow export".into());
        }
        let shapes = required_shapes();
        if !manifest.tensors.keys().eq(shapes.keys()) {
            return Err("Meanflow tensor names are incomplete or unexpected".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tensors = BTreeMap::new();
        for (name, asset) in &manifest.tensors {
            let array =
                F32Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)?;
            checked_array(&array, &shapes[name])?;
            tensors.insert(name.clone(), array);
        }
        Ok(Self {
            tensors,
            graph: None,
            source_sha256: manifest.source_sha256,
        })
    }
    fn tensor(&self, name: &str) -> &F32Npy {
        &self.tensors[name]
    }
    fn linear(&self, x: &[f32], p: &str, bias: bool) -> Vec<f32> {
        affine(
            x,
            self.tensor(&format!("{p}.weight")),
            bias.then(|| self.tensor(&format!("{p}.bias"))),
        )
    }
    /// Internally all activations are time-major [T,C]. Only the public
    /// boundary is channel-major. LayerNorm always reduces over channels.
    fn norm(&self, x: &[f32], p: &str) -> Vec<f32> {
        let weight = &self.tensor(&format!("{p}.weight")).values;
        let bias = &self.tensor(&format!("{p}.bias")).values;
        x.chunks_exact(C)
            .flat_map(|row| {
                let mean = (row.iter().map(|&v| f64::from(v)).sum::<f64>() / C as f64) as f32;
                let variance = (row
                    .iter()
                    .map(|&v| (f64::from(v) - f64::from(mean)).powi(2))
                    .sum::<f64>()
                    / C as f64) as f32;
                let inv = (variance + 1e-5).sqrt().recip();
                row.iter()
                    .zip(weight)
                    .zip(bias)
                    .map(move |((&x, &w), &b)| ((x - mean) * inv) * w + b)
            })
            .collect()
    }
    /// im2col in Torch's channel/tap weight order, causal padding kernel-1.
    fn conv(&self, x: &[f32], p: &str) -> Vec<f32> {
        let weight = self.tensor(&format!("{p}.weight"));
        let input = weight.shape[1];
        let kernel = weight.shape[2];
        let frames = x.len() / input;
        if kernel == 1 {
            return self.linear(x, p, true);
        }
        let mut columns = vec![0.; frames * input * kernel];
        for t in 0..frames {
            for c in 0..input {
                for tap in 0..kernel {
                    if t + tap >= kernel - 1 {
                        columns[(t * input + c) * kernel + tap] =
                            x[(t + tap - (kernel - 1)) * input + c];
                    }
                }
            }
        }
        affine(&columns, weight, Some(self.tensor(&format!("{p}.bias"))))
    }
    fn block(&self, x: &[f32], p: &str) -> Vec<f32> {
        let conv = self.conv(x, &format!("{p}.block.0"));
        self.norm(&conv, &format!("{p}.block.2"))
            .into_iter()
            .map(mish)
            .collect()
    }
    fn residual(&self, x: &[f32], time: &[f32], p: &str) -> Vec<f32> {
        let mut h = self.block(x, &format!("{p}.block1"));
        let time: Vec<_> = time.iter().copied().map(mish).collect();
        let time = self.linear(&time, &format!("{p}.mlp.1"), true);
        for row in h.chunks_exact_mut(C) {
            add(row, &time);
        }
        let mut h = self.block(&h, &format!("{p}.block2"));
        add(&mut h, &self.conv(x, &format!("{p}.res_conv")));
        h
    }
    /// Eight heads of 64; attention uses all keys, regardless of convolution
    /// causality. Compute one head at a time to bound score-buffer memory.
    fn transformer(&self, x: &[f32], p: &str) -> Vec<f32> {
        let frames = x.len() / C;
        let norm = self.norm(x, &format!("{p}.norm1"));
        let q = self.linear(&norm, &format!("{p}.attn1.to_q"), false);
        let k = self.linear(&norm, &format!("{p}.attn1.to_k"), false);
        let v = self.linear(&norm, &format!("{p}.attn1.to_v"), false);
        let mut heads = vec![0.; frames * INNER];
        for h in 0..8 {
            let extract = |values: &[f32]| {
                values
                    .chunks_exact(INNER)
                    .flat_map(|row| row[h * HEAD..(h + 1) * HEAD].iter().copied())
                    .collect::<Vec<_>>()
            };
            let q = extract(&q);
            let k = extract(&k);
            let v = extract(&v);
            let mut scores = product(&q, &k, frames, HEAD, frames);
            for row in scores.chunks_exact_mut(frames) {
                let maximum = row
                    .iter()
                    .map(|v| *v * 0.125)
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.;
                for v in row.iter_mut() {
                    *v = (*v * 0.125 - maximum).exp();
                    sum += *v;
                }
                for v in row {
                    *v /= sum;
                }
            }
            let mut vt = vec![0.; HEAD * frames];
            for t in 0..frames {
                for c in 0..HEAD {
                    vt[c * frames + t] = v[t * HEAD + c];
                }
            }
            let head = product(&scores, &vt, frames, frames, HEAD);
            for t in 0..frames {
                heads[t * INNER + h * HEAD..t * INNER + (h + 1) * HEAD]
                    .copy_from_slice(&head[t * HEAD..(t + 1) * HEAD]);
            }
        }
        let mut out = x.to_vec();
        add(
            &mut out,
            &self.linear(&heads, &format!("{p}.attn1.to_out.0"), true),
        );
        let norm = self.norm(&out, &format!("{p}.norm3"));
        let ff: Vec<_> = self
            .linear(&norm, &format!("{p}.ff.net.0.proj"), true)
            .into_iter()
            .map(gelu)
            .collect();
        add(&mut out, &self.linear(&ff, &format!("{p}.ff.net.2"), true));
        out
    }
    fn time(&self, t: f32) -> Vec<f32> {
        let freq = &self.tensor("time_frequencies").values;
        let angles: Vec<_> = freq.iter().map(|&f| (1000. * t) * f).collect();
        let embed: Vec<_> = angles
            .iter()
            .map(|v| v.sin())
            .chain(angles.iter().map(|v| v.cos()))
            .collect();
        let h: Vec<_> = self
            .linear(&embed, "time_mlp.linear_1", true)
            .into_iter()
            .map(silu)
            .collect();
        self.linear(&h, "time_mlp.linear_2", true)
    }
    /// Evaluate one derivative; scalar times must be increasing in [0,1].
    pub fn estimate(&self, input: &S3GenMelInput, x: &F32Npy, t: f32, r: f32) -> Result<F32Npy> {
        let frames = validate(input, x)?;
        if !t.is_finite() || !r.is_finite() || t < 0. || r > 1. || t >= r {
            return Err("Meanflow times require 0 <= t < r <= 1".into());
        }
        if let Some(graph) = &self.graph {
            return Ok(graph.borrow_mut().run(input, x, 0, t, r)?.0);
        }
        let mut time = self.time(t);
        time.extend(self.time(r));
        let time = self.linear(&time, "time_embed_mixer", false);
        let mut hidden = Vec::with_capacity(frames * 320);
        for f in 0..frames {
            for tensor in [&x.values, &input.mu.values] {
                hidden.extend((0..80).map(|c| tensor[c * frames + f]));
            }
            hidden.extend_from_slice(&input.speaker.values);
            hidden.extend((0..80).map(|c| input.cond.values[c * frames + f]));
        }
        hidden = self.residual(&hidden, &time, "down_blocks.0.0");
        for i in 0..4 {
            hidden = self.transformer(&hidden, &format!("down_blocks.0.1.{i}"));
        }
        let skip = hidden.clone();
        hidden = self.conv(&hidden, "down_blocks.0.2");
        for mid in 0..12 {
            hidden = self.residual(&hidden, &time, &format!("mid_blocks.{mid}.0"));
            for i in 0..4 {
                hidden = self.transformer(&hidden, &format!("mid_blocks.{mid}.1.{i}"));
            }
        }
        let mut cat = Vec::with_capacity(frames * 512);
        for (row, skip) in hidden.chunks_exact(C).zip(skip.chunks_exact(C)) {
            cat.extend_from_slice(row);
            cat.extend_from_slice(skip);
        }
        hidden = self.residual(&cat, &time, "up_blocks.0.0");
        for i in 0..4 {
            hidden = self.transformer(&hidden, &format!("up_blocks.0.1.{i}"));
        }
        hidden = self.conv(&hidden, "up_blocks.0.2");
        hidden = self.block(&hidden, "final_block");
        let hidden = self.conv(&hidden, "final_proj");
        let mut values = vec![0.; 80 * frames];
        for f in 0..frames {
            for c in 0..80 {
                values[c * frames + f] = hidden[f * 80 + c];
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite meanflow estimator output".into());
        }
        Ok(F32Npy {
            shape: vec![1, 80, frames],
            values,
        })
    }
    /// Native distilled meanflow uses linear t/r and no guidance. Explicit
    /// noise includes reference frames; do not overwrite these with cond.
    pub fn decode(
        &self,
        input: &S3GenMelInput,
        noise: &F32Npy,
        steps: usize,
    ) -> Result<MeanflowOutput> {
        Ok(self.execute(input, noise, steps)?.0)
    }
    /// Select the complete graph before entering cuBLAS: creating a second
    /// matrix context would reintroduce setup overhead and misleading receipts.
    pub fn run(
        &self,
        input: &S3GenMelInput,
        noise: &F32Npy,
        steps: usize,
    ) -> Result<(MeanflowOutput, crate::gpu::Stats)> {
        if self.graph.is_some() {
            self.execute(input, noise, steps)
        } else {
            crate::gpu::run_from_env(|| self.decode(input, noise, steps))
        }
    }
    fn execute(
        &self,
        input: &S3GenMelInput,
        noise: &F32Npy,
        steps: usize,
    ) -> Result<(MeanflowOutput, crate::gpu::Stats)> {
        let frames = validate(input, noise)?;
        if steps == 0 || steps > i32::MAX as usize {
            return Err("Meanflow requires a positive representable step count".into());
        }
        let mut x = F32Npy {
            shape: noise.shape.clone(),
            values: noise.values.clone(),
        };
        let mut compute = crate::gpu::Stats::default();
        if let Some(graph) = &self.graph {
            (x, compute) = graph.borrow_mut().run(input, noise, steps, 0., 1.)?;
        } else {
            for i in 0..steps {
                let t = i as f32 / steps as f32;
                let r = (i + 1) as f32 / steps as f32;
                let derivative = self.estimate(input, &x, t, r)?;
                for (v, dx) in x.values.iter_mut().zip(derivative.values) {
                    *v += (r - t) * dx;
                    if !v.is_finite() {
                        return Err("Nonfinite meanflow Euler output".into());
                    }
                }
            }
        }
        let mut generated = Vec::with_capacity(80 * input.generated_frames);
        for channel in x.values.chunks_exact(frames) {
            generated.extend_from_slice(&channel[input.reference_frames..]);
        }
        Ok((
            MeanflowOutput {
                full: x,
                generated: F32Npy {
                    shape: vec![1, 80, input.generated_frames],
                    values: generated,
                },
            },
            compute,
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn meanflow_cache_reuses_context_and_retires_failed_or_changed_sources() {
        use sha2::{Digest, Sha256};
        let root = std::env::temp_dir().join(format!("meanflow-cache-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("fake.c");
        let bridge = root.join("fake.so");
        std::fs::write(
            &source,
            r#"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
static int opens=0, live=0;
int cb_metric(int key) { return key?live:opens; }
void* cb_meanflow_open(const char* path,char* err,size_t cap) {
    ++opens;++live;return malloc(1);
}
int cb_meanflow_run(void* h,const float* x,const float* mask,const float* mu,
 const float* spks,const float* cond,size_t frames,int steps,float t,float r,
 float* out,uint64_t* counts,char* err,size_t cap) {
    if(x[0]==99) {snprintf(err,cap,"injected failure");return -1;}
    for(size_t i=0;i<80*frames;++i)out[i]=x[i]+mu[i];
    counts[0]=counts[1]=counts[2]=0;return 0;
}
void cb_meanflow_close(void* h) {--live;free(h);}
"#,
        )
        .unwrap();
        assert!(std::process::Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&source)
            .arg("-o")
            .arg(&bridge)
            .status()
            .unwrap()
            .success());
        let lib = unsafe { libloading::Library::new(&bridge).unwrap() };
        let metric = unsafe {
            lib.get::<unsafe extern "C" fn(i32) -> i32>(b"cb_metric\0")
                .unwrap()
        };
        let get = |key| unsafe { metric(key) };
        let digest = |b: &[u8]| format!("{:x}", Sha256::digest(b));
        let engine = root.join("fake.engine");
        std::fs::write(&engine, b"one").unwrap();
        let path = root.join("meanflow.json");
        let manifest = serde_json::json!({"schema_version":1,"decoder_kind":"s3gen_meanflow_trt_fp32",
            "source_sha256":"0".repeat(64),"engine":{"file":"fake.engine","sha256":digest(b"one")},
            "bridge":{"file":"fake.so","sha256":digest(&std::fs::read(&bridge).unwrap())},"max_frames":4});
        let bytes = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let pin = digest(&bytes);
        let a = |v| F32Npy {
            shape: vec![1, 80, 1],
            values: vec![v; 80],
        };
        let input = S3GenMelInput {
            mu: a(2.),
            cond: a(0.),
            mask: F32Npy {
                shape: vec![1, 1, 1],
                values: vec![1.],
            },
            speaker: F32Npy {
                shape: vec![1, 80],
                values: vec![0.; 80],
            },
            reference_frames: 0,
            generated_frames: 1,
        };
        let mut cache = MeanflowCache::default();
        for v in [1., 3.] {
            assert!(cache.admit(&path, &pin).unwrap());
            assert_eq!(
                cache.run(&input, &a(v), 2).unwrap().0.full.values,
                vec![v + 2.; 80]
            );
        }
        assert_eq!((get(0), get(1)), (1, 1));
        assert!(cache.run(&input, &a(99.), 2).is_err());
        assert_eq!(get(1), 0, "failed context must be evicted");
        cache.admit(&path, &pin).unwrap();
        assert_eq!(get(0), 2);
        // Same length and restored mtime still changes ctime, so the old
        // admitted graph cannot conceal a rewrite under its original pin.
        let modified = engine.metadata().unwrap().modified().unwrap();
        std::fs::write(&engine, b"two").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&engine)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        assert!(cache.admit(&path, &pin).unwrap_err().contains("checksum"));
        assert_eq!(get(1), 0);
        std::fs::write(&engine, b"one").unwrap();
        cache.admit(&path, &pin).unwrap();
        assert_eq!(get(0), 3);
        cache.clear();
        assert_eq!(get(1), 0, "unload releases graph context and weights");
    }
    #[test]
    fn meanflow_rejects_malformed_assets() {
        use serde_json::{json, Value};
        use sha2::{Digest, Sha256};
        let root = std::env::temp_dir().join(format!("meanflow-admission-{}", std::process::id()));
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
        let good = json!({"schema_version":1,"decoder_kind":"s3gen_meanflow_fp32",
            "source_sha256":"a".repeat(64),"tensors":tensors});
        let open = |manifest: &Value| {
            let bytes = serde_json::to_vec(manifest).unwrap();
            let path = root.join("meanflow.json");
            std::fs::write(&path, &bytes).unwrap();
            S3GenMeanflow::open(&path, &format!("{:x}", Sha256::digest(&bytes)))
                .err()
                .unwrap()
        };
        for (key, value) in [
            ("schema_version", json!(2)),
            ("decoder_kind", json!("other")),
            ("source_sha256", json!("broken")),
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
        // Only the first (small bias) tensor is needed to prove rejection before
        // any large model allocation. Keep its digest valid for shape/NaN tests.
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
    fn meanflow_contract_and_kernels() {
        assert_eq!(required_shapes().len(), 912);
        assert_eq!(
            required_shapes()["up_blocks.0.0.res_conv.weight"],
            [256, 512, 1]
        );
        assert_eq!(
            required_shapes()["down_blocks.0.1.0.attn1.to_q.weight"],
            [512, 256]
        );
        assert!(mish(-100.).is_finite());
        assert!((gelu(1.) - 0.8413447).abs() < 1e-6);
        let a = || F32Npy {
            shape: vec![1, 80, 1],
            values: vec![0.; 80],
        };
        let mut input = S3GenMelInput {
            mu: a(),
            cond: a(),
            mask: F32Npy {
                shape: vec![1, 1, 1],
                values: vec![1.],
            },
            speaker: F32Npy {
                shape: vec![1, 80],
                values: vec![0.; 80],
            },
            reference_frames: 0,
            generated_frames: 1,
        };
        assert_eq!(validate(&input, &a()).unwrap(), 1);
        input.mask.values[0] = 0.;
        assert!(validate(&input, &a()).is_err());
        input.mask.values[0] = 1.;
        input.generated_frames = 2;
        assert!(validate(&input, &a()).is_err());
        input.generated_frames = 1;
        input.mu.shape[2] = usize::MAX;
        assert!(validate(&input, &a()).is_err());
    }
}
