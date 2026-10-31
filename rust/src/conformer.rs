// Native inference derived from CosyVoice/ESPnet's UpsampleConformerEncoder.
// Ported and modified into Rust by Hailstorm contributors, 2026.
// Copyright 2019 Shigeki Karita; 2020-2021 Mobvoi Inc (Binbin Zhang, Di Wu);
// 2022 Xingchen Song; 2024 Alibaba Inc (Xiang Lyu).
// SPDX-License-Identifier: Apache-2.0
//! Single unpadded request, full attention, evaluation mode. Six blocks, causal
//! 2x upsampling, four blocks, final norm. No streaming cache or reference work.
//! The existing s3gen::S3GenFlowInput supplies the codec embedding boundary.
use crate::{
    conditioning::{admitted_bytes, asset_path, valid_sha, Asset},
    npy::F32Npy,
    Result,
};
use serde::Deserialize;
#[cfg(test)]
#[path = "conformer_cache_tests.rs"]
mod cache_tests;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const D: usize = 512;
const H: usize = 8;
const K: usize = D / H;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    encoder_kind: String,
    source_sha256: String,
    tensors: BTreeMap<String, Asset>,
}

/// Immutable admitted weights; all intermediate state belongs to encode().
pub struct S3GenConformer {
    tensors: BTreeMap<String, F32Npy>,
    graph: Option<RefCell<crate::conformer_trt::ConformerGraph>>,
    pub source_sha256: String,
}
pub struct EncoderOutput {
    pub features: F32Npy,
    /// Deployed encoder mask layout is [1,1,2T], distinct from flow's [1,T,1].
    pub mask: F32Npy,
}

/// One encoder graph shares the service inference permit and engine mutex.
/// Track original lookup routes as well as canonical identities: retargeting a
/// symlink must invalidate admission even if its previous target is unchanged.
#[derive(Default)]
pub(crate) struct ConformerCache {
    retained: Option<(
        Vec<(PathBuf, crate::tensorrt::SourceIdentity)>,
        String,
        S3GenConformer,
    )>,
}
impl ConformerCache {
    pub(crate) fn clear(&mut self) {
        self.retained = None;
    }
    pub(crate) fn admit(&mut self, path: &Path, expected: &str) -> Result<bool> {
        let result = self.try_admit(path, expected);
        if result.is_err() {
            self.clear();
        }
        result
    }
    fn try_admit(&mut self, path: &Path, expected: &str) -> Result<bool> {
        let bytes = admitted_bytes(path, expected)?;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if header["encoder_kind"] != "s3gen_conformer_trt_fp32" {
            self.clear();
            return Ok(false);
        }
        let manifest: crate::conformer_trt::Manifest =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let sources = [
            path.to_path_buf(),
            asset_path(root, &manifest.engine)?,
            asset_path(root, &manifest.bridge)?,
        ]
        .into_iter()
        .map(|p| Ok((p.clone(), crate::tensorrt::SourceIdentity::read(&p)?)))
        .collect::<Result<Vec<_>>>()?;
        let hit = self
            .retained
            .as_ref()
            .is_some_and(|(s, pin, _)| s == &sources && pin.eq_ignore_ascii_case(expected));
        if !hit {
            self.clear(); // Bound replacement peak; failed admission leaves no owner.
            let model = S3GenConformer::open(path, expected)?;
            for (route, source) in &sources {
                if crate::tensorrt::SourceIdentity::read(route)? != *source {
                    return Err("Conformer source changed during admission".into());
                }
            }
            self.retained = Some((sources, expected.to_ascii_lowercase(), model));
        }
        let _trace = crate::trace::span(if hit {
            "conformer_context_reuse"
        } else {
            "conformer_context_admitted"
        });
        Ok(true)
    }
    pub(crate) fn run(&mut self, input: &F32Npy) -> Result<(EncoderOutput, crate::gpu::Stats)> {
        let model = &self
            .retained
            .as_ref()
            .ok_or("Conformer graph not admitted")?
            .2;
        let result = model.run_cached(input);
        if result.is_err() {
            self.clear();
        }
        result
    }
}

fn required_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut result = BTreeMap::new();
    for prefix in ["embed", "up_embed"] {
        result.insert(format!("{prefix}.out.0.weight"), vec![D, D]);
        result.insert(format!("{prefix}.out.0.bias"), vec![D]);
        for suffix in ["weight", "bias"] {
            result.insert(format!("{prefix}.out.1.{suffix}"), vec![D]);
        }
    }
    result.insert("position_frequencies".into(), vec![D / 2]);
    for (name, kernel) in [
        ("pre_lookahead_layer.conv1", 4),
        ("pre_lookahead_layer.conv2", 3),
        ("up_layer.conv", 5),
    ] {
        result.insert(format!("{name}.weight"), vec![D, D, kernel]);
        result.insert(format!("{name}.bias"), vec![D]);
    }
    for suffix in ["weight", "bias"] {
        result.insert(format!("after_norm.{suffix}"), vec![D]);
    }
    for (stack, blocks) in [("encoders", 6), ("up_encoders", 4)] {
        for i in 0..blocks {
            let p = format!("{stack}.{i}");
            for (name, out, input, bias) in [
                ("self_attn.linear_q", D, D, true),
                ("self_attn.linear_k", D, D, true),
                ("self_attn.linear_v", D, D, true),
                ("self_attn.linear_out", D, D, true),
                ("self_attn.linear_pos", D, D, false),
                ("feed_forward.w_1", 2048, D, true),
                ("feed_forward.w_2", D, 2048, true),
            ] {
                result.insert(format!("{p}.{name}.weight"), vec![out, input]);
                if bias {
                    result.insert(format!("{p}.{name}.bias"), vec![out]);
                }
            }
            for name in ["norm_mha", "norm_ff"] {
                for suffix in ["weight", "bias"] {
                    result.insert(format!("{p}.{name}.{suffix}"), vec![D]);
                }
            }
            for name in ["pos_bias_u", "pos_bias_v"] {
                result.insert(format!("{p}.self_attn.{name}"), vec![H, K]);
            }
        }
    }
    result
}

/// Row-major A times row-major B-transpose. Bounds are checked before the one
/// native kernel call; output never aliases its borrowed input or weights.
pub(crate) fn product(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    if let Some(output) = crate::gpu::product(a, b, m, k, n) {
        return output;
    }
    let mut c = vec![0.0; m * n];
    // SAFETY: contiguous A[m,k], B[n,k], C[m,n]; transposed B strides; disjoint C.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    c
}
pub(crate) fn affine(x: &[f32], weight: &F32Npy, bias: Option<&F32Npy>) -> Vec<f32> {
    let n = weight.shape[0];
    let k = weight.values.len() / n;
    let mut output = product(x, &weight.values, x.len() / k, k, n);
    if let Some(bias) = bias {
        for row in output.chunks_exact_mut(n) {
            for (v, b) in row.iter_mut().zip(&bias.values) {
                *v += b;
            }
        }
    }
    output
}
pub(crate) fn norm(x: &[f32], weight: &F32Npy, bias: &F32Npy, eps: f32) -> Vec<f32> {
    // Both the 512-wide Conformer and 1280-wide codec use this same layer norm.
    let width = weight.values.len();
    let mut output = Vec::with_capacity(x.len());
    for row in x.chunks_exact(width) {
        // Wide reduction avoids cancellation; the layer boundary remains FP32.
        let mean = (row.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64) as f32;
        let variance = (row
            .iter()
            .map(|&v| (f64::from(v) - f64::from(mean)).powi(2))
            .sum::<f64>()
            / width as f64) as f32;
        let inv = (variance + eps).sqrt().recip();
        output.extend(
            row.iter()
                .zip(&weight.values)
                .zip(&bias.values)
                .map(|((&v, &w), &b)| ((v - mean) * inv) * w + b),
        );
    }
    output
}
fn add(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (x, y) in x.iter_mut().zip(y) {
        *x += y;
    }
}
/// Explicit im2col uses channel-major kernel order, as Torch Conv1d weights do.
/// left_pad=0 is future lookahead; kernel-1 is the causal history convention.
fn convolution(x: &[f32], weight: &F32Npy, bias: &F32Npy, left_pad: usize) -> Vec<f32> {
    let t = x.len() / D;
    let kernel = weight.shape[2];
    let mut columns = vec![0.0; t * D * kernel];
    for frame in 0..t {
        for channel in 0..D {
            for tap in 0..kernel {
                let at = frame as isize + tap as isize - left_pad as isize;
                if at >= 0 && (at as usize) < t {
                    columns[(frame * D + channel) * kernel + tap] = x[at as usize * D + channel];
                }
            }
        }
    }
    affine(&columns, weight, Some(bias))
}
/// Positions run T-1 ... 0 ... -(T-1). Torch-exported FP32 frequencies avoid
/// platform differences in exp(); only sin/cos and reductions require tolerance.
fn positions(t: usize, frequencies: &[f32]) -> Vec<f32> {
    let mut values = Vec::with_capacity((2 * t - 1) * D);
    for index in 0..2 * t - 1 {
        let position = (t as isize - 1 - index as isize) as f32;
        for &freq in frequencies {
            let angle = position * freq;
            values.push(angle.sin());
            values.push(angle.cos());
        }
    }
    values
}
fn validate_input(input: &F32Npy) -> Result<()> {
    let t = input.shape.get(1).copied().unwrap_or(0);
    if t == 0
        || input.shape != [1, t, D]
        || t.checked_mul(D) != Some(input.values.len())
        || input.values.iter().any(|v| !v.is_finite())
    {
        return Err("Conformer requires finite unpadded FP32 [1,T,512] input".into());
    }
    t.checked_mul(t)
        .and_then(|v| v.checked_mul(16))
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or("Conformer size overflow")?;
    Ok(())
}
impl S3GenConformer {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let _trace = crate::trace::span("tensor_model_load");
        let bytes = admitted_bytes(path, expected)?;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if header["encoder_kind"] == "s3gen_conformer_trt_fp32" {
            let manifest: crate::conformer_trt::Manifest =
                serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            return Ok(Self {
                tensors: BTreeMap::new(),
                graph: Some(RefCell::new(crate::conformer_trt::ConformerGraph::open(
                    path.parent().unwrap_or_else(|| Path::new(".")),
                    &manifest,
                )?)),
                source_sha256: manifest.source_sha256,
            });
        }
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|e| format!("Conformer manifest: {e}"))?;
        if manifest.schema_version != 1
            || manifest.encoder_kind != "s3gen_upsample_conformer_fp32"
            || !valid_sha(&manifest.source_sha256)
        {
            return Err("Unsupported S3Gen Conformer export".into());
        }
        let shapes = required_shapes();
        if !manifest.tensors.keys().eq(shapes.keys()) {
            return Err("Conformer tensor names differ from pinned architecture".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tensors = BTreeMap::new();
        for (name, shape) in shapes {
            let _trace = crate::trace::named("tensor_admission", &name);
            let asset = &manifest.tensors[&name];
            let array =
                F32Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)?;
            if array.shape != shape {
                return Err(format!("Conformer {name} requires {shape:?}"));
            }
            tensors.insert(name, array);
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
    fn linear(&self, x: &[f32], name: &str, bias: bool) -> Vec<f32> {
        affine(
            x,
            self.tensor(&format!("{name}.weight")),
            bias.then(|| self.tensor(&format!("{name}.bias"))),
        )
    }
    fn normalise(&self, x: &[f32], name: &str, eps: f32) -> Vec<f32> {
        norm(
            x,
            self.tensor(&format!("{name}.weight")),
            self.tensor(&format!("{name}.bias")),
            eps,
        )
    }
    fn conv(&self, x: &[f32], name: &str, left: usize) -> Vec<f32> {
        convolution(
            x,
            self.tensor(&format!("{name}.weight")),
            self.tensor(&format!("{name}.bias")),
            left,
        )
    }
    fn embed(&self, x: &[f32], name: &str) -> Vec<f32> {
        let x = self.linear(x, &format!("{name}.out.0"), true);
        self.normalise(&x, &format!("{name}.out.1"), 1e-5)
            .iter()
            .map(|v| v * (D as f32).sqrt())
            .collect()
    }
    fn attention(&self, x: &[f32], pos: &[f32], name: &str) -> Vec<f32> {
        let _trace = crate::trace::named("conformer_attention", name);
        let t = x.len() / D;
        let q = self.linear(x, &format!("{name}.linear_q"), true);
        let k = self.linear(x, &format!("{name}.linear_k"), true);
        let v = self.linear(x, &format!("{name}.linear_v"), true);
        let p = self.linear(pos, &format!("{name}.linear_pos"), false);
        let u = &self.tensor(&format!("{name}.pos_bias_u")).values;
        let b = &self.tensor(&format!("{name}.pos_bias_v")).values;
        let mut context = vec![0.0; t * D];
        // Process one head at a time. No eight-head quadratic score allocation.
        for head in 0..H {
            let head_rows = |values: &[f32]| -> Vec<f32> {
                values
                    .chunks_exact(D)
                    .flat_map(|r| r[head * K..(head + 1) * K].iter().copied())
                    .collect()
            };
            let qh = head_rows(&q);
            let kh = head_rows(&k);
            let ph = head_rows(&p);
            let qu: Vec<f32> = qh
                .iter()
                .enumerate()
                .map(|(i, v)| v + u[head * K + i % K])
                .collect();
            let qv: Vec<f32> = qh
                .iter()
                .enumerate()
                .map(|(i, v)| v + b[head * K + i % K])
                .collect();
            let mut scores = product(&qu, &kh, t, K, t);
            let relative = product(&qv, &ph, t, K, 2 * t - 1);
            // Algebraic equivalent of ESPnet rel_shift: relative position i-j
            // selects row T-1-i+j. This also handles T=1 without a reshape trick.
            for i in 0..t {
                let row = &mut scores[i * t..(i + 1) * t];
                for (j, score) in row.iter_mut().enumerate() {
                    *score =
                        (*score + relative[i * (2 * t - 1) + t - 1 - i + j]) / (K as f32).sqrt();
                }
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for value in row.iter_mut() {
                    *value = (*value - max).exp();
                }
                let total = row.iter().sum::<f32>();
                for value in row {
                    *value /= total;
                }
            }
            let vh = head_rows(&v);
            // Use the same backend for the attention-value reduction. Convert
            // V[t,K] to the shared product's B[K,t] row-major convention.
            let mut vt = vec![0.0; K * t];
            for frame in 0..t {
                for channel in 0..K {
                    vt[channel * t + frame] = vh[frame * K + channel];
                }
            }
            let result = product(&scores, &vt, t, t, K);
            for (out, row) in context.chunks_exact_mut(D).zip(result.chunks_exact(K)) {
                out[head * K..(head + 1) * K].copy_from_slice(row);
            }
        }
        self.linear(&context, &format!("{name}.linear_out"), true)
    }
    fn blocks(&self, mut x: Vec<f32>, stack: &str, count: usize) -> Vec<f32> {
        let pos = positions(x.len() / D, &self.tensor("position_frequencies").values);
        for i in 0..count {
            let prefix = format!("{stack}.{i}");
            let _trace = crate::trace::named("conformer_block", &prefix);
            let normal = self.normalise(&x, &format!("{prefix}.norm_mha"), 1e-12);
            add(
                &mut x,
                &self.attention(&normal, &pos, &format!("{prefix}.self_attn")),
            );
            let _ff_trace = crate::trace::span("conformer_feed_forward");
            let normal = self.normalise(&x, &format!("{prefix}.norm_ff"), 1e-12);
            let mut ff = self.linear(&normal, &format!("{prefix}.feed_forward.w_1"), true);
            for value in &mut ff {
                *value *= 1.0 / (1.0 + (-*value).exp());
            }
            add(
                &mut x,
                &self.linear(&ff, &format!("{prefix}.feed_forward.w_2"), true),
            );
        }
        x
    }
    pub fn encode(&self, input: &F32Npy) -> Result<EncoderOutput> {
        if self.graph.is_some() {
            self.run_cached(input).map(|(output, _)| output)
        } else {
            self.encode_native(input)
        }
    }
    pub(crate) fn run_cached(&self, input: &F32Npy) -> Result<(EncoderOutput, crate::gpu::Stats)> {
        if let Some(graph) = &self.graph {
            validate_input(input)?;
            return graph.borrow_mut().run(input);
        }
        crate::gpu::run_from_env(|| self.encode_native(input))
    }
    fn encode_native(&self, input: &F32Npy) -> Result<EncoderOutput> {
        validate_input(input)?;
        let t = input.shape.get(1).copied().unwrap_or(0);
        // Check every derived dimension before allocation/native strides. No
        // arbitrary utterance cap is introduced; memory grows with the request.
        t.checked_mul(t)
            .and_then(|v| v.checked_mul(16))
            .filter(|&v| v <= isize::MAX as usize)
            .ok_or("Conformer size overflow")?;
        let mut x = self.embed(&input.values, "embed");
        let mut ahead = self.conv(&x, "pre_lookahead_layer.conv1", 0);
        for v in &mut ahead {
            if *v < 0.0 {
                *v *= 0.01;
            }
        }
        add(&mut x, &self.conv(&ahead, "pre_lookahead_layer.conv2", 2));
        let x = self.blocks(x, "encoders", 6);
        let repeated: Vec<f32> = x
            .chunks_exact(D)
            .flat_map(|row| row.iter().chain(row).copied())
            .collect();
        let x = self.conv(&repeated, "up_layer.conv", 4);
        let x = self.embed(&x, "up_embed");
        let x = self.blocks(x, "up_encoders", 4);
        let values = self.normalise(&x, "after_norm", 1e-5);
        if values.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite Conformer output".into());
        }
        Ok(EncoderOutput {
            features: F32Npy {
                shape: vec![1, t * 2, D],
                values,
            },
            mask: F32Npy {
                shape: vec![1, 1, t * 2],
                values: vec![1.0; t * 2],
            },
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conformer_rejects_incomplete_corrupt_and_wrong_shape_assets() {
        use sha2::{Digest, Sha256};
        let root = std::env::temp_dir().join(format!("conformer-admission-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let sha = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        let load = |manifest: &serde_json::Value| {
            let bytes = serde_json::to_vec(manifest).unwrap();
            let path = root.join("conformer.json");
            std::fs::write(&path, &bytes).unwrap();
            S3GenConformer::open(&path, &sha(&bytes))
                .err()
                .expect("must reject")
        };
        let mut manifest = serde_json::json!({"schema_version":1,
            "encoder_kind":"s3gen_upsample_conformer_fp32", "source_sha256":"a".repeat(64),
            "tensors":{}});
        assert!(load(&manifest).contains("tensor names"));
        // All expected keys exist, but the first sorted tensor is a checksummed
        // [1] array instead of [512]. It must fail before loading any others.
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n";
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        std::fs::write(root.join("wrong.npy"), &bytes).unwrap();
        for key in required_shapes().keys() {
            manifest["tensors"][key] = serde_json::json!({"file":"wrong.npy","sha256":sha(&bytes)});
        }
        assert!(load(&manifest).contains("after_norm.bias requires"));
        let first = required_shapes().keys().next().unwrap().clone();
        manifest["tensors"][&first]["sha256"] = serde_json::json!("0".repeat(64));
        assert!(load(&manifest).contains("checksum differs"));
        manifest["tensors"][&first]["file"] = serde_json::json!("../wrong.npy");
        assert!(load(&manifest).contains("local"));
        std::fs::remove_dir_all(root).unwrap();
    }
    fn array(shape: &[usize], values: Vec<f32>) -> F32Npy {
        F32Npy {
            shape: shape.to_vec(),
            values,
        }
    }
    #[test]
    fn conformer_affine_and_causal_convolution() {
        assert_eq!(
            affine(
                &[1., 2., 3., 4.],
                &array(&[2, 2], vec![1., 2., -1., 1.]),
                Some(&array(&[2], vec![0.5, -0.5]))
            ),
            [5.5, 0.5, 11.5, 0.5]
        );
        let mut weight = vec![0.; D * D * 3];
        weight[0] = 1.;
        weight[1] = 2.;
        weight[2] = 3.;
        let mut input = vec![0.; 3 * D];
        input[0] = 1.;
        input[D] = 2.;
        input[2 * D] = 4.;
        let output = convolution(
            &input,
            &array(&[D, D, 3], weight),
            &array(&[D], vec![0.; D]),
            2,
        );
        assert_eq!([output[0], output[D], output[2 * D]], [3., 8., 17.]);
    }
    #[test]
    fn conformer_relative_shift_and_position_order() {
        // Literal deployed zero-pad/view/drop/view/slice, independently indexed.
        for t in [1, 2, 3, 7] {
            let n = 2 * t - 1;
            let original: Vec<usize> = (0..t * n).collect();
            let padded: Vec<usize> = original
                .chunks_exact(n)
                .flat_map(|r| std::iter::once(usize::MAX).chain(r.iter().copied()))
                .collect();
            for i in 0..t {
                for j in 0..t {
                    assert_eq!(padded[t + i * n + j], original[i * n + t - 1 - i + j]);
                }
            }
        }
        let pos = positions(2, &vec![1.; D / 2]);
        assert_eq!(pos[0], 1.0_f32.sin());
        assert_eq!(pos[D], 0.);
        assert_eq!(pos[2 * D], (-1.0_f32).sin());
    }
}
