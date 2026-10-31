//! Complete strict-FP32 HiFT graph behind the existing tensor contract.
//! Exclusive owners retain engine weights, context and scratch between calls.
use crate::{
    conditioning::{asset_path, valid_sha, Asset},
    gpu::Stats,
    hift::HiFTOutput,
    npy::F32Npy,
    tensorrt::{check_engine, diagnostic, process_bridge},
    Result,
};
use libloading::Library;
use serde::Deserialize;
use std::{
    ffi::{c_char, c_int, c_void, CString},
    path::Path,
    ptr::NonNull,
    sync::Arc,
    time::Instant,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub schema_version: u32,
    pub decoder_kind: String,
    pub source_sha256: String,
    pub engine: Asset,
    pub bridge: Asset,
    pub max_frames: usize,
}
type Open = unsafe extern "C" fn(*const c_char, *mut c_char, usize) -> *mut c_void;
type Close = unsafe extern "C" fn(*mut c_void);
type Run = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    usize,
    usize,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut u64,
    *mut c_char,
    usize,
) -> c_int;
pub(crate) struct HiftGraph {
    handle: NonNull<c_void>,
    run_fn: Run,
    close_fn: Close,
    _library: Arc<Library>,
    max_frames: usize,
}
// SAFETY: only the service inference permit and cache mutex move this unique
// context between workers. Native calls attach its device; no Sync is exposed.
unsafe impl Send for HiftGraph {}
impl HiftGraph {
    pub fn open(root: &Path, manifest: &Manifest) -> Result<Self> {
        if manifest.schema_version != 1
            || manifest.decoder_kind != "s3gen_hift_trt_fp32"
            || !valid_sha(&manifest.source_sha256)
            || manifest.max_frames == 0
            || manifest.max_frames > i32::MAX as usize / 480
        {
            return Err("Unsupported HiFT TensorRT manifest".into());
        }
        let engine = asset_path(root, &manifest.engine)?;
        let bridge = asset_path(root, &manifest.bridge)?;
        check_engine(&engine, &manifest.engine.sha256)?;
        check_engine(&bridge, &manifest.bridge.sha256)?;
        let library = process_bridge(&bridge)?;
        let path =
            CString::new(engine.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
        // Resolve symbols before allocating, and retain the library until after
        // close so a moved/dropped service never unloads executing native code.
        unsafe {
            let open: Open = *library.get(b"cb_hift_open\0").map_err(|e| e.to_string())?;
            let run_fn: Run = *library.get(b"cb_hift_run\0").map_err(|e| e.to_string())?;
            let close_fn: Close = *library.get(b"cb_hift_close\0").map_err(|e| e.to_string())?;
            let mut error = [0 as c_char; 1024];
            let _trace = crate::trace::span("hift_engine_context_create");
            let handle = NonNull::new(open(path.as_ptr(), error.as_mut_ptr(), error.len()))
                .ok_or_else(|| format!("HiFT graph: {}", diagnostic(&error)))?;
            Ok(Self {
                handle,
                run_fn,
                close_fn,
                _library: library,
                max_frames: manifest.max_frames,
            })
        }
    }
    // Public dimensions and finite values are checked by S3GenHiFT first.
    pub fn run(
        &mut self,
        mel: &F32Npy,
        normal: &F32Npy,
        phases: &F32Npy,
        cache: &F32Npy,
    ) -> Result<(HiFTOutput, Stats)> {
        let frames = mel.shape[2];
        if frames > self.max_frames {
            return Err(format!(
                "HiFT frame length {frames} exceeds admitted TensorRT profile {}",
                self.max_frames
            ));
        }
        let samples = frames * 480;
        let mut output = HiFTOutput {
            pitch: F32Npy {
                shape: vec![1, frames],
                values: vec![0.; frames],
            },
            source: F32Npy {
                shape: vec![1, 1, samples],
                values: vec![0.; samples],
            },
            waveform: F32Npy {
                shape: vec![1, samples],
                values: vec![0.; samples],
            },
        };
        let mut counts = [0u64; 3];
        let mut error = [0 as c_char; 1024];
        let began = Instant::now();
        // SAFETY: all arrays are contiguous, validated, exact-sized and live
        // through the synchronous native output fence; owner access is unique.
        let status = unsafe {
            (self.run_fn)(
                self.handle.as_ptr(),
                mel.values.as_ptr(),
                normal.values.as_ptr(),
                phases.values.as_ptr(),
                cache.values.as_ptr(),
                frames,
                cache.values.len(),
                output.pitch.values.as_mut_ptr(),
                output.source.values.as_mut_ptr(),
                output.waveform.values.as_mut_ptr(),
                counts.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(format!("HiFT graph: {}", diagnostic(&error)));
        }
        if output
            .pitch
            .values
            .iter()
            .chain(&output.source.values)
            .chain(&output.waveform.values)
            .any(|v| !v.is_finite())
        {
            return Err("Nonfinite HiFT graph output".into());
        }
        // A prefix is a byte-exact request contract, not numerical tolerance.
        if output.source.values[..cache.values.len()] != cache.values {
            return Err("HiFT graph changed the request source prefix".into());
        }
        Ok((
            output,
            Stats {
                backend: "fp32-tensorrt",
                host_to_device_bytes: counts[0],
                device_to_host_bytes: counts[1],
                peak_scratch_bytes: usize::try_from(counts[2])
                    .map_err(|_| "HiFT scratch size overflow")?,
                gemm_and_download_ns: began.elapsed().as_nanos(),
                ..Stats::default()
            },
        ))
    }
}
impl Drop for HiftGraph {
    fn drop(&mut self) {
        // SAFETY: unique owner; close drains work before releasing resources.
        unsafe { (self.close_fn)(self.handle.as_ptr()) };
    }
}
