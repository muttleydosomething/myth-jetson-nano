//! Strict-FP32 Conformer graph at the existing encoder tensor boundary.
//! The service exclusively retains one context; request features remain local.
use crate::{
    conditioning::{asset_path, valid_sha, Asset},
    conformer::EncoderOutput,
    gpu::Stats,
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
    pub encoder_kind: String,
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
    usize,
    *mut f32,
    *mut u64,
    *mut c_char,
    usize,
) -> c_int;
pub(crate) struct ConformerGraph {
    handle: NonNull<c_void>,
    run_fn: Run,
    close_fn: Close,
    _library: Arc<Library>,
    max_frames: usize,
}
// SAFETY: the existing service inference permit and cache mutex serialize this
// unique owner. Native entry attaches its CUDA device when workers change.
unsafe impl Send for ConformerGraph {}
impl ConformerGraph {
    pub fn open(root: &Path, manifest: &Manifest) -> Result<Self> {
        if manifest.schema_version != 1
            || manifest.encoder_kind != "s3gen_conformer_trt_fp32"
            || !valid_sha(&manifest.source_sha256)
            || !(1..=850).contains(&manifest.max_frames)
        {
            return Err("Unsupported Conformer TensorRT manifest".into());
        }
        let engine = asset_path(root, &manifest.engine)?;
        let bridge = asset_path(root, &manifest.bridge)?;
        check_engine(&engine, &manifest.engine.sha256)?;
        check_engine(&bridge, &manifest.bridge.sha256)?;
        let library = process_bridge(&bridge)?;
        let path =
            CString::new(engine.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
        // Resolve every symbol before allocation; library lives past close.
        unsafe {
            let open: Open = *library
                .get(b"cb_conformer_open\0")
                .map_err(|e| e.to_string())?;
            let run_fn: Run = *library
                .get(b"cb_conformer_run\0")
                .map_err(|e| e.to_string())?;
            let close_fn: Close = *library
                .get(b"cb_conformer_close\0")
                .map_err(|e| e.to_string())?;
            let mut error = [0 as c_char; 1024];
            let _trace = crate::trace::span("conformer_engine_context_create");
            let handle = NonNull::new(open(path.as_ptr(), error.as_mut_ptr(), error.len()))
                .ok_or_else(|| format!("Conformer graph: {}", diagnostic(&error)))?;
            Ok(Self {
                handle,
                run_fn,
                close_fn,
                _library: library,
                max_frames: manifest.max_frames,
            })
        }
    }
    // S3GenConformer validates shape, finite values and arithmetic beforehand.
    pub fn run(&mut self, input: &F32Npy) -> Result<(EncoderOutput, Stats)> {
        let frames = input.shape[1];
        if frames > self.max_frames {
            return Err(format!(
                "Conformer frame length {frames} exceeds admitted TensorRT profile {}",
                self.max_frames
            ));
        }
        let mut output = EncoderOutput {
            features: F32Npy {
                shape: vec![1, 2 * frames, 512],
                values: vec![0.; 2 * frames * 512],
            },
            mask: F32Npy {
                shape: vec![1, 1, 2 * frames],
                values: vec![1.; 2 * frames],
            },
        };
        let mut counts = [0u64; 3];
        let mut error = [0 as c_char; 1024];
        let began = Instant::now();
        // SAFETY: exact-sized validated contiguous buffers remain live through
        // the synchronous native fence; exclusive owner prevents concurrent use.
        let status = unsafe {
            (self.run_fn)(
                self.handle.as_ptr(),
                input.values.as_ptr(),
                frames,
                output.features.values.as_mut_ptr(),
                counts.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(format!("Conformer graph: {}", diagnostic(&error)));
        }
        if output.features.values.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite Conformer graph output".into());
        }
        Ok((
            output,
            Stats {
                backend: "fp32-tensorrt",
                host_to_device_bytes: counts[0],
                device_to_host_bytes: counts[1],
                peak_scratch_bytes: usize::try_from(counts[2])
                    .map_err(|_| "Conformer scratch overflow")?,
                gemm_and_download_ns: began.elapsed().as_nanos(),
                ..Stats::default()
            },
        ))
    }
}
impl Drop for ConformerGraph {
    fn drop(&mut self) {
        // SAFETY: exactly one close; native teardown drains pending kernels.
        unsafe { (self.close_fn)(self.handle.as_ptr()) };
    }
}
