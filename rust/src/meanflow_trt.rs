//! Complete FP32 meanflow graph, behind the existing model/validation boundary.
//! Native ownership is exclusive; the service may retain it between requests.
//! No host intermediate may
//! cross this ABI: inputs upload once, Euler steps ping-pong on the device.
use crate::{
    conditioning::{asset_path, valid_sha, Asset},
    gpu::Stats,
    npy::F32Npy,
    s3gen::S3GenMelInput,
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
    *const f32,
    usize,
    c_int,
    f32,
    f32,
    *mut f32,
    *mut u64,
    *mut c_char,
    usize,
) -> c_int;
pub(crate) struct MeanflowGraph {
    handle: NonNull<c_void>,
    run_fn: Run,
    close_fn: Close,
    _library: Arc<Library>,
    max_frames: usize,
}
// SAFETY: the service cache moves the unique native owner only under its
// mutex/inference permit. Native execute and teardown attach its CUDA device;
// no Sync implementation or concurrent context access is allowed.
unsafe impl Send for MeanflowGraph {}
impl MeanflowGraph {
    pub fn open(root: &Path, manifest: &Manifest) -> Result<Self> {
        if manifest.schema_version != 1
            || manifest.decoder_kind != "s3gen_meanflow_trt_fp32"
            || !valid_sha(&manifest.source_sha256)
            || manifest.max_frames == 0
            || manifest.max_frames > i32::MAX as usize
        {
            return Err("Unsupported meanflow TensorRT manifest".into());
        }
        let engine = asset_path(root, &manifest.engine)?;
        let bridge = asset_path(root, &manifest.bridge)?;
        check_engine(&engine, &manifest.engine.sha256)?;
        check_engine(&bridge, &manifest.bridge.sha256)?;
        let library = process_bridge(&bridge)?;
        let path =
            CString::new(engine.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
        // Resolve all symbols before allocation. The process bridge retains
        // plugin code; this owner releases context and weights exactly once.
        unsafe {
            let open: Open = *library
                .get(b"cb_meanflow_open\0")
                .map_err(|e| e.to_string())?;
            let run_fn: Run = *library
                .get(b"cb_meanflow_run\0")
                .map_err(|e| e.to_string())?;
            let close_fn: Close = *library
                .get(b"cb_meanflow_close\0")
                .map_err(|e| e.to_string())?;
            let mut error = [0 as c_char; 1024];
            let _trace = crate::trace::span("meanflow_engine_context_create");
            let handle = NonNull::new(open(path.as_ptr(), error.as_mut_ptr(), error.len()))
                .ok_or_else(|| format!("Meanflow graph: {}", diagnostic(&error)))?;
            Ok(Self {
                handle,
                run_fn,
                close_fn,
                _library: library,
                max_frames: manifest.max_frames,
            })
        }
    }
    // The parent model validates all public tensor fields before reaching here.
    // &mut plus RefCell at its owner prevents concurrent native-context use.
    pub fn run(
        &mut self,
        input: &S3GenMelInput,
        noise: &F32Npy,
        steps: usize,
        t: f32,
        r: f32,
    ) -> Result<(F32Npy, Stats)> {
        let frames = input.mu.shape[2];
        if frames > self.max_frames {
            return Err(format!(
                "Meanflow frame length {frames} exceeds admitted TensorRT profile {}",
                self.max_frames
            ));
        }
        let mut values = vec![0.; noise.values.len()];
        let mut counts = [0u64; 3];
        let mut error = [0 as c_char; 1024];
        let began = Instant::now();
        // SAFETY: validated contiguous arrays, exclusive native owner and exact
        // output/counter capacities are live for the entire synchronous call.
        let status = unsafe {
            (self.run_fn)(
                self.handle.as_ptr(),
                noise.values.as_ptr(),
                input.mask.values.as_ptr(),
                input.mu.values.as_ptr(),
                input.speaker.values.as_ptr(),
                input.cond.values.as_ptr(),
                frames,
                steps as c_int,
                t,
                r,
                values.as_mut_ptr(),
                counts.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(format!("Meanflow graph: {}", diagnostic(&error)));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite meanflow graph output".into());
        }
        let stats = Stats {
            backend: "fp32-tensorrt",
            host_to_device_bytes: counts[0],
            device_to_host_bytes: counts[1],
            peak_scratch_bytes: usize::try_from(counts[2])
                .map_err(|_| "Meanflow scratch size overflow")?,
            // This span includes uploads, graph execution and final download;
            // avoid suggesting it is a pure kernel measurement.
            gemm_and_download_ns: began.elapsed().as_nanos(),
            ..Stats::default()
        };
        Ok((
            F32Npy {
                shape: noise.shape.clone(),
                values,
            },
            stats,
        ))
    }
}
impl Drop for MeanflowGraph {
    fn drop(&mut self) {
        // SAFETY: exclusive ownership, native drains its stream before release.
        unsafe {
            (self.close_fn)(self.handle.as_ptr());
        }
    }
}
