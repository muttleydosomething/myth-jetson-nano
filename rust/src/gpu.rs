//! Request-scoped FP32 cuBLAS backend for the existing waveform matrix kernel.
//! Only explicitly selected waveform stages enter this scope. Voice preparation
//! and CPU-only development retain their established matrixmultiply route.
//! Host tensors stay authoritative: each call uploads current contents, so an
//! allocator-reused attention/transpose pointer can never select stale weights.
use crate::Result;
use libloading::Library;
use serde::Serialize;
use std::{cell::RefCell, ffi::c_void, ptr, time::Instant};

type Handle = *mut c_void;
type Create = unsafe extern "C" fn(*mut Handle) -> i32;
type Destroy = unsafe extern "C" fn(Handle) -> i32;
type MathMode = unsafe extern "C" fn(Handle, i32) -> i32;
type Malloc = unsafe extern "C" fn(*mut Handle, usize) -> i32;
type Free = unsafe extern "C" fn(Handle) -> i32;
type Copy = unsafe extern "C" fn(Handle, *const c_void, usize, i32) -> i32;
type Sync = unsafe extern "C" fn() -> i32;
type Gemm = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    *const f32,
    i32,
    *const f32,
    i32,
    *const f32,
    *mut f32,
    i32,
) -> i32;

#[derive(Clone, Debug, Serialize)]
pub struct HostTrim {
    pub rss_before_bytes: Option<u64>,
    pub rss_after_bytes: Option<u64>,
}
// The production Jetson path already uses malloc_trim at sequential model
// boundaries. Rust dropping a Vec returns ownership to glibc, but does not
// guarantee those pages return to Orin's shared CPU/GPU RAM budget.
fn trim_host_workspace() -> HostTrim {
    fn resident() -> Option<u64> {
        let pages = std::fs::read_to_string("/proc/self/statm")
            .ok()?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()?;
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        pages.checked_mul(u64::try_from(size).ok()?)
    }
    let before = resident();
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    HostTrim {
        rss_before_bytes: before,
        rss_after_bytes: resident(),
    }
}
struct HostWorkspaceLease(bool);
impl HostWorkspaceLease {
    fn finish(&mut self) -> HostTrim {
        self.0 = false;
        trim_host_workspace()
    }
}
impl Drop for HostWorkspaceLease {
    fn drop(&mut self) {
        if self.0 {
            trim_host_workspace();
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Stats {
    pub backend: &'static str,
    pub gemm_calls: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    /// Three reusable scratch buffers only; CUDA/cuBLAS context overhead is
    /// measured separately by the Borg process/cgroup acceptance proof.
    pub peak_scratch_bytes: usize,
    pub host_workspace_trims: Vec<HostTrim>,
    /// Host wall times at existing blocking boundaries; no extra CUDA fences.
    /// GEMM completes during the blocking download, so these are deliberately
    /// combined rather than misreported as independent GPU kernel timings.
    pub setup_ns: u128,
    pub allocation_ns: u128,
    pub upload_ns: u128,
    pub gemm_and_download_ns: u128,
    pub teardown_ns: u128,
}
impl Default for Stats {
    fn default() -> Self {
        Self {
            backend: "fp32-cpu",
            gemm_calls: 0,
            host_to_device_bytes: 0,
            device_to_host_bytes: 0,
            peak_scratch_bytes: 0,
            host_workspace_trims: Vec::new(),
            setup_ns: 0,
            allocation_ns: 0,
            upload_ns: 0,
            gemm_and_download_ns: 0,
            teardown_ns: 0,
        }
    }
}
#[derive(Default)]
struct Buffer {
    pointer: Handle,
    bytes: usize,
}

struct Cuda {
    // Libraries outlive every copied ABI pointer and native resource.
    _blas: Library,
    _runtime: Library,
    handle: Handle,
    malloc: Malloc,
    free: Free,
    copy: Copy,
    sync: Sync,
    destroy: Destroy,
    gemm: Gemm,
    buffers: [Buffer; 3],
    stats: Stats,
}
fn status(code: i32, operation: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{operation} failed (native status {code})"))
    }
}
impl Cuda {
    fn open() -> Result<Self> {
        let _trace = crate::trace::span("gpu_setup");
        let timer = Instant::now();
        // SAFETY: the installed NVIDIA libraries expose these documented C ABIs.
        // Names are fixed system dependencies, never model-supplied code paths.
        unsafe {
            let runtime = Library::new("libcudart.so").map_err(|e| format!("CUDA runtime: {e}"))?;
            let blas = Library::new("libcublas.so").map_err(|e| format!("cuBLAS: {e}"))?;
            let create: Create = *blas.get(b"cublasCreate_v2\0").map_err(|e| e.to_string())?;
            let math: MathMode = *blas
                .get(b"cublasSetMathMode\0")
                .map_err(|e| e.to_string())?;
            let mut result = Self {
                malloc: *runtime.get(b"cudaMalloc\0").map_err(|e| e.to_string())?,
                free: *runtime.get(b"cudaFree\0").map_err(|e| e.to_string())?,
                copy: *runtime.get(b"cudaMemcpy\0").map_err(|e| e.to_string())?,
                sync: *runtime
                    .get(b"cudaDeviceSynchronize\0")
                    .map_err(|e| e.to_string())?,
                destroy: *blas.get(b"cublasDestroy_v2\0").map_err(|e| e.to_string())?,
                gemm: *blas.get(b"cublasSgemm_v2\0").map_err(|e| e.to_string())?,
                handle: ptr::null_mut(),
                buffers: Default::default(),
                stats: Stats {
                    backend: "fp32-cuda-cublas",
                    ..Stats::default()
                },
                _blas: blas,
                _runtime: runtime,
            };
            status(create(&mut result.handle), "cublasCreate")?;
            // CUBLAS_PEDANTIC_MATH=2: preserve FP32 throughout, no TF32/FP16.
            status(math(result.handle, 2), "cublasSetMathMode(FP32 pedantic)")?;
            result.stats.setup_ns = timer.elapsed().as_nanos();
            Ok(result)
        }
    }
    fn reserve(&mut self, index: usize, bytes: usize) -> Result<()> {
        if bytes <= self.buffers[index].bytes {
            return Ok(());
        }
        // Calls are synchronous. Release the preceding capacity before growing,
        // so growth never temporarily duplicates a large scratch allocation.
        let buffer = &mut self.buffers[index];
        unsafe {
            if !buffer.pointer.is_null() {
                status((self.free)(buffer.pointer), "cudaFree(grow)")?;
                buffer.pointer = ptr::null_mut();
                buffer.bytes = 0;
            }
            status((self.malloc)(&mut buffer.pointer, bytes), "cudaMalloc")?;
        }
        buffer.bytes = bytes;
        self.stats.peak_scratch_bytes = self
            .stats
            .peak_scratch_bytes
            .max(self.buffers.iter().map(|b| b.bytes).sum());
        Ok(())
    }
    fn product(&mut self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        let _trace = crate::trace::span("gpu_product");
        let (mi, ki, ni, count) = dimensions(a.len(), b.len(), m, k, n)?;
        if m == 0 || n == 0 || k == 0 {
            return Ok(vec![0.; count]);
        }
        let bytes = [a.len() * 4, b.len() * 4, count * 4];
        let allocate_trace = crate::trace::span("gpu_allocate");
        let timer = Instant::now();
        for (index, &bytes) in bytes.iter().enumerate() {
            self.reserve(index, bytes)?;
        }
        let [da, db, dc] = self.buffers.each_ref().map(|b| b.pointer);
        let mut output = vec![0.; count];
        self.stats.allocation_ns += timer.elapsed().as_nanos();
        drop(allocate_trace);
        let (alpha, beta) = (1f32, 0f32);
        // SAFETY: dimensions and byte counts checked; buffers own sufficient
        // device capacity. Blocking D2H also completes all preceding GEMM work.
        unsafe {
            let upload_trace = crate::trace::span("gpu_upload");
            let timer = Instant::now();
            status(
                (self.copy)(da, a.as_ptr().cast(), bytes[0], 1),
                "cudaMemcpy(A,H2D)",
            )?;
            status(
                (self.copy)(db, b.as_ptr().cast(), bytes[1], 1),
                "cudaMemcpy(B,H2D)",
            )?;
            self.stats.upload_ns += timer.elapsed().as_nanos();
            drop(upload_trace);
            let _compute_trace = crate::trace::span("gpu_gemm_download");
            let timer = Instant::now();
            // cuBLAS is column-major: C^T[n,m] = B[n,k] * A^T[k,m].
            // Row-major B[n,k] is viewed as column-major [k,n], then transposed.
            status(
                (self.gemm)(
                    self.handle,
                    1,
                    0,
                    ni,
                    mi,
                    ki,
                    &alpha,
                    db.cast(),
                    ki,
                    da.cast(),
                    ki,
                    &beta,
                    dc.cast(),
                    ni,
                ),
                "cublasSgemm",
            )?;
            status(
                (self.copy)(output.as_mut_ptr().cast(), dc, bytes[2], 2),
                "cudaMemcpy(C,D2H)",
            )?;
            self.stats.gemm_and_download_ns += timer.elapsed().as_nanos();
        }
        self.stats.gemm_calls += 1;
        self.stats.host_to_device_bytes += (bytes[0] + bytes[1]) as u64;
        self.stats.device_to_host_bytes += bytes[2] as u64;
        Ok(output)
    }
    fn finish(&mut self) -> Result<()> {
        let _trace = crate::trace::span("gpu_teardown");
        let timer = Instant::now();
        // Completion/cleanup is inside stage ownership and therefore inside the
        // service inference permit, including cancellation and failed requests.
        let mut result = unsafe { status((self.sync)(), "cudaDeviceSynchronize") };
        // Attempt every release even if an earlier one fails. Normal success
        // includes teardown; Drop remains the unwinding/partial-load fallback.
        for buffer in &mut self.buffers {
            if !buffer.pointer.is_null() {
                let released = unsafe { status((self.free)(buffer.pointer), "cudaFree(close)") };
                if released.is_ok() {
                    buffer.pointer = ptr::null_mut();
                    buffer.bytes = 0;
                }
                if result.is_ok() {
                    result = released;
                }
            }
        }
        if !self.handle.is_null() {
            let released = unsafe { status((self.destroy)(self.handle), "cublasDestroy") };
            if released.is_ok() {
                self.handle = ptr::null_mut();
            }
            if result.is_ok() {
                result = released;
            }
        }
        self.stats.teardown_ns = timer.elapsed().as_nanos();
        result
    }
}
impl Drop for Cuda {
    fn drop(&mut self) {
        // Never reset the CUDA device: TensorRT and other services share it.
        // Explicit finish reports errors; Drop still releases every owned item
        // during unwinding or a partially completed library/handle admission.
        unsafe {
            if !self.handle.is_null() {
                (self.sync)();
            }
            for buffer in &mut self.buffers {
                if !buffer.pointer.is_null() {
                    (self.free)(buffer.pointer);
                }
            }
            if !self.handle.is_null() {
                (self.destroy)(self.handle);
            }
        }
    }
}
fn dimensions(a: usize, b: usize, m: usize, k: usize, n: usize) -> Result<(i32, i32, i32, usize)> {
    let size = |x: usize, y: usize| {
        x.checked_mul(y)
            .filter(|&v| v <= isize::MAX as usize / 4)
            .ok_or("CUDA matrix size overflow")
    };
    if size(m, k)? != a || size(n, k)? != b {
        return Err("CUDA matrix shape mismatch".into());
    }
    Ok((
        i32::try_from(m).map_err(|_| "CUDA matrix dimension exceeds ABI")?,
        i32::try_from(k).map_err(|_| "CUDA matrix dimension exceeds ABI")?,
        i32::try_from(n).map_err(|_| "CUDA matrix dimension exceeds ABI")?,
        size(m, n)?,
    ))
}
struct Session {
    cuda: Option<Cuda>,
    failure: Option<String>,
}
thread_local! { static ACTIVE: RefCell<Option<Session>> = const { RefCell::new(None) }; }
/// Select packing geometry from the executing scope, never from an environment
/// setting that might describe a different thread or an inactive GPU backend.
pub(crate) fn cuda_active() -> bool {
    ACTIVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|session| session.cuda.is_some())
    })
}
struct Scope;
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}
pub fn backend() -> Result<String> {
    match std::env::var("MYTH_WAVEFORM_BACKEND") {
        Err(std::env::VarError::NotPresent) => Ok("cpu".into()),
        Ok(value) if value == "cpu" || value == "cuda" => Ok(value),
        _ => Err("MYTH_WAVEFORM_BACKEND must be cpu or cuda".into()),
    }
}
pub fn backend_name() -> Result<&'static str> {
    Ok(if backend()? == "cuda" {
        "fp32-cuda-cublas"
    } else {
        "fp32-cpu"
    })
}
pub fn run_from_env<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, Stats)> {
    run(&backend()?, f)
}
pub fn run<T>(backend: &str, f: impl FnOnce() -> Result<T>) -> Result<(T, Stats)> {
    if ACTIVE.with(|slot| slot.borrow().is_some()) {
        return Err("Nested waveform backend scope".into());
    }
    match backend {
        "cpu" => Ok((f()?, Stats::default())),
        "cuda" => {
            // The preceding TensorRT/CPU model is already dropped. Reclaim its
            // free arenas before loading the next GPU library/model; reclaim
            // this stage only after its context, buffers and code owners drop.
            let before = trim_host_workspace();
            let mut workspace = HostWorkspaceLease(true);
            let result = session(
                Session {
                    cuda: Some(Cuda::open()?),
                    failure: None,
                },
                f,
            );
            let after = workspace.finish();
            result.map(|(value, mut stats)| {
                stats.host_workspace_trims = vec![before, after];
                (value, stats)
            })
        }
        _ => Err("Waveform backend must be cpu or cuda".into()),
    }
}
fn session<T>(session: Session, f: impl FnOnce() -> Result<T>) -> Result<(T, Stats)> {
    ACTIVE.with(|slot| {
        *slot.borrow_mut() = Some(session);
    });
    let _scope = Scope;
    let result = f();
    ACTIVE.with(|slot| {
        let mut session = slot.borrow_mut().take().expect("owned waveform scope");
        // A kernel error wins over downstream nonfinite diagnostics. Never
        // substitute CPU output or publish a success after a failed CUDA call.
        let close = session.cuda.as_mut().map(|cuda| cuda.finish()).transpose();
        if let Some(message) = session.failure {
            return Err(match close {
                Err(cleanup) => format!("{message}; cleanup: {cleanup}"),
                _ => message,
            });
        }
        close?;
        let cuda = session
            .cuda
            .as_ref()
            .ok_or("CUDA session missing context")?;
        Ok((result?, cuda.stats.clone()))
    })
}
/// Existing private matrix routines return Vec, so retain the first GPU error
/// at the surrounding Result boundary and poison this calculation. Later kernel
/// calls do no work; the scope must reject even if a downstream operation could
/// otherwise mask a NaN. No public model method or HTTP receipt can accept it.
pub(crate) fn product(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let session = slot.as_mut()?;
        let result = if let Some(message) = &session.failure {
            Err(message.clone())
        } else {
            session
                .cuda
                .as_mut()
                .ok_or("CUDA session missing context".into())
                .and_then(|cuda| cuda.product(a, b, m, k, n))
        };
        Some(match result {
            Ok(values) => values,
            Err(message) => {
                session.failure.get_or_insert(message);
                vec![f32::NAN; m.checked_mul(n).expect("validated matrix output size")]
            }
        })
    })
}
/// Small proof entry using the exact dispatch path shared by all three stages.
pub fn product_for_proof(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    dimensions(a.len(), b.len(), m, k, n)?;
    Ok(crate::conformer::product(a, b, m, k, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dimensions_reject_overflow_and_mismatch() {
        assert!(dimensions(0, 0, usize::MAX, 2, 1).is_err());
        assert!(dimensions(2, 3, 1, 2, 1).is_err());
        assert!(dimensions(0, 0, i32::MAX as usize + 1, 0, 0).is_err());
    }
    #[test]
    fn failure_never_becomes_cpu_fallback_or_success() {
        let result = session(
            Session {
                cuda: None,
                failure: Some("injected CUDA failure".into()),
            },
            || {
                let values = crate::conformer::product(&[2.], &[3.], 1, 1, 1);
                assert!(values[0].is_nan());
                assert!(run("cpu", || Ok(())).is_err());
                Ok(()) // Simulate downstream masking: the scope must still fail.
            },
        );
        assert_eq!(result.unwrap_err(), "injected CUDA failure");
        assert!(ACTIVE.with(|slot| slot.borrow().is_none()));
        assert_eq!(crate::conformer::product(&[2.], &[3.], 1, 1, 1), [6.]);
    }
    #[test]
    fn unwinding_releases_scope_and_other_threads_are_isolated() {
        let unwind = std::panic::catch_unwind(|| {
            let _: Result<((), Stats)> = session(
                Session {
                    cuda: None,
                    failure: None,
                },
                || {
                    assert!(std::thread::spawn(|| ACTIVE.with(|s| s.borrow().is_none()))
                        .join()
                        .unwrap());
                    panic!("injected stage panic");
                },
            );
        });
        assert!(unwind.is_err());
        assert!(ACTIVE.with(|slot| slot.borrow().is_none()));
    }
}
