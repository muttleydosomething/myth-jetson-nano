//! A small C ABI adapter is necessary because TensorRT exposes a C++ API.
//! Rust owns the handle exclusively, pins the engine checksum before load,
//! and retains reset service contexts plus process-registered plugin code.
//! No Python or Torch is used.
use crate::{generation::Decoder, npy::Npy, Result};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{
    ffi::{c_char, c_int, c_void, CString},
    fs::File,
    io::Read,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{Arc, Mutex, Weak},
};

type Open = unsafe extern "C" fn(*const c_char, *mut c_char, usize) -> *mut c_void;
type Step = unsafe extern "C" fn(
    *mut c_void,
    *const u16,
    usize,
    *mut f32,
    usize,
    *mut c_char,
    usize,
) -> c_int;
type Reset = unsafe extern "C" fn(*mut c_void);
type Close = unsafe extern "C" fn(*mut c_void);
type OpenSession = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> *mut c_void;

pub struct TensorRt {
    handle: NonNull<c_void>,
    step_fn: Step,
    reset_fn: Reset,
    close_fn: Close,
    _library: Arc<Library>,
    prompt: Npy,
    embeddings: Arc<Npy>,
    idle: Option<Weak<Mutex<Option<IdleContext>>>>,
    reusable: bool,
}

/// Linux source identity, including ctime so restoring size/mtime cannot hide
/// a rewrite. This is an invalidation key for already admitted objects, never
/// a substitute for SHA256 when admitting new bytes. Observe before and after
/// admission/load so a source modified during construction is not retained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SourceIdentity {
    path: PathBuf,
    stamp: [u64; 7],
}
impl SourceIdentity {
    pub(crate) fn read(path: &Path) -> Result<Self> {
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        let m = path.metadata().map_err(|e| e.to_string())?;
        if !m.is_file() {
            return Err("Admitted engine source must be a regular file".into());
        }
        Ok(Self {
            path,
            stamp: [
                m.dev(),
                m.ino(),
                m.len(),
                m.mtime() as u64,
                m.mtime_nsec() as u64,
                m.ctime() as u64,
                m.ctime_nsec() as u64,
            ],
        })
    }
    pub(crate) fn unchanged(&self) -> Result<bool> {
        Ok(Self::read(&self.path)? == *self)
    }
}

/// Exclusive idle context. A checked-out decoder holds only a weak return
/// address, so unload/engine replacement cannot resurrect an evicted owner.
struct IdleContext {
    handle: Option<NonNull<c_void>>,
    close_fn: Close,
    _library: Arc<Library>,
}
// SAFETY: an idle context has no pending CUDA work or request KV, and its only
// owner is moved under the cache mutex. The native entry points attach device.
unsafe impl Send for IdleContext {}
impl Drop for IdleContext {
    fn drop(&mut self) {
        if let Some(handle) = self.handle {
            unsafe { (self.close_fn)(handle.as_ptr()) };
        }
    }
}
pub(crate) fn diagnostic(buffer: &[c_char]) -> String {
    let bytes: Vec<u8> = buffer
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
fn file_sha256(path: &Path) -> Result<String> {
    let _trace = crate::trace::named("stream_read_sha256", path.to_str().unwrap_or("non-UTF8"));
    let mut source = File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let size = source.read(&mut buffer).map_err(|e| e.to_string())?;
        if size == 0 {
            break;
        }
        hash.update(&buffer[..size]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn check_engine(path: &Path, expected: &str) -> Result<()> {
    if expected.len() != 64 || !expected.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Engine SHA256 must be 64 hexadecimal characters".into());
    }
    let actual = file_sha256(path)?;
    if actual != expected.to_ascii_lowercase() {
        return Err(format!(
            "TensorRT engine checksum differs from admitted engine: {actual}"
        ));
    }
    Ok(())
}

// initLibNvInferPlugins registers code with TensorRT/CUDA beyond one session's
// lifetime. dlclose after each request leaves process-global callbacks pointing
// into unloaded code. Keep code resident independently of engine/session owners.
// One immutable bridge identity prevents a later load
// from silently reusing old code after a catalog/path update.
struct ProcessBridge {
    path: PathBuf,
    sha256: String,
    library: Arc<Library>,
    source: SourceIdentity,
}
static PROCESS_BRIDGE: Mutex<Option<ProcessBridge>> = Mutex::new(None);
pub(crate) fn process_bridge(path: &Path) -> Result<Arc<Library>> {
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    let source = SourceIdentity::read(&path)?;
    let mut slot = PROCESS_BRIDGE
        .lock()
        .map_err(|_| "Native process bridge lock poisoned")?;
    if let Some(bridge) = slot.as_mut() {
        if bridge.path != path || (bridge.source != source && bridge.sha256 != file_sha256(&path)?)
        {
            return Err("Native process bridge cannot be rebound; start a new process for changed bridge code".into());
        }
        if !source.unchanged()? {
            return Err("Native process bridge changed during admission".into());
        }
        bridge.source = source;
        return Ok(bridge.library.clone());
    }
    let sha256 = file_sha256(&path)?;
    // SAFETY: callers supply trusted native code; this process retains its
    // library owner until exit, including after failed session construction.
    let library = Arc::new(unsafe { Library::new(&path) }.map_err(|e| e.to_string())?);
    if !source.unchanged()? {
        return Err("Native process bridge changed during admission".into());
    }
    *slot = Some(ProcessBridge {
        path,
        sha256,
        library: library.clone(),
        source,
    });
    Ok(library)
}

/// One admitted engine, separate from request context/KV. Only the service cache
/// moves this owner between blocking workers; context construction is serialized
/// by its mutex and the existing inference permit covers the entire utterance.
struct Engine {
    path: PathBuf,
    sha256: String,
    handle: NonNull<c_void>,
    library: Arc<Library>,
    session_fn: OpenSession,
    close_fn: Close,
    step_fn: Step,
    reset_fn: Reset,
    session_close_fn: Close,
    source: SourceIdentity,
    idle: Arc<Mutex<Option<IdleContext>>>,
}
// SAFETY: the native owner uses CUDA's runtime primary context, which can move
// across host threads. It contains immutable weights, never request state.
// Access/eviction requires the service cache mutex; no Sync implementation.
unsafe impl Send for Engine {}
impl Drop for Engine {
    fn drop(&mut self) {
        // Release the retained context before its engine; checked-out sessions
        // independently retain native shared weights until their own teardown.
        if let Ok(mut idle) = self.idle.lock() {
            *idle = None;
        }
        // SAFETY: this unique owner closes once; native sessions retain shared
        // weights until their context is gone, and library remains resident.
        unsafe { (self.close_fn)(self.handle.as_ptr()) };
    }
}

/// Bounded to one token engine/idle context and one meanflow/HiFT graph each.
/// CLI callers keep their original uncached lifetime.
/// The service holds its inference permit until the request session has closed,
/// so replacement cannot overlap a preceding utterance's weights.
#[derive(Default)]
pub struct EngineCache {
    engine: Option<Engine>,
    bridge: Option<(PathBuf, SourceIdentity, String)>,
    pub(crate) prompt: crate::conditioning::PromptCache,
    pub(crate) conformer: crate::conformer::ConformerCache,
    pub(crate) meanflow: crate::meanflow::MeanflowCache,
    pub(crate) hift: crate::hift::HiftCache,
}
impl EngineCache {
    pub fn clear(&mut self) {
        self.engine = None;
        self.bridge = None;
        self.prompt.clear();
        self.conformer.clear();
        self.meanflow.clear();
        self.hift.clear();
    }
    /// Same integrity authority as engines: a warm identity hit avoids hashing
    /// native code again. Changed bytes still require a new process at load.
    pub(crate) fn admit_bridge(&mut self, path: &Path, pin: &str) -> Result<()> {
        let result =
            (|| {
                if !crate::conditioning::valid_sha(pin) {
                    return Err("Bridge SHA256 must be 64 hexadecimal characters".into());
                }
                let source = SourceIdentity::read(path)?;
                if self.bridge.as_ref().is_some_and(|(p, s, h)| {
                    p == path && s == &source && h.eq_ignore_ascii_case(pin)
                }) {
                    return Ok(());
                }
                self.bridge = None;
                check_engine(path, pin)?;
                if SourceIdentity::read(path)? != source {
                    return Err("Native bridge changed during admission".into());
                }
                self.bridge = Some((path.to_path_buf(), source, pin.to_ascii_lowercase()));
                Ok(())
            })();
        if result.is_err() {
            self.clear();
        }
        result
    }
    pub fn session(
        &mut self,
        bridge: &Path,
        engine: &Path,
        expected: &str,
        prompt: Npy,
        embeddings: impl Into<Arc<Npy>>,
    ) -> Result<(TensorRt, bool)> {
        let embeddings = embeddings.into();
        if expected.len() != 64 || !expected.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("Engine SHA256 must be 64 hexadecimal characters".into());
        }
        prompt.prompt_frames()?;
        embeddings.speech_row(0)?;
        let path = engine.canonicalize().map_err(|e| e.to_string())?;
        let sha256 = expected.to_ascii_lowercase();
        let source = SourceIdentity::read(engine)?;
        let library = process_bridge(bridge)?;
        let hit = self
            .engine
            .as_ref()
            .is_some_and(|e| e.path == path && e.sha256 == sha256 && e.source == source);
        if !hit {
            // Evict before allocating replacement weights, including failure.
            self.engine = None;
            check_engine(engine, expected)?;
            let engine_path =
                CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
            // SAFETY: resolve the full ABI before allocation, so a missing
            // symbol cannot leak an engine or fall back to per-request loading.
            unsafe {
                let open: Open = *library
                    .get(b"cb_engine_open\0")
                    .map_err(|e| e.to_string())?;
                let session_fn: OpenSession = *library
                    .get(b"cb_session_open\0")
                    .map_err(|e| e.to_string())?;
                let close_fn: Close = *library
                    .get(b"cb_engine_close\0")
                    .map_err(|e| e.to_string())?;
                let step_fn: Step = *library.get(b"cb_step\0").map_err(|e| e.to_string())?;
                let reset_fn: Reset = *library.get(b"cb_reset\0").map_err(|e| e.to_string())?;
                let session_close_fn: Close =
                    *library.get(b"cb_close\0").map_err(|e| e.to_string())?;
                let mut error = [0 as c_char; 1024];
                let _trace = crate::trace::span("token_engine_deserialize");
                let handle =
                    NonNull::new(open(engine_path.as_ptr(), error.as_mut_ptr(), error.len()))
                        .ok_or_else(|| format!("TensorRT engine open: {}", diagnostic(&error)))?;
                if source.unchanged() != Ok(true) {
                    close_fn(handle.as_ptr());
                    return Err("Token engine changed during admission".into());
                }
                self.engine = Some(Engine {
                    path,
                    sha256,
                    handle,
                    library,
                    session_fn,
                    close_fn,
                    step_fn,
                    reset_fn,
                    session_close_fn,
                    source,
                    idle: Arc::new(Mutex::new(None)),
                });
            }
        }
        let engine = self
            .engine
            .as_ref()
            .ok_or("TensorRT engine cache is empty")?;
        let mut error = [0 as c_char; 1024];
        let idle = engine
            .idle
            .lock()
            .map_err(|_| "Token context lock poisoned")?
            .take();
        let handle = if let Some(mut context) = idle {
            let _trace = crate::trace::span("token_context_reuse");
            context.handle.take().expect("idle context owns its handle")
        } else {
            let _trace = crate::trace::span("token_context_create");
            NonNull::new(unsafe {
                (engine.session_fn)(engine.handle.as_ptr(), error.as_mut_ptr(), error.len())
            })
            .ok_or_else(|| format!("TensorRT session open: {}", diagnostic(&error)))?
        };
        Ok((
            TensorRt {
                handle,
                step_fn: engine.step_fn,
                reset_fn: engine.reset_fn,
                close_fn: engine.session_close_fn,
                _library: engine.library.clone(),
                prompt,
                embeddings,
                idle: Some(Arc::downgrade(&engine.idle)),
                reusable: true,
            },
            hit,
        ))
    }
}
impl TensorRt {
    /// Bridge code and engines must be supplied from trusted local build paths:
    /// dynamic library loading executes native code.
    pub fn open(
        bridge: &Path,
        engine: &Path,
        expected: &str,
        prompt: Npy,
        embeddings: impl Into<Arc<Npy>>,
    ) -> Result<Self> {
        let embeddings = embeddings.into();
        check_engine(engine, expected)?;
        prompt.prompt_frames()?;
        embeddings.speech_row(0)?;
        let engine_path =
            CString::new(engine.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: all symbols use the versioned-in-source cb_* ABI. Both this
        // session and the process retain ownership of the registered code.
        unsafe {
            let library = process_bridge(bridge)?;
            let open: Open = *library.get(b"cb_open\0").map_err(|e| e.to_string())?;
            let step_fn: Step = *library.get(b"cb_step\0").map_err(|e| e.to_string())?;
            let reset_fn: Reset = *library.get(b"cb_reset\0").map_err(|e| e.to_string())?;
            let close_fn: Close = *library.get(b"cb_close\0").map_err(|e| e.to_string())?;
            let mut error = [0 as c_char; 1024];
            let handle = NonNull::new(open(engine_path.as_ptr(), error.as_mut_ptr(), error.len()))
                .ok_or_else(|| format!("TensorRT open: {}", diagnostic(&error)))?;
            Ok(Self {
                handle,
                step_fn,
                reset_fn,
                close_fn,
                _library: library,
                prompt,
                embeddings,
                idle: None,
                reusable: false,
            })
        }
    }
    fn step(&mut self, input: &[u16], frames: usize) -> Result<Vec<f32>> {
        if input.len() != frames * 1024 {
            return Err("Embedding length differs from frame count".into());
        }
        let mut logits = vec![0f32; 6563];
        let mut error = [0 as c_char; 1024];
        // SAFETY: vectors remain alive until synchronous cb_step returns; the
        // exclusive mutable handle prevents concurrent context use.
        // Failure/unwind retires the context; only a fully returned native
        // step may put it back in the idle slot at request teardown.
        let was_reusable = self.reusable;
        self.reusable = false;
        let count = unsafe {
            (self.step_fn)(
                self.handle.as_ptr(),
                input.as_ptr(),
                frames,
                logits.as_mut_ptr(),
                logits.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if count == 6563 {
            self.reusable = was_reusable;
        }
        if count != 6563 {
            return Err(format!("TensorRT step: {}", diagnostic(&error)));
        }
        Ok(logits)
    }
}
impl Decoder for TensorRt {
    fn reset(&mut self) -> Result<()> {
        // SAFETY: a live unique handle created by cb_open.
        unsafe { (self.reset_fn)(self.handle.as_ptr()) };
        Ok(())
    }
    fn prefill(&mut self) -> Result<Vec<f32>> {
        let frames = self.prompt.prompt_frames()?;
        // Only the short prompt is copied; keep the full embedding table resident.
        let input = self.prompt.halves.clone();
        self.step(&input, frames)
    }
    fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        let input = self.embeddings.speech_row(token)?.to_vec();
        self.step(&input, 1)
    }
}
impl Drop for TensorRt {
    fn drop(&mut self) {
        let context = IdleContext {
            handle: Some(self.handle),
            close_fn: self.close_fn,
            _library: self._library.clone(),
        };
        if self.reusable {
            if let Some(idle) = self.idle.as_ref().and_then(Weak::upgrade) {
                // Reset KV and absolute position before the context becomes
                // visible to another request, including after cancellation.
                unsafe { (self.reset_fn)(self.handle.as_ptr()) };
                if let Ok(mut slot) = idle.lock() {
                    if slot.is_none() {
                        *slot = Some(context);
                        return;
                    }
                }
            }
        }
        drop(context);
    }
}
