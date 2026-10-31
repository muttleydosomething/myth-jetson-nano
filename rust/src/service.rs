//! Native HTTP boundary for admitted, already prepared voices. This shares the
//! CLI inference routine; response transforms share audio_output and never load Python.
//! One worker owns inference. Busy callers are rejected rather than queued into
//! unbounded memory, and health stays independent of the blocking model kernels.
use crate::{
    audio_output::{self, Format},
    conditioning::{admitted_bytes, valid_sha},
    pipeline,
    sampling::Sampling,
    Result,
};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Semaphore;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Voice {
    pub display_name: String,
    pub pipeline: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    schema_version: u32,
    model: String,
    #[serde(default)]
    model_aliases: Vec<String>,
    voices: BTreeMap<String, Voice>,
    #[serde(default)]
    sampling: Sampling,
    seed: u64,
    max_gen_len: usize,
    #[serde(default)]
    workshop: Option<crate::workshop::Configuration>,
}
impl Catalog {
    /// Resolve and pin the configured catalog before opening a listener. Each
    /// request re-admits its pinned pipeline and every nested model asset.
    pub fn open(path: &Path, sha256: &str) -> Result<Self> {
        let mut catalog: Self = serde_json::from_slice(&admitted_bytes(path, sha256)?)
            .map_err(|e| format!("Service catalog: {e}"))?;
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        for voice in catalog.voices.values_mut() {
            if voice.pipeline.is_relative() {
                voice.pipeline = root.join(&voice.pipeline);
            }
            admitted_bytes(&voice.pipeline, &voice.sha256)?;
        }
        catalog.validate()?;
        Ok(catalog)
    }
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.model.trim().is_empty()
            || (self.voices.is_empty() && self.workshop.is_none())
            || self.max_gen_len == 0
            || self.max_gen_len > 1024
        {
            return Err("Invalid native service catalog".into());
        }
        for (name, voice) in &self.voices {
            // Identifiers are looked up, never joined to a request-supplied path.
            if name.is_empty()
                || name.contains(['/', '\\', '\0'])
                || name == "."
                || name == ".."
                || voice.display_name.trim().is_empty()
                || !valid_sha(&voice.sha256)
            {
                return Err("Invalid prepared voice entry".into());
            }
        }
        self.sampling.validate()
    }
}
fn wav_format() -> String {
    "wav".into()
}
fn normal_speed() -> f64 {
    1.0
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(default = "wav_format")]
    pub response_format: String,
    #[serde(default = "normal_speed")]
    pub speed: f64,
    pub seed: Option<u64>,
    #[serde(skip)]
    pub temperature: Option<f32>,
    #[serde(skip)]
    pub chunk_size: Option<usize>,
    #[serde(skip)]
    pub workshop_request: bool,
}
pub struct Waveform {
    pub pcm: crate::npy::F32Npy,
}

/// The hardware boundary permits direct ownership tests without loading models.
/// The deployed constructor always selects NativeSynthesizer.
pub trait Synthesizer: Send + Sync + 'static {
    /// Called under the inference permit on a blocking worker. Retained weights
    /// must retire before Service Multiplex considers this owner unloaded.
    fn unload(&self) -> Result<()> {
        Ok(())
    }
    fn synthesize(
        &self,
        voice: &Voice,
        request: &SpeechRequest,
        cancelled: &AtomicBool,
    ) -> Result<Waveform>;
    fn streaming_granularity(&self) -> &'static str {
        "sentence"
    }
    /// The same hardware boundary supports partial waveforms. Non-incremental
    /// implementations retain their existing one-fragment behavior.
    fn synthesize_stream(
        &self,
        voice: &Voice,
        request: &SpeechRequest,
        cancelled: &AtomicBool,
        emit: &mut dyn FnMut(Waveform, bool) -> Result<()>,
    ) -> Result<()> {
        emit(self.synthesize(voice, request, cancelled)?, true)
    }
}
struct NativeSynthesizer {
    sampling: Sampling,
    seed: u64,
    limit: usize,
    engine_cache: Mutex<crate::tensorrt::EngineCache>,
}
impl Synthesizer for NativeSynthesizer {
    fn unload(&self) -> Result<()> {
        self.engine_cache
            .lock()
            .map_err(|_| "Token engine cache lock poisoned")?
            .clear();
        Ok(())
    }
    fn streaming_granularity(&self) -> &'static str {
        "codec-prefix"
    }
    fn synthesize_stream(
        &self,
        voice: &Voice,
        request: &SpeechRequest,
        cancelled: &AtomicBool,
        emit: &mut dyn FnMut(Waveform, bool) -> Result<()>,
    ) -> Result<()> {
        let mut sampling = self.sampling.clone();
        if let Some(temperature) = request.temperature {
            sampling.temperature = temperature;
        }
        let receipt = pipeline::render_stream_with_engine_cache(
            &voice.pipeline,
            &voice.sha256,
            &request.input,
            request.seed.unwrap_or(self.seed),
            self.limit,
            &sampling,
            || cancelled.load(Ordering::Acquire),
            |pcm, finality| emit(Waveform { pcm }, finality),
            Some(&self.engine_cache),
        )?;
        eprintln!(
            "{}",
            serde_json::json!({"operation":"native_waveform_stream","receipt":receipt})
        );
        Ok(())
    }
    fn synthesize(
        &self,
        voice: &Voice,
        request: &SpeechRequest,
        cancelled: &AtomicBool,
    ) -> Result<Waveform> {
        let mut sampling = self.sampling.clone();
        if let Some(temperature) = request.temperature {
            sampling.temperature = temperature;
        }
        let rendered = pipeline::render_with_engine_cache(
            &voice.pipeline,
            &voice.sha256,
            &request.input,
            request.seed.unwrap_or(self.seed),
            self.limit,
            &sampling,
            || cancelled.load(Ordering::Acquire),
            Some(&self.engine_cache),
        )?;
        eprintln!(
            "{}",
            serde_json::json!({
                "operation":"native_waveform",
                "stages_ms":rendered.output.stages_ms,
                "waveform_compute":rendered.output.waveform_compute,
                "samples":rendered.output.samples
                ,"token_engine_cache_hit":rendered.output.token_engine_cache_hit
            })
        );
        Ok(Waveform {
            pcm: rendered.waveform,
        })
    }
}
#[derive(Clone)]
struct App {
    catalog: Arc<Catalog>,
    synthesizer: Arc<dyn Synthesizer>,
    admission: Arc<Semaphore>,
    stopping: Arc<AtomicBool>,
    loaded: Arc<AtomicBool>,
    workshop: Option<Arc<crate::workshop::Workshop>>,
}
/// Dropping a request future signals the worker but never releases its permit
/// prematurely. A submitted native kernel must finish before another can enter.
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
fn error(status: StatusCode, message: &str) -> Response {
    (
        status,
        [(header::CACHE_CONTROL, "no-store")],
        Json(serde_json::json!({"detail": message})),
    )
        .into_response()
}
async fn health(State(app): State<App>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": if app.stopping.load(Ordering::Acquire) { "stopping" } else { "ready" },
        "engine": "myth-native", "inference_runtime": "rust-tensorrt",
        "model": app.catalog.model, "busy": app.admission.available_permits() == 0,
        "loaded": app.loaded.load(Ordering::Acquire),
        "prepared_voices": app.catalog.voices.len() + app.workshop.as_ref().map(|w|w.list(false).len()+w.list(true).len()).unwrap_or(0), "new_voice_preparation": app.workshop.is_some(),
        "incremental_audio": true, "streaming_granularity": app.synthesizer.streaming_granularity(), "streaming_formats": ["wav"], "waveform_backend": crate::gpu::backend_name().unwrap_or("invalid"),
        "response_formats": ["wav", "mp3", "opus"], "response_wav_encoding": "pcm16",
        "speed_processing": "native-phase-vocoder", "speed_preserves_pitch": true,
    }))
}
async fn voices(State(app): State<App>) -> Response {
    let mut entries: Vec<serde_json::Value> = app
        .catalog
        .voices
        .iter()
        .map(|(filename, voice)| serde_json::json!({"filename":filename,"display_name":voice.display_name}))
        .collect();
    if let Some(w) = &app.workshop {
        entries.extend(w.list(false).into_iter().map(|(filename,display_name)|serde_json::json!({"filename":filename,"display_name":display_name})));
    }
    ([(header::CACHE_CONTROL, "no-store")], Json(entries)).into_response()
}
async fn speech(State(app): State<App>, Json(mut request): Json<SpeechRequest>) -> Response {
    speech_response(app, &mut request, false).await
}
async fn speech_stream(State(app): State<App>, Json(mut request): Json<SpeechRequest>) -> Response {
    speech_response(app, &mut request, true).await
}
// Both transports use the same authority checks and defaults. Streaming is an
// explicit route: existing WAV/MP3/Opus clients keep their complete-file contract.
async fn speech_response(app: App, request: &mut SpeechRequest, streaming: bool) -> Response {
    if app.stopping.load(Ordering::Acquire) {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Speech service is stopping",
        );
    }
    if !app.loaded.load(Ordering::Acquire) {
        return error(StatusCode::SERVICE_UNAVAILABLE, "Native model is unloaded");
    }
    if let Some(w) = &app.workshop {
        let config = w.config();
        if request.temperature.is_none() {
            request.temperature = config["generation_defaults"]["temperature"]
                .as_f64()
                .map(|v| v as f32);
        }
        if request.seed.is_none() {
            request.seed = config["generation_defaults"]["seed"].as_u64();
        }
    }
    if request.model != app.catalog.model && !app.catalog.model_aliases.contains(&request.model) {
        return error(StatusCode::BAD_REQUEST, "Unknown speech model");
    }
    if request.input.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "Speech input must contain text");
    }
    let format = match Format::parse(&request.response_format) {
        Ok(format) => format,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    if let Err(message) = audio_output::validate_speed(request.speed) {
        return error(StatusCode::BAD_REQUEST, &message);
    }
    if streaming && format != Format::Wav {
        return error(
            StatusCode::BAD_REQUEST,
            "Incremental speech requires WAV output",
        );
    }
    let dynamic = match &app.workshop {
        Some(w) => match w.resolve(&request.voice, true) {
            Ok(voice) => voice,
            Err(message) => {
                eprintln!("Voice admission: {message}");
                return error(StatusCode::CONFLICT, "Prepared voice admission failed");
            }
        },
        None => None,
    };
    let voice = match dynamic.or_else(|| app.catalog.voices.get(&request.voice).cloned()) {
        Some(voice) => voice,
        None => {
            return error(
                StatusCode::NOT_FOUND,
                "Voice has no admitted native preparation",
            )
        }
    };
    let permit = match app.admission.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return error(StatusCode::TOO_MANY_REQUESTS, "Speech engine is busy"),
    };
    // Unload may finish on another runtime thread during voice validation.
    // Recheck under inference ownership so that stale admission cannot silently
    // load retained weights after Service Multiplex released this owner.
    if app.stopping.load(Ordering::Acquire) || !app.loaded.load(Ordering::Acquire) {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Native model is unloaded or stopping",
        );
    }
    let request = request.clone();
    if streaming {
        return incremental_response(app, voice, request, permit);
    }
    let cancel = CancelOnDrop(Arc::new(AtomicBool::new(false)));
    let worker_cancel = cancel.0.clone();
    let started = now_ms();
    let timer = Instant::now();
    // Permit ownership moves into the worker, including during caller cancellation.
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut waveform = Waveform {
            pcm: crate::npy::F32Npy {
                shape: vec![1, 0],
                values: Vec::new(),
            },
        };
        // Workshop chunks share the permit, cancellation and output codec.
        // Ordinary OpenAI requests retain the exact preceding execution path.
        for input in crate::workshop::chunks(&request.input, request.chunk_size) {
            if worker_cancel.load(Ordering::Acquire) {
                return Err("Speech request cancelled".into());
            }
            let mut part = request.clone();
            part.input = input;
            let rendered = app.synthesizer.synthesize(&voice, &part, &worker_cancel)?;
            if request.workshop_request {
                let pcm = audio_output::tempo(&rendered.pcm, request.speed, || {
                    worker_cancel.load(Ordering::Acquire)
                })?;
                crate::workshop::join(&mut waveform.pcm.values, &pcm.values);
            } else {
                waveform.pcm.values.extend(rendered.pcm.values);
            }
        }
        waveform.pcm.shape[1] = waveform.pcm.values.len();
        if request.workshop_request {
            let peak = waveform.pcm.values.iter().fold(0f32, |p, v| p.max(v.abs()));
            if peak > 0.99 {
                for sample in &mut waveform.pcm.values {
                    *sample *= 0.95 / peak;
                }
            }
        }
        // Encoding, including native codec finalization, belongs to this worker.
        // Caller cancellation must not admit a second request during that work.
        // Tempo changes duration at the original sample rate before every codec.
        // Both transforms retain the same permit and caller cancellation signal.
        let speed = if request.workshop_request {
            1.0
        } else {
            request.speed
        };
        let pcm = audio_output::tempo(&waveform.pcm, speed, || {
            worker_cancel.load(Ordering::Acquire)
        })?;
        let bytes = audio_output::encode(&pcm, format, || worker_cancel.load(Ordering::Acquire))?;
        Ok::<_, String>((bytes, pcm.values.len()))
    })
    .await;
    let (bytes, samples) = match result {
        Ok(Ok(waveform)) => waveform,
        other => {
            eprintln!(
                "myth native request failed: {}",
                match other {
                    Ok(Err(message)) => message,
                    Err(error) => error.to_string(),
                    _ => unreachable!(),
                }
            );
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Native speech generation failed",
            );
        }
    };
    let finished = now_ms();
    let elapsed = timer.elapsed().as_millis();
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let mut response = Response::new(Body::from(bytes));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(format.content_type()),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    for (name, value) in [
        ("x-myth-started-unix-ms", started.to_string()),
        ("x-myth-finished-unix-ms", finished.to_string()),
        ("x-myth-duration-ms", elapsed.to_string()),
        ("x-myth-samples", samples.to_string()),
        ("x-myth-sha256", sha256.clone()),
    ] {
        response.headers_mut().insert(
            name,
            HeaderValue::from_str(&value).expect("numeric/checksum header"),
        );
    }
    eprintln!(
        "{}",
        serde_json::json!({"status":"waveform_generated", "started_unix_ms":started,
        "finished_unix_ms":finished, "duration_ms":elapsed, "samples":samples,
        "waveform_sha256":sha256, "reference_preparations":0})
    );
    response
}

/// The HTTP body owns cancellation for its entire lifetime, after the handler
/// returns. Dropping a stalled reader wakes blocking_send; the worker retains
/// its inference permit until native kernels and response processing have ended.
struct AudioStream {
    receiver: tokio::sync::mpsc::Receiver<axum::body::Bytes>,
    _cancel: CancelOnDrop,
}
impl futures_core::Stream for AudioStream {
    type Item = std::result::Result<axum::body::Bytes, std::convert::Infallible>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx).map(|item| item.map(Ok))
    }
}
fn incremental_response(
    app: App,
    voice: Voice,
    mut request: SpeechRequest,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let cancel = CancelOnDrop(Arc::new(AtomicBool::new(false)));
    let worker_cancel = cancel.0.clone();
    // One pending frame, plus the worker's current segment. A slow reader cannot
    // turn an arbitrarily long narration into an unbounded service-side queue.
    let (sender, receiver) = tokio::sync::mpsc::channel::<axum::body::Bytes>(1);
    let started = now_ms();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let timer = Instant::now();
        let stopped = || {
            worker_cancel.load(Ordering::Acquire)
                || app.stopping.load(Ordering::Acquire)
                || sender.is_closed()
        };
        let send = |value: serde_json::Value| -> Result<()> {
            if stopped() {
                return Err("Speech request cancelled".into());
            }
            let mut line = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
            line.push(b'\n');
            sender
                .blocking_send(line.into())
                .map_err(|_| "Speech reader disconnected".into())
        };
        let result = (|| -> Result<serde_json::Value> {
            send(
                serde_json::json!({"type":"start","version":1,"format":"wav","sample_rate":24000,
                "channels":1,"granularity":if request.speed == 1. { app.synthesizer.streaming_granularity() } else { "sentence" },"started_unix_ms":started}),
            )?;
            // Reuse the workshop splitter and its explicit split/size controls.
            // The OpenAI streaming route selects the same sentence boundary.
            if !request.workshop_request {
                request.chunk_size = Some(120);
            }
            let inputs = crate::workshop::chunks(&request.input, request.chunk_size);
            let mut pending = Vec::new();
            let mut samples = 0usize;
            let mut frames = 0usize;
            let mut first_audio_ms = None;
            let mut digest = Sha256::new();
            for (index, input) in inputs.iter().enumerate() {
                if stopped() {
                    return Err("Speech request cancelled".into());
                }
                let mut part = request.clone();
                part.input = input.clone();
                let mut first_fragment = true;
                let mut segment_finished = false;
                let mut emit = |rendered: Waveform, finality: bool| -> Result<()> {
                    if segment_finished {
                        return Err("Waveform fragment after segment completion".into());
                    }
                    if stopped() {
                        return Err("Speech request cancelled".into());
                    }
                    let pcm = audio_output::tempo(&rendered.pcm, request.speed, stopped)?;
                    if request.workshop_request && first_fragment {
                        crate::workshop::join(&mut pending, &pcm.values);
                    } else {
                        pending.extend_from_slice(&pcm.values);
                    }
                    // Retain only the tail that the next smart join may fade. This
                    // gives exactly the established joins without withholding the
                    // complete utterance. OpenAI speech never adds workshop pauses.
                    first_fragment = false;
                    segment_finished = finality;
                    let retain =
                        if request.workshop_request && (!finality || index + 1 < inputs.len()) {
                            480.min(pending.len())
                        } else {
                            0
                        };
                    let boundary = pending.len() - retain;
                    if boundary == 0 {
                        return Ok(());
                    }
                    let tail = pending.split_off(boundary);
                    let pcm = crate::npy::F32Npy {
                        shape: vec![1, pending.len()],
                        values: std::mem::replace(&mut pending, tail),
                    };
                    // A whole-utterance gain pass would defeat incremental delivery.
                    // PCM16's existing finite/clipping policy applies per frame.
                    let wav = audio_output::encode(&pcm, Format::Wav, stopped)?;
                    digest.update(&wav[44..]);
                    let ready = timer.elapsed().as_millis();
                    first_audio_ms.get_or_insert(ready);
                    send(
                        serde_json::json!({"type":"audio","index":frames,"offset_samples":samples,
                    "samples":pcm.values.len(),"ready_ms":ready,"wav":base64::encode(&wav)}),
                    )?;
                    frames += 1;
                    samples += pcm.values.len();
                    Ok(())
                };
                if request.speed == 1. {
                    app.synthesizer
                        .synthesize_stream(&voice, &part, &worker_cancel, &mut emit)?;
                } else {
                    // Phase-vocoder tempo owns a complete sentence. Stretching
                    // independent tiny fragments would reset its spectral state.
                    emit(
                        app.synthesizer.synthesize(&voice, &part, &worker_cancel)?,
                        true,
                    )?;
                }
                if !segment_finished {
                    return Err("Waveform stream ended without final segment".into());
                }
            }
            if frames == 0 {
                return Err("No audio generated".into());
            }
            Ok(
                serde_json::json!({"type":"complete","frames":frames,"samples":samples,"pcm_sha256":format!("{:x}",digest.finalize()),
                "started_unix_ms":started,"finished_unix_ms":now_ms(),"duration_ms":timer.elapsed().as_millis(),"first_audio_ms":first_audio_ms,
                "reference_preparations":0}),
            )
        })();
        match result {
            Ok(receipt) => {
                if send(receipt.clone()).is_ok() {
                    eprintln!(
                        "{}",
                        serde_json::json!({"status":"stream_generated","receipt":receipt})
                    );
                }
            }
            Err(message) => {
                eprintln!("Native speech stream failed: {message}");
                // HTTP 200 means transport admission, never successful synthesis.
                // Clients require complete; disconnect, panic and errors cannot
                // silently promote partial speech to a successful output.
                let _ = send(
                    serde_json::json!({"type":"error","detail":"Native speech generation failed",
                    "started_unix_ms":started,"finished_unix_ms":now_ms(),"duration_ms":timer.elapsed().as_millis()}),
                );
            }
        }
    });
    let mut response = Response::new(Body::from_stream(AudioStream {
        receiver,
        _cancel: cancel,
    }));
    for (name, value) in [
        (header::CONTENT_TYPE, "application/x-ndjson"),
        (header::CACHE_CONTROL, "no-store"),
        (
            axum::http::HeaderName::from_static("x-accel-buffering"),
            "no",
        ),
    ] {
        response
            .headers_mut()
            .insert(name, HeaderValue::from_static(value));
    }
    response
}
// The existing workshop uses the same inference worker as OpenAI speech.
// A service reload validates admission in-process; it never restarts Hailstorm.
fn model_value(app: &App) -> serde_json::Value {
    serde_json::json!({"loaded":app.loaded.load(Ordering::Acquire),"type":"turbo",
    "class_name":"ChatterboxTurboTTS","device":"cuda","sample_rate":24000,
    "supports_paralinguistic_tags":true,"available_paralinguistic_tags":["[laugh]","[chuckle]","[cough]","[sigh]","[gasp]","[sniff]","[groan]","[shush]","[clear throat]"],
    "turbo_available_in_package":true,"multilingual_available_in_package":false,
    "supports_multilingual":false,"supported_languages":{"en":"English"},
    "engine":"myth-native","waveform_backend":crate::gpu::backend_name().unwrap_or("invalid"),"resident_model":false,"incremental_audio":true,"streaming_formats":["wav"],
    "loading_policy":"request-owned; unload disables admission and retains prepared voices"})
}
async fn model_info(State(app): State<App>) -> Json<serde_json::Value> {
    Json(model_value(&app))
}
fn status_message(message: &str) -> Response {
    Json(serde_json::json!({"message":message,"restart_needed":false})).into_response()
}
async fn initial_data(State(app): State<App>) -> Response {
    let Some(w) = &app.workshop else {
        return error(StatusCode::NOT_FOUND, "Workshop is not configured");
    };
    let response = voices(State(app.clone())).await;
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let voices: serde_json::Value = serde_json::from_slice(&body).unwrap();
    Json(serde_json::json!({"config":w.config(),"reference_files":w.list(true).into_iter().map(|v|v.0).collect::<Vec<_>>(),
     "predefined_voices":voices,"presets":[],"initial_gen_result":{"outputUrl":null,"filename":null,"genTime":null,"submittedVoiceMode":null,"submittedPredefinedVoice":null,"submittedCloneFile":null},"model_info":model_value(&app)})).into_response()
}
async fn reference_files(State(app): State<App>) -> Response {
    Json(
        app.workshop
            .as_ref()
            .map(|w| w.list(true).into_iter().map(|v| v.0).collect::<Vec<_>>())
            .unwrap_or_default(),
    )
    .into_response()
}
async fn settings(State(app): State<App>, Json(update): Json<serde_json::Value>) -> Response {
    let Some(w) = app.workshop.clone() else {
        return error(StatusCode::NOT_FOUND, "Workshop is not configured");
    };
    // Settings never allocate models. Their mutex serializes the durable state
    // envelope while active speech retains its already captured controls.
    match tokio::task::spawn_blocking(move || w.apply(update)).await {
        Ok(Ok(())) => status_message("Settings applied"),
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
        Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "Settings task failed"),
    }
}
async fn reset_settings(State(app): State<App>) -> Response {
    let Some(w) = app.workshop.clone() else {
        return error(StatusCode::NOT_FOUND, "Workshop is not configured");
    };
    match tokio::task::spawn_blocking(move || w.reset(app.catalog.seed)).await {
        Ok(Ok(())) => status_message("Default settings applied"),
        _ => error(StatusCode::INTERNAL_SERVER_ERROR, "Settings reset failed"),
    }
}
async fn workshop_speech(
    State(app): State<App>,
    Json(request): Json<crate::workshop::Request>,
) -> Response {
    let Some(w) = &app.workshop else {
        return error(StatusCode::NOT_FOUND, "Workshop is not configured");
    };
    let request = match request.speech(app.catalog.model.clone(), &w.config()) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, &e),
    };
    let extension = request.response_format.clone();
    let mut response = speech(State(app), Json(request)).await;
    if response.status() == StatusCode::OK {
        response.headers_mut().insert(
            header::CONTENT_DISPOSITION,
            HeaderValue::from_str(&format!("attachment; filename=\"myth-output.{extension}\""))
                .unwrap(),
        );
    }
    response
}
async fn workshop_stream(
    State(app): State<App>,
    Json(request): Json<crate::workshop::Request>,
) -> Response {
    let Some(w) = &app.workshop else {
        return error(StatusCode::NOT_FOUND, "Workshop is not configured");
    };
    let mut request = match request.speech(app.catalog.model.clone(), &w.config()) {
        Ok(request) => request,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    speech_response(app, &mut request, true).await
}
async fn unload_model(State(app): State<App>) -> Response {
    let Ok(permit) = app.admission.clone().try_acquire_owned() else {
        return error(StatusCode::TOO_MANY_REQUESTS, "Speech engine is busy");
    };
    // TensorRT destruction can synchronize CUDA. Keep it off the async runtime,
    // retaining inference ownership until the actual engine has retired.
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        app.synthesizer.unload()?;
        app.loaded.store(false, Ordering::Release);
        Ok::<_, String>(())
    })
    .await
    {
        Ok(Ok(())) => {
            status_message("Native model unloaded; engine released, prepared voices retained")
        }
        other => {
            eprintln!("Native unload: {other:?}");
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Native model unload failed",
            )
        }
    }
}
async fn load_model(State(app): State<App>) -> Response {
    let Ok(permit) = app.admission.clone().try_acquire_owned() else {
        return error(StatusCode::TOO_MANY_REQUESTS, "Speech engine is busy");
    };
    if app.stopping.load(Ordering::Acquire) {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Speech service is stopping",
        );
    }
    let loaded = app.loaded.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        for voice in app.catalog.voices.values() {
            pipeline::admit(&voice.pipeline, &voice.sha256)?;
        }
        if let Some(w) = app.workshop {
            for reference in [false, true] {
                for (name, _) in w.list(reference) {
                    if let Some(v) = w.resolve(&name, true)? {
                        pipeline::admit(&v.pipeline, &v.sha256)?;
                    }
                }
            }
        }
        // Successful re-admission invalidates any former engine identity. The
        // next utterance opens lazily; failed admission retains the former state.
        app.synthesizer.unload()?;
        loaded.store(true, Ordering::Release);
        Ok::<_, String>(())
    })
    .await;
    match result {
        Ok(Ok(())) => {
            status_message("Native model assets admitted; ready for request-owned loading")
        }
        other => {
            eprintln!("Native reload: {other:?}");
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Native model admission failed; previous state retained",
            )
        }
    }
}
async fn upload_reference(State(app): State<App>, multipart: Multipart) -> Response {
    upload(app, multipart, true).await
}
async fn upload_voice(State(app): State<App>, multipart: Multipart) -> Response {
    upload(app, multipart, false).await
}
async fn upload(app: App, mut multipart: Multipart, reference: bool) -> Response {
    use std::io::Write;
    let Some(w) = app.workshop.clone() else {
        return error(StatusCode::NOT_FOUND, "Workshop is not configured");
    };
    if app.stopping.load(Ordering::Acquire) {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Speech service is stopping",
        );
    }
    let Ok(permit) = app.admission.clone().try_acquire_owned() else {
        return error(StatusCode::TOO_MANY_REQUESTS, "Speech engine is busy");
    };
    let cancel = CancelOnDrop(Arc::new(AtomicBool::new(false)));
    let mut staged = Vec::new();
    // Multipart is streamed to the governed state volume with no body-size cap.
    // No decoder/model runs until the complete transport has been admitted.
    loop {
        let mut field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(_) => return error(StatusCode::BAD_REQUEST, "Malformed voice multipart"),
        };
        if field.name() != Some("files") {
            return error(StatusCode::BAD_REQUEST, "Expected multipart files");
        }
        let name = field.file_name().unwrap_or("").to_owned();
        if let Err(e) = crate::workshop::filename(&name) {
            return error(StatusCode::BAD_REQUEST, &e);
        }
        if app.catalog.voices.contains_key(&name) {
            return error(
                StatusCode::CONFLICT,
                "Voice name belongs to a retained catalog entry",
            );
        }
        let stage = match w.stage() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Upload staging: {e}");
                return error(StatusCode::INTERNAL_SERVER_ERROR, "Voice staging failed");
            }
        };
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(stage.join("recording"))
        {
            Ok(f) => f,
            Err(_) => return error(StatusCode::INTERNAL_SERVER_ERROR, "Voice staging failed"),
        };
        loop {
            match field.chunk().await {
                Ok(Some(bytes)) => {
                    if file.write_all(&bytes).is_err() {
                        return error(StatusCode::INTERNAL_SERVER_ERROR, "Voice upload failed");
                    }
                }
                Ok(None) => break,
                Err(_) => return error(StatusCode::BAD_REQUEST, "Incomplete voice multipart"),
            }
        }
        if file.sync_all().is_err() {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Voice upload flush failed",
            );
        }
        staged.push((name, stage));
    }
    if staged.is_empty() {
        return error(StatusCode::BAD_REQUEST, "No voice recordings supplied");
    }
    let worker_cancel = cancel.0.clone();
    let worker = w.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut uploaded = Vec::new();
        let mut errors = Vec::new();
        let mut preparations = 0;
        for (name, stage) in staged {
            match worker.upload(&name, &stage, reference, &worker_cancel) {
                Ok(n) => {
                    uploaded.push(name);
                    preparations += n;
                }
                Err(e) => {
                    eprintln!("Native upload: {e}");
                    errors.push(serde_json::json!({"filename":name,"error":e}));
                }
            }
        }
        (uploaded, errors, preparations)
    })
    .await;
    let (uploaded, errors, preparations) = match result {
        Ok(r) => r,
        Err(_) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Voice preparation task failed",
            )
        }
    };
    eprintln!(
        "{}",
        serde_json::json!({"status":"voice_upload","reference_preparations":preparations,"uploaded_files":uploaded})
    );
    let status = if uploaded.is_empty() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::OK
    };
    let mut body = serde_json::json!({"message":"Voice upload processed","uploaded_files":uploaded,"errors":errors,"reference_preparations":preparations});
    if reference {
        body["all_reference_files"] =
            serde_json::json!(w.list(true).into_iter().map(|v| v.0).collect::<Vec<_>>());
    } else {
        let response = voices(State(app)).await;
        body["all_predefined_voices"] = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
    }
    (status, Json(body)).into_response()
}
#[derive(Deserialize)]
struct DeleteQuery {
    filename: String,
}
async fn delete_voice(
    State(app): State<App>,
    headers: axum::http::HeaderMap,
    Query(query): Query<DeleteQuery>,
) -> Response {
    let Some(w) = app.workshop else {
        return error(StatusCode::NOT_FOUND, "Workshop is not configured");
    };
    // The existing gateway checks the operator's PIN and assignments upstream.
    // No authority credential is returned by initial-data or persisted settings.
    if !w.authorize_delete(&headers) {
        return error(
            StatusCode::FORBIDDEN,
            "Voice deletion requires gateway authority",
        );
    }
    let Ok(permit) = app.admission.clone().try_acquire_owned() else {
        return error(StatusCode::TOO_MANY_REQUESTS, "Speech engine is busy");
    };
    if let Err(e) = crate::workshop::filename(&query.filename) {
        return error(StatusCode::BAD_REQUEST, &e);
    }
    let filename = query.filename.clone();
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        w.delete(&filename)
    })
    .await
    {
        Ok(Ok(())) => Json(serde_json::json!({"ok":true,"deleted":query.filename})).into_response(),
        Ok(Err(e)) => {
            eprintln!("Voice deletion: {e}");
            error(StatusCode::BAD_REQUEST, &e)
        }
        Err(_) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Voice deletion task failed",
        ),
    }
}
async fn static_asset(State(app): State<App>, uri: axum::http::Uri) -> Response {
    let Some(w) = app.workshop else {
        return error(StatusCode::NOT_FOUND, "Workshop is not configured");
    };
    if uri.path() == "/docs" {
        return axum::response::Html("<html><title>myth native API</title><h1>myth native API</h1><p>GET /health, /api/model-info, /api/ui/initial-data, /get_predefined_voices, /get_reference_files. POST /tts, /v1/audio/speech, /upload_reference, /upload_predefined_voice, /save_settings, /reset_settings, /restart_server, /load_model, /unload_model. DELETE /delete_predefined_voice requires gateway authority.</p></html>").into_response();
    }
    let relative = if uri.path() == "/" {
        "index.html"
    } else {
        uri.path()
            .trim_start_matches('/')
            .strip_prefix("ui/")
            .unwrap_or(uri.path().trim_start_matches('/'))
    };
    if relative.split('/').any(|s| {
        s.is_empty() || s == "." || s == ".." || s.starts_with('.') || s.contains(['%', '\\'])
    }) {
        return error(StatusCode::NOT_FOUND, "Unknown workshop asset");
    }
    let path = match std::fs::canonicalize(w.configuration.ui_root.join(relative)) {
        Ok(p) if p.starts_with(&w.configuration.ui_root) => p,
        _ => return error(StatusCode::NOT_FOUND, "Unknown workshop asset"),
    };
    let media = match path.extension().and_then(|s| s.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "wasm" => "application/wasm",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "txt" => "text/plain; charset=utf-8",
        _ => return error(StatusCode::NOT_FOUND, "Unknown workshop asset type"),
    };
    match std::fs::read(path) {
        Ok(bytes) => (
            [
                (header::CONTENT_TYPE, media),
                (header::CACHE_CONTROL, "no-store"),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => error(StatusCode::NOT_FOUND, "Unknown workshop asset"),
    }
}
fn app_router(app: App) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/model-info", get(model_info))
        .route("/get_predefined_voices", get(voices))
        .route("/v1/audio/speech", post(speech))
        .route("/v1/audio/speech/stream", post(speech_stream))
        .route("/tts", post(workshop_speech))
        .route("/tts/stream", post(workshop_stream))
        .route("/api/ui/initial-data", get(initial_data))
        .route("/get_reference_files", get(reference_files))
        .route("/save_settings", post(settings))
        .route("/reset_settings", post(reset_settings))
        .route("/restart_server", post(load_model))
        .route("/load_model", post(load_model))
        .route("/unload_model", post(unload_model))
        .route("/upload_reference", post(upload_reference))
        .route("/upload_predefined_voice", post(upload_voice))
        .route("/delete_predefined_voice", delete(delete_voice))
        .fallback(get(static_asset))
        .layer(DefaultBodyLimit::disable())
        .with_state(app)
}
pub fn router(catalog: Catalog, synthesizer: Arc<dyn Synthesizer>) -> Result<Router> {
    catalog.validate()?;
    let workshop = catalog
        .workshop
        .clone()
        .map(|c| crate::workshop::Workshop::open(c, catalog.seed))
        .transpose()?
        .map(Arc::new);
    Ok(app_router(App {
        catalog: Arc::new(catalog),
        synthesizer,
        admission: Arc::new(Semaphore::new(1)),
        stopping: Arc::new(AtomicBool::new(false)),
        loaded: Arc::new(AtomicBool::new(true)),
        workshop,
    }))
}
pub async fn serve(path: &Path, sha256: &str, listen: SocketAddr) -> Result<()> {
    crate::gpu::backend()?; // Reject invalid selection before opening the listener.
    let catalog = Catalog::open(path, sha256)?;
    let synthesizer = Arc::new(NativeSynthesizer {
        sampling: catalog.sampling.clone(),
        seed: catalog.seed,
        limit: catalog.max_gen_len,
        engine_cache: Mutex::new(crate::tensorrt::EngineCache::default()),
    });
    let stopping = Arc::new(AtomicBool::new(false));
    let workshop = catalog
        .workshop
        .clone()
        .map(|c| crate::workshop::Workshop::open(c, catalog.seed))
        .transpose()?
        .map(Arc::new);
    let app = app_router(App {
        catalog: Arc::new(catalog),
        synthesizer,
        admission: Arc::new(Semaphore::new(1)),
        stopping: stopping.clone(),
        loaded: Arc::new(AtomicBool::new(true)),
        workshop,
    });
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .map_err(|e| format!("Native speech listener: {e}"))?;
    eprintln!(
        "{}",
        serde_json::json!({"status":"listening", "engine":"myth-native", "listen":listener.local_addr().map_err(|e| e.to_string())?})
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            #[cfg(unix)]
            {
                let mut terminate =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("SIGTERM handler");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            }
            #[cfg(not(unix))]
            let _ = tokio::signal::ctrl_c().await;
            // Finish existing requests before dropping the process and its GPU state.
            stopping.store(true, Ordering::Release);
        })
        .await
        .map_err(|e| format!("Native speech server: {e}"))
}
