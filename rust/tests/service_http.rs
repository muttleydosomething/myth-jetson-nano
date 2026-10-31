//! Exercise HTTP ownership without loading neural weights; Borg acceptance uses
//! the same router with NativeSynthesizer and the real checksummed pipeline.
use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use chatterbox_rs::service::{self, Catalog, SpeechRequest, Synthesizer, Voice, Waveform};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use tower::ServiceExt;

struct Probe {
    calls: AtomicUsize,
    unloads: AtomicUsize,
    fail_unload: AtomicBool,
    block: AtomicBool,
    cancelled: AtomicBool,
}
impl Probe {
    fn new(block: bool) -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            unloads: AtomicUsize::new(0),
            fail_unload: AtomicBool::new(false),
            block: AtomicBool::new(block),
            cancelled: AtomicBool::new(false),
        })
    }
}
impl Synthesizer for Probe {
    fn unload(&self) -> chatterbox_rs::Result<()> {
        self.unloads.fetch_add(1, Ordering::SeqCst);
        while self.block.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        if self.fail_unload.load(Ordering::SeqCst) {
            return Err("injected engine teardown failure".into());
        }
        Ok(())
    }
    fn synthesize(
        &self,
        _: &Voice,
        _: &SpeechRequest,
        cancel: &AtomicBool,
    ) -> chatterbox_rs::Result<Waveform> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        while self.block.load(Ordering::SeqCst) {
            if cancel.load(Ordering::SeqCst) {
                self.cancelled.store(true, Ordering::SeqCst);
                return Err("Speech request cancelled".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        Ok(Waveform {
            pcm: chatterbox_rs::npy::F32Npy {
                shape: vec![1, 2],
                values: vec![0., 0.25],
            },
        })
    }
}
fn app(probe: Arc<Probe>) -> axum::Router {
    let catalog: Catalog = serde_json::from_value(serde_json::json!({
        "schema_version": 1, "model": "myth", "model_aliases": ["chatterbox-borg"],
        "voices": { "tpol.wav": { "display_name": "tpol", "pipeline": "/unused/pipeline.json", "sha256": "a".repeat(64) } },
        "sampling": {}, "seed": 42, "max_gen_len": 600
    })).unwrap();
    service::router(catalog, probe).unwrap()
}
fn speech(body: serde_json::Value) -> Request<Body> {
    Request::post("/v1/audio/speech")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}
fn input() -> serde_json::Value {
    serde_json::json!({"model": "myth", "input": "Ready, Holmes.", "voice": "tpol.wav", "response_format": "wav"})
}

// Stream transport must make useful audio observable while the next neural
// call is still running. The second call blocks independently of the first.
struct StreamProbe {
    calls: AtomicUsize,
    cancelled: AtomicBool,
    release: AtomicBool,
    fail_second: bool,
}
impl Synthesizer for StreamProbe {
    fn synthesize(
        &self,
        _: &Voice,
        _: &SpeechRequest,
        cancel: &AtomicBool,
    ) -> chatterbox_rs::Result<Waveform> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call > 0 {
            if self.fail_second {
                return Err("private model diagnostic".into());
            }
            while !self.release.load(Ordering::SeqCst) {
                if cancel.load(Ordering::SeqCst) {
                    self.cancelled.store(true, Ordering::SeqCst);
                    // Simulate a kernel that still owns native state after
                    // noticing cancellation. Admission must remain occupied.
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            if cancel.load(Ordering::SeqCst) {
                return Err("cancelled".into());
            }
        }
        Ok(Waveform {
            pcm: chatterbox_rs::npy::F32Npy {
                shape: vec![1, 2400],
                values: vec![0.25; 2400],
            },
        })
    }
}
fn stream_app(fail_second: bool) -> (axum::Router, Arc<StreamProbe>) {
    let probe = Arc::new(StreamProbe {
        calls: AtomicUsize::new(0),
        cancelled: AtomicBool::new(false),
        release: AtomicBool::new(false),
        fail_second,
    });
    let catalog: Catalog = serde_json::from_value(serde_json::json!({
        "schema_version":1,"model":"myth","voices":{"tpol.wav":{"display_name":"tpol","pipeline":"/unused/pipeline.json","sha256":"a".repeat(64)}},
        "sampling":{},"seed":42,"max_gen_len":600
    })).unwrap();
    (service::router(catalog, probe.clone()).unwrap(), probe)
}
fn stream_request() -> Request<Body> {
    let mut data = input();
    data["input"] = serde_json::json!(
        ("This is a complete sentence with enough words to form its own segment. ").repeat(4)
    );
    Request::post("/v1/audio/speech/stream")
        .header("content-type", "application/json")
        .body(Body::from(data.to_string()))
        .unwrap()
}
async fn stream_next(stream: &mut axum::body::BodyDataStream) -> serde_json::Value {
    use futures_core::Stream;
    let data = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        std::future::poll_fn(|cx| std::pin::Pin::new(&mut *stream).poll_next(cx)),
    )
    .await
    .unwrap()
    .unwrap()
    .unwrap();
    serde_json::from_slice(&data).unwrap()
}
async fn wait_for(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn stream_before_completion() {
    let (router, probe) = stream_app(false);
    let response = router.clone().oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/x-ndjson");
    assert!(!response.headers().contains_key("x-myth-finished-unix-ms"));
    let mut stream = response.into_body().into_data_stream();
    assert_eq!(stream_next(&mut stream).await["type"], "start");
    let first = stream_next(&mut stream).await;
    assert_eq!(first["type"], "audio");
    assert_eq!(first["index"], 0);
    assert_eq!(first["offset_samples"], 0);
    let wav = base64::decode(first["wav"].as_str().unwrap()).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(wav.len(), 44 + 2400 * 2);
    wait_for(|| probe.calls.load(Ordering::SeqCst) == 2).await;
    assert!(
        !probe.release.load(Ordering::SeqCst),
        "later synthesis still blocked"
    );
    let busy = router.clone().oneshot(speech(input())).await.unwrap();
    assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
    probe.release.store(true, Ordering::SeqCst);
    let mut samples = first["samples"].as_u64().unwrap();
    let mut frames = 1;
    let mut digest = sha2::Sha256::default();
    use sha2::Digest;
    digest.update(&wav[44..]);
    loop {
        let item = stream_next(&mut stream).await;
        if item["type"] == "complete" {
            assert_eq!(item["frames"], frames);
            assert_eq!(item["samples"], samples);
            assert_eq!(item["pcm_sha256"], format!("{:x}", digest.finalize()));
            assert!(
                item["finished_unix_ms"].as_u64().unwrap()
                    >= item["started_unix_ms"].as_u64().unwrap()
            );
            break;
        }
        assert_eq!(item["type"], "audio");
        assert_eq!(item["index"], frames);
        assert_eq!(item["offset_samples"], samples);
        let wav = base64::decode(item["wav"].as_str().unwrap()).unwrap();
        digest.update(&wav[44..]);
        samples += item["samples"].as_u64().unwrap();
        frames += 1;
    }
    assert_eq!(probe.calls.load(Ordering::SeqCst), 4);
}
#[tokio::test]
async fn stream_cancel_retains_admission_until_worker_exit() {
    let (router, probe) = stream_app(false);
    let response = router.clone().oneshot(stream_request()).await.unwrap();
    let mut stream = response.into_body().into_data_stream();
    stream_next(&mut stream).await;
    stream_next(&mut stream).await;
    wait_for(|| probe.calls.load(Ordering::SeqCst) == 2).await;
    drop(stream);
    wait_for(|| probe.cancelled.load(Ordering::SeqCst)).await;
    assert_eq!(
        router
            .clone()
            .oneshot(speech(input()))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    probe.release.store(true, Ordering::SeqCst);
    // A new turn can enter only after the old worker drops its native permit.
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let r = router.clone().oneshot(speech(input())).await.unwrap();
            if r.status() == StatusCode::OK {
                break;
            }
            assert_eq!(r.status(), StatusCode::TOO_MANY_REQUESTS);
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        probe.calls.load(Ordering::SeqCst),
        3,
        "cancelled remainder never synthesized"
    );
}
#[tokio::test]
async fn stream_failure_has_no_success_receipt_or_private_diagnostic() {
    let (router, _) = stream_app(true);
    let response = router.oneshot(stream_request()).await.unwrap();
    let mut stream = response.into_body().into_data_stream();
    stream_next(&mut stream).await;
    stream_next(&mut stream).await;
    let failure = stream_next(&mut stream).await;
    assert_eq!(failure["type"], "error");
    assert!(!failure.to_string().contains("private"));
    use futures_core::Stream;
    assert!(
        std::future::poll_fn(|cx| std::pin::Pin::new(&mut stream).poll_next(cx))
            .await
            .is_none()
    );
}
#[tokio::test]
async fn stream_slow_reader_bounds_producer() {
    let (router, probe) = stream_app(false);
    probe.release.store(true, Ordering::SeqCst);
    let response = router.oneshot(stream_request()).await.unwrap();
    // Without polling the body, only the start frame fits. At most one waveform
    // can be generated while its send waits; no duration-sized producer queue.
    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
    assert!(probe.calls.load(Ordering::SeqCst) <= 1);
    drop(response);
}
async fn wait_calls(probe: &Probe, count: usize) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while probe.calls.load(Ordering::SeqCst) < count {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn single_inference() {
    let probe = Probe::new(true);
    let router = app(probe.clone());
    let first = tokio::spawn(router.clone().oneshot(speech(input())));
    wait_calls(&probe, 1).await;
    let health = router
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(health.into_body(), 10000).await.unwrap()).unwrap();
    assert_eq!(body["busy"], true);
    let unload = router
        .clone()
        .oneshot(Request::post("/unload_model").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(unload.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(body["new_voice_preparation"], false);
    let busy = router.clone().oneshot(speech(input())).await.unwrap();
    assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
    probe.block.store(false, Ordering::SeqCst);
    let response = first.await.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "audio/wav");
    for name in [
        "x-myth-started-unix-ms",
        "x-myth-finished-unix-ms",
        "x-myth-duration-ms",
        "x-myth-sha256",
    ] {
        assert!(response.headers().contains_key(name));
    }
    let bytes = to_bytes(response.into_body(), 10000).await.unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    // The permit returns after native work finishes, including on subsequent requests.
    let replay = router.oneshot(speech(input())).await.unwrap();
    assert_eq!(bytes, to_bytes(replay.into_body(), 10000).await.unwrap());
}

#[tokio::test]
async fn unload_retains_permit_until_engine_teardown_and_preserves_state_on_failure() {
    let probe = Probe::new(true);
    let router = app(probe.clone());
    let unloading = tokio::spawn(
        router
            .clone()
            .oneshot(Request::post("/unload_model").body(Body::empty()).unwrap()),
    );
    wait_for(|| probe.unloads.load(Ordering::SeqCst) == 1).await;
    // Resource release is still executing; neither speech nor another unload
    // may acquire inference ownership, and health must remain responsive.
    assert_eq!(
        router
            .clone()
            .oneshot(speech(input()))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    let health = router
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let state: serde_json::Value =
        serde_json::from_slice(&to_bytes(health.into_body(), 10000).await.unwrap()).unwrap();
    assert_eq!(state["loaded"], true);
    assert_eq!(state["busy"], true);
    probe.block.store(false, Ordering::SeqCst);
    assert_eq!(unloading.await.unwrap().unwrap().status(), StatusCode::OK);
    assert_eq!(
        router
            .clone()
            .oneshot(speech(input()))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );

    let failing = Probe::new(false);
    failing.fail_unload.store(true, Ordering::SeqCst);
    let router = app(failing.clone());
    assert_eq!(
        router
            .clone()
            .oneshot(Request::post("/unload_model").body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(
        router.oneshot(speech(input())).await.unwrap().status(),
        StatusCode::OK
    );
}
#[tokio::test]
async fn cancelled_request() {
    let probe = Probe::new(true);
    let router = app(probe.clone());
    let handle = tokio::spawn(router.clone().oneshot(speech(input())));
    wait_calls(&probe, 1).await;
    handle.abort();
    let _ = handle.await;
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !probe.cancelled.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        loop {
            let response = router
                .clone()
                .oneshot(Request::get("/health").body(Body::empty()).unwrap())
                .await
                .unwrap();
            let body: serde_json::Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 10000).await.unwrap())
                    .unwrap();
            if body["busy"] == false {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    probe.block.store(false, Ordering::SeqCst);
    assert_eq!(
        router.oneshot(speech(input())).await.unwrap().status(),
        StatusCode::OK
    );
}
#[tokio::test]
async fn response_formats_use_native_codecs_and_truthful_headers() {
    let probe = Probe::new(false);
    let router = app(probe.clone());
    for (format, media, signature) in [
        ("wav", "audio/wav", b"RIFF".as_slice()),
        ("mp3", "audio/mpeg", b"".as_slice()),
        ("opus", "audio/ogg", b"OggS".as_slice()),
    ] {
        let mut body = input();
        body["response_format"] = serde_json::json!(format);
        let response = router.clone().oneshot(speech(body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{format}");
        assert_eq!(response.headers()["content-type"], media);
        assert_eq!(response.headers()["x-myth-samples"], "2");
        let digest = response.headers()["x-myth-sha256"]
            .to_str()
            .unwrap()
            .to_owned();
        let bytes = to_bytes(response.into_body(), 10000).await.unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.starts_with(signature));
        use sha2::{Digest, Sha256};
        assert_eq!(digest, format!("{:x}", Sha256::digest(&bytes)));
    }
    assert_eq!(probe.calls.load(Ordering::SeqCst), 3);
}
#[tokio::test]
async fn speed_changes_all_response_lengths_and_normal_replay() {
    let probe = Probe::new(false);
    let router = app(probe.clone());
    let normal = router.clone().oneshot(speech(input())).await.unwrap();
    let normal = to_bytes(normal.into_body(), 10000).await.unwrap();
    for format in ["wav", "mp3", "opus"] {
        for (speed, samples) in [(0.5, "4"), (2., "1")] {
            let mut body = input();
            body["speed"] = serde_json::json!(speed);
            body["response_format"] = serde_json::json!(format);
            let response = router.clone().oneshot(speech(body)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["x-myth-samples"], samples);
            assert!(!to_bytes(response.into_body(), 10000)
                .await
                .unwrap()
                .is_empty());
        }
    }
    let replay = router.oneshot(speech(input())).await.unwrap();
    assert_eq!(normal, to_bytes(replay.into_body(), 10000).await.unwrap());
}
#[tokio::test]
async fn invalid_controls_never_invoke_inference() {
    let probe = Probe::new(false);
    let router = app(probe.clone());
    for (key, value, status) in [
        (
            "voice",
            serde_json::json!("../tpol.wav"),
            StatusCode::NOT_FOUND,
        ),
        (
            "voice",
            serde_json::json!("unprepared.wav"),
            StatusCode::NOT_FOUND,
        ),
        ("model", serde_json::json!("other"), StatusCode::BAD_REQUEST),
        ("input", serde_json::json!("   "), StatusCode::BAD_REQUEST),
        (
            "response_format",
            serde_json::json!("aac"),
            StatusCode::BAD_REQUEST,
        ),
        ("speed", serde_json::json!(0.), StatusCode::BAD_REQUEST),
        ("speed", serde_json::json!(-1.), StatusCode::BAD_REQUEST),
        (
            "temperature",
            serde_json::json!(0.2),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
    ] {
        let mut body = input();
        body[key] = value;
        assert_eq!(
            router.clone().oneshot(speech(body)).await.unwrap().status(),
            status,
            "{key}"
        );
    }
    assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
    let voices = router
        .oneshot(
            Request::get("/get_predefined_voices")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let list: serde_json::Value =
        serde_json::from_slice(&to_bytes(voices.into_body(), 10000).await.unwrap()).unwrap();
    assert_eq!(
        list,
        serde_json::json!([{"filename": "tpol.wav", "display_name": "tpol"}])
    );
}
#[test]
fn native_pipeline_cancellation_precedes_asset_loading() {
    let result = chatterbox_rs::pipeline::render(
        std::path::Path::new("/not-admitted"),
        &"0".repeat(64),
        "hello",
        42,
        600,
        &Default::default(),
        || true,
    );
    assert!(result.err().unwrap().contains("cancelled"));
}

// Unlike StreamProbe's sentence boundaries, these three fragments all belong
// to one synthesizer invocation for one short utterance.
struct TokenProbe {
    inner: Arc<StreamProbe>,
}
#[tokio::test]
async fn token_stream_workshop_joins_sentences_once_and_admits_dynamic_only_catalog() {
    let root = std::env::temp_dir().join(format!(
        "myth-token-stream-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(root.join("ui")).unwrap();
    std::fs::create_dir_all(root.join("state")).unwrap();
    let (_, probe) = stream_app(false);
    probe.release.store(true, Ordering::SeqCst);
    let mut value = serde_json::json!({"schema_version":1,"model":"myth","voices":{},"sampling":{},"seed":42,"max_gen_len":600,
        "workshop":{"ui_root":root.join("ui"),"state_root":root.join("state"),
            "preparation":{"path":"/unused","sha256":"a".repeat(64)},"prompt":{"path":"/unused","sha256":"a".repeat(64)},
            "pipeline":{"path":"/unused","sha256":"a".repeat(64)}}});
    // An imported workshop inventory must not need a dummy fourth voice.
    let dynamic = service::router(
        serde_json::from_value(value.clone()).unwrap(),
        Arc::new(TokenProbe {
            inner: probe.clone(),
        }),
    )
    .unwrap();
    let r = dynamic
        .clone()
        .oneshot(
            Request::get("/get_predefined_voices")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&to_bytes(r.into_body(), 1000).await.unwrap())
            .unwrap(),
        serde_json::json!([])
    );
    drop(dynamic);
    value["voices"] = serde_json::json!({"tpol.wav":{"display_name":"tpol","pipeline":"/unused","sha256":"a".repeat(64)}});
    let router = service::router(
        serde_json::from_value(value).unwrap(),
        Arc::new(TokenProbe { inner: probe }),
    )
    .unwrap();
    let text = "This is one complete sentence long enough for a segment. This is another complete sentence long enough for a segment.";
    assert_eq!(chatterbox_rs::workshop::chunks(text, Some(50)).len(), 2);
    let request = Request::post("/tts/stream").header("content-type", "application/json").body(Body::from(serde_json::json!({
        "text":text,"predefined_voice_id":"tpol.wav","split_text":true,"chunk_size":50,"speed_factor":1.,"output_format":"wav"}).to_string())).unwrap();
    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    stream_next(&mut stream).await;
    let mut bytes = Vec::new();
    let mut count = 0;
    loop {
        let item = stream_next(&mut stream).await;
        if item["type"] == "complete" {
            assert_eq!(item["frames"], 6);
            assert_eq!(item["samples"], count);
            break;
        }
        assert_eq!(item["offset_samples"], count);
        let wav = base64::decode(item["wav"].as_str().unwrap()).unwrap();
        bytes.extend_from_slice(&wav[44..]);
        count += item["samples"].as_u64().unwrap();
    }
    let mut expected = Vec::new();
    for _ in 0..2 {
        chatterbox_rs::workshop::join(&mut expected, &vec![0.25; 7200]);
    }
    let wav = chatterbox_rs::audio_output::encode(
        &chatterbox_rs::npy::F32Npy {
            shape: vec![1, expected.len()],
            values: expected,
        },
        chatterbox_rs::audio_output::Format::Wav,
        || false,
    )
    .unwrap();
    assert_eq!(bytes, wav[44..]);
}
impl Synthesizer for TokenProbe {
    fn streaming_granularity(&self) -> &'static str {
        "codec-prefix"
    }
    fn synthesize(
        &self,
        voice: &Voice,
        request: &SpeechRequest,
        cancel: &AtomicBool,
    ) -> chatterbox_rs::Result<Waveform> {
        self.inner.synthesize(voice, request, cancel)
    }
    fn synthesize_stream(
        &self,
        voice: &Voice,
        request: &SpeechRequest,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Waveform, bool) -> chatterbox_rs::Result<()>,
    ) -> chatterbox_rs::Result<()> {
        for index in 0..3 {
            emit(self.inner.synthesize(voice, request, cancel)?, index == 2)?;
        }
        Ok(())
    }
}
fn token_app(fail: bool) -> (axum::Router, Arc<StreamProbe>) {
    let (_, inner) = stream_app(fail);
    let catalog: Catalog = serde_json::from_value(serde_json::json!({
        "schema_version":1,"model":"myth","voices":{"tpol.wav":{"display_name":"tpol","pipeline":"/unused/pipeline.json","sha256":"a".repeat(64)}},
        "sampling":{},"seed":42,"max_gen_len":600
    })).unwrap();
    (
        service::router(
            catalog,
            Arc::new(TokenProbe {
                inner: inner.clone(),
            }),
        )
        .unwrap(),
        inner,
    )
}
fn token_request(speed: f64) -> Request<Body> {
    let mut body = input();
    body["speed"] = serde_json::json!(speed);
    Request::post("/v1/audio/speech/stream")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}
#[tokio::test]
async fn token_stream_emits_within_one_sentence_and_completes_once() {
    let (router, probe) = token_app(false);
    let response = router.clone().oneshot(token_request(1.)).await.unwrap();
    let mut stream = response.into_body().into_data_stream();
    assert_eq!(
        stream_next(&mut stream).await["granularity"],
        "codec-prefix"
    );
    let first = stream_next(&mut stream).await;
    assert_eq!(first["offset_samples"], 0);
    wait_for(|| probe.calls.load(Ordering::SeqCst) == 2).await;
    assert_eq!(
        router
            .clone()
            .oneshot(speech(input()))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    probe.release.store(true, Ordering::SeqCst);
    for index in 1..3 {
        let item = stream_next(&mut stream).await;
        assert_eq!(item["index"], index);
        assert_eq!(item["offset_samples"], index * 2400);
    }
    let complete = stream_next(&mut stream).await;
    assert_eq!(complete["type"], "complete");
    assert_eq!(complete["samples"], 7200);
    assert_eq!(complete["frames"], 3);
    use futures_core::Stream;
    assert!(
        std::future::poll_fn(|cx| std::pin::Pin::new(&mut stream).poll_next(cx))
            .await
            .is_none()
    );
}
#[tokio::test]
async fn token_stream_cancellation_keeps_permit_until_kernel_retires_then_reenters() {
    let (router, probe) = token_app(false);
    let response = router.clone().oneshot(token_request(1.)).await.unwrap();
    let mut stream = response.into_body().into_data_stream();
    stream_next(&mut stream).await;
    stream_next(&mut stream).await;
    wait_for(|| probe.calls.load(Ordering::SeqCst) == 2).await;
    drop(stream);
    wait_for(|| probe.cancelled.load(Ordering::SeqCst)).await;
    assert_eq!(
        router
            .clone()
            .oneshot(speech(input()))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    probe.release.store(true, Ordering::SeqCst);
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if router
                .clone()
                .oneshot(speech(input()))
                .await
                .unwrap()
                .status()
                == StatusCode::OK
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn token_stream_failure_has_no_completion_and_tempo_uses_whole_sentence() {
    let (router, _) = token_app(true);
    let mut stream = router
        .oneshot(token_request(1.))
        .await
        .unwrap()
        .into_body()
        .into_data_stream();
    stream_next(&mut stream).await;
    stream_next(&mut stream).await;
    let failure = stream_next(&mut stream).await;
    assert_eq!(failure["type"], "error");
    assert!(!failure.to_string().contains("private"));
    let (router, probe) = token_app(false);
    let mut stream = router
        .oneshot(token_request(2.))
        .await
        .unwrap()
        .into_body()
        .into_data_stream();
    assert_eq!(stream_next(&mut stream).await["granularity"], "sentence");
    assert_eq!(stream_next(&mut stream).await["samples"], 1200);
    assert_eq!(stream_next(&mut stream).await["type"], "complete");
    assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
}
