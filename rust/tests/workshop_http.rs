//! Workshop ownership and durable state use the production router with a tiny
//! synthesizer; Borg acceptance separately exercises the native preparation.
use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use chatterbox_rs::service::{self, Catalog, SpeechRequest, Synthesizer, Voice, Waveform};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tower::ServiceExt;
struct Probe;
fn import_owner(p: &std::path::Path) -> chatterbox_rs::workshop::Workshop {
    let pin = serde_json::json!({"path":p.join("template.json"),"sha256":"a".repeat(64)});
    let configuration = serde_json::from_value(serde_json::json!({
        "ui_root":p.join("ui"),"state_root":p.join("state"),
        "preparation":pin,"prompt":pin,"pipeline":pin
    }))
    .unwrap();
    chatterbox_rs::workshop::Workshop::open(configuration, 42).unwrap()
}
fn import_input(p: &std::path::Path, value: &serde_json::Value) -> (PathBuf, String) {
    use sha2::{Digest, Sha256};
    let bytes = serde_json::to_vec(value).unwrap();
    let path = p.join("inventory.json");
    std::fs::write(&path, &bytes).unwrap();
    (path, format!("{:x}", Sha256::digest(bytes)))
}
#[test]
fn native_import_rejects_incomplete_inventory_and_missing_preparations() {
    use sha2::{Digest, Sha256};
    let p = root();
    let source = p.join("legacy");
    std::fs::create_dir(&source).unwrap();
    std::fs::write(source.join("retained.wav"), b"retained-recording").unwrap();
    let sha = format!("{:x}", Sha256::digest(b"retained-recording"));
    let w = import_owner(&p);
    let mut input = serde_json::json!({"schema_version":1,"config":{},
        "sources":[{"root":source,"reference":false,"recordings":{}}]});
    let (path, digest) = import_input(&p, &input);
    assert!(w
        .import(&path, &digest, 42)
        .unwrap_err()
        .contains("complete source recording inventory"));
    input["sources"][0]["recordings"] = serde_json::json!({"retained.wav":sha});
    let (path, digest) = import_input(&p, &input);
    let error = w.import(&path, &digest, 42).unwrap_err();
    assert!(
        error.contains("retained.wav") && error.contains("reference preparation was not run"),
        "{error}"
    );
    assert!(!p.join("state/state.json").exists());
    assert!(w.list(false).is_empty());
    assert_eq!(
        std::fs::read(source.join("retained.wav")).unwrap(),
        b"retained-recording"
    );
    // Source changes cannot be reconciled by silently dropping or remapping it.
    std::fs::write(source.join("retained.wav"), b"changed-recording").unwrap();
    assert!(w
        .import(&path, &digest, 42)
        .unwrap_err()
        .contains("inventory differs"));
    assert_eq!(w.config()["generation_defaults"]["seed"], 42);
}

#[test]
#[ignore = "requires existing pinned native prompt tables"]
fn native_import_atomic_reopen_replay_and_changed_pin() {
    use chatterbox_rs::{npy::F32Npy, s3gen::S3GenReference, voice_preparation::PreparedVoice};
    use sha2::{Digest, Sha256};
    let p = root();
    let source = p.join("legacy");
    std::fs::create_dir_all(source.join(".native/prepared")).unwrap();
    std::fs::write(source.join(".native/.lock"), b"").unwrap();
    // Deliberately undecodable bytes: a successful import proves this route
    // neither decodes the recording nor prepares its already retained voice.
    std::fs::write(source.join("retained.wav"), b"not-new-audio").unwrap();
    let reference_source = p.join("legacy-reference");
    std::fs::create_dir(&reference_source).unwrap();
    std::fs::write(reference_source.join("retained.wav"), b"not-new-audio").unwrap();
    let recording_sha = format!("{:x}", Sha256::digest(b"not-new-audio"));
    let prompt = PathBuf::from(std::env::var("CHATTERBOX_VOICE_PROMPT_ASSETS").unwrap())
        .join("retained-assets.json");
    let prompt_sha = format!("{:x}", Sha256::digest(std::fs::read(&prompt).unwrap()));
    let voice = PreparedVoice {
        t3_speaker: F32Npy {
            shape: vec![1, 256],
            values: vec![0.0625; 256],
        },
        t3_tokens: vec![3, 4],
        reference: S3GenReference::from_parts(
            "a".repeat(64),
            vec![5, 6],
            F32Npy {
                shape: vec![1, 5, 80],
                values: vec![-0.125; 400],
            },
            None,
            F32Npy {
                shape: vec![1, 192],
                values: vec![0.25; 192],
            },
        )
        .unwrap(),
    };
    let mut variants = serde_json::json!({});
    for key in ["normalised", "original"] {
        let published = voice
            .publish(
                &prompt,
                &prompt_sha,
                &source.join(".native/prepared").join(key),
            )
            .unwrap();
        variants[key] = serde_json::json!({
            "prompt":{"path":published.prompt.path.strip_prefix(&source).unwrap(),"sha256":published.prompt.sha256},
            "reference":{"path":published.reference.path.strip_prefix(&source).unwrap(),"sha256":published.reference.sha256}});
    }
    let marker = source
        .join(".native")
        .join(format!("{:x}.json", Sha256::digest(b"retained.wav")));
    let binding = serde_json::json!({"schema_version":1,"recording_sha256":recording_sha,
        "models_sha256":"a".repeat(64),"variants":variants});
    std::fs::write(&marker, binding.to_string()).unwrap();
    std::fs::write(p.join("small"), b"{}").unwrap();
    let mut pipeline = serde_json::json!({"schema_version":1});
    for key in [
        "bridge",
        "engine",
        "prompt",
        "reference",
        "flow",
        "conformer",
        "mel",
        "meanflow",
        "hift",
    ] {
        pipeline[key] =
            serde_json::json!({"path":"small","sha256":format!("{:x}", Sha256::digest(b"{}"))});
    }
    let bytes = serde_json::to_vec(&pipeline).unwrap();
    std::fs::write(p.join("template.json"), &bytes).unwrap();
    let configuration:chatterbox_rs::workshop::Configuration = serde_json::from_value(serde_json::json!({
        "ui_root":p.join("ui"),"state_root":p.join("state"),
        "preparation":{"path":"/unused","sha256":"a".repeat(64)},
        "prompt":{"path":prompt,"sha256":prompt_sha},
        "pipeline":{"path":p.join("template.json"),"sha256":format!("{:x}",Sha256::digest(bytes))}
    })).unwrap();
    let input = serde_json::json!({"schema_version":1,"config":{
        "server":{"port":8004},"paths":{"output":"outputs"},
        "generation_defaults":{"temperature":0.35,"seed":17},"ui_state":{"last_text":"retained"}},
        "sources":[{"root":source,"reference":false,"recordings":{"retained.wav":recording_sha}},
            {"root":reference_source,"reference":true,"recordings":{"retained.wav":recording_sha}}]});
    let (path, sha) = import_input(&p, &input);
    let w = chatterbox_rs::workshop::Workshop::open(configuration.clone(), 42).unwrap();
    // Incomplete staging from a terminated attempt has no visible registration.
    std::fs::create_dir(p.join("state/.upload-interrupted")).unwrap();
    assert!(w.list(false).is_empty());
    // The established source owner can defer migration without either process
    // mutating the other's state. The lock is held through native publication.
    let source_lock = std::fs::File::open(source.join(".native/.lock")).unwrap();
    use std::os::fd::AsRawFd;
    assert_eq!(
        unsafe { libc::flock(source_lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    assert!(w
        .import(&path, &sha, 42)
        .unwrap_err()
        .contains("being changed by its owner"));
    assert!(!p.join("state/state.json").exists());
    drop(source_lock);
    let result = w.import(&path, &sha, 42).unwrap();
    assert_eq!(result["voices"], 1);
    assert_eq!(result["reference_preparations"], 0);
    assert_eq!(w.list(false)[0].0, "retained.wav");
    assert_eq!(w.list(true)[0].0, "retained.wav");
    assert_eq!(w.config()["generation_defaults"]["temperature"], 0.35);
    assert_eq!(w.config()["generation_defaults"]["seed"], 17);
    assert_eq!(w.config()["ui_state"]["last_text"], "retained");
    assert!(w.config().get("server").is_none());
    let native = w.source("retained.wav").unwrap().unwrap();
    assert_ne!(native, source.join("retained.wav"));
    assert_eq!(std::fs::read(&native).unwrap(), b"not-new-audio");
    let normal = w.resolve("retained.wav", true).unwrap().unwrap();
    let original = w.resolve("retained.wav", false).unwrap().unwrap();
    assert_ne!(normal.pipeline, original.pipeline);
    // The operator's offline command uses the same transaction and emits its
    // timing receipt. A running owner must reject a competing importer.
    let config_path = p.join("workshop.json");
    let config_bytes = serde_json::to_vec(&serde_json::json!({
        "ui_root":configuration.ui_root,"state_root":configuration.state_root,
        "preparation":{"path":configuration.preparation.path,"sha256":configuration.preparation.sha256},
        "prompt":{"path":configuration.prompt.path,"sha256":configuration.prompt.sha256},
        "pipeline":{"path":configuration.pipeline.path,"sha256":configuration.pipeline.sha256}
    })).unwrap();
    std::fs::write(&config_path, &config_bytes).unwrap();
    let config_sha = format!("{:x}", Sha256::digest(&config_bytes));
    let cli = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args([
                "import-workshop",
                config_path.to_str().unwrap(),
                &config_sha,
                path.to_str().unwrap(),
                &sha,
                "42",
            ])
            .output()
            .unwrap()
    };
    assert!(!cli().status.success());
    w.apply(serde_json::json!({"ui_state":{"last_text":"later Apply"}}))
        .unwrap();
    drop(w);
    let command = cli();
    assert!(
        command.status.success(),
        "{}",
        String::from_utf8_lossy(&command.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&command.stdout).unwrap();
    assert_eq!(receipt["status"], "workshop_imported");
    assert_eq!(receipt["result"]["reused"], true);
    let reopened = chatterbox_rs::workshop::Workshop::open(configuration.clone(), 42).unwrap();
    assert_eq!(reopened.import(&path, &sha, 42).unwrap()["reused"], true);
    assert_eq!(reopened.config()["ui_state"]["last_text"], "later Apply");
    // Native tensors are independent of the old owner after migration.
    let source_manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(source.join(".native/prepared/normalised/reference.json")).unwrap(),
    )
    .unwrap();
    let source_tensor = source
        .join(".native/prepared/normalised")
        .join(source_manifest["embedding"]["file"].as_str().unwrap());
    std::fs::write(source_tensor, b"retired-old-tensor").unwrap();
    assert_eq!(reopened.import(&path, &sha, 42).unwrap()["reused"], true);
    // Corruption inside the native bundle must invalidate exact replay even
    // when its aggregate and nested manifest checksums remain unchanged.
    let aggregate: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&normal.pipeline).unwrap()).unwrap();
    let reference_path = PathBuf::from(aggregate["reference"]["path"].as_str().unwrap());
    let reference_manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&reference_path).unwrap()).unwrap();
    let tensor = reference_path
        .parent()
        .unwrap()
        .join(reference_manifest["embedding"]["file"].as_str().unwrap());
    let original_tensor = std::fs::read(&tensor).unwrap();
    std::fs::write(&tensor, b"corrupt-native-tensor").unwrap();
    assert!(reopened.import(&path, &sha, 42).is_err());
    std::fs::write(&tensor, &original_tensor).unwrap();
    // Exact replay cannot conceal changed pins, and an unrelated manifest
    // cannot replace an existing native owner's state.
    let changed = serde_json::json!({"schema_version":1,"config":{},"sources":input["sources"]});
    let (changed_path, changed_sha) = import_input(&p, &changed);
    assert!(reopened
        .import(&changed_path, &changed_sha, 42)
        .unwrap_err()
        .contains("already exists"));
    let (path, sha) = import_input(&p, &input);
    std::fs::write(&normal.pipeline, b"changed").unwrap();
    assert!(reopened.import(&path, &sha, 42).is_err());
    assert_eq!(reopened.list(false).len(), 1);
    assert_eq!(
        std::fs::read(source.join("retained.wav")).unwrap(),
        b"not-new-audio"
    );
}
#[test]
fn reload_admission_hashes_large_assets_in_bounded_memory() {
    use sha2::{Digest, Sha256};
    if std::env::var_os("MYTH_RELOAD_MEMORY_CHILD").is_some() {
        // The admitted asset is twice this process allowance. Whole-asset
        // buffering must fail; streaming checksum admission must still pass.
        let limit = libc::rlimit {
            rlim_cur: 128 * 1024 * 1024,
            rlim_max: 128 * 1024 * 1024,
        };
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &limit) }, 0);
        let path = PathBuf::from(std::env::var("MYTH_RELOAD_PIPELINE").unwrap());
        let bytes = std::fs::read(&path).unwrap();
        chatterbox_rs::pipeline::admit(&path, &format!("{:x}", Sha256::digest(bytes))).unwrap();
        return;
    }
    let p = root();
    let large = p.join("large.engine");
    let file = std::fs::File::create(&large).unwrap();
    file.set_len(256 * 1024 * 1024).unwrap();
    drop(file);
    let mut checksum = Sha256::new();
    let zeros = [0u8; 65536];
    for _ in 0..4096 {
        checksum.update(zeros);
    }
    let large_sha = format!("{:x}", checksum.finalize());
    std::fs::write(p.join("small"), b"{}").unwrap();
    let small_sha = format!("{:x}", Sha256::digest(b"{}"));
    let mut manifest = serde_json::json!({"schema_version":1});
    for key in [
        "bridge",
        "engine",
        "prompt",
        "reference",
        "flow",
        "conformer",
        "mel",
        "meanflow",
        "hift",
    ] {
        manifest[key] = if key == "engine" {
            serde_json::json!({"path":"large.engine","sha256":large_sha})
        } else {
            serde_json::json!({"path":"small","sha256":small_sha})
        };
    }
    let pipeline = p.join("pipeline.json");
    std::fs::write(&pipeline, manifest.to_string()).unwrap();
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "reload_admission_hashes_large_assets_in_bounded_memory",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("MYTH_RELOAD_MEMORY_CHILD", "1")
        .env("MYTH_RELOAD_PIPELINE", &pipeline)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
impl Synthesizer for Probe {
    fn synthesize(
        &self,
        _: &Voice,
        _: &SpeechRequest,
        cancel: &AtomicBool,
    ) -> chatterbox_rs::Result<Waveform> {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        Ok(Waveform {
            pcm: chatterbox_rs::npy::F32Npy {
                shape: vec![1, 240],
                values: vec![0.; 240],
            },
        })
    }
}
#[test]
fn workshop_defaults_sentence_controls_and_fixed_join_pause() {
    use chatterbox_rs::workshop::{chunks, join, Request as WorkshopRequest};
    let config = serde_json::json!({"generation_defaults":{"temperature":0.35,"seed":17,"speed_factor":1.5},"audio_output":{"format":"opus"}});
    let request: WorkshopRequest = serde_json::from_value(
        serde_json::json!({"text":"Ready.","predefined_voice_id":"retained.wav"}),
    )
    .unwrap();
    let request = request.speech("myth".into(), &config).unwrap();
    assert_eq!(request.temperature, Some(0.35));
    assert_eq!(request.seed, Some(17));
    assert_eq!(request.speed, 1.5);
    assert_eq!(request.response_format, "opus");
    let long="This sentence is kept whole even though it exceeds the requested target size. Next sentence.";
    assert_eq!(
        chunks(long, Some(30)),
        vec![
            "This sentence is kept whole even though it exceeds the requested target size.",
            "Next sentence."
        ]
    );
    let mut output = vec![1.; 2400];
    join(&mut output, &vec![1.; 2400]);
    assert_eq!(output.len(), 9600);
    assert!(output[2400..7200].iter().all(|v| *v == 0.));
    assert_eq!(output[0], 1.);
    assert_eq!(*output.last().unwrap(), 1.);
}
fn root() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "myth-workshop-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(p.join("ui")).unwrap();
    std::fs::create_dir_all(p.join("state")).unwrap();
    std::fs::write(p.join("ui/index.html"), "<html>myth workshop</html>").unwrap();
    p
}
fn app(p: &std::path::Path) -> axum::Router {
    let c: Catalog=serde_json::from_value(serde_json::json!({
 "schema_version":1,"model":"myth","voices":{"retained.wav":{"display_name":"retained","pipeline":"/unused","sha256":"a".repeat(64)}},
 "seed":42,"max_gen_len":600,
 "workshop":{"ui_root":p.join("ui"),"state_root":p.join("state"),"preparation":{"path":"/unused","sha256":"a".repeat(64)},
 "prompt":{"path":"/unused","sha256":"a".repeat(64)},"pipeline":{"path":"/unused","sha256":"a".repeat(64)},
 "delete_authority_sha256":format!("{:x}",<sha2::Sha256 as sha2::Digest>::digest(b"operator-token"))}
 })).unwrap();
    service::router(c, Arc::new(Probe)).unwrap()
}
fn req(path: &str, value: serde_json::Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(value.to_string()))
        .unwrap()
}
async fn json(r: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(&to_bytes(r.into_body(), 100000).await.unwrap()).unwrap()
}
#[tokio::test]
async fn settings_apply_reopen_reset_and_reject_unsupported() {
    let p = root();
    let a = app(&p);
    let r=a.clone().oneshot(req("/save_settings",serde_json::json!({"generation_defaults":{"temperature":0.35,"speed_factor":1.5},"ui_state":{"last_text":"retained"}}))).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(json(r).await["restart_needed"], false);
    drop(a);
    let reopened = app(&p);
    let r = reopened
        .clone()
        .oneshot(
            Request::get("/api/ui/initial-data")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = json(r).await;
    assert_eq!(body["config"]["generation_defaults"]["temperature"], 0.35);
    assert_eq!(body["config"]["generation_defaults"]["speed_factor"], 1.5);
    assert_eq!(body["config"]["ui_state"]["last_text"], "retained");
    assert_eq!(body["model_info"]["type"], "turbo");
    for bad in [
        serde_json::json!({"model":{"repo_id":"multilingual"}}),
        serde_json::json!({"paths":{"output":"/etc"}}),
        serde_json::json!({"audio_output":{"sample_rate":48000}}),
        serde_json::json!({"generation_defaults":{"speed_factor":0}}),
    ] {
        assert_eq!(
            reopened
                .clone()
                .oneshot(req("/save_settings", bad))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    assert_eq!(
        reopened
            .clone()
            .oneshot(req("/reset_settings", serde_json::json!({})))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    drop(reopened);
    let reset = app(&p)
        .oneshot(
            Request::get("/api/ui/initial-data")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        json(reset).await["config"]["generation_defaults"]["speed_factor"],
        1.
    );
}
#[tokio::test]
async fn workshop_tts_and_unload_share_native_boundary() {
    let p = root();
    let a = app(&p);
    let body = serde_json::json!({"text":"Ready, Holmes.","temperature":0.8,"exaggeration":0.5,"cfg_weight":0.5,"speed_factor":1.,"seed":42,"language":"en","voice_mode":"predefined","split_text":false,"chunk_size":120,"output_format":"wav","predefined_voice_id":"retained.wav"});
    let r = a.clone().oneshot(req("/tts", body.clone())).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert!(r.headers()["content-disposition"]
        .to_str()
        .unwrap()
        .contains(".wav"));
    assert_eq!(
        a.clone()
            .oneshot(req("/unload_model", serde_json::json!({})))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    let info = json(
        a.clone()
            .oneshot(Request::get("/api/model-info").body(Body::empty()).unwrap())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(info["loaded"], false);
    assert_eq!(
        a.oneshot(req("/tts", body)).await.unwrap().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}
#[tokio::test]
async fn static_paths_and_delete_authority_are_bounded() {
    let p = root();
    let a = app(&p);
    let r = a
        .clone()
        .oneshot(Request::get("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert!(
        std::str::from_utf8(&to_bytes(r.into_body(), 1000).await.unwrap())
            .unwrap()
            .contains("myth workshop")
    );
    for path in ["/ui/../state/state.json", "/vendor/%2e%2e/state.json"] {
        assert_ne!(
            a.clone()
                .oneshot(Request::get(path).body(Body::empty()).unwrap())
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }
    let denied = a
        .oneshot(
            Request::delete("/delete_predefined_voice?filename=retained.wav")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::FORBIDDEN);
}
