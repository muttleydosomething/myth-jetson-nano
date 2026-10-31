//! Exercise engine identity and request lifetimes without CUDA. The native fake
//! counts actual opens, active owners and context-local positions; timing and
//! memory acceptance still require the separate real Borg route.
use chatterbox_rs::{
    generation::{self, Decoder},
    npy::Npy,
    sampling::Sampling,
    tensorrt::EngineCache,
};
use sha2::{Digest, Sha256};
use std::{path::Path, process::Command};

fn arrays(mark: u16) -> (Npy, Npy) {
    (
        Npy {
            shape: vec![1, 1, 1024],
            halves: vec![mark; 1024],
        },
        Npy {
            shape: vec![6563, 1024],
            halves: vec![mark; 6563 * 1024],
        },
    )
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
#[ignore = "set CHATTERBOX_TOKEN_PIPELINE to a pinned deployed pipeline; requires Borg CUDA"]
fn native_token_buffers_replay_growth_and_reset() {
    use chatterbox_rs::conditioning::PromptAssets;
    use std::sync::Arc;
    let path = std::env::var("CHATTERBOX_TOKEN_PIPELINE").unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let pin = |name: &str, key: &str| manifest[name][key].as_str().unwrap();
    let assets =
        PromptAssets::open(Path::new(pin("prompt", "path")), pin("prompt", "sha256")).unwrap();
    let prompt = assets
        .assemble(&assets.tokenize("Ready, Holmes.").unwrap())
        .unwrap();
    let embeddings = Arc::new(assets.into_speech_embeddings());
    let mut cache = EngineCache::default();
    let mut open = || {
        cache
            .session(
                Path::new(pin("bridge", "path")),
                Path::new(pin("engine", "path")),
                pin("engine", "sha256"),
                Npy {
                    shape: prompt.shape.clone(),
                    halves: prompt.halves.clone(),
                },
                embeddings.clone(),
            )
            .unwrap()
            .0
    };
    let mut decoder = open();
    let initial = decoder.prefill().unwrap();
    // Teacher-force different tokens through both KV banks and across multiple
    // capacity boundaries. Compare every logit on replay, not only an argmax.
    let mut replay = Vec::new();
    for i in 0..257 {
        replay.push(decoder.decode((i * 13 % 6560) as u32).unwrap());
    }
    decoder.reset().unwrap();
    assert_eq!(decoder.prefill().unwrap(), initial);
    for (i, expected) in replay.iter().enumerate() {
        assert_eq!(
            decoder.decode((i * 13 % 6560) as u32).unwrap(),
            *expected,
            "KV replay differs at step {i}"
        );
    }
    // Cancellation must release request state even after advancing the cache.
    let mut checks = 0;
    let error = generation::generate(&mut decoder, &Sampling::default(), 600, 42, || {
        checks += 1;
        checks == 4
    })
    .unwrap_err();
    assert!(error.contains("cancelled"), "{error}");
    assert_eq!(decoder.prefill().unwrap(), initial);
    drop(decoder);
    let mut reused = open();
    assert_eq!(reused.prefill().unwrap(), initial);
    drop(reused);
    cache.clear();
}

#[test]
fn engine_cache_reuses_admitted_contexts_and_retires_changed_or_failed_state() {
    let root =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("engine-cache-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("engine.c");
    let bridge = root.join("engine.so");
    std::fs::write(&source, r#"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
typedef struct { int id, refs; } Engine;
typedef struct { Engine* engine; int position; } Session;
static int opens=0, live=0, peak=0, sessions=0, fail_session=0, context_opens=0;
int cb_metric(int key) { return key==0?opens:key==1?live:key==2?peak:key==3?sessions:context_opens; }
void cb_fail_session(void) { fail_session=1; }
void* cb_engine_open(const char* path,char* error,size_t cap) {
    FILE* f=fopen(path,"rb"); int tag=f?fgetc(f):'!'; if(f)fclose(f);
    if(tag=='!') { snprintf(error,cap,"injected engine failure");return NULL; }
    Engine* e=calloc(1,sizeof(*e));e->id=++opens;e->refs=1;
    if(++live>peak)peak=live;return e;
}
void cb_engine_close(void* ptr) {
    Engine* e=ptr;if(--e->refs==0){--live;free(e);}
}
void* cb_session_open(void* ptr,char* error,size_t cap) {
    if(fail_session){fail_session=0;snprintf(error,cap,"injected context failure");return NULL;}
    Session* s=calloc(1,sizeof(*s));s->engine=ptr;++s->engine->refs;++sessions;++context_opens;return s;
}
int cb_step(void* ptr,const unsigned short* input,size_t frames,float* out,size_t n,char* err,size_t cap) {
    Session* s=ptr;
    if(input[0]==99){snprintf(err,cap,"injected inference failure");return -1;}
    for(size_t i=0;i<n;++i)out[i]=0;
    out[0]=(float)s->engine->id;out[1]=(float)input[0];out[2]=(float)s->position;
    s->position+=(int)frames;return 6563;
}
void cb_reset(void* ptr) { ((Session*)ptr)->position=0; }
void cb_close(void* ptr) { Session* s=ptr;cb_engine_close(s->engine);--sessions;free(s); }
"#).unwrap();
    assert!(Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(&source)
        .arg("-o")
        .arg(&bridge)
        .status()
        .unwrap()
        .success());
    let engine = root.join("weights");
    std::fs::write(&engine, b"one").unwrap();
    let mut cache = EngineCache::default();
    // Test instrumentation is a second reference to the same registered code.
    let lib = unsafe { libloading::Library::new(&bridge).unwrap() };
    let metric = unsafe {
        lib.get::<unsafe extern "C" fn(i32) -> i32>(b"cb_metric\0")
            .unwrap()
    };
    let fail_session = unsafe {
        lib.get::<unsafe extern "C" fn()>(b"cb_fail_session\0")
            .unwrap()
    };
    let get = |key| unsafe { metric(key) };
    let (p, e) = arrays(7);
    let (mut first, hit) = cache
        .session(&bridge, &engine, &digest(b"one"), p, e)
        .unwrap();
    assert!(!hit);
    assert_eq!(&first.prefill().unwrap()[..3], &[1., 7., 0.]);
    assert_eq!(first.decode(5).unwrap()[2], 1.);
    drop(first);
    assert_eq!((get(0), get(1), get(3), get(4)), (1, 1, 1, 1));

    let (p, e) = arrays(8);
    let (mut second, hit) = cache
        .session(&bridge, &engine, &digest(b"one"), p, e)
        .unwrap();
    assert!(hit);
    assert_eq!(&second.prefill().unwrap()[..3], &[1., 8., 0.]);
    assert_eq!(get(4), 1, "unchanged weights reuse the actual context");
    // Cancellation runs the real generation cleanup; the following request
    // must start with empty KV even after a partially advanced context.
    let mut checks = 0;
    assert!(
        generation::generate(&mut second, &Sampling::default(), 5, 42, || {
            checks += 1;
            checks > 1
        })
        .is_err()
    );
    assert_eq!(second.prefill().unwrap()[2], 0.);
    drop(second);

    let (p, e) = arrays(99);
    let (mut failed, hit) = cache
        .session(&bridge, &engine, &digest(b"one"), p, e)
        .unwrap();
    assert!(hit && failed.prefill().is_err());
    assert_eq!(get(4), 1, "cancelled request returned a clean context");
    drop(failed);
    unsafe { fail_session() };
    let (p, e) = arrays(4);
    assert!(cache
        .session(&bridge, &engine, &digest(b"one"), p, e)
        .err()
        .unwrap()
        .contains("context failure"));
    assert_eq!((get(0), get(1), get(3)), (1, 1, 0));
    let (p, e) = arrays(7);
    let (mut recovered, hit) = cache
        .session(&bridge, &engine, &digest(b"one"), p, e)
        .unwrap();
    assert!(hit);
    assert_eq!(&recovered.prefill().unwrap()[..3], &[1., 7., 0.]);
    drop(recovered);

    // A hit still checks physical content; a changed pin replaces the engine
    // rather than retaining old weights or allowing a second resident owner.
    std::fs::write(&engine, b"two").unwrap();
    let (p, e) = arrays(7);
    assert!(cache
        .session(&bridge, &engine, &digest(b"one"), p, e)
        .err()
        .unwrap()
        .contains("checksum"));
    let (p, e) = arrays(7);
    let (mut replaced, hit) = cache
        .session(&bridge, &engine, &digest(b"two"), p, e)
        .unwrap();
    assert!(!hit);
    assert_eq!(replaced.prefill().unwrap()[0], 2.);
    drop(replaced);
    assert_eq!((get(0), get(1), get(2)), (2, 1, 1));
    let other = root.join("other.so");
    std::fs::copy(&bridge, &other).unwrap();
    let (p, e) = arrays(7);
    assert!(cache
        .session(&other, &engine, &digest(b"two"), p, e)
        .err()
        .unwrap()
        .contains("process bridge"));

    // Failed replacement evicts the old engine and leaves a retryable empty
    // owner, while clearing an active session is safe via native shared refs.
    std::fs::write(&engine, b"!").unwrap();
    let (p, e) = arrays(7);
    assert!(cache
        .session(&bridge, &engine, &digest(b"!"), p, e)
        .err()
        .unwrap()
        .contains("engine failure"));
    assert_eq!((get(1), get(3)), (0, 0));
    std::fs::write(&engine, b"one").unwrap();
    let (p, e) = arrays(7);
    let (mut final_session, hit) = cache
        .session(&bridge, &engine, &digest(b"one"), p, e)
        .unwrap();
    assert!(!hit);
    cache.clear();
    assert_eq!(get(1), 1);
    assert_eq!(final_session.prefill().unwrap()[2], 0.);
    drop(final_session);
    assert_eq!((get(1), get(2), get(3)), (0, 1, 0));
}
