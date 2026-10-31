//! TensorRT plugins register process-lifetime callbacks. Closing a request must
//! release its session, while the bridge code remains loaded for later requests.
use chatterbox_rs::{generation::Decoder, npy::Npy, tensorrt::TensorRt};
use sha2::{Digest, Sha256};
use std::{path::Path, process::Command};
fn session(bridge: &Path, engine: &Path, digest: &str) -> Result<TensorRt, String> {
    TensorRt::open(
        bridge,
        engine,
        digest,
        Npy {
            shape: vec![1, 1, 1024],
            halves: vec![0; 1024],
        },
        Npy {
            shape: vec![6563, 1024],
            halves: vec![0; 6563 * 1024],
        },
    )
}
#[test]
fn bridge_code_outlives_request_handles_and_rejects_rebinding() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("bridge-lifetime-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("bridge.c");
    let library = root.join("bridge.so");
    let marker = root.join("unloaded");
    std::fs::write(&source, r#"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
static int opens = 0;
void* cb_open(const char* path, char* error, size_t cap) { (void)path; (void)error; (void)cap; ++opens; return malloc(1); }
int cb_step(void* p, const unsigned short* in, size_t f, float* out, size_t n, char* e, size_t cap) {
    (void)p; (void)in; (void)f; (void)e; (void)cap;
    for (size_t i=0;i<n;++i) out[i]=0;
    out[0]=(float)opens; return 6563;
}
void cb_reset(void* p) { (void)p; }
void cb_close(void* p) { free(p); }
__attribute__((destructor)) static void unloaded(void) { FILE* f=fopen(MARKER,"w"); if(f) fclose(f); }
"#).unwrap();
    assert!(Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(&source)
        .arg(format!("-DMARKER=\"{}\"", marker.display()))
        .arg("-o")
        .arg(&library)
        .status()
        .unwrap()
        .success());
    let engine = root.join("engine");
    std::fs::write(&engine, b"fixture").unwrap();
    let digest = format!("{:x}", Sha256::digest(b"fixture"));
    {
        let mut first = session(&library, &engine, &digest).unwrap();
        assert_eq!(first.prefill().unwrap()[0], 1.);
    }
    assert!(
        !marker.exists(),
        "request teardown unloaded registered native code"
    );
    {
        let mut second = session(&library, &engine, &digest).unwrap();
        assert_eq!(second.prefill().unwrap()[0], 2.);
    }
    assert!(!marker.exists());
    let other = root.join("other.so");
    std::fs::copy(&library, &other).unwrap();
    let error = session(&other, &engine, &digest).err().unwrap();
    assert!(error.contains("process bridge"), "{error}");
}
