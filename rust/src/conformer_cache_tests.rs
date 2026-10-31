//! Fake-ABI ownership proof; arithmetic belongs to the deployed encoder oracle.
use super::*;

#[test]
fn conformer_graph_lifecycle() {
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::symlink;
    // Plugin code has process lifetime. Isolate this fake ABI instead of
    // weakening PROCESS_BRIDGE's immutable binding for parallel unit tests.
    if std::env::var_os("MYTH_CONFORMER_GRAPH_LIFECYCLE_CHILD").is_none() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "conformer::cache_tests::conformer_graph_lifecycle",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("MYTH_CONFORMER_GRAPH_LIFECYCLE_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "child stdout:\n{}\nchild stderr:\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        // A wrong module/filter must not produce a passing zero-test child.
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
        return;
    }
    let root = std::env::temp_dir().join(format!("conformer-cache-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("fake.c");
    let bridge = root.join("fake.so");
    std::fs::write(
        &source,
        r#"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
static int opens=0,live=0,calls=0;
int cb_conformer_metric(int key) {return key==0?opens:key==1?live:calls;}
void* cb_conformer_open(const char* path,char* err,size_t cap) {
    ++opens;++live;return malloc(1);
}
int cb_conformer_run(void* h,const float* input,size_t frames,float* features,
    uint64_t* counts,char* err,size_t cap) {
    ++calls;
    if(input[0]==99) {snprintf(err,cap,"injected Conformer failure");return -1;}
    for(size_t i=0;i<2*frames*512;++i) features[i]=input[(i/1024)*512+i%512]+1.0f;
    if(input[0]==98) features[0]=NAN;
    counts[0]=frames*512*4;counts[1]=2*frames*512*4;
    counts[2]=counts[0]+counts[1];return 0;
}
void cb_conformer_close(void* h) {--live;free(h);}
"#,
    )
    .unwrap();
    assert!(std::process::Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(&source)
        .arg("-o")
        .arg(&bridge)
        .status()
        .unwrap()
        .success());
    // SAFETY: this test compiled the ABI, and the library outlives metrics.
    let library = unsafe { libloading::Library::new(&bridge).unwrap() };
    let metric = unsafe {
        library
            .get::<unsafe extern "C" fn(i32) -> i32>(b"cb_conformer_metric\0")
            .unwrap()
    };
    let get = |key| unsafe { metric(key) };
    let digest = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let engine = root.join("fake.engine");
    std::fs::write(&engine, b"one").unwrap();
    let path = root.join("conformer.json");
    let manifest = serde_json::json!({"schema_version":1,"encoder_kind":"s3gen_conformer_trt_fp32",
        "source_sha256":"0".repeat(64),"max_frames":4,
        "engine":{"file":"fake.engine","sha256":digest(b"one")},
        "bridge":{"file":"fake.so","sha256":digest(&std::fs::read(&bridge).unwrap())}});
    let bytes = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let pin = digest(&bytes);
    let array = |frames: usize, mark: f32| F32Npy {
        shape: vec![1, frames, D],
        values: (0..frames * D).map(|i| mark + i as f32 / 1024.).collect(),
    };
    let mut cache = ConformerCache::default();
    assert!(cache.run(&array(1, 1.)).is_err());
    let mut first = None;
    for (frames, mark) in [(2, 1.), (4, 2.), (1, 3.), (3, 4.)] {
        assert!(cache.admit(&path, &pin).unwrap());
        let input = array(frames, mark);
        let (output, stats) = cache.run(&input).unwrap();
        assert_eq!(output.features.shape, [1, 2 * frames, D]);
        let expected: Vec<_> = input
            .values
            .chunks_exact(D)
            .flat_map(|row| row.iter().chain(row.iter()).map(|&v| v + 1.))
            .collect();
        assert_eq!(output.features.values, expected);
        assert_eq!(output.mask.shape, [1, 1, 2 * frames]);
        assert_eq!(output.mask.values, vec![1.; 2 * frames]);
        assert_eq!(stats.backend, "fp32-tensorrt");
        assert_eq!(stats.host_to_device_bytes, (frames * D * 4) as u64);
        assert_eq!(stats.device_to_host_bytes, (2 * frames * D * 4) as u64);
        assert_eq!((get(0), get(1)), (1, 1), "warm calls reuse one owner");
        if first.is_none() {
            first = Some(output.features.values);
        }
    }
    assert_eq!(
        cache.run(&array(2, 1.)).unwrap().0.features.values,
        first.unwrap()
    );
    // Shape, finite-input and profile rejection occur before native code.
    // Any failed request retires the mutable cached execution context.
    for malformed in 0..4 {
        cache.admit(&path, &pin).unwrap();
        let mut input = array(2, 1.);
        match malformed {
            0 => {
                input.values.pop();
            }
            1 => input.values[0] = f32::NAN,
            2 => input.shape[0] = 2,
            _ => input = array(5, 1.),
        }
        let calls = get(2);
        assert!(cache.run(&input).is_err());
        assert_eq!(get(2), calls, "invalid input crossed native ABI");
        assert_eq!(get(1), 0, "request failure must retire its context");
    }
    for (mark, message) in [(99., "injected Conformer failure"), (98., "Nonfinite")] {
        cache.admit(&path, &pin).unwrap();
        let calls = get(2);
        assert!(cache.run(&array(1, mark)).err().unwrap().contains(message));
        assert_eq!(get(2), calls + 1);
        assert_eq!(get(1), 0, "native/output failure must evict context");
    }
    cache.admit(&path, &pin).unwrap();
    let opens = get(0);
    let modified = engine.metadata().unwrap().modified().unwrap();
    std::fs::write(&engine, b"two").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&engine)
        .unwrap()
        .set_modified(modified)
        .unwrap();
    assert_eq!(engine.metadata().unwrap().len(), 3);
    assert_eq!(engine.metadata().unwrap().modified().unwrap(), modified);
    assert!(cache.admit(&path, &pin).unwrap_err().contains("checksum"));
    assert_eq!(
        (get(0), get(1)),
        (opens, 0),
        "failed admission cannot retain old owner"
    );
    std::fs::write(&engine, b"one").unwrap();
    cache.admit(&path, &pin).unwrap();
    assert_eq!((get(0), get(1)), (opens + 1, 1));
    // The former canonical target stays intact while the lookup route changes.
    let same = root.join("same.engine");
    let corrupt = root.join("corrupt.engine");
    std::fs::write(&same, b"one").unwrap();
    std::fs::write(&corrupt, b"two").unwrap();
    std::fs::remove_file(&engine).unwrap();
    symlink(&same, &engine).unwrap();
    cache.admit(&path, &pin).unwrap();
    let opens = get(0);
    assert_eq!(get(1), 1);
    std::fs::remove_file(&engine).unwrap();
    symlink(&corrupt, &engine).unwrap();
    assert!(cache.admit(&path, &pin).unwrap_err().contains("checksum"));
    assert_eq!((get(0), get(1)), (opens, 0));
    std::fs::remove_file(&engine).unwrap();
    symlink(&same, &engine).unwrap();
    cache.admit(&path, &pin).unwrap();
    assert!(cache.admit(&path, &"0".repeat(64)).is_err());
    assert_eq!(get(1), 0, "changed pin must evict the former owner");
    cache.admit(&path, &pin).unwrap();
    cache.clear();
    assert_eq!(get(1), 0, "unload releases context and weights");
    assert!(cache.run(&array(1, 1.)).is_err());
}
