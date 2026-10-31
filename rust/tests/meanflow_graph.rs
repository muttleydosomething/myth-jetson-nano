//! Native ABI ownership/validation checks without CUDA. Numerical correctness
//! and actual copy budgets are proved separately against Borg's deployed oracle.
use chatterbox_rs::{meanflow::S3GenMeanflow, npy::F32Npy, s3gen::S3GenMelInput};
use sha2::{Digest, Sha256};
use std::{path::Path, process::Command};
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn array(shape: Vec<usize>, value: f32) -> F32Npy {
    F32Npy {
        values: vec![value; shape.iter().product()],
        shape,
    }
}
fn input(frames: usize, mark: f32) -> S3GenMelInput {
    S3GenMelInput {
        mu: array(vec![1, 80, frames], mark),
        cond: array(vec![1, 80, frames], mark * 2.),
        mask: array(vec![1, 1, frames], 1.),
        speaker: array(vec![1, 80], mark * 3.),
        reference_frames: 1,
        generated_frames: frames - 1,
    }
}
#[test]
fn meanflow_graph_replay() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("meanflow-graph-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let source = root.join("graph.c");
    let bridge = root.join("graph.so");
    std::fs::write(
        &source,
        r#"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
static int live=0,calls=0;
int cb_mf_metric(int key) { return key?calls:live; }
void* cb_meanflow_open(const char* p,char* e,size_t n) { ++live;return malloc(1); }
void cb_meanflow_close(void* p) { --live;free(p); }
int cb_meanflow_run(void* h,const float* x,const float* mask,const float* mu,
    const float* spks,const float* cond,size_t f,int steps,float t,float r,
    float* out,uint64_t* bytes,char* e,size_t n) {
    ++calls;
    for(size_t c=0;c<80;++c)for(size_t j=0;j<f;++j) {
        size_t k=c*f+j;float dx=mu[k]+cond[k]+spks[c];
        out[k]=steps?x[k]+dx:dx;
    }
    bytes[0]=(241*f+80)*4+8*(steps?steps:1);bytes[1]=80*f*4;bytes[2]=401*f*4+328;
    return 0;
}
"#,
    )
    .unwrap();
    assert!(Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(&source)
        .arg("-o")
        .arg(&bridge)
        .status()
        .unwrap()
        .success());
    let lib = unsafe { libloading::Library::new(&bridge).unwrap() };
    let metric = unsafe {
        lib.get::<unsafe extern "C" fn(i32) -> i32>(b"cb_mf_metric\0")
            .unwrap()
    };
    let calls = || unsafe { metric(1) };
    std::fs::write(root.join("meanflow.engine"), b"pinned-graph").unwrap();
    let manifest = serde_json::json!({"schema_version":1,"decoder_kind":"s3gen_meanflow_trt_fp32",
        "source_sha256":"a".repeat(64),"max_frames":8,
        "engine":{"file":"meanflow.engine","sha256":sha(b"pinned-graph")},
        "bridge":{"file":"graph.so","sha256":sha(&std::fs::read(&bridge).unwrap())}});
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let path = root.join("meanflow.json");
    std::fs::write(&path, &bytes).unwrap();
    assert!(S3GenMeanflow::open(&path, &"0".repeat(64)).is_err());
    let model = S3GenMeanflow::open(&path, &sha(&bytes)).unwrap();
    assert_eq!(unsafe { metric(0) }, 1);
    let a = input(3, 1.);
    let noise = array(vec![1, 80, 3], 0.5);
    let (first, stats) = model.run(&a, &noise, 2).unwrap();
    assert_eq!(first.full.values, vec![6.5; 240]);
    assert_eq!(first.generated.shape, vec![1, 80, 2]);
    assert_eq!(stats.backend, "fp32-tensorrt");
    assert_eq!(stats.host_to_device_bytes, (241 * 3 + 80) * 4 + 2 * 8);
    let b = input(6, 7.);
    assert_eq!(
        model
            .decode(&b, &array(vec![1, 80, 6], 2.), 3)
            .unwrap()
            .full
            .values,
        vec![44.; 480]
    );
    let before = calls();
    let mut bad = input(3, 1.);
    bad.mask.values[1] = 0.;
    assert!(model.decode(&bad, &noise, 2).is_err());
    assert!(model.decode(&a, &noise, 0).is_err());
    assert!(model.estimate(&a, &noise, 0., f32::NAN).is_err());
    assert!(model
        .decode(&input(9, 1.), &array(vec![1, 80, 9], 1.), 2)
        .is_err());
    assert_eq!(calls(), before, "invalid inputs crossed native ABI");
    assert_eq!(
        model.decode(&a, &noise, 2).unwrap().full.values,
        first.full.values
    );
    assert_eq!(
        model.estimate(&a, &noise, 0.25, 0.5).unwrap().values,
        vec![6.; 240]
    );
    drop(model);
    assert_eq!(unsafe { metric(0) }, 0);
    // Pin changes are refused before opening another native owner.
    std::fs::write(root.join("meanflow.engine"), b"changed").unwrap();
    assert!(S3GenMeanflow::open(&path, &sha(&bytes))
        .err()
        .unwrap()
        .contains("checksum differs"));
    assert_eq!(unsafe { metric(0) }, 0);
}
