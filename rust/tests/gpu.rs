//! Real CUDA layout/replay proof. Normal developer tests need no CUDA installation.
use chatterbox_rs::gpu;
#[test]
#[ignore = "requires Borg CUDA and cuBLAS"]
fn cuda_product_layout_replay() {
    let run = || {
        gpu::run("cuda", || {
            let a = [1., 2., 3., 4., 5., 6.];
            let b = [7., 8., 9., -1., 2., -3., 4., 0., 1., 1., 1., 1.];
            let c = gpu::product_for_proof(&a, &b, 2, 3, 4)?;
            assert_eq!(c, [50., -6., 7., 6., 122., -12., 22., 15.]);
            // Intervening shape and changed values deliberately reuse device buffers.
            assert_eq!(
                gpu::product_for_proof(&[2., -3.], &[4., 5.], 1, 2, 1)?,
                [-7.]
            );
            assert_eq!(gpu::product_for_proof(&a, &b, 2, 3, 4)?, c);
            Ok(c)
        })
        .unwrap()
    };
    let (first, stats) = run();
    assert_eq!(stats.backend, "fp32-cuda-cublas");
    assert_eq!(stats.gemm_calls, 3);
    assert!(stats.peak_scratch_bytes > 0);
    let (second, _) = run(); // context teardown/re-entry, no stale request state
    assert_eq!(first, second);
    eprintln!("{}", serde_json::to_string(&stats).unwrap());
}
