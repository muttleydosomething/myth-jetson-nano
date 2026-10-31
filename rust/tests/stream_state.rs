//! Request-state contracts independent of expensive model inference.
use chatterbox_rs::{
    hift::{HiFTOutput, WaveformOverlap},
    npy::F32Npy,
    sampling::RequestNoise,
};
fn tensor(shape: &[usize], values: Vec<f32>) -> F32Npy {
    F32Npy {
        shape: shape.to_vec(),
        values,
    }
}
fn output(frames: usize, value: f32) -> HiFTOutput {
    HiFTOutput {
        pitch: tensor(&[1, frames], vec![100.; frames]),
        source: tensor(
            &[1, 1, frames * 480],
            (0..frames * 480).map(|i| i as f32).collect(),
        ),
        waveform: tensor(&[1, frames * 480], vec![value; frames * 480]),
    }
}
fn prefix_eq(a: &F32Npy, b: &F32Npy) {
    let small = a.shape[2];
    let large = b.shape[2];
    for channel in 0..a.shape[1] {
        assert_eq!(
            &a.values[channel * small..(channel + 1) * small],
            &b.values[channel * large..channel * large + small]
        );
    }
}
#[test]
fn stream_state_noise_is_coordinate_stable_across_uneven_growth_and_request_drop() {
    let mut request = RequestNoise::new(71);
    let a = request.meanflow(3).unwrap();
    let b = request.meanflow(7).unwrap();
    let c = request.meanflow(12).unwrap();
    prefix_eq(&a, &b);
    prefix_eq(&b, &c);
    assert_eq!(c.values, request.meanflow(12).unwrap().values);
    let (h1, p1) = request.hift(3).unwrap();
    let (h2, p2) = request.hift(7).unwrap();
    prefix_eq(&h1, &h2);
    assert_eq!(p1.values, p2.values);
    assert!(request.meanflow(11).is_err());
    assert!(request.hift(6).is_err());
    assert!(request.meanflow(usize::MAX).is_err());
    assert_eq!(c.values, request.meanflow(12).unwrap().values);
    assert!(c.values.iter().all(|v| v.is_finite()));
    // Independent lanes must not be shifted copies of a single PRNG stream.
    assert_ne!(&c.values[2..12], &c.values[24..34]);
    drop(request); // Cancellation drops every channel and generator with the request.
    let mut fresh = RequestNoise::new(71);
    assert_eq!(fresh.meanflow(12).unwrap().values, c.values);
    assert_eq!(fresh.hift(7).unwrap().0.values, h2.values);
    assert_ne!(RequestNoise::new(72).meanflow(12).unwrap().values, c.values);
}
#[test]
fn stream_state_overlap_commits_once_flushes_and_preserves_source_coordinates() {
    let mut state = WaveformOverlap::new(960).unwrap();
    let a = state.push(output(4, 1.), false).unwrap();
    assert_eq!(a.offset_samples, 0);
    assert_eq!(a.pcm.values.len(), 960);
    assert!(a.pcm.values[..480].iter().all(|&v| v == 0.));
    let source = state.cached_source().values.clone();
    assert_eq!(source, output(4, 1.).source.values);
    let b = state.push(output(7, 2.), false).unwrap();
    assert_eq!(b.offset_samples, 960);
    assert_eq!(b.pcm.values.len(), 1440);
    assert_eq!(b.pcm.values[0], 1.);
    assert_eq!(b.pcm.values[959], 2.);
    assert!(b.pcm.values[..960].windows(2).all(|v| v[0] <= v[1]));
    assert!(b.pcm.values[960..].iter().all(|&v| v == 2.));
    let c = state.push(output(7, 3.), true).unwrap();
    assert_eq!(c.offset_samples, 2400);
    assert_eq!(c.pcm.values.len(), 960);
    assert_eq!(c.pcm.values[0], 2.);
    assert_eq!(c.pcm.values[959], 3.);
    assert_eq!(
        a.pcm.values.len() + b.pcm.values.len() + c.pcm.values.len(),
        7 * 480
    );
    assert!(state.push(output(8, 4.), true).is_err());
}
#[test]
fn stream_state_rejects_stale_malformed_and_short_updates_without_advancing() {
    assert!(WaveformOverlap::new(0).is_err());
    let mut state = WaveformOverlap::new(960).unwrap();
    let first = state.push(output(2, 1.), false).unwrap();
    assert!(first.pcm.values.is_empty());
    assert!(state.push(output(2, 1.), false).is_err());
    assert!(state.push(output(1, 1.), true).is_err());
    let mut bad = output(4, 2.);
    bad.source.values[0] = f32::NAN;
    assert!(state.push(bad, false).is_err());
    let mut bad = output(4, 2.);
    bad.waveform.values.pop();
    assert!(state.push(bad, false).is_err());
    let mut bad = output(4, 2.);
    bad.source.values[0] += 1.;
    assert!(state.push(bad, false).is_err());
    let good = state.push(output(4, 2.), true).unwrap();
    assert_eq!(good.offset_samples, 0);
    assert_eq!(good.pcm.values.len(), 1920);
    let mut fresh = WaveformOverlap::new(960).unwrap();
    let full = fresh.push(output(4, 2.), true).unwrap();
    assert_eq!(&good.pcm.values[960..], &full.pcm.values[960..]);
}

#[test]
#[ignore = "requires the independently admitted meanflow and HiFT checkpoint assets"]
fn stream_state_native_prefixes_cancel_and_replay() {
    use chatterbox_rs::{
        hift::S3GenHiFT, meanflow::S3GenMeanflow, pipeline::WaveformStream, s3gen::S3GenMelInput,
    };
    use std::path::Path;
    let meanroot = std::env::var("CHATTERBOX_MEANFLOW_ASSETS").unwrap();
    let hiftroot = std::env::var("CHATTERBOX_HIFT_CACHE_ASSETS").unwrap();
    let load = |root: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(Path::new(root).join("oracle.json")).unwrap())
            .unwrap()
    };
    let mean = S3GenMeanflow::open(
        &Path::new(&meanroot).join("meanflow.json"),
        load(&meanroot)["manifest_sha256"].as_str().unwrap(),
    )
    .unwrap();
    let hift = S3GenHiFT::open(
        &Path::new(&hiftroot).join("hift.json"),
        load(&hiftroot)["manifest_sha256"].as_str().unwrap(),
    )
    .unwrap();
    // Tiny synthetic conditions exercise real kernels without a new recording
    // or repeated expensive whole-reference inference. Numerical stage parity
    // remains covered by the independent captures for these exact checkpoints.
    let input = |generated| {
        let frames = 2 + generated;
        S3GenMelInput {
            mu: tensor(&[1, 80, frames], vec![0.01; 80 * frames]),
            mask: tensor(&[1, 1, frames], vec![1.; frames]),
            cond: tensor(&[1, 80, frames], vec![0.; 80 * frames]),
            speaker: tensor(&[1, 80], vec![0.01; 80]),
            reference_frames: 2,
            generated_frames: generated,
        }
    };
    let render = || {
        let mut state = WaveformStream::new(42, 960).unwrap();
        let mut output = Vec::new();
        for (frames, finality) in [(3, false), (6, false), (8, true)] {
            let part = state
                .decode_prefix(&input(frames), &mean, &hift, finality, || false)
                .unwrap();
            assert_eq!(part.offset_samples, output.len());
            assert!(part.pcm.values.iter().all(|v| v.is_finite()));
            output.extend(part.pcm.values);
        }
        assert_eq!(output.len(), 8 * 480);
        assert!(state
            .decode_prefix(&input(9), &mean, &hift, true, || false)
            .is_err());
        output
    };
    let first = render();
    let mut abandoned = WaveformStream::new(42, 960).unwrap();
    let mut checks = 0;
    assert!(abandoned
        .decode_prefix(&input(3), &mean, &hift, false, || {
            checks += 1;
            checks == 2
        })
        .is_err());
    assert!(abandoned
        .decode_prefix(&input(3), &mean, &hift, false, || false)
        .err()
        .unwrap()
        .contains("closed"));
    drop(abandoned);
    let mut invalid = WaveformStream::new(42, 960).unwrap();
    let mut bad = input(5);
    bad.mask.values.pop();
    assert!(invalid
        .decode_prefix(&bad, &mean, &hift, false, || false)
        .is_err());
    assert!(invalid
        .decode_prefix(&input(3), &mean, &hift, false, || false)
        .err()
        .unwrap()
        .contains("closed"));
    assert_eq!(first, render());
}
