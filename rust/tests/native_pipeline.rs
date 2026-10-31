//! Composition contracts: token cleanup once, truthful stop, lossless WAV output.
use chatterbox_rs::{generation::Generated, npy::F32Npy, pipeline};
#[test]
fn token_stream_prefixes_keep_silence_only_at_finality() {
    let mut tokens = vec![12; 70];
    tokens.extend([4299; 3]);
    assert_eq!(
        pipeline::streaming_prefixes(&tokens).unwrap(),
        vec![(32, false), (64, false), (73, true)]
    );
    assert_eq!(
        pipeline::streaming_prefixes(&[12, 4299, 4299, 4299]).unwrap(),
        vec![(4, true)]
    );
    assert!(pipeline::streaming_prefixes(&[4299; 3]).is_err());
    assert!(pipeline::streaming_prefixes(&[12; 70]).is_err());
    tokens[0] = 6561;
    assert!(pipeline::streaming_prefixes(&tokens).is_err());
}
#[test]
fn native_pipeline_handoff_requires_eos_and_cleans_once() {
    assert_eq!(
        pipeline::decoder_tokens(&Generated {
            tokens: vec![12, 6561, 23],
            reached_eos: true
        })
        .unwrap(),
        vec![12, 23, 4299, 4299, 4299]
    );
    assert!(pipeline::decoder_tokens(&Generated {
        tokens: vec![12],
        reached_eos: false
    })
    .is_err());
    assert!(pipeline::decoder_tokens(&Generated {
        tokens: vec![],
        reached_eos: true
    })
    .is_err());
}
#[test]
fn native_pipeline_wav_preserves_float_samples_without_gain_or_clipping() {
    let input = F32Npy {
        shape: vec![1, 4],
        values: vec![0., -0.5, 1.25, -1.5],
    };
    let wav = pipeline::wav_bytes(&input).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(
        u32::from_le_bytes(wav[4..8].try_into().unwrap()) as usize,
        wav.len() - 8
    );
    assert_eq!(&wav[8..16], b"WAVEfmt ");
    assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 3); // IEEE FP32
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24000);
    assert_eq!(&wav[48..52], b"data");
    let samples: Vec<f32> = wav[56..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(samples, input.values);
    assert_eq!(wav, pipeline::wav_bytes(&input).unwrap());
    for bad in [
        F32Npy {
            shape: vec![1, 0],
            values: vec![],
        },
        F32Npy {
            shape: vec![2, 2],
            values: vec![0.; 4],
        },
        F32Npy {
            shape: vec![1, 2],
            values: vec![0.],
        },
        F32Npy {
            shape: vec![1, 1],
            values: vec![f32::NAN],
        },
    ] {
        assert!(pipeline::wav_bytes(&bad).is_err());
    }
}
