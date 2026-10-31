//! Response encoding is distinct from reference preparation and lossless proof WAVs.
use chatterbox_rs::{
    audio_output::{self, Format},
    npy::F32Npy,
};
fn signal() -> F32Npy {
    F32Npy {
        shape: vec![1, 24000],
        values: (0..24000)
            .map(|i| 0.2 * (std::f32::consts::TAU * 440. * i as f32 / 24000.).sin())
            .collect(),
    }
}
#[test]
fn pcm16_response_matches_deployed_clipping_and_truncation() {
    let input = F32Npy {
        shape: vec![1, 7],
        values: vec![0., -0.5, 0.5, 1., -1., 1.25, -1.5],
    };
    let bytes = audio_output::encode(&input, Format::Wav, || false).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize,
        bytes.len() - 8
    );
    assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 24000);
    assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
    let decoded: Vec<_> = bytes[44..]
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        decoded,
        vec![0, -16383, 16383, 32767, -32767, 32767, -32767]
    );
}
#[test]
fn output_admission_and_cancellation_precede_codec_loading() {
    assert!(Format::parse("aac").is_err());
    for format in [Format::Wav, Format::Mp3, Format::Opus] {
        assert!(audio_output::encode(&signal(), format, || true)
            .unwrap_err()
            .contains("cancelled"));
        for bad in [
            F32Npy {
                shape: vec![1, 0],
                values: vec![],
            },
            F32Npy {
                shape: vec![2, 1],
                values: vec![0., 0.],
            },
            F32Npy {
                shape: vec![1, 2],
                values: vec![0.],
            },
            F32Npy {
                shape: vec![1, 1],
                values: vec![f32::NAN],
            },
            F32Npy {
                shape: vec![1, 1],
                values: vec![f32::INFINITY],
            },
        ] {
            assert!(audio_output::encode(&bad, format, || false).is_err());
        }
    }
}
#[test]
fn native_codecs_replay_and_cancel_between_chunks() {
    let input = signal();
    for format in [Format::Wav, Format::Mp3, Format::Opus] {
        let encoded = audio_output::encode(&input, format, || false).unwrap();
        assert!(!encoded.is_empty());
        let intervening = F32Npy {
            shape: vec![1, 720],
            values: vec![0.; 720],
        };
        audio_output::encode(&intervening, format, || false).unwrap();
        let replay = audio_output::encode(&input, format, || false).unwrap();
        // Ogg's stream serial is independently assigned by libsndfile.
        if format != Format::Opus {
            assert_eq!(encoded, replay);
        }
        let mut checkpoints = 0;
        assert!(audio_output::encode(&input, format, || {
            checkpoints += 1;
            checkpoints > 2
        })
        .unwrap_err()
        .contains("cancelled"));
        let recovered = audio_output::encode(&input, format, || false).unwrap();
        assert_eq!(encoded.len(), recovered.len());
        if let Ok(root) = std::env::var("CHATTERBOX_OUTPUT_PROOF_DIR") {
            std::fs::write(format!("{root}/tone.{}", format.name()), &encoded).unwrap();
        }
    }
}
