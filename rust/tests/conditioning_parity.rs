use chatterbox_rs::{
    conditioning::{assemble, project_reference, PromptAssets},
    npy::Npy,
};
use serde::Deserialize;
use std::path::Path;

fn array(shape: Vec<usize>, value: u16) -> Npy {
    let count = shape.iter().product();
    Npy {
        shape,
        halves: vec![value; count],
    }
}

/// Distinct rows expose a transposed projection, missing bias, and reordered
/// reference tokens. Accumulate before rounding to FP16, as the linear does.
#[test]
fn prepared_reference_projection_and_admission() {
    use half::f16;
    let one = f16::from_f32(1.0).to_bits();
    let mut speaker = array(vec![1, 256], 0);
    speaker.halves[0] = one;
    speaker.halves[1] = f16::from_f32(2.0).to_bits();
    let mut weight = array(vec![1024, 256], 0);
    weight.halves[0] = one;
    weight.halves[1] = one;
    weight.halves[256] = f16::from_f32(-1.0).to_bits();
    let bias = array(vec![1024], one);
    let mut speech = array(vec![6563, 1024], 0);
    speech.halves[1024..2048].fill(one);
    speech.halves[2048..3072].fill(f16::from_f32(2.0).to_bits());
    let prefix = project_reference(&speaker, &weight, &bias, &[2, 1, 2], &speech).unwrap();
    assert_eq!(prefix.shape, [1, 4, 1024]);
    assert_eq!(prefix.halves[0], f16::from_f32(4.0).to_bits());
    assert_eq!(prefix.halves[1], 0);
    assert_eq!(prefix.halves[2], one);
    assert_eq!(prefix.halves[1024], f16::from_f32(2.0).to_bits());
    assert_eq!(prefix.halves[2048], one);
    assert_eq!(
        project_reference(&speaker, &weight, &bias, &[], &speech)
            .unwrap()
            .shape,
        [1, 1, 1024]
    );
    assert_eq!(
        project_reference(&speaker, &weight, &bias, &vec![0; 375], &speech)
            .unwrap()
            .shape,
        [1, 376, 1024]
    );
    for ids in [vec![0; 376], vec![6561], vec![6562], vec![u32::MAX]] {
        assert!(project_reference(&speaker, &weight, &bias, &ids, &speech).is_err());
    }
    for bad in [
        array(vec![2, 256], 0),
        array(vec![1, 255], 0),
        array(vec![1, 256], 0x7c00),
        Npy {
            shape: vec![1, 256],
            halves: vec![],
        },
    ] {
        assert!(project_reference(&bad, &weight, &bias, &[0], &speech).is_err());
    }
    assert!(project_reference(&speaker, &array(vec![256, 1024], 0), &bias, &[0], &speech).is_err());
    assert!(project_reference(&speaker, &weight, &array(vec![1023], 0), &[0], &speech).is_err());
    assert!(project_reference(
        &speaker,
        &array(vec![1024, 256], 0x7bff),
        &bias,
        &[0],
        &speech
    )
    .is_err());
}

/// Different row markers expose token-order and BOS/prefix boundary mistakes.
#[test]
fn assembly_order_and_profile_admission() {
    let mut text = array(vec![50276, 1024], 0);
    text.halves[1024..2048].fill(1);
    text.halves[2048..3072].fill(2);
    let mut speech = array(vec![6563, 1024], 0);
    speech.halves[6561 * 1024..6562 * 1024].fill(3);
    let cond = array(vec![1, 2, 1024], 4);
    let prompt = assemble(&cond, &text, &speech, &[2, 1, 2]).unwrap();
    assert_eq!(prompt.shape, [1, 6, 1024]);
    assert_eq!(
        prompt
            .halves
            .chunks_exact(1024)
            .map(|r| r[0])
            .collect::<Vec<_>>(),
        [4, 4, 2, 1, 2, 3]
    );
    assert_eq!(
        assemble(&cond, &text, &speech, &vec![1; 509])
            .unwrap()
            .shape,
        [1, 512, 1024]
    );
    assert!(assemble(&cond, &text, &speech, &vec![1; 510])
        .unwrap_err()
        .contains("512"));
    assert!(assemble(&cond, &text, &speech, &[]).is_err());
    assert!(assemble(&cond, &text, &speech, &[50276]).is_err());
    assert!(assemble(&cond, &text, &speech, &[u32::MAX]).is_err());
}

#[test]
fn malformed_arrays_fail_instead_of_panicking() {
    let cond = array(vec![1, 1, 1024], 0);
    let text = array(vec![50276, 1024], 0);
    let speech = array(vec![6563, 1024], 0);
    let bad = [
        Npy {
            shape: vec![1, 1, 1024],
            halves: vec![],
        },
        array(vec![2, 1, 1024], 0),
        array(vec![1, 0, 1024], 0),
        array(vec![1, 512, 1024], 0),
        array(vec![1, 1, 1023], 0),
        array(vec![1, 1, 1024], 0x7c00),
    ];
    for c in bad {
        assert!(assemble(&c, &text, &speech, &[0]).is_err());
    }
    assert!(assemble(&cond, &array(vec![50275, 1024], 0), &speech, &[0]).is_err());
    assert!(assemble(&cond, &text, &array(vec![6562, 1024], 0), &[0]).is_err());
}

#[derive(Deserialize)]
struct Oracle {
    assets_sha256: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    input: String,
    input_ids: Vec<u32>,
    prompt_sha256: String,
    frames: usize,
    admitted: bool,
}

/// External weights remain outside Git; the oracle contains their hashes and
/// independently executed Python method results, including profile rejection.
#[test]
#[ignore = "set CHATTERBOX_CONDITIONING_ASSETS to the pinned asset manifest"]
fn deployed_prompt_assembly_parity() {
    let root = std::env::var("CHATTERBOX_CONDITIONING_ASSETS").expect("asset manifest");
    let oracle: Oracle = serde_json::from_str(include_str!("conditioning-oracle.json")).unwrap();
    let assets = PromptAssets::open(Path::new(&root), &oracle.assets_sha256).unwrap();
    assert!(oracle.cases.len() >= 25);
    for case in oracle.cases {
        let encoded = assets.tokenize(&case.input).unwrap();
        assert_eq!(encoded.input_ids, case.input_ids, "{:?}", case.input);
        let result = assets.assemble(&encoded);
        if case.admitted {
            let prompt = result.unwrap();
            assert_eq!(prompt.shape, [1, case.frames, 1024]);
            assert_eq!(
                chatterbox_rs::conditioning::prompt_digest(&prompt),
                case.prompt_sha256,
                "{:?}",
                case.input
            );
        } else {
            assert!(result.unwrap_err().contains("512"), "{:?}", case.input);
        }
    }
}

/// Independent deployed T3CondEnc.forward oracle, with retained and synthetic
/// tensor probes. Weights stay outside Git; pin the complete oracle receipt.
#[test]
#[ignore = "set CHATTERBOX_PREPARED_ASSETS to the portable export directory"]
fn deployed_prepared_reference_parity() {
    use sha2::{Digest, Sha256};
    use std::{fs, process::Command};
    let root = std::env::var("CHATTERBOX_PREPARED_ASSETS").unwrap();
    let root = Path::new(&root);
    let bytes = fs::read(root.join("prepared-oracle.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "dd60d65ba66e9bd735d94fcdd9609a10f927550142cb4d3c4c7164eca6964bcd"
    );
    let oracle: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(oracle["reference_preparations"], 0);
    assert_eq!(oracle["cuda_initialized"], false);
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for case in cases {
        let path = root.join(case["assets_file"].as_str().unwrap());
        let sha = case["assets_sha256"].as_str().unwrap();
        let assets = PromptAssets::open(&path, sha).unwrap();
        let text = assets.tokenize(case["input"].as_str().unwrap()).unwrap();
        let mut prompt = assets.assemble(&text).unwrap();
        let native_digest = chatterbox_rs::conditioning::prompt_digest(&prompt);
        // The real CLI uses this same manifest route. A second text request
        // reuses the admitted prefix through the public library interface.
        let second = assets.tokenize("The prepared voice is ready.").unwrap();
        let second_prompt = assets.assemble(&second).unwrap();
        let expected_path = root.join(case["expected"]["file"].as_str().unwrap());
        let expected_bytes = fs::read(expected_path).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&expected_bytes)),
            case["expected"]["sha256"].as_str().unwrap()
        );
        let expected = Npy::from_bytes(&expected_bytes).unwrap();
        assert_eq!(
            second_prompt.halves[..expected.halves.len()],
            prompt.halves[..expected.halves.len()]
        );
        // Torch's intermediate reduction and our wide sum may differ by one
        // final FP16 quantum. Use sign-aware ordering (including signed zero),
        // and allow that difference only in the synthetic speaker row.
        let ordered = |bits: u16| -> i32 {
            if bits & 0x8000 == 0 {
                0x8000 + bits as i32
            } else {
                0x8000 - (bits & 0x7fff) as i32
            }
        };
        let differences: Vec<_> = prompt.halves[..1024]
            .iter()
            .zip(&expected.halves[..1024])
            .map(|(&a, &b)| (ordered(a) - ordered(b)).unsigned_abs())
            .collect();
        let max_ulp = differences.iter().copied().max().unwrap();
        let differing = differences.iter().filter(|&&n| n != 0).count();
        assert!(max_ulp <= 1, "{}: {max_ulp} ULP", case["name"]);
        if case["name"] != "changed-speaker" {
            assert_eq!(
                native_digest,
                case["prompt_sha256"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
        assert_eq!(
            prompt.halves[1024..expected.halves.len()],
            expected.halves[1024..]
        );
        // Replace only the speaker row with the oracle's value. Matching the
        // full oracle digest then proves every other prompt byte exactly.
        prompt.halves[..1024].copy_from_slice(&expected.halves[..1024]);
        assert_eq!(
            chatterbox_rs::conditioning::prompt_digest(&prompt),
            case["prompt_sha256"].as_str().unwrap()
        );
        println!(
            "{}",
            serde_json::json!({"case": case["name"], "speaker_differing_values": differing, "speaker_max_fp16_ulp": max_ulp, "native_prompt_sha256": native_digest})
        );
        let result = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args([
                "prompt",
                path.to_str().unwrap(),
                sha,
                case["input"].as_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let receipt: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(receipt["prompt_sha256"], native_digest);
    }
}
