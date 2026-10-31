//! Publication must round-trip the existing inference formats, with no overwrite.
use chatterbox_rs::{npy::F32Npy, s3gen::S3GenReference, voice_preparation::PreparedVoice};
use std::{
    path::Path,
    sync::{Arc, Barrier},
};
fn voice() -> PreparedVoice {
    PreparedVoice {
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
    }
}
#[test]
#[ignore = "requires existing pinned prompt tables"]
fn voice_publication() {
    use chatterbox_rs::conditioning::{prompt_digest, PromptAssets};
    use sha2::{Digest, Sha256};
    let root = std::env::var("CHATTERBOX_VOICE_PROMPT_ASSETS").unwrap();
    let template = Path::new(&root).join("retained-assets.json");
    let sha = format!("{:x}", Sha256::digest(std::fs::read(&template).unwrap()));
    let dir = std::env::temp_dir().join(format!("voice-publication-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let target = dir.join("voice");
    let first = voice().publish(&template, &sha, &target).unwrap();
    let reference = S3GenReference::open(&first.reference.path, &first.reference.sha256).unwrap();
    assert_eq!(
        serde_json::to_value(reference.receipt()).unwrap(),
        serde_json::to_value(voice().reference.receipt()).unwrap()
    );
    let assets = PromptAssets::open(&first.prompt.path, &first.prompt.sha256).unwrap();
    let text = assets.tokenize("Ready, Holmes.").unwrap();
    let prompt = assets.assemble(&text).unwrap();
    assert_eq!(prompt.shape[1], 1 + 2 + text.input_ids.len() + 1);
    let digest = prompt_digest(&prompt);
    assert!(voice().publish(&template, &sha, &target).is_err());
    assert_eq!(
        prompt_digest(
            &PromptAssets::open(&first.prompt.path, &first.prompt.sha256)
                .unwrap()
                .assemble(&text)
                .unwrap()
        ),
        digest
    );
    // Concurrent publishers use different staging directories. Only one may
    // commit; even an existing empty directory must survive untouched.
    let race = Arc::new(dir.join("race"));
    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let (race, barrier, template, sha) =
                (race.clone(), barrier.clone(), template.clone(), sha.clone());
            std::thread::spawn(move || {
                barrier.wait();
                voice().publish(&template, &sha, &race).is_ok()
            })
        })
        .collect();
    assert_eq!(
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|&ok| ok)
            .count(),
        1
    );
    let empty = dir.join("empty");
    std::fs::create_dir(&empty).unwrap();
    assert!(voice().publish(&template, &sha, &empty).is_err());
    assert_eq!(std::fs::read_dir(&empty).unwrap().count(), 0);
    let bad = dir.join("bad");
    assert!(voice().publish(&template, &"0".repeat(64), &bad).is_err());
    assert!(!bad.exists());
    let mut malformed = voice();
    malformed.t3_speaker.values.pop();
    assert!(malformed
        .publish(&template, &sha, &dir.join("malformed"))
        .is_err());
    assert!(!dir.join("malformed").exists());
    let mut overflow = voice();
    overflow.t3_speaker.values[0] = f32::MAX;
    assert!(overflow
        .publish(&template, &sha, &dir.join("overflow"))
        .is_err());
    assert!(!dir.join("overflow").exists());
    let alternate = voice()
        .publish(&template, &sha, &dir.join("replay"))
        .unwrap();
    assert_eq!(first.prompt.sha256, alternate.prompt.sha256);
    assert_eq!(first.reference.sha256, alternate.reference.sha256);
    // Reopening output needs published assets only, never an encoder or audio.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&first.prompt.path).unwrap()).unwrap();
    assert_eq!(manifest["prepared"]["speaker"]["file"], "speaker.npy");
    assert_eq!(
        manifest["prepared"]["speech_tokens"]["file"],
        "t3-tokens.json"
    );
}
