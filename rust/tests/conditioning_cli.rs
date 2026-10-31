use serde_json::Value;
use std::{path::Path, process::Command};

#[test]
#[ignore = "set CHATTERBOX_CONDITIONING_ASSETS to the pinned asset manifest"]
fn fresh_text_cli_assembles_and_rejects_overflow_before_engine_loading() {
    let assets = std::env::var("CHATTERBOX_CONDITIONING_ASSETS").unwrap();
    let oracle: Value = serde_json::from_str(include_str!("conditioning-oracle.json")).unwrap();
    let sha = oracle["assets_sha256"].as_str().unwrap();
    let executable = env!("CARGO_BIN_EXE_chatterbox-rs");
    let result = Command::new(executable)
        .args(["prompt", &assets, sha, "ready, Holmes"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
    let case = oracle["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["input"] == "ready, Holmes")
        .unwrap();
    assert_eq!(receipt["prompt_sha256"], case["prompt_sha256"]);
    assert_eq!(receipt["shape"][1], case["frames"]);
    assert!(
        receipt["finished_unix_ms"].as_u64().unwrap()
            >= receipt["started_unix_ms"].as_u64().unwrap()
    );

    // Oversize text must stop before attempting to open the deliberately absent
    // engine/library. This exercises the fresh-text generation entry route.
    let oversized = "x ".repeat(1100);
    let result = Command::new(executable)
        .args([
            "text-tokens",
            "absent-bridge.so",
            "absent.engine",
            &"0".repeat(64),
            &assets,
            sha,
            &oversized,
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("512"), "{error}");
    assert!(!error.contains("absent"), "{error}");

    let result = Command::new(executable)
        .args(["prompt", &assets, &"0".repeat(64), "ready"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("checksum"));

    // Ensure runtime asset resolution stays anchored to this fixture's bundle.
    assert!(Path::new(&assets).is_file());
}
