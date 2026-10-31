use chatterbox_rs::tokenizer::TurboTokenizer;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Oracle {
    tokenizer_sha256: String,
    model_max_length: usize,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    input: String,
    normalized: String,
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    untruncated_length: usize,
    truncated: bool,
}

/// Requires only the exported tokenizer JSON, never speech weights or a GPU.
#[test]
#[ignore = "set CHATTERBOX_TOKENIZER_JSON to the pinned tokenizer export"]
fn deployed_turbo_tokenizer_parity() {
    let oracle: Oracle = serde_json::from_str(include_str!("tokenizer-oracle.json")).unwrap();
    let asset = std::env::var("CHATTERBOX_TOKENIZER_JSON").expect("tokenizer export path");
    let tokenizer = TurboTokenizer::open(Path::new(&asset), &oracle.tokenizer_sha256).unwrap();
    assert_eq!(oracle.model_max_length, 1024);
    assert!(oracle.cases.len() >= 70);
    for case in oracle.cases {
        let actual = tokenizer.encode(&case.input).unwrap();
        assert_eq!(actual.normalized, case.normalized, "{:?}", case.input);
        assert_eq!(actual.input_ids, case.input_ids, "{:?}", case.input);
        assert_eq!(
            actual.attention_mask, case.attention_mask,
            "{:?}",
            case.input
        );
        assert_eq!(actual.untruncated_length, case.untruncated_length);
        assert_eq!(actual.truncated, case.truncated);
    }
    // Reusing one immutable tokenizer must not inherit a preceding long request.
    assert_eq!(
        tokenizer.encode("ready").unwrap(),
        tokenizer.encode("ready").unwrap()
    );
}

#[test]
fn tokenizer_checksum_is_checked_before_parsing() {
    // This existing fixture is deliberately not tokenizer JSON.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/punctuation-oracle.json");
    let error = TurboTokenizer::open(&path, &"0".repeat(64)).err().unwrap();
    assert!(error.contains("checksum"), "{error}");
}
