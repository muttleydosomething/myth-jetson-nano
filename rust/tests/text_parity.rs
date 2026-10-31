use chatterbox_rs::text::{finish_speech_tokens, punc_norm};
#[test]
fn deployed_python_punctuation_oracle() {
    // AST-extracted punc_norm from the pinned deployed tts_turbo.py; the oracle
    // executed without importing Chatterbox or loading its model.
    let cases: Vec<(String, String)> =
        serde_json::from_str(include_str!("punctuation-oracle.json")).unwrap();
    assert_eq!(cases.len(), 41);
    for (input, expected) in cases {
        assert_eq!(punc_norm(&input), expected, "input={input:?}");
    }
}
#[test]
fn token_cutoff_silence_and_reference_input_remain_unchanged() {
    let input = [0, 6560, 6561, 6562, 4299, 10, 10, -1, i64::MAX];
    let expected = [0, 6560, 4299, 10, 10, -1, 4299, 4299, 4299];
    assert_eq!(finish_speech_tokens(&input), expected);
    assert_eq!(finish_speech_tokens(&input), expected);
    assert_eq!(input, [0, 6560, 6561, 6562, 4299, 10, 10, -1, i64::MAX]);
    assert_eq!(finish_speech_tokens(&[]), [4299; 3]);
}
