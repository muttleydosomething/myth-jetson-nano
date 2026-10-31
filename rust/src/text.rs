// Copyright (c) 2025 Resemble AI
// Copyright (c) 2026 Hailstorm contributors
// SPDX-License-Identifier: MIT
pub const S3GEN_SR: u32 = 24_000;
pub const S3GEN_SIL: i64 = 4299;
pub const SPEECH_TOKEN_LIMIT: i64 = 6561;
fn python_whitespace(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '\u{001c}'..='\u{001f}')
}
/// Preserve deployed ordering: casing precedes whitespace collapse.
pub fn punc_norm(text: &str) -> String {
    let Some(first) = text.chars().next() else {
        return "You need to add some text for me to talk.".to_owned();
    };
    // Python upper() can expand one Unicode character to several, e.g. ß -> SS.
    let capitalized = if first.is_lowercase() {
        first.to_uppercase().collect::<String>() + &text[first.len_utf8()..]
    } else {
        text.to_owned()
    };
    let mut normalized = capitalized
        .split(python_whitespace)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    // Replacement order is significant for spaces introduced by ellipsis.
    for (old, new) in [
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ] {
        normalized = normalized.replace(old, new);
    }
    normalized.truncate(normalized.trim_end_matches(' ').len());
    if !normalized.ends_with(['.', '!', '?', '-', ',']) {
        normalized.push('.');
    }
    normalized
}
/// Mirror the Python cutoff literally, preserving order and input ownership.
pub fn finish_speech_tokens(tokens: &[i64]) -> Vec<i64> {
    let mut result: Vec<_> = tokens
        .iter()
        .copied()
        .filter(|&t| t < SPEECH_TOKEN_LIMIT)
        .collect();
    result.extend([S3GEN_SIL; 3]);
    result
}
