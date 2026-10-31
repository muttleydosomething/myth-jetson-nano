//! Native Turbo text front end. Hugging Face's Rust core consumes an exported
//! tokenizer JSON; Python, Torch, network access and speech weights are absent.
//! The checksum admits the exact vocabulary, merge rules and added voice tags.
use crate::{text::punc_norm, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use tokenizers::Tokenizer;

/// Pinned by the deployed Turbo tokenizer_config.json, independently of the
/// TensorRT prompt profile (conditioning embeddings consume additional space).
pub const MAX_TEXT_TOKENS: usize = 1024;

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct TokenizedText {
    pub normalized: String,
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub untruncated_length: usize,
    pub truncated: bool,
}

pub struct TurboTokenizer {
    tokenizer: Tokenizer,
}

impl TurboTokenizer {
    pub fn open(path: &Path, expected_sha256: &str) -> Result<Self> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("Tokenizer {}: {e}", path.display()))?;
        Self::from_bytes(&bytes, expected_sha256)
    }

    fn from_bytes(bytes: &[u8], expected_sha256: &str) -> Result<Self> {
        if expected_sha256.len() != 64 || !expected_sha256.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("Tokenizer SHA256 must be 64 hexadecimal characters".into());
        }
        // Parse exactly the bytes hashed here; do not reopen a mutable asset.
        let actual = format!("{:x}", Sha256::digest(bytes));
        if actual != expected_sha256.to_ascii_lowercase() {
            return Err(format!(
                "Tokenizer checksum differs from admitted export: {actual}"
            ));
        }
        let mut tokenizer =
            Tokenizer::from_bytes(bytes).map_err(|e| format!("Tokenizer JSON: {e}"))?;
        if tokenizer.get_vocab_size(true) != 50276
            || tokenizer.token_to_id("<|endoftext|>") != Some(50256)
        {
            return Err("Tokenizer is not the Turbo 50276-token vocabulary".into());
        }
        // Transformers mutates its backend's padding/truncation settings during
        // calls. Reset any exported request state; this wrapper owns the policy.
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        Ok(Self { tokenizer })
    }

    /// Mirror generate(): punctuation first, special-tag recognition, then
    /// right truncation at 1024. A single request adds no padding or BOS/EOS.
    /// Expose truncation so future conditioning assembly can report it truthfully.
    pub fn encode(&self, text: &str) -> Result<TokenizedText> {
        let normalized = punc_norm(text);
        let encoded = self
            .tokenizer
            .encode(normalized.as_str(), true)
            .map_err(|e| format!("Tokenizer encode: {e}"))?;
        let untruncated_length = encoded.len();
        let length = untruncated_length.min(MAX_TEXT_TOKENS);
        Ok(TokenizedText {
            normalized,
            input_ids: encoded.get_ids()[..length].to_vec(),
            attention_mask: encoded.get_attention_mask()[..length].to_vec(),
            untruncated_length,
            truncated: untruncated_length > length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admitted_bytes_still_require_a_turbo_vocabulary() {
        let bytes = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","vocab":{"x":0},"merges":[]}}"#;
        let digest = format!("{:x}", Sha256::digest(bytes));
        let error = TurboTokenizer::from_bytes(bytes, &digest).err().unwrap();
        assert!(error.contains("50276-token"), "{error}");
    }

    #[test]
    fn malformed_checksum_is_rejected() {
        let error = TurboTokenizer::from_bytes(b"{}", "invalid").err().unwrap();
        assert!(error.contains("64 hexadecimal"), "{error}");
    }
}
