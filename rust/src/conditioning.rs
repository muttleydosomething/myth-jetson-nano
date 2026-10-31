//! Turbo prompt assembly at T3.prepare_input_embeds: prepared voice prefix,
//! text-table rows, then speech BOS. Load either an already projected prefix or
//! retained speaker/reference-token tensors and project once at admission.
//! Reuse this immutable state across requests; audio preparation is separate.
use crate::{
    npy::Npy,
    tokenizer::{TokenizedText, TurboTokenizer},
    Result,
};
use half::f16;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

const WIDTH: usize = 1024;
const BOS: usize = 6561;

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Asset {
    pub(crate) file: String,
    pub(crate) sha256: String,
}

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    voice_prefix_kind: String,
    voice_source_sha256: String,
    tokenizer: Asset,
    text: Asset,
    speech: Asset,
    conditioning: Option<Asset>,
    prepared: Option<PreparedReference>,
}

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct PreparedReference {
    source_sha256: String,
    speaker: Asset,
    projection_weight: Asset,
    projection_bias: Asset,
    speech_tokens: Asset,
}

/// One checksummed export binds vocabulary, embedding tables and prepared voice.
/// It loads local data only, never downloads or invokes a Python interpreter.
pub struct PromptAssets {
    tokenizer: TurboTokenizer,
    text: Npy,
    speech: Arc<Npy>,
    conditioning: Npy,
}

/// Keep the original lookup routes as well as canonical source identities.
/// Checking only an old canonical target would miss a retargeted symlink in
/// the manifest's route, its parent directory, or one of its leaf assets.
struct PromptSources {
    path: PathBuf,
    pin: String,
    sources: Vec<(PathBuf, crate::tensorrt::SourceIdentity)>,
}
impl PromptSources {
    fn read(path: &Path, pin: &str) -> Result<Self> {
        // Observe the manifest before reading its bytes, then observe every
        // dependency before PromptAssets admits and parses any large tensor.
        let source = crate::tensorrt::SourceIdentity::read(path)?;
        let manifest: Manifest = serde_json::from_slice(&admitted_bytes(path, pin)?)
            .map_err(|e| format!("Conditioning manifest: {e}"))?;
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut assets = vec![&manifest.tokenizer, &manifest.text, &manifest.speech];
        if let Some(asset) = &manifest.conditioning {
            assets.push(asset);
        }
        if let Some(reference) = &manifest.prepared {
            assets.extend([
                &reference.speaker,
                &reference.projection_weight,
                &reference.projection_bias,
                &reference.speech_tokens,
            ]);
        }
        let mut sources = vec![(path.to_path_buf(), source)];
        for asset in assets {
            let path = asset_path(root, asset)?;
            let source = crate::tensorrt::SourceIdentity::read(&path)?;
            sources.push((path, source));
        }
        Ok(Self {
            path: path.to_path_buf(),
            pin: pin.to_ascii_lowercase(),
            sources,
        })
    }
    fn matches(&self, path: &Path, pin: &str) -> Result<bool> {
        if self.path != path || !self.pin.eq_ignore_ascii_case(pin) {
            return Ok(false);
        }
        for (route, source) in &self.sources {
            if crate::tensorrt::SourceIdentity::read(route)? != *source {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// One service-owned immutable prompt, cleared with the existing EngineCache.
/// Only request text and assembled prompt bytes vary; speech-table sharing
/// avoids copying its thirteen-megabyte allocation into each token session.
#[derive(Default)]
pub(crate) struct PromptCache {
    retained: Option<(PromptSources, PromptAssets)>,
}
impl PromptCache {
    pub(crate) fn clear(&mut self) {
        self.retained = None;
    }
    pub(crate) fn prepare(
        &mut self,
        path: &Path,
        pin: &str,
        input: &str,
    ) -> Result<(TokenizedText, Npy, Arc<Npy>)> {
        let result = (|| {
            if !valid_sha(pin) {
                return Err("Asset SHA256 must be 64 hexadecimal characters".into());
            }
            let hit = match &self.retained {
                Some((sources, _)) => sources.matches(path, pin)?,
                None => false,
            };
            if !hit {
                // Evict first: failed replacement must not preserve the old
                // voice or overlap two large embedding-table allocations.
                self.clear();
                let sources = PromptSources::read(path, pin)?;
                let assets = PromptAssets::open(path, pin)?;
                if !sources.matches(path, pin)? {
                    return Err("Prompt source changed during admission".into());
                }
                self.retained = Some((sources, assets));
            }
            let _trace = crate::trace::span(if hit {
                "prompt_assets_reuse"
            } else {
                "prompt_assets_admitted"
            });
            let assets = &self.retained.as_ref().expect("Prompt assets admitted").1;
            let text = assets.tokenize(input)?;
            let prompt = assets.assemble(&text)?;
            Ok((text, prompt, Arc::clone(&assets.speech)))
        })();
        if result.is_err() {
            self.clear();
        }
        result
    }
}

pub(crate) fn valid_sha(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|c| c.is_ascii_hexdigit())
}
pub(crate) fn admitted_bytes(path: &Path, expected: &str) -> Result<Vec<u8>> {
    if !valid_sha(expected) {
        return Err("Asset SHA256 must be 64 hexadecimal characters".into());
    }
    let _trace = crate::trace::named("asset_admission", path.to_str().unwrap_or("non-UTF8"));
    let read_trace = crate::trace::span("asset_read");
    let bytes = fs::read(path).map_err(|e| format!("Asset {}: {e}", path.display()))?;
    drop(read_trace);
    let _hash_trace = crate::trace::span("asset_sha256");
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected.to_ascii_lowercase() {
        return Err(format!(
            "Asset checksum differs for {}: {actual}",
            path.display()
        ));
    }
    Ok(bytes)
}
pub(crate) fn asset_path(root: &Path, asset: &Asset) -> Result<std::path::PathBuf> {
    // Exports are self-contained. A manifest must not redirect asset resolution
    // to unrelated absolute/parent paths.
    let mut components = Path::new(&asset.file).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err("Asset must name one local filename".into());
    }
    Ok(root.join(&asset.file))
}
fn array(root: &Path, asset: &Asset) -> Result<Npy> {
    Npy::from_bytes(&admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?)
}
fn validate(array: &Npy, expected: &[usize], name: &str) -> Result<()> {
    if array.shape != expected || array.halves.len() != expected.iter().product::<usize>() {
        return Err(format!(
            "{name} shape/data length differs from {expected:?}"
        ));
    }
    Ok(())
}

/// T3.prepare_conditioning -> T3CondEnc.forward for Turbo: one linear speaker
/// projection followed by speech-table rows. No perceiver, emotion, positions,
/// audio encoder or waveform state participates in this model's prefix.
pub fn project_reference(
    speaker: &Npy,
    weight: &Npy,
    bias: &Npy,
    tokens: &[u32],
    speech: &Npy,
) -> Result<Npy> {
    validate(speaker, &[1, 256], "Prepared speaker")?;
    validate(weight, &[WIDTH, 256], "Speaker projection weight")?;
    validate(bias, &[WIDTH], "Speaker projection bias")?;
    validate(speech, &[6563, WIDTH], "Speech embedding")?;
    if tokens.len() > 375 || tokens.iter().any(|&t| t >= 6561) {
        return Err("Reference speech must contain 0..375 codec tokens in 0..6561".into());
    }
    for values in [&speaker.halves, &weight.halves, &bias.halves] {
        if values.iter().any(|&n| !f16::from_bits(n).is_finite()) {
            return Err("Nonfinite speaker projection input".into());
        }
    }
    let mut halves = Vec::with_capacity((1 + tokens.len()) * WIDTH);
    // FP16 operands are exact in f64. A wide accumulator avoids a half round
    // after every multiply/add; only the final linear output is rounded. This
    // tiny 1024x256 operation runs once per loaded voice, outside the decoder.
    for (row, &offset) in weight.halves.chunks_exact(256).zip(&bias.halves) {
        let mut sum = f16::from_bits(offset).to_f64();
        for (&w, &s) in row.iter().zip(&speaker.halves) {
            sum += f16::from_bits(w).to_f64() * f16::from_bits(s).to_f64();
        }
        let value = f16::from_f64(sum);
        if !value.is_finite() {
            return Err("Speaker projection overflows FP16".into());
        }
        halves.push(value.to_bits());
    }
    for &token in tokens {
        let row = speech.speech_row(token)?;
        if row.iter().any(|&n| !f16::from_bits(n).is_finite()) {
            return Err("Nonfinite reference speech embedding".into());
        }
        halves.extend_from_slice(row);
    }
    Ok(Npy {
        shape: vec![1, 1 + tokens.len(), WIDTH],
        halves,
    })
}

impl PromptAssets {
    /// Publish through the existing prepared_turbo format. Shared tables are
    /// copied from their admitted bytes, never linked to mutable source assets.
    /// The caller owns a private staging directory and the final atomic commit.
    pub(crate) fn export_reference(
        template: &Path,
        expected: &str,
        speaker: &crate::npy::F32Npy,
        tokens: &[u32],
        destination: &Path,
    ) -> Result<Asset> {
        use crate::voice_preparation::{float_array, publish_file};
        let mut manifest: Manifest = serde_json::from_slice(&admitted_bytes(template, expected)?)
            .map_err(|e| format!("Conditioning template: {e}"))?;
        if manifest.schema_version != 2
            || manifest.voice_prefix_kind != "prepared_turbo"
            || manifest.conditioning.is_some()
            || !valid_sha(&manifest.voice_source_sha256)
        {
            return Err("Publication requires the existing prepared_turbo model template".into());
        }
        if speaker.shape != [1, 256]
            || speaker.values.len() != 256
            || tokens.is_empty()
            || tokens.len() > 375
            || tokens.iter().any(|&t| t >= 6561)
        {
            return Err("Invalid native T3 publication tensors".into());
        }
        let root = template.parent().unwrap_or_else(|| Path::new("."));
        let copy = |asset: &Asset, name: &str| -> Result<Asset> {
            publish_file(
                destination,
                name,
                &admitted_bytes(&asset_path(root, asset)?, &asset.sha256)?,
            )
        };
        let reference = manifest
            .prepared
            .as_mut()
            .ok_or("Missing prepared model template")?;
        reference.projection_weight = copy(&reference.projection_weight, "projection-weight.npy")?;
        reference.projection_bias = copy(&reference.projection_bias, "projection-bias.npy")?;
        reference.speaker = publish_file(destination, "speaker.npy", &float_array(speaker)?)?;
        reference.speech_tokens = publish_file(
            destination,
            "t3-tokens.json",
            &serde_json::to_vec(tokens).map_err(|e| e.to_string())?,
        )?;
        // Bind provenance to these new tensors, rather than the template voice.
        reference.source_sha256 = format!(
            "{:x}",
            Sha256::digest(
                format!(
                    "{}{}",
                    reference.speaker.sha256, reference.speech_tokens.sha256
                )
                .as_bytes()
            )
        );
        manifest.tokenizer = copy(&manifest.tokenizer, "tokenizer.json")?;
        manifest.text = copy(&manifest.text, "text.npy")?;
        manifest.speech = copy(&manifest.speech, "speech.npy")?;
        let pin = publish_file(
            destination,
            "prompt.json",
            &serde_json::to_vec(&manifest).map_err(|e| e.to_string())?,
        )?;
        // Re-admit and project with the inference reader before publication.
        Self::open(&destination.join(&pin.file), &pin.sha256)?;
        Ok(pin)
    }

    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(&admitted_bytes(path, expected)?)
            .map_err(|e| format!("Conditioning manifest: {e}"))?;
        let projected = manifest.schema_version == 1
            && manifest.voice_prefix_kind == "projected_turbo"
            && manifest.conditioning.is_some()
            && manifest.prepared.is_none();
        let prepared = manifest.schema_version == 2
            && manifest.voice_prefix_kind == "prepared_turbo"
            && manifest.conditioning.is_none()
            && manifest.prepared.is_some();
        if !(projected || prepared) || !valid_sha(&manifest.voice_source_sha256) {
            return Err(
                "Conditioning manifest must select exactly one supported Turbo voice export".into(),
            );
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let tokenizer = TurboTokenizer::open(
            &asset_path(root, &manifest.tokenizer)?,
            &manifest.tokenizer.sha256,
        )?;
        let speech = array(root, &manifest.speech)?;
        let conditioning = if let Some(reference) = manifest.prepared {
            if !valid_sha(&reference.source_sha256) {
                return Err("Prepared reference source SHA256 is invalid".into());
            }
            let tokens: Vec<u32> = serde_json::from_slice(&admitted_bytes(
                &asset_path(root, &reference.speech_tokens)?,
                &reference.speech_tokens.sha256,
            )?)
            .map_err(|e| format!("Reference speech tokens: {e}"))?;
            project_reference(
                &array(root, &reference.speaker)?,
                &array(root, &reference.projection_weight)?,
                &array(root, &reference.projection_bias)?,
                &tokens,
                &speech,
            )?
        } else {
            array(
                root,
                &manifest
                    .conditioning
                    .ok_or("Missing projected voice prefix")?,
            )?
        };
        let assets = Self {
            tokenizer,
            text: array(root, &manifest.text)?,
            speech: Arc::new(speech),
            conditioning,
        };
        // Validate all boundaries once before requests reach TensorRT.
        assemble(&assets.conditioning, &assets.text, &assets.speech, &[0])?;
        Ok(assets)
    }
    pub fn tokenize(&self, input: &str) -> Result<TokenizedText> {
        self.tokenizer.encode(input)
    }
    /// Replace a voice's conditioning at preparation time using the original
    /// projection. Vocabulary and speech/text tables remain immutable.
    pub fn with_reference(
        mut self,
        speaker: &Npy,
        weight: &Npy,
        bias: &Npy,
        tokens: &[u32],
    ) -> Result<Self> {
        self.conditioning = project_reference(speaker, weight, bias, tokens, &self.speech)?;
        assemble(&self.conditioning, &self.text, &self.speech, &[0])?;
        Ok(self)
    }
    pub fn assemble(&self, text: &TokenizedText) -> Result<Npy> {
        // A tokenizer receipt may truthfully report truncation; a speech request
        // must never silently lose the end of its message.
        if text.truncated {
            return Err("Text was truncated and exceeds the TensorRT 512-frame profile".into());
        }
        assemble(
            &self.conditioning,
            &self.text,
            &self.speech,
            &text.input_ids,
        )
    }
    pub fn into_speech_embeddings(self) -> Npy {
        // CLI ownership remains a move. A caller explicitly sharing the table
        // can still request the original owned return type without changing it.
        Arc::try_unwrap(self.speech).unwrap_or_else(|speech| Npy {
            shape: speech.shape.clone(),
            halves: speech.halves.clone(),
        })
    }
}

/// Turbo uses no text/speech position embeddings and no CFG duplication. BOS is
/// a speech-table row, not a text special token. Profile admission includes every
/// voice-prefix frame; the tokenizer's independent 1024-token ceiling is too big.
pub fn assemble(conditioning: &Npy, text: &Npy, speech: &Npy, ids: &[u32]) -> Result<Npy> {
    let frames = conditioning.shape.get(1).copied().unwrap_or(0);
    if !(1..=376).contains(&frames) {
        return Err("Prepared Turbo voice prefix must contain 1..376 frames".into());
    }
    validate(conditioning, &[1, frames, WIDTH], "Prepared voice prefix")?;
    validate(text, &[50276, WIDTH], "Text embedding")?;
    validate(speech, &[6563, WIDTH], "Speech embedding")?;
    if ids.is_empty() {
        return Err("Text tokens must be nonempty".into());
    }
    let total = frames
        .checked_add(ids.len())
        .and_then(|n| n.checked_add(1))
        .ok_or("Prompt frame count overflow")?;
    if total > 512 {
        return Err(format!(
            "Prompt has {total} frames; TensorRT permits at most 512"
        ));
    }
    let mut halves = Vec::with_capacity(total * WIDTH);
    halves.extend_from_slice(&conditioning.halves);
    for &id in ids {
        if id >= 50276 {
            return Err(format!("Text token {id} outside Turbo vocabulary"));
        }
        let start = id as usize * WIDTH;
        halves.extend_from_slice(&text.halves[start..start + WIDTH]);
    }
    halves.extend_from_slice(&speech.halves[BOS * WIDTH..(BOS + 1) * WIDTH]);
    if halves.iter().any(|&n| !f16::from_bits(n).is_finite()) {
        return Err("Prompt contains a nonfinite embedding".into());
    }
    Ok(Npy {
        shape: vec![1, total, WIDTH],
        halves,
    })
}

/// Hash the actual FP16 bytes submitted to TensorRT, independent of NPY headers.
pub fn prompt_digest(prompt: &Npy) -> String {
    let mut hash = Sha256::new();
    for &value in &prompt.halves {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Source-invalidation fixtures deliberately contain tiny leaf bytes.
    /// Tensor admission has independent numerical tests; these regressions
    /// exercise every route without allocating full embedding tables.
    struct SourceFixture {
        root: PathBuf,
        path: PathBuf,
        pin: String,
    }
    impl SourceFixture {
        fn new(prepared: bool) -> Self {
            static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "prompt-sources-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            let leaf = |name: &str| {
                fs::write(root.join(name), b"leaf").unwrap();
                serde_json::json!({"file":name,"sha256":format!("{:x}",Sha256::digest(b"leaf"))})
            };
            let mut manifest = serde_json::json!({
                "schema_version": if prepared { 2 } else { 1 },
                "voice_prefix_kind": if prepared { "prepared_turbo" } else { "projected_turbo" },
                "voice_source_sha256":"a".repeat(64),
                "tokenizer":leaf("tokenizer.json"),
                "text":leaf("text.npy"),
                "speech":leaf("speech.npy")
            });
            if prepared {
                manifest["prepared"] = serde_json::json!({
                    "source_sha256":"b".repeat(64),
                    "speaker":leaf("speaker.npy"),
                    "projection_weight":leaf("weight.npy"),
                    "projection_bias":leaf("bias.npy"),
                    "speech_tokens":leaf("tokens.json")
                });
            } else {
                manifest["conditioning"] = leaf("conditioning.npy");
            }
            let bytes = serde_json::to_vec(&manifest).unwrap();
            let path = root.join("prompt.json");
            fs::write(&path, &bytes).unwrap();
            Self {
                root,
                path,
                pin: format!("{:x}", Sha256::digest(bytes)),
            }
        }
    }
    impl Drop for SourceFixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }

    #[test]
    fn prompt_sources_detect_every_leaf_rewrite_even_with_restored_mtime() {
        for prepared in [false, true] {
            let fixture = SourceFixture::new(prepared);
            let sources = PromptSources::read(&fixture.path, &fixture.pin).unwrap();
            assert!(sources.matches(&fixture.path, &fixture.pin).unwrap());
            assert!(sources
                .matches(&fixture.path, &fixture.pin.to_ascii_uppercase())
                .unwrap());
            assert!(!sources.matches(&fixture.path, &"0".repeat(64)).unwrap());
            assert_eq!(sources.sources.len(), if prepared { 8 } else { 5 });
            for (route, _) in &sources.sources[1..] {
                let before = PromptSources::read(&fixture.path, &fixture.pin).unwrap();
                let original = fs::metadata(route).unwrap().modified().unwrap();
                fs::write(route, b"evil").unwrap();
                fs::OpenOptions::new()
                    .write(true)
                    .open(route)
                    .unwrap()
                    .set_times(fs::FileTimes::new().set_modified(original))
                    .unwrap();
                assert!(
                    !before.matches(&fixture.path, &fixture.pin).unwrap(),
                    "rewrite concealed in {}",
                    route.display()
                );
                fs::write(route, b"leaf").unwrap();
            }
            let before = PromptSources::read(&fixture.path, &fixture.pin).unwrap();
            fs::write(&fixture.path, b"{}").unwrap();
            assert!(!before.matches(&fixture.path, &fixture.pin).unwrap());
        }
    }

    #[test]
    fn prompt_sources_resolve_original_alias_routes_again() {
        use std::os::unix::fs::symlink;
        let fixture = SourceFixture::new(false);
        let second = SourceFixture::new(false);
        let alias = fixture.root.join("alias");
        symlink(&fixture.root, &alias).unwrap();
        let path = alias.join("prompt.json");
        let sources = PromptSources::read(&path, &fixture.pin).unwrap();
        assert!(sources.matches(&path, &fixture.pin).unwrap());
        // A different spelling of the bundle is admitted afresh, because its
        // relative leaf routes need not resolve through the same directory.
        assert!(!sources.matches(&fixture.path, &fixture.pin).unwrap());
        fs::remove_file(&alias).unwrap();
        symlink(&second.root, &alias).unwrap();
        assert!(!sources.matches(&path, &fixture.pin).unwrap());

        let leaf = fixture.root.join("text.npy");
        fs::remove_file(&leaf).unwrap();
        symlink(second.root.join("text.npy"), &leaf).unwrap();
        let sources = PromptSources::read(&fixture.path, &fixture.pin).unwrap();
        assert!(sources.matches(&fixture.path, &fixture.pin).unwrap());
        fs::remove_file(&leaf).unwrap();
        symlink(fixture.root.join("speech.npy"), &leaf).unwrap();
        assert!(!sources.matches(&fixture.path, &fixture.pin).unwrap());
    }

    #[test]
    fn prompt_cache_failed_cold_admission_has_no_retained_owner() {
        let fixture = SourceFixture::new(false);
        let mut cache = PromptCache::default();
        assert!(cache
            .prepare(&fixture.path, "invalid", "Ready.")
            .err()
            .unwrap()
            .contains("64 hexadecimal"));
        assert!(cache.retained.is_none());
        // The manifest and leaves have genuine checksums, but the tokenizer
        // bytes are not a model. No partially admitted prompt may survive.
        assert!(cache
            .prepare(&fixture.path, &fixture.pin, "Ready.")
            .is_err());
        assert!(cache.retained.is_none());
        cache.clear();
        assert!(cache.retained.is_none());
    }
    #[test]
    fn manifest_asset_paths_stay_with_the_export() {
        for file in ["/foreign.npy", "../foreign.npy", "nested/foreign.npy", ""] {
            assert!(asset_path(
                Path::new("."),
                &Asset {
                    file: file.into(),
                    sha256: "0".repeat(64)
                }
            )
            .is_err());
        }
    }
    #[test]
    fn checksums_reject_unadmitted_bytes() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conditioning-oracle.json");
        assert!(admitted_bytes(&path, "invalid")
            .unwrap_err()
            .contains("64 hexadecimal"));
        assert!(admitted_bytes(&path, &"0".repeat(64))
            .unwrap_err()
            .contains("checksum"));
    }
}
