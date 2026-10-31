//! Join the existing audio and three neural branches into complete Turbo Conds.
//! This request orchestrator is separate from audio analysis and synthesis: no
//! reference is prepared during ordinary retained-voice speech requests.
use crate::{
    audio_preparation::AudioPreparation,
    campplus::CampPlus,
    conditioning::{admitted_bytes, valid_sha, Asset, PromptAssets},
    npy::{F32Npy, Npy},
    s3_tokenizer::S3Tokenizer,
    s3gen::{ReferenceReceipt, S3GenReference, S3ReferenceAnalysis},
    voice_encoder::T3VoiceEncoder,
    Result,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

/// One create-only output primitive for both established manifest formats.
/// Every file is flushed before the directory becomes visible to consumers.
pub(crate) fn publish_file(root: &Path, name: &str, bytes: &[u8]) -> Result<Asset> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))
        .map_err(|e| format!("Voice output {name}: {e}"))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("Voice output {name}: {e}"))?;
    Ok(Asset {
        file: name.into(),
        sha256: format!("{:x}", Sha256::digest(bytes)),
    })
}

/// NumPy v1 envelope for the small conditioning arrays, retaining every FP32
/// bit. Large immutable model tables are copied directly from admitted bytes.
pub(crate) fn float_array(array: &F32Npy) -> Result<Vec<u8>> {
    if array.shape.is_empty()
        || array
            .shape
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            != Some(array.values.len())
        || array.values.is_empty()
        || array.values.iter().any(|v| !v.is_finite())
    {
        return Err("Publication requires complete finite FP32 tensors".into());
    }
    let shape = array
        .shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape},), }}");
    let pad = (64 - (10 + header.len() + 1) % 64) % 64;
    header.extend(std::iter::repeat_n(' ', pad));
    header.push('\n');
    let length = u16::try_from(header.len()).map_err(|_| "NumPy header too large")?;
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(array.values.iter().flat_map(|v| v.to_le_bytes()));
    Ok(bytes)
}

#[derive(Serialize)]
pub struct PublishedAsset {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Serialize)]
pub struct PublishedVoice {
    pub prompt: PublishedAsset,
    pub reference: PublishedAsset,
}

/// Linux no-replace directory rename is the single visibility boundary. Plain
/// rename may replace an existing empty directory, so it cannot meet this gate.
#[cfg(target_os = "linux")]
fn commit_directory(staging: &Path, destination: &Path) -> Result<()> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let source = CString::new(staging.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
    let target = CString::new(destination.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            target.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result != 0 {
        return Err(format!("Voice commit: {}", std::io::Error::last_os_error()));
    }
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn commit_directory(_: &Path, _: &Path) -> Result<()> {
    Err("Atomic voice publication requires Linux renameat2".into())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pinned {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    preparation_kind: String,
    voice_encoder: Pinned,
    codec: Pinned,
    campplus: Pinned,
    analysis: Pinned,
}

/// Immutable preparation configuration; large neural models load sequentially.
/// Keeping all three encoders resident would unnecessarily increase peak RAM.
pub struct VoicePreparation {
    manifest: Manifest,
    audio: AudioPreparation,
    analysis: S3ReferenceAnalysis,
}
pub struct PreparedVoice {
    /// Deployed Conds retain the speaker in FP32. Only T3 projection casts FP16.
    pub t3_speaker: F32Npy,
    pub t3_tokens: Vec<u32>,
    pub reference: S3GenReference,
}
#[derive(Serialize)]
pub struct VoiceReceipt<'a> {
    pub t3_speaker: &'a [f32],
    pub t3_tokens: &'a [u32],
    pub prompt_tokens: &'a [i64],
    pub prompt_feat_shape: &'a [usize],
    pub prompt_feat: &'a [f32],
    pub embedding: &'a [f32],
    pub reference: ReferenceReceipt,
}
impl PreparedVoice {
    /// Publish a newly encoded voice once, without altering retained voices or
    /// re-running encoders. Consumers pin these existing inference manifests.
    pub fn publish(
        &self,
        prompt_template: &Path,
        expected: &str,
        destination: &Path,
    ) -> Result<PublishedVoice> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = destination
            .file_name()
            .ok_or("Voice destination requires a name")?;
        let parent = std::fs::canonicalize(
            destination
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new(".")),
        )
        .map_err(|e| format!("Voice parent: {e}"))?;
        let destination = parent.join(name);
        if destination.symlink_metadata().is_ok() {
            return Err("Voice destination already exists".into());
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let staging = parent.join(format!(
            ".voice-staging-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&staging).map_err(|e| format!("Voice staging: {e}"))?;
        let result = (|| {
            let prompt = PromptAssets::export_reference(
                prompt_template,
                expected,
                &self.t3_speaker,
                &self.t3_tokens,
                &staging,
            )?;
            let reference = self.reference.export(&staging)?;
            std::fs::File::open(&staging)
                .and_then(|f| f.sync_all())
                .map_err(|e| e.to_string())?;
            commit_directory(&staging, &destination)?;
            std::fs::File::open(&parent)
                .and_then(|f| f.sync_all())
                .map_err(|e| {
                    format!(
                        "Voice visible at {}; parent flush failed: {e}",
                        destination.display()
                    )
                })?;
            let pin = |asset: Asset| PublishedAsset {
                path: destination.join(asset.file),
                sha256: asset.sha256,
            };
            Ok(PublishedVoice {
                prompt: pin(prompt),
                reference: pin(reference),
            })
        })();
        // Failed staging remains recoverable. Never delete an operator's data.
        result.map_err(|e: String| format!("{e}; publication staging: {}", staging.display()))
    }

    pub fn receipt(&self) -> VoiceReceipt<'_> {
        VoiceReceipt {
            t3_speaker: &self.t3_speaker.values,
            t3_tokens: &self.t3_tokens,
            prompt_tokens: self.reference.prompt_tokens(),
            prompt_feat_shape: &self.reference.prompt_feat().shape,
            prompt_feat: &self.reference.prompt_feat().values,
            embedding: &self.reference.embedding().values,
            reference: self.reference.receipt(),
        }
    }
    /// Reuse the established projection and prompt assembly without preparing
    /// audio again. The caller supplies admitted, model-compatible FP16 tables.
    pub fn condition_prompt(
        &self,
        prompt: PromptAssets,
        weight: &Npy,
        bias: &Npy,
    ) -> Result<PromptAssets> {
        if self.t3_speaker.shape != [1, 256] || self.t3_speaker.values.len() != 256 {
            return Err("Native T3 speaker must have complete shape [1,256]".into());
        }
        let halves = self
            .t3_speaker
            .values
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        prompt.with_reference(
            &Npy {
                shape: vec![1, 256],
                halves,
            },
            weight,
            bias,
            &self.t3_tokens,
        )
    }
}
fn pcm(values: &[f32]) -> F32Npy {
    F32Npy {
        shape: vec![values.len()],
        values: values.to_vec(),
    }
}
impl VoicePreparation {
    pub fn open(path: &Path, expected: &str) -> Result<Self> {
        let mut manifest: Manifest = serde_json::from_slice(&admitted_bytes(path, expected)?)
            .map_err(|e| format!("Voice preparation manifest: {e}"))?;
        if manifest.schema_version != 1 || manifest.preparation_kind != "turbo_native_voice" {
            return Err("Unsupported native voice preparation manifest".into());
        }
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        for pin in [
            &mut manifest.voice_encoder,
            &mut manifest.codec,
            &mut manifest.campplus,
            &mut manifest.analysis,
        ] {
            if pin.path.as_os_str().is_empty() || !valid_sha(&pin.sha256) {
                return Err("Native voice preparation requires pinned model manifests".into());
            }
            if pin.path.is_relative() {
                pin.path = root.join(&pin.path);
            }
            // Admit every manifest before any expensive encoder execution.
            admitted_bytes(&pin.path, &pin.sha256)?;
        }
        let analysis =
            S3ReferenceAnalysis::open(&manifest.analysis.path, &manifest.analysis.sha256)?;
        Ok(Self {
            manifest,
            audio: AudioPreparation::open()?,
            analysis,
        })
    }
    pub fn prepare(&self, path: &Path, normalise: bool) -> Result<PreparedVoice> {
        let audio = self.audio.prepare(path, normalise)?;
        // Full normalized SoXR 16k is for T3's speaker; its codec uses only15s.
        let t3_speaker = {
            let pin = &self.manifest.voice_encoder;
            T3VoiceEncoder::open(&pin.path, &pin.sha256)?
                .encode_pcm16k(&pcm(&audio.pcm16))?
                .embedding
        };
        let reference24 = pcm(audio.decoder_reference());
        let s3_pcm = self.analysis.resample(&reference24)?;
        let mels = self.analysis.mels(&reference24)?;
        let (t3_tokens, mut s3_tokens) = {
            let pin = &self.manifest.codec;
            let model = S3Tokenizer::open(&pin.path, &pin.sha256)?;
            // Two independent utterances through one immutable encoder model.
            (
                model.encode_pcm16k(&pcm(audio.t3_reference()))?,
                model.encode_pcm16k(&s3_pcm)?,
            )
        };
        if t3_tokens.is_empty() || t3_tokens.len() > 375 {
            return Err("T3 native reference requires 1..375 codec tokens".into());
        }
        // Deployed embed_ref trims only codec IDs when lengths differ. It retains
        // every mel frame, including an odd final frame, and leaves feat_len None.
        if mels.shape[1] != 2 * s3_tokens.len() {
            s3_tokens.truncate(mels.shape[1] / 2);
        }
        let (embedding, source) = {
            let pin = &self.manifest.campplus;
            let model = CampPlus::open(&pin.path, &pin.sha256)?;
            (model.encode_pcm16k(&s3_pcm)?, model.source_sha256)
        };
        let reference = S3GenReference::from_parts(
            source,
            s3_tokens.into_iter().map(i64::from).collect(),
            mels,
            None,
            embedding,
        )?;
        Ok(PreparedVoice {
            t3_speaker,
            t3_tokens,
            reference,
        })
    }
}
