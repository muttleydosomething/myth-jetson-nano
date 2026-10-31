//! Existing Myth workshop contracts, hosted by service's single admission gate.
//! One atomic state envelope is the visibility boundary for settings and voices.
//! Hidden preparation directories remain recoverable after interrupted Apply.
use crate::{
    conditioning::admitted_bytes,
    service::{SpeechRequest, Voice},
    voice_preparation::VoicePreparation,
    Result,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub ui_root: PathBuf,
    pub state_root: PathBuf,
    pub preparation: Pin,
    pub prompt: Pin,
    pub pipeline: Pin,
    #[serde(default)]
    pub delete_authority_sha256: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Upload {
    pub recording: PathBuf,
    pub recording_sha256: String,
    pub normalised: Voice,
    pub original: Voice,
    pub reference: bool,
    /// Older states infer the predefined membership from reference=false.
    /// Import can retain one identical voice in both existing UI selectors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predefined: Option<bool>,
}
impl Upload {
    fn listed(&self, reference: bool) -> bool {
        if reference {
            self.reference
        } else {
            self.predefined.unwrap_or(!self.reference)
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct Persistent {
    schema_version: u32,
    config: Value,
    uploads: BTreeMap<String, Upload>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    imported_from: Option<String>,
}

/// Offline migration names the complete recording inventory, never a guessed
/// voice mapping. Existing .native bindings prove each recording's preparation.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Import {
    schema_version: u32,
    config: Value,
    sources: Vec<ImportSource>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportSource {
    root: PathBuf,
    reference: bool,
    recordings: BTreeMap<String, String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    schema_version: u32,
    recording_sha256: String,
    models_sha256: String,
    variants: BTreeMap<String, Variant>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Variant {
    prompt: Pin,
    reference: Pin,
}

fn source_inventory(source: &ImportSource) -> Result<()> {
    let mut actual = BTreeMap::new();
    for entry in std::fs::read_dir(&source.root).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let extension = Path::new(&name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !["wav", "mp3"].contains(&extension.as_str()) {
            continue;
        }
        filename(&name)?;
        if !entry.file_type().map_err(|e| e.to_string())?.is_file() {
            return Err(format!("Import recording must be regular: {name}"));
        }
        actual.insert(name, digest_file(&entry.path())?);
    }
    if actual != source.recordings {
        return Err("Import inventory differs from the complete source recording inventory".into());
    }
    Ok(())
}

fn admit_imported_voice(voice: &Voice) -> Result<()> {
    crate::pipeline::admit(&voice.pipeline, &voice.sha256)?;
    let pipeline: Value = serde_json::from_slice(&admitted_bytes(&voice.pipeline, &voice.sha256)?)
        .map_err(|e| e.to_string())?;
    let root = voice
        .pipeline
        .parent()
        .ok_or("Imported pipeline parent absent")?;
    for key in ["prompt", "reference"] {
        let mut pin: Pin =
            serde_json::from_value(pipeline[key].clone()).map_err(|e| e.to_string())?;
        if pin.path.is_relative() {
            pin.path = root.join(pin.path);
        }
        if key == "prompt" {
            crate::conditioning::PromptAssets::open(&pin.path, &pin.sha256)?;
        } else {
            crate::s3gen::S3GenReference::open(&pin.path, &pin.sha256)?;
        }
    }
    Ok(())
}

fn imported_settings(source: &Value, seed: u64) -> Result<Value> {
    if !source.is_object() {
        return Err("Imported settings must be an object".into());
    }
    // Server paths, model-cache locations and debug controls belong to the old
    // process, not the workshop. Keep the full source in the import receipt;
    // project only controls already owned by this native settings authority.
    let mut config = defaults(seed);
    for (section, fields) in config.as_object_mut().unwrap() {
        for (key, target) in fields.as_object_mut().unwrap() {
            if let Some(value) = source.get(section).and_then(|v| v.get(key)) {
                *target = value.clone();
            }
        }
    }
    if let Some(ui) = source.get("ui_state") {
        for (key, value) in ui
            .as_object()
            .ok_or("Imported UI state must be an object")?
        {
            if UI_STATE_FIELDS.contains(&key.as_str()) {
                config["ui_state"][key] = value.clone();
            }
        }
    }
    // Turbo supports English only; the existing native workshop hides this
    // unavailable selector rather than suggesting other languages are admitted.
    config["ui"]["show_language_select"] = json!(false);
    validate_settings(&config)?;
    Ok(config)
}
pub struct Workshop {
    pub configuration: Configuration,
    state: Mutex<Persistent>,
    _ownership: std::fs::File,
}
pub fn filename(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 128
        || name.trim() != name
        || name.contains(['/', '\\'])
        || name.chars().any(|c| c.is_control())
        || !["wav", "mp3"].contains(
            &Path::new(name)
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str(),
        )
        || Path::new(name).file_stem().is_none_or(|s| s.is_empty())
    {
        return Err("Invalid reference voice filename".into());
    }
    Ok(())
}
fn defaults(seed: u64) -> Value {
    json!({"model":{"repo_id":"chatterbox-turbo"},
 "tts_engine":{"device":"cuda","default_voice_id":""},
 "generation_defaults":{"temperature":0.8,"exaggeration":0.5,"cfg_weight":0.5,"seed":seed,"speed_factor":1.0,"language":"en"},
 "audio_output":{"format":"wav","sample_rate":24000,"save_to_disk":false,"max_reference_duration_sec":30},
 "ui_state":{},"ui":{"title":"myth","show_language_select":false}})
}
fn merge(target: &mut Value, update: &Value) {
    if let (Some(dst), Some(src)) = (target.as_object_mut(), update.as_object()) {
        for (key, value) in src {
            if let Some(old) = dst.get_mut(key) {
                merge(old, value);
            } else {
                dst.insert(key.clone(), value.clone());
            }
        }
    } else {
        *target = update.clone();
    }
}
const UI_STATE_FIELDS: &[&str] = &[
    "last_text",
    "last_voice_mode",
    "last_predefined_voice",
    "last_reference_file",
    "last_seed",
    "last_chunk_size",
    "last_split_text_enabled",
    "hide_chunk_warning",
    "hide_generation_warning",
    "theme",
    "last_preset_name",
];
fn validate_settings(v: &Value) -> Result<()> {
    let root = v.as_object().ok_or("Settings must be an object")?;
    for (section, fields) in root {
        let allowed: &[&str] = match section.as_str() {
            "model" => &["repo_id"],
            "tts_engine" => &["device", "default_voice_id"],
            "generation_defaults" => &[
                "temperature",
                "exaggeration",
                "cfg_weight",
                "seed",
                "speed_factor",
                "language",
            ],
            "audio_output" => &[
                "format",
                "sample_rate",
                "save_to_disk",
                "max_reference_duration_sec",
            ],
            "ui_state" => UI_STATE_FIELDS,
            "ui" => &["title", "show_language_select"],
            _ => {
                return Err(format!(
                    "Unsupported native configuration section: {section}"
                ))
            }
        };
        for key in fields
            .as_object()
            .ok_or("Settings sections must be objects")?
            .keys()
        {
            if !allowed.contains(&key.as_str()) {
                return Err(format!("Unsupported native setting: {section}.{key}"));
            }
        }
    }
    if v["model"]["repo_id"] != "chatterbox-turbo"
        || v["tts_engine"]["device"] != "cuda"
        || v["audio_output"]["sample_rate"] != 24000
        || v["audio_output"]["save_to_disk"] != false
        || v["generation_defaults"]["language"] != "en"
    {
        return Err("Native service requires Turbo, CUDA token generation, English and 24 kHz in-memory output".into());
    }
    let g = &v["generation_defaults"];
    let temperature = g["temperature"].as_f64().ok_or("Invalid temperature")?;
    if !(0.0..=1.5).contains(&temperature)
        || g["seed"].as_u64().is_none()
        || g["exaggeration"] != 0.5
        || g["cfg_weight"] != 0.5
    {
        return Err("Invalid Turbo generation defaults; exaggeration and CFG are fixed".into());
    }
    crate::audio_output::validate_speed(g["speed_factor"].as_f64().ok_or("Invalid speed")?)?;
    crate::audio_output::Format::parse(
        v["audio_output"]["format"]
            .as_str()
            .ok_or("Invalid response format")?,
    )?;
    Ok(())
}
fn atomic_json(path: &Path, v: &impl Serialize) -> Result<()> {
    // Same-directory create-only staging, flush, rename and directory flush form
    // the commit boundary. Never expose partially written settings or voice pins.
    let pending = path.with_extension(format!(
        "pending-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos()
    ));
    let bytes = serde_json::to_vec(v).map_err(|e| e.to_string())?;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&pending)
        .map_err(|e| e.to_string())?;
    f.write_all(&bytes)
        .and_then(|_| f.sync_all())
        .map_err(|e| e.to_string())?;
    std::fs::rename(&pending, path).map_err(|e| e.to_string())?;
    std::fs::File::open(path.parent().ok_or("State parent absent")?)
        .and_then(|f| f.sync_all())
        .map_err(|e| format!("State visible; parent flush failed: {e}"))
}
impl Workshop {
    pub fn open(mut configuration: Configuration, seed: u64) -> Result<Self> {
        configuration.ui_root = std::fs::canonicalize(&configuration.ui_root)
            .map_err(|e| format!("Workshop UI: {e}"))?;
        configuration.state_root = std::fs::canonicalize(&configuration.state_root)
            .map_err(|e| format!("Workshop state: {e}"))?;
        // A second process must never independently mutate this service state.
        // The descriptor is retained for the lifetime of the workshop owner.
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(configuration.state_root.join(".lock"))
            .map_err(|e| e.to_string())?;
        use std::os::fd::AsRawFd;
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err("Workshop state is owned by another service".into());
        }
        let path = configuration.state_root.join("state.json");
        let state = if path.exists() {
            let state: Persistent =
                serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            if state.schema_version != 1 {
                return Err("Unsupported workshop state".into());
            }
            validate_settings(&state.config)?;
            for (name, entry) in &state.uploads {
                filename(name)?;
                if !entry.recording.starts_with(&configuration.state_root) {
                    return Err("Upload escaped workshop authority".into());
                }
            }
            state
        } else {
            Persistent {
                schema_version: 1,
                config: defaults(seed),
                uploads: BTreeMap::new(),
                imported_from: None,
            }
        };
        Ok(Self {
            configuration,
            state: Mutex::new(state),
            _ownership: lock,
        })
    }
    pub fn config(&self) -> Value {
        self.state.lock().unwrap().config.clone()
    }
    /// Bootstrap an empty native state from a complete, checksummed inventory.
    /// This transaction consumes retained preparations only; it cannot invoke a
    /// reference encoder. Missing voices fail before any state becomes visible.
    pub fn import(&self, path: &Path, expected: &str, seed: u64) -> Result<Value> {
        let bytes = admitted_bytes(path, expected)?;
        let mut import: Import = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if import.schema_version != 1 || import.sources.is_empty() {
            return Err("Import requires schema version 1 and complete voice sources".into());
        }
        let config = imported_settings(&import.config, seed)?;
        let mut names = BTreeMap::<String, (String, bool, bool)>::new();
        let mut entries = BTreeMap::new();
        let mut source_locks = Vec::new();
        for source in &mut import.sources {
            if !source.root.is_absolute() {
                return Err("Import source roots must be absolute".into());
            }
            source.root = std::fs::canonicalize(&source.root).map_err(|e| e.to_string())?;
            source_inventory(source)?;
            if source.root.join(".native").exists() {
                // Python's upload/delete owner uses this exact flock. Keep its
                // bundles stable until native-owned copies and state commit.
                let lock = std::fs::File::open(source.root.join(".native/.lock"))
                    .map_err(|e| format!("Import source ownership: {e}"))?;
                use std::os::fd::AsRawFd;
                if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } != 0 {
                    return Err("Import source is being changed by its owner".into());
                }
                source_locks.push(lock);
            }
            for (name, sha) in &source.recordings {
                let identity = names
                    .entry(name.clone())
                    .or_insert((sha.clone(), false, false));
                if identity.0 != *sha {
                    return Err(format!(
                        "Different recordings share the imported voice name: {name}"
                    ));
                }
                if source.reference {
                    identity.1 = true;
                } else {
                    identity.2 = true;
                }
                let marker = source
                    .root
                    .join(".native")
                    .join(format!("{:x}.json", Sha256::digest(name.as_bytes())));
                if !marker.exists() {
                    continue;
                }
                let binding: Binding =
                    serde_json::from_slice(&std::fs::read(&marker).map_err(|e| e.to_string())?)
                        .map_err(|e| format!("Import binding {name}: {e}"))?;
                if binding.schema_version != 1
                    || binding.recording_sha256 != *sha
                    || binding.models_sha256 != self.configuration.preparation.sha256
                    || binding.variants.len() != 2
                {
                    return Err(format!(
                        "Import binding disagrees with recording or model: {name}"
                    ));
                }
                entries
                    .entry(name.clone())
                    .or_insert((source.root.clone(), sha.clone(), binding));
            }
        }
        let missing = names
            .keys()
            .filter(|name| !entries.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!("Retained native preparations are not known for: {}. Import did not commit; reference preparation was not run", missing.join(", ")));
        }
        if entries.is_empty() {
            return Err("Import voice inventory is empty".into());
        }
        let mut state = self.state.lock().map_err(|_| "Workshop state poisoned")?;
        let state_path = self.configuration.state_root.join("state.json");
        if state_path.exists() || !state.uploads.is_empty() {
            if state.imported_from.as_deref() != Some(expected)
                || state.uploads.len() != entries.len()
            {
                return Err("Native state already exists; import cannot replace it".into());
            }
            // A replay never overwrites subsequently applied settings. Validate
            // the retained copies and both pipeline pins before acknowledging.
            for (name, (_, sha, _)) in &entries {
                let old = state
                    .uploads
                    .get(name)
                    .ok_or("Imported voice was withdrawn")?;
                let (_, reference, predefined) = &names[name];
                if old.reference != *reference
                    || old.listed(false) != *predefined
                    || old.recording_sha256 != *sha
                    || digest_file(&old.recording)? != *sha
                {
                    return Err("Imported recording changed".into());
                }
                admit_imported_voice(&old.normalised)?;
                admit_imported_voice(&old.original)?;
            }
            return Ok(json!({"voices":entries.len(),"reused":true,"reference_preparations":0}));
        }
        let template_bytes = admitted_bytes(
            &self.configuration.pipeline.path,
            &self.configuration.pipeline.sha256,
        )?;
        let mut template: Value =
            serde_json::from_slice(&template_bytes).map_err(|e| e.to_string())?;
        let base = self
            .configuration
            .pipeline
            .path
            .parent()
            .ok_or("Pipeline parent absent")?;
        for key in [
            "bridge",
            "engine",
            "prompt",
            "reference",
            "flow",
            "conformer",
            "mel",
            "meanflow",
            "hift",
        ] {
            let name = template[key]["path"]
                .as_str()
                .ok_or("Invalid pipeline template pin")?;
            let p = Path::new(name);
            if p.is_relative() {
                template[key]["path"] = json!(base.join(p));
            }
        }
        let mut next = Persistent {
            schema_version: 1,
            config,
            uploads: BTreeMap::new(),
            imported_from: Some(expected.into()),
        };
        for (name, (root, sha, binding)) in entries {
            let authority =
                std::fs::canonicalize(root.join(".native")).map_err(|e| e.to_string())?;
            let stage = self.stage()?;
            let recording = stage.join("recording");
            // std::io::copy streams without an utterance or recording-size cap.
            // Retain source recordings and every original conditioning bundle.
            let mut input = std::fs::File::open(root.join(&name)).map_err(|e| e.to_string())?;
            let mut output = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&recording)
                .map_err(|e| e.to_string())?;
            std::io::copy(&mut input, &mut output)
                .and_then(|_| output.sync_all())
                .map_err(|e| e.to_string())?;
            if digest_file(&recording)? != sha {
                return Err("Import recording changed while copying".into());
            }
            let mut variants = Vec::new();
            for key in ["normalised", "original"] {
                let variant = binding
                    .variants
                    .get(key)
                    .ok_or("Import requires both normalization variants")?;
                let mut pins = Vec::new();
                for pin in [&variant.prompt, &variant.reference] {
                    let path =
                        std::fs::canonicalize(root.join(&pin.path)).map_err(|e| e.to_string())?;
                    if !path.starts_with(&authority) {
                        return Err("Import preparation escaped its source authority".into());
                    }
                    admitted_bytes(&path, &pin.sha256)?;
                    pins.push((path, pin.sha256.clone()));
                }
                if pins[0].0.parent() != pins[1].0.parent() {
                    return Err("Import preparation manifests disagree".into());
                }
                // Imported voice tensors become native-owned. Later retirement
                // of a Python voice cannot invalidate these registrations.
                let bundle = stage.join(key);
                std::fs::create_dir(&bundle).map_err(|e| e.to_string())?;
                for item in
                    std::fs::read_dir(pins[0].0.parent().unwrap()).map_err(|e| e.to_string())?
                {
                    let item = item.map_err(|e| e.to_string())?;
                    if !item.file_type().map_err(|e| e.to_string())?.is_file() {
                        return Err("Import preparation contains a nonregular asset".into());
                    }
                    let mut input = std::fs::File::open(item.path()).map_err(|e| e.to_string())?;
                    let mut output = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(bundle.join(item.file_name()))
                        .map_err(|e| e.to_string())?;
                    std::io::copy(&mut input, &mut output)
                        .and_then(|_| output.sync_all())
                        .map_err(|e| e.to_string())?;
                }
                for (path, _) in &mut pins {
                    *path = bundle.join(path.file_name().ok_or("Import manifest name absent")?);
                }
                // Admit the existing tensors, including every nested checksum;
                // no audio analysis or reference-neural-model inference occurs.
                crate::conditioning::PromptAssets::open(&pins[0].0, &pins[0].1)?;
                crate::s3gen::S3GenReference::open(&pins[1].0, &pins[1].1)?;
                std::fs::File::open(&bundle)
                    .and_then(|f| f.sync_all())
                    .map_err(|e| e.to_string())?;
                let mut pipeline = template.clone();
                pipeline["prompt"] = json!({"path":pins[0].0,"sha256":pins[0].1});
                pipeline["reference"] = json!({"path":pins[1].0,"sha256":pins[1].1});
                let pipeline_bytes = serde_json::to_vec(&pipeline).map_err(|e| e.to_string())?;
                let file = format!("{key}-pipeline.json");
                let pin = crate::voice_preparation::publish_file(&stage, &file, &pipeline_bytes)?;
                let path = stage.join(file);
                crate::pipeline::admit(&path, &pin.sha256)?;
                variants.push(Voice {
                    display_name: name.clone(),
                    pipeline: path,
                    sha256: pin.sha256,
                });
            }
            std::fs::File::open(&stage)
                .and_then(|f| f.sync_all())
                .map_err(|e| e.to_string())?;
            let (_, reference, predefined) = names[&name];
            next.uploads.insert(
                name,
                Upload {
                    recording,
                    recording_sha256: sha,
                    normalised: variants.remove(0),
                    original: variants.remove(0),
                    reference,
                    predefined: Some(predefined),
                },
            );
        }
        // Catch a concurrently changed source inventory before the sole commit.
        // Failed copies remain hidden and recoverable, just like interrupted Apply.
        for source in &import.sources {
            source_inventory(source)?;
        }
        let receipt = format!("import-{expected}.json");
        let retained = self.configuration.state_root.join(&receipt);
        if retained.exists() {
            admitted_bytes(&retained, expected)?;
        } else {
            crate::voice_preparation::publish_file(
                &self.configuration.state_root,
                &receipt,
                &bytes,
            )?;
        }
        atomic_json(&state_path, &next)?;
        let count = next.uploads.len();
        *state = next;
        Ok(json!({"voices":count,"reused":false,"reference_preparations":0}))
    }
    pub fn apply(&self, update: Value) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| "Workshop state poisoned")?;
        let mut next = state.clone();
        merge(&mut next.config, &update);
        validate_settings(&next.config)?;
        atomic_json(&self.configuration.state_root.join("state.json"), &next)?;
        *state = next;
        Ok(())
    }
    pub fn reset(&self, seed: u64) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| "Workshop state poisoned")?;
        let mut next = state.clone();
        next.config = defaults(seed);
        atomic_json(&self.configuration.state_root.join("state.json"), &next)?;
        *state = next;
        Ok(())
    }
    pub fn list(&self, reference: bool) -> Vec<(String, String)> {
        self.state
            .lock()
            .unwrap()
            .uploads
            .iter()
            .filter(|(_, u)| u.listed(reference))
            .map(|(name, _)| {
                (
                    name.clone(),
                    Path::new(name)
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .replace('_', " "),
                )
            })
            .collect()
    }
    pub fn resolve(&self, name: &str, normalise: bool) -> Result<Option<Voice>> {
        let state = self.state.lock().map_err(|_| "Workshop state poisoned")?;
        if let Some(u) = state.uploads.get(name) {
            // A committed binding never falls back to a raw recording or encoder.
            let sha = digest_file(&u.recording)?;
            if sha != u.recording_sha256 {
                return Err("Voice recording checksum differs".into());
            }
            let v = if normalise {
                &u.normalised
            } else {
                &u.original
            };
            admitted_bytes(&v.pipeline, &v.sha256)?;
            Ok(Some(v.clone()))
        } else {
            Ok(None)
        }
    }
    pub fn source(&self, name: &str) -> Result<Option<PathBuf>> {
        filename(name)?;
        Ok(self
            .state
            .lock()
            .map_err(|_| "Workshop state poisoned")?
            .uploads
            .get(name)
            .map(|u| u.recording.clone()))
    }
    pub fn stage(&self) -> Result<PathBuf> {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let stage = self
            .configuration
            .state_root
            .join(format!(".upload-{}-{id}", std::process::id()));
        std::fs::create_dir(&stage).map_err(|e| e.to_string())?;
        Ok(stage)
    }
    pub fn upload(
        &self,
        name: &str,
        stage: &Path,
        reference: bool,
        cancel: &AtomicBool,
    ) -> Result<usize> {
        filename(name)?;
        let recording = stage.join("recording");
        let sha = digest_file(&recording)?;
        {
            let state = self.state.lock().map_err(|_| "Workshop state poisoned")?;
            if let Some(old) = state.uploads.get(name) {
                if !old.listed(reference) || old.recording_sha256 != sha {
                    return Err("Voice name belongs to a different recording".into());
                }
                drop(state);
                self.resolve(name, true)?;
                self.resolve(name, false)?;
                return Ok(0);
            }
        }
        // Borg's native runtime mounts libsndfile 1.2.2, including MPEG decode.
        // Both formats share AudioPreparation; no child decoder is required.
        let engine = VoicePreparation::open(
            &self.configuration.preparation.path,
            &self.configuration.preparation.sha256,
        )?;
        let pipeline_bytes = admitted_bytes(
            &self.configuration.pipeline.path,
            &self.configuration.pipeline.sha256,
        )?;
        let mut template: Value =
            serde_json::from_slice(&pipeline_bytes).map_err(|e| e.to_string())?;
        let base = self
            .configuration
            .pipeline
            .path
            .parent()
            .ok_or("Pipeline parent absent")?;
        // Preserve all model pins while resolving relative paths before publication.
        for key in [
            "bridge",
            "engine",
            "prompt",
            "reference",
            "flow",
            "conformer",
            "mel",
            "meanflow",
            "hift",
        ] {
            let path = PathBuf::from(
                template[key]["path"]
                    .as_str()
                    .ok_or("Incomplete pipeline template")?,
            );
            if path.is_relative() {
                template[key]["path"] = json!(base.join(path));
            }
        }
        let mut variants = Vec::new();
        for (normalise, key) in [(true, "normalised"), (false, "original")] {
            if cancel.load(Ordering::Acquire) {
                return Err("Voice Apply cancelled".into());
            }
            let prepared = engine.prepare(&recording, normalise)?;
            let bundle = stage.join(key);
            let published = prepared.publish(
                &self.configuration.prompt.path,
                &self.configuration.prompt.sha256,
                &bundle,
            )?;
            let mut pipeline = template.clone();
            pipeline["prompt"] =
                json!({"path":published.prompt.path,"sha256":published.prompt.sha256});
            pipeline["reference"] =
                json!({"path":published.reference.path,"sha256":published.reference.sha256});
            let path = bundle.join("pipeline.json");
            let bytes = serde_json::to_vec(&pipeline).map_err(|e| e.to_string())?;
            crate::voice_preparation::publish_file(&bundle, "pipeline.json", &bytes)?;
            // Publication added the aggregate after the existing bundle commit;
            // flush its directory before the state envelope can expose it.
            std::fs::File::open(&bundle)
                .and_then(|f| f.sync_all())
                .map_err(|e| e.to_string())?;
            let voice = Voice {
                display_name: name.into(),
                pipeline: path,
                sha256: format!("{:x}", Sha256::digest(&bytes)),
            };
            variants.push(voice);
        }
        if cancel.load(Ordering::Acquire) {
            return Err("Voice Apply cancelled".into());
        }
        std::fs::File::open(stage)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
        let mut state = self.state.lock().map_err(|_| "Workshop state poisoned")?;
        let mut next = state.clone();
        next.uploads.insert(
            name.into(),
            Upload {
                recording,
                recording_sha256: sha,
                normalised: variants.remove(0),
                original: variants.remove(0),
                reference,
                predefined: None,
            },
        );
        // Completed immutable bundles are hidden until this single envelope commits.
        atomic_json(&self.configuration.state_root.join("state.json"), &next)?;
        *state = next;
        Ok(2)
    }
    pub fn delete(&self, name: &str) -> Result<()> {
        filename(name)?;
        let mut state = self.state.lock().map_err(|_| "Workshop state poisoned")?;
        let upload = state
            .uploads
            .get(name)
            .ok_or("Reference voice is not known")?;
        let directory = upload.recording.parent().ok_or("Upload parent absent")?;
        if directory.parent() != Some(self.configuration.state_root.as_path())
            || directory
                .symlink_metadata()
                .map_err(|e| e.to_string())?
                .file_type()
                .is_symlink()
            || !directory
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".upload-")
        {
            return Err("Deletion escaped upload authority".into());
        }
        let directory = directory.to_path_buf();
        let mut next = state.clone();
        next.uploads.remove(name);
        atomic_json(&self.configuration.state_root.join("state.json"), &next)?;
        *state = next;
        std::fs::remove_dir_all(&directory)
            .map_err(|e| format!("Voice withdrawn; preparation reclamation failed: {e}"))?;
        std::fs::File::open(&self.configuration.state_root)
            .and_then(|f| f.sync_all())
            .map_err(|e| format!("Voice removed; parent flush failed: {e}"))?;
        Ok(())
    }
    pub fn authorize_delete(&self, headers: &axum::http::HeaderMap) -> bool {
        let expected = &self.configuration.delete_authority_sha256;
        if expected.len() != 64 {
            return false;
        }
        let Some(token) = headers
            .get("x-hailstorm-voice-authority")
            .and_then(|s| s.to_str().ok())
            .filter(|s| !s.is_empty())
        else {
            return false;
        };
        let actual = format!("{:x}", Sha256::digest(token.as_bytes()));
        actual
            .bytes()
            .zip(expected.bytes())
            .fold(0u8, |difference, (a, b)| difference | (a ^ b))
            == 0
    }
}
pub fn digest_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let metadata = path.symlink_metadata().map_err(|e| e.to_string())?;
    if !metadata.file_type().is_file() {
        return Err("Expected a regular recording".into());
    }
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut sha = Sha256::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = f.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        sha.update(&chunk[..n]);
    }
    Ok(format!("{:x}", sha.finalize()))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub text: String,
    #[serde(default)]
    pub voice_mode: Option<String>,
    pub predefined_voice_id: Option<String>,
    pub reference_audio_filename: Option<String>,
    pub output_format: Option<String>,
    pub temperature: Option<f32>,
    pub exaggeration: Option<f64>,
    pub cfg_weight: Option<f64>,
    pub seed: Option<u64>,
    pub speed_factor: Option<f64>,
    pub language: Option<String>,
    pub split_text: Option<bool>,
    pub chunk_size: Option<usize>,
}
impl Request {
    pub fn speech(self, model: String, config: &Value) -> Result<SpeechRequest> {
        if self.language.as_deref().unwrap_or("en") != "en"
            || self.exaggeration.unwrap_or(0.5) != 0.5
            || self.cfg_weight.unwrap_or(0.5) != 0.5
        {
            return Err("Native Turbo supports English with fixed exaggeration and CFG".into());
        }
        let temperature = self.temperature.unwrap_or(
            config["generation_defaults"]["temperature"]
                .as_f64()
                .unwrap_or(0.8) as f32,
        );
        if !(0.0..=1.5).contains(&temperature) {
            return Err("Invalid temperature".into());
        }
        if self.chunk_size.is_some_and(|n| !(50..=500).contains(&n)) {
            return Err("Invalid chunk size".into());
        }
        let mode = self.voice_mode.as_deref().unwrap_or("predefined");
        let voice = match mode {
            "predefined" => self.predefined_voice_id,
            "clone" => self.reference_audio_filename,
            _ => return Err("Unknown voice mode".into()),
        }
        .ok_or("A prepared voice is required")?;
        Ok(SpeechRequest {
            model,
            input: self.text,
            voice,
            response_format: self.output_format.unwrap_or_else(|| {
                config["audio_output"]["format"]
                    .as_str()
                    .unwrap_or("wav")
                    .into()
            }),
            speed: self.speed_factor.unwrap_or(
                config["generation_defaults"]["speed_factor"]
                    .as_f64()
                    .unwrap_or(1.),
            ),
            seed: self
                .seed
                .or_else(|| config["generation_defaults"]["seed"].as_u64()),
            temperature: Some(temperature),
            workshop_request: true,
            // Finite text chunks share one admission and one final response (step 2 adds
            // incremental playback). This preserves the workshop's long-text control.
            chunk_size: if self.split_text.unwrap_or(true) {
                Some(self.chunk_size.unwrap_or(120))
            } else {
                None
            },
        })
    }
}
/// Preserve complete sentences; a long sentence remains one chunk. Splitting
/// activates only above the deployed 1.5x threshold. Tokens inside [] are kept.
pub fn chunks(text: &str, size: Option<usize>) -> Vec<String> {
    let Some(size) = size else {
        return vec![text.into()];
    };
    if text.chars().count() <= size.saturating_mul(3) / 2 {
        return vec![text.into()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut brackets: usize = 0;
    for (index, &ch) in chars.iter().enumerate() {
        if ch == '[' {
            brackets += 1;
        } else if ch == ']' {
            brackets = brackets.saturating_sub(1);
        }
        if brackets == 0
            && ['.', '!', '?', '\n'].contains(&ch)
            && (index + 1 == chars.len() || chars[index + 1].is_whitespace())
        {
            sentences.push(chars[start..=index].iter().collect::<String>());
            start = index + 1;
        }
    }
    if start < chars.len() {
        sentences.push(chars[start..].iter().collect());
    }
    for sentence in sentences {
        let word = sentence.trim();
        if word.is_empty() {
            continue;
        }
        if !current.is_empty() && current.chars().count() + 1 + word.chars().count() > size {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Deployed /tts smart join: 20 ms equal-power edges around 200 ms of silence.
/// No extra fade is applied to a one-chunk request or the OpenAI response.
pub fn join(output: &mut Vec<f32>, next: &[f32]) {
    if output.is_empty() {
        output.extend_from_slice(next);
        return;
    }
    let left = 480.min(output.len());
    let right = 480.min(next.len());
    let n = output.len();
    for i in 0..left {
        let t = (i as f32) * std::f32::consts::FRAC_PI_2 / ((left - 1).max(1) as f32);
        output[n - left + i] *= t.cos().powi(2);
    }
    // An oversized silence buffer loses one overlap at each end, preserving
    // the deployed exact response length even for unusually short chunks.
    output.resize(n + 5760 - left - right, 0.);
    let offset = output.len();
    output.extend_from_slice(next);
    for i in 0..right {
        let t = (i as f32) * std::f32::consts::FRAC_PI_2 / ((right - 1).max(1) as f32);
        output[offset + i] *= t.sin().powi(2);
    }
}
