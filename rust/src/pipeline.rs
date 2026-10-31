//! One final, unpadded retained-voice request through the existing native stages.
//! Shared by the offline command and native HTTP service. Service token and
//! meanflow owners retain verified engines/contexts; token KV retires before
//! waveform work. Other models and CLI keep their sequential lifetimes.
use crate::{
    conditioning::{admitted_bytes, prompt_digest, PromptAssets},
    conformer::S3GenConformer,
    generation::{self, Generated},
    hift::{finish_waveform, S3GenHiFT},
    meanflow::S3GenMeanflow,
    npy::F32Npy,
    s3gen::{S3GenFlowAssets, S3GenMelAssets, S3GenReference},
    sampling::{self, Sampling},
    tensorrt::{check_engine, EngineCache, TensorRt},
    text, Result,
};
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pinned {
    path: PathBuf,
    sha256: String,
}
impl Pinned {
    // Aggregate manifests deliberately reference existing export directories.
    // Each referenced manifest still enforces its own local asset/checksum rules.
    fn resolve(&mut self, root: &Path) {
        if self.path.is_relative() {
            self.path = root.join(&self.path);
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    bridge: Pinned,
    engine: Pinned,
    prompt: Pinned,
    reference: Pinned,
    flow: Pinned,
    conformer: Pinned,
    mel: Pinned,
    meanflow: Pinned,
    hift: Pinned,
}
#[derive(Serialize)]
pub struct Output {
    pub seed: u64,
    pub text: crate::tokenizer::TokenizedText,
    pub prompt_sha256: String,
    pub speech_tokens: Vec<u32>,
    pub decoder_tokens: Vec<i64>,
    pub reference_frames: usize,
    pub generated_frames: usize,
    pub samples: usize,
    pub peak: f32,
    pub waveform_sha256: String,
    pub token_engine_cache_hit: bool,
    pub stages_ms: std::collections::BTreeMap<&'static str, u128>,
    pub waveform_compute: std::collections::BTreeMap<&'static str, crate::gpu::Stats>,
}

/// Complete in-memory response; HTTP never creates a temporary output artifact.
pub struct Rendered {
    pub wav: Vec<u8>,
    /// Retain the finished samples for response codecs; never decode the proof
    /// WAV or quantise the inference receipt merely to serve another format.
    pub waveform: F32Npy,
    pub output: Output,
}

fn check_cancelled(cancelled: &mut impl FnMut() -> bool) -> Result<()> {
    if cancelled() {
        Err("Speech request cancelled".into())
    } else {
        Ok(())
    }
}

/// One growing-token utterance owns all stochastic and vocoder history. Models
/// remain immutable and callers retain the existing inference permit until this
/// value and any executing kernels are retired. No voice preparation occurs.
pub struct WaveformStream {
    noise: sampling::RequestNoise,
    overlap: crate::hift::WaveformOverlap,
    reference_frames: Option<usize>,
    generated_frames: usize,
    closed: bool,
}
impl WaveformStream {
    pub fn new(seed: u64, overlap_samples: usize) -> Result<Self> {
        Ok(Self {
            noise: sampling::RequestNoise::new(seed),
            overlap: crate::hift::WaveformOverlap::new(overlap_samples)?,
            reference_frames: None,
            generated_frames: 0,
            closed: false,
        })
    }
    /// Feed a full, conditioned mel prefix, preserving absolute noise/source
    /// coordinates. Input token-prefix stability belongs to the caller. The
    /// noncausal model still requires audible join acceptance before promotion.
    pub fn decode_prefix(
        &mut self,
        input: &crate::s3gen::S3GenMelInput,
        meanflow: &S3GenMeanflow,
        hift: &S3GenHiFT,
        finalize: bool,
        cancelled: impl FnMut() -> bool,
    ) -> Result<crate::hift::WaveformFragment> {
        self.decode_with(
            input,
            |noise| Ok(meanflow.decode(input, noise, 2)?.generated),
            |mel, normal, phases, source| hift.synthesize_cached(mel, normal, phases, source),
            finalize,
            cancelled,
        )
    }

    /// The production loader can retire one stage before opening the next;
    /// retained-model callers above share this same state transition.
    fn decode_with(
        &mut self,
        input: &crate::s3gen::S3GenMelInput,
        meanflow: impl FnOnce(&F32Npy) -> Result<F32Npy>,
        hift: impl FnOnce(&F32Npy, &F32Npy, &F32Npy, &F32Npy) -> Result<crate::hift::HiFTOutput>,
        finalize: bool,
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<crate::hift::WaveformFragment> {
        if self.closed {
            return Err("Streaming request is closed after finalization or failure".into());
        }
        // A failed/cancelled attempt may have advanced RNG coordinates. Retire
        // it rather than letting a caller reuse a superficially unchanged cursor.
        // Only a successful non-final update reopens this request.
        self.closed = true;
        check_cancelled(&mut cancelled)?;
        if self
            .reference_frames
            .is_some_and(|frames| frames != input.reference_frames)
            || input.generated_frames < self.generated_frames
            || (!finalize && input.generated_frames == self.generated_frames)
            || input.generated_frames < 2
        {
            return Err("Streaming mel prefix is stale, inconsistent, or finalized".into());
        }
        let frames = input
            .mu
            .shape
            .get(2)
            .copied()
            .ok_or("Missing streaming mel frames")?;
        let noise = self.noise.meanflow(frames)?;
        let mel = meanflow(&noise)?;
        check_cancelled(&mut cancelled)?;
        let (normal, phases) = self.noise.hift(input.generated_frames)?;
        let raw = hift(&mel, &normal, &phases, self.overlap.cached_source())?;
        check_cancelled(&mut cancelled)?;
        let fragment = self.overlap.push(raw, finalize)?;
        self.reference_frames = Some(input.reference_frames);
        self.generated_frames = input.generated_frames;
        self.closed = finalize;
        Ok(fragment)
    }
}

/// A partial token sequence is never presented as a successful speech request.
/// Use the established cleanup exactly once; flow.assemble consumes codec IDs.
pub fn decoder_tokens(generated: &Generated) -> Result<Vec<i64>> {
    if !generated.reached_eos {
        return Err("Speech generation stopped before EOS; no WAV produced".into());
    }
    if !generated.tokens.iter().any(|&n| n < generation::BOS) {
        return Err("Speech generation produced no codec tokens; no WAV produced".into());
    }
    Ok(text::finish_speech_tokens(
        &generated
            .tokens
            .iter()
            .map(|&n| i64::from(n))
            .collect::<Vec<_>>(),
    ))
}

/// Lossless mono IEEE-float WAV. Include a fact chunk for non-PCM readers.
/// No output amplification, clipping, quantisation or second fade is introduced.
pub fn wav_bytes(waveform: &F32Npy) -> Result<Vec<u8>> {
    if waveform.shape.len() != 2
        || waveform.shape[0] != 1
        || waveform.shape[1] == 0
        || waveform.shape[1] != waveform.values.len()
        || waveform.values.iter().any(|v| !v.is_finite())
    {
        return Err("WAV requires finite complete mono samples [1,N]".into());
    }
    let size = waveform
        .values
        .len()
        .checked_mul(4)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or("Waveform exceeds RIFF WAV size")?;
    let riff_size = size
        .checked_add(48)
        .ok_or("Waveform exceeds RIFF WAV size")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(riff_size as usize + 8)
        .map_err(|e| e.to_string())?;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&24000u32.to_le_bytes());
    bytes.extend_from_slice(&96000u32.to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&32u16.to_le_bytes());
    bytes.extend_from_slice(b"fact");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&(waveform.values.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&size.to_le_bytes());
    for v in &waveform.values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Ok(bytes)
}

/// The checksummed aggregate pins both voice states and every model boundary.
/// No export/capture subprocess or audio/reference preparation occurs here.
pub fn synthesize(
    manifest: &Path,
    sha256: &str,
    input: &str,
    output: &Path,
    seed: u64,
    limit: usize,
    sampling: &Sampling,
) -> Result<Output> {
    if output.exists() {
        return Err("WAV destination already exists".into());
    }
    let rendered = render(manifest, sha256, input, seed, limit, sampling, || false)?;
    // Preserve CLI create-only publication; HTTP returns these same bytes directly.
    let mut destination = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| format!("WAV destination: {e}"))?;
    destination
        .write_all(&rendered.wav)
        .map_err(|e| format!("WAV output incomplete: {e}"))?;
    destination
        .sync_all()
        .map_err(|e| format!("WAV output flush: {e}"))?;
    Ok(rendered.output)
}

/// One inference routine for both entry points. Cancellation is checked between
/// Admit aggregate model pins for the service reload action without allocating
/// inference state. Service inference later admits reusable native contexts.
pub fn admit(path: &Path, sha256: &str) -> Result<()> {
    let manifest: Manifest =
        serde_json::from_slice(&admitted_bytes(path, sha256)?).map_err(|e| e.to_string())?;
    if manifest.schema_version != 1 {
        return Err("Unsupported pipeline manifest".into());
    }
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    for mut pin in [
        manifest.bridge,
        manifest.engine,
        manifest.prompt,
        manifest.reference,
        manifest.flow,
        manifest.conformer,
        manifest.mel,
        manifest.meanflow,
        manifest.hift,
    ] {
        pin.resolve(root);
        // Reuse the existing 64 KiB checksum admission. The engine is large;
        // reload must never buffer it beside a process's retained CPU arenas.
        check_engine(&pin.path, &pin.sha256)
            .map_err(|e| format!("Reload asset {}: {e}", pin.path.display()))?;
    }
    Ok(())
}

struct PreparedRequest {
    manifest: Manifest,
    text: crate::tokenizer::TokenizedText,
    prompt_sha256: String,
    generated: Generated,
    tokens: Vec<i64>,
    stages_ms: std::collections::BTreeMap<&'static str, u128>,
    token_engine_cache_hit: bool,
}

/// Subspans overlap the existing inclusive stage totals. Time synchronous work
/// in place: profiling must not introduce a second decode or a CUDA fence.
fn timed<T>(
    spans: &mut std::collections::BTreeMap<&'static str, u128>,
    name: &'static str,
    work: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _trace = crate::trace::span(name);
    let began = Instant::now();
    let result = work();
    spans.insert(name, began.elapsed().as_millis());
    result
}

/// Share admission, tokenization, EOS and cleanup between complete and streaming
/// responses. Request KV retires before either waveform path admits models.
fn prepare_request(
    manifest: &Path,
    sha256: &str,
    input: &str,
    seed: u64,
    limit: usize,
    sampling: &Sampling,
    mut cancelled: impl FnMut() -> bool,
    engine_cache: Option<&Mutex<EngineCache>>,
) -> Result<PreparedRequest> {
    check_cancelled(&mut cancelled)?;
    sampling.validate()?;
    let root = manifest.parent().unwrap_or_else(|| Path::new("."));
    let mut manifest: Manifest = serde_json::from_slice(&admitted_bytes(manifest, sha256)?)
        .map_err(|e| e.to_string())
        .and_then(|m: Manifest| {
            if m.schema_version != 1 {
                Err("Unsupported pipeline manifest".into())
            } else {
                Ok(m)
            }
        })?;
    for asset in [
        &mut manifest.bridge,
        &mut manifest.engine,
        &mut manifest.prompt,
        &mut manifest.reference,
        &mut manifest.flow,
        &mut manifest.conformer,
        &mut manifest.mel,
        &mut manifest.meanflow,
        &mut manifest.hift,
    ] {
        asset.resolve(root);
    }
    let mut stages_ms = std::collections::BTreeMap::new();
    let timer = Instant::now();
    let prompt_trace = crate::trace::span("prompt_admission");
    let (text, prompt, embeddings) = if let Some(cache) = engine_cache {
        cache
            .lock()
            .map_err(|_| "Prompt cache lock poisoned")?
            .prompt
            .prepare(&manifest.prompt.path, &manifest.prompt.sha256, input)?
    } else {
        let assets = PromptAssets::open(&manifest.prompt.path, &manifest.prompt.sha256)?;
        let text = assets.tokenize(input)?;
        let prompt = assets.assemble(&text)?;
        (
            text,
            prompt,
            std::sync::Arc::new(assets.into_speech_embeddings()),
        )
    };
    let prompt_sha256 = prompt_digest(&prompt);
    // Full bridge checksum on real admission, identity checks on warm hits.
    // The process owner also prevents silently rebinding native plugin code.
    if let Some(cache) = engine_cache {
        cache
            .lock()
            .map_err(|_| "Bridge cache lock poisoned")?
            .admit_bridge(&manifest.bridge.path, &manifest.bridge.sha256)?;
    } else {
        check_engine(&manifest.bridge.path, &manifest.bridge.sha256)?;
    }
    stages_ms.insert("prompt_admission", timer.elapsed().as_millis());
    drop(prompt_trace);
    check_cancelled(&mut cancelled)?;
    let timer = Instant::now();
    let mut token_engine_cache_hit = false;
    let generated = {
        let mut decoder = timed(&mut stages_ms, "token_model_load", || {
            if let Some(cache) = engine_cache {
                let (decoder, hit) = cache
                    .lock()
                    .map_err(|_| "Token engine cache lock poisoned")?
                    .session(
                        &manifest.bridge.path,
                        &manifest.engine.path,
                        &manifest.engine.sha256,
                        prompt,
                        embeddings,
                    )?;
                token_engine_cache_hit = hit;
                // Explicit receipt distinguishes retained weights from merely
                // warm filesystem pages. A reused context has empty request KV.
                eprintln!(
                    "{}",
                    serde_json::json!({"operation":"native_token_engine",
                    "cache_hit":hit,"engine_sha256":manifest.engine.sha256})
                );
                return Ok(decoder);
            }
            TensorRt::open(
                &manifest.bridge.path,
                &manifest.engine.path,
                &manifest.engine.sha256,
                prompt,
                embeddings,
            )
        })?;
        timed(&mut stages_ms, "token_inference", || {
            generation::generate(&mut decoder, sampling, limit, seed, &mut cancelled)
        })?
    }; // Return the reset context to the service; request KV has been released.
    stages_ms.insert("tokens", timer.elapsed().as_millis());
    let tokens = decoder_tokens(&generated)?;
    Ok(PreparedRequest {
        manifest,
        text,
        prompt_sha256,
        generated,
        tokens,
        stages_ms,
        token_engine_cache_hit,
    })
}

/// Complete and incremental routes share the same service admission owner.
/// Only TensorRT graphs are retained; CPU oracle/CLI models keep their original
/// sequential release. An inference failure retires a possibly poisoned graph.
fn run_meanflow(
    pin: &Pinned,
    input: &crate::s3gen::S3GenMelInput,
    noise: &F32Npy,
    stages: &mut std::collections::BTreeMap<&'static str, u128>,
    cache: Option<&Mutex<EngineCache>>,
) -> Result<(crate::meanflow::MeanflowOutput, crate::gpu::Stats)> {
    if let Some(cache) = cache {
        let mut owner = cache.lock().map_err(|_| "Engine cache lock poisoned")?;
        let admitted = timed(stages, "meanflow_model_load", || {
            owner.meanflow.admit(&pin.path, &pin.sha256)
        });
        match admitted {
            Ok(true) => {
                return timed(stages, "meanflow_inference", || {
                    owner.meanflow.run(input, noise, 2)
                })
            }
            Ok(false) => (),
            Err(e) => {
                owner.meanflow.clear();
                return Err(e);
            }
        }
    }
    let model = timed(stages, "meanflow_model_load", || {
        S3GenMeanflow::open(&pin.path, &pin.sha256)
    })?;
    timed(stages, "meanflow_inference", || model.run(input, noise, 2))
}

/// Shared HiFT dispatch for complete and incremental output. Graph requests
/// use the service owner; native tensor models keep their existing CUDA scope.
fn run_hift(
    pin: &Pinned,
    mel: &F32Npy,
    normal: &F32Npy,
    phases: &F32Npy,
    source: &F32Npy,
    stages: &mut std::collections::BTreeMap<&'static str, u128>,
    cache: Option<&Mutex<EngineCache>>,
) -> Result<(crate::hift::HiFTOutput, crate::gpu::Stats)> {
    if let Some(cache) = cache {
        let mut owner = cache.lock().map_err(|_| "Engine cache lock poisoned")?;
        match timed(stages, "hift_model_load", || {
            owner.hift.admit(&pin.path, &pin.sha256)
        }) {
            Ok(true) => {
                return timed(stages, "hift_inference", || {
                    owner.hift.run(mel, normal, phases, source)
                })
            }
            Ok(false) => (),
            Err(e) => {
                owner.hift.clear();
                return Err(e);
            }
        }
    }
    crate::gpu::run_from_env(|| {
        let model = timed(stages, "hift_model_load", || {
            S3GenHiFT::open(&pin.path, &pin.sha256)
        })?;
        timed(stages, "hift_inference", || {
            model.synthesize_cached(mel, normal, phases, source)
        })
    })
}

/// Complete and incremental requests share one encoder authority. Device graph
/// inputs and final features cross once; tensor/CLI callers retain their scope.
fn run_conformer(
    pin: &Pinned,
    input: &F32Npy,
    stages: &mut std::collections::BTreeMap<&'static str, u128>,
    cache: Option<&Mutex<EngineCache>>,
) -> Result<(crate::conformer::EncoderOutput, crate::gpu::Stats)> {
    if let Some(cache) = cache {
        let mut owner = cache.lock().map_err(|_| "Conformer cache lock poisoned")?;
        if timed(stages, "conformer_model_load", || {
            owner.conformer.admit(&pin.path, &pin.sha256)
        })? {
            return timed(stages, "conformer_inference", || owner.conformer.run(input));
        }
    }
    crate::gpu::run_from_env(|| {
        let model = timed(stages, "conformer_model_load", || {
            S3GenConformer::open(&pin.path, &pin.sha256)
        })?;
        timed(stages, "conformer_inference", || model.encode(input))
    })
}

/// One inference routine for CLI and complete HTTP responses. Its original
/// numerical path and random sequence remain unchanged by streaming support.
pub fn render(
    manifest: &Path,
    sha256: &str,
    input: &str,
    seed: u64,
    limit: usize,
    sampling: &Sampling,
    mut cancelled: impl FnMut() -> bool,
) -> Result<Rendered> {
    render_with_engine_cache(
        manifest,
        sha256,
        input,
        seed,
        limit,
        sampling,
        &mut cancelled,
        None,
    )
}

/// The native service supplies its existing, bounded engine owner. Offline
/// callers use render() and do not retain a process-global model allocation.
pub fn render_with_engine_cache(
    manifest: &Path,
    sha256: &str,
    input: &str,
    seed: u64,
    limit: usize,
    sampling: &Sampling,
    mut cancelled: impl FnMut() -> bool,
    engine_cache: Option<&Mutex<EngineCache>>,
) -> Result<Rendered> {
    let mut trace = crate::trace::Root::begin("complete");
    let PreparedRequest {
        manifest,
        text,
        prompt_sha256,
        generated,
        tokens,
        mut stages_ms,
        token_engine_cache_hit,
    } = prepare_request(
        manifest,
        sha256,
        input,
        seed,
        limit,
        sampling,
        &mut cancelled,
        engine_cache,
    )?;
    let mut waveform_compute = std::collections::BTreeMap::new();
    check_cancelled(&mut cancelled)?;
    let timer = Instant::now();
    let (mel_input, compute) = {
        let reference = S3GenReference::open(&manifest.reference.path, &manifest.reference.sha256)?;
        let flow = S3GenFlowAssets::open(&manifest.flow.path, &manifest.flow.sha256)?;
        let prepared = flow.prepare(&reference)?;
        let input = prepared.assemble(&tokens)?;
        let (encoded, compute) = run_conformer(
            &manifest.conformer,
            input.embeddings(),
            &mut stages_ms,
            engine_cache,
        )?;
        let mel = S3GenMelAssets::open(&manifest.mel.path, &manifest.mel.sha256)?;
        // Preserve the established projection kernel and its numerical route.
        // It is outside the retained encoder graph and owns its original scope.
        let (conditioned, projection) =
            crate::gpu::run_from_env(|| mel.condition(&input, &encoded, true))?;
        waveform_compute.insert("mel_projection", projection);
        (conditioned, compute)
    };
    waveform_compute.insert("conformer", compute);
    stages_ms.insert("encoder_conditioning", timer.elapsed().as_millis());
    check_cancelled(&mut cancelled)?;
    let reference_frames = mel_input.reference_frames;
    let generated_frames = mel_input.generated_frames;
    let timer = Instant::now();
    let noise = sampling::s3gen_meanflow_noise(mel_input.mu.shape[2], seed)?;
    let (mel, compute) = run_meanflow(
        &manifest.meanflow,
        &mel_input,
        &noise,
        &mut stages_ms,
        engine_cache,
    )?;
    let mel = mel.generated;
    waveform_compute.insert("meanflow", compute);
    drop(mel_input);
    stages_ms.insert("meanflow", timer.elapsed().as_millis());
    check_cancelled(&mut cancelled)?;
    let timer = Instant::now();
    let (normal, phases) = sampling::s3gen_hift_random(mel.shape[2], seed)?;
    let empty_source = F32Npy {
        shape: vec![1, 1, 0],
        values: Vec::new(),
    };
    let (output, compute) = run_hift(
        &manifest.hift,
        &mel,
        &normal,
        &phases,
        &empty_source,
        &mut stages_ms,
        engine_cache,
    )?;
    let waveform = finish_waveform(&output.waveform)?;
    waveform_compute.insert("hift", compute);
    stages_ms.insert("hift_finish", timer.elapsed().as_millis());
    check_cancelled(&mut cancelled)?;
    if waveform.values.len()
        != generated_frames
            .checked_mul(480)
            .ok_or("Waveform length overflow")?
    {
        return Err("Waveform length differs from generated mel frames".into());
    }
    let wav = wav_bytes(&waveform)?;
    use sha2::{Digest, Sha256};
    let waveform_sha256 = format!("{:x}", Sha256::digest(&wav));
    let output = Output {
        seed,
        text,
        prompt_sha256,
        speech_tokens: generated.tokens,
        decoder_tokens: tokens,
        reference_frames,
        generated_frames,
        samples: waveform.values.len(),
        peak: waveform.values.iter().fold(0f32, |a, v| a.max(v.abs())),
        waveform_sha256,
        token_engine_cache_hit,
        stages_ms,
        waveform_compute,
    };
    trace.success();
    Ok(Rendered {
        wav,
        waveform,
        output,
    })
}

/// Codec IDs have already passed EOS admission and cleanup. Only the final
/// prefix includes the three established silence tokens. Non-final calls leave
/// six mel lookahead frames to the conditioning boundary.
pub fn streaming_prefixes(tokens: &[i64]) -> Result<Vec<(usize, bool)>> {
    if tokens.len() < 4
        || tokens[tokens.len() - 3..] != [4299; 3]
        || tokens.iter().any(|&token| !(0..6561).contains(&token))
    {
        return Err("Streaming requires cleaned codec tokens and one final silence suffix".into());
    }
    let mut prefixes: Vec<_> = (32..tokens.len() - 3)
        .step_by(32)
        .map(|end| (end, false))
        .collect();
    prefixes.push((tokens.len(), true));
    Ok(prefixes)
}

/// Incremental waveform synthesis after T3 has reached EOS. Release its request
/// context/KV first; no simultaneous token/waveform contexts or background
/// inference queue are introduced. The same immutable
/// waveform tables serve each prefix; only noise/source/tail state stays resident
/// between the established, separately admitted waveform stages.
pub fn render_stream(
    manifest: &Path,
    sha256: &str,
    input: &str,
    seed: u64,
    limit: usize,
    sampling: &Sampling,
    mut cancelled: impl FnMut() -> bool,
    mut emit: impl FnMut(F32Npy, bool) -> Result<()>,
) -> Result<serde_json::Value> {
    render_stream_with_engine_cache(
        manifest,
        sha256,
        input,
        seed,
        limit,
        sampling,
        &mut cancelled,
        &mut emit,
        None,
    )
}

/// Service entry shares the exact waveform/random/transport path with CLI.
pub fn render_stream_with_engine_cache(
    manifest: &Path,
    sha256: &str,
    input: &str,
    seed: u64,
    limit: usize,
    sampling: &Sampling,
    mut cancelled: impl FnMut() -> bool,
    mut emit: impl FnMut(F32Npy, bool) -> Result<()>,
    engine_cache: Option<&Mutex<EngineCache>>,
) -> Result<serde_json::Value> {
    let mut trace = crate::trace::Root::begin("stream");
    let PreparedRequest {
        manifest,
        tokens,
        mut stages_ms,
        token_engine_cache_hit,
        ..
    } = prepare_request(
        manifest,
        sha256,
        input,
        seed,
        limit,
        sampling,
        &mut cancelled,
        engine_cache,
    )?;
    let prefixes = streaming_prefixes(&tokens)?;
    let timer = Instant::now();
    // Bind one admitted reference for the entire request. Large neural models
    // keep the established sequential stage ownership and host-arena release.
    let reference = S3GenReference::open(&manifest.reference.path, &manifest.reference.sha256)?;
    let flow = S3GenFlowAssets::open(&manifest.flow.path, &manifest.flow.sha256)?;
    let prepared = flow.prepare(&reference)?;
    let mut state = WaveformStream::new(seed, 4800)?;
    let mut samples = 0usize;
    let mut updates = Vec::new();
    for (end, finalize) in prefixes {
        let _trace = crate::trace::iteration("waveform_prefix", end);
        check_cancelled(&mut cancelled)?;
        let began = Instant::now();
        let mut conditioning_spans = std::collections::BTreeMap::new();
        let (conditioned, conformer_compute) = {
            let input = prepared.assemble(&tokens[..end])?;
            check_cancelled(&mut cancelled)?;
            let (encoded, compute) = run_conformer(
                &manifest.conformer,
                input.embeddings(),
                &mut conditioning_spans,
                engine_cache,
            )?;
            check_cancelled(&mut cancelled)?;
            let mel = S3GenMelAssets::open(&manifest.mel.path, &manifest.mel.sha256)?;
            let (conditioned, _projection) =
                crate::gpu::run_from_env(|| mel.condition(&input, &encoded, finalize))?;
            (conditioned, compute)
        };
        let conditioning_ms = began.elapsed().as_millis();
        let generated_frames = conditioned.generated_frames;
        let mut meanflow_compute = None;
        let mut hift_compute = None;
        let mut meanflow_ms = 0;
        let mut hift_ms = 0;
        // Separate maps let the sequential closures borrow their own timing
        // state without changing the existing stage/model ownership boundary.
        let mut meanflow_spans = std::collections::BTreeMap::new();
        let mut hift_spans = std::collections::BTreeMap::new();
        let fragment = state.decode_with(
            &conditioned,
            |noise| {
                let began = Instant::now();
                let (mel, compute) = run_meanflow(
                    &manifest.meanflow,
                    &conditioned,
                    noise,
                    &mut meanflow_spans,
                    engine_cache,
                )?;
                meanflow_compute = Some(compute);
                meanflow_ms = began.elapsed().as_millis();
                Ok(mel.generated)
            },
            |mel, normal, phases, source| {
                let began = Instant::now();
                let (output, compute) = run_hift(
                    &manifest.hift,
                    mel,
                    normal,
                    phases,
                    source,
                    &mut hift_spans,
                    engine_cache,
                )?;
                hift_compute = Some(compute);
                hift_ms = began.elapsed().as_millis();
                Ok(output)
            },
            finalize,
            &mut cancelled,
        )?;
        if fragment.offset_samples != samples {
            return Err("Streaming waveform sample offset mismatch".into());
        }
        let count = fragment.pcm.values.len();
        // Withheld-only prefixes are valid state updates, not empty WAVs.
        if count != 0 {
            emit(fragment.pcm, finalize)?;
            samples += count;
        }
        conditioning_spans.extend(meanflow_spans);
        conditioning_spans.extend(hift_spans);
        conditioning_spans.extend([
            ("conditioning", conditioning_ms),
            ("meanflow", meanflow_ms),
            ("hift", hift_ms),
        ]);
        updates.push(serde_json::json!({"codec_tokens":end,"final":finalize,
            "generated_frames":generated_frames,"emitted_samples":count,"duration_ms":began.elapsed().as_millis(),
            "stages_ms":conditioning_spans,
            "waveform_compute":{"conformer":conformer_compute,"meanflow":meanflow_compute,"hift":hift_compute}}));
        if finalize
            && samples
                != generated_frames
                    .checked_mul(480)
                    .ok_or("Waveform length overflow")?
        {
            return Err("Final streaming waveform frame accounting mismatch".into());
        }
    }
    stages_ms.insert("waveform_prefixes", timer.elapsed().as_millis());
    trace.success();
    Ok(
        serde_json::json!({"stages_ms":stages_ms,"prefixes":updates,"reference_preparations":0,
        "token_engine_cache_hit":token_engine_cache_hit}),
    )
}
