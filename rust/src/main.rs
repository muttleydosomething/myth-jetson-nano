use chatterbox_rs::{
    conditioning::{prompt_digest, PromptAssets},
    generation,
    npy::Npy,
    sampling::Sampling,
    tensorrt::TensorRt,
    text,
    tokenizer::TokenizedText,
    Result,
};
use serde::Serialize;
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Serialize)]
struct Receipt {
    status: &'static str,
    started_unix_ms: u128,
    finished_unix_ms: u128,
    duration_ms: u128,
    seed: u64,
    reached_eos: bool,
    speech_tokens: Vec<u32>,
    decoder_tokens: Vec<i64>,
    prompt_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<TokenizedText>,
}
fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "import-workshop") {
        if args.len() != 6 {
            return Err("Usage: chatterbox-rs import-workshop WORKSHOP.json SHA256 INVENTORY.json SHA256 SEED".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        // Reuse the existing configuration and its process-lifetime state lock.
        // Offline import cannot race an active workshop or prepare a reference.
        let configuration_bytes = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
        let actual = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(&configuration_bytes))
        };
        if actual != args[2] {
            return Err("Workshop configuration checksum differs".into());
        }
        let configuration =
            serde_json::from_slice(&configuration_bytes).map_err(|e| e.to_string())?;
        let seed = args[5].parse().map_err(|_| "Invalid import seed")?;
        let workshop = chatterbox_rs::workshop::Workshop::open(configuration, seed)?;
        let result = workshop.import(Path::new(&args[3]), &args[4], seed)?;
        println!(
            "{}",
            serde_json::json!({"status":"workshop_imported", "result":result,
            "started_unix_ms":started_unix_ms,"finished_unix_ms":now_ms(),"duration_ms":timer.elapsed().as_millis()})
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "publish-voice") {
        if !(args.len() == 7 || (args.len() == 8 && args[7] == "--no-normalise")) {
            return Err("Usage: chatterbox-rs publish-voice MODELS.json SHA256 RECORDING PROMPT.json SHA256 DESTINATION [--no-normalise]".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let engine = chatterbox_rs::voice_preparation::VoicePreparation::open(
            Path::new(&args[1]),
            &args[2],
        )?;
        let voice = engine.prepare(Path::new(&args[3]), args.len() == 7)?;
        let published = voice.publish(Path::new(&args[4]), &args[5], Path::new(&args[6]))?;
        println!(
            "{}",
            serde_json::json!({
                "status":"voice_published", "started_unix_ms":started_unix_ms,
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "published":published, "reference":voice.reference.receipt(),
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "s3-analyze-reference") {
        if args.len() != 4 {
            return Err(
                "Usage: chatterbox-rs s3-analyze-reference ANALYSIS.json SHA256 MONO-24000HZ.npy"
                    .into(),
            );
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let analysis =
            chatterbox_rs::s3gen::S3ReferenceAnalysis::open(Path::new(&args[1]), &args[2])?;
        let audio = chatterbox_rs::npy::F32Npy::from_bytes(
            &std::fs::read(&args[3]).map_err(|e| e.to_string())?,
        )?;
        let pcm16 = analysis.resample(&audio)?;
        let mels = analysis.mels(&audio)?;
        println!(
            "{}",
            serde_json::json!({
                "status":"s3_reference_analyzed","started_unix_ms":started_unix_ms,
                "finished_unix_ms":now_ms(),"duration_ms":timer.elapsed().as_millis(),
                "source_sha256":analysis.source_sha256,
                "pcm16":pcm16.values,"mel_shape":mels.shape,"mels":mels.values,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "prepare-voice") {
        if !(args.len() == 4 || (args.len() == 5 && args[4] == "--no-normalise")) {
            return Err(
                "Usage: chatterbox-rs prepare-voice MODELS.json SHA256 RECORDING [--no-normalise]"
                    .into(),
            );
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let engine = chatterbox_rs::voice_preparation::VoicePreparation::open(
            Path::new(&args[1]),
            &args[2],
        )?;
        let voice = engine.prepare(Path::new(&args[3]), args.len() == 4)?;
        println!(
            "{}",
            serde_json::json!({
                "status":"voice_prepared", "started_unix_ms":started_unix_ms,
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "voice":voice.receipt(),
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "prepare-audio") {
        if !(args.len() == 2 || (args.len() == 3 && args[2] == "--no-normalise")) {
            return Err("Usage: chatterbox-rs prepare-audio RECORDING [--no-normalise]".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let engine = chatterbox_rs::audio_preparation::AudioPreparation::open()?;
        let audio = engine.prepare(Path::new(&args[1]), args.len() == 2)?;
        println!(
            "{}",
            serde_json::json!({
                "status":"audio_prepared", "started_unix_ms":started_unix_ms,
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "source_sample_rate":audio.source_sample_rate, "source_channels":audio.source_channels,
                "loudness_lufs":audio.loudness_lufs, "gain":audio.gain,
                "decoder_reference_samples":audio.decoder_reference().len(),
                "t3_reference_samples":audio.t3_reference().len(),
                "pcm24":audio.pcm24, "pcm16":audio.pcm16,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "campplus-encode") {
        if args.len() != 4 {
            return Err(
                "Usage: chatterbox-rs campplus-encode MODEL.json SHA256 MONO-16000HZ.npy".into(),
            );
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let model = chatterbox_rs::campplus::CampPlus::open(Path::new(&args[1]), &args[2])?;
        let audio = chatterbox_rs::npy::F32Npy::from_bytes(
            &std::fs::read(&args[3]).map_err(|e| e.to_string())?,
        )?;
        let features = model.features(&audio)?;
        let embedding = model.encode_features(&features)?;
        println!(
            "{}",
            serde_json::json!({
                "status":"campplus_encoded", "started_unix_ms":started_unix_ms,
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "source_sha256":model.source_sha256, "sample_rate":16000,
                "mel_frames":features.shape[0], "embedding":embedding.values,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "s3-tokenize") {
        if args.len() != 4 {
            return Err(
                "Usage: chatterbox-rs s3-tokenize MODEL.json SHA256 MONO-16000HZ.npy".into(),
            );
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let model = chatterbox_rs::s3_tokenizer::S3Tokenizer::open(Path::new(&args[1]), &args[2])?;
        let audio = chatterbox_rs::npy::F32Npy::from_bytes(
            &std::fs::read(&args[3]).map_err(|e| e.to_string())?,
        )?;
        let features = model.features(&audio)?;
        let tokens = model.encode_mels(&features)?;
        println!(
            "{}",
            serde_json::json!({
                "status":"s3_tokenized", "started_unix_ms":started_unix_ms,
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "source_sha256":model.source_sha256, "sample_rate":16000,
                "mel_frames":features.shape[2], "tokens":tokens,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "voice-encode") {
        if args.len() != 4 {
            return Err(
                "Usage: chatterbox-rs voice-encode MODEL.json SHA256 MONO-16000HZ.npy".into(),
            );
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let model =
            chatterbox_rs::voice_encoder::T3VoiceEncoder::open(Path::new(&args[1]), &args[2])?;
        let audio = chatterbox_rs::npy::F32Npy::from_bytes(
            &std::fs::read(&args[3]).map_err(|e| e.to_string())?,
        )?;
        let features = model.features(&audio)?;
        let result = model.encode_mels(&features.mels)?;
        println!(
            "{}",
            serde_json::json!({
                "status":"t3_speaker_encoded", "started_unix_ms":started_unix_ms,
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "source_sha256":model.source_sha256, "sample_rate":16000,
                "trim":features.trim, "mel_frames":features.mels.shape[0],
                "partials":result.partial_embeddings.shape[0],
                "embedding":result.embedding.values,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "serve") {
        if args.len() != 4 {
            return Err("Usage: chatterbox-rs serve SERVICE.json SHA256 LISTEN_IP:PORT".into());
        }
        let listen = args[3]
            .parse()
            .map_err(|e| format!("Native listen address: {e}"))?;
        return tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?
            .block_on(chatterbox_rs::service::serve(
                Path::new(&args[1]),
                &args[2],
                listen,
            ));
    }
    if args.first().is_some_and(|s| s == "synthesize") {
        if !(5..=8).contains(&args.len()) {
            return Err("Usage: chatterbox-rs synthesize PIPELINE.json SHA256 TEXT OUTPUT.wav [SEED] [MAX_GEN_LEN] [SAMPLING.json]".into());
        }
        let seed = args
            .get(5)
            .map(|s| s.parse::<u64>())
            .transpose()
            .map_err(|e| format!("Seed: {e}"))?
            .unwrap_or(42);
        let limit = args
            .get(6)
            .map(|s| s.parse::<usize>())
            .transpose()
            .map_err(|e| format!("Limit: {e}"))?
            .unwrap_or(1000);
        let sampling: Sampling = if let Some(path) = args.get(7) {
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?
        } else {
            Sampling::default()
        };
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let result = chatterbox_rs::pipeline::synthesize(
            Path::new(&args[1]),
            &args[2],
            &args[3],
            Path::new(&args[4]),
            seed,
            limit,
            &sampling,
        )?;
        println!(
            "{}",
            serde_json::json!({
                "status":"waveform_generated", "started_unix_ms":started_unix_ms,
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "sample_rate":24000, "reference_preparations":0,
                "trim_fade_applied":true, "watermark_applied":false,
                "reached_eos":true, "wav":args[4], "result":result,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "normalize") {
        if args.len() != 2 {
            return Err("Usage: chatterbox-rs normalize TEXT".into());
        }
        println!("{}", text::punc_norm(&args[1]));
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "s3-finish") {
        if args.len() != 2 {
            return Err("Usage: chatterbox-rs s3-finish RAW-WAVEFORM.npy".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let raw = chatterbox_rs::npy::F32Npy::from_bytes(
            &std::fs::read(&args[1]).map_err(|e| e.to_string())?,
        )?;
        let waveform = chatterbox_rs::hift::finish_waveform(&raw)?;
        println!(
            "{}",
            serde_json::json!({
                "status":"waveform_finished", "started_unix_ms":started_unix_ms,
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "sample_rate":24000, "reference_preparations":0,
                "trim_fade_applied":true, "watermark_applied":false,
                "waveform":{"shape":waveform.shape,"values":waveform.values},
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "s3-hift") {
        if args.len() != 6 {
            return Err(
                "Usage: chatterbox-rs s3-hift MODEL.json SHA256 MEL.npy NORMAL.npy PHASES.npy\n       chatterbox-rs s3-hift MODEL.json SHA256 MEL.npy --seed SEED"
                    .into(),
            );
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let read = |index: usize| -> Result<chatterbox_rs::npy::F32Npy> {
            chatterbox_rs::npy::F32Npy::from_bytes(
                &std::fs::read(&args[index]).map_err(|e| e.to_string())?,
            )
        };
        let mel = read(3)?;
        let random_seed: Option<u64> = if args[4] == "--seed" {
            Some(args[5].parse().map_err(|_| "Invalid random seed")?)
        } else {
            None
        };
        let (normal, phases) = if let Some(seed) = random_seed {
            let frames = *mel.shape.get(2).ok_or("Missing HiFT mel frame dimension")?;
            chatterbox_rs::sampling::s3gen_hift_random(frames, seed)?
        } else {
            (read(4)?, read(5)?)
        };
        let load = Instant::now();
        let model = chatterbox_rs::hift::S3GenHiFT::open(Path::new(&args[1]), &args[2])?;
        let model_load_ms = load.elapsed().as_millis();
        let execute = Instant::now();
        let (output, compute) =
            chatterbox_rs::gpu::run_from_env(|| model.synthesize(&mel, &normal, &phases))?;
        let execution_ms = execute.elapsed().as_millis();
        let tensor =
            |a: &chatterbox_rs::npy::F32Npy| serde_json::json!({"shape":a.shape,"values":a.values});
        println!(
            "{}",
            serde_json::json!({
                "status":"hift_waveform_generated", "started_unix_ms":started_unix_ms,
                "stages_ms":{"model_load":model_load_ms,"execution":execution_ms},
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "source_sha256":model.source_sha256, "reference_preparations":0,
                "random_seed":random_seed,
                "sample_rate":24000, "trim_fade_applied":false, "watermark_applied":false,
                "pitch":tensor(&output.pitch), "source":tensor(&output.source),
                "waveform":tensor(&output.waveform),
                "waveform_compute":compute,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "s3-meanflow") {
        let native_random = args.get(7).is_some_and(|s| s == "--seed");
        if args.len() != (if native_random { 11 } else { 10 }) {
            return Err("Usage: chatterbox-rs s3-meanflow MODEL.json SHA256 MU.npy MASK.npy SPKS.npy COND.npy NOISE.npy STEPS REFERENCE_FRAMES\n       chatterbox-rs s3-meanflow MODEL.json SHA256 MU.npy MASK.npy SPKS.npy COND.npy --seed SEED STEPS REFERENCE_FRAMES".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let offset = usize::from(native_random);
        let random_seed: Option<u64> = if native_random {
            Some(args[8].parse().map_err(|_| "Invalid random seed")?)
        } else {
            None
        };
        let steps: usize = args[8 + offset]
            .parse()
            .map_err(|_| "Invalid meanflow step count")?;
        let reference_frames: usize = args[9 + offset]
            .parse()
            .map_err(|_| "Invalid reference frame count")?;
        let read = |index: usize| -> Result<chatterbox_rs::npy::F32Npy> {
            chatterbox_rs::npy::F32Npy::from_bytes(
                &std::fs::read(&args[index]).map_err(|e| e.to_string())?,
            )
        };
        let mu = read(3)?;
        let frames = mu
            .shape
            .get(2)
            .copied()
            .ok_or("Missing meanflow frame dimension")?;
        let generated_frames = frames
            .checked_sub(reference_frames)
            .ok_or("Reference exceeds meanflow input")?;
        let input = chatterbox_rs::s3gen::S3GenMelInput {
            mu,
            mask: read(4)?,
            speaker: read(5)?,
            cond: read(6)?,
            reference_frames,
            generated_frames,
        };
        let noise = if let Some(seed) = random_seed {
            chatterbox_rs::sampling::s3gen_meanflow_noise(frames, seed)?
        } else {
            read(7)?
        };
        let load = Instant::now();
        let model = chatterbox_rs::meanflow::S3GenMeanflow::open(Path::new(&args[1]), &args[2])?;
        let model_load_ms = load.elapsed().as_millis();
        let execute = Instant::now();
        let (output, compute) = model.run(&input, &noise, steps)?;
        let execution_ms = execute.elapsed().as_millis();
        let tensor = |array: &chatterbox_rs::npy::F32Npy| serde_json::json!({"shape":array.shape,"values":array.values});
        println!(
            "{}",
            serde_json::json!({
                "status":"meanflow_mel_generated", "started_unix_ms":started_unix_ms,
                "stages_ms":{"model_load":model_load_ms,"execution":execution_ms},
                "finished_unix_ms":now_ms(), "duration_ms":timer.elapsed().as_millis(),
                "source_sha256":model.source_sha256, "reference_preparations":0,
                "random_seed":random_seed,
                "waveform_synthesis":false, "steps":steps, "reference_frames":reference_frames,
                "full":tensor(&output.full), "generated":tensor(&output.generated),
                "waveform_compute":compute,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "s3-encode") {
        if args.len() != 4 {
            return Err("Usage: chatterbox-rs s3-encode CONFORMER.json SHA256 INPUT.npy".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let model = chatterbox_rs::conformer::S3GenConformer::open(Path::new(&args[1]), &args[2])?;
        let input = chatterbox_rs::npy::F32Npy::from_bytes(
            &std::fs::read(&args[3]).map_err(|e| e.to_string())?,
        )?;
        let (output, compute) = chatterbox_rs::gpu::run_from_env(|| model.encode(&input))?;
        println!(
            "{}",
            serde_json::json!({
                "status": "conformer_encoded", "started_unix_ms": started_unix_ms,
                "finished_unix_ms": now_ms(), "duration_ms": timer.elapsed().as_millis(),
                "source_sha256": model.source_sha256, "reference_preparations": 0,
                "shape": output.features.shape, "features": output.features.values,
                "mask_shape": output.mask.shape, "mask": output.mask.values,
                "waveform_compute":compute,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "s3-mel") {
        if args.len() != 10 {
            return Err("Usage: chatterbox-rs s3-mel MEL.json SHA256 REFERENCE.json SHA256 FLOW.json SHA256 TOKENS.json ENCODED.npy FINALIZE(true|false)".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let finalize: bool = args[9]
            .parse()
            .map_err(|_| "FINALIZE must be true or false")?;
        let model = chatterbox_rs::s3gen::S3GenMelAssets::open(Path::new(&args[1]), &args[2])?;
        let reference = chatterbox_rs::s3gen::S3GenReference::open(Path::new(&args[3]), &args[4])?;
        let flow = chatterbox_rs::s3gen::S3GenFlowAssets::open(Path::new(&args[5]), &args[6])?;
        let tokens: Vec<i64> =
            serde_json::from_slice(&std::fs::read(&args[7]).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let features = chatterbox_rs::npy::F32Npy::from_bytes(
            &std::fs::read(&args[8]).map_err(|e| e.to_string())?,
        )?;
        let frames = features.shape.get(1).copied().unwrap_or(0);
        // Proof entry: portable full encoder features, with an unpadded mask.
        // Native library callers pass S3GenConformer::encode's actual output.
        let encoded = chatterbox_rs::conformer::EncoderOutput {
            features,
            mask: chatterbox_rs::npy::F32Npy {
                shape: vec![1, 1, frames],
                values: vec![1.0; frames],
            },
        };
        let prepared = flow.prepare(&reference)?;
        let input = prepared.assemble(&tokens)?;
        let output = model.condition(&input, &encoded, finalize)?;
        let tensor = |array: &chatterbox_rs::npy::F32Npy| serde_json::json!({"shape": array.shape, "values": array.values});
        println!(
            "{}",
            serde_json::json!({
                "status": "mel_conditioned", "started_unix_ms": started_unix_ms,
                "finished_unix_ms": now_ms(), "duration_ms": timer.elapsed().as_millis(),
                "reference_preparations": 0, "waveform_synthesis": false, "finalize": finalize,
                "mu": tensor(&output.mu), "cond": tensor(&output.cond),
                "mask": tensor(&output.mask), "speaker": tensor(&output.speaker),
                "reference_frames": output.reference_frames, "generated_frames": output.generated_frames,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "s3-reference") {
        if args.len() != 3 {
            return Err("Usage: chatterbox-rs s3-reference ASSETS.json SHA256".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let reference = chatterbox_rs::s3gen::S3GenReference::open(Path::new(&args[1]), &args[2])?;
        // Admission only. Report exact state without loading TensorRT/CUDA or
        // suggesting that unported waveform generation has occurred.
        println!(
            "{}",
            serde_json::json!({
                "status": "reference_loaded", "started_unix_ms": started_unix_ms,
                "finished_unix_ms": now_ms(), "duration_ms": timer.elapsed().as_millis(),
                "reference_preparations": 0, "reference": reference.receipt(),
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "s3-flow-input") {
        if args.len() != 6 {
            return Err("Usage: chatterbox-rs s3-flow-input REFERENCE.json SHA256 FLOW.json SHA256 TOKENS.json".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let reference = chatterbox_rs::s3gen::S3GenReference::open(Path::new(&args[1]), &args[2])?;
        let assets = chatterbox_rs::s3gen::S3GenFlowAssets::open(Path::new(&args[3]), &args[4])?;
        // This consumes codec IDs (e.g. the existing decoder_tokens receipt),
        // never raw decoder BOS/EOS. Assembly rejects IDs outside the codec.
        let tokens: Vec<i64> =
            serde_json::from_slice(&std::fs::read(&args[5]).map_err(|e| e.to_string())?)
                .map_err(|e| format!("S3Gen generated tokens: {e}"))?;
        let prepared = assets.prepare(&reference)?;
        let input = prepared.assemble(&tokens)?;
        println!(
            "{}",
            serde_json::json!({
                "status": "flow_input_assembled", "started_unix_ms": started_unix_ms,
                "finished_unix_ms": now_ms(), "duration_ms": timer.elapsed().as_millis(),
                "reference_preparations": 0, "flow_input": input.receipt(),
                "normalized_speaker": prepared.normalized_speaker().values,
                "speaker": prepared.speaker().values,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "tokenize") {
        if args.len() != 4 {
            return Err("Usage: chatterbox-rs tokenize TOKENIZER.json SHA256 TEXT".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let tokenizer =
            chatterbox_rs::tokenizer::TurboTokenizer::open(Path::new(&args[1]), &args[2])?;
        let output = tokenizer.encode(&args[3])?;
        println!(
            "{}",
            serde_json::json!({
                "status": "tokenized",
                "started_unix_ms": started_unix_ms,
                "finished_unix_ms": now_ms(),
                "duration_ms": timer.elapsed().as_millis(),
                "text": output,
            })
        );
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "prompt") {
        if args.len() != 4 {
            return Err("Usage: chatterbox-rs prompt ASSETS.json SHA256 TEXT".into());
        }
        let started_unix_ms = now_ms();
        let timer = Instant::now();
        let assets = PromptAssets::open(Path::new(&args[1]), &args[2])?;
        let text = assets.tokenize(&args[3])?;
        let prompt = assets.assemble(&text)?;
        println!(
            "{}",
            serde_json::json!({
                "status": "assembled", "started_unix_ms": started_unix_ms,
                "finished_unix_ms": now_ms(), "duration_ms": timer.elapsed().as_millis(),
                "text": text, "shape": prompt.shape, "prompt_sha256": prompt_digest(&prompt),
            })
        );
        return Ok(());
    }
    let fresh_text = args.first().is_some_and(|s| s == "text-tokens");
    let seed_index = if fresh_text { 7 } else { 6 };
    if (!fresh_text && args.first().is_none_or(|s| s != "tokens"))
        || args.len() < seed_index
        || args.len() > seed_index + 3
    {
        return Err("Usage: chatterbox-rs tokens BRIDGE ENGINE SHA256 PROMPT.npy SPEECH-EMBEDDING.npy [SEED] [MAX_GEN_LEN] [SAMPLING.json]\n       chatterbox-rs text-tokens BRIDGE ENGINE SHA256 ASSETS.json ASSETS_SHA256 TEXT [SEED] [MAX_GEN_LEN] [SAMPLING.json]\nThese commands produce speech tokens. Use synthesize for retained-voice WAV output.".into());
    }
    let seed = args
        .get(seed_index)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("Seed: {e}"))?
        .unwrap_or(42);
    let limit = args
        .get(seed_index + 1)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("Limit: {e}"))?
        .unwrap_or(1000usize);
    let sampling: Sampling = if let Some(path) = args.get(seed_index + 2) {
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        Sampling::default()
    };
    sampling.validate()?;
    let started_unix_ms = now_ms();
    let timer = Instant::now();
    // Both entry routes use the same decoder and generation loop. Fresh text
    // reaches profile admission before any native code or engine is loaded.
    let (prompt, embeddings, tokenized) = if fresh_text {
        let assets = PromptAssets::open(Path::new(&args[4]), &args[5])?;
        let text = assets.tokenize(&args[6])?;
        let prompt = assets.assemble(&text)?;
        (prompt, assets.into_speech_embeddings(), Some(text))
    } else {
        (
            Npy::read(Path::new(&args[4]))?,
            Npy::read(Path::new(&args[5]))?,
            None,
        )
    };
    let prompt_sha256 = prompt_digest(&prompt);
    let mut decoder = TensorRt::open(
        Path::new(&args[1]),
        Path::new(&args[2]),
        &args[3],
        prompt,
        embeddings,
    )?;
    let output = generation::generate(&mut decoder, &sampling, limit, seed, || false)?;
    let decoder_tokens =
        text::finish_speech_tokens(&output.tokens.iter().map(|&n| n as i64).collect::<Vec<_>>());
    let receipt = Receipt {
        status: if output.reached_eos {
            "eos"
        } else {
            "length_or_mask_stop"
        },
        started_unix_ms,
        finished_unix_ms: now_ms(),
        duration_ms: timer.elapsed().as_millis(),
        seed,
        reached_eos: output.reached_eos,
        speech_tokens: output.tokens,
        decoder_tokens,
        prompt_sha256,
        text: tokenized,
    };
    println!(
        "{}",
        serde_json::to_string(&receipt).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    if let Err(message) = run() {
        eprintln!("chatterbox-rs: {message}");
        std::process::exit(1);
    }
}
