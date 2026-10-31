# Chatterbox Turbo in Rust

This is the native Chatterbox Turbo conversion for Capricorn 15. It assembles
arbitrary text into a TensorRT prompt using exported embedding tables and either
an **already projected voice prefix** or **retained speaker/reference tensors**.
Rust projects the latter once when loading a voice. Both fresh-text and captured-prompt commands
feed the existing speech-token generator. The offline `synthesize` command joins
those stages through Conformer, meanflow, HiFT and final trim/fade into a WAV.
The current Python speech service remains the operational engine.

New recordings can also be prepared into complete native T3/S3Gen conditional
tensors through the offline/library route described below. Production reference
uploads and workshop Apply now use this native preparation and publication.
The operational Python synthesizer consumes the resulting conditioning tensors;
replacement of that service remains separate.

Rust owns punctuation normalisation, FP16/FP32 NumPy capture inputs, sampling
(temperature, top-k, top-p, repetition), generation limits, token cleanup,
request reset, checksums and receipts. A small C ABI bridge calls TensorRT's
C++ interface and CUDA; it owns engine/context lifetime and K/V buffers.
Neither the Rust executable nor this bridge invokes Python or PyTorch.
Text tokenisation uses Hugging Face's native Rust core, pinned to the deployed
0.22.2 version. It loads a checksummed JSON export of the existing GPT-2 vocabulary,
merge rules and voice tags; it never downloads assets or loads speech weights.

This component lives beside the existing Borg source because it replaces that
runtime boundary. Its independent Cargo workspace permits building for Borg
without compiling the Hailstorm gateway.

## Build and proof

### Native T3 speaker encoder

`T3VoiceEncoder` owns the first of the three new-reference neural branches:
mono FP32 PCM already at 16 kHz to the 256-value T3 speaker embedding. Its
checksummed `t3_voice_encoder_fp32` manifest admits the exact three-layer LSTM
and projection tensors plus the deployed 40-channel Slaney mel basis. Training
similarity parameters are excluded. Inference uses the existing native matrix
kernel; checkpoint tables remain immutable and recurrent state is request-owned.

`features` applies the deployed default 20 dB relative-RMS trim, reflect-centered
400-sample periodic-Hann STFT, 160-sample hop and unscaled power mels. The fixed
Fourier tables are retained with the model. `encode_mels` uses 160-frame partials,
77-frame steps (the default rate 1.3), minimum coverage 0.8, zero padding, final
ReLU and both partial/utterance L2 normalizations. Partials run sequentially to
bound recurrent workspace. Empty, malformed, negative/nonfinite mel input,
overflowing analysis/projection and zero/nonfinite embeddings are rejected.
There is no arbitrary utterance-length cap.

The proof CLI is:

```sh
chatterbox-rs voice-encode MODEL.json SHA256 MONO-16000HZ.npy
```

It returns embedding values, trim bounds, mel/partial counts, checkpoint checksum
and start/finish/duration. The NumPy input contract is mono `[samples]` FP32;
the caller must supply the declared 16 kHz rate. This is not a new-voice upload
endpoint. S3 codec tokenization and CAMPPlus x-vectors are separate native branches
described below. Audio decoding/resampling, loudness normalization and assembling
a complete native voice remain separate gates.

`tests/capture_voice_encoder_oracle.py` loads only `ve.safetensors` and executes
the actual deployed `embeds_from_wavs`, partial inference and mel analysis against
seven new synthetic PCM fixtures: one sample, silence, a short chirp, a complete
partial, both sides of the coverage boundary and padded audio requiring trimming.
Run with CPU-only `python3 -S` in a separate 1536 MiB, network-disabled container;
no retained service voice is prepared again. Set `CHATTERBOX_VOICE_ENCODER_ASSETS`
and run `--test voice_encoder -- --ignored`. The independent mel gate permits
absolute/relative error 2e-5; embeddings permit absolute error 3e-5 plus relative
error 3e-4. Exact native library/CLI identity and retained-model replay are required.
Synthetic numerical parity does not establish fresh-recording voice quality.

The 30 October 2026 proof passed all seven deployed-method comparisons on Saturn,
exact library/CLI identity and retained-model replay. A coverage/empty-output unit
regression and the asset-admission regression also passed. Admission covers schema,
model kind, source digest, missing/extra tensors, shape, checksum, local path and
nonfinite array rejection. The ARM executable then passed all seven embedding
comparisons in a separate 1536 MiB, network-disabled scratch container on Borg.
Maximum ARM embedding absolute error was 2.7567148208618164e-7; complete CLI calls
took 255–727 ms for these small fixtures. These are encoder timings, not synthesis
or production latency. ARM replay was exact and a wrong manifest digest failed.

Runtime source is `edc0e6d7`; subsequent candidate changes add admission tests and
records only. Binary SHA256 is
`7ccd0749407dacba399a5acaa0805e7b609ce609635411e19422cea6df2f6f37`.
Evidence is under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-voice-encoder-20261030/`.
Borg retains `chatterbox-voice-encoder-rs`, `voice-encoder-assets` and
`voice-encoder-receipts` in its existing native proof directory. Neither the native
HTTP trial nor operational speech service was replaced. All retained voice
conditionals remain untouched; no fresh-recording or audible-quality claim is made.

### Native S3 speech-codec encoder

`S3Tokenizer` supplies the second new-reference branch: one mono FP32 PCM
utterance already at 16 kHz to S3 v2/25 Hz codec IDs (0..6560). Its checksummed
`s3_tokenizer_v2_25hz_fp32` manifest admits the exact two strided convolutions,
six 1280-wide rotary/FSMN transformer blocks, eight-value FSQ projection and
deployed mel/window/rotary analysis tables. Weights remain immutable and all
intermediate state belongs to the request. The port reuses the existing native
matrix, layer-normalization and exact-erf GELU kernels.

The proof CLI is:

```sh
chatterbox-rs s3-tokenize MODEL.json SHA256 MONO-16000HZ.npy
```

Receipts retain codec IDs, mel-frame count, source checksum and
start/finish/duration. Reflect-centered 400-sample STFT with 160-sample hop,
last-frame removal, 128 Slaney power mels, the logarithmic floor and 80 dB clamp
match the deployed analysis. Torch's reflect-padding precondition requires more
than 200 PCM samples; malformed, nonfinite and overflowing inputs fail explicitly.
Audio is not silently padded to a new token length. Decoding, resampling and
loudness normalization belong to the surrounding preparation pipeline.

Long audio retains the deployed 3000-frame windows, 2600-frame steps and
50-token half-overlap trim. Segments run sequentially. Masked tail rows are
omitted from native computation; upstream masks them before both convolutions,
attention keys and FSMN values, so valid outputs remain equivalent. The
independent long-input comparison exercises the actual padded upstream batch.
This bounds attention workspace without imposing an utterance-length cap.

`tests/capture_s3_tokenizer_oracle.py` executes the deployed wrapper/encoder/FSQ
against seven new synthetic PCM fixtures: minimum length, both sides of a token
hop, the exact hop, silence, a chirp and 3001 mel frames crossing the long-input
boundary. Only checkpoint `tokenizer.*` tensors are loaded into a separate
1536 MiB, CPU-only, network-disabled container. No retained voice is prepared.
Set `CHATTERBOX_S3_TOKENIZER_ASSETS` and run `--test s3_tokenizer -- --ignored`.
Hidden states and FSQ logits permit absolute/relative error 3e-4; every final
codec ID must match exactly. Native library/CLI identity and immutable-model
replay after intervening lengths are also required.

The initial mel gate (2e-5 absolute/relative) rejected the long fixture. Diagnosis
showed that a float64 FFT of the same FP32-windowed PCM differs from the deployed
FP32 FFT by 5.340576171875e-5 after log scaling in quiet bins. Explicit FP32
windowing/FFT reproduces the deployed STFT exactly. The corrected mel-only gate
is 8e-5 absolute plus 2e-5 relative; hidden/logit tolerances and the exact-token
gate are unchanged. The native maximum mel error was 5.340576e-5. This is an
explicit numerical limit, not universal bit parity for intermediate features.

On 30 October 2026 all seven comparisons passed on Saturn and with the ARM
executable on Borg, including 751 exact tokens for the long fixture. Schema,
kind, source digest, missing/extra tensors, shape, checksum, local-path and
nonfinite-asset regressions passed; malformed/short/nonfinite/overflowing audio
and malformed mels were rejected. Read-only source review found no blocking
defect. The wrong manifest digest also failed at Borg's real CLI boundary.

Runtime source is `af33d2e1`; the ARM build uses `a4fc050b`, which adds tests,
license attribution and a comment correction. Binary SHA256 is
`3e994813208932b39048ade03d7d5c227a560c69131845ce755b02c24b50c6ad`.
Borg CLI calls took 4205–20277 ms including admission of the checkpoint tables;
these CPU measurements do not establish GPU or production latency. Each ran
in a 1536 MiB scratch container without Python or GPU access. Host Python only
checked receipts. Evidence, binary, exported assets and receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-s3-tokenizer-20261030/`.
Borg retains `chatterbox-s3-tokenizer-rs`, `s3-tokenizer-assets` and
`s3-tokenizer-receipts` in the existing native proof directory.

The production service and native HTTP trial were not replaced. Complete
fresh-recording preparation still needs audio preparation and joining all reference
tensors. Fresh-recording voice quality remains unverified.

### Native CAMPPlus speaker encoder

`CampPlus` supplies the third new-reference neural branch: one mono FP32 PCM
utterance already at 16 kHz to an unnormalised 192-value S3Gen speaker x-vector.
Its checksummed `campplus_192_fp32` manifest admits the exact FCM convolutions,
evaluation BatchNorm tables, TDNN, 52 context-attentive dense layers, transitions,
statistics pool and final projection, plus the deployed Kaldi mel/window tables.
Training counters are excluded. The existing native matrix kernel is reused;
weights remain immutable and all utterance state belongs to the request.

The proof CLI is:

```sh
chatterbox-rs campplus-encode MODEL.json SHA256 MONO-16000HZ.npy
```

Receipts retain the vector, feature-frame count, source checksum and
start/finish/duration. Kaldi analysis uses snipped 400-sample frames, 160-sample
hop, frame DC removal, 0.97 preemphasis with first-sample replication, the
nonperiodic Povey window, zero-padded 512-point power spectrum, 80 mel filters,
FP32-epsilon log floor and utterance column-mean removal. It deliberately does
not share either other encoder's different feature semantics. The caller supplies
the declared mono 16 kHz rate; decoding, resampling and loudness normalization
are not performed by this command.

FCM strides downsample frequency only, preserving channel/frequency flatten order.
TDNN halves the temporal length by ceiling. CAM context sums the global mean
and per-100-frame mean; a short final segment uses its actual length. Its two
pointwise gate layers run once per segment. Final statistics use sample standard
deviation with denominator T-1 and no floor; the final BatchNorm has no affine
parameters or ReLU. S3Gen owns downstream vector normalization. At least three
feature frames (720 PCM samples) are required for a finite speaker vector.
Malformed/short/nonfinite/overflowing input and invalid checkpoint tables fail
explicitly, including overflow before a nonlinearity could conceal it. Tiled
2D-convolution workspace avoids an utterance-length cap.

`tests/capture_campplus_oracle.py` executes the deployed `inference` and feature
methods against seven new synthetic PCM fixtures: minimum finite length,
silence, chirp, and TDNN lengths 99, 100, 101 and 201 around segment-pool
boundaries. Only checkpoint `speaker_encoder.*` tensors are loaded in a separate
1536 MiB, CPU-only, network-disabled container. No retained voice is prepared.
Set `CHATTERBOX_CAMPPLUS_ASSETS` and run `--test campplus -- --ignored`.
Speaker vectors from both independent features and native PCM features permit
absolute/relative error 3e-4. Native library/CLI identity and retained-model replay
after different utterance lengths must be exact.

The initial feature gate (8e-5 absolute plus 2e-5 relative) rejected the chirp.
An independent diagnostic reproduced deployed features exactly, then measured
up to 2.193450927734375e-4 from FP32 mean-reduction and FFT roundoff in quiet
log-feature bins. The corrected feature-only bound is 3e-4 absolute plus 2e-5
relative; the speaker-vector bound is unchanged. This is numerical tolerance,
not universal bit parity with PyTorch intermediates.

On 30 October 2026 all seven comparisons passed on Saturn and with the ARM
executable on Borg. Maximum PCM-to-vector absolute error on both was
2.874135971069336e-4. The pooling/statistics regression, malformed and extreme
input checks, schema/kind/source/shape/checksum/path/nonfinite/negative-variance
asset-admission checks passed. Read-only source review found no remaining blocking
defect. Borg replay was exact and a wrong manifest checksum failed at the CLI.

Runtime source is `4f758c0f`; the ARM build uses `cf6e6ffa`, adding measured-gate
correction and license attribution only. Binary SHA256 is
`e1bbedb52426baf612420be18bb035cf7dd83507456b6449c07746a0759761de`.
Borg CLI calls took 267–775 ms including checkpoint admission, in separate
1536 MiB, network-disabled scratch containers without Python or GPU access.
Host Python only checked receipts. These are CPU encoder timings, not synthesis
or production latency. Evidence, assets, executable and receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-campplus-20261030/`.
Borg retains `chatterbox-campplus-rs`, `campplus-assets` and `campplus-receipts`
in the existing native proof directory. The production speech service and native
HTTP trial were not replaced. Complete new-voice preparation still needs audio
preparation and joining the encoder outputs; real-recording voice quality remains
unverified.

### Native reference-audio preparation

`AudioPreparation` supplies the audio stage before the neural encoders. The
Rust process calls the installed `libsndfile.so.1` and `libsoxr.so.0` C ABIs
through the existing native-library loader. Decoding and HQ resampling therefore
retain native library dependencies, with no Python interpreter or subprocess.
Codecs are those supported by the installed libsndfile; librosa's legacy
audioread fallback is not implemented. Source paths must resolve to regular
recordings; nonblocking open followed by descriptor metadata validation prevents
a FIFO from hanging before rejection.

The decoder mixes FP32 channels before resampling to 24 kHz, preserves librosa's
ceil/final-zero-pad length convention and enforces Turbo's strict longer-than-five
seconds contract. Rust implements default pyloudnorm K-weighting, 400 ms/75%
overlap absolute/relative loudness gates and the -27 LUFS gain, including the
FP32 round after each FP64 biquad stage. Silence and below-gate input remain
unchanged. Nonfinite input and arithmetic overflow fail explicitly; no peak
limiter or arbitrary recording-duration cap is introduced.

Normalization uses the full recording before any cut. `pcm24` and `pcm16` retain
the complete normalized 24 kHz and librosa/SoXR-resampled 16 kHz recordings.
`decoder_reference()` borrows the first ten seconds at 24 kHz;
`t3_reference()` borrows the first fifteen seconds at 16 kHz. T3's speaker
encoder consumes the full 16 kHz recording. S3Gen's separate TorchAudio
24-to-16 kHz resampling and 24 kHz reference-mel analysis remain part of the
next assembly boundary; they must not silently consume T3's differently
resampled copy.

The proof CLI returns both complete sample arrays plus source rate/channel count,
LUFS/gain, reference-cut lengths and start/finish/duration:

```sh
chatterbox-rs prepare-audio RECORDING [--no-normalise]
```

`tests/capture_audio_preparation_oracle.py` executes the deployed
`norm_loudness` method and librosa functions without constructing a speech model.
Its eight new synthetic recordings cover mono/stereo, FP32/PCM16 WAV, PCM24 FLAC,
16/22.05/24/32/44.1/48 kHz source rates, silence, below-gate input, disabled
normalization and a 17-second gated recording crossing both reference cuts.
Run in a separate CPU-only, network-disabled 1536 MiB container. Set
`CHATTERBOX_AUDIO_PREPARATION_ASSETS` and run the `audio_preparation` tests
with ignored tests included. Sample bounds are 2e-6 absolute plus 2e-5 relative;
finite loudness permits 1e-5 LUFS. Replay and CLI/library identity must be exact.
Admission checks cover empty/invalid/short/exactly-five-second/nonfinite inputs,
invalid rates, overflow and FIFO rejection. No retained voice is prepared.

On 30 October 2026 all eight comparisons passed on Saturn and with the ARM
executable on Borg. Maximum sample absolute error was 1.1920929e-7 on Saturn
and 1.7881393432617188e-7 on Borg, with the original gates unchanged. Exact
CLI/library identity, replay after different recordings, malformed/short/nonfinite
input rejection and FIFO rejection passed. Borg preparation calls took 15–73 ms,
excluding CLI sample serialization, container startup and all neural encoders.
The initial Borg harness FIFO assertion saw permission denial because its
host-owned mode-0600 fixture was unreadable by cap-dropped container UID0.
After correcting only that fixture to mode 0444, only its admission was repeated;
the passing numerical runs were retained.

Runtime source is `9d5ac3ce`; later candidate revisions contain attribution and
records only. Binary SHA256 is
`c9645abe7ac7baf1d16cf31bba3e7b0cddd8d4c70cac4e8356c83ce566d2960f`.
Evidence, assets, executable and receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-audio-preparation-20261030/`.
Borg retains `chatterbox-audio-preparation-rs`, `audio-preparation-assets` and
`audio-preparation-receipts` in its existing native proof directory. This is an
offline audio-stage proof; complete voice assembly, production speech replacement
and real-recording audible quality remain unverified.

### Native complete reference preparation

`VoicePreparation` joins the existing decoder, normalization and three neural
branches into the deployed Turbo conditional tensors. The aggregate manifest
pins the existing voice-encoder, codec, CAMPPlus and S3 analysis manifests.
Neural models load sequentially, keeping only their small resulting tensors;
an ordinary speech request never prepares an existing voice again.

The complete normalized SoXR 16 kHz recording feeds T3's speaker encoder.
Only its first fifteen seconds feed T3's codec (at most 375 IDs). Separately,
the first ten seconds at 24 kHz feed S3Gen's own Hann-sinc 24-to-16 kHz
resampler, codec and CAMPPlus. These resampled arrays are deliberately distinct.
S3 reference mels use FFT/window 1920, hop 480, explicit reflect padding 720,
periodic Hann, magnitude sqrt(real²+imag²+1e-9), the deployed 80-channel Slaney
basis spanning 0–8000 Hz, and natural log with a 1e-5 floor. This is neither
other encoder's power-mel analysis.

The existing `S3GenReference::from_parts` admission boundary validates the
assembled reference. Only codec IDs are shortened to floor(mel_frames/2) on
a mismatch; the complete mel array, including an odd final frame, survives.
Feature length remains absent, matching deployed `embed_ref`. T3's speaker
remains FP32 until `PreparedVoice::condition_prompt` explicitly casts for the
existing FP16 projection. That method reuses `PromptAssets` and
`project_reference`; the S3 reference feeds the existing flow admission.
These library objects are in-memory conditionals, not a new persisted voice
format, upload route or replacement speech endpoint.

Proof commands:

```sh
chatterbox-rs s3-analyze-reference ANALYSIS.json SHA256 MONO-24000HZ.npy
chatterbox-rs prepare-voice MODELS.json SHA256 RECORDING [--no-normalise]
```

The first returns resampled PCM and time-major mel values. The second returns
both complete encoder branches and the existing reference tensor checksums.
Both include start/finish/duration. Manifests are checksummed before encoder
execution. No interpreter, subprocess, model download or retained-voice rewrite
is part of native preparation.

`tests/capture_voice_preparation_oracle.py` executes the actual deployed
individual encoders in a separate CPU-only, network-disabled 1536 MiB
container. The actual `prepare_conditionals` and `embed_ref` methods then run
against captured neural outputs with exact PCM input assertions, avoiding a
full speech-model allocation. Six small analysis fixtures cover reflect/hop
boundaries, impulse and silence; three complete synthetic recordings cover
ordinary preparation, an odd mel tail and both recording cuts. This proves
method composition against independent encoder outputs, not audible quality.

Set `CHATTERBOX_VOICE_PREPARATION_ASSETS`, `CHATTERBOX_VOICE_PROMPT_ASSETS`
and `CHATTERBOX_VOICE_FLOW_ASSETS` for the `voice_preparation` tests. Resampling
uses the previous sample gate, 2e-6 absolute plus 2e-5 relative. T3 embeddings
use 3e-5 absolute plus 3e-4 relative; CAMPPlus retains 3e-4 absolute/relative.
Codec IDs must match exactly. Native library/CLI identity and replay after
other recordings must also be exact.

The first log-mel gate (3e-5 absolute plus 2e-5 relative) rejected quiet bins.
An independent reconstruction reproduced deployed FP32 FFT mels exactly on
all nine fixtures. A float64 FFT of the same FP32 windowed samples matched
native analysis within 9.5367431640625e-7 log units, while both differed from
deployed FP32 FFT by up to 8.974075317382812e-4 near the 1e-5 mel floor.
The measured mel-only bound is therefore 1e-3 absolute plus 2e-5 relative;
sample, embedding and exact-token gates are unchanged. This is numerical
agreement rather than bit identity with PyTorch's FFT intermediates.

On 30 October 2026 all six analysis fixtures and all three complete voice
comparisons passed on Saturn and with the ARM executable on Borg. Codec IDs
were exact: ordinary voice 150/150 T3/S3 IDs and 300 mel frames; odd-tail voice
151/150 IDs and 301 frames; long voice 375/250 IDs and 500 frames. Maximum
mel error was 8.974075317382812e-4; maximum CAMPPlus vector error was
1.9121170043945312e-4. Native prompt/flow connection, same-configuration replay,
FP32 CLI/library identity and six existing S3-reference admission regressions
passed. Borg separately verified receipt tensor hashes, exact CLI replay and
wrong manifest digests, short/exact-five-second/nonfinite/absent input rejection.

The initial CLI identity assertion compared JSON parser FP64 tails rather
than the declared FP32 tensors. Its 609 differing decimal values all decoded
to identical FP32 bits. The corrected test compares decoded tensors and exact
reference checksums; no numerical runtime correction was needed.

The ARM build used `77edefcc`; later changes alter records and a crate-level
comment only. Binary SHA256 is
`ae08a8cd856860543ea4c937a4f837ed6f76d7390074c9907c00184f15194192`.
Borg complete-voice commands took 13,676–20,945 ms including checkpoint
admission in separate 1536 MiB scratch containers without Python or GPU access.
Host Python checked receipts only. These are CPU preparation measurements,
not speech latency or production performance. Evidence, capture, independent
FFT diagnosis, executable and receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-voice-assembly-20261030/`.
Borg retains `chatterbox-voice-preparation-rs`, `voice-preparation-assets`
and `voice-preparation-receipts` in the established native proof area.

The native preparation tensors are proven. The publication boundary below
connects them to the existing synthesis formats. Production uploads, the
workshop contract and real-recording audible quality remain separate.

### Native voice publication

`PreparedVoice::publish` consumes an already encoded voice and a pinned
version-2 `prepared_turbo` prompt template. The command joins preparation
and publication for one new recording:

```sh
chatterbox-rs publish-voice MODELS.json MODELS_SHA256 RECORDING \
  PROMPT_TEMPLATE.json PROMPT_SHA256 DESTINATION [--no-normalise]
```

The destination's parent must already exist. Publication emits the existing
`prompt.json` and `reference.json` inference formats, plus their local
checksummed arrays. The template supplies only the tokenizer, text/speech
tables and speaker projection. Its retained speaker and tokens are never
copied or prepared again. New FP32 speaker/mel tensors retain their exact bits;
T3's existing reader performs the established FP16 projection at admission.

The aggregate receipt returns absolute prompt/reference paths and SHA256
pins, reference tensor checksums, and start/finish/duration. Set an existing
pipeline's `prompt` and `reference` entries to these returned pins, then admit
that pipeline through the existing service catalog. No new synthesis format,
HTTP upload endpoint, automatic catalog mutation or production route is added.
Shared tables are copied from admitted bytes because the existing prompt
format requires local assets; publication therefore produces a self-contained
voice without introducing mutable shared links.

Both manifests are re-admitted with the normal inference readers in a private
sibling staging directory. After all output and directory flushes succeed,
Linux `renameat2(RENAME_NOREPLACE)` makes the entire voice visible at once.
Concurrent publishers cannot replace an existing voice, including an empty
destination or symbolic link. The parent is then flushed. A parent-flush
failure explicitly reports that the destination is already visible; no
success receipt is issued. Earlier failures retain the named staging area for
operator inspection and leave the final destination absent. This entry point
performs no deletion or overwrite.

`tests/voice_publication.rs`, with `CHATTERBOX_VOICE_PROMPT_ASSETS` pointing
to the existing exported prompt tables, exercises exact reference round-trip,
new prompt conditioning, replay, duplicate/concurrent publication, malformed
tensors, FP16 overflow and bad template pins. The Borg harness in
`tests/verify_voice_publication_borg.py` prepares a synthetic recording,
compares its published tensors with the previous native receipt, and consumes
that publication through an isolated instance of the existing native HTTP
service. These are compatibility gates; they do not establish audible quality
or production latency.

On 30 October 2026 the focused host regression passed, including the concurrent
publication race and existing-empty-destination rejection. The ARM executable
then prepared and published the existing six-second synthetic tone fixture on
Borg. Its published FP32 reference tensors and codec IDs matched the earlier
native preparation receipt exactly. The existing HTTP catalog listed the new
voice and two native requests produced the same 39,360-sample WAV byte for byte,
SHA256 `1e8eff192afc0cdebdc8d64b4235654e20acab30bd779a62843027939b83de34`.
The existing-destination and wrong-model-pin failures reported their expected
causes. No retained voice was prepared, and synthesis consumed only the
published conditioning.

Runtime source is `9b30b786`; the ARM build began at `c32de2b0`. Subsequent
changes affect the acceptance harness and records only. Binary SHA256 is
`ae0ff7e5a7a48cebfc55fb4900c7b459aef1fad070d3c6aa27775e135fe53170`.
Preparation/publication took 16,714 ms; HTTP requests took 28,831 and 27,791 ms,
including model admission and CPU waveform work. These remain isolated trial
measurements. The task-owned scratch trial listened only on Borg's private
`10.200.0.2:8767`, with a 1536 MiB ceiling and no swap; it was stopped after
acceptance. The production container retained its running state and original
start time `2026-10-30T12:06:46.911067898Z`.

Evidence, executable, claims and copied receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-voice-publication-20261030/`.
Borg retains the executable, harness and `voice-publication-20261030` outputs in
the established native proof area. The publication boundary has been exercised;
production upload/workshop integration, service replacement, GPU waveform
performance, streaming and real-recording audible quality remain unverified.

### Production reference-upload integration: active and accepted

Candidate `5e4fa1ac`, integrated and pushed at `1a76216b`, connects the existing
`/upload_predefined_voice` handler to `patches/native_reference.py`. This is the
shared entry point for Hailstorm reference uploads and the Myth workshop's Apply
action. Rust prepares and publishes both normalization variants before the public
recording is admitted. Borg's pinned libsndfile cannot decode MP3, so the existing
native FFmpeg executable first produces a floating-point WAV for that format;
the original uploaded bytes remain the recording and source checksum.

The binding, manifests and tensor assets are checked before the production
adapter constructs the existing Torch `Conditionals`. Replay of an admitted
upload does not execute an encoder. The generation lock also owns preparation,
and the existing authorised deletion patch reclaims the upload's native assets.
Interrupted publication after binding commitment recovers the recording without
preparing it again. Retained legacy voices keep their established path.

The focused adapter/conditioning suite passed 26 checks; the deletion endpoint
passed three. A real Borg MP3 transaction passed both normalization variants,
exact tensor admission, original-byte preservation and encoder-free replay. Its
native child peak was 587,452 KiB. The receipt and supervisor claim are under
`myth-native-voice-integration-20261030/mp3-native-admission-run` in the existing
host engineering evidence area.

Production activation was attempted with image `myth:native-reference-1a76216b`
and a bounded 6 GiB service allowance. The model became locally ready, but the
real gateway upload returned HTTP 502 while Borg completed native publication.
An encoder-free replay also failed. Remote GETs subsequently failed while local
Borg requests succeeded. Borg's TCP diagnostics showed 2,532 bytes unacknowledged
and retransmissions; this establishes a transport failure, not its cause. The
browser workshop and production synthesis gates were not reached successfully.

The preserved prior speech implementation is the recovery authority at
`/srv/borg-recovery/myth-native-reference-20261030` on Borg. Recovery restored its
source, conditioning wrapper and 5 GiB limit, and disabled native admission. The
restored model became locally ready; the remote catalogue returned HTTP 200 with
2,259 bytes and the public Myth homepage returned HTTP 200 with 525,132 bytes.
This is recovery evidence, not proof of which change caused the transport fault.

On 30 October 2026 the exact integrated release was reactivated after the host
resource heartbeat and source/checksum checks, retaining that recovery authority
and the bounded 6 GiB service allowance. The live acceptance gates then passed:

- Hailstorm's `/api/borg/voices` accepted a fresh MP3 recording, preserving its
  original bytes and publishing both normalization variants natively.
- The real Myth workshop accepted a fresh WAV through browser Apply. Draft
  selection sent no upload; Apply sent exactly one, closed the workshop and
  selected the admitted voice. No browser errors occurred. Model-info remained
  responsive during preparation.
- The production speech endpoint returned validated WAVs using both new voices,
  including reuse of the MP3 voice. The three measured requests took 3,341,
  3,266 and 3,401 ms. Production operations recorded
  `native_reference_loaded ... reference_preparations=0`; generation did not
  prepare either recording again.
- The existing Hailstorm `/api/borg/synthesize` Task Route returned HTTP 200 and
  a validated WAV using Watson's configured voice. Duplicate MP3 upload also
  passed. No agent voice assignment or Task Route was changed for this proof.

Simultaneous TCP-header captures at Saturn and Borg contained no repeated payload
segments during the resumed upload/Apply run. The earlier transport failure did
not recur; its cause remains unverified and no network configuration was changed.
The acceptance harness initially misused Node fetch's `ok` property in two
assertions. Both failed runs were claimed and retained; only their remaining
gates were resumed after correction, without repeating successful upload or
browser proof. The final gateway/replay run was also claimed.

Native upload admission remains active. Recordings, prepared tensors, claims,
deployment, recovery and acceptance evidence are retained under the existing
`myth-native-voice-integration-20261030` engineering evidence directory. The final
receipt is `acceptance-resumed-final.json`; browser evidence is
`workshop-resumed-applied.png`, and production reuse evidence is
`production-native-acceptance-verified.log`. This proves production native
reference integration with synthetic recordings. Audible voice quality,
streaming, GPU waveform performance and replacement of the Python synthesizer
remain unverified.

### Existing synthesis proof

Within a governed Hailstorm candidate, run Cargo through
`scripts/hailstorm-engineering-target.sh` with the candidate provenance:

```sh
scripts/hailstorm-engineering-target.sh run --provenance /path/to/candidate.json -- \
  cargo test --manifest-path borg/chatterbox/rust/Cargo.toml --locked
```

Build the native bridge on the TensorRT/CUDA host (Borg):

```sh
bash borg/chatterbox/rust/native/build.sh /absolute/path/libchatterbox_trt.so
```

Tests include 41 punctuation cases obtained by executing the deployed Python
function independently, eight authentic Hugging Face FP32 processor comparisons,
and replay, cancellation, failure, EOS, length and array-contract checks.
See `source-provenance.json` for the pinned source hashes and remaining work.

The tokenizer regression uses 79 comparisons captured from Borg's actual
Transformers wrapper: empty input, Unicode, control characters, every added voice
tag, adjacent tags and the 1024-token boundary. To exercise that test, set
`CHATTERBOX_TOKENIZER_JSON` to the pinned exported JSON and append
`-- --include-ignored` to the Cargo test command. Without that asset this external
parity test is explicitly ignored; the checksum/vocabulary rejection tests still run.

`tests/capture_tokenizer_oracle.py` captures the JSON export and oracle receipt
from the deployed model directory. Invoke it as `python3 -S` inside the existing
speech container with `CUDA_VISIBLE_DEVICES=''`, `USE_TORCH=0`, `USE_TF=0` and
`USE_FLAX=0`. The startup hook is excluded. Transformers 5.1 still imports CPU
Torch; the script asserts that Chatterbox and a CUDA context remain unloaded.
The receipt's `tokenizer_json` string is the native asset, while its
`tokenizer_sha256` admits the exact bytes. Speech weights are never exported.

Conditioning proof uses 32 Python oracle cases: 24 retained speech captures and
eight further cases covering fresh text, empty input, voice tags, Unicode,
profile overflow and truncation. Admitted prompts must match every FP16 byte;
oversized requests must fail before engine loading. Synthetic checks cover the
exact 512/513-frame boundary, invalid IDs, malformed arrays and nonfinite values.
Set `CHATTERBOX_CONDITIONING_ASSETS` to the exported `assets.json` and include
ignored tests to exercise the numerical and CLI comparisons.

`tests/capture_conditioning_oracle.py` takes the checkpoint, retained-capture
directory, native tokenizer JSON and a new output directory. Run it with the same
CPU-only `python3 -S` setup as the tokenizer probe. It maps only the text/speech
embedding tables and extracts the prepared prefix from the retained prompt. It
executes the deployed `prepare_input_embeds` method without constructing a speech
model, reproduces the 24 captured prompts exactly, and prepares no reference audio.
The output manifest pins every local asset by SHA256; keep weights outside Git.
The version-1 projected prefix already includes speaker projection and reference
speech embeddings. Version 2 (`prepared_turbo`) instead admits a retained FP16
speaker tensor `[1,256]`, FP16 projection weight `[1024,256]` and bias `[1024]`,
and JSON reference codec tokens (0..375 values, each below 6561). Every asset is
checksummed. Rust applies the linear projection, rounds its output to FP16, then
appends speech-table rows in reference order. The loaded `PromptAssets` retains
that prefix across text requests. Versions are exclusive: a bundle cannot mix
prepared tensors and an existing projected prefix.

Supply the retained `conditionals.pt` as an optional fifth argument to the existing
oracle exporter to produce the portable version-2 bundles and `prepared-oracle.json`.
Only this CPU proof/export tool reads a Torch archive with `weights_only=True`;
the Rust runtime reads portable arrays and JSON and never prepares audio. The
export carries only T3 conditioning; S3Gen's separate reference state is outside
this slice. New-voice audio encoding remains unported.

The prepared-reference regression independently executes the deployed
`T3CondEnc.forward` with Torch's linear layer. Its six tensor probes cover the
retained voice, no reference tokens, reversed tokens, the 375-token maximum,
zero speaker and a changed speaker. Set `CHATTERBOX_PREPARED_ASSETS` to the export
directory and run the `deployed_prepared_reference_parity` test with `--ignored`.
It checks exact retained-voice prompts, reuse across texts and the CLI manifest
route. Synthetic speaker projections permit at most one FP16 step; reference,
text and BOS rows remain byte-exact. The scalar projection uses a wide accumulator
and one final FP16 rounding. The initial strict test exposed one subnormal value
in the changed-speaker probe where Torch's reduction rounded one FP16 step higher;
that result is retained rather than presented as universal bit parity.

S3Gen's separate prepared state loads through `S3GenReference`. Its checksummed
version-1 `s3gen_prepared` manifest names JSON codec IDs, their exact length,
FP32 C-order NumPy mel features and the independent 192-value speaker x-vector.
`prompt_feat_len` remains null or an explicit exact length. This single-voice,
unpadded contract admits 2N or 2N+1 mel frames for N codec tokens: the retained
voice has 201 tokens and 403 frames, including its original odd final frame.
The loader preserves every FP32 bit, rejects malformed/nonfinite/mismatched
assets and retains immutable state across requests without reopening assets.
It shares NumPy envelope validation with T3 while keeping the two float
representations separate. Reference admission does not project the x-vector.

`S3GenFlowAssets` separately loads checksummed FP32 codec embedding rows
`[6561,512]`, speaker affine weight `[80,192]` and bias `[80]`. Its version-1
`s3gen_encoder_input` manifest pins the source checkpoint and each local asset.
`prepare` borrows the retained reference and assets, normalises the 192-value
x-vector with FP32 L2/epsilon 1e-12, and projects it once to 80 values. `assemble`
then joins reference codec IDs followed by generated codec IDs, copies their
embedding rows exactly and supplies the FP32 `[1,T,1]` validity mask and integral
total length. This boundary admits a single unpadded voice/request. Empty or
out-of-range generated IDs and incompatible/nonfinite weights are rejected.
The retained mel features remain borrowed and unchanged for later flow stages.
The native Conformer and post-encoder conditioning described below consume these
embeddings. The native meanflow and HiFT boundaries are described below.
The assembly therefore accepts codec IDs
such as the existing `decoder_tokens` receipt, without silently removing specials
or appending silence a second time.

`tests/capture_s3gen_reference_oracle.py` exports only the retained archive's
`gen` state, using the same CPU-only `python3 -S` route. It executes the deployed
`S3Token2Mel.forward` against a recording flow stub, never constructing a model
or invoking `embed_ref`. Set `CHATTERBOX_S3GEN_ASSETS` to its export directory
and run the `s3gen_reference_deployed_handoff_parity` test with `--ignored`.
The authentic state and an even-mel/explicit-length NumPy-input probe match
the native library and CLI exactly. Python casts reference IDs and lengths to
its model float dtype before flow; native state keeps integers integral. The
oracle checks exact numerical values through that cast, not identical dtypes.
The initial exporter retained Fortran strides from the transposed mel tensor
and was correctly rejected. Its corrected export uses C order without changing
values; no native-loader correction was needed.

Supply the Turbo `s3gen_meanflow.safetensors` as the oracle exporter's optional
third argument to export the three pre-Conformer tables. An optional fourth
argument names a previous native generation receipt whose `decoder_tokens`
become the authentic flow probe. The exporter executes the deployed
`CausalMaskedDiffWithXvec.inference` and `make_pad_mask`, intercepting the encoder
call before any Conformer computation. Its six cases cover retained/reordered
generated tokens, codec limits, a single token, zero speaker and sub-epsilon
speaker. Set `CHATTERBOX_S3GEN_FLOW_ASSETS` to that directory and run
`s3gen_flow_deployed_encoder_boundary_parity` with `--ignored`. IDs, lengths,
mask and embedding rows must match every byte. Speaker normalisation/projection
permit absolute error 2e-6 plus relative error 1e-5 because native and Torch
reductions differ. Run this CPU probe in a separate memory-bounded container,
with no GPU/network and the model volume read-only; never inside the live speech
container. Checkpoint hashing streams bounded blocks.

`S3GenConformer` loads a checksummed version-1 `s3gen_upsample_conformer_fp32`
manifest. Admission requires the exact pinned tensor names, FP32 shapes, finite
values and local asset checksums. It retains immutable weights across requests.
Its `encode` method consumes the existing flow input's `[1,T,512]` embeddings:
linear embedding, ESPnet relative positions, three-frame pre-lookahead, six
Conformer blocks, nearest-neighbour twofold upsampling with causal convolution,
another linear embedding, four blocks and the final norm. Output is FP32
`[1,2T,512]` plus a FP32 `[1,1,2T]` validity mask representing the deployed
encoder's boolean mask values; mask dtype parity is not claimed. This is evaluation-mode,
full-attention, single-request inference; padding, streaming/chunked attention,
dropout and training are not supported. No arbitrary utterance cap is introduced.
The implementation uses the native Rust `matrixmultiply` CPU kernel. GPU
acceleration and production latency remain unproven. This boundary performs no
post-encoder trimming, mel projection, reference preparation or synthesis itself.

`tests/capture_conformer_oracle.py` imports only the deployed encoder modules
through inert package shells and loads only `flow.encoder.*` checkpoint weights.
Run it with CPU-only `python3 -S` in a separate memory-bounded, network-disabled
container, passing the meanflow checkpoint, previous flow-input oracle directory
and a new export directory. It never constructs the tokenizer, speaker encoder,
decoder or service. It exports the deployed FP32 positional frequencies and
four oracle cases: authentic retained/generated codec embeddings, reversed
generated IDs, one frame and two frames. Set `CHATTERBOX_CONFORMER_ASSETS` to
that directory and run `conformer_replay_deployed_parity` with `--ignored`.
Library/CLI outputs must agree exactly; deployed outputs permit absolute error
2e-4 plus relative error 2e-4. Replay checks reuse after intervening lengths.
Unit regressions independently check the relative shift, Conv1d layout, causal
padding, affine orientation and incomplete/corrupt/wrong-shaped asset rejection.
The port retains the original CosyVoice/ESPnet Apache-2.0 terms in `LICENSE`.

`S3GenMelAssets` admits the checksummed FP32 encoder projection `[80,512]`
and bias `[80]` from the same checkpoint as the existing flow assets. Its
`condition` method consumes `S3GenFlowInput` and full `EncoderOutput`, using the
existing Conformer matrix kernel. Final requests retain all encoded frames;
non-final requests remove six frames before projection. Output `mu` and `cond`
are contiguous `[1,80,F]`: conditioning copies every retained reference-mel bit,
including the odd final frame, then supplies zeros for generated frames. Speaker
projection is borrowed from the prepared voice and copied into the request output.
Reference/generated frame counts identify the eventual decoder-output crop.
Incomplete, padded, nonfinite or mismatched encoder outputs and non-final inputs
too short for the retained reference are rejected before projection.

The deployed flow computes its decoder mask from the original encoder mask,
without applying the non-final six-frame trim. Native conditioning deliberately
preserves that longer `[1,1,2T]` mask. This boundary establishes decoder-call
parity; it does not establish streaming decoder execution or waveform synthesis.
All weights and voice state remain immutable across requests.

`tests/capture_mel_oracle.py` executes the deployed flow's actual `inference`
method through a recording decoder, with retained reference arrays and the
previously proven encoder outputs. Run with CPU-only `python3 -S` in a separate
1536 MiB, network-disabled container. Arguments are the meanflow checkpoint,
previous flow-input export, previous Conformer export and a new output directory.
It constructs no Conformer or waveform decoder and prepares no reference audio.
Seven probes cover authentic/reordered outputs, final/non-final trimming, odd
and even reference lengths and a one-frame generated tail. Shorter probes slice
captured encoder features to isolate this boundary; they do not claim full
Conformer recomputation for those token prefixes. Set `CHATTERBOX_MEL_ASSETS`
to the export and run `mel_conditioning_replay_deployed_parity` with `--ignored`.
Projection/speaker comparisons permit absolute error 2e-6 plus relative error
1e-5; conditioning and mask values match every FP32 bit. Native library/CLI
outputs agree exactly. Replay retains the same voice, speaker projection and
mel weights through intervening shorter inputs. Synthetic admission checks
reject corrupt, missing, wrong-shaped, nonfinite and path-invalid mel assets.

`S3GenMeanflow` loads the 911 deployed FP32 estimator tensors plus exported
sinusoid frequencies. It runs the actual single-width causal-convolution network:
one down block, twelve middle blocks and one up block, each with four
full-context attention blocks. Time embeddings mix both Euler endpoints through
the learned checkpoint matrix. Inference uses linear Euler steps, exact erf GELU,
and no classifier-free guidance or cosine schedule. The existing Conformer matrix
kernel supplies the shared native affine/attention multiplication boundary.

`decode` accepts the existing `S3GenMelInput`, explicit FP32 initial noise of
the same shape, and a positive step count. Noise includes the reference segment;
it is never replaced with the reference conditioning. The returned full mel
includes reference frames; the generated mel removes exactly `reference_frames`.
Weights are retained immutably, and all intermediates belong to the request.
This is a final, unpadded, single-request CPU boundary. A longer non-final mask
is rejected explicitly: the deployed estimator itself fails on the earlier
six-frame feature/mask mismatch. No streaming repair is included.

`tests/capture_meanflow_oracle.py` imports the actual deployed estimator and CFM
methods in a separate 1536 MiB CPU-only, network-disabled container, constructs
the estimator on the meta device, and assigns only its checkpoint tensors. It
uses previously captured retained-voice conditioning and fixed noise to execute
the real meanflow forward/Euler path. It prepares no reference audio and loads
no Conformer, HiFT or speech service. Arguments are the checkpoint, previous
mel-conditioning export and a new output directory. Five probes cover retained
and reversed generated tokens, one/two/odd frame lengths and one/two/three Euler
steps. Set `CHATTERBOX_MEANFLOW_ASSETS` to that export and run
`meanflow_replay_deployed_parity` with `--ignored` in the release profile.

Per-step derivatives, full mel and generated crop permit absolute error 3e-4
plus relative error 3e-4. Retained-model replay and CLI/library outputs agree
exactly. Focused admission regressions cover malformed/missing/extra tensor
names, wrong shapes, bad digests, nonfinite values, invalid paths, inconsistent
frame accounting, padded/mismatched masks and invalid steps/times. Initial noise
is supplied by the caller; native Gaussian sampling, Torch random-sequence parity,
GPU acceleration and production latency are not established by the meanflow proof.

`S3GenHiFT` admits a checksummed version-1 `s3gen_hift_fp32` manifest with 246
effective FP32 tensors and the periodic Hann window. The independent exporter
reads the deployed Torch parametrizations once, including dimension-zero weight
normalisation for transposed convolutions; the native runtime loads only portable
NumPy arrays, retaining all weights immutably across requests.

`synthesize` takes generated mel `[1,80,F]`, caller-supplied standard-normal
excitation `[1,9,480F]` and harmonic phases in radians `[1,9,1]`. It predicts pitch
with the five-layer Conv/ELU network, repeats pitch 480 times per mel frame,
accumulates nine harmonic phases and combines voiced/unvoiced noise through the
learned source projection. The fundamental's supplied phase is ignored, matching
the deployed generator. `decode` separately accepts explicit source `[1,1,480F]`.
The vocoder combines source STFT with transposed-convolution rates 8/5/3 and
Snake residual blocks, then produces a centred inverse STFT waveform `[1,480F]`
at 24 kHz, clamped to ±0.99. Tiled convolutions reuse the existing native matrix
kernel without an utterance-sized im2col workspace. The fixed 16-point transforms
need no additional FFT dependency. Spectrum and waveform outputs reject nonfinite
results, including overflow from otherwise finite input.

This is the raw, final, single-request CPU HiFT boundary. It does not apply
S3Token2Wav's separate 960-sample trim/fade, streaming source caching, Perth
watermarking or service transport. Native Gaussian/phase supply, Torch random
sequence parity, GPU acceleration, listening assessment and production latency
remain unverified. No retained voice is prepared again.

`tests/capture_hift_oracle.py` loads only the deployed HiFT/F0 modules and their
checkpoint tensors in a separate 1536 MiB CPU-only, network-disabled container.
Arguments are the meanflow checkpoint, previous meanflow oracle export and a new
output directory. It executes actual deployed `HiFTGenerator.inference` with
captured random values and exports effective weights, pitch, excitation, source
STFT and waveform. Six cases cover retained/reversed generated mel, one/two/odd
frames and zero mel; two source probes cover voiced threshold and phase
accumulation. Set `CHATTERBOX_HIFT_ASSETS` and run
`hift_replay_deployed_parity -- --ignored` in the release profile. Pitch, source,
STFT and waveform permit absolute error 3e-4 plus relative error 3e-4. Native
CLI/library identity and retained-model replay are exact. Focused checks reject
malformed assets, unsupported manifests, missing/extra tensors, invalid shapes,
nonfinite arrays, bad checksums, path traversal and incompatible randomness.

The optional `s3gen_hift_trt_fp32` manifest selects the complete resident HiFT
graph through that same `synthesize`/`synthesize_cached` boundary. It pins one
engine and the existing native bridge, plus an explicit frame-profile ceiling.
`native/export_hift.py` consumes admitted effective weights without reference
preparation and clears TensorRT's TF32 flag. Complex STFT export is replaced by
real DFT16 matrix multiplication, explicit reflection and four-frame overlap-add
with the Hann-squared envelope. The initial convolution-based transform lowering
failed numerical acceptance; the matrix graph preserves the existing tolerance.

The service's existing `EngineCache` retains one HiFT graph/context alongside
token and meanflow owners. Complete and incremental speech share this dispatcher.
Warm admission checks the same canonical-path/inode/size/mtime/ctime identities;
changed dependencies are checksummed again, and any inference error retires the
owner. Unload releases it. Random inputs and exact sample-prefix replacement
remain request-owned. Internal buffers stay on the device; only inputs and final
pitch/source/waveform cross the native ABI. Direct pitch, source-spectrum and
decode helpers remain tensor-model interfaces and explicitly reject graph models.

`hift_graph_lifecycle` isolates its fake bridge in a child process to prove reuse,
source invalidation, failure retirement and unload without rebinding another
test's process bridge. `hift_graph_deployed_parity -- --ignored` uses
`CHATTERBOX_HIFT_GRAPH`, `CHATTERBOX_HIFT_ASSETS` and
`CHATTERBOX_HIFT_CACHE_ASSETS` for the existing six oracle cases, exact cached
prefixes, five warm fixed-input replays, profile-capacity execution and subsequent
small-request replay. Its absolute/relative tolerance remains 3e-4/3e-4.

## Run

`hift::finish_waveform` applies the separate S3Token2Wav final-output boundary
once to a raw FP32 mono waveform `[1,N]` at 24 kHz. Its retained immutable window
silences the first 480 samples and fades the following 480 with the deployed
cosine ramp. It removes no samples and preserves every later sample bit. Inputs
shorter than 960 samples, incomplete arrays, nonfinite samples and batches are
rejected before processing; the deployed wrapper also rejects short inputs.
The raw HiFT interface remains available separately. This boundary has no model
weights, voice state, reference preparation, randomness or service transport.

`tests/capture_output_oracle.py` executes the deployed S3Token2Wav constructor's
actual fade-buffer expression and its `inference` output operation, with inert
flow/HiFT boundaries fed previously captured raw waveforms. It imports no
Chatterbox package or weights. Run it with CPU-only `python3 -S` in a separate
memory-bounded, network-disabled container, passing the prior HiFT oracle export
and a fresh output directory. Seven probes cover two authentic waveforms, exact
window length, odd lengths, zero-mel audio, unit amplitude and alternating polarity;
two short-input probes capture deployed rejection. Set `CHATTERBOX_OUTPUT_ASSETS`
and run `output_finish_deployed_wrapper_parity -- --ignored`; the FP32 absolute
error gate is 3e-7. Native replay and CLI/library results must agree exactly.

Perth watermarking is deliberately omitted by Holmes's decision on 2026-10-30.
It is not a required remaining conversion stage. Licence notices remain intact.

```sh
chatterbox-rs normalize 'ready, Holmes'
chatterbox-rs tokenize /path/tokenizer.json TOKENIZER_SHA256 'ready, Holmes'
chatterbox-rs prompt /path/assets.json ASSETS_SHA256 'ready, Holmes'
chatterbox-rs s3-reference /path/s3gen.json ASSETS_SHA256
chatterbox-rs s3-flow-input /path/s3gen.json REFERENCE_SHA256 \
  /path/flow.json FLOW_SHA256 /path/generated-codec-ids.json
chatterbox-rs s3-encode /path/conformer.json CONFORMER_SHA256 /path/encoder-input.npy
chatterbox-rs s3-mel /path/mel.json MEL_SHA256 /path/s3gen.json REFERENCE_SHA256 \
  /path/flow.json FLOW_SHA256 /path/generated-codec-ids.json /path/encoded.npy true
chatterbox-rs s3-meanflow /path/meanflow.json MEANFLOW_SHA256 /path/mu.npy \
  /path/mask.npy /path/spks.npy /path/cond.npy /path/noise.npy 2 REFERENCE_FRAMES
chatterbox-rs s3-hift /path/hift.json HIFT_SHA256 /path/generated-mel.npy \
  /path/standard-normal.npy /path/harmonic-phases.npy
chatterbox-rs s3-finish /path/raw-waveform.npy
chatterbox-rs text-tokens /path/libchatterbox_trt.so /path/decode.engine \
  ENGINE_SHA256 /path/assets.json ASSETS_SHA256 'ready, Holmes' 42 1000
chatterbox-rs tokens /path/libchatterbox_trt.so /path/decode.engine \
  ENGINE_SHA256 /path/prompt-0.npy /path/speech-embedding.npy 42 1000
```

Only trusted local libraries and engines may be loaded. The engine checksum is
required. Prompt shape is [1,1..512,1024], embedding table [6563,1024].
The existing TensorRT past-length profile is enforced rather than silently
changing engine or truncating a request.

An optional final argument supplies a JSON sampling configuration with
`temperature`, `top_k`, `top_p`, and `repetition_penalty`.
Output is JSON with speech/decoder tokens, termination reason, seed, start/finish
times and duration. A length or all-masked stop is explicitly distinguished from
EOS. Rust's PRNG is reproducible but does not reproduce PyTorch's CUDA sequence
from the same seed.

`tokenize` returns normalized text, input IDs, attention mask, original token
count and a truncation flag, plus start/finish times and duration. Like the
deployed single-request wrapper it adds no BOS/EOS or padding and truncates on the
right at 1024 text tokens. This limit is distinct from TensorRT's 512-embedding
prompt profile; tokenisation alone does not establish prompt admission or speech.

`prompt` reports the assembled shape and checksum without loading TensorRT.
`s3-reference` reports shapes, original dtypes and tensor-byte checksums of the
prepared S3Gen state, plus start/finish times and duration. It runs without
Python, Torch, CUDA or an engine; it admits reference state only, not waveforms.
`s3-flow-input` reports exact encoder-input tensor checksums and numerical
normalised/projected speaker values, with start/finish times and duration. Its
status is assembly only; it executes no Conformer or waveform generator.
`s3-encode` returns encoded feature values, shape, validity mask, checkpoint
checksum, start/finish times and duration. This proof entry reads portable FP32
embeddings; library callers use `S3GenFlowInput::embeddings()` directly. Neither
route invokes Python, Torch or CUDA. No waveform is produced.
`s3-mel` returns post-encoder `mu`, reference conditioning, the original validity
mask, projected speaker, frame counts and request timing. It reads a portable
full encoder output for proof; library callers pass `S3GenConformer::encode`'s
output directly. It produces no waveform and invokes no Python, Torch or CUDA.
`text-tokens` connects that assembly to the same decoder as `tokens`, retaining
the normalized text and prompt checksum in its receipt. It rejects text truncation
and combined prompts over 512 frames. `PromptAssets` can be retained across
requests; the CLI loads the export once per invocation. Native assembly and CLI
admission are checked on Saturn. The release ARM executable was also exercised
on Borg on 2026-10-30: two fresh inputs (including a voice tag) generated 66 and
58 speech tokens and reached EOS. A third input, `Ready, Holmes.`, generated
28 tokens; both the fresh-text and retained-capture entry routes returned exactly
the same speech and decoder tokens. All three prompt checksums matched the
independent Python assembly oracle. No reference preparation was repeated.

This proof used deployed sampling defaults (temperature 0.8, top-k 1000,
top-p 0.95, repetition penalty 1.2) and Rust seed 42. An initial test forced
top-k 1 and repeated tokens until the 600-step limit without EOS; that failure is
retained and greedy termination is not claimed. The runtime source was unchanged
between the two attempts. The standard-library Python harness only launched the
ARM executable and checked JSON receipts; speech inference used Rust, TensorRT
and CUDA without Python or Torch in its process.

The checked source is commit `6ec150718e7d0a4de9cfceafc3c8b606e7ee864f`.
The ARM binary SHA256 is
`c0ca5356597bdaebca6685ceb199ce47915c1c460a31e7f86245f40661a2e788`.
Retained receipts and the proof harness are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-rust-borg-proof-20261030/`
on Saturn; the successful summary is
`fresh-text-default-proof-20261030/result.json`. Borg retains the same receipts
under `/home/muttleydosomething/chatterbox-rust-build-20261030/`.
The executable is a proof artifact; it has not replaced the speech service.

Prepared-reference loading was exercised on Borg on 2026-10-30 with the retained
unprojected speaker and 202 reference codec tokens. Rust produced the same prompt
bytes and seeded speech/decoder tokens as the previously proven projected-prefix
route: `ready, Holmes` returned 28 speech tokens and the arbitrary-text comparison
returned 66. Both reached EOS. No reference preparation was repeated. On Saturn,
five of the six deployed-method tensor probes were byte-exact; the changed-speaker
probe differed in one speaker value by one FP16 step, within the explicit numerical
gate. All remaining prompt bytes and the CLI/library route comparisons passed.

The first Borg request took 82,616 ms and the second 10,723 ms. The difference was
not profiled; this proof establishes loading/projection and token parity, not a
latency improvement. The ARM release binary SHA256 is
`c4c93103b78b1e72efc4b9af6409ceff55e1ea292e7f335e4508729b9c2e9498`.
Its runtime source is commit `bcfd4d2c` (subsequent changes are tests and records).
Retained assets, independent oracle, initial strict-comparison failure, corrected
comparison and Borg receipts are referenced by
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-prepared-reference-20261030/verification.json`.
Borg retains the inference receipts under
`/home/muttleydosomething/chatterbox-rust-build-20261030/prepared-reference-proof-20261030/`.
The portable T3 loader has not replaced the live Python service and does not load
S3Gen's separate reference state or synthesize waveforms.

The separate S3Gen loader was exercised with the ARM release executable on Borg
on 2026-10-30. Both the original 403-frame retained voice and the 402-frame
explicit-length probe matched every reference tensor checksum and loaded in
1 ms each. Invalid manifest checksums were rejected. The independent deployed
method, native library/CLI comparisons, FP32 fidelity, malformed-state and
immutable-reuse regressions passed; no reference audio was prepared.
The ARM binary SHA256 is
`3e5e413880bea77c57ffb3cfc09384a30e8d6b38e88bac0fd72be81e0a0be6c0`.
Its runtime behaviour is from commit `cde8d7c1`; subsequent changes correct the
proof export's array order and update records/library documentation.
Evidence, original Fortran-order export, corrected assets and Borg receipts are
referenced by
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-s3gen-reference-20261030/verification.json`.
Borg retains the executable and receipts under
`/home/muttleydosomething/chatterbox-rust-build-20261030/s3gen-reference-proof-20261030/`.
This admits retained reference state only. No waveform generator or live-service
integration was exercised or deployed.

Pre-Conformer flow-input assembly was exercised with the ARM release executable
on Borg on 2026-10-30. All six oracle cases matched codec IDs, total length, mask
and encoder embedding bytes exactly. Normalised speaker values were exact for
these probes; the largest affine-projection error was 8.940696716308594e-08,
within the explicit FP32 gate. Full CLI load/assembly took 101–103 ms per case;
this is not waveform or end-user speech latency. Invalid codec IDs were rejected.
No reference audio preparation, Conformer or waveform generation occurred.
The binary SHA256 is
`5bf76d58e22ea8c0a6990033579bd42e0dafc2f0436b1f107647080e27701fbf`;
runtime source is commit `e0400d37`. Later commits change proof tooling/tests
and records only. Evidence is referenced by
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-s3gen-flow-input-20261030/verification.json`.
Borg retains `flow-input-assets`, `flow-input-receipts`, `verify_flow_borg.py`
and `chatterbox-s3gen-flow-rs` under
`/home/muttleydosomething/chatterbox-rust-build-20261030/`.

The initial CPU oracle ran inside the live speech container and exited 137;
Docker recorded that container as OOM-killed at 02:30:05 UTC. Unbounded checkpoint
hashing was the likely cause. The existing container was started without changing
its configuration, and its model endpoint confirmed Turbo loaded on CUDA at
02:32:06 UTC. The corrected exporter streams the checkpoint hash and passed in
a separate 1.5 GiB CPU-only container. The Borg acceptance also confirmed the live
model remained loaded after that proof. No service artifact was replaced.

The native FP32 Conformer was exercised with the ARM release executable on Borg
on 2026-10-30. All four independently captured cases passed: authentic retained
voice plus generated tokens, reordered generated tokens, one frame and two frames.
The two authentic inputs produced `[1,540,512]` outputs and matching doubled
validity-mask shapes/values (native FP32, deployed boolean). Maximum absolute
output error was 1.6689300537109375e-6, within the
predeclared numerical gate. The native library/CLI comparison and immutable-weight
replay after different lengths passed on Saturn; shape, checksum, tensor-name,
path and nonfinite-input rejection regressions also passed. No reference audio
was prepared. The read-only review found no blocking runtime defect.
The ARM binary SHA256 is
`c817c08cc4097695c4348ecc6eb6c00d12b159d351e5a320709e86315ef7f109`;
runtime source is commit `2d931383`. Subsequent changes update records only.
Borg's full CLI load plus CPU encode took 3318/3324 ms for the authentic inputs
and 1194/1195 ms for the tiny inputs. These include weight admission; they do not
establish retained-model, GPU or end-user speech latency. The native process uses
Rust and the CPU matrix kernel; the standard-library Python harness only checks
its output. No waveform or live-service integration was exercised or deployed.
Evidence, exported weights, oracle, retained binary and Borg receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-conformer-20261030/`.
Borg retains `conformer-assets`, `conformer-receipts`, `chatterbox-conformer-rs`
and `verify_conformer_borg.py` under
`/home/muttleydosomething/chatterbox-rust-build-20261030/`.
The independent Torch oracle passed in an isolated 1.5 GiB CPU-only container
with the model volume read-only, no GPU and no network; it never entered the live
speech container. The unrelated Cargo cache-cleanup permission warning remained
visible; compilation and focused proofs succeeded without an adjacent repair.

Native post-encoder conditioning was exercised with the ARM release executable
on Borg on 2026-10-30. All seven independently captured decoder-call cases
passed. Maximum projection error was 1.1920928955078125e-7; maximum speaker
error was 8.940696716308594e-8. Retained conditioning and masks matched every
FP32 bit. CLI/library equivalence, same-voice replay after intervening lengths,
malformed-asset and invalid-encoder checks passed on Saturn. Borg also rejected
an incorrect manifest checksum and a non-final request too short for the
reference. No reference audio was prepared and no waveform was synthesized.

Runtime source is commit `0e3293e1`; subsequent changes update records only.
The ARM binary SHA256 is
`23d452c5685966b0f289c61463de13890534a23d7c1c8b627c73bf4bbf310bb6`.
Borg CLI load plus conditioning took 118–121 ms per probe, including loading
the existing embedding and voice assets. This does not establish retained-model,
GPU or end-user speech latency. The standard-library Python harness only
compares receipts; the native process invokes no Python or Torch.
Evidence, assets, binary and Borg receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-mel-conditioning-20261030/`.
Borg retains `mel-assets`, `mel-receipts`, `chatterbox-mel-rs` and
`verify_mel_borg.py` under
`/home/muttleydosomething/chatterbox-rust-build-20261030/`.
The live speech service has not been replaced. GPU acceleration, streaming
decoder execution, waveform synthesis and audible parity remain unverified.

Native meanflow was exercised with the ARM release executable on Borg on
2026-10-30. All five independently captured cases passed: retained/reversed
authentic conditioning, one frame, two frames and an odd five-frame input, using
one/two/three Euler steps. The largest full-mel absolute error was
1.52587890625e-5. Both authentic inputs returned `[1,80,540]` full mels and
`[1,80,137]` generated mels after the retained 403-frame reference crop.
Invalid manifest checksums and mismatched masks were rejected at the real CLI.
On Saturn, per-step derivatives, malformed-asset/request checks, exact CLI/library
identity and exact retained-model replay after different lengths also passed.
The read-only source review found no blocking inference defect.

The checked and ARM-built source is commit `704ed4ac`; subsequent changes update
records only. The ARM binary SHA256 is
`34c70aa792a07f561231450acdb9f7e13a9f4d824506c6c6c591929ec21d4022`.
Borg CLI load plus CPU decode took 18,699/18,590 ms for the full inputs and
2,430–2,742 ms for the small probes. These include weight admission and establish
neither GPU performance nor a production latency improvement. The executable
uses native Rust and libm; the standard-library Python harness only checks its
receipts. No reference audio was prepared, waveform synthesized or service
artifact deployed. The existing speech service remains the operational engine.

Evidence, exported weights, independent oracle, binary and receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-meanflow-20261030/`.
Borg retains `meanflow-assets`, `meanflow-receipts`, `chatterbox-meanflow-rs` and
`verify_meanflow_borg.py` under
`/home/muttleydosomething/chatterbox-rust-build-20261030/`.

Native HiFT was exercised with the ARM release executable on Borg on 2026-10-30.
All six independent waveform cases passed, including the retained and reversed
137-frame generated mels, which each produced 65,760 samples. The largest
waveform absolute error was 1.4271587133407593e-5, within the declared 3e-4
absolute plus 3e-4 relative gate. Pitch and excitation passed independently;
the threshold/phase probes, source STFT, fixed-source decode, exact native
CLI/library identity and retained-model replay passed on Saturn. Four focused
kernel/admission tests passed, including finite-input spectral overflow rejection.

The ARM-built runtime source is commit `dd6d2017`; subsequent changes update
records only. The preceding numerical/replay proof used `8259f1bf`; the later
runtime correction adds a finite-spectrum rejection check without changing
valid-input arithmetic. That correction and malformed-asset tests were proved
separately before the ARM build and Borg waveform comparison. The binary SHA256 is
`149d0c4e9d2f3a4dbcfd98f573f8862381efb53265e8588705786174cfb41762`.
Borg CLI load plus CPU synthesis took 7,012/6,980 ms for the two full probes and
748–924 ms for the small probes, including weight admission. This is not a
retained-model latency measurement or a production speed improvement claim.

Evidence, exported assets, binary, receipts and `native-retained.wav` are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-hift-20261030/`
(the WAV is inside `receipts/`). Borg retains `hift-assets`, `hift-receipts`,
`chatterbox-hift-rs` and `verify_hift_borg.py` under
`/home/muttleydosomething/chatterbox-rust-build-20261030/`.
The independent oracle ran in a separate CPU-only/network-disabled 1536 MiB
container. No reference audio was prepared and the existing speech service
remained running with its original start time. The waveform executable is a proof
artifact; it has not replaced the service. Listening quality and full native
speech-pipeline/service integration remain unverified.

## Remaining conversion

The final output boundary was exercised with the ARM release executable on Borg
on 2026-10-30. All seven deployed-wrapper cases passed with maximum absolute
error 1.1920928955078125e-7 (gate 3e-7); both short inputs were rejected. Native
contract checks, exact CLI/library identity and replay passed on Saturn. The
previous native HiFT waveform also passed through the native finishing command,
producing `receipts/native-finished.wav` with unchanged length and tail bits.
This composes the last two waveform boundaries, not the complete speech service.

Runtime source is commit `7809e1b9`; later code changes only correct the oracle's
type-annotation environment, and the ARM build used `c8713da6`. The binary SHA256
is `fed6266575af824463c800713dfb8818b1c85111a60e16e5e58eff26b233c3c9`.
Evidence, oracle, binary and receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-output-20261030/`.
Borg retains `output-assets`, `output-receipts`, `chatterbox-output-rs` and
`verify_output_borg.py` under `/home/muttleydosomething/chatterbox-rust-build-20261030/`.
No reference audio was prepared and no service artifact was installed. Listening
quality, full native pipeline/service integration and production latency remain
unverified. Perth is intentionally omitted.

### Native request randomness

The existing sampling module now supplies standard-normal meanflow noise
`[1,80,F]`, including reference frames, and HiFT excitation `[1,9,480F]` plus
uniform harmonic phases in radians `[1,9,1]`. Box-Muller reuses the native
SplitMix64 generator. Separate stream domains isolate token sampling, meanflow,
HiFT noise and HiFT phases; phases are independent of utterance length. State
belongs to each call, so replay after another request cannot advance its seed.
Zero/overflowing tensor sizes and allocation failures return explicit errors;
no arbitrary utterance limit or additional dependency is introduced.

The existing proof commands accept native randomness in place of captured arrays:

```sh
chatterbox-rs s3-meanflow /path/meanflow.json MEANFLOW_SHA256 /path/mu.npy \
  /path/mask.npy /path/spks.npy /path/cond.npy --seed 42 2 REFERENCE_FRAMES
chatterbox-rs s3-hift /path/hift.json HIFT_SHA256 /path/generated-mel.npy --seed 42
```

Receipts include `random_seed` (null for the original captured-array route).
The native sequence differs from Torch/CUDA even with the same seed. This
supplies decoder inputs; it does not connect text generation to the waveform
stages or replace the operational Python service.

This boundary was exercised with the ARM release executable on Borg on
2026-10-30. Native-seeded meanflow and HiFT replay matched exactly after a
different-seed request; different seeds changed the outputs. Six invalid seeds
were rejected, and one small captured-array case per command still matched its
independent deployed oracle. The HiFT probe used the retained 137-frame mel and
produced 65,760 samples; the meanflow probe used two frames. The native finishing
command produced `receipts/native-random-finished.wav`. These are separate
decoder probes, not a connected text-to-waveform run or a listening assessment.
Four focused Saturn tests passed, including distribution moments/tails, tensor
shape/admission checks, library/CLI identity and immutable-model replay.

The ARM-built runtime source is commit `d50ea824`; subsequent changes update
records only. The binary SHA256 is
`0ae2834f888c43d2e0926184aa9085c618c607291a5cb8429389476980dee4cd`.
Evidence and receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-native-random-20261030/`.
Borg retains `chatterbox-native-random-rs`, `verify_native_random_borg.py` and
`native-random-receipts` under
`/home/muttleydosomething/chatterbox-rust-build-20261030/`.
Each native Borg process had a 1536 MiB address-space limit. No reference audio
was prepared or service artifact installed. Initial Borg CLI load plus decode
took 2804 ms for the tiny meanflow probe and 9185 ms for HiFT; these include
asset admission and do not establish production latency or GPU performance.

### Connected retained-voice request

The existing CLI now supplies one native, final, unpadded text-to-WAV route:

```sh
chatterbox-rs synthesize /path/pipeline.json PIPELINE_SHA256 \
  'Ready, Holmes.' /path/output.wav 42 600 /path/sampling.json
```

The checksummed version-1 aggregate manifest has `schema_version: 1` and nine
entries: `bridge`, `engine`, `prompt`, `reference`, `flow`, `conformer`,
`mel`, `meanflow` and `hift`. Each entry contains `path` and `sha256`.
Relative paths resolve against the aggregate's directory; absolute paths permit
reuse of the existing exported assets. Every nested manifest keeps its existing
asset admission. Bridge and engine checksums are checked before native loading.
The prompt and S3Gen reference must describe the same prepared voice; the proof
checks their common reference-archive checksum. New audio is never encoded here.

Text admission and sampling reuse the existing routines. A generation stop before
EOS rejects the request without a WAV. Speech cleanup runs exactly once before
flow assembly, including the three silence codec IDs. The offline route releases
TensorRT before waveform decoding, Conformer before meanflow, and meanflow before
HiFT. The HTTP service retains one admitted token engine as described below,
while releasing request context/KV before waveform inference.
Request-local seed domains supply both decoders. Meanflow uses two Euler steps
and crops the retained reference before HiFT; the final waveform must contain
exactly 480 samples per generated mel frame. Output is lossless mono IEEE FP32
WAV at 24 kHz with a fact chunk; no gain, clipping or watermark is added.
The RIFF container's 32-bit length limit is explicitly checked.

The destination is created only after all inference and WAV validation succeeds.
An existing destination is never overwritten. A failed final output operation
returns an error and can leave an incomplete artifact; no success receipt is
issued. Successful receipts retain text, prompt checksum, both token sequences,
reference/generated frame counts, sample count, WAV checksum, peak amplitude,
per-stage timing, start/finish times and total duration.

The ARM executable passed the complete route on Borg on 2026-10-30:
`Ready, Holmes.` produced 29,280 samples; a fresh longer sentence produced
65,760 samples. Both speech/decoder token sequences matched the earlier native
prepared-reference receipts. Repeating the first input after the longer input
reproduced every WAV byte. Reference preparation count remained zero.
The bad aggregate checksum, oversized text, existing output destination and
forced pre-EOS length stop were rejected; failed requests published no new WAV,
and the existing output retained its checksum.

The two focused handoff/WAV tests passed on Saturn at runtime commit
`8dcee094`. The final ARM build used `fed07f66`, whose only subsequent source
change corrects the token command's help text. Its binary SHA256 is
`dd9565671ca2550ca585a0ce8c97837f463768794c57e2816185aefc75c8eb76`.
Initial complete-request times were 30,762 ms, 41,043 ms and 29,963 ms, including
asset admission and CPU waveform decoding. These establish neither production
latency nor audible quality. Each native request ran in its own 1536 MiB memory
scope with no swap. The live speech container retained its original start time
`2026-10-30T12:06:46.911067898Z` and running state.

Evidence, aggregate manifest, standard-library receipt harness, final ARM binary
and WAVs are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/chatterbox-native-pipeline-20261030/`.
The listening artifacts are `receipts/retained.wav` and `receipts/fresh-text.wav`.
Borg retains `chatterbox-pipeline-rs`, `verify_pipeline_borg.py` and
`pipeline-receipts` under
`/home/muttleydosomething/chatterbox-rust-build-20261030/`.
The harness performs no speech inference. The runtime uses Rust and the existing
TensorRT C++/CUDA bridge, with no Python or Torch in its process. This remains
an offline executable; no service artifact was installed or gateway restarted.
The unrelated Cargo registry auto-cleanup permission warning remains outside
this slice; both builds and the focused proof succeeded.

Native preparation and publication are described above. GPU acceleration and
performance, production speech-service integration and audible voice-quality
proof remain required. Do not replace the current speech service with this crate.

### Native HTTP entry point

`chatterbox-rs serve SERVICE.json SERVICE_SHA256 LISTEN_IP:PORT` runs a native
HTTP service for explicitly admitted, already prepared voices. This is an
isolated service trial; the operational Python service remains in place.
The listener address is explicit so a trial need not expose a public interface.

The service owns a bounded, lazy TensorRT engine cache. Every request still
checks the admitted engine checksum and process bridge identity. A cache hit
opens a fresh execution context with empty KV and the current voice prompt;
context/KV retire before waveform inference. Replacement evicts the old engine
before allocating another. Unload and successful reload clear the cache under
the existing inference permit on a blocking worker. Offline CLI ownership is
unchanged. Engine residence requires measured combined RAM admission; the
retention proof below does not admit the former 1536 MiB trial envelope.

The checksummed service catalog has this shape:

```json
{
  "schema_version": 1,
  "model": "myth",
  "model_aliases": ["chatterbox-borg"],
  "voices": {
    "retained-reference": {
      "display_name": "retained reference",
      "pipeline": "/admitted/pipeline.json",
      "sha256": "PIPELINE_SHA256"
    }
  },
  "sampling": {"temperature": 0.8, "top_k": 1000, "top_p": 0.95, "repetition_penalty": 1.2},
  "seed": 42,
  "max_gen_len": 600
}
```

`POST /v1/audio/speech` accepts `model`, `input`, `voice`, optional `seed`,
`response_format` of `wav`, `mp3` or `opus` and a finite positive `speed`
(default 1.0). Invalid controls fail explicitly. A native phase vocoder changes
duration while retaining pitch and the 24 kHz rate before response encoding.
It uses the deployed librosa 0.11 defaults: centered zero-padded 2048-point
periodic Hann analysis, 512-sample hop, interpolated magnitudes, phase unwrapping
and window-normalized overlap/add. Output length follows nearest-even
`round(samples / speed)`; empty, overflowing or unallocatable output fails.
Normal speed borrows the original waveform without transforming or copying it.
Only two input spectra and one inverse frame are retained; output samples and
their normalization are the duration-dependent buffers. All state is request-owned,
with cancellation checks each frame and each normalization chunk. This uses the
pinned native Rust FFT crate, with no additional runtime library or subprocess.
WAV uses the operational service's clipping and
FP32 multiply/truncate to PCM16; the offline inference proof keeps its separate,
lossless FP32 WAV. MP3 uses installed `libmp3lame.so.0` at mono 24 kHz/64 kbps
with a finalized Info/LAME gapless tag. Opus uses installed `libsndfile.so.1`
and an anonymous seekable descriptor to finalize its Ogg container. There is no
Python, FFmpeg subprocess, reference preparation or named response artifact.
Media types are `audio/wav`, `audio/mpeg` and `audio/ogg`. Responses include
`x-myth-started-unix-ms`, `x-myth-finished-unix-ms`, `x-myth-duration-ms`,
`x-myth-samples` and `x-myth-sha256` headers; the checksum covers the encoded
response bytes. Codec state and finalization remain inside the inference permit;
cancellation is checked between encoding chunks. Compressed outputs are not
claimed bit-identical to Python/FFmpeg output or as audible-quality acceptance.
`GET /get_predefined_voices` lists only admitted native preparations.
`GET /health` and `/api/model-info` report the engine and its present limits.
They do not claim that merely opening a listener has exercised inference.

One inference worker owns admission; overlapping requests receive HTTP 429
instead of accumulating model allocations. Blocking model work leaves health
responsive. Dropping the handler signals cancellation every token and between
waveform stages; the worker retains admission until native work actually exits.
SIGTERM and Ctrl+C stop new admission and drain existing work.

The adjacent `Dockerfile` packages only the governed ARM executable in a scratch
image. Borg must supply read-only compatible Linux, CUDA/TensorRT and pinned
asset mounts, NVIDIA runtime access and a bounded memory allocation. The image
contains no Python interpreter, inference wrapper or model downloads.
New-reference preparation, uploads/deletion, incremental audio and GPU waveform
computation are not supplied by this trial.

#### Borg HTTP acceptance, 30 October 2026

The isolated `myth-native-trial` container passed real HTTP acceptance from
Saturn at runtime commit `ec98e4b2`. Its scratch image contains only the native
executable; Docker's process view showed `myth`, with no Python process. Each
response matched the earlier offline `retained.wav` byte for byte, including
replay in the same process: 29,280 samples, WAV SHA256
`0345801f9080a1dc8899d74ccc368e86511975c786070b3fb688489c27d65c27`.
The two requests took 35,356 and 35,479 ms. This is not production latency.

Real transport disconnect returned admission in 1,424 ms. Health remained
responsive during inference; overlapping synthesis returned HTTP 429.
Unprepared/traversing voice IDs, unsupported model/format/speed, empty input
and unknown controls were rejected. A wrong catalog checksum failed before
listener creation. Reference preparation count remained zero. The original
speech container retained its running state and original start time.

The initial HTTP trial returned the correct first WAV but crashed with exit
139 on replay, without an OOM kill. The corrected bridge retains registered
native code for the process lifetime while releasing request-owned engine,
context and KV state. A focused fixture verifies that the bridge remains
loaded across handle close/reopen and rejects rebinding its identity. Six
HTTP/pipeline tests and that additional lifetime regression passed. The
corrected real replay passed; the failed trial is retained separately.

The corrected ARM binary SHA256 is
`95af931be5531979248ee840bd35ff2dd48c65c1d86b6badb0aa6b6fc0b3bd98`.
Its image is `hailstorm/myth-native:ec98e4b2`, installed on Borg with a 1536 MiB
memory ceiling, no swap, read-only root and read-only assets. The service listens
only on Borg's private `10.200.0.2:8766`; the operational speech route was not
changed. Acceptance, deployment, build and watch-consumption receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-native-service-20261030/`.

This proves the retained-voice HTTP boundary, not complete conversion or
production replacement. New-reference encoders, GPU waveform execution,
incremental speech and the existing workshop/service contract remain open.
Audible quality was not reassessed. Cargo also emitted its existing unrelated
registry auto-cleanup permission warning; both governed release builds succeeded.
The classifier under-classifies this Borg crate as docs-only; the explicit
lifecycle/concurrency acceptance above was applied instead of weakening proof.

#### Native response-format acceptance, 30 October 2026

Runtime candidate `177e0d06810ea649bc888b02bc5064423708dd09`, integrated and
pushed at `40a3006d`, extends the existing HTTP boundary to PCM16 WAV, MP3 and
Ogg/Opus. Ten focused response-codec, HTTP ownership and pipeline regressions
passed, followed by the governed ARM release build. Independent FFmpeg decoding
of all three one-second synthetic outputs recovered exactly 24,000 samples;
tone correlation checked rate, pitch, gross gain and encoder-delay handling.
Cancellation was exercised between codec chunks and before codec admission.

The updated isolated `myth-native-trial` then passed three real speech requests
on Borg's existing private listener. Each independently decoded to exactly
29,280 mono samples at 24 kHz. The HTTP PCM16 WAV matched the earlier native
FP32 waveform's deployed clip/multiply/truncate conversion exactly. MP3 and
Opus decoded SNR against that PCM16 reference was 24.1132 and 22.4698 dB.
These checks detect response corruption; they are not audible-quality proof
or a claim of byte parity with the Python/FFmpeg compressed encoders.

The WAV, MP3 and Opus requests took 30,369, 29,726 and 30,252 ms, including the
existing CPU waveform stages and model admission. Their encoded sizes were
58,604, 10,368 and 6,196 bytes. The response media types, exact-byte checksums,
timing and sample headers passed. Health remained responsive and overlapping
inference returned HTTP 429. Native operations recorded zero reference
preparations for all three requests; the container process was `myth`, without
Python. Production speech retained start time `2026-10-30T16:30:55.141432741Z`.

The ARM binary SHA256 is
`34c001ee76d58f2e2045c073292f4b5e6fecb99ef8899f1b8bb6f17b2bd225b3`.
Image `hailstorm/myth-native:output-177e0d06` replaces only the isolated native
trial, with the existing 1536 MiB/no-swap ceiling and read-only mounts.
The prior trial remains stopped as `myth-native-trial-before-output-177e0d06`
for recovery. The resource heartbeat preceded deployment. Evidence, response
audio, deployment and consumed supervisor receipts are retained under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-native-output-20261030/`;
the live speech gate is `acceptance.json`, and the independent codec gate is
`codec-decoding.json`. Both task-owned watches were consumed. The candidate
was retired after releasing the idle compiler-cache process's target reference.

The integration gate initially refused an incorrectly supplied full candidate
hash; no merge occurred until its exact Git identity was read and supplied.
Candidate retirement initially refused the cache process's retained target
environment; no Cargo/rustc process was active when that idle cache was stopped.
Cargo's unrelated registry-cleanup permission warning recurred without failing
either build. The classifier still under-classifies the Borg crate as docs-only;
the explicit native response/lifecycle proof above supplements that result.

Production native reference upload/Apply remains accepted as recorded earlier.
At this output-encoding slice, production synthesis still used Python. Native
speed processing, the remaining service/workshop handlers, streaming, GPU
waveform acceleration and real-recording audible acceptance remained open.
This slice changed output encoding only.

#### Native speed acceptance, 30 October 2026

Runtime candidate `4602b5077da27998b6f4699b8181c1db3796bf24`, integrated and
pushed at `eb4a4e05`, adds native pitch-preserving response tempo. Four focused
tempo tests and six HTTP ownership/response tests passed, followed by the
governed ARM release build. Twelve independent deployed-librosa 0.11 captures
cover tones, chirp, silence, short input and the earlier native speech waveform.
All lengths matched; silence was exact, maximum sample error was 5.9604645e-8,
and nonzero-case SNR was at least 136.9975 dB against the declared 40 dB gate.
An independent zero-crossing test retained 440 Hz pitch at five speed settings.
Replay after intervening input, nearest-even sample counts, malformed/nonfinite
input, invalid speeds, output-size overflow and mid-transform cancellation passed.
Normal speed borrows the input, retaining its bits and avoiding another copy.

Five real requests then passed through the isolated Borg trial. WAV at 0.75x
and 1.5x decoded to exactly 39,040 and 19,520 samples at 24 kHz. Agreement with
the independently captured deployed transform after PCM16 conversion was
109.2663 and 96.9348 dB. MP3 at 1.5x and Opus at 0.75x decoded to the same
expected counts; decoded SNR versus the corresponding native WAV was 24.4977
and 20.8734 dB. Normal-speed WAV after these requests reproduced every byte
of the preceding output-format trial. Media types, response checksums and timing
headers passed. Invalid speeds returned HTTP 400, health remained responsive,
and overlapping inference returned HTTP 429. These are numerical/transport
checks, not listening-quality acceptance.

The ARM binary SHA256 is
`2f8a674253128b31e0740e207cb99ba50c86ad2485d92c4e880c8cdfb1238884`.
Image `hailstorm/myth-native:speed-4602b507` replaces only `myth-native-trial`
on its existing private listener, with the existing 1536 MiB/no-swap ceiling
and read-only mounts. Recovery retains
`myth-native-trial-before-speed-4602b507`. Docker's process view showed only
`myth`; no Python interpreter or additional native tempo library is required.
All five operations recorded zero reference preparations. Request times were
29,922–34,092 ms, dominated by existing model admission and CPU waveform stages;
these measurements do not isolate tempo latency or prove production performance.
The production speech service retained start time
`2026-10-30T16:30:55.141432741Z` and its running state.

Evidence, oracle, response audio, exact ARM binary and consumed supervisor
receipts are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-native-speed-20261030/`.
The live gate is `acceptance.json`; independent fixtures are in `oracle/`,
and `verification.json` records the focused proof. Both task watches were
consumed. The integrated candidate was retired after confirming no active
Cargo/rustc processes and releasing its idle compiler-cache target reference.
The predeployment heartbeat flagged Saturn's load at 5.73 on six CPUs, with
29% RAM use; Borg reported load 0.06 and 77% RAM use and passed the measured
headroom gate. Cargo's existing unrelated registry-cleanup permission warning
did not fail either command. The classifier has no native Borg crate rule and
reported no risk classes; explicit lifecycle/concurrency/tempo proof was applied.

Production synthesis still uses Python. Remaining service/workshop handlers,
incremental streaming, GPU waveform acceleration and real-recording audible
acceptance remain required before production replacement.

### Native service and workshop contract

The optional `workshop` entry in the existing checksummed service catalog
activates the established Myth HTML/JavaScript against the native HTTP owner.
It contains `ui_root`, `state_root` and three `{path, sha256}` pins:
`preparation` (the existing voice-preparation manifest), `prompt` (the existing
publication template), and `pipeline` (the existing aggregate inference template).
Both directories must exist before startup. `delete_authority_sha256` supplies
the existing gateway-authority verifier; its plaintext credential is never part
of UI initial data or mutable settings.

The native owner supplies `/`, the existing static assets and `/docs`,
`/api/ui/initial-data`, `/get_reference_files`, `/get_predefined_voices`,
`/upload_reference`, `/upload_predefined_voice`, `/delete_predefined_voice`,
`/save_settings`, `/reset_settings`, `/restart_server`, `/load_model`,
`/unload_model` and `/tts`, alongside the earlier OpenAI speech and health routes.
The deployed workshop assets are served unchanged. There is no Python process,
HTTP proxy to the Python service, model download, or inference subprocess.

Multipart recordings stream to hidden preparation directories without an upload
body-size cap. Apply uses the existing `VoicePreparation` and `PreparedVoice`
publication routines for both normalization choices, then pins native pipelines.
Source bytes, both complete variants and their directory entries are flushed
before a single atomic `state.json` envelope makes the voice visible. That
envelope owns native workshop settings and voice registration. A process-lifetime
flock rejects another service owner of the same state root. Interrupted work
remains hidden and recoverable; the service never lists a half-prepared voice.
Same-name/same-byte replay validates existing pins without encoding; conflicting
bytes are rejected. The native runtime uses Borg's host libsndfile 1.2.2 for both
WAV and MP3, including source formats without filename extensions in staging.
The earlier Python image's libsndfile 1.0.31 limitation does not apply here.

`/tts` maps the existing workshop request to the shared native synthesizer,
including voice selection, temperature, seed, output format and speed. Sentence
chunks share one admission, cancellation signal and final encoded response.
Their join preserves the deployed 20 ms fade edges and 200 ms silence; tempo
is applied before each join. Existing native deterministic-seed semantics apply.
Supported Turbo defaults take effect on the next request. UI history persistence
has its own short state transaction and remains available during inference.
Unsupported models, languages, sample rates, output destinations and unknown
configuration keys are rejected explicitly rather than silently ignored.

Upload, inference, deletion and model lifecycle share admission. Reload validates
aggregate pins in-process; it never restarts the service or Hailstorm. Model
contexts remain request-owned, so unloaded means synthesis admission is disabled
while prepared voices are retained. Busy unload/reload receives HTTP 429.
Deletion requires the existing gateway-authority header; the gateway remains
responsible for operator PIN and assignment checks. A successful native deletion
withdraws the registered voice and removes its owned recording and preparations.

This native state authority is scoped to the service's configured state root.
The trial does not import or mutate production settings or the Python service's
existing voice registry. Production cutover must bind or migrate the accepted
preparations explicitly, without encoding retained recordings again.

#### Borg workshop acceptance, 30 October 2026

Runtime `f98c64b495fada3aabf32c2995a4cce3df01fff9` runs as
`hailstorm/myth-native:reload-f98c64b4` on the existing private trial listener.
Its ARM executable SHA256 is
`be6f0337b387ae3ef3fc50c46bb605e396708f7417a149c8c310e69a48f48700`.
Ten focused HTTP/workshop tests passed; a separate bounded-memory regression
admitted a 256 MiB asset within a 128 MiB address-space allowance. The governed
ARM release build passed. The classifier still has no native Borg crate rule;
the explicit persistence, concurrency, lifecycle and browser gates below supply
the proof that its empty risk list omits.

The browser served the unchanged deployed workshop assets through Rust. Selecting
a fresh reference caused no upload; Apply sent one upload, committed both native
normalization variants, closed the editor and selected the new voice. Generate
returned a 59,564-byte PCM16 WAV: the browser's playback Blob matched its SHA256,
decoded to 29,760 mono samples at 24 kHz and played through the actual Play button.
Browser exceptions and failed UI requests were absent. Health stayed responsive
and unload during synthesis returned HTTP 429. The fixture repeats the preceding
native waveform to form a valid 7.32-second reference; this is transport/ownership
proof, not real-recording voice-quality acceptance.

The real API also accepted an MP3 reference through native libsndfile and generated
clone speech. Both uploads prepared two variants only once. Duplicate upload and
post-restart replay reported zero preparations. Different bytes under the same
name, traversal, malformed audio and unsupported configuration were rejected.
Applied settings and both prepared references survived the initial trial failure
and subsequent ordinary restart. Unloaded speech returned HTTP 503. Cold reload,
clone synthesis, unload and warm reload all passed within the unchanged 1536 MiB
no-swap ceiling. Clone replay after reload matched its earlier WAV byte for byte.
Deletion without gateway authority returned HTTP 403; authorised deletion removed
both test voices, their owned recordings and preparation directories. Lists and
speech admission reflected withdrawal. Settings reset returned native defaults.
Docker's final process inspection showed only `myth`, with no Python process.

Acceptance found and corrected one native defect: reload originally buffered the
large engine while the warm process retained CPU arenas, causing cgroup OOM.
Reload now reuses the existing 64 KiB streaming checksum admission. The OOM trial
is retained as `myth-native-oom-before-reload-f98c64b4`; recovery also retains the
preceding speed trial as `myth-native-trial-before-workshop-7916abb0`.
Initial deployment also exposed that FFmpeg existed only inside the Python image;
the native host decoder made that extra executable unnecessary. The first short
probe was rejected by the established five-second reference minimum before the
valid-length fixture was used. Browser proof needed corrections for a collapsed
control and Chromium's empty instrumentation buffer; its final gate hashes the
actual playback Blob. The management harness reached its final inspection after
all assertions, then failed on SSH quoting. The corrected diagnostic completed
only that remaining inspection, preserving the earlier receipt and mutations.

Evidence is retained under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-native-workshop-20261030/`.
`browser/browser-acceptance.json` retains the successful Apply and the later
control-visibility failure; `browser-final/browser-acceptance.json` is the passed
playback gate. `http-acceptance.json` and
`lifecycle-acceptance-run/command.log` jointly record lifecycle/deletion proof and
the final diagnostic correction. Supervisor claims consume the exact task-owned
watches; failed receipts remain available. Cargo's existing unrelated registry
auto-cleanup permission warning did not fail the builds. The resource heartbeat
preceded both trial activations; no gateway, Redis or Qdrant restart occurred.

Production `chatterbox-tts` retained its running state and start time
`2026-10-30T16:30:55.141432741Z`. Step 1's native service/workshop gate is accepted
on the isolated trial. Incremental streaming, GPU waveform execution, listening
acceptance and production cutover (including existing registry/settings binding)
remain separate required stages. Production synthesis still uses Python.

#### Sentence-streaming transport acceptance, 30 October 2026

Step 2 is **partial**: playable segments arrive before generation finishes and
interruption is accepted on the isolated native workshop, but the current native
path cannot yet sustain continuous real-time playback. Production remains on
Python. GPU waveform execution, listening acceptance and production cutover remain
separate gates; these measurements do not establish faster-than-real-time speech.

`POST /tts/stream` and `POST /v1/audio/speech/stream` reuse their existing request
shapes, defaults, prepared-voice admission, synthesizer and single inference
permit. They accept WAV output only. Their response is `application/x-ndjson`,
version 1: a `start` record, ordered `audio` records carrying independent base64
PCM16 mono 24 kHz WAVs, then exactly one `complete` receipt. Audio records specify
their zero-based index, sample offset and sample count. The receipt retains start,
finish, duration, first-audio readiness, total frames/samples, aggregate PCM SHA256
and preparation count. HTTP 200 admits transport only: an error record, missing
terminal receipt or disconnect cannot become a successful download. Private
inference diagnostics remain in service operations.

The stream reuses the workshop sentence splitter and its controls; the OpenAI
stream selects its existing 120-character grouping. Disabling workshop splitting
still produces one complete segment. This is sentence streaming, not incremental
attention or sample generation inside a sentence: each segment needs its complete
token sequence before the current waveform decoder runs. Smart joins retain only
the 20 ms tail needed for the next fade and preserve the established pause. Tempo
is applied per segment. Streaming uses the native PCM16 clipping policy instead
of the buffered workshop's whole-utterance peak-normalization pass, which would
require retaining the entire output. Existing complete-file WAV, MP3 and Opus
routes retain their preceding contracts; the workshop automatically streams its
WAV selection and retains compressed downloads through the buffered route.

One queued service frame bounds producer lookahead. Dropping the response body
signals cancellation and wakes a blocked sender; native ownership is retained
until the worker exits. The browser parses fragmented records, schedules Web
Audio sources on one sample timeline, limits scheduled lookahead, retains PCM for
the existing finished download/player, and requires completion before publishing
that download. Stop, a new stream, input/voice changes, page hiding and page exit
abort transport and stop queued sources. A stale asynchronous continuation cannot
resume playback. The stream module integrates through the existing Generate
routine; its patch and build entry live beside the existing workshop extension.

Fifteen native HTTP/workshop checks passed, including first-frame delivery while
later synthesis was blocked, ordered/checksummed completion, bounded producer
lookahead, terminal failure, and cancellation without premature permit release.
The browser client proof passed fragmented transport, contiguous scheduling with
a sufficiently fast producer, failure, out-of-order audio, truncated output,
interruption and re-entry. The real Borg workshop then played its first frame at
29.163 seconds while later inference still held admission. Generation finished
in 100.191 seconds for 135,840 samples / 5.66 seconds of audio: real-time factor
17.70, with measured playback starvation gaps of 41.173 and 25.254 seconds. The
browser decoded the finished mono 24 kHz download to the expected sample count.
Thus earlier delivery is proven; uninterrupted playback on this native execution
path is explicitly **not achieved**. Model admission and CPU waveform execution
still need acceleration; these timings do not isolate their individual costs.

The resumed live interruption check passed: Stop aborted an active request,
emitted no stale audio or completion, and the worker returned admission 2.224
seconds after interruption. Browser exceptions were absent in that gate. The
separate client proof covers stopping already scheduled playback. Three complete
segments and cancellation ran within the unchanged 1536 MiB/no-swap allowance;
the trial remained running without OOM and its process was `myth`.

The ARM binary is from `7440497ff8bf54368f2a4c71cd84e9cf8e280e89`, SHA256
`25a527d1da637d770e1d0b5d1d8e46b92e384f831d9f5a1f4dbadbec1ee1a70a`, in image
`hailstorm/myth-native:stream-7440497f`. Only the workshop script changed after
that build: the real UI exposed a replay-handler reference scoped inside the old
player function; document-level delegation corrected it. The activated script
SHA256 is `a34b45cae65b55b028b250ccb3c8275de81eb54aab3c436f985daf5688996437`.
Recovery retains `myth-native-trial-before-stream-7440497f`. The resource heartbeat
preceded deployment. Production retained its running state and start time
`2026-10-30T16:30:55.141432741Z`; no gateway, Redis or Qdrant restart occurred.

Evidence is under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-native-stream-20261030/`.
The successful generation/playback and timing evidence is in
`browser-corrected/browser-live.json`; its overall harness status is failed because
the second submission stopped at the existing acknowledgement dialogue before
interruption. Only that unfinished gate was resumed in
`browser-cancel-corrected/browser-cancel.json`. The first interruption-only harness
also needed to open the collapsed parameter section. The generation receipt's
download `bytes` field is zero because Web Audio detached its input buffer before
instrumentation read byteLength; decoded sample count and live playback passed.
`browser-client/browser-client.json` contains the independent browser client
gate. Failed runs and consumed supervisor claims remain available. The classifier
still reports no Borg risk classes; explicit streaming, concurrency and browser
proof supplement it. The shared asset builder found an unchanged missing RNNoise
dependency, so only the new streaming module was bundled using the existing
primary frontend toolchain and deployed reference assets were retained.

#### GPU waveform acceleration acceptance, 30 October 2026

Step 3's isolated native GPU-acceleration gate passes. Conformer, meanflow and
HiFT now execute their shared matrix workload through native FP32 cuBLAS on
Borg. Production synthesis remains Python. This is mixed CPU/GPU execution:
normalisation, softmax, activations, im2col, Fourier work and audio assembly
remain native CPU operations. It is not a claim that every waveform operation
runs on the GPU, nor that speech is faster than real time.

The existing `conformer::product` is the single dispatch boundary. Conformer's
attention-value reduction now joins it through an explicit V transpose;
Meanflow and HiFT already use it. `MYTH_WAVEFORM_BACKEND=cuda` selects cuBLAS;
the absent setting or `cpu` retains matrixmultiply. Unknown selections fail.
Only waveform scopes enter CUDA, after the TensorRT token session closes.
Prepared-reference encoders and uploads retain their preceding execution path.

Each scope owns a cuBLAS handle and three reusable device scratch buffers.
Current host contents are uploaded on every call: pointer reuse cannot select
stale weights or attention state. Pedantic FP32 math avoids TF32/FP16 changes.
Kernel errors remain sticky through downstream nonfinite handling; no CPU
fallback or success receipt follows a CUDA failure. Successful completion checks
synchronisation, every buffer release and handle destruction. Unwinding also
releases ownership. Device reset is never used. CPU model and GPU scratch
lifetimes remain inside the service's existing inference permit.

The first real streaming run completed two segments, then its third segment
was OOM-killed under the existing 1536 MiB/no-swap limit. Its state and operations
are retained in `oom-runtime.json`; the preceding CPU trial was restored before
repair. Following the existing Jetson Python implementation, the corrected
native scopes call glibc `malloc_trim(0)` before admission and after teardown,
including unwind cleanup. Recorded resident-page measurements show substantial
released host arenas; the retained meanflow stage released about 310 MB in the
first completed segment. The correction does not change matrix arithmetic or
raise the memory ceiling.

Forty-two focused host tests passed. Borg also passed the three injected
failure/overflow/unwind tests and a real CUDA layout, changed-shape replay and
context-reentry proof. All 15 retained independent numerical fixtures passed
their existing tolerances: four Conformer, five meanflow and six HiFT cases.
Maximum absolute errors were 1.9073486e-6, 2.0027161e-5 and 1.4048070e-5,
respectively. Bad manifest digests produced no success receipt. No reference
voice was prepared again.

The initial arithmetic build, `64bb2d37`, measured the following retained cases
with one executable and matching inputs. These CLI durations include admission,
model loading and compute, excluding container startup. They precede the host
arena correction and are not final production benchmarks:

| Stage | CPU, ms | CUDA, ms |
| --- | ---: | ---: |
| Conformer | 3476 | 2263 |
| Meanflow | 19121 | 10215 |
| HiFT | 7082 | 4166 |

The corrected `5be06167` build passed the actual workshop's three-segment
stream, completed-download decoding, Stop and worker release. First audio was
ready at 20.717 seconds; complete generation took 68.738 seconds for 135840
samples / 5.66 seconds of speech. The maximum playback starvation gap was
27.317 seconds. Interruption returned admission in 2.031 seconds with no stale
audio or completion. A subsequent independent streaming request succeeded after
cancellation. Browser exceptions were absent. Step 2 remains partial.

The same running native process completed four waveform operations with GPU
counters for all three stages. It reached the enforced 1536 MiB cgroup ceiling,
with zero swap, zero OOM events/kills and zero restarts after correction; this
does not establish spare memory headroom. It ran as `myth`. The operational
Python service retained its start time `2026-10-30T16:30:55.141432741Z`.
No gateway, Redis or Qdrant restart occurred.

The exact ARM binary, numerical and timing receipts, failed first activation,
corrected browser proof and consumed supervisor claims are retained under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-waveform-cuda-20261030/`.
The accepted image is `hailstorm/myth-native:cuda-5be06167`, with
`myth-native-trial-before-cuda-5be06167` retained for rollback. The runtime source
after the numerical gate changes host-arena reclamation and its measurements
only. The classifier still has no Borg-specific risk rule; explicit lifecycle,
numerical, resource and real browser proof supply that missing contract.
Human listening acceptance, production cutover and the already recorded
post-conversion performance investigation remain separate work.

### Retained voice inventory migration

The native workshop has an offline, checksummed import transaction:

```sh
chatterbox-rs import-workshop WORKSHOP.json SHA256 INVENTORY.json SHA256 SEED
```

`WORKSHOP.json` is the existing service catalog's `workshop` configuration.
The inventory has `schema_version: 1`, the supported source `config` sections,
and `sources`, each containing an absolute recording `root`, a `reference`
boolean, and the complete `recordings` map of filename to SHA256. Operational
server settings and credentials are not needed in this input.

Every recording must match its existing `.native/<filename-sha256>.json`
binding, including the preparation-model pin and both normalization variants.
Missing preparations, partial inventories, changed recordings and different
recordings sharing one name reject without committing native state. Identical
recordings listed as both predefined and reference voices keep both memberships.
Older native states retain their preceding single-membership interpretation.

Import takes the existing source ownership lock, streams independent recording
and tensor copies into the existing hidden upload staging, checks all nested
prompt/reference assets, then commits settings and registrations in one state
envelope. Original sources remain intact. No encoder runs, and no input-size
ceiling is added. Interrupted copies remain hidden. Exact replay validates native
tensors and preserves settings subsequently changed through Apply. An active
native owner or a different existing native state rejects offline replacement.

HiFT's remaining source-prefix operation also has a native library entry point,
`synthesize_cached`. It preserves the deployed prefix replacement in samples and
returns the resulting source; ordinary synthesis passes an empty cache. Four
independent deployed-method probes cover 0, 1, 481 and 2400 cached samples, with
maximum waveform absolute error 2.2500753e-6 on the host. This boundary does not
connect overlap ownership to the sentence-stream HTTP service or establish
uninterrupted playback. Production cutover and listening acceptance remain open.

#### Inventory import acceptance, 30 October 2026

Seven focused host workshop checks passed, including complete-inventory rejection,
atomic import/reopen, exact replay after Apply, source-lock conflict, independent
native copies, nested-tensor corruption and dual selector membership. The success
fixture uses deliberately undecodable recording bytes with already prepared
tensors: any reference preparation would fail. The governed ARM release built.

The ARM executable then imported all four existing native-bound proof voices
inside an isolated 1536 MiB/no-swap Borg container, admitted both variants, and
passed exact replay. All 40 original predefined recordings retained their checksums.
A separate complete-40-voice inventory was rejected with no state commitment,
explicitly naming missing retained preparations, including Seven and the Doctor.
The reference selector also contains Gianna and Robert; no complete native-bound
production inventory is currently available. The retained synthesis capture is
T'Pol's voice and must never be relabelled as another crew member.

The initial read-only diagnostic hit source permission errors; its corrected
execution obtains the recording owner's actual UID/GID from source metadata.
No original permissions changed. Both failed receipts remain available. Native
import performed zero reference preparations. Neither production nor the existing
native HTTP trial was replaced or restarted; their start times remained
`2026-10-30T16:30:55.141432741Z` and `2026-10-30T18:40:59.218715566Z`.

Runtime source is `352e4252`; later candidate changes update this record only.
The ARM binary SHA256 is
`c8f0f201864f2843f91c7354b83b5b3bd563d5d51e85be1ee904d517a50070b9`.
Evidence and consumed supervisor claims are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-native-import-20261030/`;
the successful ARM receipt is `borg-result.json`. This proves the isolated import
transaction, not production cutover, listening quality or uninterrupted speech.

#### Three retained voices, 30 October 2026

The Commander narrowed the migration inventory to ship's computer
(`voyager-computer.wav`), tpol (`tpol_amplified_clean.wav`) and the doctor
(`the-doctor.wav`), and explicitly approved preparing these three once for
migration. The deployed native publication transaction prepared both existing
normalization variants for each recording. Duplicate admission then validated
the committed bindings without encoding again.

The existing ARM importer admitted exactly these three voices into an isolated
native workshop state. Its receipt reports `reference_preparations: 0`;
exact import replay also reports reuse and zero preparations. Both operations
left all prepared source assets unchanged, and all 40 original recordings
retain their checksums. The other recordings were not imported or deleted.

Borg's production Python owner was temporarily stopped to release its resident
models during preparation, then restored. Its real `/api/model-info` route
confirmed the Turbo model loaded on CUDA. The initial preparation attempt
failed before encoding because its user could not read the private recordings;
the first import attempt likewise could not enter private prepared directories.
Both failures are retained. The corrected operations run as the recordings'
verified owner, UID/GID 2002:2002, without broadening original permissions.
The import correction consumed existing tensors only.

The retained state is
`/home/muttleydosomething/myth-native-import-20261030-ownership-verified/retained-three-owner/state`
on Borg. Its preparation bindings and import/replay receipts remain beside it.
Saturn evidence and consumed supervisor claims remain under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-native-import-20261030/`;
the final receipt is `retained-three-result.json`. The importer uses the already
built ARM binary pinned above; no new build or service artifact was promoted.
Production speech still uses Python. Incremental waveform streaming, native
production cutover and human listening acceptance remain open.

#### Non-final conditioning acceptance, 30 October 2026

The native non-final conditioning boundary now trims its mask to the same span
as its features after withholding six lookahead frames. The deployed Python flow
instead retains the full encoder mask: its actual decoder rejects 534 feature
frames against 540 mask frames. The historical mismatch captures above remain
valid evidence of that original defect; native execution no longer preserves it.
Non-final prefixes leaving fewer than one generated frame reject before projection.
Reference mels, including an odd tail, determine the generated-frame count.

`capture_mel_oracle.py --align-nonfinal` executes the deployed flow with one
explicit AST correction: compute `h_lengths` from `h_masks[..., :h.size(1)]`.
Its receipt identifies this departure from the unmodified package. Eight native
conditioning comparisons passed with unchanged tolerances, exact mask/reference
values and replay. All preceding final captures remain byte-identical; for
preceding non-final captures only the mask changed. The short-prefix regression
covers odd/even references, one through five generated tokens, zero generated
frames and final replay after intervening non-final calls.

`capture_meanflow_oracle.py --nonfinal-only` consumes that corrected capture and
supplies explicit full fixed noise to the actual deployed CFM. The focused
`meanflow_replay_deployed_parity` test passed all three partial cases, per-step
derivatives, malformed-mask rejection, replay and CLI/library identity. This
three-case capture is not the input for the separate native-randomness test.
Maximum host full-mel absolute error was 1.5258789e-5, inside the existing bound.

The ARM CLI then fed its own native conditioning outputs directly into native
CUDA meanflow for retained, short odd-reference and short even-reference cases.
Generated frame counts were 131, 1 and 2. Maximum full-mel absolute error was
1.430511474609375e-5; zero-audio rejection also passed at the real CLI boundary.
Each isolated container retained the 1536 MiB/no-swap limit. No reference was
prepared; neither production nor the native HTTP trial was replaced or restarted.

Runtime source is `4e507fd1`; ARM binary SHA256 is
`83dcb931a3479a05ce2312249b389a6bf3c8e0a4dfac66943c291b6bf2e484b1`.
Evidence and consumed supervisor claims remain under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-nonfinal-conditioning-20261030/`;
the connected receipt is `native-receipts/result.json`.
This proves the conditioning-to-meanflow prerequisite only. Full attention over
the available token prefix does not establish prefix stability as tokens arrive.
Request noise retention, waveform overlap ownership, incremental HTTP delivery,
continuous playback, production cutover and listening acceptance remain open.

#### Request-owned streaming state, 30 October 2026

`sampling::RequestNoise` retains independent channel streams for meanflow and
HiFT excitation, including the unused member of an odd Box-Muller pair. Growing
channel-major tensors keep every preceding channel/time coordinate unchanged;
simply regenerating a flat tensor with the same seed would change its channel
strides. Harmonic phases retain the preceding length-independent policy.
Complete-response randomness and inference remain unchanged.

`hift::WaveformOverlap` owns the absolute source prefix, withheld PCM tail,
committed sample cursor and finalization state. Full-prefix cached HiFT calls
must return exactly the preceding source at those coordinates. A raised-cosine
blend reconciles only the undecided tail; already emitted samples never repeat.
The initial suppression applies at utterance origin. Finalization flushes once;
shrinking, repeated non-final, malformed and differently aligned updates reject.

`pipeline::WaveformStream` joins these owners to the existing immutable meanflow
and HiFT stages, checks cancellation between stages, and rejects changes in
reference-frame accounting. Cancellation retires this request by dropping it.
The caller remains responsible for token-prefix order and inference admission.
Models still use full attention over the available prefix: this explicit sample
commit policy does not establish equivalence with final utterance inference.

Three focused state regressions and two existing pipeline regressions passed.
A connected test using the admitted meanflow and HiFT checkpoints passed three
growing synthetic mel prefixes, exact sample offsets and final count, cancellation
after meanflow, and bit-identical replay in a fresh request. These synthetic
conditions prove native state composition, not speech quality or voice identity.
Evidence and consumed supervisor claims are under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-stream-state-20261030/`.
The classifier currently has no Borg-native rule; explicit lifecycle/streaming
proof applies. Cargo emitted the existing unrelated registry-cleanup permission
warning for `mediatype`; both focused proof commands exited successfully.
No reference was prepared. HTTP delivery has not yet adopted this state owner;
ARM execution, continuous playback, production cutover and listening acceptance
remain unverified for this boundary.

#### Codec-prefix transport acceptance and failed pacing gate, 30 October 2026

The isolated native workshop now uses the request-state owner through
`pipeline::render_stream` and the existing one-frame HTTP channel. T3 reaches EOS
and releases its engine first; cleaned codec prefixes advance in 32-token steps,
and only final conditioning includes the three established silence tokens.
Conformer, meanflow and HiFT retain sequential model/GPU-scope ownership. Only
request noise, source and the undecided waveform tail survive their boundaries.
This is incremental waveform delivery after token generation, not simultaneous
T3 and waveform inference or a causal-attention model contract.

Normal-speed requests emit several WAV fragments within one sentence. Other
speeds retain whole-sentence tempo processing; independently stretching each
fragment would reset the phase vocoder. Workshop joins run once per sentence,
not per fragment. Static catalogues may be empty when the configured workshop
owns the voice inventory, avoiding a dummy fourth retained voice.

Three pipeline checks, fourteen HTTP checks and six workshop checks passed.
The focused workshop test proves the exact PCM bytes of two sentence joins over
multiple fragments. The governed ARM release passed. Chromium then verified
four ordered fragments from one real tpol utterance, playback before completion,
final-download decoding, Stop, worker retirement and successful fresh speech.
Stop released the worker in 1616 ms. The native selector contains exactly the
ship's computer, tpol and the doctor; no voice preparation was performed.
The 1536 MiB/no-swap trial had zero OOM events, kills or restarts.

**Continuous playback failed; production cutover does not pass this gate.**
The 4.62-second utterance required 93156 ms, first audio arrived at
41462 ms in the browser, and the largest measured playback gap was
17.772666666666666 seconds. Full-prefix recomputation does not meet real-time
pacing, and the separate whole-WAV Hailstorm route has not adopted this NDJSON
transport. Authentic voice quality and audible join acceptance remain unverified.

A single temporary 3 GiB comparison, with the Python owner paused, produced
bit-identical PCM in 71057 ms with first audio at 20751 ms server-side. Memory
peaked at 2179227648 bytes and the memory-limit event count stayed unchanged;
the 1536 MiB run had accumulated 18263 such events. Later starvation still
reached 17.650999999999996 seconds. Removing reclamation pressure helps initial
generation but does not solve waveform throughput. Both the original trial
ceiling and production Python service were restored after this comparison.

The native trial uses the imported state's verified UID/GID 2002:2002 and its
existing video/render groups, without broadening asset permissions. The initial
resource preflight rejected low free RAM before changing either service. The
first activation then rejected root's write to the owner's lock with capabilities
dropped; a subsequent browser harness incorrectly counted its selector's `none`
placeholder. Each failure remains recorded, and automatic recovery restored the
preceding trial and Python owner. Corrected acceptance and comparison watches
were consumed after inspecting their terminal results.

Runtime source is `095be8b8`; ARM binary SHA256 is
`941ddda3039e940cc519c3a1c69c291014b6eee4e85168ce48785b4f6cd2a895`.
The isolated image is `hailstorm/myth-native:prefix-095be8b8` on the existing
8766 trial route. Evidence is under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-token-stream-20261030/`:
`browser-prefix-corrected/browser-prefix.json`, `runtime.json`,
`memory-comparison.json` and the consumed supervisor claims. The browser's
download `bytes` field is zero because `decodeAudioData` detached the input
buffer before that field was read; decoded sample count/rate/channels passed.
Production synthesis remains Python. The gateway, Redis and Qdrant were not
restarted.

#### Bounded HiFT CUDA batching, 30 October 2026

Holmes authorised optimisations during conversion rather than deferring them
until conversion ends. The detailed rationale and acceptance contract are in
the existing Vault register `agents/watson/documents/tts-development.md2`, under
“Immediate native HiFT batching optimisation”. No new planning authority exists.

HiFT's existing ordinary/dilated and transposed convolution now packs up to
1024 adjacent time rows per CUDA product, bounded by 8 MiB for that call's
combined packed input and result. The executing CUDA scope selects this geometry;
CPU execution retains 128 rows. Larger batches reuse each uploaded weight matrix
for more rows and reduce GEMM launches and blocking download boundaries. They
preserve indexing, FP32 precision and transposed-convolution overlap-add order.
No pointer-based weight cache, model/voice preparation, attention change, service
contract change or retained-model memory expansion is introduced.

The 8 MiB budget does **not** cap total GPU memory: independent input/weight/result
buffer high-water marks and host allocations coexist. Runtime measurements remain
the memory authority. Six focused HiFT checks passed, including independent scalar
convolution/transpose references across 128/455/1024-row boundaries, uneven tails,
stride, padding and dilation. The governed ARM release built successfully.

A paired fixed-noise 137-frame Borg fixture, with the same assets/backend and
1536 MiB/no-swap ceiling, produced these single observations:

| Measurement | Previous binary | Batched binary |
| --- | ---: | ---: |
| CLI duration, including model admission | 4046 ms | 3222 ms |
| GPU matrix calls | 4722 | 659 |
| Host-to-device bytes | 2869829164 | 1750619692 |
| Device-to-host bytes | 224792236 | 224792236 |
| Peak reusable GPU scratch | 12853120 bytes | 21837824 bytes |

Separate admission/kernel timing was not captured; these are not pure kernel
latencies or percentile estimates. The transfer/call reduction directly confirms
the intended mechanism. Do not infer a total RAM reduction from cgroup peaks:
shared page charging and warm asset pages differ between sequential containers.
All paired/additional diagnostic runs had zero memory-limit and OOM events.

The candidate passed the existing 3e-4 absolute plus 3e-4 relative deployed-oracle
tolerance on the retained/reversed 137-frame, single-frame and odd-frame fixtures.
Its largest waveform difference from the preceding binary was
9.950250387191772e-6; pitch differed by at most 2.2351741790771484e-7 and source
samples were identical. Fresh-process replay was bit-identical to the candidate.
The waveform is not bit-identical to the old binary; changing GEMM geometry can
change FP32 rounding. These numerical checks do not replace listening acceptance.

The existing isolated workshop then delivered four real tpol fragments before
completion, exercised running Web Audio, and decoded the final download at the
expected 110880 samples/24 kHz/mono. Browser duration was 70485 ms, first server
audio 21367 ms, and maximum playback gap 17.004666666666665 seconds. The final
231-frame prefix used 1055 HiFT calls and uploaded 2886921388 bytes, versus the
previous receipt's 7946 calls and 4803761324 bytes. Its HiFT duration was 4503 ms
versus the earlier 5573 ms. Overall request timings are not a controlled speedup
comparison: initial loading and reclamation vary materially.

The first receipt collector read only Docker standard output; stage receipts
were on standard error, so it incorrectly found zero operations. The supervised
run failed after the browser had passed, restored the previous trial and restored
Python. The collector was corrected to combine both streams and select the newest
request from the retained container history. Only the missing memory/request proof
was rerun; the passing browser proof and built binary were not repeated.

The corrected 1536 MiB run reproduced the new browser PCM hash exactly, with no
OOM, OOM kill, swap or restart. It reached the ceiling and recorded 3785 max-limit
events. Total request duration was 105678 ms, first server audio 58054 ms and
maximum starvation 16.345 seconds. Token-stage time rose from 7382 ms in the
browser run to 44515 ms; waveform-prefix time was 61855/59939 ms respectively.
HiFT's final-prefix call count remained 1055 in both runs. Memory pressure and
meanflow/full-prefix work still prevent continuous playback. This optimisation
does not admit production cutover or establish the full conversion as finished.

Runtime revision is `ead89ddb3e6e8d0ffcdaa91d05c1e0ef6dbf2b71`; binary SHA256 is
`ff91b1ba62f752429ae9d00a86bece98399ec643e3581d1d05336ab7802c7154`.
The isolated image is `hailstorm/myth-native:hift-batch-ead89ddb`; the preceding
trial container remains available for recovery. Python production is restored.
No reference was prepared; the gateway, Redis and Qdrant were not restarted.
Listening, continuous pacing and production cutover remain open.

Evidence is under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-hift-batching-20261030/`:
`comparison/result.json`, `browser/browser.json`, `memory.json`,
`runtime-corrected.json`, `deployment.json`, classifiers and exact consumed
supervisor claims. The unrelated existing Cargo registry-cleanup permission
warning for `mediatype` did not prevent either focused checks or the release build.

#### Native timing boundaries, 30 October 2026

The existing complete-request and streaming-prefix receipts now separate
`*_model_load` from `*_inference` in `stages_ms`, including TensorRT token
generation and the Conformer/meanflow/HiFT boundaries. These subspans overlap
the original inclusive stage totals; do not add them to those totals. Model
loading includes the existing checksum/asset admission. The `s3-meanflow` and
`s3-hift` proof commands likewise report separate model-load and execution times.

Each CUDA stage's existing `waveform_compute` receipt records host wall-clock
nanoseconds for context/library setup, matrix buffer allocation, blocking input
uploads, GEMM plus blocking result download, and teardown. The combined
`gemm_and_download_ns` includes GPU completion and transfer: it is deliberately
not labelled a pure kernel measurement. No new synchronisation, model retention,
weight cache, numerical operation or inference iteration is introduced.

The source comparison found different execution lifetimes despite numerical
parity: Python retains its TensorRT engine across requests; the native pipeline
retires it before waveform generation. Both retain autoregressive KV within a
request and consume prepared voice state. Native waveform matrices upload both
operands on every call and download each result; streaming admits the waveform
models again for every growing prefix. The existing stochastic source/tail
cache does not eliminate the full-prefix meanflow calculation. Treat these as
profiling leads until timing receipts establish their individual costs.

The ARM diagnostic executable passed the existing GPU failure/isolation and
pipeline handoff/WAV checks. Borg replay returned these single observations
using the existing retained-final meanflow and HiFT fixtures:

| Host wall time | Meanflow | HiFT |
| --- | ---: | ---: |
| Model admission | 2573 ms | 736 ms |
| Execution including CUDA lifecycle | 7714 ms | 2770 ms |
| Input uploads | 506.460 ms | 257.497 ms |
| GEMM plus result download | 1116.433 ms | 429.112 ms |

Both numerical oracle comparisons and timing-accounting checks passed, with
zero reference preparations. Meanflow's largest full-output error was
1.239776611328125e-5; HiFT's waveform error was 5.409121513366699e-6.
Meanflow completed with exit zero and no OOM, but its 600 MiB envelope recorded
1961 max-limit events: the resource gate failed and its timings are explicitly
pressure-affected. Initial 640 MiB admission was refused before any inference.
Meanflow was not repeated. HiFT's separately executed 600 MiB fixture had no
limit or OOM events. These observations are neither a matched Python/native
benchmark nor evidence that uploads dominate total latency. Meanflow spent
5563.035 ms outside the measured CUDA boundaries; this includes host operations,
reclamation and other unseparated work, not a pure CPU-kernel measurement.

The binary SHA256 is
`3fe4a15b3191757712ca0a038b1993be6b6ff9c0b3115e07cb73bd2bc916076e`.
Evidence is under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-native-timing-20261030/`:
`meanflow.json`, `meanflow-verification.json`, `result.json` and the retained
proof scripts. Sibling supervised `-check`, `-borg`, `-borg-corrected` and
`-hift` directories retain the build/admission/resource outcomes and consumed
claims. No running service was replaced or restarted. Per-request service
activation, a pressure-free meanflow measurement, matched cold/warm Python/native
timings and finer attribution of host work remain unverified.

#### Service-owned TensorRT engines, 30 October 2026

The authorised plan and rationale are committed in the existing Vault register
`agents/watson/documents/tts-development.md2`, under “Persistent engines and
device-resident waveform optimisation”. This first slice changes token-engine
lifetime only. Meanflow/HiFT graph export, device-resident intermediates, waveform
model retention and incremental-prefix semantics remain separate work.

The additive native ABI separates immutable engine/runtime ownership from each
request's context, stream and KV. Sessions retain shared engine ownership until
their own teardown; plugin bridge code remains process-resident. Each engine
records its CUDA device and binds it before creating a context on a later worker
and before engine teardown. Legacy `cb_open` retains its CLI behavior.

Rust's existing service owns one engine cache keyed by canonical engine path and
admitted SHA256. Hits still verify actual engine content and process bridge
identity. Changed pins evict the preceding entry before replacement; failed
replacement leaves an empty, retryable cache. Every utterance supplies a fresh
prompt, context and KV, then releases those allocations before waveform work.
No new daemon, queue, numerical operation or voice preparation is introduced.

Unload now clears the owner on a blocking worker while retaining the inference
permit; successful reload invalidates the old cache, and failed admission keeps
the preceding state. Speech rechecks loaded/stopping state after acquiring the
permit, closing the race with an unload during voice validation. Native operation
receipts expose `token_engine_cache_hit`; cache hits still include checksum and
fresh-context work in `token_model_load`.

Checked: 20 focused bridge/cache, pipeline and HTTP checks. The fake native
boundary exercises actual engine/context counts, checksum replacement, bridge
rebinding refusal, context/inference failure, cancellation and teardown ordering.
The governed ARM release built; the native bridge built directly on Borg.
The classifier requires focused proof and no browser states for this change.

The real isolated HTTP route returned six complete WAVs and admitted one
cancelled streaming request. Engine operations were cold, warm, warm after a
voice change, warm after returning to the original voice, warm for cancellation,
warm on recovery, then cold after explicit unload/reload. All five identical
tpol requests produced the same PCM16 WAV SHA256
`eaf369d484097013de49d899acbc3822e7b802f3550b4b741638467c65057a34`,
with 28320 samples. The changed ship's-computer request produced different audio.
Unloaded speech returned HTTP 503. No reference was prepared.

The first 3 GiB trial passed those functional checks but failed its resource gate:
1061 cgroup max-limit events, no OOM or kill, and a peak at the ceiling. The
preceding native trial and Python owner were automatically restored. Retained
samples show at least 4271243264 host bytes available, so the cgroup ceiling
bound this run despite remaining host headroom. Its cold/warm request durations
were 19583/17489 ms; they are pressure-affected observations.

Only the missing RAM gate was repeated, with unchanged artifacts, an explicitly
different 4 GiB/no-swap envelope, and anonymous/file/kernel memory accounting.
Cold, warm and unload/reload requests reproduced the identical WAV. No voice
switching, cancellation, source checks or builds were repeated. The resource
gate passed with zero max-limit/OOM/kill/swap events; recorded cgroup peak was
2068303872 bytes and minimum host availability 4713943040 bytes. One sampled
peak comprised 768450560 anonymous, 1164054528 file-backed and 8658944 kernel
bytes. Page charging differs between runs; this does not prove retention uses
less RAM or that a smaller ceiling is admitted.

The corrected run's cold/warm/reloaded durations were 68657/17715/18003 ms.
These are single observations, including admission and context setup, with no
matched Python baseline. They establish reuse, not a general speed multiplier
or continuous playback. The aggregate cgroup counter did not fall immediately
on unload; physical RAM return is not proven by the owner teardown check.

The candidate is activated only in the existing isolated trial, using image
`hailstorm/myth-native:engine-cache-1d4308d0` and the measured 4 GiB profile.
Its preceding container remains available for rollback. Prepared workshop state
was copied for the trial with new bridge pins; the preceding owner's state was
unchanged. After acceptance the engine was explicitly unloaded before restoring
Python, then assets were lazily re-admitted without retaining token weights.
Production synthesis remains Python. No gateway, Redis or Qdrant restart occurred.

Runtime revision: `1d4308d0b218d1728810e22213be406c5ca7e820`.
Rust binary SHA256:
`b8c345d9d5c4491bee6e5423316d55e6790d0973a2514880567e9168dcc01317`.
Bridge SHA256:
`ac250c8f960dc5d0221df2511dbed19266328109ee97beaf1eaee5c6c6c2a4f0`.
The Rust build began at `b9805924`; only independently built native bridge code
changed afterward. Rust sources and the verified executable were unchanged.

Evidence is under
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-engine-retention-20261030/`:
`requests.json`, `runtime-operations.log`, `memory-samples.json`,
`resource-4g/result.json`, `resource-4g/restoration.json`, `deployment.json`,
classifier and exact claimed build/failed-acceptance/corrected-resource runs.
Nsight Systems 2026.3.1 is installed on Borg; a meanflow trace has not yet run.
Whole waveform TensorRT graphs, the former trial ceiling, continuous playback,
listening acceptance and production cutover remain unverified.

#### Complete meanflow TensorRT graph, 30 October 2026

The existing meanflow owner now admits an additional pinned export kind,
`s3gen_meanflow_trt_fp32`. Its engine contains the entire deployed estimator
and Euler update. Inputs upload once, device buffers ping-pong between Euler
steps, and only the final full mel downloads. The derivative output stays on
the device except during the independent numerical oracle. Normalisation,
softmax, residuals, convolution and attention no longer return host vectors
between operators. The original FP32 tensor export remains available.

`native/export_meanflow.py` is build-time tooling, separate from the independent
oracle capture: it consumes the already pinned oracle tensors and imports only
the deployed estimator, without preparing a voice or starting the speech service.
Run with `python3 -S`, an explicit `--opt-frames` and `--max-frames`, and `--build`.
`--build-only` resumes an existing checksum-verified ONNX export. Compilation
uses Borg's established `/host-cuda`, `/host-trt` and `/trt-python` mounts, disables
TF32, uses FP32, and limits TensorRT tactic workspace to 256 MiB. Its private
loader route contains NVIDIA tactic-resource links only; the host libc directory
must never be added to the container's library search path. Export-time Torch
allocations retire before graph compilation.

The graph manifest, engine and bridge form one self-contained admitted directory:

```json
{
  "schema_version": 1,
  "decoder_kind": "s3gen_meanflow_trt_fp32",
  "source_sha256": "CHECKPOINT_SHA256",
  "engine": {"file": "meanflow.engine", "sha256": "ENGINE_SHA256"},
  "bridge": {"file": "libchatterbox_trt.so", "sha256": "BRIDGE_SHA256"},
  "max_frames": 1700
}
```

Both token and meanflow admission must select that same canonical bridge path.
The graph owner checks dense FP32 device I/O, retains one exclusive execution
context/nonblocking stream, and reuses bounded high-water input/output buffers.
Uploads share the execution stream; every error path drains queued work before
returning Rust slices. Native inference failure retires that context. All public
shape, finite-value, mask, step-count, time and reference-cropping checks remain
at the existing model boundary. Out-of-profile requests fail explicitly; there
is no crop, precision reduction or automatic fallback. This admitted graph's
profile is 1–1700 frames, with an optimum at 512. The maximum covers the current
600-token generation ceiling plus the longest retained reference of 250 tokens:
`2 * (600 + 250)`. The initial 1024-frame diagnostic profile did not cover that
existing contract and is superseded. A different profile requires
another explicit export and resource admission.

The pipeline releases the owner before HiFT and between streaming prefixes,
preserving sequential model ownership. This slice establishes device residency
within meanflow decoding, not waveform-model retention across requests. Engine
checksum work and fresh token-context allocation remain separate optimisation
work. No new service, scheduler or reference-preparation route was introduced.

Focused native-ABI replay, pipeline and stream-state checks passed. The ARM release
and Borg bridge built. The existing deployed-oracle test accepts the optional
`CHATTERBOX_MEANFLOW_GRAPH` and `CHATTERBOX_MEANFLOW_GRAPH_SHA256` pins while
keeping its expected arrays and tolerance unchanged. All five final and three
corrected non-final cases passed, including per-step derivatives, rejected-input
recovery, changing frame lengths, exact replay, cropping and CLI/library identity.
The largest full-mel error was 0.0000371933, inside `3e-4 + 3e-4 * abs(expected)`.
The retained 540-frame case uploaded 520896 bytes and downloaded 172800 bytes;
its graph-call wall time was 314873834 ns. These timings exclude model admission
and do not establish complete-request speed. The numerical container peaked at
1044033536 bytes with zero limit/OOM/swap events and at least 5002149888 host
bytes available. Reference preparations: zero.

Runtime source is `549a98a3`; ARM binary SHA256 is
`03afefeab644900d24c30991de4eca2ff0ba0465b21f85c2b0a9c080f4cb5173`.
The initial 1024-frame diagnostic engine SHA256 is
`6b5b23f348054e5c189989efddf700304609059dca65c2d592bd687837d4ff27`.
Evidence authority is
`/home/muttleydosomething/.local/state/hailstorm/engineering/myth-meanflow-trt-20261030/`,
mirrored on Borg at `/home/muttleydosomething/myth-meanflow-trt-20261030/`.
HTTP activation, matched Nsight attribution and production cutover are not
established by the numerical gate alone.

The final service-contract engine is
`ed90915257b0e9b295f4a57460565cad130373cbe4be142f2dbc6a255c3e5bf8`.
Its 1700-frame profile passed the same eight numerical cases and derivatives
without relaxing tolerances. A separate maximum-boundary call returned a finite
`[1,80,1700]` full mel and `[1,80,1200]` generated mel, exercising a synthetic
500-frame reference boundary while preserving the existing 600-token ceiling. It uploaded
1639136 bytes and downloaded 544000 bytes. That container peaked at 1605812224
bytes, with zero limit/OOM/swap events. No voice was prepared.

The additive bridge also now uses one process-lifetime TensorRT logger shared
by token and waveform runtimes. The first HTTP trial exposed repeated logger
registration warnings; sharing the logger removes their cause and prevents a
retired model owner from supplying another runtime's logger. Cold, warm and
unload/reload requests on the corrected bridge returned the same 28320-sample
WAV, with token cache states false/true/false, no logger warning, no reference
preparation and zero limit/OOM/swap events. The graph's numerical engine was
unchanged by this bridge-only correction.

The final 1700-frame release then passed the actual existing HTTP route, warm
Nsight capture, and a fresh unprofiled route replay. The retained seed-42 tpol
workload and both output lengths were unchanged. The initial and final profile
produced identical short and long WAV hashes; the graph changes FP32 rounding
relative to the preceding CPU/cuBLAS implementation, so its WAV hash differs
from that older binary. Listening acceptance remains separate.

| Warm traced longer sentence | Previous native | Final meanflow graph |
| --- | ---: | ---: |
| Complete WAV availability | 26.335265 s | 18.865618 s |
| Meanflow execution | 9130 ms | 513 ms |
| Meanflow including admission | 11481 ms | 3881 ms |
| Meanflow boundary H2D + D2H | 5865170176 bytes | 814392 bytes |
| Whole-trace H2D + D2H | 9849872348 bytes | 4279126348 bytes |
| Meanflow admission | 2302 ms | 3255 ms |
| Token engine/context admission | 4266 ms | 4311 ms |

These are individual matched traced observations, not percentile or median
estimates. The preceding capture used 6 GiB and this release 4 GiB; both had
zero max-limit/OOM/swap events. GPU kernel interval union changed from
1.769241096 to 1.959579721 seconds, while kernel-plus-copy union changed from
2.429563831 to 2.359994501 seconds. The wall-time improvement primarily removes
host-side graph work and transfers; it is not a claim that every GPU kernel sped
up. Whole-trace bytes include other stages and model admission, unlike the
explicit meanflow boundary counters. HiFT and Conformer still transfer host
intermediates. Repeated hashing, fresh token contexts and waveform admission
remain material costs; meanflow's new engine is not retained across utterances.

The final profiled container peaked at 3501125632 bytes with at least 3858534400
host bytes available. Its events and the fresh unprofiled route's events were
all zero for max-limit/OOM/swap. The unprofiled short request reproduced the
warmup WAV exactly. After proof, native weights were unloaded before restoring
the original Python service on CUDA, then native assets were lazily re-admitted.
The native trial remains idle/ready. Production synthesis remains Python.
Continuous streaming, listening acceptance and production cutover are not
verified by this slice. The gateway, Redis and Qdrant were not restarted.

The active image is `hailstorm/myth-native:meanflow-full-31fda930`; the preceding
native owner remains available for rollback. Runtime bridge source is
`31fda93083b16daa83c0d0d0db3a4f9ef3fa02da`, bridge SHA256
`11b14f5265e87b309eb464c50f43c1e8a7a392918220abdf346e61e0f568278d`.
The Rust binary remains the checksum stated above; no Rust code changed after
its build. Final assets are under Borg's existing evidence root at
`full-profile-release/`. Canonical deployment receipt is `deployment.json` at
that root's parent. Saturn's `full-profile/analysis.json` derives the table from
the retained SQLite capture and runtime receipts.

The initial tactic-library lookup, the installed binding's None-returning
`set_shape`, and SCP into a root-owned artifact directory failed before their
respective activation gates. Corrections used a selective NVIDIA library
route, the actual binding contract, and the established privileged publication
route; release permissions were not broadened. The first 1024-frame profile
was superseded after checking the existing 600-token service contract.
The unrelated existing Cargo `mediatype` cache-cleanup permission warning did
not fail the focused checks or ARM build. Supervisor terminal results were
inspected and claimed; no already-passing unchanged numerical/build programme
was repeated for the logger or publication corrections.

#### Opt-in call and loop attribution, 30 October 2026

The existing complete/streaming pipeline roots now accept the diagnostic
environment flag `HAILSTORM_NATIVE_TRACE=1`. A private shared collector buffers
host-monotonic spans for existing admission, token, Conformer, HiFT and GPU
matrix owners. It emits one `native_call_trace` JSON receipt to stderr when the
root exits, including failure/unwind. Each span records its parent, start,
duration, completion and optional asset/layer name or iteration index. The
32768-span capacity is bounded; `dropped_spans` explicitly invalidates a complete
attribution claim if exceeded. Disabled spans allocate no labels, read no clocks
and acquire no locks. No numerical operation, model lifetime, cache, public HTTP
schema or CUDA synchronization changed. Scalar element loops are not individually
observed. A private helper is used because the existing pipeline timing ledger
has no reusable nested collector and the GPU owner does not own admission/token
diagnostics.

The trace separates admitted-asset reads, SHA256 and NumPy parse/validation;
streaming engine read-plus-SHA from native token-context creation; token prefill,
each decode and sampling iteration; Conformer blocks/attention/feed-forward;
HiFT pitch/excitation/spectrum/inverse and upsample/residual iterations; and
convolution packing, matrix allocation/upload/GEMM-plus-download and scatter.
Meanflow's native engine/context constructor remains one combined span.
Inclusive spans overlap children. Self time subtracts only immediate observed
children and includes untimed work and observer overhead; it is not pure CPU or
GPU execution time. Root duration excludes receipt serialization/output.

Checked: 39 focused library/cache/pipeline/stream checks passed, with one existing
intentional ignored test. Four new cases cover disabled tracing, nested timing
and root ownership, explicit overflow, and error/unwind cleanup. Built: governed
ARM release from `f5d4a13f`; executable SHA256
`f5aa6716ee132898b2c5b595f51f0eb8edc9728723ebd5bfe4f00d366c175613`.
Its exact supervisor result was inspected and claimed. No bridge rebuild was
needed.

The isolated HTTP diagnostic compared disabled and enabled tracing sequentially,
each with a cold warmup and warm short/long requests. All six WAVs reproduced the
admitted short/long meanflow-release hashes stated above. The long enabled trace
contains 11608 spans, all complete, with zero dropped spans. Both 4 GiB/no-swap
clones had zero max-limit/OOM/swap events; minimum host availability was
3955269632 bytes. The different peaks, 1614766080/2807652352 bytes, include shared
page charging and do not establish tracing's independent memory cost.

| Long-request host measurement | Observed time |
| --- | ---: |
| Cached token engine streaming read + SHA256 | 4063.633 ms |
| Fresh token execution context | 220.050 ms |
| Meanflow engine streaming read + SHA256 | 1914.828 ms |
| Meanflow native engine + context creation | 1198.298 ms |
| 476 admitted-asset SHA256 calls | 2230.830 ms |
| 466 NumPy parse/validation calls | 379.999 ms |
| 1040 HiFT convolution packing tiles | 1566.971 ms |
| 36 HiFT residual iterations, self time | 889.472 ms |
| HiFT source spectrum / inverse | 135.453 / 132.720 ms |

All non-overlapping native streaming-read-plus-SHA and admitted-asset SHA spans
total 8211.501 ms. Token cache admission is dominated by repeated checksum work,
not its 220 ms fresh context. Meanflow admission combines repeated checksum work
with engine/context construction. HiFT's packing and residual self time establish
substantial remaining host work; residual self time includes activation,
accumulation and other unseparated operations. HiFT inference was 3786.342 ms,
Conformer 644.190 ms and meanflow 510.385 ms in this trace.

After inference stopped, Python `_hashlib` read the exact same engine bytes with
the same 65536-byte buffer and reproduced both admitted checksums. Its token
engine read-plus-hash took 957.369 ms (191.752 ms read, 752.879 ms hash); meanflow
took 436.348 ms (70.475 ms read, 359.713 ms hash). Native streaming spans include
both reading and hashing, so their entire durations must not be called pure
hashing time. The ARM dependency fingerprint contains only `default,std` for
`sha2 0.10.9`; that installed dependency selects its AArch64 accelerated backend
only with `asm`, while Borg advertises SHA2 support. This supplies a specific
implementation lead without weakening checksum admission. Dependencies remain
unchanged; the speed of an accelerated native build is not yet measured.

The evidence changes the immediate priority: measure an ARM SHA backend with
unchanged checksum/admission semantics and byte-identical speech proof before
altering context lifetime, then continue the agreed resident HiFT graph. Fresh
token context creation is only 220 ms here. Native token inference was 2086.677 ms
across 113 decode iterations; Python's ordinary longer-request token receipt was
3176 ms for 103 generated speech tokens. Differing sequences prevent a strict
per-token equivalence claim, but these observations do not identify the native
sampling loop as the cause of the overall delay. No optimisation is implemented
by this diagnostic slice.

The corresponding Python diagnostic captured 18832/70656 selected model calls
for short/long speech, with no overflow or open spans and unchanged WAVs when
profiling. Long warm HTTP durations were Python 5.973776 s unprofiled and
7.443748 s profiled, native 17.631724 s disabled and 18.458292 s enabled. Both
profilers add overhead. Python's selected long HiFT/Conformer host spans were
688.9/240.3 ms; these asynchronous call boundaries are not isolated GPU times.
Python and native produce different codec sequences and 4.22/4.62-second audio,
so this remains a sentence-workload comparison. Production Python retains its
models and avoids native per-request model/checksum admission; its tensor path
also avoids native host packing/intermediate round trips. The prior CUDA trace
provides transfer evidence; these new host spans identify the additional work.

The original Python CUDA owner and native ready/idle owner were restored with
unchanged image identities. No reference was prepared. Production routing and
the original trial executable remain unchanged; this executable was used only
in disposable diagnostic clones. No gateway, Redis or Qdrant restart occurred.
Listening, continuous streaming, production cutover and dependency optimisation
remain unverified by this profiling slice.

Evidence: the existing `myth-meanflow-trt-20261030/call-timing/` engineering
authority contains `analysis.json`, `python/{result.json,runtime.log}` and
`native/{result.json,runtime.log}`, beneath
`/home/muttleydosomething/.local/state/hailstorm/engineering/`. The separate
`myth-call-timing-20261030/` build evidence retains the copied executable,
`arm-lib-sha2-fingerprint.json`, classifier and the claimed build supervisor
receipt. This extends the existing TTS development authority rather than
creating a parallel plan.

#### Retained engine admission and ARM SHA2, 30 October 2026

The existing service `EngineCache` now owns one token engine/idle execution
context and one meanflow graph/context. CLI and CPU-oracle models retain their
original lifetimes. The existing inference permit and cache mutex serialize
ownership; there is no second service or background inference owner. Token KV
and position are reset before returning a context to its idle slot. Failed token
steps discard their context; failed meanflow inference evicts its graph. Unload
retires both retained owners. Native token entry points attach their CUDA device
when an idle context moves to another blocking worker.

Complete-byte SHA256 still gates admission. Retained engines use their checksum
pin plus canonical path, device, inode, size, mtime and ctime (including nanoseconds)
as invalidation keys. Dependencies are observed before and after construction;
changes trigger fresh checksum verification and replacement. Same-length writes
with restored mtime cannot conceal changed ctime. Bridge code retains its existing
process-identity rule and cannot be rebound to changed native code. This is reuse
of an admitted in-memory instance, not a global exemption for future file reads.

The AArch64 dependency enables sha2 0.10.9's `asm-aarch64` runtime-dispatched
backend. It selects SHA2 instructions only when supported and keeps the portable
fallback. The final executable contains `sha256h`, `sha256h2` and `sha256su0`;
the dependency feature fingerprint is retained. No custom hash implementation
or integrity check removal is involved. Other model assets still pass their
existing admission checks on each request.

The established retained voice/text/seed HTTP trial produced identical accepted
short and longer WAV hashes in all eight requests: three uninstrumented and five
instrumented, including unload/reload and its next warm request. Warm traces had
no engine read/checksum or context-creation spans. Longer-request token admission
was 0.117 ms and meanflow admission 0.108 ms. Cold engine read-plus-SHA was
1035.477 ms for token and 487.301 ms for meanflow, versus the earlier software
observations of 4063.633/1914.828 ms. The native constructor spans remain separate.

| Warm HTTP observation | Previous | Retained admission |
| --- | ---: | ---: |
| Short, uninstrumented | 12.993855 s | 3.806543 s |
| Longer, uninstrumented | 17.631724 s | 8.391060 s |
| Longer, instrumented | 18.458292 s | 9.391652 s |

These are individual observations with identical native WAVs, not percentiles.
The instrumented longer request still spent 3794.700 ms in HiFT, including
1573.928 ms convolution packing and 889.088 ms residual-iteration self time.
Conformer inference was 612.966 ms, meanflow inference 507.247 ms and token
inference 2048.516 ms. Other asset SHA calls totalled 495.930 ms. Python's earlier
ordinary longer request remains 5.973776 s; different native/Python codec tensors
still prevent a strict inference parity claim. The next HiFT graph proof must
feed identical retained tokens/mel/noise to both implementations.

Checked: 40 focused library/cache/pipeline/stream regressions, with one existing
intentional ignored case. New/extended ownership cases cover actual context
open counts, cancellation/reset, inference and construction failure, changed
pins/sources, same-size/restored-mtime rewrites and explicit unload. AArch64
release built from `72cb9155` (ownership comments follow the tested `2bb0a57f`
runtime implementation), SHA256
`57255153e281f10eb2e8833262ecebb39ab46d81b201a9b414853ebf7d4deb50`.
The Borg bridge SHA256 is
`58bbeeffd658e234c107adb36d97bbb0518a6f3837f679b67e61c37c1dad7175`.
The classifier requires focused proof and no browser states for this slice.

The 4 GiB/no-swap disabled/enabled clones peaked at 2620960768/2257403904 bytes,
with zero limit/OOM/swap events and at least 3703357440 host bytes available.
The original Python CUDA and native ready owners were restored with unchanged
image identities after candidate proof. No reference was prepared. Production
routing remains Python; uninterrupted streaming, listening acceptance and native
production cutover are not verified here. Gateway, Redis and Qdrant are untouched.

Evidence on Saturn is the existing engineering authority's sibling
`myth-resident-admission-20261030/`: `analysis.json`, `proof/result.json`,
`proof/runtime.log`, `proof/{disabled,enabled}/`, `classifier.json`, feature
fingerprint and claimed supervisor receipts. Borg has the same task root;
release assets are its `release/`. Native-trial activation uses the existing
canonical `myth-meanflow-trt-20261030/deployment.json` authority with an exact
image/binary/bridge receipt and retained rollback container. The initial absent
supervisor parent and unprivileged Docker inspection were corrected through the
existing evidence directory and sudo routes. The unrelated Cargo mediatype
cache-cleanup permission warning did not fail checks or builds.

#### Retained prompt assets and strict-FP32 Conformer, 31 October 2026

The existing service EngineCache now also retains one admitted PromptAssets
instance and one complete Conformer engine/context. Prompt tables, tokenizer and
prepared voice projection are immutable; the speech table is shared with token
sessions through Arc rather than copied for every request. Only one prompt is
retained, so changing voice evicts its predecessor. Full-byte checksums remain
mandatory on admission. Original lookup paths plus canonical source identity,
including inode/size/mtime/ctime nanoseconds, invalidate every manifest/leaf and
detect symlink retargeting. Failed admission clears the old owner. Bridge
verification uses the same identity policy and immutable process-code authority.

The new build-time export extends the existing tensor/bridge boundary: FP32
[1,T,512] embeddings enter, [1,2T,512] features leave, and Rust constructs the
exact all-one mask. Six encoder blocks, causal 2x upsampling and four further
blocks remain device-resident. Relative-position indexing, padding, normalization
epsilons and the existing independent oracle are preserved. TensorRT TF32 is
disabled; the profile is 1..850 input frames, matching 1700 downstream frames.
The build workspace is explicitly 512 MiB: 256 MiB could not admit a 324 MiB
relative-attention tactic. The original ONNX/eager proof was retained for that
corrected build rather than repeated.

Complete and incremental synthesis share run_conformer. Mel projection retains
its existing numerical kernel. Unload clears prompt and encoder owners; native
failure drains queued work, poisons its context and evicts the cached graph.
No reference voice is prepared again.

The compiled encoder passed retained, reversed, one-frame and two-frame oracle
cases at the unchanged 2e-4 absolute plus 2e-4 relative tolerance. Maximum
absolute error was 0.000004529953. Five fixed-input warm replays had a
53.308576 ms median. Odd lengths, full 850-frame capacity, subsequent small
replay and profile/input rejection passed; capacity probes establish shape,
finite output and replay, not an independent numerical oracle at every length.

Five warm HTTP requests per text, excluding warmups:

| Sentence | Previous native median | Retained prompt/Conformer median |
| --- | ---: | ---: |
| Ready, Holmes. | 2.633741281 s | 0.956557290 s |
| All systems are operating within normal parameters. The ship is ready to proceed. | 4.657704721 s | 2.891217864 s |

Previous values are retained accepted HiFT evidence, not rerun baselines. These
measure complete-WAV availability. Native sample counts remain 28320/110880;
maximum difference against retained native PCM is one PCM16 step under the
unchanged waveform tolerance. Prompt-admission stage medians are 1 ms for both
texts; Conformer admission is 0 ms warm, inference 37/59 ms and total encoder
conditioning 95/119 ms. Nested stage observations must not be added twice.

Checked: 36 focused host regressions, compiled encoder oracle, exact seeded
replay, switching to the already-prepared Doctor voice and back, streaming
cancellation after audio, next-request recovery and unload/reload. The 4 GiB
no-swap HTTP container peaked at 4122562560 charged bytes, with zero limit/OOM
events; minimum host available memory was 2961305600 bytes. Retention increases
charged peak relative to the preceding trial. The isolated encoder oracle
peaked at 756875264 bytes; its cgroup was retained until peak/events collection.
The classifier has no rules for this independent Borg component; explicit
runtime/lifecycle proof was enforced despite its empty risk-class result.

Evidence authority:
/home/muttleydosomething/.local/state/hailstorm/engineering/myth-conformer-trt-20261031/
contains analysis.json, accept-result.json, parity.log, runtime.log, release.json
and claimed supervised build/acceptance receipts. Integration and activation
receipts determine delivery state. Runtime build source is fff0a0b3.
Production routing remains Python; human listening, continuous-playback pacing
and native production cutover are not verified by this slice.


## Request-local token buffer reuse (2026-10-31)

The token session now reuses embedding, position, diagnostic-output and two
separate KV banks during an utterance. KV allocation capacity grows in 128-frame
blocks; logical tensor shapes and compact KV strides remain unchanged. Only an
output bank may grow. Reset, cancellation and teardown release all scratch before
waveform work, so no request KV is retained by the idle context. Sampling,
precision, engine graphs and the admitted executable are unchanged.

The clean Nsight diagnostic recorded 5996 CUDA allocations and 5997 releases,
with 739.401824 ms combined CUDA runtime API time across the longer request.
API time includes synchronization and must not be added to wall/GPU time. The
first 4 GiB profiled capture hit its memory ceiling without OOM; the corrected
6 GiB diagnostic peaked at 4682207232 bytes with no limit/OOM/swap events.
Neither profiled duration is an ordinary latency baseline.

Five unprofiled warm HTTP requests per sentence, warmups excluded:

| Sentence | Previous complete-WAV median | Token-buffer median | Token stage before / after |
| --- | ---: | ---: | ---: |
| Ready, Holmes. | 0.956557290 s | 0.849956006 s | 482 / 373 ms |
| All systems are operating within normal parameters. The ship is ready to proceed. | 2.891217864 s | 2.314435981 s | 2039 / 1461 ms |

Every native comparison had identical PCM and unchanged 28320/110880 sample
counts. The existing numerical tolerances were not relaxed. Checked: 12 host
regressions; the compiled native regression replayed all logits over 257
teacher-forced steps across capacity boundaries, then tested reset, cancellation
and context reuse. Voice switch/recovery, streaming cancellation and unload/reload
also passed. Unprofiled HTTP peak was 3612454912 charged bytes within 4 GiB,
with zero limit/OOM/swap events; minimum host available was 2949558272 bytes.
These charged peaks include reclaimable pages and do not establish an equal
reduction in resident model memory against the preceding run.

Evidence authority:
/home/muttleydosomething/.local/state/hailstorm/engineering/myth-token-loop-20261031/
contains profile-analysis.json, accept-result.json, parity.log, runtime.log,
classifier.json and claimed supervised build/acceptance receipts. Native bridge
source is 63d10bfb; the admitted executable and all engine bytes are reused.
Integration and activation receipts determine delivery state. Production routing
remains Python; human listening, continuous-playback pacing and native production
cutover are not verified by this slice.

## Native production launcher (2026-10-31)

`borg/chatterbox/start-gpu.sh` now defaults to the admitted token-buffer native
image and release. It verifies the executable, bridge and catalog checksums,
then creates `chatterbox-tts` on Borg's existing `10.200.0.2:8004` listener.
The bind address is `0.0.0.0:8004`, matching Python's existing socket so both
the local media coordinator and LAN reverse proxy can connect.
The pinned `service-production-chatterbox.json` is derived from the admitted
trial catalog by adding the existing Task Route model ID `chatterbox` to
`model_aliases`; all asset, preparation and deletion-authority pins are retained.
The trial catalog remains unchanged. This production catalog is a deployment
artifact beside the admitted release, with its checksum pinned by the launcher.
The 4 GiB/no-swap envelope, read-only assets, unprivileged runtime and Service
Multiplex ownership remain in force. No Task Route, Caddy or gateway change is
required. `MYTH_RUNTIME=python` retains the original launcher for explicit
recovery; the retained production Python container is the exact rollback owner.

Activation must stop the isolated native trial before reusing its prepared-state
authority, retain the previous Python owner under a recovery name, migrate only
supported workshop settings, and exercise ordinary HTTP and browser synthesis.
The three imported voices must retain their existing preparation pins. Deployment
and acceptance receipts are retained in the existing engineering evidence root
under `myth-native-cutover-20261031`. Launcher integration alone does not prove
activation, human listening acceptance or continuous-stream pacing.
