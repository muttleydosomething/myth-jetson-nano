#!/usr/bin/env python3
"""Actual deployed VoiceEncoder oracle, CPU-only in a separate bounded container.
Uses new deterministic audio fixtures; never re-prepares a retained service voice.
Only ve.safetensors is loaded. No T3, S3Gen, CUDA, or service is constructed.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

assert sys.flags.no_site
sys.path.append("/usr/local/lib/python3.10/dist-packages")
import numpy as np
import torch
from safetensors.torch import load_file

torch.set_num_threads(2)
source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox")
for name, path in [
    ("chatterbox", source),
    ("chatterbox.models", source / "models"),
    ("chatterbox.models.voice_encoder", source / "models/voice_encoder"),
]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
module = importlib.import_module("chatterbox.models.voice_encoder.voice_encoder")
mel_module = importlib.import_module("chatterbox.models.voice_encoder.melspec")
checkpoint, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=False)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(name, value):
    path = output / (name + ".npy")
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    np.save(path, np.ascontiguousarray(value, dtype="<f4"), allow_pickle=False)
    return dict(file=path.name, sha256=sha(path))


model = module.VoiceEncoder()
model.load_state_dict(load_file(checkpoint), strict=True)
model.eval()
# Similarity weight/bias are training-only and unused by inference.
tensors = {
    name: export(name, value)
    for name, value in model.state_dict().items()
    if name.startswith(("lstm.", "proj."))
}
tensors["mel_basis"] = export("mel_basis", mel_module.mel_basis(model.hp))
manifest = dict(
    schema_version=1, encoder_kind="t3_voice_encoder_fp32",
    source_sha256=sha(checkpoint), tensors=tensors,
)
(output / "voice-encoder.json").write_text(json.dumps(manifest, indent=2) + "\n")
cases = []
# Different lengths exercise partial padding, rate=1.3 frame-step rounding,
# coverage inclusion/exclusion and the second normalization across partials.
for name, length, pad in [
    ("one-sample", 1, 0),
    ("silence", 3200, 0),
    ("short-tone", 8000, 0),
    ("one-partial", 25440, 0),
    ("coverage-below", 32480, 0),
    ("coverage-above", 32640, 0),
    ("trimmed-chirp", 48000, 4096),
]:
    t = np.arange(length, dtype=np.float64) / 16000
    wav = (0.17 * np.sin(2 * np.pi * (173 * t + 57 * t * t))
           + 0.05 * np.sin(2 * np.pi * 517 * t)).astype(np.float32)
    if name == "silence":
        wav[:] = 0
    if pad:
        wav = np.pad(wav, (pad, pad * 2))
    trimmed, indices = module.librosa.effects.trim(wav, top_db=20)
    mels = mel_module.melspectrogram(trimmed, model.hp).T.copy()
    partials = module.stride_as_partials(mels, model.hp, rate=1.3)
    with torch.inference_mode():
        partial_embeds = model(torch.from_numpy(partials.copy()))
        embedding = model.embeds_from_wavs([wav], sample_rate=16000)
    cases.append(dict(
        name=name, audio=export(name + "-audio", wav),
        trim=indices.tolist(), mels=export(name + "-mels", mels),
        partials=len(partials), partial_embeddings=export(name + "-partials", partial_embeds),
        embedding=export(name + "-embedding", embedding),
    ))
oracle = dict(
    manifest_sha256=sha(output / "voice-encoder.json"),
    source_hashes={
        str(path.relative_to(source)): sha(path)
        for path in [
            source / "models/voice_encoder/voice_encoder.py",
            source / "models/voice_encoder/melspec.py",
            source / "models/voice_encoder/config.py",
        ]
    },
    cases=cases, cuda_initialized=torch.cuda.is_initialized(),
    retained_voice_preparations=0,
)
assert not oracle["cuda_initialized"]
(output / "oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")
print(json.dumps(dict(cases=len(cases), manifest_sha256=oracle["manifest_sha256"],
                      cuda_initialized=False, retained_voice_preparations=0)), flush=True)
