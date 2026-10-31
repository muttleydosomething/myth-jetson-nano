#!/usr/bin/env python3
"""Independent deployed S3 v2 codec proof. CPU only, in a bounded isolated container.
No service, retained reference preparation, decoder, speaker encoder or network.
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
from safetensors import safe_open

torch.set_num_threads(2)
base = Path("/usr/local/lib/python3.10/dist-packages")
for name, path in [
    ("chatterbox", base / "chatterbox"),
    ("chatterbox.models", base / "chatterbox/models"),
    ("chatterbox.models.s3tokenizer", base / "chatterbox/models/s3tokenizer"),
    ("s3tokenizer", base / "s3tokenizer"),
]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
module = importlib.import_module("chatterbox.models.s3tokenizer.s3tokenizer")
core = importlib.import_module("s3tokenizer.model_v2")
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


# Assign mapped checkpoint tensors to a meta model: do not allocate a second
# half-gigabyte copy merely to discard its randomly initialized parameters.
with torch.device("meta"):
    model = module.S3Tokenizer()
state = {}
with safe_open(checkpoint, framework="pt", device="cpu") as weights:
    for key in weights.keys():
        if key.startswith("tokenizer."):
            state[key.removeprefix("tokenizer.")] = weights.get_tensor(key).float()
# These analysis/position buffers are deliberately absent from the checkpoint.
import librosa
state["_mel_filters"] = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128))
state["window"] = torch.hann_window(400)
model.load_state_dict(state, strict=True, assign=True)
model.encoder.freqs_cis = core.precompute_freqs_cis(64, 2048)
model.eval()
tensors = {name: export(name, value) for name, value in model.state_dict().items()}
tensors["rotary_cos"] = export("rotary_cos", model.encoder.freqs_cis.real)
tensors["rotary_sin"] = export("rotary_sin", model.encoder.freqs_cis.imag)
manifest = dict(schema_version=1, encoder_kind="s3_tokenizer_v2_25hz_fp32",
                source_sha256=sha(checkpoint), tensors=tensors)
(output / "s3-tokenizer.json").write_text(json.dumps(manifest, indent=2) + "\n")
cases = []
with torch.inference_mode():
    for name, length in [
        ("minimum", 201), ("below-token-hop", 639), ("token-hop", 640),
        ("above-token-hop", 641), ("silence", 3200), ("short-chirp", 12800),
        ("long-segment-boundary", 480160),
    ]:
        t = np.arange(length, dtype=np.float64) / 16000
        wav = (0.17 * np.sin(2 * np.pi * (173 * t + 57 * t * t))
               + 0.05 * np.sin(2 * np.pi * 517 * t)).astype(np.float32)
        if name == "silence":
            wav[:] = 0
        audio = torch.from_numpy(wav).unsqueeze(0)
        mels = model.log_mel_spectrogram(audio)
        codes, lengths = model([wav])
        case = dict(name=name, audio=export(name + "-audio", wav),
                    mels=export(name + "-mels", mels),
                    codes=codes[0, :lengths[0]].tolist(), tokens=int(lengths[0]))
        # Hidden and quantizer logits localize arithmetic error before discrete FSQ.
        if mels.shape[-1] <= 3000:
            hidden, _ = model.encoder(mels, torch.tensor([mels.shape[-1]]))
            logits = model.quantizer._codebook.project_down(hidden)
            case["hidden"] = export(name + "-hidden", hidden)
            case["logits"] = export(name + "-logits", logits)
        cases.append(case)
        print(json.dumps(dict(name=name, frames=mels.shape[-1], tokens=case["tokens"])), flush=True)
sources = [
    base / "chatterbox/models/s3tokenizer/s3tokenizer.py",
    base / "s3tokenizer/model_v2.py", base / "s3tokenizer/model.py",
    base / "s3tokenizer/utils.py",
]
oracle = dict(manifest_sha256=sha(output / "s3-tokenizer.json"), cases=cases,
              source_hashes={str(p.relative_to(base)): sha(p) for p in sources},
              cuda_initialized=torch.cuda.is_initialized(), retained_voice_preparations=0)
assert not oracle["cuda_initialized"]
(output / "oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")
print(json.dumps(dict(cases=len(cases), manifest_sha256=oracle["manifest_sha256"],
                      cuda_initialized=False, retained_voice_preparations=0)), flush=True)
