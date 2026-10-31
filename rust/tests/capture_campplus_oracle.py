#!/usr/bin/env python3
"""Execute deployed CAMPPlus alone; synthetic audio, CPU, no retained voices."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

assert sys.flags.no_site
sys.path.append("/usr/local/lib/python3.10/dist-packages")
import numpy as np
import torch
from safetensors import safe_open
import torchaudio.compliance.kaldi as kaldi

torch.set_num_threads(2)
base = Path("/usr/local/lib/python3.10/dist-packages")
source = base / "chatterbox/models/s3gen/xvector.py"
spec = importlib.util.spec_from_file_location("deployed_xvector", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
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

with torch.device("meta"):
    model = module.CAMPPlus(feat_dim=80, embedding_size=192)
state = {}
with safe_open(checkpoint, framework="pt", device="cpu") as weights:
    for key in weights.keys():
        if key.startswith("speaker_encoder."):
            value = weights.get_tensor(key)
            state[key.removeprefix("speaker_encoder.")] = value if not value.is_floating_point() else value.float()
model.load_state_dict(state, strict=True, assign=True)
model.eval()
tensors = {name: export(name, value) for name, value in model.state_dict().items()
           if not name.endswith(".num_batches_tracked")}
basis, _ = kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
tensors["mel_basis"] = export("mel_basis", torch.nn.functional.pad(basis, (0, 1)))
tensors["window"] = export("window", torch.hann_window(400, periodic=False).pow(0.85))
manifest = dict(schema_version=1, encoder_kind="campplus_192_fp32",
                source_sha256=sha(checkpoint), tensors=tensors)
(output / "campplus.json").write_text(json.dumps(manifest, indent=2)+"\n")
cases = []
with torch.inference_mode():
    for name, frames in [
        ("minimum", 3), ("silence", 18), ("chirp", 78),
        ("below-pool", 197), ("pool-boundary", 199),
        ("above-pool", 201), ("multiple-pools", 401),
    ]:
        length = 400 + (frames - 1) * 160
        t = np.arange(length, dtype=np.float64) / 16000
        wav = (0.17*np.sin(2*np.pi*(173*t+57*t*t)) + 0.05*np.sin(2*np.pi*517*t)).astype(np.float32)
        if name == "silence":
            wav[:] = 0
        if name == "multiple-pools":
            wav += np.random.default_rng(42).normal(0, 0.013, length).astype(np.float32)
        audio = torch.from_numpy(wav)
        features, _, _ = module.extract_feature([audio])
        embedding = model.inference([audio])
        assert torch.isfinite(embedding).all()
        head = model.head(features.permute(0, 2, 1))
        stages = {}
        current = head
        for stage_name, layer in model.xvector.named_children():
            current = layer(current)
            if stage_name in ("tdnn", "transit1", "transit2", "transit3", "stats"):
                stages[stage_name] = export(name+"-"+stage_name, current)
        assert torch.equal(current, embedding)
        cases.append(dict(name=name, audio=export(name+"-audio", wav),
                          features=export(name+"-features", features.squeeze(0)),
                          head=export(name+"-head", head), stages=stages,
                          embedding=export(name+"-embedding", embedding), frames=frames))
        print(json.dumps(dict(name=name, frames=frames)), flush=True)
oracle = dict(manifest_sha256=sha(output/"campplus.json"), cases=cases,
              source_hashes={str(source.relative_to(base)):sha(source),
                             "torchaudio/compliance/kaldi.py":sha(base/"torchaudio/compliance/kaldi.py")},
              cuda_initialized=torch.cuda.is_initialized(), retained_voice_preparations=0)
assert not oracle["cuda_initialized"]
(output/"oracle.json").write_text(json.dumps(oracle, indent=2)+"\n")
print(json.dumps(dict(passed=True, cases=len(cases), cuda_initialized=False)), flush=True)
