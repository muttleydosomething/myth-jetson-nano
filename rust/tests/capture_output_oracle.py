#!/usr/bin/env python3
"""Execute deployed S3Token2Wav constructor/inference with inert model boundaries.
Only the real trim_fade buffer and inference output operation run in Torch.
Existing HiFT waveforms supply authentic inputs; no weights or references load.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType
from typing import Optional

assert sys.flags.no_site
sys.path.append("/usr/local/lib/python3.10/dist-packages")
import numpy as np
import torch
torch.set_num_threads(2)
inputs, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=False)
source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox/models/s3gen/s3gen.py")
source_bytes = source.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == "b3ea20b6f3f98dacfa3c2c99502bbf3dfbe7a77362f2ce0d16783d4324e6fc96"
tree = ast.parse(source_bytes)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "S3Token2Wav")
cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "inference")]
class InertMel(torch.nn.Module):
    def __init__(self, meanflow=False):
        super().__init__()
        self.meanflow = meanflow
    @property
    def dtype(self):
        return torch.float32
def inert_hift(**kwargs):
    return torch.nn.Identity()
namespace = dict(torch=torch, Optional=Optional, S3Token2Mel=InertMel, S3GEN_SR=24000,
                 HiFTGenerator=inert_hift, ConvRNNF0Predictor=torch.nn.Identity)
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
model = namespace["S3Token2Wav"](meanflow=True).eval()
assert list(model.state_dict()) == []  # No checkpoint or prepared voice.
assert model.trim_fade.shape == (960,)
calls = []
current = None
marker = torch.zeros(1, 80, 2)
def flow(self, *args, **kwargs):
    assert kwargs["finalize"] is True
    assert kwargs["ref_wav"] is None and kwargs["ref_dict"] is None
    calls.append("flow")
    return marker
def hift(self, mels, cache):
    assert mels is marker and cache is None
    calls.append("hift")
    return current.clone(), torch.zeros(1, 1, 0)
model.flow_inference = MethodType(flow, model)
model.hift_inference = MethodType(hift, model)
def export(name, value):
    np.save(output / name, np.ascontiguousarray(value.numpy(), dtype="<f4"), allow_pickle=False)
    return name
cases = []
# Authentic raw waveforms were already captured from deployed HiFT with fixed excitation.
for name in ["retained-final", "reversed-final", "two-frame", "odd-frames", "zero-mel"]:
    current = torch.from_numpy(np.load(inputs / (name + "-waveform.npy"), allow_pickle=False).copy())
    original = current.clone()
    finished, _ = model.inference(torch.tensor([[1, 2]]))
    assert torch.equal(current, original) and torch.equal(finished[:, 960:], current[:, 960:])
    cases.append(dict(name=name, input=export(name + "-raw.npy", current),
                      output=export(name + "-finished.npy", finished)))
# Unit amplitude exposes the exact window; alternating polarity catches sign mistakes.
for name, data in [("unit-window", torch.ones(1, 960)),
                   ("alternating", torch.tensor([[(-1.0 if i % 2 else 1.0) for i in range(1441)]]))]:
    current = data
    finished, _ = model.inference(torch.tensor([[1, 2]]))
    cases.append(dict(name=name, input=export(name + "-raw.npy", current),
                      output=export(name + "-finished.npy", finished)))
rejections = []
for name, count in [("single-frame", 480), ("short-window", 959)]:
    current = torch.ones(1, count)
    export(name + ".npy", current)
    try:
        model.inference(torch.tensor([[1]]))
    except RuntimeError as error:
        rejections.append(dict(name=name, error=str(error)))
    else:
        raise AssertionError("Deployed wrapper unexpectedly accepted a short fade window")
assert not torch.cuda.is_initialized()
assert not any(n == "chatterbox" or n.startswith("chatterbox.") for n in sys.modules)
receipt = dict(schema_version=1, source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    cases=cases, deployed_short_input_rejections=rejections, calls=calls,
    sample_rate=24000, unchanged_length=True, reference_preparations=0,
    watermark_applied=False, cuda_initialized=False, tolerance_atol=3e-7,
    scope="Actual deployed constructor fade buffer and inference output operation; flow/HiFT are inert stubs fed captured raw waveforms.")
(output / "oracle.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(dict(passed=True, cases=len(cases), short_rejections=len(rejections))))
