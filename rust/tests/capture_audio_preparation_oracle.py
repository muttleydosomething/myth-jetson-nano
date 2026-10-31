#!/usr/bin/env python3
"""Capture deployed audio preparation only: no model, service or retained voice."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

assert sys.flags.no_site
sys.path.append("/usr/local/lib/python3.10/dist-packages")
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as ln

source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox/tts_turbo.py")
tree = ast.parse(source.read_text())
owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChatterboxTurboTTS")
method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "norm_loudness")
scope = dict(ln=ln, math=math)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
normalise = scope["norm_loudness"]
output = Path(sys.argv[1])
output.mkdir(exist_ok=False)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def array(name, values):
    path = output / (name + ".npy")
    np.save(path, np.ascontiguousarray(values, dtype="<f4"), allow_pickle=False)
    return dict(file=path.name, sha256=sha(path))

cases = []
for name, rate, seconds, channels, kind, subtype, enabled in [
    ("mono24", 24000, 6.0, 1, "tone", "FLOAT", True),
    ("stereo44", 44100, 6.123, 2, "chirp", "PCM_16", True),
    ("mono48-flac", 48000, 6.01, 1, "chirp", "PCM_24", True),
    ("mono16", 16000, 6.001, 1, "tone", "PCM_16", True),
    ("silence", 24000, 6.0, 1, "silence", "FLOAT", True),
    ("quiet-gated", 22050, 6.051, 1, "quiet", "FLOAT", True),
    ("long-gated", 32000, 17.05, 2, "gated", "FLOAT", True),
    ("disabled", 48000, 6.013, 1, "chirp", "FLOAT", False),
]:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    y = (.13 * np.sin(2*np.pi*(173*t + (31*t*t if kind == "chirp" else 0)))
         + .04 * np.sin(2*np.pi*711*t)).astype(np.float32)
    if kind == "silence": y[:] = 0
    if kind == "quiet": y *= 1e-6
    if kind == "gated":
        y[:rate*3] *= .001
        y[rate*10:] *= .04
    if channels == 2: y = np.stack([y, y*.31 + .02*np.sin(2*np.pi*337*t)], axis=1).astype(np.float32)
    path = output / (name + (".flac" if name.endswith("flac") else ".wav"))
    sf.write(path, y, rate, subtype=subtype)
    decoded, actual_rate = librosa.load(path, sr=None)
    pcm24, sr = librosa.load(path, sr=24000)
    assert len(pcm24)/sr > 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loudness = float(ln.Meter(sr).integrated_loudness(pcm24))
        normalised = normalise(None, pcm24.copy(), sr) if enabled else pcm24.copy()
    pcm16 = librosa.resample(normalised, orig_sr=24000, target_sr=16000)
    cases.append(dict(name=name, audio=path.name, audio_sha256=sha(path),
        rate=actual_rate, channels=channels, normalise=enabled,
        loudness=loudness if math.isfinite(loudness) else None,
        decoded=array(name+"-decoded", decoded),
        pcm24=array(name+"-pcm24", normalised),
        pcm16=array(name+"-pcm16", pcm16)))
# Boundary/malformed admissions do not need models or retained recordings.
for name, size in [("exact-five", 120000), ("short", 10)]:
    sf.write(output/(name+".wav"), np.zeros(size,dtype=np.float32),24000,subtype="FLOAT")
sf.write(output/"nonfinite.wav", np.full(144000,np.nan,dtype=np.float32),24000,subtype="FLOAT")
(output/"invalid.wav").write_bytes(b"not audio")
receipt=dict(cases=cases, source_sha256=sha(source),
    library_versions=dict(librosa=librosa.__version__, soundfile=sf.__version__, sndfile=sf.__libsndfile_version__),
    retained_voice_preparations=0, cuda_initialized=False)
(output/"oracle.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)
