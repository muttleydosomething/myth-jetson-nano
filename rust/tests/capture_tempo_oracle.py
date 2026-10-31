#!/usr/bin/env python3
"""Capture the deployed librosa tempo boundary without loading any speech model.
Run in the existing isolated CPU-only, network-disabled, memory-bounded oracle
container. The real retained waveform is input only; no voice is prepared.
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import librosa
from scipy.io import wavfile
source, output = map(Path,sys.argv[1:])
output.mkdir(exist_ok=False)
started=time.time_ns()//1000000
assert librosa.__version__=="0.11.0"
rate,speech=wavfile.read(source)
assert rate==24000 and speech.dtype==np.float32 and speech.ndim==1
t=np.arange(24000,dtype=np.float64)/24000
fixtures=[
    ("tone",(.2*np.sin(2*np.pi*440*t)).astype(np.float32),[.5,.75,1.5,2.]),
    ("chirp",(.2*np.sin(2*np.pi*(100*t+900*t*t))).astype(np.float32),[.75,1.3]),
    ("silence",np.zeros(513,dtype=np.float32),[.5,2.]),
    ("short",np.array([.2,-.1,.3],dtype=np.float32),[.5,2.]),
    ("speech",speech,[.75,1.5]),
]
cases=[]
for name,pcm,speeds in fixtures:
    np.save(output/(name+"-input.npy"),pcm[None,:],allow_pickle=False)
    for speed in speeds:
        result=librosa.effects.time_stretch(pcm,rate=speed)
        filename=f"{name}-{speed}.npy"
        np.save(output/filename,result[None,:],allow_pickle=False)
        cases.append(dict(name=f"{name}-{speed}",speed=speed,input=name+"-input.npy",output=filename,samples=len(result)))
finished=time.time_ns()//1000000
receipt=dict(librosa_version=librosa.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    cases=cases,reference_preparations=0,inference=False,cuda=False,
    waveform_gate="SNR > 40 dB against deployed librosa; exact lengths; silence exact",
    started_unix_ms=started,finished_unix_ms=finished,duration_ms=finished-started)
(output/"oracle.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
