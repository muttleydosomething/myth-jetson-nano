#!/usr/bin/env python3
"""Independent deployed meanflow proof, CPU-only in a bounded isolated container.
Load only the estimator, use retained conditioning and fixed captured noise.
Never import speech service initializers or prepare reference audio.
"""
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types
assert sys.flags.no_site
sys.path.append("/usr/local/lib/python3.10/dist-packages")
import numpy as np
import torch
from safetensors import safe_open
torch.set_num_threads(2)
torch.manual_seed(123)
source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox")
for name, path in [("chatterbox",source),("chatterbox.models",source/"models"),
                   ("chatterbox.models.s3gen",source/"models/s3gen")]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
decoder_module = importlib.import_module("chatterbox.models.s3gen.decoder")
cfm_module = importlib.import_module("chatterbox.models.s3gen.flow_matching")
checkpoint, inputs, output = map(Path,sys.argv[1:4])
nonfinal_only = sys.argv[4:] == ["--nonfinal-only"]
assert len(sys.argv) == 4 or nonfinal_only
if nonfinal_only:
    assert json.loads((inputs/"oracle.json").read_text())["upstream_mask_correction"] is True
output.mkdir(parents=True,exist_ok=False)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def export(name,tensor):
    path=output/(name+".npy")
    np.save(path,np.ascontiguousarray(tensor.detach().numpy(),dtype="<f4"),allow_pickle=False)
    return dict(file=path.name,sha256=sha(path))
# Meta construction avoids an extra randomly initialised 280 MiB model.
with torch.device("meta"):
    estimator = decoder_module.ConditionalDecoder(in_channels=320,out_channels=80,
        causal=True,channels=[256],dropout=0.,attention_head_dim=64,n_blocks=4,
        num_mid_blocks=12,num_heads=8,act_fn="gelu",meanflow=True)
state={}
with safe_open(checkpoint,framework="pt",device="cpu") as weights:
    for key in weights.keys():
        if key.startswith("flow.decoder.estimator."):
            state[key.removeprefix("flow.decoder.estimator.")]=weights.get_tensor(key).float()
estimator.load_state_dict(state,strict=True,assign=True)
estimator.eval()
tensors={name:export(name,tensor) for name,tensor in state.items()}
del state
freq=torch.exp(torch.arange(160,dtype=torch.float32)*-(math.log(10000)/159))
tensors["time_frequencies"]=export("time_frequencies",freq)
manifest=dict(schema_version=1,decoder_kind="s3gen_meanflow_fp32",
              source_sha256=sha(checkpoint),tensors=tensors)
(output/"meanflow.json").write_text(json.dumps(manifest,indent=2)+"\n")
cfm=cfm_module.CausalConditionalCFM(estimator=estimator).eval()
cases=[]
# Retained and reversed full conditioning; short probes isolate decoder behaviour.
probes=[("retained-final",None,2),("reversed-final",None,2),
        ("single-frame",1,1),("two-frame",2,2),("odd-frames",5,3)]
if nonfinal_only:
    probes=[("retained-nonfinal",None,2),("odd-short-nonfinal",None,2),
            ("even-short-nonfinal",None,2)]
with torch.inference_mode():
    for name,length,steps in probes:
        base=name if length is None else "retained-final"
        def read(key):
            x=torch.from_numpy(np.load(inputs/(base+"-"+key+".npy"),allow_pickle=False).copy())
            return x if length is None or key=="spks" else x[...,:length].contiguous()
        mu,mask,spks,cond=[read(k) for k in ["mu","mask","spks","cond"]]
        noise=torch.randn_like(mu)
        span=torch.linspace(0,1,steps+1,dtype=torch.float32)
        derivatives=[]
        # Hooks record real calls in actual deployed forward/basic_euler.
        hook=estimator.register_forward_hook(lambda m,a,y:derivatives.append(y.clone()))
        original=torch.randn_like
        torch.randn_like=lambda x:noise.clone()
        try:
            full,_=cfm(mu,mask,steps,spks=spks,cond=cond,meanflow=True)
        finally:
            torch.randn_like=original
            hook.remove()
        arrays={k:export(name+"-"+k,v) for k,v in
            dict(mu=mu,mask=mask,spks=spks,cond=cond,noise=noise,output=full).items()}
        deriv=[export(name+"-derivative-"+str(i),v) for i,v in enumerate(derivatives)]
        if nonfinal_only:
            conditioning=json.loads((inputs/"oracle.json").read_text())
            reference_frames=next(c["reference_frames"] for c in conditioning["cases"] if c["name"]==name)
        else:
            reference_frames=403 if length is None else 0
        arrays["generated"]=export(name+"-generated",full[...,reference_frames:])
        cases.append(dict(name=name,steps=steps,reference_frames=reference_frames,
                          arrays=arrays,derivatives=deriv))
# Nonfinal mismatch from the previous boundary must fail, never silently crop mask.
try:
    if nonfinal_only:
        # Retain the historical failure as an explicit negative gate using the
        # exact old six-frame excess. Never relax native mask validation.
        def read_nonfinal(key):
            x=torch.from_numpy(np.load(inputs/("retained-nonfinal-"+key+".npy"),allow_pickle=False).copy())
            return torch.nn.functional.pad(x,(0,6),value=1) if key=="mask" else x
    else:
        def read_nonfinal(key):
            return torch.from_numpy(np.load(inputs/("retained-nonfinal-"+key+".npy"),allow_pickle=False).copy())
    with torch.inference_mode():
        estimator(read_nonfinal("mu"),mask=read_nonfinal("mask"),mu=read_nonfinal("mu"),
                  spks=read_nonfinal("spks"),cond=read_nonfinal("cond"),
                  t=torch.tensor([0.]),r=torch.tensor([.5]))
    raise AssertionError("Expected upstream nonfinal mask mismatch")
except RuntimeError as e:
    nonfinal_error=str(e)
assert not torch.cuda.is_initialized()
sources={str(p.relative_to(source)):sha(p) for p in [
    source/"models/s3gen"/n for n in ["decoder.py","flow_matching.py","matcha/decoder.py",
                                    "matcha/transformer.py","utils/intmeanflow.py"]]}
receipt=dict(schema_version=1,manifest_sha256=sha(output/"meanflow.json"),
             source_sha256=manifest["source_sha256"],sources=sources,cases=cases,
             diffusers_sources={name:sha(source.parent/"diffusers/models"/name)
                                for name in ["attention_processor.py","activations.py"]},
             nonfinal_error=nonfinal_error,reference_preparations=0,
             corrected_nonfinal_conditions=nonfinal_only,
             cuda_initialized=False,waveform_synthesis=False,tolerance=dict(atol=3e-4,rtol=3e-4))
(output/"oracle.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(cases=[c["name"] for c in cases],sources=sources,
                     nonfinal_error=nonfinal_error,reference_preparations=0)))
