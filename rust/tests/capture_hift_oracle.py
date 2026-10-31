#!/usr/bin/env python3
"""Independent deployed HiFT inference, isolated CPU-only, no reference preparation.
Exports Torch's effective parametrized weights once, never speech service state.
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
torch.manual_seed(124)
source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox")
for name, path in [("chatterbox",source),("chatterbox.models",source/"models"),
                   ("chatterbox.models.s3gen",source/"models/s3gen")]:
    package=types.ModuleType(name); package.__path__=[str(path)]; sys.modules[name]=package
hift_module=importlib.import_module("chatterbox.models.s3gen.hifigan")
f0_module=importlib.import_module("chatterbox.models.s3gen.f0_predictor")
checkpoint,inputs,output=map(Path,sys.argv[1:4])
cache_only=sys.argv[4:]==["--cache-only"]
assert len(sys.argv)==4 or cache_only
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
with torch.device("meta"):
    model=hift_module.HiFTGenerator(sampling_rate=24000,upsample_rates=[8,5,3],
        upsample_kernel_sizes=[16,11,7],source_resblock_kernel_sizes=[7,7,11],
        source_resblock_dilation_sizes=[[1,3,5]]*3,
        f0_predictor=f0_module.ConvRNNF0Predictor())
state={}
with safe_open(checkpoint,framework="pt",device="cpu") as weights:
    for key in weights.keys():
        if key.startswith("mel2wav."): state[key.removeprefix("mel2wav.")]=weights.get_tensor(key).float()
model.load_state_dict(state,strict=True,assign=True); model.eval(); del state
# Access actual Torch parametrizations, including ConvTranspose1d's dim0 norm.
tensors={}
for name,module in model.named_modules():
    if isinstance(module,(torch.nn.Conv1d,torch.nn.ConvTranspose1d,torch.nn.Linear)):
        tensors[name+".weight"]=export(name+".weight",module.weight)
        if module.bias is not None: tensors[name+".bias"]=export(name+".bias",module.bias)
    if isinstance(module,hift_module.Snake): tensors[name+".alpha"]=export(name+".alpha",module.alpha)
tensors["stft_window"]=export("stft_window",model.stft_window)
manifest=dict(schema_version=1,decoder_kind="s3gen_hift_fp32",
              source_sha256=sha(checkpoint),tensors=tensors)
(output/"hift.json").write_text(json.dumps(manifest,indent=2)+"\n")
cases=[]
probes=[] if cache_only else [("retained-final",None),("reversed-final",None),("single-frame",1),
        ("two-frame",2),("odd-frames",5),("zero-mel",3)]
with torch.inference_mode():
    for name,length in probes:
        base=name if length is None else "retained-final"
        mel=torch.from_numpy(np.load(inputs/(base+"-generated.npy"),allow_pickle=False).copy())
        if length is not None: mel=mel[...,:length].contiguous()
        if name=="zero-mel": mel.zero_()
        frames=mel.shape[-1]
        normal=torch.randn(1,9,frames*480)
        phases=torch.distributions.Uniform(-np.pi,np.pi).sample((1,9,1))
        # These are the exact random values supplied to deployed SineGen.
        original_normal=torch.randn_like
        original_phase=torch.distributions.Uniform.sample
        calls=[]
        def randn_like(x):
            calls.append(list(x.shape))
            return normal.clone() if x.shape==normal.shape else torch.zeros_like(x)
        torch.randn_like=randn_like
        torch.distributions.Uniform.sample=lambda self,sample_shape=torch.Size():phases.clone()
        try:
            waveform,excitation=model.inference(mel)
        finally:
            torch.randn_like=original_normal
            torch.distributions.Uniform.sample=original_phase
        f0=model.f0_predictor(mel)
        real,imag=model._stft(excitation.squeeze(1))
        fixed=model.decode(mel,excitation)
        assert torch.equal(fixed,waveform)
        # Capture actual decoder stages without implementing a second oracle.
        trace={}
        hooks=[]
        for key,mod in [("pre",model.conv_pre),("post",model.conv_post)]:
            hooks.append(mod.register_forward_hook(lambda m,a,y,key=key:trace.update({key:y.clone()})))
        try: model.decode(mel,excitation)
        finally:
            for hook in hooks: hook.remove()
        arrays={key:export(name+"-"+key,value) for key,value in dict(
            mel=mel,normal=normal,phases=phases,f0=f0,source=excitation,
            source_stft=torch.cat([real,imag],dim=1),waveform=waveform,**trace).items()}
        cases.append(dict(name=name,frames=frames,arrays=arrays,random_calls=calls))
        print(json.dumps(dict(case=name,frames=frames,samples=waveform.numel())),flush=True)
# Explicit source probes exercise threshold and fractional phase accumulation.
source_cases=[]
for name,f0 in ([] if cache_only else [("thresholds",torch.tensor([[0.,10.,10.01,100.,240.,0.]])),
                ("constant",torch.full((1,3),173.25))]):
    normal=torch.randn(1,9,f0.shape[1]*480)
    phases=torch.distributions.Uniform(-np.pi,np.pi).sample((1,9,1))
    original_normal=torch.randn_like; original_phase=torch.distributions.Uniform.sample
    torch.randn_like=lambda x:normal.clone() if x.shape==normal.shape else torch.zeros_like(x)
    torch.distributions.Uniform.sample=lambda self,sample_shape=torch.Size():phases.clone()
    try: excitation=model.m_source(model.f0_upsamp(f0[:,None]).transpose(1,2))[0].transpose(1,2)
    finally:
        torch.randn_like=original_normal; torch.distributions.Uniform.sample=original_phase
    source_cases.append(dict(name=name,arrays={k:export(name+"-"+k,v) for k,v in
        dict(f0=f0,normal=normal,phases=phases,source=excitation).items()}))
# Cache parity calls the actual deployed inference method with controlled random
# values. Include a non-frame-aligned prefix and the full-source boundary; the
# algorithm replaces exactly the requested prefix and returns that same source.
cache_cases=[]
with torch.inference_mode():
    mel=torch.from_numpy(np.load(inputs/"retained-final-generated.npy",allow_pickle=False).copy())[...,:5].contiguous()
    frames=mel.shape[-1]
    normal=torch.randn(1,9,frames*480)
    phases=torch.distributions.Uniform(-np.pi,np.pi).sample((1,9,1))
    for count in [0,1,481,frames*480]:
        cache=torch.linspace(-0.1,0.1,count).reshape(1,1,count)
        original_normal=torch.randn_like; original_phase=torch.distributions.Uniform.sample
        torch.randn_like=lambda x:normal.clone() if x.shape==normal.shape else torch.zeros_like(x)
        torch.distributions.Uniform.sample=lambda self,sample_shape=torch.Size():phases.clone()
        try: waveform,excitation=model.inference(mel,cache_source=cache)
        finally:
            torch.randn_like=original_normal; torch.distributions.Uniform.sample=original_phase
        assert torch.equal(excitation[...,:count],cache)
        name="cached-"+str(count)
        arrays={key:export(name+"-"+key,value) for key,value in dict(
            mel=mel,normal=normal,phases=phases,cache=cache,
            source=excitation,waveform=waveform).items()}
        cache_cases.append(dict(name=name,frames=frames,cache_samples=count,arrays=arrays))
assert not torch.cuda.is_initialized()
receipt=dict(schema_version=1,manifest_sha256=sha(output/"hift.json"),
    source_sha256=manifest["source_sha256"],
    sources={str(p.relative_to(source)):sha(p) for p in [
        source/"models/s3gen/hifigan.py",source/"models/s3gen/f0_predictor.py"]},
    cases=cases,source_cases=source_cases,cache_cases=cache_cases,reference_preparations=0,cuda_initialized=False,
    sample_rate=24000,samples_per_frame=480,trim_fade_applied=False,
    tolerance=dict(atol=3e-4,rtol=3e-4))
(output/"oracle.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(passed=True,cases=len(cases),tensor_count=len(tensors),reference_preparations=0)))
