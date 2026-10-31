#!/usr/bin/env python3
"""CPU-only conversion proof. Import encoder modules through inert package shells;
never import Chatterbox's initializers, construct a speech service or encode audio.
Run in an isolated no-network/no-GPU memory-bounded container, never live service.
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
torch.manual_seed(0)
source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox")
# Avoid tts/__init__ side effects and tokenizer/speaker/decoder construction.
for name, path in [("chatterbox",source),("chatterbox.models",source/"models"),
                   ("chatterbox.models.s3gen",source/"models/s3gen")]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
module = importlib.import_module("chatterbox.models.s3gen.transformer.upsample_encoder")
checkpoint, inputs, output = map(Path,sys.argv[1:4])
output.mkdir(parents=True,exist_ok=False)
def file_sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()
def export(name,tensor):
    path = output/(name+".npy")
    np.save(path,np.ascontiguousarray(tensor.detach().cpu().numpy(),dtype="<f4"),allow_pickle=False)
    return dict(file=path.name,sha256=file_sha(path))
# This is the deployed S3Token2Mel constructor's exact encoder configuration.
encoder = module.UpsampleConformerEncoder(
    output_size=512,attention_heads=8,linear_units=2048,num_blocks=6,
    dropout_rate=0.1,positional_dropout_rate=0.1,attention_dropout_rate=0.1,
    normalize_before=True,input_layer="linear",pos_enc_layer_type="rel_pos_espnet",
    selfattention_layer_type="rel_selfattn",input_size=512,
    use_cnn_module=False,macaron_style=False).eval()
state = {}
with safe_open(checkpoint,framework="pt",device="cpu") as weights:
    for key in weights.keys():
        if key.startswith("flow.encoder."):
            state[key.removeprefix("flow.encoder.")] = weights.get_tensor(key).float()
encoder.load_state_dict(state,strict=True)
tensors = {name:export(name,tensor) for name,tensor in state.items()}
del state
frequencies = torch.exp(torch.arange(0,512,2,dtype=torch.float32)*-(np.log(10000.0)/512))
tensors["position_frequencies"] = export("position_frequencies",frequencies)
manifest = dict(schema_version=1,encoder_kind="s3gen_upsample_conformer_fp32",
                source_sha256=file_sha(checkpoint),tensors=tensors)
(output/"conformer.json").write_text(json.dumps(manifest,indent=2)+"\n")
# Use the previous pre-Conformer proof's retained voice plus native decoder IDs.
oracle = json.loads((inputs/"oracle.json").read_text())
cases=[]
with safe_open(checkpoint,framework="pt",device="cpu") as weights:
    table = weights.get_tensor("flow.input_embedding.weight").float()
retained = oracle["flow"]["cases"][0]
ref_manifest = json.loads((inputs/retained["manifest"]).read_text())
prefix = json.loads((inputs/ref_manifest["prompt_tokens"]["file"]).read_text())
generated = json.loads((inputs/retained["generated"]).read_text())
probes = [("retained",prefix+generated),("reversed",prefix+list(reversed(generated))),
          ("single-frame",[6560]),("two-frame",[0,6560])]
with torch.inference_mode():
    for label,ids in probes:
        x = torch.nn.functional.embedding(torch.tensor([ids]),table)
        y,mask = encoder(x,torch.tensor([len(ids)]))
        assert y.shape == (1,2*len(ids),512) and bool(mask.all())
        cases.append(dict(name=label,input=export(label+"-input",x)["file"],
                          output=export(label+"-output",y)["file"],
                          mask_shape=list(mask.shape)))
assert not torch.cuda.is_initialized()
assert "chatterbox.tts" not in sys.modules
sources={str(path.relative_to(source)):file_sha(path) for path in [
    source/"models/s3gen/transformer"/name for name in [
        "upsample_encoder.py","encoder_layer.py","attention.py","embedding.py",
        "subsampling.py","positionwise_feed_forward.py"]]}
receipt = dict(schema_version=1,manifest_sha256=file_sha(output/"conformer.json"),
               source_sha256=manifest["source_sha256"],sources=sources,cases=cases,
               reference_preparations=0,cuda_initialized=False,waveform_synthesis=False,
               tolerance=dict(atol=2e-4,rtol=2e-4))
(output/"oracle.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
