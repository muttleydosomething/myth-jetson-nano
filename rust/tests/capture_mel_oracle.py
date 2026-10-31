#!/usr/bin/env python3
"""Independent post-encoder oracle: execute the deployed flow to a recording
decoder, using retained reference tensors and previously proven encoder outputs.
CPU-only bounded container. No reference preparation, Conformer or synthesis.
The optional --align-nonfinal probe explicitly repairs only upstream h_masks
trimming, so corrected results are never attributed to the unmodified package.
"""
import ast
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
assert sys.flags.no_site
sys.path.append("/usr/local/lib/python3.10/dist-packages")
import numpy as np
import torch
from safetensors import safe_open
torch.set_num_threads(2)
checkpoint, flow_root, encoder_root, output = map(Path, sys.argv[1:5])
align_nonfinal = sys.argv[5:] == ["--align-nonfinal"]
assert len(sys.argv) == 5 or align_nonfinal
output.mkdir(parents=True, exist_ok=False)
source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox/models/s3gen/flow.py")
mask_source = source.parent / "utils/mask.py"
def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()
def export(name, tensor):
    path = output / (name + ".npy")
    np.save(path, np.ascontiguousarray(tensor.detach().numpy(), dtype="<f4"), allow_pickle=False)
    return dict(file=path.name, sha256=sha(path))
# Copy only admitted portable arrays, never the retained Torch archive.
for manifest_name in ["flow.json", "retained-s3gen.json", "explicit-length-s3gen.json"]:
    manifest = json.loads((flow_root / manifest_name).read_text())
    shutil.copyfile(flow_root / manifest_name, output / manifest_name)
    for asset in manifest.values():
        if isinstance(asset, dict) and "file" in asset:
            shutil.copyfile(flow_root / asset["file"], output / asset["file"])
flow_manifest = json.loads((output/"flow.json").read_text())
with safe_open(checkpoint, framework="pt", device="cpu") as weights:
    weight = weights.get_tensor("flow.encoder_proj.weight").float()
    bias = weights.get_tensor("flow.encoder_proj.bias").float()
manifest = dict(schema_version=1, mel_kind="s3gen_post_encoder_fp32",
    source_sha256=sha(checkpoint), weight=export("mel-weight", weight), bias=export("mel-bias", bias))
assert manifest["source_sha256"] == flow_manifest["source_sha256"]
(output/"mel.json").write_text(json.dumps(manifest, indent=2)+"\n")
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=="CausalMaskedDiffWithXvec")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=="inference")
if align_nonfinal:
    # Modify the actual method at its existing length authority. All projection,
    # reference placement and decoder-call code remains deployed source.
    lengths = [n for n in ast.walk(method) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "h_lengths" for t in n.targets)]
    assert len(lengths) == 1
    lengths[0].value = ast.parse("h_masks[..., :h.size(1)].sum(dim=-1).squeeze(dim=-1)", mode="eval").body
    ast.fix_missing_locations(method)
repeat = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=="_repeat_batch_dim")
mask = next(n for n in ast.parse(mask_source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name=="make_pad_mask")
env = dict(torch=torch, F=torch.nn.functional, logger=logging.getLogger("mel-oracle"))
exec(compile(ast.Module(body=[mask, repeat, method], type_ignores=[]), str(source), "exec"), env)
projection = torch.nn.Linear(512,80)
projection.weight = torch.nn.Parameter(weight, requires_grad=False)
projection.bias = torch.nn.Parameter(bias, requires_grad=False)
def array(name):
    return torch.from_numpy(np.load(flow_root/name, allow_pickle=False).copy())
table = array(flow_manifest["token_embedding"]["file"])
speaker = torch.nn.Linear(192,80)
speaker.weight = torch.nn.Parameter(array(flow_manifest["speaker_weight"]["file"]), requires_grad=False)
speaker.bias = torch.nn.Parameter(array(flow_manifest["speaker_bias"]["file"]), requires_grad=False)
flow_oracle = json.loads((flow_root/"oracle.json").read_text())
generated = json.loads((flow_root/flow_oracle["flow"]["cases"][0]["generated"]).read_text())
retained = torch.from_numpy(np.load(encoder_root/"retained-output.npy",allow_pickle=False).copy())
reversed_output = torch.from_numpy(np.load(encoder_root/"reversed-output.npy",allow_pickle=False).copy())
class DecoderBoundary(Exception):
    pass
cases=[]
probes = [
    ("retained-final", "retained", generated, retained, True),
    ("retained-nonfinal", "retained", generated, retained, False),
    ("reversed-final", "retained", list(reversed(generated)), reversed_output, True),
    ("reversed-nonfinal", "retained", list(reversed(generated)), reversed_output, False),
    ("odd-single-final", "retained", generated[:1], retained[:,:404].contiguous(), True),
    ("odd-short-nonfinal", "retained", generated[:4], retained[:,:410].contiguous(), False),
    ("even-final", "explicit-length", generated[:2], retained[:,:406].contiguous(), True),
]
if align_nonfinal:
    probes.append(("even-short-nonfinal", "explicit-length", generated[:4],
                   retained[:, :410].contiguous(), False))
for name, ref_name, ids, encoded, finalize in probes:
    ref_manifest = json.loads((output/(ref_name+"-s3gen.json")).read_text())
    ref_tokens = json.loads((output/ref_manifest["prompt_tokens"]["file"]).read_text())
    def ref_array(key):
        return torch.from_numpy(np.load(output/ref_manifest[key]["file"],allow_pickle=False).copy())
    captured={}
    def encoder(x, lengths):
        assert x.shape[1]*2 == encoded.shape[1]
        return encoded.clone(), torch.ones((1,1,encoded.shape[1]),dtype=torch.bool)
    def decoder(**kwargs):
        captured.update(kwargs)
        raise DecoderBoundary()
    owner = SimpleNamespace(vocab_size=6561, spk_embed_affine_layer=speaker,
        input_embedding=torch.nn.Embedding.from_pretrained(table),
        encoder=encoder, encoder_proj=projection, decoder=decoder,
        pre_lookahead_len=3,token_mel_ratio=2,output_size=80)
    try:
        env["inference"](owner,token=torch.tensor([ids]),token_len=torch.tensor([len(ids)]),
            prompt_token=torch.tensor([ref_tokens]),prompt_token_len=torch.tensor([len(ref_tokens)]),
            prompt_feat=ref_array("prompt_feat"),prompt_feat_len=None if ref_manifest["prompt_feat_len"] is None
                else torch.tensor([ref_manifest["prompt_feat_len"]]),
            embedding=ref_array("embedding"),finalize=finalize,meanflow=True)
        raise AssertionError("Decoder boundary not reached")
    except DecoderBoundary:
        pass
    token_path=output/(name+"-tokens.json")
    token_path.write_text(json.dumps(ids)+"\n")
    tensors={key:export(name+"-"+key,captured[key]) for key in ["mu","mask","spks","cond"]}
    cases.append(dict(name=name,reference=ref_name+"-s3gen.json",
        reference_sha256=sha(output/(ref_name+"-s3gen.json")),generated=token_path.name,
        input=export(name+"-encoded",encoded)["file"],finalize=finalize,tensors=tensors,
        reference_frames=ref_manifest["prompt_feat_len"] or ref_array("prompt_feat").shape[1],
        generated_frames=captured["mu"].shape[2]-ref_array("prompt_feat").shape[1]))
assert not torch.cuda.is_initialized()
assert not any(n=="chatterbox" or n.startswith("chatterbox.") for n in sys.modules)
receipt=dict(schema_version=1,manifest_sha256=sha(output/"mel.json"),
    flow_sha256=sha(output/"flow.json"),source_sha256=sha(source),mask_source_sha256=sha(mask_source),
    cases=cases,reference_preparations=0,conformer_executions=0,waveform_synthesis=False,
    tolerance=dict(atol=2e-6,rtol=1e-5),
    nonfinal_mask_contract=("Corrected deployed method: trim h_masks to h.size(1) before computing h_lengths"
        if align_nonfinal else "Original encoder length retained; six-frame feature trim is not applied to mask"),
    upstream_mask_correction=align_nonfinal)
(output/"oracle.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
