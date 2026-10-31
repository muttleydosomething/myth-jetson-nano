#!/usr/bin/env python3
"""Export retained S3Gen state and exercise its deployed ref_dict handoff.

CPU proof tooling only. Optional flow mode reads three pre-Conformer tables.
No Chatterbox import, Conformer/decoder construction, CUDA context, reference
audio, embed_ref, tokenizer or speaker encoder is permitted.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import sys

assert sys.flags.no_site
sys.path.append("/usr/local/lib/python3.10/dist-packages")
import numpy as np
import torch

torch.set_num_threads(2)
retained, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=False)
source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox/models/s3gen/s3gen.py")
flow_source = source.with_name("flow.py")
source_sha = hashlib.sha256(retained.read_bytes()).hexdigest()
reference = torch.load(retained, map_location="cpu", weights_only=True)["gen"]
assert set(reference) == {"prompt_token", "prompt_token_len", "prompt_feat", "prompt_feat_len", "embedding"}
assert reference["prompt_token"].dtype == torch.int64
assert reference["prompt_token_len"].dtype == torch.int64
assert reference["prompt_feat"].dtype == reference["embedding"].dtype == torch.float32

tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "S3Token2Mel")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
namespace = {"torch": torch, "np": np, "Optional": Optional}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_sha(path):
    # A model checkpoint can exceed the live container's remaining RAM. Hash
    # bounded blocks even though only three small tables are materialised.
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def tensor_receipt(value):
    if value is None:
        return None
    dtype = "int64" if value.dtype == torch.int64 else "float32"
    assert value.dtype in (torch.int64, torch.float32)
    values = value.detach().contiguous().numpy().astype("<i8" if dtype == "int64" else "<f4")
    return dict(shape=list(values.shape), dtype=dtype, sha256=sha(values.tobytes()))

def forbidden(*args, **kwargs):
    raise AssertionError("Reference preparation must never occur")

cases = []
# The authentic odd mel tail is retained; the second probe exercises optional
# explicit lengths and NumPy conversion in the existing prod API boundary.
even = {k: v.clone() if torch.is_tensor(v) else v for k, v in reference.items()}
even["prompt_feat"] = even["prompt_feat"][:, :2 * even["prompt_token"].shape[1], :].contiguous()
even["prompt_feat_len"] = torch.tensor([even["prompt_feat"].shape[1]], dtype=torch.int64)
for label, ref, numpy_input in [("retained", reference, False), ("explicit-length", even, True)]:
    handoff = {}
    def inference(**kwargs):
        handoff.update(kwargs)
        return torch.empty((1, 80, 0)), None

    owner = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32,
                            meanflow=True, embed_ref=forbidden,
                            flow=SimpleNamespace(inference=inference))
    incoming = {k: v.detach().numpy().copy() if numpy_input and torch.is_tensor(v)
                else v.clone() if torch.is_tensor(v) else v for k, v in ref.items()}
    namespace["forward"](owner, speech_tokens=torch.tensor([[1, 2, 3]]),
                         ref_wav=None, ref_sr=None, ref_dict=incoming, finalize=True)
    # The Python wrapper casts *all* reference tensors to the model float dtype,
    # including codec IDs/lengths. Native state keeps integers integral. Check
    # exact values through that cast explicitly rather than claiming dtype parity.
    for key, value in ref.items():
        if value is None:
            assert handoff[key] is None
        else:
            actual = handoff[key]
            if value.dtype == torch.int64:
                assert torch.equal(actual.long(), value)
            else:
                assert actual.dtype == value.dtype
                assert actual.contiguous().numpy().tobytes() == value.contiguous().numpy().tobytes()

    def array_asset(key):
        name = label + "-" + key + ".npy"
        path = output / name
        # Retained mel tensors are transposed; np.save otherwise preserves their
        # Fortran strides. Portable native assets require C order, same values.
        np.save(path, np.ascontiguousarray(ref[key].numpy(), dtype="<f4"), allow_pickle=False)
        return dict(file=name, sha256=sha(path.read_bytes()))

    token_path = output / (label + "-tokens.json")
    token_path.write_text(json.dumps(ref["prompt_token"][0].tolist()) + "\n")
    manifest = dict(schema_version=1, reference_kind="s3gen_prepared",
                    source_sha256=source_sha,
                    prompt_tokens=dict(file=token_path.name, sha256=sha(token_path.read_bytes())),
                    prompt_token_len=int(ref["prompt_token_len"][0]),
                    prompt_feat=array_asset("prompt_feat"),
                    prompt_feat_len=None if ref["prompt_feat_len"] is None else int(ref["prompt_feat_len"][0]),
                    embedding=array_asset("embedding"))
    manifest_path = output / (label + "-s3gen.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cases.append(dict(name=label, manifest=manifest_path.name,
                      manifest_sha256=sha(manifest_path.read_bytes()),
                      reference=dict(source_sha256=source_sha,
                                     **{k: tensor_receipt(v) for k, v in ref.items()}),
                      handoff_exact_values=True,
                      handoff_dtypes={k: None if handoff[k] is None else str(handoff[k].dtype)
                                      for k in ref}))

assert not torch.cuda.is_initialized()
assert not any(n == "chatterbox" or n.startswith("chatterbox.") for n in sys.modules)
receipt = dict(schema_version=1, source_sha256=sha(source.read_bytes()),
               flow_source_sha256=sha(flow_source.read_bytes()),
               retained_source_sha256=source_sha,
               reference_preparations=0, cuda_initialized=False, cases=cases)
(output / "oracle.json").write_text(json.dumps(receipt, indent=2) + "\n")

# Optional next boundary: materialise only the three pre-Conformer tables from
# the Turbo meanflow checkpoint and run the actual flow method to its encoder.
# The exception is intentional: no Conformer, decoder or waveform can run.
if len(sys.argv) >= 4:
    import logging
    from safetensors import safe_open
    checkpoint = Path(sys.argv[3])
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        table = weights.get_tensor("flow.input_embedding.weight").float()
        weight = weights.get_tensor("flow.spk_embed_affine_layer.weight").float()
        bias = weights.get_tensor("flow.spk_embed_affine_layer.bias").float()
    assert list(table.shape) == [6561, 512]
    assert list(weight.shape) == [80, 192] and list(bias.shape) == [80]

    def export_array(name, tensor):
        path = output / (name + ".npy")
        np.save(path, np.ascontiguousarray(tensor.numpy(), dtype="<f4"), allow_pickle=False)
        return dict(file=path.name, sha256=sha(path.read_bytes()))

    flow_manifest = dict(schema_version=1, flow_kind="s3gen_encoder_input",
                         source_sha256=file_sha(checkpoint),
                         token_embedding=export_array("token_embedding", table),
                         speaker_weight=export_array("speaker_weight", weight),
                         speaker_bias=export_array("speaker_bias", bias))
    flow_path = output / "flow.json"
    flow_path.write_text(json.dumps(flow_manifest, indent=2) + "\n")
    mask_source = source.parent / "utils/mask.py"
    mask_tree = ast.parse(mask_source.read_text())
    mask_method = next(n for n in mask_tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_pad_mask")
    flow_tree = ast.parse(flow_source.read_text())
    repeat_method = next(n for n in flow_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_repeat_batch_dim")
    flow_class = next(n for n in flow_tree.body if isinstance(n, ast.ClassDef) and n.name == "CausalMaskedDiffWithXvec")
    flow_method = next(n for n in flow_class.body if isinstance(n, ast.FunctionDef) and n.name == "inference")
    env = dict(torch=torch, F=torch.nn.functional, logger=logging.getLogger("oracle"))
    exec(compile(ast.Module(body=[mask_method, repeat_method, flow_method], type_ignores=[]), str(flow_source), "exec"), env)

    class EncoderBoundary(Exception):
        pass

    linear = torch.nn.Linear(192, 80)
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    linear.bias = torch.nn.Parameter(bias, requires_grad=False)
    embedding_table = torch.nn.Embedding.from_pretrained(table)
    flow_cases = []
    real_tokens = json.loads(Path(sys.argv[4]).read_text())["decoder_tokens"] if len(sys.argv) >= 5 else [1, 2, 3]
    probes = [("retained", reference, real_tokens),
              ("reversed", reference, list(reversed(real_tokens))),
              ("codec-boundary", even, [0, 6560]),
              ("single-token", reference, [429])]
    for label, value in [("zero-speaker", 0.0), ("tiny-speaker", 1e-15)]:
        ref = {k: v.clone() if torch.is_tensor(v) else v for k, v in reference.items()}
        ref["embedding"].fill_(value)
        probes.append((label, ref, [0, 1, 6560]))
        manifest = json.loads((output / "retained-s3gen.json").read_text())
        manifest["embedding"] = export_array(label, ref["embedding"])
        (output / (label + "-s3gen.json")).write_text(json.dumps(manifest, indent=2) + "\n")

    for index, (label, ref, ids) in enumerate(probes):
        captured = {}
        def affine(x):
            captured["normalized_speaker"] = x.clone()
            captured["speaker"] = linear(x)
            return captured["speaker"]
        def embed(x):
            captured["token"] = x.clone()
            return embedding_table(x)
        def mask(lengths):
            result = env["make_pad_mask_original"](lengths)
            captured["mask"] = (~result).unsqueeze(-1).float()
            return result
        def encoder(x, lengths):
            captured["encoder_input"] = x.clone()
            # The deployed wrapper casts reference lengths to FP32. The native
            # contract retains integral lengths; compare exact integer values.
            captured["token_len"] = lengths.long()
            raise EncoderBoundary()
        if "make_pad_mask_original" not in env:
            env["make_pad_mask_original"] = env["make_pad_mask"]
        env["make_pad_mask"] = mask
        owner = SimpleNamespace(spk_embed_affine_layer=affine, input_embedding=embed,
                                encoder=encoder, vocab_size=6561)
        incoming = {k: v.float() if torch.is_tensor(v) else v for k, v in ref.items()}
        try:
            env["inference"](owner, token=torch.tensor([ids]), token_len=torch.tensor([len(ids)]),
                             finalize=True, **incoming)
            raise AssertionError("Encoder boundary was not reached")
        except EncoderBoundary:
            pass
        prefix = "explicit-length" if label == "codec-boundary" else label if label in ("zero-speaker", "tiny-speaker") else "retained"
        manifest_path = output / (prefix + "-s3gen.json")
        tokens_path = output / (label + "-generated.json")
        tokens_path.write_text(json.dumps(ids) + "\n")
        flow_cases.append(dict(name=label, manifest=manifest_path.name,
                               manifest_sha256=sha(manifest_path.read_bytes()),
                               generated=tokens_path.name,
                               tensors={k: tensor_receipt(v) for k, v in captured.items()},
                               speaker_values={k: captured[k].tolist()[0] for k in ("normalized_speaker", "speaker")}))
    receipt["flow"] = dict(manifest=flow_path.name, manifest_sha256=sha(flow_path.read_bytes()),
                            mask_source_sha256=sha(mask_source.read_bytes()),
                            cases=flow_cases, conformer_executions=0,
                            tolerance=dict(atol=2e-6, rtol=1e-5))
    (output / "oracle.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    assert not any(n == "chatterbox" or n.startswith("chatterbox.") for n in sys.modules)
print(json.dumps(receipt))
