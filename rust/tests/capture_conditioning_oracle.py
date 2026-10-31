#!/usr/bin/env python3
"""Export only embedding tables and an already-prepared Turbo voice prefix.

Proof tooling: run python3 -S with CUDA_VISIBLE_DEVICES='' in the existing
container. No Chatterbox import, model constructor, CUDA context, reference
encoding, transformer or S3Gen weights are needed.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

assert sys.flags.no_site
sys.path.append("/usr/local/lib/python3.10/dist-packages")
import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

torch.set_num_threads(2)
checkpoint, captures, tokenizer_path, output = map(Path, sys.argv[1:5])
retained_reference = Path(sys.argv[5]) if len(sys.argv) == 6 else None
output.mkdir(parents=True, exist_ok=False)
source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox/models/t3/t3.py")
turbo_source = source.parents[2] / "tts_turbo.py"

def sha(data):
    return hashlib.sha256(data).hexdigest()

# Execute the deployed method without constructing a transformer or importing
# the package (its module imports would pull in unrelated speech machinery).
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "T3")
method = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "prepare_input_embeds")
namespace = {"torch": torch, "T3Cond": object}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
tree = ast.parse(turbo_source.read_text())
normalize_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "punc_norm")
exec(compile(ast.Module(body=[normalize_node], type_ignores=[]), str(turbo_source), "exec"), namespace)
normalize = namespace["punc_norm"]
tokenizer = Tokenizer.from_file(str(tokenizer_path))
tokenizer.no_padding()
tokenizer.no_truncation()

# safe_open maps the checkpoint and materialises only these two tables.
with safe_open(checkpoint, framework="pt", device="cpu") as weights:
    text_table = weights.get_tensor("text_emb.weight").half()
    speech_table = weights.get_tensor("speech_emb.weight").half()
assert tuple(text_table.shape) == (50276, 1024)
assert tuple(speech_table.shape) == (6563, 1024)

capture = json.loads((captures / "capture.json").read_text())
first = capture["cases"][0]
first_ids = tokenizer.encode(normalize(first["text"])).ids
first_prompt = np.load(captures / "prompt-0.npy", allow_pickle=False).astype("<f2")
prefix_frames = first_prompt.shape[1] - len(first_ids) - 1
assert 1 <= prefix_frames <= 376
prefix = first_prompt[:, :prefix_frames, :].copy()
owner = SimpleNamespace(
    prepare_conditioning=lambda cond: torch.from_numpy(prefix),
    text_emb=torch.nn.Embedding.from_pretrained(text_table),
    speech_emb=torch.nn.Embedding.from_pretrained(speech_table),
    hp=SimpleNamespace(input_pos_emb=None), is_gpt=True)

assets = {}
for name, data in (("text", text_table.numpy()), ("speech", speech_table.numpy()),
                   ("conditioning", prefix)):
    path = output / (name + ".npy")
    np.save(path, data.astype("<f2"), allow_pickle=False)
    assets[name] = {"file": path.name, "sha256": sha(path.read_bytes())}
path = output / "tokenizer.json"
path.write_bytes(tokenizer_path.read_bytes())
assets["tokenizer"] = {"file": path.name, "sha256": sha(path.read_bytes())}
manifest = dict(schema_version=1, voice_prefix_kind="projected_turbo",
                voice_source_sha256=capture["voice_sha256"], **assets)
manifest_path = output / "assets.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

# Fresh words, tags, Unicode, profile overflow and tokenizer truncation join
# the historical end-to-end captures. The largest arrays are only ~2 MiB.
texts = [c["text"] for c in capture["cases"]]
texts.extend(["", "ready, Holmes", "A fresh sentence that was never in the captured prompts.",
              "we are ready [laugh] proceed", "こんにちは世界",
              "x " * (511 - prefix_frames), "x " * (512 - prefix_frames),
              "x " * 1100])
cases = []
for index, raw in enumerate(texts):
    ids = tokenizer.encode(normalize(raw)).ids[:1024]
    prompt, length = namespace["prepare_input_embeds"](
        owner, t3_cond=None, text_tokens=torch.tensor([ids], dtype=torch.long),
        speech_tokens=torch.tensor([[6561]]), cfg_weight=0.0)
    actual = prompt.numpy().astype("<f2")
    assert length == prefix_frames
    if index < len(capture["cases"]):
        existing = np.load(captures / f"prompt-{index}.npy", allow_pickle=False).astype("<f2")
        assert np.array_equal(existing, actual), f"capture parity {index}"
    cases.append(dict(input=raw, input_ids=ids, frames=actual.shape[1],
                      prompt_sha256=sha(actual.tobytes()), admitted=actual.shape[1] <= 512))

assert not torch.cuda.is_initialized()
assert not any(n == "chatterbox" or n.startswith("chatterbox.") for n in sys.modules)
receipt = dict(schema_version=1, source_sha256=sha(source.read_bytes()),
               assets_sha256=sha(manifest_path.read_bytes()), prefix_frames=prefix_frames,
               reference_preparations=0, cuda_initialized=False,
               existing_prompt_comparisons=len(capture["cases"]), cases=cases)
(output / "oracle.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "cases"}))

if retained_reference is not None:
    # The retained archive is a proof/export input only. The native runtime
    # consumes checksummed arrays and JSON, never Torch pickle or audio.
    retained = torch.load(retained_reference, map_location="cpu", weights_only=True)["t3"]
    assert retained["clap_emb"] is None
    speaker = retained["speaker_emb"]
    tokens = retained["cond_prompt_speech_tokens"]
    assert speaker.dtype == torch.float16 and tuple(speaker.shape) == (1, 256)
    assert tokens.dtype == torch.int64 and tokens.ndim == 2 and tokens.shape[0] == 1
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        weight = weights.get_tensor("cond_enc.spkr_enc.weight").half()
        bias = weights.get_tensor("cond_enc.spkr_enc.bias").half()
    projection = torch.nn.Linear(256, 1024).half()
    projection.weight = torch.nn.Parameter(weight, requires_grad=False)
    projection.bias = torch.nn.Parameter(bias, requires_grad=False)
    cond_source = source.parent / "modules/cond_enc.py"
    tree = ast.parse(cond_source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "T3CondEnc")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(cond_source), "exec"), namespace)
    cond_owner = SimpleNamespace(spkr_enc=projection, hp=SimpleNamespace(
        speaker_embed_size=256, use_perceiver_resampler=False, emotion_adv=False))

    def export_array(name, value):
        path = output / (name + ".npy")
        np.save(path, value.detach().numpy().astype("<f2"), allow_pickle=False)
        return dict(file=path.name, sha256=sha(path.read_bytes()))

    weight_asset = export_array("projection-weight", weight)
    bias_asset = export_array("projection-bias", bias)
    prepared_cases = []
    # Include reference-free, reversed, repeated and maximum-length token rows,
    # plus a changed speaker. These are tensor probes, never new audio encoding.
    probes = [("retained", speaker, tokens[0].tolist()),
              ("speaker-only", speaker, []),
              ("reversed", speaker, tokens[0].flip(0).tolist()),
              ("maximum", speaker, [0, 6560, 0] * 125),
              ("zero-speaker", torch.zeros_like(speaker), [0, 1, 6560]),
              ("changed-speaker", -speaker, [2, 1, 2])]
    for label, voice, ids in probes:
        token_path = output / (label + "-tokens.json")
        token_path.write_text(json.dumps(ids) + "\n")
        prepared = dict(source_sha256=sha(retained_reference.read_bytes()),
                        speaker=export_array(label + "-speaker", voice),
                        projection_weight=weight_asset, projection_bias=bias_asset,
                        speech_tokens=dict(file=token_path.name, sha256=sha(token_path.read_bytes())))
        bundle = dict(schema_version=2, voice_prefix_kind="prepared_turbo",
                      voice_source_sha256=capture["voice_sha256"],
                      **{k: assets[k] for k in ("tokenizer", "text", "speech")}, prepared=prepared)
        bundle_path = output / (label + "-assets.json")
        bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
        embedded = torch.nn.functional.embedding(torch.tensor([ids], dtype=torch.long), speech_table)
        cond = SimpleNamespace(speaker_emb=voice, clap_emb=None,
                               cond_prompt_speech_tokens=torch.tensor([ids], dtype=torch.long),
                               cond_prompt_speech_emb=embedded, emotion_adv=None)
        with torch.inference_mode():
            expected = namespace["forward"](cond_owner, cond)
        # Execute the existing assembly method with the independently projected
        # prefix. This compares the final prompt as well as the speaker row.
        probe_owner = SimpleNamespace(**vars(owner))
        probe_owner.prepare_conditioning = lambda cond: expected
        text_ids = tokenizer.encode(normalize("ready, Holmes")).ids
        prompt, _ = namespace["prepare_input_embeds"](
            probe_owner, t3_cond=None, text_tokens=torch.tensor([text_ids]),
            speech_tokens=torch.tensor([[6561]]), cfg_weight=0.0)
        expected_asset = export_array(label + "-expected", expected)
        case = dict(name=label, assets_file=bundle_path.name, assets_sha256=sha(bundle_path.read_bytes()),
                    expected=expected_asset, input="ready, Holmes", input_ids=text_ids,
                    prompt_sha256=sha(prompt.numpy().astype("<f2").tobytes()))
        if label == "retained":
            delta = np.abs(expected.numpy().astype(np.float32) - prefix.astype(np.float32))
            case["captured_prefix_exact"] = bool(np.array_equal(expected.numpy(), prefix))
            case["captured_prefix_max_abs"] = float(delta.max())
        prepared_cases.append(case)
    prepared_receipt = dict(schema_version=1, conditioning_source_sha256=sha(cond_source.read_bytes()),
                            retained_reference_sha256=sha(retained_reference.read_bytes()),
                            reference_preparations=0, cuda_initialized=torch.cuda.is_initialized(),
                            cases=prepared_cases)
    assert not prepared_receipt["cuda_initialized"]
    assert not any(n == "chatterbox" or n.startswith("chatterbox.") for n in sys.modules)
    (output / "prepared-oracle.json").write_text(json.dumps(prepared_receipt, indent=2) + "\n")
    print(json.dumps(prepared_receipt))
