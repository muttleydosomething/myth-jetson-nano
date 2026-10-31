#!/usr/bin/env python3
"""Capture the deployed Turbo tokenizer oracle without importing a speech model.

Run python3 -S with USE_TORCH=0 USE_TF=0 USE_FLAX=0 in the speech container.
The JSON receipt includes the portable native tokenizer and exact wrapper output.
This is proof tooling only; the Rust runtime never invokes this script.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

# -S excludes the service's sitecustomize hook, which eagerly imports Torch.
# Add installed packages directly; site.main() would run that same hook again.
assert sys.flags.no_site, "Invoke python3 -S to bypass service startup hooks"
sys.path.append("/usr/local/lib/python3.10/dist-packages")

from transformers import AutoTokenizer
import tokenizers
import transformers

source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox/tts_turbo.py")
model_dir = Path(sys.argv[1])
tree = ast.parse(source.read_text())
function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "punc_norm")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
     namespace)
normalize = namespace["punc_norm"]
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
assert tokenizer.is_fast
assert len(tokenizer) == 50276
# Transformers 5.1 imports Torch even with USE_TORCH=0. Keep it CPU-only and
# prove no Chatterbox module or CUDA context (hence no speech model) was loaded.
assert not any(name == "chatterbox" or name.startswith("chatterbox.")
               for name in sys.modules), "Oracle must not import Chatterbox"
torch_module = sys.modules.get("torch")
assert torch_module is None or not torch_module.cuda.is_initialized()

# Capture the unconfigured native data; Transformers installs per-call truncation.
native = tokenizer.backend_tokenizer.to_str()
cases = [
    "", " ", "\t\n", "ready, Holmes", " leading whitespace", "hello… world",
    "a:b—c–d", "“quoted” and ‘single’", "ßeta", "école", "e\u0301cole",
    "I'm ready; we've arrived, hadn't we?", "123.45 -6.7 +8 1e9",
    "path/to/file \\ bracket [unknown]", "<|endoftext|>", "one<|endoftext|>two",
    "[laugh]", "before[laugh]after", "[laugh][cough]", "[LAUGH]",
    "[clear throat] indeed", "[clear  throat]", "[ clear throat ]",
    "Привет мир", "こんにちは世界", "你好世界", "مرحبا بالعالم", "שלום עולם",
    "안녕하세요", "🙂🚀✨", "\x00\x01\x1c\x1f\x7f", "a\u00a0b\u2003c",
    "a\u200bb\u200dc", "!!!???", "...", "x " * 1023,
    "x " * 1024, "x " * 1025, "[laugh]" * 1030,
]
# Include every configured added token, both alone and touching ordinary text.
for item in tokenizer.added_tokens_decoder.values():
    cases.extend([item.content, f"left{item.content}right"])
records = []
for raw in cases:
    text = normalize(raw)
    full = tokenizer(text, padding=False, truncation=False)["input_ids"]
    result = tokenizer(text, padding=True, truncation=True)
    records.append(dict(input=raw, normalized=text, input_ids=result["input_ids"],
                        attention_mask=result["attention_mask"],
                        untruncated_length=len(full),
                        truncated=len(full) > len(result["input_ids"])))

assets = {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
          for name in ["vocab.json", "merges.txt", "tokenizer_config.json",
                       "added_tokens.json", "special_tokens_map.json"]}
print(json.dumps(dict(
    schema_version=1, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    snapshot=model_dir.name, transformers_version=transformers.__version__,
    tokenizers_version=tokenizers.__version__, tokenizer_class=type(tokenizer).__name__,
    tokenizer_sha256=hashlib.sha256(native.encode()).hexdigest(), assets=assets,
    model_max_length=tokenizer.model_max_length, tokenizer_json=native, cases=records),
    ensure_ascii=False))
