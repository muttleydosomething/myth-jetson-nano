#!/usr/bin/env python3
# Copyright (c) 2026 Hailstorm contributors.
# ConditionalDecoder: Copyright (c) 2024 Alibaba Inc, Apache-2.0.
"""Build-time only: export the admitted estimator plus Euler update as one graph.
No reference preparation or service initializers. Runtime Rust consumes only the
pinned TensorRT engine. A dynamic frame profile is explicit, never a silent crop.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def build_engine(args, receipt):
    # TensorRT opens tactic resources by filename in its own loader scope.
    # Give the loader a directory containing only these NVIDIA resources;
    # preloading them does not satisfy that separate lookup. Re-exec also
    # releases export-time Torch/model RAM before tactic compilation begins.
    route = args.output/"trt-build-libs"
    route.mkdir(exist_ok=True)
    for resource in Path("/host-trt").glob("libnvinfer_builder_resource_*.so.*"):
        link = route/resource.name
        if not link.is_symlink():
            link.symlink_to(resource)
    if str(route) not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(route)+":"+env.get("LD_LIBRARY_PATH", "")
        os.execvpe(sys.executable, [sys.executable, "-S", __file__, str(args.assets),
            str(args.output), "--opt-frames", str(args.opt_frames),
            "--max-frames", str(args.max_frames), "--build-only"], env)
    # Load only the established host CUDA/TRT libraries, never put the host's
    # entire library tree on LD_LIBRARY_PATH (it also contains another libc).
    sys.path.insert(0, "/trt-python")
    import ctypes
    for library in ("/host-cuda/libcudart.so.13", "/host-trt/libnvinfer.so.10",
                    "/host-trt/libnvonnxparser.so.10", "/host-trt/libnvinfer_plugin.so.10"):
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
    import tensorrt_bindings as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.output/"meanflow.onnx")):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256*1024**2)
    config.builder_optimization_level = 2
    profile = builder.create_optimization_profile()
    for name in ["x", "mask", "mu", "spks", "cond", "t", "r"]:
        base = (1,80) if name == "spks" else (1,) if name in ["t","r"] else (1,1,1) if name == "mask" else (1,80,1)
        shapes = [base if len(base)<3 else (*base[:2],n)
                  for n in [1,args.opt_frames,args.max_frames]]
        # Borg's binding returns None and raises ValueError on invalid shapes.
        profile.set_shape(name, *shapes)
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT meanflow engine build failed")
    engine = args.output/"meanflow.engine"
    engine.write_bytes(bytes(plan))
    receipt["engine_sha256"] = sha(engine)
    receipt["tensorrt_version"] = trt.__version__
    (args.output/"export.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--opt-frames", type=int, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-only", action="store_true",
                        help="compile a previously exported ONNX without repeating export")
    args = parser.parse_args()
    if not 1 <= args.opt_frames <= args.max_frames <= 16384:
        parser.error("require 1 <= opt <= max <= 16384")
    # -S prevents the deployed site's service patch from loading models.
    assert sys.flags.no_site, "run with python3 -S"
    if args.build_only:
        receipt = json.loads((args.output/"export.json").read_text())
        assert receipt["onnx_sha256"] == sha(args.output/"meanflow.onnx")
        assert (receipt["opt_frames"],receipt["max_frames"]) == (args.opt_frames,args.max_frames)
        build_engine(args, receipt)
        return
    sys.path.append("/usr/local/lib/python3.10/dist-packages")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox")
    for name, path in [("chatterbox", source), ("chatterbox.models", source/"models"),
                       ("chatterbox.models.s3gen", source/"models/s3gen")]:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    decoder = importlib.import_module("chatterbox.models.s3gen.decoder")
    admitted = json.loads((args.assets/"oracle.json").read_text())
    manifest_path = args.assets/"meanflow.json"
    assert sha(manifest_path) == admitted["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["decoder_kind"] == "s3gen_meanflow_fp32"
    with torch.device("meta"):
        estimator = decoder.ConditionalDecoder(in_channels=320, out_channels=80,
            causal=True, channels=[256], dropout=0., attention_head_dim=64,
            n_blocks=4, num_mid_blocks=12, num_heads=8, act_fn="gelu", meanflow=True)
    state = {}
    for name, asset in manifest["tensors"].items():
        path = args.assets/asset["file"]
        assert path.parent.resolve() == args.assets.resolve()
        assert sha(path) == asset["sha256"]
        if name != "time_frequencies":
            state[name] = torch.from_numpy(np.load(path, allow_pickle=False).copy())
    estimator.load_state_dict(state, strict=True, assign=True)
    estimator.eval()
    # Explicit attention avoids export of a framework-specific fused SDPA op;
    # TensorRT still owns softmax, reductions and all intermediate tensors.
    from diffusers.models.attention_processor import AttnProcessor
    for module in estimator.modules():
        if hasattr(module, "set_processor"):
            module.set_processor(AttnProcessor())
    class Euler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.estimator = estimator
        def forward(self, x, mask, mu, spks, cond, t, r):
            dx = self.estimator(x, mask=mask, mu=mu, spks=spks, cond=cond, t=t, r=r)
            return x + (r-t).reshape(1, 1, 1)*dx, dx
    graph = Euler().eval()
    args.output.mkdir(parents=True, exist_ok=False)
    names = ["x", "mask", "mu", "spks", "cond", "t", "r"]
    length = min(5, args.max_frames)
    examples = (torch.zeros(1,80,length), torch.ones(1,1,length),
                torch.zeros(1,80,length), torch.zeros(1,80),
                torch.zeros(1,80,length), torch.tensor([0.]), torch.tensor([.5]))
    onnx_path = args.output/"meanflow.onnx"
    with torch.inference_mode():
        torch.onnx.export(graph, examples, str(onnx_path), input_names=names,
            output_names=["next_x", "derivative"], opset_version=17,
            dynamic_axes={name:{2:"frames"} for name in
                          ["x","mask","mu","cond","next_x","derivative"]},
            do_constant_folding=True, dynamo=False)
    receipt = dict(schema_version=1, source_sha256=manifest["source_sha256"],
        oracle_manifest_sha256=admitted["manifest_sha256"], onnx_sha256=sha(onnx_path),
        min_frames=1, opt_frames=args.opt_frames, max_frames=args.max_frames,
        reference_preparations=0, precision="fp32", tf32=False)
    (args.output/"export.json").write_text(json.dumps(receipt, indent=2)+"\n")
    if args.build:
        build_engine(args, receipt)
    else:
        print(json.dumps(receipt), flush=True)

if __name__ == "__main__":
    main()
