#!/usr/bin/env python3
"""Isolated TensorRT decode experiment for the existing Chatterbox benchmark.

The current speech image exports its own HF model at its deployed FP16 precision.
Borg's host TensorRT compiles/runs it separately: no PyTorch downgrade, service
replacement, voice change, precision change to the audio generator, or host
package installation. This gate measures transformer decode, not audible speech.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def available():
    return int(next(x for x in Path("/proc/meminfo").read_text().splitlines()
                    if x.startswith("MemAvailable:")).split()[1]) * 1024


def guarded(argv, root, reserve=512 * 1024**2, available=available, timeout=600):
    """Own one process group; pressure, deadline and cancellation stop only it."""
    root.mkdir(parents=True, exist_ok=True)
    result = {"argv": argv, "started_at": now(), "reason": None}
    started = time.monotonic()
    prior = {}
    process = None
    try:
        def interrupted(signum, frame):
            raise InterruptedError(f"signal {signum}")
        for sig in (signal.SIGTERM, signal.SIGINT):
            prior[sig] = signal.signal(sig, interrupted)
        with (root / "command.log").open("w") as output:
            process = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            result["pid"] = process.pid
            minimum = available()
            while process.poll() is None:
                minimum = min(minimum, available())
                if minimum < reserve:
                    result["reason"] = "memory-reserve"
                    break
                if time.monotonic() - started > timeout:
                    result["reason"] = "timeout"
                    break
                time.sleep(0.1)
            result["minimum_available_bytes"] = minimum
    except BaseException as error:
        result["reason"] = repr(error)
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # Natural exit raced the pressure/cancellation check.
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            result["returncode"] = process.wait()
        result.update(finished_at=now(), duration_seconds=time.monotonic() - started)
        write_json(root / "command.json", result)
        for sig, handler in prior.items():
            signal.signal(sig, handler)
    return result


def export(root, native_only=False, unified=False, compare_mask=False, backend_probe=False):
    """Export the actual HF implementation with a tensor-only KV boundary."""
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import GPT2Config, GPT2Model
    from transformers.cache_utils import DynamicCache
    from chatterbox.models.t3.llama_configs import GPT2_MEDIUM_CONFIG

    torch.set_num_threads(4)
    torch.manual_seed(42)
    checkpoints = list(Path("/app/hf_cache").rglob("t3_turbo_v1.safetensors"))
    assert len(checkpoints) == 1, checkpoints
    config = GPT2Config(**GPT2_MEDIUM_CONFIG)
    config._attn_implementation = "sdpa"
    # Meta construction avoids allocating the unused token table and 24 copies
    # of the 8196-square causal buffer. SDPA does not read that eager-only buffer.
    with torch.device("meta"):
        model = GPT2Model(config)
        head = torch.nn.Linear(config.n_embd, 6563)
    del model.wte
    weights = {}
    with safe_open(checkpoints[0], framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if key.startswith("tfmr.") and not key.startswith("tfmr.wte."):
                weights[key.removeprefix("tfmr.")] = checkpoint.get_tensor(key).half()
        head.load_state_dict({key.removeprefix("speech_head."):
                              checkpoint.get_tensor(key).half()
                              for key in checkpoint.keys() if key.startswith("speech_head.")},
                             assign=True)
        embedding = checkpoint.get_tensor("speech_emb.weight").half()
    model.load_state_dict(weights, assign=True)
    for block in model.h:
        block.attn.bias = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    model = model.eval().cuda()
    head = head.eval().cuda()

    class Decode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.head = head
            self.explicit = False

        def forward(self, embeds, position, *past):
            cache = DynamicCache(tuple(zip(past[::2], past[1::2])))
            # A one-token decode has no future keys. Explicit 4-D mask prevents
            # tracing HF's Python mask-construction heuristics as a fixed shape.
            if unified and (torch.onnx.is_in_onnx_export() or self.explicit):
                keys = torch.arange(past[0].shape[2] + embeds.shape[1], device=embeds.device)
                allowed = keys.unsqueeze(0) <= position.unsqueeze(1)
                mask = torch.where(allowed, 0.0, -65504.0).to(embeds.dtype).unsqueeze(0).unsqueeze(0)
            elif unified:
                # Reference/timing calls must retain native HF/SDPA kernel
                # selection. Only ONNX tracing needs the explicit dynamic mask.
                mask = None
            else:
                mask = embeds.new_zeros((1, 1, 1, past[0].shape[2] + 1))
            output = self.model(inputs_embeds=embeds, position_ids=position.view(1, -1),
                                cache_position=position, attention_mask=mask,
                                past_key_values=cache, use_cache=True, return_dict=True)
            return (self.head(output.last_hidden_state),
                    *((output.last_hidden_state,) if unified else ()),
                    *(tensor for layer in output.past_key_values for tensor in layer[:2]))

    wrapper = Decode().eval()
    names = ["embeds", "position"] + [
        f"past_{kind}_{i}" for i in range(config.n_layer) for kind in ("k", "v")]
    outputs = ["logits"] + (["hidden"] if unified else []) + [
        f"present_{kind}_{i}" for i in range(config.n_layer) for kind in ("k", "v")]
    if compare_mask:
        compare_masks(wrapper, root, names, outputs, backend_probe=backend_probe)
        return
    report = {"started_at": now(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "precision": "float16",
              "checkpoint": str(checkpoints[0]), "cases": [],
              "scope": "prefill/decode" if unified else "one-token decode; no speech quality claim"}
    with torch.inference_mode():
        # Real model-generated KV, rather than a zero-filled cache, reveals bad
        # position handling and cache updates. Different lengths test the profile.
        cases = ((128, 1), (384, 1), (0, 17), (0, 128)) if unified else ((128, 1), (384, 1))
        for length, query in cases:
            ids = torch.arange(length) % embedding.shape[0]
            prefix = embedding[ids].unsqueeze(0).cuda()
            if length:
                cache = model(inputs_embeds=prefix, use_cache=True).past_key_values
                past = tuple(t for layer in cache for t in layer[:2])
            else:
                past = tuple(torch.empty(1, 16, 0, 64, device="cuda", dtype=torch.float16)
                             for _ in range(48))
            embeds = embedding[(torch.arange(query) + 42) % embedding.shape[0]].unsqueeze(0).cuda()
            position = torch.arange(length, length + query, device="cuda", dtype=torch.int64)
            args = (embeds, position, *past)
            reference = None if native_only else wrapper(*args)
            # Compare the exported call with the ordinary deployed HF call.
            if not native_only:
                native_cache = DynamicCache(tuple(zip(past[::2], past[1::2])))
                native = model(inputs_embeds=embeds, past_key_values=native_cache, use_cache=True)
                torch.testing.assert_close(reference[0], head(native.last_hidden_state), rtol=0.02, atol=0.05)
            timings = []
            for index in range(12):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                if native_only:
                    # Use the ordinary HF call, without the export-only mask, so
                    # kernel selection matches the deployed PyTorch decode.
                    native_cache = DynamicCache(tuple(zip(past[::2], past[1::2])))
                    result = head(model(inputs_embeds=embeds, past_key_values=native_cache,
                                        use_cache=True).last_hidden_state)
                else:
                    result = wrapper(*args)
                torch.cuda.synchronize()
                if index >= 2:
                    timings.append(time.perf_counter() - tick)
            if native_only:
                report["cases"].append({"past_length": length, "pytorch_native_seconds": timings})
                continue
            values = {name: tensor.cpu().numpy() for name, tensor in zip(names, args)}
            values.update({name: tensor.cpu().numpy() for name, tensor in zip(outputs, reference)})
            np.savez(root / f"case-{length}-{query}.npz" if unified else root / f"case-{length}.npz", **values)
            report["cases"].append({"past_length": length, "query_length": query, "pytorch_seconds": timings,
                                    "wrapper_matches_native": True})
            if length == 128:
                dynamic = {name: {2: "past_length"} for name in names[2:]}
                dynamic.update({name: {2: "present_length"} for name in outputs[1:]})
                if unified:
                    dynamic.update({"embeds": {1: "query_length"}, "position": {0: "query_length"},
                                    "logits": {1: "query_length"}, "hidden": {1: "query_length"}})
                torch.onnx.export(wrapper, args, str(root / "decode.onnx"),
                                  input_names=names, output_names=outputs, dynamic_axes=dynamic,
                                  opset_version=17, dynamo=False)
    report["finished_at"] = now()
    write_json(root / ("native.json" if native_only else "export.json"), report)


class Cuda:
    """Minimal CUDA Runtime binding; no new host Python dependency."""
    def __init__(self):
        self.lib = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(self.stream)))
        self.pointers = []

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA Runtime status {code}")

    def allocate(self, size):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), size))
        self.pointers.append(ptr)
        return ptr

    def close(self):
        for ptr in self.pointers:
            self.check(self.lib.cudaFree(ptr))
        self.check(self.lib.cudaStreamDestroy(self.stream))


def infer(root):
    """Check outputs and time resident input/output bindings, excluding copies."""
    import numpy as np
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine((root / "decode.engine").read_bytes())
    assert engine is not None, "TensorRT engine could not be deserialised"
    context = engine.create_execution_context()
    report = {"started_at": now(), "tensorrt": trt.__version__, "cases": [],
              "timing_scope": "resident decode plus synchronisation, excludes host copies",
              "quality": "not listened to; numerical gate only"}
    try:
        for case in sorted(root.glob("case-*.npz")):
            data = np.load(case)
            length = data["past_k_0"].shape[2]
            cuda = Cuda()
            arrays = {}
            pointers = {}
            try:
                for name in engine:
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        assert context.set_input_shape(name, data[name].shape)
                for name in engine:
                    arrays[name] = np.empty(tuple(context.get_tensor_shape(name)),
                                            dtype=trt.nptype(engine.get_tensor_dtype(name)))
                    pointers[name] = cuda.allocate(max(arrays[name].nbytes, 1))
                    assert context.set_tensor_address(name, pointers[name].value)
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        values = np.ascontiguousarray(data[name])
                        assert values.dtype == arrays[name].dtype and values.nbytes == arrays[name].nbytes, name
                        cuda.check(cuda.lib.cudaMemcpy(pointers[name], values.ctypes.data,
                                                       values.nbytes, 1))
                timings = []
                for index in range(12):
                    cuda.check(cuda.lib.cudaDeviceSynchronize())
                    tick = time.perf_counter()
                    assert context.execute_async_v3(cuda.stream.value)
                    cuda.check(cuda.lib.cudaDeviceSynchronize())
                    if index >= 2:
                        timings.append(time.perf_counter() - tick)
                metrics = []
                for name in engine:
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                        actual = arrays[name]
                        cuda.check(cuda.lib.cudaMemcpy(actual.ctypes.data, pointers[name],
                                                       actual.nbytes, 2))
                        # Inspect appended KV separately: unchanged cache prefixes
                        # otherwise dominate a superficial full-array comparison.
                        expected = data[name]
                        retained_matches = not name.startswith("present_") or np.array_equal(
                            actual[:, :, :length], expected[:, :, :length])
                        if name.startswith("present_"):
                            actual, expected = actual[:, :, length:], expected[:, :, length:]
                        delta = actual.astype("float32") - expected.astype("float32")
                        passed = bool(retained_matches and np.isfinite(actual).all() and
                                      np.allclose(actual, expected, rtol=0.02, atol=0.05))
                        metrics.append({"name": name, "passed": passed,
                                        "max_abs_error": float(np.max(np.abs(delta)))})
                token_matches = bool(arrays["logits"][:, -1].argmax() == data["logits"][:, -1].argmax())
                row = {"past_length": length, "query_length": data["embeds"].shape[1], "tensorrt_seconds": timings,
                       "top_token_matches": token_matches,
                       "outputs": metrics, "passed": token_matches and all(x["passed"] for x in metrics)}
                report["cases"].append(row)
                write_json(root / "inference.json", report)
                assert row["passed"], f"Numerical gate failed at past length {length}"
            finally:
                cuda.close()
    finally:
        report["finished_at"] = now()
        write_json(root / "inference.json", report)


def run(root, unified=False):
    """Admit the stopped deployment, export, compile and compare sequentially."""
    root.mkdir(parents=True, exist_ok=False)
    docker = ["sudo", "-n", "docker"]
    deployed = json.loads(subprocess.check_output(docker + ["inspect", "chatterbox-tts"], text=True))[0]
    assert not deployed["State"]["Running"], "Refuse to trial alongside live Chatterbox"
    assert subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 1, "LFM3 must remain stopped"
    assert available() > 5 * 1024**3, "Insufficient memory for guarded export"
    name = "chatterbox-trt-export-" + str(os.getpid())
    script = str(Path(__file__).resolve())
    report = {"started_at": now(), "image": deployed["Image"], "phases": [],
              "script_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest()}
    try:
        argv = docker + ["run", "--name", name, "--runtime", "nvidia", "--network", "none",
                         "--restart", "no", "--memory", "5g", "--memory-swap", "5g",
                         "--volumes-from", "chatterbox-tts", "-v", f"{root}:/trial",
                         "-v", f"{script}:/tensorrt-trial.py:ro", "--entrypoint", "python3"]
        # Preserve the working CUDA compatibility mounts and environment, while
        # leaving allocation caching enabled for the comparator.
        for entry in deployed["Config"]["Env"]:
            if entry.split("=", 1)[0] != "PYTORCH_NO_CUDA_MEMORY_CACHING":
                argv += ["-e", entry]
        argv += [deployed["Image"], "/tensorrt-trial.py", "export-unified" if unified else "export", "--output-dir", "/trial"]
        try:
            phase = guarded(argv, root / "export-command")
            report["phases"].append(phase)
            assert phase["returncode"] == 0 and phase["reason"] is None, "Export failed"
        finally:
            subprocess.run(docker + ["stop", "-t", "5", name], capture_output=True, timeout=20)
            state = json.loads(subprocess.check_output(docker + ["inspect", name], text=True))[0]
            report["export_container_stopped"] = not state["State"]["Running"]
            assert report["export_container_stopped"], "Owned export container did not stop"
        names = [f"past_{kind}_{i}" for i in range(24) for kind in ("k", "v")]
        def shapes(length, query=1):
            return ",".join([f"embeds:1x{query}x1024", f"position:{query}"] +
                            [f"{name}:1x16x{length}x64" for name in names])
        argv = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={root}/decode.onnx",
                f"--saveEngine={root}/decode.engine", "--fp16", "--noTF32",
                "--memPoolSize=workspace:256", "--builderOptimizationLevel=2",
                "--maxAuxStreams=0", "--skipInference",
                "--minShapes=" + shapes(0 if unified else 1), "--optShapes=" + shapes(128),
                "--maxShapes=" + shapes(1024 if unified else 512, 512 if unified else 1)]
        phase = guarded(argv, root / "build-command")
        report["phases"].append(phase)
        assert phase["returncode"] == 0 and phase["reason"] is None, "Engine build failed"
        phase = guarded(["python3", script, "infer", "--output-dir", str(root)],
                        root / "infer-command")
        report["phases"].append(phase)
        assert phase["returncode"] == 0 and phase["reason"] is None, "Inference gate failed"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report.update(finished_at=now(),
                      lfm3_running_at_finish=bool(subprocess.run(
                          ["pgrep", "-x", "llama-server"], capture_output=True).stdout))
        write_json(root / "result.json", report)


def validate(root):
    """Resolve review findings without repeating the accepted export/build."""
    docker = ["sudo", "-n", "docker"]
    deployed = json.loads(subprocess.check_output(docker + ["inspect", "chatterbox-tts"], text=True))[0]
    assert not deployed["State"]["Running"], "Speech became active"
    assert subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 1
    previous = json.loads((root / "result.json").read_text())
    argv = previous["phases"][0]["argv"][:]
    name = "chatterbox-trt-native-" + str(os.getpid())
    argv[argv.index("--name") + 1] = name
    argv[argv.index("export")] = "native"
    report = {"started_at": now(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        try:
            phase = guarded(argv, root / "native-command")
            report["native"] = phase
            assert phase["returncode"] == 0 and phase["reason"] is None, "Native comparison failed"
        finally:
            subprocess.run(docker + ["stop", "-t", "5", name], capture_output=True, timeout=20)
            state = json.loads(subprocess.check_output(docker + ["inspect", name], text=True))[0]
            report["native_container_stopped"] = not state["State"]["Running"]
            assert report["native_container_stopped"]
        phase = guarded(["python3", str(Path(__file__).resolve()), "infer", "--output-dir", str(root)],
                        root / "infer-reviewed-command")
        report["inference"] = phase
        assert phase["returncode"] == 0 and phase["reason"] is None, "Reviewed numerical gate failed"
    finally:
        report["finished_at"] = now()
        write_json(root / "validation.json", report)


def probe_export(root, replay=False, attention=False):
    """Instrument the first block with the retained failed prompt's exact inputs.

    All references use deployed FP16 weights. Explicit-mask PyTorch separates
    export mask semantics from TensorRT arithmetic; hooks expose each existing
    module boundary without substituting an attention implementation.
    """
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import GPT2Config, GPT2Model
    from transformers.cache_utils import DynamicCache
    from chatterbox.models.t3.llama_configs import GPT2_MEDIUM_CONFIG

    torch.set_num_threads(4)
    config = GPT2Config(**GPT2_MEDIUM_CONFIG)
    config.n_layer = 1
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = GPT2Model(config)
    del model.wte
    wanted = set(model.state_dict())
    checkpoints = list(Path("/app/hf_cache").rglob("t3_turbo_v1.safetensors"))
    assert len(checkpoints) == 1
    with safe_open(checkpoints[0], framework="pt", device="cpu") as checkpoint:
        model.load_state_dict({key.removeprefix("tfmr."): checkpoint.get_tensor(key).half()
                               for key in checkpoint.keys() if key.startswith("tfmr.")
                               and key.removeprefix("tfmr.") in wanted}, assign=True)
    model.h[0].attn.bias = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    model = model.eval().cuda()
    data = np.load(root / "source-case.npz")
    embeds = torch.from_numpy(data["embeds"]).cuda()
    position = torch.from_numpy(data["position"]).cuda()
    captured = {}
    captured_inputs = {}
    block = model.h[0]
    stages = {"ln1": block.ln_1, "qkv": block.attn.c_attn,
              "attention_projection": block.attn.c_proj, "ln2": block.ln_2,
              "mlp_fc": block.mlp.c_fc, "mlp_activation": block.mlp.act,
              "mlp_projection": block.mlp.c_proj}
    for name, module in stages.items():
        module.register_forward_pre_hook(lambda module, inputs, name=name:
                                         captured_inputs.__setitem__(name, inputs[0]))
        module.register_forward_hook(lambda module, inputs, output, name=name:
                                     captured.__setitem__(name, output))
    block.register_forward_pre_hook(lambda module, inputs:
                                    captured.__setitem__("block_input", inputs[0]))
    block.attn.c_proj.register_forward_pre_hook(lambda module, inputs:
                                               captured.__setitem__("attention_context", inputs[0]))
    block.register_forward_hook(lambda module, inputs, output:
                                captured.__setitem__("block_output", output[0]))
    outputs = ["block_input", "ln1", "qkv", "attention_context", "attention_projection",
               "ln2", "mlp_fc", "mlp_activation", "mlp_projection", "block_output",
               "present_k_0", "present_v_0"]

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.explicit = False

        def forward(self, embeds, position):
            mask = None
            if self.explicit:
                allowed = position.unsqueeze(0) <= position.unsqueeze(1)
                mask = torch.where(allowed, 0.0, -65504.0).to(embeds.dtype)[None, None]
            result = self.model(inputs_embeds=embeds, position_ids=position[None],
                                cache_position=position, attention_mask=mask,
                                past_key_values=DynamicCache(), use_cache=True)
            return (*(captured[name] for name in outputs[:-2]),
                    *next(iter(result.past_key_values))[:2])

    wrapper = Probe().eval()
    with torch.inference_mode():
        if attention:
            # Observe the actual SDPA boundary, including its layout and flags;
            # rebuilding Q/K/V from projections could introduce another cause.
            functional = torch.nn.functional
            original = functional.scaled_dot_product_attention
            calls = []

            def observe(query, key, value, *args, **kwargs):
                result = original(query, key, value, *args, **kwargs)
                calls.append(((query.clone(), key.clone(), value.clone()),
                              args, dict(kwargs), result.clone()))
                return result

            try:
                functional.scaled_dot_product_attention = observe
                native = tuple(t.clone() for t in wrapper(embeds, position))
            finally:
                functional.scaled_dot_product_attention = original
            assert len(calls) == 1
        else:
            native = tuple(t.clone() for t in wrapper(embeds, position))
        # This anchors the reduced probe to the original 24-layer failed case.
        for name, tensor in zip(outputs[-2:], native[-2:]):
            assert np.array_equal(tensor.cpu().numpy(), data[name]), name
        if attention:
            qkv, args, kwargs, context = calls[0]
            merged = context.transpose(1, 2).reshape_as(captured["attention_context"])
            assert torch.equal(merged, captured["attention_context"])
            export_attention(root, qkv, args, kwargs, context)
            return
        if replay:
            # Replay each module against its *native* input, never the preceding
            # TensorRT result. This distinguishes local from propagated error.
            stage_names = list(stages)
            inputs = tuple(captured_inputs[name].clone() for name in stage_names)
            expected = {name: captured[name].clone() for name in stage_names}
            values = {"input_" + name: tensor.cpu().numpy()
                      for name, tensor in zip(stage_names, inputs)}
            values.update({name: tensor.cpu().numpy() for name, tensor in expected.items()})
            for name, tensor in zip(stage_names, inputs):
                module = stages[name]
                x = tensor.float()
                if isinstance(module, torch.nn.LayerNorm):
                    high = torch.nn.functional.layer_norm(x, module.normalized_shape,
                                module.weight.float(), module.bias.float(), module.eps)
                elif hasattr(module, "weight"):
                    # HF Conv1D stores its matrix in input/output order.
                    high = torch.addmm(module.bias.float(), x.reshape(-1, x.shape[-1]),
                                       module.weight.float()).view(*x.shape[:-1], -1)
                else:
                    high = module(x)
                values["fp32_" + name] = high.cpu().numpy()

            class Replay(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.operations = torch.nn.ModuleList(stages.values())

                def forward(self, *values):
                    return tuple(module(value) for module, value in zip(self.operations, values))

            np.savez(root / "probe.npz", **values)
            torch.onnx.export(Replay().eval(), inputs, str(root / "probe.onnx"),
                              input_names=["input_" + name for name in stage_names],
                              output_names=stage_names, opset_version=17, dynamo=False)
            write_json(root / "probe-export.json", {
                "first_layer_kv_bitwise_matches_retained_case": True,
                "method": "Independent module replay from native inputs, with FP32 references",
                "stages": stage_names, "finished_at": now()})
            return
        wrapper.explicit = True
        explicit = tuple(t.clone() for t in wrapper(embeds, position))
        values = {"embeds": data["embeds"], "position": data["position"]}
        values.update({name: tensor.cpu().numpy() for name, tensor in zip(outputs, native)})
        values.update({"masked_" + name: tensor.cpu().numpy() for name, tensor in zip(outputs, explicit)})
        np.savez(root / "probe.npz", **values)
        torch.onnx.export(wrapper, (embeds, position), str(root / "probe.onnx"),
                          input_names=["embeds", "position"], output_names=outputs,
                          opset_version=17, dynamo=False)
    write_json(root / "probe-export.json", {"started_from": "retained case-0-128.npz",
               "first_layer_kv_bitwise_matches_retained_case": True,
               "stages": outputs, "torch": torch.__version__, "finished_at": now()})


def export_attention(root, qkv, args, kwargs, context):
    """Replay only attention from the native first-layer inputs, without weights."""
    import numpy as np
    import torch
    from torch.nn import functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    started = now()
    assert not args and kwargs.get("is_causal") is True
    assert kwargs.get("attn_mask") is None and kwargs.get("dropout_p", 0) == 0
    assert not kwargs.get("enable_gqa", False)
    scale = kwargs.get("scale")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        flash = functional.scaled_dot_product_attention(*qkv, **kwargs)
    assert torch.equal(flash, context), "Captured native attention must be FlashAttention"
    query, key, value = qkv
    allowed = torch.ones(query.shape[-2], key.shape[-2], device=query.device,
                         dtype=torch.bool).tril()
    mask = torch.where(allowed, 0.0, -65504.0).to(query.dtype)[None, None]

    class Attention(torch.nn.Module):
        def forward(self, query, key, value):
            # This is the existing export path, now isolated from projections,
            # residuals and every other transformer layer.
            return functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=0.0,
                is_causal=False, scale=scale)

    operation = Attention().eval()
    masked = operation(*qkv)
    q, k, v = (tensor.cpu().double() for tensor in qkv)
    scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5 if scale is None else scale)
    reference = torch.softmax(scores.masked_fill(~allowed.cpu(), float("-inf")), -1) @ v
    np.savez(root / "probe.npz", input_query=query.cpu().numpy(),
             input_key=key.cpu().numpy(), input_value=value.cpu().numpy(),
             attention_context=context.cpu().numpy(),
             masked_attention_context=masked.cpu().numpy(),
             float64_attention_context=reference.numpy())
    torch.onnx.export(operation, qkv, str(root / "probe.onnx"),
                      input_names=["input_query", "input_key", "input_value"],
                      output_names=["attention_context"], opset_version=17, dynamo=False)
    write_json(root / "probe-export.json", {
        "started_at": started, "finished_at": now(), "torch": torch.__version__,
        "first_layer_kv_bitwise_matches_retained_case": True,
        "native_attention_bitwise_matches_forced_flash": True,
        "context_bitwise_matches_native_projection_input": True,
        "input_shapes": [list(t.shape) for t in qkv],
        "native_input_strides": [list(t.stride()) for t in qkv],
        "method": "Identical native Q/K/V; isolated exported attention; float64 CPU reference",
        "scope": "First layer of retained 128-token prompt; no full-model or speech claim"})


def probe_infer(root):
    """Report every first-block discrepancy without relaxing the speech gate."""
    import numpy as np
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine((root / "probe.engine").read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    data = np.load(root / "probe.npz")
    cuda = Cuda()
    arrays, pointers = {}, {}
    report = {"started_at": now(), "comparisons": {},
              "limitation": "Exposed intermediate outputs can change TensorRT fusion; no speech or latency claim."}
    def metric(actual, expected):
        a, e = actual.astype("float32"), expected.astype("float32")
        error = np.abs(a - e)
        tolerance = 0.05 + 0.02 * np.abs(e)
        return {"passed": bool(np.isfinite(a).all() and np.all(error <= tolerance)),
                "max_abs_error": float(error.max()),
                "rms_error": float(np.sqrt(np.mean((a - e)**2))),
                "outside_tolerance": int(np.count_nonzero(error > tolerance)),
                "elements": int(error.size)}
    try:
        # Layer sweeps reuse the dynamic full-model graph and its exact retained
        # inputs. Resolve shapes before allocating any diagnostic output buffers.
        for name in engine:
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                assert context.set_input_shape(name, data[name].shape), name
        for name in engine:
            arrays[name] = np.empty(tuple(context.get_tensor_shape(name)),
                                    dtype=trt.nptype(engine.get_tensor_dtype(name)))
            pointers[name] = cuda.allocate(max(1, arrays[name].nbytes))
            assert context.set_tensor_address(name, pointers[name].value)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                value = np.ascontiguousarray(data[name])
                assert value.dtype == arrays[name].dtype and value.nbytes == arrays[name].nbytes
                cuda.check(cuda.lib.cudaMemcpy(pointers[name], value.ctypes.data, value.nbytes, 1))
        assert context.execute_async_v3(cuda.stream.value)
        cuda.check(cuda.lib.cudaDeviceSynchronize())
        for name in engine:
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                actual = arrays[name]
                cuda.check(cuda.lib.cudaMemcpy(actual.ctypes.data, pointers[name], actual.nbytes, 2))
                report["comparisons"][name] = {"trt_vs_native": metric(actual, data[name])}
                if "masked_" + name in data:
                    report["comparisons"][name].update({
                        "mask_vs_native": metric(data["masked_" + name], data[name]),
                        "trt_vs_mask": metric(actual, data["masked_" + name])})
                if "fp32_" + name in data:
                    report["comparisons"][name].update({
                        "native_vs_fp32": metric(data[name], data["fp32_" + name]),
                        "trt_vs_fp32": metric(actual, data["fp32_" + name])})
                if "float64_" + name in data:
                    # Keep the independent reference in float64 throughout the
                    # comparison; casting it down can hide half-rounding errors.
                    reference = data["float64_" + name]
                    report["comparisons"][name]["float64_reference"] = {
                        label: {"max_abs_error": float(np.abs(value.astype("float64") - reference).max()),
                                "rms_error": float(np.sqrt(np.mean((value.astype("float64") - reference)**2)))}
                        for label, value in (("native_flash", data[name]),
                                             ("pytorch_mask", data["masked_" + name]),
                                             ("tensorrt", actual))}
        if "logits" in arrays:
            # Match the original final-token gate, while exposing disagreements
            # elsewhere in the prompt rather than hiding them behind one token.
            tokens = arrays["logits"].argmax(-1)
            expected_tokens = data["logits"].argmax(-1)
            report["token_disagreement_positions"] = np.flatnonzero(
                tokens != expected_tokens).tolist()
            report["final_token_matches"] = bool(np.array_equal(
                tokens[:, -1], expected_tokens[:, -1]))
            report["first_failing_block"] = next((i for i in range(24)
                if f"block_{i}" in report["comparisons"] and not
                report["comparisons"][f"block_{i}"]["trt_vs_native"]["passed"]), None)
        np.savez(root / "probe-actual.npz", **arrays)
    finally:
        cuda.close()
        report["finished_at"] = now()
        write_json(root / "probe-inference.json", report)


def layer_graph(root):
    """Expose existing block boundaries; reuse weights and retained references."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    source = Path("/source-model")
    graph = onnx.load(str(source / "decode.onnx"), load_external_data=False)
    nodes = {node.name: node for node in graph.graph.node}
    with np.load(source / "case-0-128.npz") as inputs, np.load(
            "/source-trace/mask-outputs.npz") as trace:
        # Identity against the previously accepted native capture prevents a
        # changed checkpoint, input or backend from becoming a new baseline.
        values = {name: inputs[name] for name in inputs.files}
        for output in graph.graph.output:
            assert np.array_equal(inputs[output.name], trace[output.name]), output.name
        for index in range(24):
            name = f"block_{index}"
            boundary = nodes[f"/model/h.{index}/Add_1"]
            assert boundary.op_type == "Add" and len(boundary.output) == 1
            graph.graph.node.append(helper.make_node(
                "Identity", [boundary.output[0]], [name], name="observe_" + name))
            graph.graph.output.append(helper.make_tensor_value_info(
                name, TensorProto.FLOAT16, [1, "query_length", 1024]))
            values[name] = trace[name]
        values.update({name: trace[name] for name in trace.files if name.startswith("masked_")})
        np.savez(root / "probe.npz", **values)
    # Keep external weight references unchanged; the orchestrator has already
    # linked their exact original companions beside this diagnostic graph.
    onnx.save_model(graph, str(root / "probe.onnx"))
    write_json(root / "probe-export.json", {
        "native_bitwise_matches_retained_case": True, "finished_at": now(),
        "scope": "24 block outputs of the retained synthetic 128-token failing prompt",
        "limitation": "Observing outputs can alter fusion; this is localisation, not a speech gate."})


def layer_sweep(root):
    """Trace the latest rejected precision candidate without another model export."""
    root.mkdir(parents=True, exist_ok=False)
    source = root.parent / "tensorrt-unified-native-reference"
    corrected = root.parent / "tensorrt-projections-fp32"
    traces = root.parent / "tensorrt-mask-comparison"
    docker = ["sudo", "-n", "docker"]
    deployed = json.loads(subprocess.check_output(docker + ["inspect", "chatterbox-tts"], text=True))[0]
    assert not deployed["State"]["Running"]
    assert subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 1
    assert available() > 3 * 1024**3
    # External ONNX weights can be many companions. Link the originals, rather
    # than duplicating them or reading them into another framework's memory.
    for companion in source.iterdir():
        if companion.is_file() and companion.name != "probe.onnx":
            (root / companion.name).symlink_to(companion)
    previous = json.loads((source / "result.json").read_text())
    argv = previous["phases"][0]["argv"][:]
    name = "chatterbox-trt-layers-" + str(os.getpid())
    argv[argv.index("--name") + 1] = name
    argv = [f"{root}:/trial" if a.endswith(":/trial") else
            f"{Path(__file__).resolve()}:/tensorrt-trial.py:ro" if a.endswith(":/tensorrt-trial.py:ro") else
            "layer_graph" if a == "export-unified" else a for a in argv]
    argv[argv.index("--entrypoint"):argv.index("--entrypoint")] = [
        "-v", f"{source}:/source-model:ro", "-v", f"{traces}:/source-trace:ro"]
    report = {"started_at": now(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "precision_source": str(corrected / "result.json")}
    try:
        try:
            report["export"] = guarded(argv, root / "export-command")
            assert report["export"]["returncode"] == 0 and report["export"]["reason"] is None
        finally:
            subprocess.run(docker + ["stop", "-t", "5", name], capture_output=True, timeout=20)
            state = json.loads(subprocess.check_output(docker + ["inspect", name], text=True))[0]
            report["container_stopped"] = not state["State"]["Running"]
            assert report["container_stopped"]
        build = json.loads((corrected / "result.json").read_text())["build"]["argv"]
        build = [f"--onnx={root}/probe.onnx" if a.startswith("--onnx=") else
                 f"--saveEngine={root}/probe.engine" if a.startswith("--saveEngine=") else a
                 for a in build]
        report["build"] = guarded(build, root / "build-command")
        # The old builder crossed the reserve during its post-build load.
        # Retain that fact and test a fully serialised engine separately, never
        # pretending a pressure-stopped command passed or lowering the reserve.
        ready = (root / "probe.engine").is_file() and (root / "probe.engine").stat().st_size > 0
        assert ready and report["build"]["reason"] in (None, "memory-reserve")
        assert report["build"]["returncode"] == 0 or report["build"]["reason"] == "memory-reserve"
        report["infer"] = guarded(["python3", str(Path(__file__).resolve()), "probe_infer",
                                   "--output-dir", str(root)], root / "infer-command")
        assert report["infer"]["returncode"] == 0 and report["infer"]["reason"] is None
    finally:
        report["finished_at"] = now()
        write_json(root / "probe-result.json", report)


def compare_masks(wrapper, root, names, outputs, backend_probe=False):
    """Isolate the export mask in PyTorch, before any ONNX/TensorRT arithmetic.

    The entire native result must match the retained failed case bit for bit.
    Hooks observe existing boundaries; they do not replace model operations.
    """
    import numpy as np
    import torch

    started = now()
    captured = {}
    handles = []
    for index, block in enumerate(wrapper.model.h):
        handles.append(block.register_forward_hook(
            lambda module, inputs, result, index=index: captured.__setitem__(
                f"block_{index}", result[0].detach().cpu().numpy().copy())))
        handles.append(block.attn.c_proj.register_forward_pre_hook(
            lambda module, inputs, index=index: captured.__setitem__(
                f"attention_context_{index}", inputs[0].detach().cpu().numpy().copy())))

    def run():
        captured.clear()
        # Reconstructing the cache on each call prevents one run mutating the
        # next run's inputs. Decode.forward already owns that cache boundary.
        tensors = wrapper(*args)
        values = {name: tensor.detach().cpu().numpy().copy()
                  for name, tensor in zip(outputs, tensors)}
        values.update(captured)
        return values

    def metric(actual, expected):
        a, e = actual.astype("float32"), expected.astype("float32")
        error = np.abs(a - e)
        outside = error > (0.05 + 0.02 * np.abs(e))
        return {"passed": bool(np.isfinite(a).all() and not outside.any()),
                "max_abs_error": float(error.max()),
                "rms_error": float(np.sqrt(np.mean((a - e)**2))),
                "outside_tolerance": int(outside.sum()), "elements": int(error.size)}

    try:
        with np.load(root / "source-case.npz") as data, torch.inference_mode():
            args = tuple(torch.from_numpy(data[name]).cuda() for name in names)
            native = run()
            for name in outputs:
                assert np.array_equal(native[name], data[name]), name
            if backend_probe:
                compare_mask_backends(wrapper, root, run, native, metric)
                return
            wrapper.explicit = True
            explicit = run()
        comparisons = {name: metric(explicit[name], value) for name, value in native.items()}
        write_json(root / "mask-comparison.json", {
            "started_at": started, "finished_at": now(), "torch": torch.__version__,
            "native_bitwise_matches_retained_case": True,
            "passed": all(value["passed"] for value in comparisons.values()),
            "top_token_matches": bool(np.array_equal(explicit["logits"].argmax(-1),
                                                       native["logits"].argmax(-1))),
            "comparisons": comparisons,
            "scope": "Retained 128-token prompt; native versus explicit-mask PyTorch only"})
        # Retain both sides for a subsequent arithmetic comparison without
        # repeating the trace. No engine or production settings are changed.
        np.savez(root / "mask-outputs.npz", **native,
                 **{"masked_" + name: value for name, value in explicit.items()})
    finally:
        wrapper.explicit = False
        for handle in handles:
            handle.remove()


def compare_mask_backends(wrapper, root, run, native, metric):
    """Distinguish mask semantics from kernel dispatch, with identical Q/K/V.

    This is an isolated diagnostic: restore the functional entry and backend
    flags even on failure. Profile only one CPU call per variant, without CUDA
    activities, allocation stacks or a retained full-model timeline.
    """
    import numpy as np
    import torch
    from torch.nn import functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    started = now()
    original = functional.scaled_dot_product_attention
    first = {}
    selected = {}
    label = None

    def observe(query, key, value, *args, **kwargs):
        # Retain only the first layer, avoiding 24 layers of duplicate tensors.
        if label not in first:
            first[label] = ((query.detach().clone(), key.detach().clone(), value.detach().clone()),
                            args, dict(kwargs))
            # Python dispatch observes only the shared SDPA entry. CPU operator
            # profiling also exposes its selected fused backend underneath it.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                result = original(query, key, value, *args, **kwargs)
            selected[label] = sorted({event.key for event in profile.key_averages()
                                      if "attention" in event.key})
            return result
        return original(query, key, value, *args, **kwargs)

    def full_case(case, explicit, backend=None):
        nonlocal label
        label = case
        wrapper.explicit = explicit
        from contextlib import nullcontext
        with (sdpa_kernel(backend) if backend is not None else nullcontext()):
            return run()

    def compare(actual, expected):
        result = {name: metric(actual[name], value) for name, value in expected.items()}
        return {"passed": all(x["passed"] for x in result.values()),
                "bitwise_equal": all(np.array_equal(actual[k], v) for k, v in expected.items()),
                "comparisons": result}

    report = {"started_at": started, "native_bitwise_matches_retained_case": True}
    try:
        functional.scaled_dot_product_attention = observe
        auto_native = full_case("auto_native", False)
        assert all(np.array_equal(auto_native[k], v) for k, v in native.items())
        auto_mask = full_case("auto_mask", True)
        report["automatic"] = compare(auto_mask, auto_native)
        # Both sides use exactly the same backend. A difference here would
        # remain attributable to masking or mask-dependent arithmetic.
        math_native = full_case("math_native", False, SDPBackend.MATH)
        math_mask = full_case("math_mask", True, SDPBackend.MATH)
        report["forced_math"] = compare(math_mask, math_native)
        report["math_native_vs_original"] = compare(math_native, auto_native)
        report["math_mask_vs_automatic_mask"] = compare(math_mask, auto_mask)
        report["dispatch_operations"] = selected
        qkv, args, kwargs = first["auto_native"]
        masked_qkv, mask_args, mask_kwargs = first["auto_mask"]
        assert all(torch.equal(a, b) for a, b in zip(qkv, masked_qkv))
        report["first_layer_qkv_identical"] = True

        def describe(positional, named):
            mask = named.get("attn_mask", positional[0] if positional else None)
            causal = named.get("is_causal", positional[2] if len(positional) > 2 else False)
            return {"is_causal": bool(causal), "mask_shape": list(mask.shape) if mask is not None else None,
                    "mask_values": mask.unique().cpu().tolist() if mask is not None else None}

        report["first_layer_calls"] = {"native": describe(args, kwargs),
                                       "explicit": describe(mask_args, mask_kwargs)}
        # Independent float64 CPU arithmetic tests the actual first-layer
        # masking effect, rather than assuming -65504 and -inf are equivalent.
        q, k, v = (x.cpu().double() for x in qkv)
        scale = kwargs.get("scale")
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        scores = (q @ k.transpose(-2, -1)) * scale
        allowed = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
        reference = torch.softmax(scores.masked_fill(~allowed, float("-inf")), -1) @ v
        mask = mask_kwargs.get("attn_mask", mask_args[0] if mask_args else None)
        assert mask is not None and not args and kwargs.get("is_causal")
        assert torch.equal(mask.cpu()[0, 0] == 0, allowed)
        probabilities = torch.softmax(scores + mask.cpu().double(), -1)
        masked_reference = probabilities @ v
        report["float64_mask_semantics"] = {
            "identical_allowed_keys": True,
            "bitwise_equal_outputs": bool(torch.equal(reference, masked_reference)),
            "max_abs_error": float((reference - masked_reference).abs().max()),
            "largest_blocked_probability": float(probabilities[..., ~allowed].max())}
        # Replay supported kernels on fixed native Q/K/V. Unsupported kernels
        # are reported explicitly; they cannot silently fall back to another.
        replay = {}
        for backend in (SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
            for kind, positional, named in (("causal", args, kwargs), ("mask", mask_args, mask_kwargs)):
                try:
                    with sdpa_kernel(backend):
                        value = original(*qkv, *positional, **named)
                    error = (value.cpu().double() - reference).abs()
                    replay[f"{backend.name}_{kind}"] = {
                        "max_abs_error_vs_float64": float(error.max()),
                        "rms_error_vs_float64": float(error.square().mean().sqrt())}
                except RuntimeError as error:
                    replay[f"{backend.name}_{kind}"] = {"unsupported": str(error)}
        report["identical_input_replay"] = replay
        report["torch"] = torch.__version__
        report["finished_at"] = now()
        write_json(root / "mask-backends.json", report)
    finally:
        functional.scaled_dot_product_attention = original
        wrapper.explicit = False


def probe(root, replay=False, mask=False, backend_probe=False, attention=False):
    """Run one bounded first-layer diagnosis; reuse image, source and guards."""
    root.mkdir(parents=True, exist_ok=False)
    original = root.parent / "tensorrt-unified-native-reference"
    (root / "source-case.npz").symlink_to(original / "case-0-128.npz")
    docker = ["sudo", "-n", "docker"]
    deployed = json.loads(subprocess.check_output(docker + ["inspect", "chatterbox-tts"], text=True))[0]
    assert not deployed["State"]["Running"]
    assert subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 1
    assert available() > 3 * 1024**3
    previous = json.loads((original / "result.json").read_text())
    argv = previous["phases"][0]["argv"][:]
    name = "chatterbox-trt-probe-" + str(os.getpid())
    argv[argv.index("--name") + 1] = name
    argv = [f"{root}:/trial" if a.endswith(":/trial") else
            f"{Path(__file__).resolve()}:/tensorrt-trial.py:ro" if a.endswith(":/tensorrt-trial.py:ro") else
            ("attention_export" if attention else "mask_backends_export" if backend_probe else "mask_export" if mask else "replay_export" if replay else "probe_export")
            if a == "export-unified" else a for a in argv]
    # Source evidence is mounted separately, read-only, for the retained inputs.
    argv[argv.index("--entrypoint"):argv.index("--entrypoint")] = ["-v", f"{original}:{original}:ro"]
    report = {"started_at": now(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        try:
            report["export"] = guarded(argv, root / "export-command")
            assert report["export"]["returncode"] == 0 and report["export"]["reason"] is None
        finally:
            subprocess.run(docker + ["stop", "-t", "5", name], capture_output=True, timeout=20)
            state = json.loads(subprocess.check_output(docker + ["inspect", name], text=True))[0]
            report["container_stopped"] = not state["State"]["Running"]
            assert report["container_stopped"]
        if mask:
            return  # This comparison requires no engine compilation/inference.
        argv = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={root}/probe.onnx",
                f"--saveEngine={root}/probe.engine", "--fp16", "--noTF32",
                "--memPoolSize=workspace:256", "--builderOptimizationLevel=2",
                "--maxAuxStreams=0", "--skipInference"]
        report["build"] = guarded(argv, root / "build-command")
        assert report["build"]["returncode"] == 0 and report["build"]["reason"] is None
        report["infer"] = guarded(["python3", str(Path(__file__).resolve()), "probe_infer",
                                   "--output-dir", str(root)], root / "infer-command")
        assert report["infer"]["returncode"] == 0 and report["infer"]["reason"] is None
    finally:
        report["finished_at"] = now()
        write_json(root / "probe-result.json", report)


def replay_precision(root):
    """Change only projection arithmetic in the already-exported replay graph."""
    source = root.parent / "tensorrt-first-layer-replay"
    root.mkdir(parents=True, exist_ok=False)
    (root / "probe.npz").symlink_to(source / "probe.npz")
    # HF's Gemm imports as a matrix multiply plus a bias addition. Keep both
    # arithmetic stages in FP32; graph boundary tensors and weights stay FP16.
    layers = [f"/operations.{i}/Gemm*" for i in (1, 2, 4, 6)]
    constraints = ",".join(name + ":fp32" for name in layers)
    argv = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={source}/probe.onnx",
            f"--saveEngine={root}/probe.engine", "--fp16", "--noTF32",
            "--memPoolSize=workspace:256", "--builderOptimizationLevel=2",
            "--maxAuxStreams=0", "--skipInference", "--precisionConstraints=obey",
            "--layerPrecisions=" + constraints, "--layerOutputTypes=" + constraints]
    report = {"started_at": now()}
    try:
        report["build"] = guarded(argv, root / "build-command")
        assert report["build"]["returncode"] == 0 and report["build"]["reason"] is None
        report["infer"] = guarded(["python3", str(Path(__file__).resolve()), "probe_infer",
                                   "--output-dir", str(root)], root / "infer-command")
        assert report["infer"]["returncode"] == 0 and report["infer"]["reason"] is None
    finally:
        report["finished_at"] = now()
        write_json(root / "probe-result.json", report)


def precision(root, projections=False):
    """Retest the failed unified gate with explicit FP32 attention arithmetic.

    Reuse the verified model export and native references. Weights, projection
    precision and tolerances are unchanged; only the attention accumulation and
    softmax path receive explicit builder constraints.
    """
    original = root.parent / "tensorrt-unified-native-reference"
    root.mkdir(parents=True, exist_ok=False)
    for source in original.glob("case-*.npz"):
        (root / source.name).symlink_to(source)
    previous = json.loads((original / "result.json").read_text())
    argv = previous["phases"][1]["argv"][:]
    argv = [f"--saveEngine={root}/decode.engine" if a.startswith("--saveEngine=") else a for a in argv]
    layers = [f"/model/h.{i}/attn/{op}" for i in range(24)
              for op in ("Sqrt_1", "Sqrt_2", "Mul", "Mul_1", "MatMul", "Add", "Softmax", "MatMul_1")]
    argv += ["--precisionConstraints=obey", "--layerPrecisions=" + ",".join(f"{n}:fp32" for n in layers),
             "--layerOutputTypes=" + ",".join(f"{n}:fp16" if n.endswith("/MatMul_1") else f"{n}:fp32" for n in layers)]
    if projections:
        # Preserve the prior attention correction and test the independently
        # demonstrated projection error in the original, uninstrumented graph.
        names = [f"/model/h.{i}/{module}/Gemm*" for i in range(24)
                 for module in ("attn/c_attn", "attn/c_proj", "mlp/c_fc", "mlp/c_proj")]
        names += ["/head/MatMul"]
        extra = ",".join(name + ":fp32" for name in names)
        argv[-2] += "," + extra
        argv[-1] += "," + extra
    result = {"started_at": now()}
    try:
        result["build"] = guarded(argv, root / "build-command")
        assert result["build"]["returncode"] == 0 and result["build"]["reason"] is None
        result["infer"] = guarded(["python3", str(Path(__file__).resolve()), "infer", "--output-dir", str(root)],
                                  root / "infer-command")
        assert result["infer"]["returncode"] == 0 and result["infer"]["reason"] is None
    finally:
        result["finished_at"] = now()
        write_json(root / "result.json", result)


REAL_TEXTS = (
    "Ready, Holmes.",
    "Calculating route.",
    "Initialising subspace communications.",
    "All systems are operating within normal parameters. The ship is ready to proceed.",
    "This star system has seventy-five documents relating to The Wind in the Willows.",
    "Seven of Nine is standing by on the bridge.",
    "Watson, check the engineering report before we proceed.",
    "The reference voice remains unchanged.",
    "Please turn on the kitchen lights.",
    "The bedroom panel will remain dark until you touch it.",
    "There are three messages awaiting your attention.",
    "Our estimated arrival is nine thirty tomorrow morning.",
    "Do you wish to continue with the current heading?",
    "The temperature is eighteen degrees Celsius.",
    "Borg is ready to generate the next passage.",
    "The Doctor has received your health report.",
    "We have twenty-four samples, three inference paths, and one reference voice.",
    "The ship turns smoothly until the destination reaches the centre of the grid.",
    "Keep the original recording available for comparison, and retain the prepared voice between requests.",
    "A quiet river wound between the willows. The evening light rested on the water, and a small boat drifted towards the bank.",
    "The engineering comparison examines pronunciation, speaker identity, termination, and generation time. We will assess the evidence before changing the speech route.",
    "When the observer changes direction, the distant stars remain steady. Nearby planets reveal their movement through parallax, while the background remains beyond reach.",
    "We are approaching the selected star system. The camera will turn from its present orientation, accelerate gently, then slow as the target reaches the central square. No journey begins until that alignment is established.",
    "Please read this longer report carefully. The reference recording was prepared once and retained throughout the comparison. Each implementation receives the same words and voice conditioning. The transformer predicts speech tokens, and the separate audio generator turns those tokens into a waveform. We must listen for missing words, repeated phrases, unnatural pauses, and changes in the speaker's identity before deciding whether the accelerated path is ready.",
)


def real_audio_model():
    """Avoid eager-only 8196-square masks during the diagnostic cold load.

    The deployed SDPA implementation does not read these buffers (the existing
    export already uses one-element placeholders). Release each buffer as its
    attention module is constructed, rather than accumulating 24 of them before
    loading the voice/audio components. The replacement is process-local only.
    """
    import torch
    sys.path.insert(0, "/app")
    import engine
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2PreTrainedModel
    original = GPT2Attention.__init__
    original_init_weights = GPT2PreTrainedModel._init_weights
    def initialise(module, *args, **kwargs):
        original(module, *args, **kwargs)
        module.bias = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    GPT2Attention.__init__ = initialise
    def initialise_weights(owner, module):
        # Transformers 5.1 fills the causal buffer again in post_init. That
        # attention-only branch initialises no parameters; all projection and
        # norm modules must retain their original initialisation/loading paths.
        if not isinstance(module, GPT2Attention):
            original_init_weights(owner, module)
    GPT2PreTrainedModel._init_weights = initialise_weights
    try:
        assert engine.load_model() and engine.loaded_model_type == "turbo"
    finally:
        GPT2Attention.__init__ = original
        GPT2PreTrainedModel._init_weights = original_init_weights
    model = engine.chatterbox_model
    assert model.t3.tfmr.config._attn_implementation == "sdpa", "Small mask is valid only for SDPA"
    return model


def real_capture(root):
    """Prepare one governed voice, then capture the actual model's input boundary.

    Conditioning is laid down as trial runtime state so the later audio phase
    reuses it rather than re-encoding the reference after every precision run.
    """
    import numpy as np
    import torch
    from chatterbox.tts_turbo import punc_norm
    torch.set_num_threads(4)
    model = real_audio_model()
    assert next(model.s3gen.parameters()).dtype == torch.float32
    voice = Path("/app/voices/tpol_amplified_clean.wav")
    report = {"started_at": now(), "voice_sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
              "reference_preparations": 1, "cases": [], "seed": 42,
              "bos": model.t3.hp.start_speech_token, "eos": model.t3.hp.stop_speech_token}
    with torch.inference_mode():
        model.prepare_conditionals(str(voice), exaggeration=0.0, norm_loudness=True)
        prepared = model.conds
        for index, text in enumerate(REAL_TEXTS):
            tokens = model.tokenizer(punc_norm(text), return_tensors="pt", padding=True,
                                     truncation=False).input_ids.cuda()
            bos = torch.full_like(tokens[:, :1], report["bos"])
            embeds, _ = model.t3.prepare_input_embeds(
                t3_cond=prepared.t3, text_tokens=tokens, speech_tokens=bos, cfg_weight=0.0)
            assert embeds.shape[1] <= 512, "Real prompt exceeds retained engine profile"
            np.save(root / f"prompt-{index}.npy", embeds.cpu().numpy())
            if index in (0, len(REAL_TEXTS) - 1):
                # Anchor the reconstructed comparator to actual deployed model
                # calls, including a real one-token cached decode. Argmax here
                # selects an identical replay input, not the audio sampling rule.
                actual = model.t3.tfmr(inputs_embeds=embeds, use_cache=True)
                logits = model.t3.speech_head(actual.last_hidden_state[:, -1:])[:, -1]
                token = logits.argmax(-1).view(1, 1)
                following = model.t3.tfmr(inputs_embeds=model.t3.speech_emb(token),
                    past_key_values=actual.past_key_values, use_cache=True)
                np.savez(root / f"deployed-anchor-{index}.npz", prompt_logits=logits.cpu().numpy(),
                    token=token.cpu().numpy(), decode_logits=model.t3.speech_head(
                        following.last_hidden_state)[:, -1].cpu().numpy())
                del actual, following, logits, token
            report["cases"].append({"index": index, "text": text,
                                     "prompt_length": embeds.shape[1]})
        np.save(root / "speech-embedding.npy", model.t3.speech_emb.weight.detach().cpu().numpy())
        assert model.conds is prepared
        # Move the already-prepared object, without invoking reference encoding.
        prepared.to("cpu").save(root / "conditionals.pt")
    report.update(finished_at=now(), conditioning_reused=True,
                  audio_generator_dtype=str(next(model.s3gen.parameters()).dtype))
    write_json(root / "capture.json", report)


def real_native_model(dtype, observer=None):
    """Load only the existing transformer/head, avoiding simultaneous FP32 S3Gen.

    FP32 starts from the checkpoint, never from weights already rounded to FP16.
    Both paths consume the same captured FP16 input embeddings, upcast if needed.
    """
    import torch
    from safetensors import safe_open
    from transformers import GPT2Config, GPT2Model
    from chatterbox.models.t3.llama_configs import GPT2_MEDIUM_CONFIG
    config = GPT2Config(**GPT2_MEDIUM_CONFIG)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = GPT2Model(config)
        head = torch.nn.Linear(config.n_embd, 6563)
    del model.wte
    checkpoints = list(Path("/app/hf_cache").rglob("t3_turbo_v1.safetensors"))
    assert len(checkpoints) == 1
    with safe_open(checkpoints[0], framework="pt", device="cpu") as checkpoint:
        weights = {key.removeprefix("tfmr."): checkpoint.get_tensor(key).to(dtype)
                   for key in checkpoint.keys()
                   if key.startswith("tfmr.") and not key.startswith("tfmr.wte.")}
        head.load_state_dict({key.removeprefix("speech_head."): checkpoint.get_tensor(key).to(dtype)
                              for key in checkpoint.keys() if key.startswith("speech_head.")}, assign=True)
    model.load_state_dict(weights, assign=True)
    del weights
    for block in model.h:
        block.attn.bias = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    model, head = model.eval().cuda(), head.eval().cuda()

    if observer is not None:
        # Observe the actual Conv1D inputs; GPT2 stores these weights in input /
        # output order, unlike Linear. Hooks belong only to this trial model.
        for name, module in model.named_modules():
            if name.startswith("h.") and name.endswith(
                    ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")):
                module.register_forward_pre_hook(
                    lambda owner, args, name=name: observer(name, args[0], owner))

    def step(embeds, past):
        from transformers.cache_utils import DynamicCache
        cache = None if past is None else DynamicCache(tuple(zip(past[::2], past[1::2])))
        output = model(inputs_embeds=embeds.to(dtype), past_key_values=cache,
                       use_cache=True, return_dict=True)
        return head(output.last_hidden_state[:, -1:])[:, -1], tuple(
            tensor for layer in output.past_key_values for tensor in layer[:2])
    return step


def real_trt_model(root):
    """Bind the unchanged uninstrumented engine to resident PyTorch CUDA tensors.

    Host TensorRT libraries are read-only trial mounts; the deployed image and
    its CUDA 12 compatibility libraries remain unchanged.
    """
    import torch
    sys.path.insert(0, "/trt-python")
    for path in ("/host-cuda/libcudart.so.13", "/host-trt/libnvinfer.so.10",
                 "/host-trt/libnvonnxparser.so.10", "/host-trt/libnvinfer_plugin.so.10"):
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    import tensorrt_bindings as trt
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine((root / "decode.engine").read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    dtypes = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32,
              trt.DataType.INT64: torch.int64, trt.DataType.INT32: torch.int32}
    names = [f"past_{kind}_{i}" for i in range(24) for kind in ("k", "v")]

    def step(embeds, past):
        # Keep runtime/engine alive for the context's entire lifetime.
        assert runtime is not None and engine is not None
        length = 0 if past is None else past[0].shape[2]
        assert length <= 1024 and embeds.shape[1] <= 512
        if past is None:
            past = tuple(torch.empty(1, 16, 0, 64, device="cuda", dtype=torch.float16)
                         for _ in names)
        values = {"embeds": embeds.contiguous(), "position": torch.arange(
            length, length + embeds.shape[1], device="cuda", dtype=torch.int64),
            **dict(zip(names, past))}
        # Resolve every dynamic input before allocating any output.
        for name, tensor in values.items():
            assert context.set_input_shape(name, tuple(tensor.shape)), name
        assert not context.infer_shapes()
        for name in engine:
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(context.get_tensor_shape(name))
                assert all(size >= 0 for size in shape), (name, shape)
                values[name] = torch.empty(shape, device="cuda", dtype=dtypes[engine.get_tensor_dtype(name)])
        # Empty K/V still need a valid address; own their tiny placeholders until
        # execution finishes. Actual zero-length shapes are preserved above.
        empty = []
        for name, tensor in values.items():
            if tensor.numel() == 0:
                empty.append(torch.empty(1, device="cuda", dtype=tensor.dtype))
                address = empty[-1].data_ptr()
            else:
                address = tensor.data_ptr()
            assert context.set_tensor_address(name, address), name
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return values["logits"][:, -1], tuple(
            values[name.replace("past_", "present_")] for name in names)
    return step


def real_processors():
    """Use Turbo's installed processor order and native tensor precision."""
    from transformers.generation.logits_process import (
        LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper,
        TopPLogitsWarper, RepetitionPenaltyLogitsProcessor)
    return LogitsProcessorList([TemperatureLogitsWarper(0.8), TopKLogitsWarper(1000),
                               TopPLogitsWarper(0.95), RepetitionPenaltyLogitsProcessor(1.2)])


def distribution_metrics(reference, actual):
    """Finite JS/TV plus honest KL, including disjoint post-filter supports."""
    import numpy as np
    p, q = (np.asarray(value, dtype=np.float64) for value in (reference, actual))
    assert np.isfinite(p).all() and np.isfinite(q).all()
    assert np.all(p >= 0) and np.all(q >= 0) and p.sum() > 0 and q.sum() > 0
    p, q = p / p.sum(), q / q.sum()
    middle = (p + q) / 2
    def kl(left, right):
        selected = left > 0
        if np.any(right[selected] == 0):
            return None  # JSON null plus explicit flag; never epsilon-clipped.
        return float(np.sum(left[selected] * np.log(left[selected] / right[selected])))
    raw = kl(p, q)
    return {"kl": raw, "kl_infinite": raw is None,
            "js": (kl(p, middle) + kl(q, middle)) / 2,
            "total_variation": float(np.abs(p - q).sum() / 2)}


def real_tokens(root, variant, case_index=None, case_indices=None):
    """Free synthesis and teacher-forced replay are separate measurements.

    Replay uses the native FP16 token history with each path's own evolving KV,
    preserving accumulated numerical drift instead of injecting reference KV.
    """
    import gc
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    capture = json.loads((root / "capture.json").read_text())
    embedding = torch.from_numpy(np.load(root / "speech-embedding.npy")).cuda()
    step = real_trt_model(root) if variant == "trt" else real_native_model(
        torch.float32 if variant == "fp32" else torch.float16)
    processors = real_processors()
    report = {"started_at": now(), "variant": variant, "cases": [],
              "sampling": {"temperature": 0.8, "top_k": 1000, "top_p": 0.95, "repetition_penalty": 1.2},
              "cache_prefix_check": "first decode and every 128 positions",
              "deployed_reference_identity": [],
              "timing_scope": "transformer, sampling, synchronisation; excludes model loading/audio rendering"}
    if case_index is not None:
        old_path = root / f"{variant}-tokens.json"
        previous = json.loads(old_path.read_text())
        assert len(previous["cases"]) == len(REAL_TEXTS)
        backup = root / f"{variant}-tokens-before-case-{case_index}.json"
        assert not backup.exists(), "Targeted repeat requires a new diagnosis"
        write_json(backup, previous)
        report["previous_report"] = backup.name
        report["cases"] = [case for case in previous["cases"] if case["index"] != case_index]
        report["deployed_reference_identity"] = [item for item in previous["deployed_reference_identity"]
                                                  if item["index"] != case_index]
        original = root / f"{variant}-{case_index}.npz"
        original.rename(root / f"{variant}-{case_index}-cap512.npz")
    try:
        with torch.inference_mode():
            for case in capture["cases"]:
                index = case["index"]
                if case_indices is not None and index not in case_indices:
                    continue
                if case_index is not None and index != case_index:
                    continue
                baseline = None if variant == "fp16" else np.load(root / f"fp16-{index}.npz")
                row = {"index": index, "text": case["text"], "runs": {}}
                for kind in (("free",) if baseline is None else ("free", "replay")):
                    torch.manual_seed(42)
                    embeds = torch.from_numpy(np.load(root / f"prompt-{index}.npy")).cuda()
                    if variant == "fp16" and index in (0, len(REAL_TEXTS) - 1):
                        anchor = np.load(root / f"deployed-anchor-{index}.npz")
                        anchored_logits, anchored_past = step(embeds, None)
                        assert np.array_equal(anchored_logits.cpu().numpy(), anchor["prompt_logits"]), "Deployed prompt identity failed"
                        next_embed = embedding[int(anchor["token"].item())].view(1, 1, -1)
                        anchored_logits, anchored_past = step(next_embed, anchored_past)
                        assert np.array_equal(anchored_logits.cpu().numpy(), anchor["decode_logits"]), "Deployed decode identity failed"
                        report["deployed_reference_identity"].append({"index": index, "prompt": True, "cached_decode": True})
                        del anchored_logits, anchored_past, next_embed
                    past, tokens, logits_trace, differences, times = None, [], [], [], []
                    started = time.perf_counter()
                    # A real long passage can exceed 512 speech tokens. The
                    # existing engine permits past length 1024; cap at that
                    # profile boundary, never at an arbitrary shorter duration.
                    maximum = 1024 - embeds.shape[1]
                    stopped = False
                    for position in range(maximum if kind == "free" else len(baseline["tokens"])):
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        logits, present = step(embeds, past)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - tick)
                        assert torch.isfinite(logits).all(), "Nonfinite pre-filter logits"
                        length = 0 if past is None else past[0].shape[2]
                        assert all(t.shape[2] == length + embeds.shape[1] for t in present), "Invalid KV shape"
                        # One host synchronisation rather than 48; check newly
                        # appended values because retained prefixes were checked
                        # when created and are required to remain unchanged.
                        assert torch.stack([torch.isfinite(t[:, :, length:]).all()
                                            for t in present]).all(), "Nonfinite appended KV"
                        if past is not None and (position == 1 or position % 128 == 0):
                            assert all(torch.equal(t[:, :, :length], p) for t, p in zip(present, past)), "KV prefix mutation"
                        history = torch.tensor([tokens or [capture["bos"]]], device="cuda", dtype=torch.long)
                        probabilities = torch.softmax(processors(history, logits.clone()), dim=-1)
                        assert torch.isfinite(probabilities).all() and probabilities.sum() > 0
                        if kind == "replay":
                            expected = torch.from_numpy(baseline["logits"][position:position + 1]).cuda()
                            expected_prob = torch.softmax(processors(history, expected.clone()), dim=-1)
                            metrics = distribution_metrics(expected_prob.cpu().numpy(), probabilities.cpu().numpy())
                            metrics["raw"] = distribution_metrics(
                                torch.softmax(expected.double(), -1).cpu().numpy(),
                                torch.softmax(logits.double(), -1).cpu().numpy())
                            differences.append(metrics)
                            token = int(baseline["tokens"][position])
                        else:
                            logits_trace.append(logits.cpu().numpy()[0])
                            token = int(torch.multinomial(probabilities, 1).item())
                        tokens.append(token)
                        # Turbo checks EOS in its repeated-decode loop, after
                        # the first sampled token. Preserve that exact boundary.
                        if position > 0 and token == capture["eos"]:
                            stopped = True
                            break
                        embeds = embedding[token].view(1, 1, -1)
                        past = present
                    torch.cuda.synchronize()
                    row["runs"][kind] = {"generation_seconds": time.perf_counter() - started,
                        "step_seconds": times, "tokens": len(tokens), "natural_eos": stopped,
                        # The final sampled token (usually EOS) is not fed back:
                        # cache length is P + T - 1, last processed index P + T - 2.
                        "maximum_position": case["prompt_length"] + len(tokens) - 2,
                        "cache_length": case["prompt_length"] + len(tokens) - 1,
                        "repeated_token_run": max((sum(1 for _ in group) for _, group in
                            __import__("itertools").groupby(tokens)), default=0),
                        "distributions": differences}
                    if kind == "free":
                        np.savez(root / f"{variant}-{index}.npz", tokens=np.array(tokens, dtype=np.int64),
                                 logits=np.array(logits_trace))
                    del past, present, embeds, logits
                    gc.collect()
                    torch.cuda.empty_cache()
                report["cases"].append(row)
                report["cases"].sort(key=lambda case: case["index"])
                write_json(root / f"{variant}-tokens.json", report)
                print(json.dumps({"variant": variant, "case": index, "free": row["runs"]["free"]["generation_seconds"]}), flush=True)
    finally:
        report["finished_at"] = now()
        write_json(root / f"{variant}-tokens.json", report)


def real_render(root, case_index=None, case_indices=None, variants=("fp16", "fp32", "trt")):
    """Render every token stream through one unchanged FP32 audio generator."""
    import numpy as np
    import torch
    import soundfile as sf
    sys.path.insert(0, "/app")
    from chatterbox.tts_turbo import Conditionals
    from chatterbox.models.s3gen.const import S3GEN_SIL
    import librosa
    model = real_audio_model()
    # The transformer is not needed for rendering. Reclaim it before loading
    # retained conditioning so unified-memory headroom stays bounded.
    del model.t3
    # The deployed cache also owns the conditionals, but does not own T3.
    __import__("gc").collect()
    torch.cuda.empty_cache()
    model.conds = Conditionals.load(root / "conditionals.pt", map_location="cpu").to("cuda")
    prepared = model.conds
    assert next(model.s3gen.parameters()).dtype == torch.float32
    capture = json.loads((root / "capture.json").read_text())
    reference, _ = librosa.load("/app/voices/tpol_amplified_clean.wav", sr=16000)
    with torch.inference_mode():
        voice = model.ve.embeds_from_wavs([reference], sample_rate=16000)[0]
        report = {"started_at": now(), "reference_reprepared": False, "cases": []}
        if case_index is not None:
            previous = json.loads((root / "audio.json").read_text())
            assert len(previous["cases"]) == 3 * len(REAL_TEXTS)
            backup = root / f"audio-before-case-{case_index}.json"
            assert not backup.exists()
            write_json(backup, previous)
            report["previous_report"] = backup.name
            report["cases"] = [case for case in previous["cases"] if case["index"] != case_index]
        try:
            # A discarded warmup excludes first-use initialisation from timings.
            sample = np.load(root / "fp16-0.npz")["tokens"]
            sample = torch.tensor(sample[sample < 6561], device="cuda")
            model.s3gen.inference(speech_tokens=torch.cat([sample, torch.full((3,), S3GEN_SIL, device="cuda")]),
                                  ref_dict=prepared.gen, n_cfm_timesteps=2)
            for case in capture["cases"]:
                if case_indices is not None and case["index"] not in case_indices:
                    continue
                if case_index is not None and case["index"] != case_index:
                    continue
                for variant in variants:
                    tokens = np.load(root / f"{variant}-{case['index']}.npz")["tokens"]
                    tokens = torch.tensor(tokens[tokens < 6561], device="cuda")
                    tokens = torch.cat([tokens, torch.full((3,), S3GEN_SIL, device="cuda")])
                    torch.manual_seed(42)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    waveform, _ = model.s3gen.inference(speech_tokens=tokens, ref_dict=prepared.gen, n_cfm_timesteps=2)
                    waveform = waveform.squeeze(0).detach().cpu().numpy()
                    waveform = model.watermarker.apply_watermark(waveform, sample_rate=model.sr)
                    elapsed = time.perf_counter() - started
                    assert waveform.size and np.isfinite(waveform).all()
                    path = root / f"{variant}-{case['index']}.wav"
                    if case_index is not None:
                        path.rename(root / f"{variant}-{case_index}-cap512.wav")
                    sf.write(path, waveform, model.sr, subtype="PCM_16")
                    # Speaker identity uses the same installed voice encoder and
                    # resampling for every variant, without a new model service.
                    resampled = librosa.resample(waveform, orig_sr=model.sr, target_sr=16000)
                    observed = model.ve.embeds_from_wavs([resampled], sample_rate=16000)[0]
                    similarity = float(np.dot(voice, observed) / (np.linalg.norm(voice) * np.linalg.norm(observed)))
                    report["cases"].append({"index": case["index"], "variant": variant,
                        "render_seconds": elapsed, "audio_seconds": len(waveform) / model.sr,
                        "speaker_cosine": similarity, "wav": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                    report["cases"].sort(key=lambda item: (item["index"], item["variant"]))
                    assert model.conds is prepared
                    write_json(root / "audio.json", report)
        finally:
            report["finished_at"] = now()
            write_json(root / "audio.json", report)


def word_errors(expected, observed):
    """Count transcription edits; missing ASR must fail before this function."""
    import re
    reference = re.findall(r"[a-z0-9]+", expected.lower())
    hypothesis = re.findall(r"[a-z0-9]+", observed.lower())
    assert reference
    previous = list(range(len(hypothesis) + 1))
    for index, word in enumerate(reference, 1):
        current = [index]
        for column, other in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (word != other)))
        previous = current
    return previous[-1], len(reference)


def paired_bound(differences, denominators=None):
    """One-sided paired 95% bound with the predeclared seed/resample count."""
    import numpy as np
    values = np.asarray(differences, dtype=float)
    assert len(values) == len(REAL_TEXTS) and np.isfinite(values).all()
    samples = np.random.default_rng(20261029).integers(0, len(values), (10000, len(values)))
    if denominators is None:
        estimates = values[samples].mean(axis=1)
        point = values.mean()
    else:
        words = np.asarray(denominators, dtype=float)
        estimates = values[samples].sum(axis=1) / words[samples].sum(axis=1)
        point = values.sum() / words.sum()
    return {"difference": float(point), "upper_95": float(np.quantile(estimates, 0.95))}


def real_analyse(root):
    """Use existing local Whisper; preserve every transcript and predeclared gate.

    The sequential diagnostic's generation/render sums are waveform-ready cost
    estimates. They are not a measured HTTP request or browser first-audible time.
    """
    import numpy as np
    capture = json.loads((root / "capture.json").read_text())
    audio = json.loads((root / "audio.json").read_text())
    reports = {variant: json.loads((root / f"{variant}-tokens.json").read_text())
               for variant in ("fp16", "fp32", "trt")}
    assert len(audio["cases"]) == 3 * len(REAL_TEXTS)
    transcripts = {"started_at": now(), "cases": []}
    try:
        for case in capture["cases"]:
            for variant in reports:
                path = root / f"{variant}-{case['index']}.wav"
                destination = root / f"{variant}-{case['index']}-asr.json"
                if destination.exists():
                    receipt = json.loads(destination.read_text())
                else:
                    output = subprocess.check_output(["curl", "--fail-with-body", "--silent",
                        "--show-error", "--max-time", "120", "-F", f"file=@{path}",
                        "-F", "language=en", "-F", "temperature=0", "-F", "response_format=json",
                        "http://127.0.0.1:8005/v1/audio/transcriptions"], timeout=125, text=True)
                    receipt = json.loads(output)
                    assert isinstance(receipt.get("text"), str) and receipt["text"].strip(), "Missing transcription"
                    write_json(destination, receipt)
                errors, words = word_errors(case["text"], receipt["text"])
                transcripts["cases"].append({"index": case["index"], "variant": variant,
                    "text": receipt["text"], "errors": errors, "words": words})
                write_json(root / "transcriptions.json", transcripts)
    finally:
        transcripts["finished_at"] = now()
        write_json(root / "transcriptions.json", transcripts)

    def select(rows, variant):
        selected = sorted((row for row in rows if row["variant"] == variant), key=lambda row: row["index"])
        assert len(selected) == len(REAL_TEXTS) and [row["index"] for row in selected] == list(range(len(REAL_TEXTS)))
        return selected

    base = select(transcripts["cases"], "fp16")
    base_voice = select(audio["cases"], "fp16")
    summary = {"started_at": now(), "variants": {}, "production_admission": False,
               "listening": "not verified; requires Holmes's paired-sample assessment",
               "first_audible": "not verified", "waveform_ready_latency": "not measured end to end",
               "reference_preparations": capture["reference_preparations"]}
    baseline_steps = [time for case in reports["fp16"]["cases"] for time in case["runs"]["free"]["step_seconds"][1:]]
    for variant in reports:
        speech = select(transcripts["cases"], variant)
        voices = select(audio["cases"], variant)
        assert len(reports[variant]["cases"]) == len(REAL_TEXTS)
        error_delta = [row["errors"] - reference["errors"] for row, reference in zip(speech, base)]
        voice_delta = [reference["speaker_cosine"] - row["speaker_cosine"] for row, reference in zip(voices, base_voice)]
        wer = paired_bound(error_delta, [row["words"] for row in base])
        similarity = paired_bound(voice_delta)
        free = [case["runs"]["free"] for case in reports[variant]["cases"]]
        steps = [time for case in free for time in case["step_seconds"][1:]]
        estimates = [case["generation_seconds"] + voice["render_seconds"] for case, voice in zip(free, voices)]
        metrics = [value for case in reports[variant]["cases"]
                   for value in case["runs"].get("replay", {}).get("distributions", [])]
        row = {"wer": sum(x["errors"] for x in speech) / sum(x["words"] for x in speech),
            "wer_delta": wer, "speaker_mean": float(np.mean([x["speaker_cosine"] for x in voices])),
            "speaker_decline": similarity, "worst_speaker_decline": max(voice_delta),
            "all_natural_eos": all(x["natural_eos"] for x in free),
            "maximum_position": max(case["prompt_length"] + run["tokens"] - 2
                                    for case, run in zip(capture["cases"], free)),
            "longest_identical_token_run": max(x["repeated_token_run"] for x in free),
            "median_decode_step_seconds": float(np.median(steps)),
            "median_decode_speedup": float(np.median(baseline_steps) / np.median(steps)),
            "median_transformer_seconds": float(np.median([sum(x["step_seconds"]) for x in free])),
            "median_diagnostic_generation_seconds": float(np.median([x["generation_seconds"] for x in free])),
            "median_sum_generation_render_seconds": float(np.median(estimates)),
            "token_repetition_listening": "not verified; identical-token runs cannot exclude phrase loops"}
        if metrics:
            row["replay"] = {"positions": len(metrics), "mean_js": float(np.mean([x["js"] for x in metrics])),
                "p99_js": float(np.quantile([x["js"] for x in metrics], 0.99)),
                "mean_tv": float(np.mean([x["total_variation"] for x in metrics])),
                "processed_kl_infinite_positions": sum(x["kl_infinite"] for x in metrics),
                "mean_raw_kl": float(np.mean([x["raw"]["kl"] for x in metrics]))}
            row["distribution_screen_passed"] = (row["replay"]["mean_js"] <= 0.001 and
                row["replay"]["p99_js"] <= 0.01 and row["replay"]["mean_tv"] <= 0.05)
        row["objective_audio_gate_passed"] = (wer["upper_95"] <= 0.02 and similarity["upper_95"] <= 0.02
            and max(voice_delta) <= 0.05 and row["all_natural_eos"] and row["longest_identical_token_run"] < 20)
        summary["variants"][variant] = row
    base_seconds = summary["variants"]["fp16"]["median_transformer_seconds"]
    for row in summary["variants"].values():
        row["transformer_generation_speedup"] = base_seconds / row["median_transformer_seconds"]
        row["transformer_speed_gate_passed"] = row["transformer_generation_speedup"] >= 2.0
        row["waveform_latency_gate"] = "not verified; sum of isolated costs is not an end-to-end measurement"
    summary["finished_at"] = now()
    write_json(root / "assessment.json", summary)
    print(json.dumps(summary, indent=2))


def emergency_projection(weights, maxima, transposed=False):
    """Smooth input-channel outliers, then quantize each output channel to INT8.

    A/S @ (S*W) preserves the original projection before rounding. Scales are
    rounded to their actual ONNX FP16 representation before integer rounding.
    Bias, attention arithmetic and the speech head retain their original types.
    """
    import numpy as np
    values = np.asarray(weights, dtype=np.float32)
    if transposed:
        values = values.T
    maxima = np.asarray(maxima, dtype=np.float32)
    if (values.ndim != 2 or maxima.shape != (values.shape[0],)
            or not np.isfinite(values).all() or not np.isfinite(maxima).all()
            or np.any(maxima < 0)):
        raise ValueError("Invalid projection weights or calibration channel ranges")
    weight_max = np.abs(values).max(axis=1)
    smooth = np.clip(np.sqrt(np.maximum(maxima, 1e-5) /
                             np.maximum(weight_max, 1e-5)), 1 / 256, 256)
    inverse = (1 / smooth).astype(np.float16)
    # Use the actual rounded multiplier in both sides of the equivalence.
    smoothed = values / inverse.astype(np.float32)[:, None]
    scales = np.maximum(np.abs(smoothed).max(axis=0) / 127,
                        np.finfo(np.float16).tiny).astype(np.float16)
    quantized = np.clip(np.rint(smoothed / scales.astype(np.float32)),
                        -127, 127).astype(np.int8)
    activation = np.float16(max(float(np.max(maxima * inverse)) / 127,
                               np.finfo(np.float16).tiny))
    return (quantized.T if transposed else quantized), scales, inverse, activation


def emergency_calibrate(root):
    """Reuse the complete retained prompt corpus and its existing voice state."""
    import numpy as np
    import torch
    source = root.parent / "tensorrt-real-input"
    capture = json.loads((source / "capture.json").read_text())
    assert len(capture["cases"]) == len(REAL_TEXTS)
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "calibration.json").exists(), "Calibration already has a result"
    for pattern in ("capture.json", "conditionals.pt", "speech-embedding.npy",
                    "prompt-*.npy", "fp16-*.npz"):
        for path in source.glob(pattern):
            (root / path.name).symlink_to(path)
    maxima = {}

    def observe(name, values, owner):
        extent = values.detach().abs().amax(dim=tuple(range(values.ndim - 1)))
        maxima[name] = extent if name not in maxima else torch.maximum(maxima[name], extent)

    torch.set_num_threads(4)
    step = real_native_model(torch.float16, observer=observe)
    embedding = torch.from_numpy(np.load(source / "speech-embedding.npy")).cuda()
    report = {"started_at": now(), "cases": [], "reference_reprepared": False,
              "method": "per-input-channel max; SmoothQuant alpha=0.5; symmetric W8A8"}
    with torch.inference_mode():
        for case in capture["cases"]:
            index = case["index"]
            embeds = torch.from_numpy(np.load(source / f"prompt-{index}.npy")).cuda()
            _, past = step(embeds, None)
            tokens = np.load(source / f"fp16-{index}.npz")["tokens"][:32]
            for token in tokens:
                _, past = step(embedding[int(token)].view(1, 1, -1), past)
            report["cases"].append({"index": index, "decode_tokens": len(tokens)})
        assert len(maxima) == 96, "Unexpected GPT2 projection ownership"
        report["channel_maxima"] = {name: values.float().cpu().tolist()
                                    for name, values in maxima.items()}
    report["finished_at"] = now()
    write_json(root / "calibration.json", report)


def emergency_quantize(root, activations=True):
    """Quantize weights, optionally activations; never mutate the FP16 source."""
    import re
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, version_converter
    original = root.parent / "tensorrt-unified-native-reference" / "decode.onnx"
    calibration = json.loads((root / "calibration.json").read_text())
    # Opset 19 admits HALF Q/DQ directly. The official converter also adapts
    # changed reduction schemas; merely replacing the opset number is invalid.
    graph = version_converter.convert_version(onnx.load(str(original)), 19)
    initializers = {value.name: value for value in graph.graph.initializer}
    rewritten, added, replaced, selected = [], [], set(), []
    for node in graph.graph.node:
        match = re.fullmatch(r"/model/(h\.\d+)/(attn|mlp)/(c_attn|c_proj|c_fc)/Gemm", node.name)
        if match is None:
            rewritten.append(node)
            continue
        assert node.op_type == "Gemm" and node.input[1] in initializers
        attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        assert not attributes.get("transA", 0)
        assert attributes.get("alpha", 1) == attributes.get("beta", 1) == 1
        transposed = bool(attributes.get("transB", 0))
        key = ".".join(match.groups())
        weights = numpy_helper.to_array(initializers[node.input[1]])
        assert weights.dtype == np.float16
        quantized, scales, inverse, activation = emergency_projection(
            weights, calibration["channel_maxima"][key], transposed)
        prefix = node.name + "/myth-e"
        def add(suffix, value):
            name = prefix + "/" + suffix
            added.append(numpy_helper.from_array(np.asarray(value), name))
            return name
        weight = add("weight", quantized)
        scale = add("weight-scale", scales)
        zero = add("weight-zero", np.zeros(scales.shape, dtype=np.int8))
        multiplier = add("inverse-smooth", inverse)
        smooth, integers, restored, dequantized = [prefix + "/" + suffix
            for suffix in ("smooth", "integers", "restored", "dequantized")]
        rewritten.append(helper.make_node("Mul", [node.input[0], multiplier],
                                          [smooth], name=prefix + "/Mul"))
        if activations:
            input_scale = add("activation-scale", activation)
            input_zero = add("activation-zero", np.array(0, dtype=np.int8))
            rewritten.extend([
                helper.make_node("QuantizeLinear", [smooth, input_scale, input_zero], [integers], name=prefix + "/Q"),
                helper.make_node("DequantizeLinear", [integers, input_scale, input_zero], [restored], name=prefix + "/DQ-input")])
        rewritten.append(helper.make_node("DequantizeLinear", [weight, scale, zero], [dequantized],
                                          axis=0 if transposed else 1, name=prefix + "/DQ-weight"))
        replaced.add(node.input[1])
        node.input[0], node.input[1] = restored if activations else smooth, dequantized
        rewritten.append(node)
        selected.append(node.name)
    assert len(selected) == 96, f"Expected 96 projections, found {len(selected)}"
    remaining_inputs = {name for node in rewritten for name in node.input}
    kept = [value for value in graph.graph.initializer
            if value.name not in replaced or value.name in remaining_inputs]
    del graph.graph.node[:]
    graph.graph.node.extend(rewritten)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(kept + added)
    onnx.checker.check_model(graph)
    destination = root / "decode.onnx"
    onnx.save_model(graph, str(destination))
    write_json(root / "quantization.json", {"finished_at": now(), "projections": selected,
        "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "onnx_bytes": destination.stat().st_size, "original_bytes": original.stat().st_size,
        "activation_dtype": "int8" if activations else "float16", "weight_dtype": "int8", "boundary_dtype": "float16",
        "audio_generator_dtype": "float32", "reference_reprepared": False})


def emergency_weights(root):
    # The retained replay found severe signal loss from activation quantization.
    # Preserve activations and measure whether TensorRT retains compressed weights.
    emergency_quantize(root, activations=False)


def emergency_build(root):
    """Use the existing bounded builder with the original dynamic shape profile."""
    original = root.parent / "tensorrt-unified-native-reference"
    previous = json.loads((original / "result.json").read_text())
    argv = [f"--onnx={root}/decode.onnx" if value.startswith("--onnx=") else
            f"--saveEngine={root}/decode.engine" if value.startswith("--saveEngine=") else
            value for value in previous["phases"][1]["argv"]]
    argv += ["--profilingVerbosity=detailed", f"--exportLayerInfo={root}/layers.json"]
    result = guarded(argv, root / "build-command", timeout=600)
    assert result["returncode"] == 0 and result["reason"] is None, "INT8 build rejected"
    engine = root / "decode.engine"
    write_json(root / "engine.json", {"finished_at": now(), "bytes": engine.stat().st_size,
        "original_bytes": (original / "decode.engine").stat().st_size,
        "sha256": hashlib.sha256(engine.read_bytes()).hexdigest()})
    assert engine.stat().st_size < (original / "decode.engine").stat().st_size, \
        "Builder expanded the compressed weights; this engine is not smaller"


def emergency_tokens(root):
    # Include a retained long passage; a short greeting cannot reveal late drift.
    real_tokens(root, "trt", case_indices=(0, 7, len(REAL_TEXTS) - 1))
    result = json.loads((root / "trt-tokens.json").read_text())
    assert len(result["cases"]) == 3
    assert all(case["runs"]["free"]["natural_eos"] for case in result["cases"]), \
        "Emergency speech hit the cache limit instead of finishing naturally"


def emergency_render(root):
    real_render(root, case_indices=(0, 7, len(REAL_TEXTS) - 1), variants=("trt",))


def emergency_diagnose(root):
    """Distinguish weight error, activation error and TensorRT import error."""
    import gc
    import numpy as np
    import torch
    torch.set_num_threads(4)
    calibration = json.loads((root / "calibration.json").read_text())["channel_maxima"]
    prompt = torch.from_numpy(np.load(root / "prompt-0.npy")).cuda()
    reference = np.load(root / "fp16-0.npz")["logits"][0].astype(np.float32)
    report = {"started_at": now(), "variants": {}, "reference_reprepared": False}
    with torch.inference_mode():
        for variant in ("native", "weights", "w8a8", "mlp-w8a8"):
            parameters, statistics = {}, {}
            def observe(name, values, owner):
                if variant == "native" or (variant == "mlp-w8a8" and ".mlp." not in name):
                    return None
                if name not in parameters:
                    q, scale, inverse, activation = emergency_projection(
                        owner.weight.detach().cpu().numpy(), calibration[name])
                    owner.weight.copy_(torch.from_numpy(q.astype(np.float32) * scale).to(
                        device=owner.weight.device, dtype=owner.weight.dtype))
                    parameters[name] = (torch.from_numpy(inverse).to(values.device),
                                        torch.tensor(activation, device=values.device))
                inverse, scale = parameters[name]
                smoothed = values * inverse
                if variant == "weights":
                    return smoothed
                integers = torch.round(smoothed / scale)
                if name not in statistics:
                    statistics[name] = (torch.mean((integers.abs() > 127).float()),
                                        torch.mean((integers == 0).float()))
                return integers.clamp(-127, 127) * scale
            step = real_native_model(torch.float16, observer=observe)
            logits, past = step(prompt, None)
            actual = logits.float().cpu().numpy()[0]
            report["variants"][variant] = {"rmse": float(np.sqrt(np.mean((actual - reference)**2))),
                "logit_min": float(actual.min()), "logit_max": float(actual.max()),
                "top_token": int(actual.argmax()), "reference_top_token": int(reference.argmax()),
                "projections": {name: {"clipped_fraction": float(a), "zero_fraction": float(b)}
                                for name, (a, b) in statistics.items()}}
            del step, logits, past, parameters, statistics
            gc.collect()
            torch.cuda.empty_cache()
        candidate = np.load(root / "trt-0.npz")["logits"][0].astype(np.float32)
        report["variants"]["tensorrt-w8a8"] = {"rmse": float(np.sqrt(np.mean((candidate-reference)**2))),
            "logit_min": float(candidate.min()), "logit_max": float(candidate.max()),
            "top_token": int(candidate.argmax()), "reference_top_token": int(reference.argmax())}
    report["finished_at"] = now()
    write_json(root / "diagnosis.json", report)


def emergency_native(root):
    """Exercise resident INT8 weights with Whisper and the retained LFM3 setup."""
    import urllib.request
    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.tts_turbo import Conditionals
    torch.set_num_threads(4)
    model = real_audio_model()
    assert model.t3.tfmr._hailstorm_int8_projections == 96
    assert next(model.s3gen.parameters()).dtype == torch.float32
    # Measure retained CPU arenas separately from live model tensors. A cold
    # load can leave freed checkpoint/conversion pages in libc's heap cache.
    resident = lambda: int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    trim = {"resident_before": resident(), "available_before": available()}
    trim["returncode"] = ctypes.CDLL(None).malloc_trim(0)
    trim.update(resident_after=resident(), available_after=available())
    trim["loading_released_bytes"] = getattr(sys.modules.get("sitecustomize"), "CPU_TRIM_RELEASED_BYTES", 0)
    print(json.dumps({"allocator_trim": trim}), flush=True)
    model.conds = Conditionals.load(root / "conditionals.pt", map_location="cpu").to("cuda")
    prepared = model.conds
    buffers = [value for name, value in model.t3.tfmr.named_buffers() if name.endswith("weight_int8")]
    assert len(buffers) == 96 and all(value.dtype == torch.int8 for value in buffers)
    report = {"started_at": now(), "reference_reprepared": False, "cases": [], "cycles": [],
              "projection_bytes": sum(value.numel() * value.element_size() for value in buffers),
              "projection_fp16_bytes": sum(value.numel() * 2 for value in buffers),
              "scratch_bytes_per_stream": model.t3.tfmr._hailstorm_int8_scratch_elements * 2,
              "audio_generator_dtype": str(next(model.s3gen.parameters()).dtype), "allocator_trim": trim}
    prior_result = root / "native-resumed-result.json"
    if not prior_result.exists():
        prior_result = root / "native-result.json"
    report_path = prior_result
    if prior_result.exists():
        previous = json.loads(prior_result.read_text())
        # A completed case already includes its actual Whisper response. Keep
        # those samples; only the interrupted long passage and cycles remain.
        for row in previous.get("cases", []):
            assert (root / row["wav"]).is_file()
            assert row["word_errors"] <= max(1, row["words"] // 10)
        report["cases"] = previous.get("cases", [])
        report["prior_result_sha256"] = hashlib.sha256(prior_result.read_bytes()).hexdigest()
        source_key = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        report_path = root / ("native-" + source_key + "-result.json")
        assert not report_path.exists(), "A repeated native failure requires fresh diagnosis"
    runtime = json.loads((root.parent / "lfm3-prior-runtime.json").read_text())["cmd"]
    llm_url = "http://" + runtime[runtime.index("--host") + 1] + ":" + runtime[runtime.index("--port") + 1]

    def speak(text, name):
        torch.manual_seed(42)
        tick = time.monotonic()
        waveform = model.generate(text, audio_prompt_path=None, exaggeration=0.0,
                                  cfg_weight=0.0, temperature=0.8)
        values = waveform.squeeze().detach().cpu().numpy()
        assert values.size and np.isfinite(values).all()
        destination = root / (name + ".wav")
        assert not destination.exists(), "Completed speech samples must remain unchanged"
        sf.write(destination, values, model.sr, subtype="PCM_16")
        assert model.conds is prepared, "Reference conditioning was replaced"
        return {"text": text, "wav": destination.name,
                "seconds": time.monotonic() - tick, "audio_seconds": len(values) / model.sr,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "cpu_trim_released_total": getattr(sys.modules.get("sitecustomize"), "CPU_TRIM_RELEASED_BYTES", 0)}

    def listen(path):
        response = subprocess.check_output(["curl", "-fsS", "--max-time", "90",
            "http://127.0.0.1:8005/v1/audio/transcriptions", "-F", "file=@" + str(path),
            "-F", "response_format=json", "-F", "temperature=0"], text=True)
        text = json.loads(response)["text"].strip()
        assert text and "inaudible" not in text.lower(), "Speech failed transcription"
        return text

    try:
        for index in (0, 7, len(REAL_TEXTS) - 1):
            if any(row["text"] == REAL_TEXTS[index] for row in report["cases"]):
                continue
            row = speak(REAL_TEXTS[index], "native-" + str(index))
            row["transcript"] = listen(root / row["wav"])
            row["word_errors"], row["words"] = word_errors(row["text"], row["transcript"])
            report["cases"].append(row)
            write_json(report_path, report)
        for index in range(3):
            heard = listen(root / "native-0.wav")
            body = json.dumps({"model": "lfm3-1.7b", "messages": [
                {"role": "user", "content": "You heard: " + heard +
                 " Reply with exactly these words: Emergency systems ready."}],
                "max_tokens": 512, "temperature": 0, "stream": False}).encode()
            request = urllib.request.Request(llm_url + "/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                answer = json.load(response)
            write_json(root / (report_path.stem + "-llm-" + str(index) + ".json"), answer)
            text = answer["choices"][0]["message"]["content"].strip()
            assert text, "LFM3 returned no spoken answer"
            row = speak(text, "cycle-" + str(index))
            row.update(heard=heard, transcript=listen(root / row["wav"]), llm_usage=answer.get("usage"))
            row["word_errors"], row["words"] = word_errors(text, row["transcript"])
            assert row["word_errors"] <= max(1, row["words"] // 10), "Spoken answer changed its words"
            report["cycles"].append(row)
            write_json(report_path, report)
        assert all(row["word_errors"] <= max(1, row["words"] // 10) for row in report["cases"]), \
            "Emergency speech did not preserve the requested words"
    finally:
        report["finished_at"] = now()
        write_json(report_path, report)


def emergency_trial(root, backup_receipt, phases=None):
    """Own the experiment and restore the exact existing service on every exit.

    Admission requires a completed recovery receipt. No live mount, image or
    container definition is replaced; the candidate's files stay in its trial.
    """
    import urllib.request
    docker = ["sudo", "-n", "docker"]
    backup = json.loads(backup_receipt.read_text())
    deployed = json.loads(subprocess.check_output(docker + ["inspect", "chatterbox-tts"], text=True))[0]
    assert backup["archive_readback"] and backup["image_layers_verified"]
    assert backup["original_image"] == deployed["Image"], "Backup belongs to another service image"
    assert deployed["State"]["Running"], "Trial requires the healthy original service"
    assert subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 1
    if phases is None:
        root.mkdir(parents=True, exist_ok=False)
    else:
        assert root.is_dir(), "Focused continuation requires retained trial evidence"
    report = {"started_at": now(), "backup_receipt": str(backup_receipt), "phases": [],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "production_activation": False, "reference_reprepared": False}
    script = str(Path(__file__).resolve())
    report_path = root / ("emergency-result.json" if phases is None else
                          "emergency-" + report["source_sha256"][:12] + "-result.json")
    owned = None
    lfm3 = None
    prior = {}

    def owned_stopped(name):
        state = subprocess.run(docker + ["inspect", name], capture_output=True, text=True, timeout=20)
        if state.returncode:
            if name in state.stderr and any(marker in state.stderr.lower() for marker in
                    ("no such container", "no such object")):
                return True  # Docker rejected creation; no trial can be resident.
            raise RuntimeError("Owned trial state could not be verified: " + state.stderr)
        return not json.loads(state.stdout)[0]["State"]["Running"]

    try:
        def interrupted(signum, frame):
            raise InterruptedError(f"signal {signum}")
        for sig in (signal.SIGINT, signal.SIGTERM):
            prior[sig] = signal.signal(sig, interrupted)
        subprocess.run(docker + ["stop", "-t", "30", "chatterbox-tts"], check=True, timeout=45)
        assert available() > 4 * 1024**3, "Insufficient headroom after original stopped"
        for mode in phases or ("emergency_calibrate", "emergency_quantize", "emergency_build",
                               "emergency_tokens", "emergency_render"):
            if mode == "emergency_native":
                prior_runtime = json.loads((root.parent / "lfm3-prior-runtime.json").read_text())
                report["patch_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                          for path in (root / "patches").glob("*.py")}
                # Reuse the exact previous 32k-context LFM3 command. This task
                # owns only the new child process, never a pre-existing server.
                with (root / "lfm3.log").open("w") as output:
                    lfm3 = subprocess.Popen(prior_runtime["cmd"], cwd=prior_runtime["cwd"],
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                report["lfm3_pid"] = lfm3.pid
            if mode == "emergency_build":
                argv = ["python3", script, mode, "--output-dir", str(root)]
            else:
                owned = "myth-e-" + mode.removeprefix("emergency_") + "-" + str(os.getpid())
                argv = docker + ["run", "--name", owned, "--runtime", "nvidia", "--network",
                    "host" if mode == "emergency_native" else "none",
                    "--restart", "no", "--memory", "5g", "--memory-swap", "5g",
                    "--volumes-from", "chatterbox-tts:ro",
                    "-v", f"{root.parent}:{root.parent}:ro", "-v", f"{root}:{root}:rw",
                    "-v", f"{script}:/tensorrt-trial.py:ro",
                    "--entrypoint", "python3"]
                for entry in deployed["Config"]["Env"]:
                    if entry.split("=", 1)[0] not in ("CHATTERBOX_TENSORRT_ENGINE", "CHATTERBOX_TENSORRT_SHA256",
                                                     "PYTORCH_NO_CUDA_MEMORY_CACHING"):
                        argv += ["-e", entry]
                # A committed image inherits its container's environment.
                # Filtering run flags alone does not clear those image defaults.
                argv += ["-e", "CHATTERBOX_TENSORRT_ENGINE=", "-e", "CHATTERBOX_TENSORRT_SHA256=",
                         "-e", "PYTORCH_NO_CUDA_MEMORY_CACHING=0"]
                if mode == "emergency_native":
                    argv += ["-v", f"{root}/patches:/app/patches:ro", "-e", "CHATTERBOX_INT8_WEIGHTS=1",
                             "-e", "PYTORCH_NO_CUDA_MEMORY_CACHING=1"]
                # Identical absolute paths keep sibling authorities and their
                # retained symlink targets valid both here and on the host.
                argv += [backup["snapshot_image"], "/tensorrt-trial.py", mode, "--output-dir", str(root)]
            try:
                command_root = root / (mode + "-command" + (
                    "-" + report["source_sha256"][:12] if phases is not None else ""))
                assert not command_root.exists(), "A repeated failure requires a fresh diagnosis"
                result = guarded(argv, command_root, timeout=650)
                report["phases"].append({"mode": mode, **result})
                assert result["reason"] is None and result["returncode"] == 0, f"{mode} failed"
            finally:
                if owned:
                    subprocess.run(docker + ["stop", "-t", "5", owned], capture_output=True, timeout=20)
                    assert owned_stopped(owned), "Owned trial did not stop"
                    owned = None
            write_json(report_path, report)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        # Cancellation retires trial work. Restoration returns the original
        # service; it never promotes the unreviewed emergency engine.
        report["original_restored"] = False
        try:
            if owned is not None:
                # A failed graceful stop must not put two speech models in RAM.
                # Escalate only this task's uniquely named container, then prove
                # it stopped before admitting the original service again.
                subprocess.run(docker + ["kill", owned], capture_output=True, timeout=20)
                assert owned_stopped(owned), "Owned trial cleanup not verified"
            if lfm3 is not None and lfm3.poll() is None:
                try:
                    os.killpg(lfm3.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The owned server exited between poll and signalling.
                try:
                    lfm3.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(lfm3.pid, signal.SIGKILL)
                    lfm3.wait(timeout=5)
                report["owned_lfm3_stopped"] = True
            subprocess.run(docker + ["start", "chatterbox-tts"], check=True, timeout=30)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8004/api/ui/initial-data", timeout=5) as response:
                        info = json.load(response)
                    if info.get("model_info", {}).get("loaded") is True:
                        report["original_restored"] = True
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            assert report["original_restored"], "Original service restoration needs operator attention"
        except BaseException as error:
            report["restoration_error"] = repr(error)
            raise
        finally:
            report["finished_at"] = now()
            write_json(report_path, report)
            for sig, handler in prior.items():
                signal.signal(sig, handler)


def real_trial(root, resume=False, case_index=None):
    """Sequential, guarded phases; no build, service replacement or activation."""
    if resume:
        assert root.is_dir()
        assert json.loads((root / "protocol.json").read_text())["corpus"] == list(REAL_TEXTS)
    else:
        root.mkdir(parents=True, exist_ok=False)
    source = root.parent / "tensorrt-unified-native-reference"
    engine = source / "decode.engine"
    assert engine.is_file()
    if not resume:
        (root / "decode.engine").symlink_to(engine)
    assert not json.loads(subprocess.check_output(
        ["sudo", "-n", "docker", "inspect", "chatterbox-tts"], text=True))[0]["State"]["Running"]
    assert subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 1
    assert available() > 5 * 1024**3
    previous = json.loads((source / "result.json").read_text())
    report = {"started_at": now(), "phases": [], "engine_source": str(engine),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "corpus": REAL_TEXTS, "production_activation": False}
    report["case_index"] = case_index
    if not resume:
        write_json(root / "protocol.json", report)
    source_key = report["source_sha256"][:12]
    report_path = root / (f"real-resume-{source_key}.json" if resume else "real-result.json")
    try:
        for mode in ("real_capture", "real_fp16", "real_fp32", "real_trt", "real_render"):
            outputs = {"real_capture": "capture.json", "real_fp16": "fp16-tokens.json",
                       "real_fp32": "fp32-tokens.json", "real_trt": "trt-tokens.json",
                       "real_render": "audio.json"}
            destination = root / outputs[mode]
            success_path = root / (mode + "-success.json")
            if resume and destination.exists() and success_path.exists() and (case_index is None or mode == "real_capture"):
                completed = json.loads(destination.read_text())
                success = json.loads(success_path.read_text())
                command = json.loads(Path(success["command_receipt"]).read_text())
                expected = len(REAL_TEXTS) * (3 if mode == "real_render" else 1)
                if (completed.get("finished_at") and len(completed.get("cases", [])) == expected
                        and command["returncode"] == 0 and command["reason"] is None
                        and success["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()):
                    report["phases"].append({"mode": mode, "reused_completed_receipt": str(destination)})
                    continue
            name = "chatterbox-" + mode.replace("_", "-") + "-" + str(os.getpid())
            argv = previous["phases"][0]["argv"][:]
            argv[argv.index("--name") + 1] = name
            argv = [f"{root}:/trial" if a.endswith(":/trial") else
                    f"{Path(__file__).resolve()}:/tensorrt-trial.py:ro" if a.endswith(":/tensorrt-trial.py:ro") else
                    mode if a == "export-unified" else a for a in argv]
            if case_index is not None:
                argv += ["--case-index", str(case_index)]
            # Loading S3Gen retains the deployed 5 GiB cgroup ceiling. FP32
            # transformer-only runs do not also load S3Gen or the voice encoder.
            argv[argv.index("--entrypoint"):argv.index("--entrypoint")] = [
                "-v", f"{source}:{source}:ro",
                "-v", f"{root.parent}/tensorrt-python310:/trt-python:ro",
                "-v", "/usr/lib/aarch64-linux-gnu:/host-trt:ro",
                "-v", "/usr/local/cuda/lib64:/host-cuda:ro"]
            try:
                command_root = root / (mode + "-command")
                if command_root.exists():
                    command_root = root / (mode + "-command-" + source_key)
                assert not command_root.exists(), "Repeated failure requires a fresh explicit diagnosis"
                phase = guarded(argv, command_root, timeout=1500)
                report["phases"].append({"mode": mode, **phase})
                assert phase["reason"] is None and phase["returncode"] == 0, f"{mode} failed"
                # A finally-written report alone cannot establish success.
                # Bind reuse to this exact successful command and output bytes.
                prior_success = json.loads(success_path.read_text()) if success_path.exists() else None
                write_json(success_path, {"source_sha256": report["source_sha256"],
                    "case_index": case_index, "prior_success": prior_success,
                    "command_receipt": str(command_root / "command.json"),
                    "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
            finally:
                subprocess.run(["sudo", "-n", "docker", "stop", "-t", "5", name], capture_output=True, timeout=20)
                state = json.loads(subprocess.check_output(["sudo", "-n", "docker", "inspect", name], text=True))[0]
                assert not state["State"]["Running"], "Owned trial container did not stop"
                write_json(report_path, report)
    finally:
        report["finished_at"] = now()
        write_json(report_path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(
        "run", "export", "infer", "native", "validate", "run-unified", "export-unified",
        "precision", "probe", "probe_export", "probe_infer", "replay", "replay_export",
        "replay_precision", "projection_precision", "guarded-infer", "mask", "mask_export",
        "mask_backends", "mask_backends_export", "attention", "attention_export",
        "layer_graph", "layer_sweep", "real_trial", "real_capture", "real_fp16",
        "real_fp32", "real_trt", "real_render", "real_analyse", "real_resume",
        "emergency_calibrate", "emergency_quantize", "emergency_build",
        "emergency_tokens", "emergency_render", "emergency_trial", "emergency_diagnose", "emergency_weights",
        "emergency_native"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-index", type=int, choices=range(len(REAL_TEXTS)))
    parser.add_argument("--backup-receipt", type=Path)
    parser.add_argument("--phase", action="append", choices=("emergency_calibrate", "emergency_quantize",
        "emergency_build", "emergency_tokens", "emergency_render", "emergency_diagnose", "emergency_weights",
        "emergency_native"))
    args = parser.parse_args()
    if args.mode == "emergency_trial":
        if args.backup_receipt is None:
            parser.error("emergency_trial requires --backup-receipt")
        emergency_trial(args.output_dir.resolve(), args.backup_receipt.resolve(), phases=args.phase)
    elif args.mode == "real_resume":
        real_trial(args.output_dir.resolve(), resume=True, case_index=args.case_index)
    elif args.mode in ("real_fp16", "real_fp32", "real_trt"):
        real_tokens(args.output_dir.resolve(), args.mode.removeprefix("real_"), case_index=args.case_index)
    elif args.mode == "real_render":
        real_render(args.output_dir.resolve(), case_index=args.case_index)
    elif args.mode == "native":
        export(args.output_dir.resolve(), native_only=True)
    elif args.mode == "export-unified":
        export(args.output_dir.resolve(), unified=True)
    elif args.mode == "run-unified":
        run(args.output_dir.resolve(), unified=True)
    elif args.mode == "mask":
        probe(args.output_dir.resolve(), mask=True)
    elif args.mode == "mask_export":
        export(args.output_dir.resolve(), unified=True, compare_mask=True)
    elif args.mode == "mask_backends":
        probe(args.output_dir.resolve(), mask=True, backend_probe=True)
    elif args.mode == "mask_backends_export":
        export(args.output_dir.resolve(), unified=True, compare_mask=True, backend_probe=True)
    elif args.mode == "attention":
        probe(args.output_dir.resolve(), attention=True)
    elif args.mode == "attention_export":
        probe_export(args.output_dir.resolve(), attention=True)
    elif args.mode == "replay":
        probe(args.output_dir.resolve(), replay=True)
    elif args.mode == "replay_export":
        probe_export(args.output_dir.resolve(), replay=True)
    elif args.mode == "projection_precision":
        precision(args.output_dir.resolve(), projections=True)
    elif args.mode == "guarded-infer":
        # A completed engine can be checked after a builder's post-build load
        # breaches the reserve; do not rebuild or relax memory safety to do so.
        root = args.output_dir.resolve()
        result = guarded(["python3", str(Path(__file__).resolve()), "infer",
                          "--output-dir", str(root)], root / "infer-standalone-command")
        assert result["returncode"] == 0 and result["reason"] is None
    else:
        globals()[args.mode](args.output_dir.resolve())
