#!/usr/bin/env python3
# Copyright (c) 2026 Hailstorm contributors.
# UpsampleConformer: CosyVoice/ESPnet contributors, Apache-2.0.
"""Build-time strict-FP32 Conformer graph from admitted tensors.
No service imports, reference preparation, audio encoding or runtime Python.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def build_engine(args, receipt):
    # Reuse the established isolated host TensorRT library route.
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
    if not parser.parse_from_file(str(args.output/"conformer.onnx")):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    # The 1700-output-frame relative-attention tactic needs 324 MiB.
    # Keep this explicit and bounded inside the existing 4 GiB build envelope.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512*1024**2)
    config.builder_optimization_level = 2
    profile = builder.create_optimization_profile()
    profile.set_shape("input", *[(1,n,512) for n in [1,args.opt_frames,args.max_frames]])
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT Conformer engine build failed")
    engine = args.output/"conformer.engine"
    engine.write_bytes(bytes(plan))
    receipt.update(engine_sha256=sha(engine), tensorrt_version=trt.__version__)
    (args.output/"export.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--opt-frames", type=int, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.opt_frames <= args.max_frames <= 850:
        parser.error("require 1 <= opt <= max <= 850 (1700 output frames)")
    assert sys.flags.no_site, "run with python3 -S"
    if args.build_only:
        receipt = json.loads((args.output/"export.json").read_text())
        assert receipt["onnx_sha256"] == sha(args.output/"conformer.onnx")
        assert (receipt["opt_frames"],receipt["max_frames"]) == (args.opt_frames,args.max_frames)
        build_engine(args, receipt)
        return
    sys.path.append("/usr/local/lib/python3.10/dist-packages")
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    oracle = json.loads((args.assets/"oracle.json").read_text())
    assert sha(args.assets/"conformer.json") == oracle["manifest_sha256"]
    manifest = json.loads((args.assets/"conformer.json").read_text())
    assert manifest["encoder_kind"] == "s3gen_upsample_conformer_fp32"

    class Graph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.keys = {}
            for index,(name,asset) in enumerate(manifest["tensors"].items()):
                path = args.assets/asset["file"]
                assert Path(asset["file"]).name == asset["file"]
                assert sha(path) == asset["sha256"]
                key = "w"+str(index)
                self.register_buffer(key, torch.from_numpy(np.load(path,allow_pickle=False).copy()))
                self.keys[name] = key
        def w(self,name):
            return getattr(self,self.keys[name])
        def linear(self,x,name,bias=True):
            return F.linear(x,self.w(name+".weight"),self.w(name+".bias") if bias else None)
        def norm(self,x,name,eps):
            return F.layer_norm(x,(512,),self.w(name+".weight"),self.w(name+".bias"),eps)
        def conv(self,x,name,left,right):
            return F.conv1d(F.pad(x.transpose(1,2),(left,right)),
                self.w(name+".weight"),self.w(name+".bias")).transpose(1,2)
        def embed(self,x,name):
            return self.norm(self.linear(x,name+".out.0"),name+".out.1",1e-5)*(512**.5)
        def blocks(self,x,stack,count):
            t = x.shape[1]
            # Deployed relative positions descend T-1 ... -(T-1), interleaved.
            angle = torch.arange(t-1,-t,-1,device=x.device,dtype=torch.float32)[:,None]*self.w("position_frequencies")[None,:]
            pos = torch.stack((angle.sin(),angle.cos()),dim=-1).reshape(1,-1,512)
            # ESPnet relative shift as an explicit dynamic gather. For query i
            # and key j, the selected relative column is T-1-i+j, including T=1.
            row = torch.arange(t,device=x.device)
            index = (t-1-row[:,None]+row[None,:]).reshape(1,1,t,t).expand(1,8,t,t)
            for i in range(count):
                p = stack+"."+str(i)
                z = self.norm(x,p+".norm_mha",1e-12)
                a = p+".self_attn"
                q = self.linear(z,a+".linear_q").reshape(1,t,8,64).transpose(1,2)
                k = self.linear(z,a+".linear_k").reshape(1,t,8,64).transpose(1,2)
                v = self.linear(z,a+".linear_v").reshape(1,t,8,64).transpose(1,2)
                relative = self.linear(pos,a+".linear_pos",False).reshape(1,-1,8,64).transpose(1,2)
                ac = (q+self.w(a+".pos_bias_u")[None,:,None,:])@k.transpose(-2,-1)
                bd = (q+self.w(a+".pos_bias_v")[None,:,None,:])@relative.transpose(-2,-1)
                scores = (ac+torch.gather(bd,3,index))/8.
                attended = (scores.softmax(dim=-1)@v).transpose(1,2).reshape(1,t,512)
                x = x+self.linear(attended,a+".linear_out")
                x = x+self.linear(F.silu(self.linear(self.norm(x,p+".norm_ff",1e-12),p+".feed_forward.w_1")),p+".feed_forward.w_2")
            return x
        def forward(self,x):
            x = self.embed(x,"embed")
            ahead = F.leaky_relu(self.conv(x,"pre_lookahead_layer.conv1",0,3),.01)
            x = x+self.conv(ahead,"pre_lookahead_layer.conv2",2,0)
            x = self.blocks(x,"encoders",6)
            # Expand/reshape preserves a dynamic time dimension during export.
            x = x.unsqueeze(2).expand(-1,-1,2,-1).reshape(1,-1,512)
            x = self.conv(x,"up_layer.conv",4,0)
            x = self.embed(x,"up_embed")
            return self.norm(self.blocks(x,"up_encoders",4),"after_norm",1e-5)

    graph = Graph().eval()
    args.output.mkdir(parents=True,exist_ok=False)
    cases = []
    with torch.inference_mode():
        for case in oracle["cases"]:
            x = torch.from_numpy(np.load(args.assets/case["input"],allow_pickle=False))
            expected = np.load(args.assets/case["output"],allow_pickle=False)
            actual = graph(x).numpy()
            assert actual.shape == expected.shape
            assert np.all(np.abs(actual-expected) <= 2e-4+2e-4*np.abs(expected)),case["name"]
            cases.append(dict(name=case["name"],maximum_absolute_error=float(np.max(np.abs(actual-expected)))))
        x = torch.zeros(1,min(5,args.max_frames),512)
        onnx = args.output/"conformer.onnx"
        torch.onnx.export(graph,(x,),str(onnx),input_names=["input"],output_names=["features"],
            opset_version=17,dynamic_axes={"input":{1:"frames"},"features":{1:"output_frames"}},
            do_constant_folding=True,dynamo=False)
    receipt = dict(schema_version=1,source_sha256=manifest["source_sha256"],
        oracle_manifest_sha256=oracle["manifest_sha256"],onnx_sha256=sha(onnx),
        min_frames=1,opt_frames=args.opt_frames,max_frames=args.max_frames,
        reference_preparations=0,precision="fp32",tf32=False,eager_cases=cases)
    (args.output/"export.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)

if __name__ == "__main__":
    main()
