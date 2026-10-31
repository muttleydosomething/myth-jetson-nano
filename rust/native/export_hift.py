#!/usr/bin/env python3
# HiFT/F0: Copyright (c) 2024 Alibaba Inc, Apache-2.0.
"""Build-time export of admitted HiFT. No service or voice preparation is loaded."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

def build_engine(args, receipt):
    # ONNX frame axes are dynamic; admission bounds belong to the engine build.
    receipt.update(min_frames=1,opt_frames=args.opt_frames,max_frames=args.max_frames)
    # Use the established host TensorRT route without importing service startup.
    route = args.output/"trt-build-libs"
    route.mkdir(exist_ok=True)
    for resource in Path("/host-trt").glob("libnvinfer_builder_resource_*.so.*"):
        link = route/resource.name
        if not link.is_symlink():
            link.symlink_to(resource)
    if str(route) not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(route)+":"+env.get("LD_LIBRARY_PATH", "")
        os.execvpe(sys.executable, [sys.executable,"-S",__file__,str(args.assets),str(args.output),
            "--opt-frames",str(args.opt_frames),"--max-frames",str(args.max_frames),"--build-only"],env)
    sys.path.insert(0,"/trt-python")
    import ctypes
    for library in ["/host-cuda/libcudart.so.13","/host-trt/libnvinfer.so.10",
                    "/host-trt/libnvonnxparser.so.10","/host-trt/libnvinfer_plugin.so.10"]:
        ctypes.CDLL(library,mode=ctypes.RTLD_GLOBAL)
    import tensorrt_bindings as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(args.output/"hift.onnx")):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,256*1024**2)
    config.builder_optimization_level = 2
    profile = builder.create_optimization_profile()
    for name,channels,scale in [("mel",80,1),("normal",9,480),("cache_source",1,480)]:
        profile.set_shape(name,*[(1,channels,n*scale) for n in [1,args.opt_frames,args.max_frames]])
    profile.set_shape("phases",(1,9,1),(1,9,1),(1,9,1))
    profile.set_shape("cache_samples",(1,),(1,),(1,))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network,config)
    if plan is None:
        raise RuntimeError("TensorRT HiFT engine build failed")
    (args.output/"hift.engine").write_bytes(bytes(plan))
    receipt.update(engine_sha256=sha(args.output/"hift.engine"),tensorrt_version=trt.__version__)
    (args.output/"export.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opt-frames",type=int,default=256)
    parser.add_argument("--max-frames",type=int,default=1700)
    parser.add_argument("--build-only",action="store_true")
    args = parser.parse_args()
    assert sys.flags.no_site, "run with python3 -S"
    assert 1 <= args.opt_frames <= args.max_frames <= 16384
    if args.build_only:
        receipt = json.loads((args.output/"export.json").read_text())
        assert receipt["onnx_sha256"] == sha(args.output/"hift.onnx")
        build_engine(args,receipt)
        return
    sys.path.append("/usr/local/lib/python3.10/dist-packages")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    source = Path("/usr/local/lib/python3.10/dist-packages/chatterbox")
    for name, path in [("chatterbox",source),("chatterbox.models",source/"models"),
                       ("chatterbox.models.s3gen",source/"models/s3gen")]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    hift = importlib.import_module("chatterbox.models.s3gen.hifigan")
    f0 = importlib.import_module("chatterbox.models.s3gen.f0_predictor")
    oracle = json.loads((args.assets/"oracle.json").read_text())
    manifest_path = args.assets/"hift.json"
    assert sha(manifest_path) == oracle["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["decoder_kind"] == "s3gen_hift_fp32"
    with torch.device("meta"):
        model = hift.HiFTGenerator(sampling_rate=24000,upsample_rates=[8,5,3],
            upsample_kernel_sizes=[16,11,7],source_resblock_kernel_sizes=[7,7,11],
            source_resblock_dilation_sizes=[[1,3,5]]*3,
            f0_predictor=f0.ConvRNNF0Predictor())
    # Exported assets are the effective normalized weights, so remove only
    # build-time parametrizations and replace meta parameters directly.
    for name, module in model.named_modules():
        for key in ["weight","bias","alpha"]:
            asset = manifest["tensors"].get(name+"."+key)
            if asset is None:
                continue
            path = args.assets/asset["file"]
            assert path.resolve().parent == args.assets.resolve()
            assert sha(path) == asset["sha256"]
            if torch.nn.utils.parametrize.is_parametrized(module,key):
                torch.nn.utils.parametrize.remove_parametrizations(module,key,leave_parametrized=True)
            setattr(module,key,torch.nn.Parameter(torch.from_numpy(np.load(path,allow_pickle=False).copy()),requires_grad=False))
    window_asset = manifest["tensors"]["stft_window"]
    window_path = args.assets/window_asset["file"]
    assert window_path.resolve().parent == args.assets.resolve()
    assert sha(window_path) == window_asset["sha256"]
    model.stft_window = torch.from_numpy(np.load(window_path,allow_pickle=False).copy())
    model.eval()
    # The direct deployed-method export failed on complex STFT. Its fixed
    # DFT16 and its inverse are explicit FP32 matrix operations. The first
    # convolution-based lowering passed eager Torch but failed the compiled
    # spectrum boundary. Explicit reflection and overlap-add also keep padding
    # and normalization visible in the graph instead of hiding them in tactics.
    taps = np.arange(16,dtype=np.float64)
    angles = 2*np.pi*np.arange(9,dtype=np.float64)[:,None]*taps/16
    window = model.stft_window.numpy().astype(np.float64)
    forward = np.concatenate([np.cos(angles),-np.sin(angles)])*window
    factors = np.array([1.]+[2.]*7+[1.])[:,None]
    inverse = np.concatenate([np.cos(angles)*factors,-np.sin(angles)*factors])*window/16
    def stft(self,x):
        padded = torch.cat([x[:,1:9].flip(-1),x,x[:,-9:-1].flip(-1)],dim=-1)
        starts = torch.arange(0,x.shape[-1]+1,4)
        frames = padded[:,starts[:,None]+torch.arange(16)]
        values = torch.matmul(frames,torch.from_numpy(forward.astype("float32")).T).transpose(1,2)
        return values[:,:9],values[:,9:]
    def istft(self,magnitude,phase):
        magnitude = torch.clamp(magnitude,max=100.)
        spectrum = torch.cat([magnitude*torch.cos(phase),magnitude*torch.sin(phase)],dim=1)
        frames = torch.matmul(spectrum.transpose(1,2),torch.from_numpy(inverse.astype("float32")))
        count = frames.shape[1]
        flattened = frames.reshape(1,-1)
        samples = torch.arange(8,(count-1)*4+8)
        signal = torch.zeros_like(samples,dtype=magnitude.dtype)[None]
        envelope = torch.zeros_like(signal)
        window_square = torch.from_numpy((window**2).astype("float32"))
        # FFT16/hop4 means exactly four candidate overlaps per output sample.
        # Mask absent edge frames before normalization; no discarded 0/0 is
        # produced and no sample-dependent loop or scatter atomic is required.
        for overlap in range(4):
            index = torch.div(samples,4,rounding_mode="floor")-overlap
            valid = ((index>=0)&(index<count))[None]
            bounded = torch.clamp(index,min=0,max=count-1)
            tap = samples%4+overlap*4
            value = flattened.gather(1,(bounded*16+tap)[None])
            signal = signal+torch.where(valid,value,torch.zeros_like(value))
            envelope = envelope+torch.where(valid,window_square[tap][None],torch.zeros_like(signal))
        return signal/envelope
    model._stft = types.MethodType(stft,model)
    model._istft = types.MethodType(istft,model)
    class Graph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
        def forward(self,mel,normal,phases,cache_source,cache_samples):
            pitch = self.model.f0_predictor(mel)
            frequencies = self.model.f0_upsamp(pitch[:,None])
            harmonics = torch.arange(1,10,dtype=mel.dtype).reshape(1,9,1)
            theta = 2*np.pi*(torch.cumsum(frequencies*harmonics/24000,dim=-1)%1)
            phase = phases*torch.tensor([0.]+[1.]*8).reshape(1,9,1)
            uv = (frequencies>10).to(mel.dtype)
            signal = .1*torch.sin(theta+phase)*uv+(uv*.003+(1-uv)*.1/3)*normal
            excitation = self.model.m_source.l_tanh(self.model.m_source.l_linear(signal.transpose(1,2))).transpose(1,2)
            # A full-sized carrier avoids zero-sized TensorRT bindings. Only
            # the caller's exact sample prefix can replace fresh excitation.
            excitation = torch.where(torch.arange(excitation.shape[2]).reshape(1,1,-1)<cache_samples,
                cache_source,excitation)
            return pitch,excitation,self.model.decode(mel,excitation)
    args.output.mkdir(parents=True,exist_ok=False)
    names = ["mel","normal","phases","cache_source","cache_samples"]
    with torch.inference_mode():
        torch.onnx.export(Graph().eval(),(torch.zeros(1,80,3),torch.zeros(1,9,1440),torch.zeros(1,9,1),
            torch.zeros(1,1,1440),torch.tensor([0],dtype=torch.int32)),
            str(args.output/"hift.onnx"),input_names=names,output_names=["pitch","source","waveform"],
            opset_version=17,dynamic_axes={"mel":{2:"frames"},"normal":{2:"samples"},
                "cache_source":{2:"samples"},"pitch":{1:"frames"},"source":{2:"samples"},"waveform":{1:"samples"}},
            do_constant_folding=True,dynamo=False)
    receipt = dict(schema_version=1,source_sha256=manifest["source_sha256"],
        oracle_manifest_sha256=oracle["manifest_sha256"],onnx_sha256=sha(args.output/"hift.onnx"),
        min_frames=1,opt_frames=args.opt_frames,max_frames=args.max_frames,
        reference_preparations=0,precision="fp32",tf32=False,transforms="real-dft16")
    (args.output/"export.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)
if __name__ == "__main__":
    main()
