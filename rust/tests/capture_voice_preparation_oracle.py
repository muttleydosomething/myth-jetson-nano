#!/usr/bin/env python3
"""Actual deployed individual methods; exact-input replay proves their assembly."""
import ast
import gc
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from functools import lru_cache
from types import SimpleNamespace
assert sys.flags.no_site
sys.path.append('/usr/local/lib/python3.10/dist-packages')
import numpy as np
import torch
import torchaudio as ta
import librosa
import pyloudnorm as ln
import soundfile as sf
from safetensors import safe_open
from safetensors.torch import load_file
torch.set_num_threads(2)
checkpoint, output = map(Path, sys.argv[1:3])
output.mkdir(exist_ok=False)
base = Path('/usr/local/lib/python3.10/dist-packages')
source = base/'chatterbox'
s3source = source/'models/s3gen/s3gen.py'
ttssource = source/'tts_turbo.py'
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()
def export(name,value):
    if torch.is_tensor(value): value=value.detach().cpu().numpy()
    path=output/(name+'.npy')
    np.save(path,np.ascontiguousarray(value,dtype='<f4'),allow_pickle=False)
    return dict(file=path.name,sha256=sha(path))
def module_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
melmod=module_file('deployed_reference_mel',source/'models/s3gen/utils/mel.py')
env=dict(torch=torch,np=np,ta=ta,lru_cache=lru_cache,logging=__import__('logging'),
         S3GEN_SR=24000,S3_SR=16000,librosa=librosa,ln=ln,math=math)
s3tree=ast.parse(s3source.read_text())
s3cls=next(n for n in s3tree.body if isinstance(n,ast.ClassDef) and n.name=='S3Token2Mel')
embed=next(n for n in s3cls.body if isinstance(n,ast.FunctionDef) and n.name=='embed_ref')
resampler=next(n for n in s3tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_resampler')
ttstree=ast.parse(ttssource.read_text())
ttscls=next(n for n in ttstree.body if isinstance(n,ast.ClassDef) and n.name=='ChatterboxTurboTTS')
methods=[n for n in ttscls.body if isinstance(n,ast.FunctionDef) and n.name in ('norm_loudness','prepare_conditionals')]
exec(compile(ast.Module(body=[resampler,embed]+methods,type_ignores=[]),'deployed-reference-methods','exec'),env)
transform=ta.transforms.Resample(24000,16000)
tensors=dict(window=export('window',torch.hann_window(1920)),
             mel_basis=export('mel_basis',librosa.filters.mel(sr=24000,n_fft=1920,n_mels=80,fmin=0,fmax=8000)),
             resample_kernel=export('resample_kernel',transform.kernel))
manifest=dict(schema_version=1,analysis_kind='s3gen_reference_analysis_fp32',
              source_sha256=sha(s3source),tensors=tensors)
(output/'analysis.json').write_text(json.dumps(manifest,indent=2)+'\n')
analysis_cases=[]
for name,length,kind in [('minimum',721,'chirp'),('below-hop',959,'chirp'),('hop',960,'chirp'),
                         ('odd-tail',1440,'chirp'),('silence',2401,'silence'),('impulse',1921,'impulse')]:
    t=np.arange(length,dtype=np.float64)/24000
    pcm=(.17*np.sin(2*np.pi*(173*t+57*t*t))+.05*np.sin(2*np.pi*517*t)).astype(np.float32)
    if kind=='silence': pcm[:]=0
    if kind=='impulse': pcm[:]=0;pcm[length//2]=.5
    audio=torch.from_numpy(pcm)
    mels=melmod.mel_spectrogram(audio).transpose(1,2).contiguous()
    analysis_cases.append(dict(name=name,pcm24=export(name+'-audio',pcm),
                               pcm16=export(name+'-pcm16',transform(audio)),mels=export(name+'-mels',mels)))
cases=[]
runtime=[]
for name,length,kind in [('tone',144000,'tone'),('odd-chirp',144480,'chirp'),('long-gated',384121,'gated')]:
    t=np.arange(length,dtype=np.float64)/24000
    pcm=(.17*np.sin(2*np.pi*(173*t+(57*t*t if kind=='chirp' else 0)))+.05*np.sin(2*np.pi*517*t)).astype(np.float32)
    if kind=='gated': pcm[:72000]*=.001;pcm[240000:]*=.04
    path=output/(name+'.wav')
    sf.write(path,pcm,24000,subtype='FLOAT')
    audio,_=librosa.load(path,sr=24000)
    audio=env['norm_loudness'](None,audio,24000)
    pcm16=librosa.resample(audio,orig_sr=24000,target_sr=16000)
    reference24=audio[:240000].copy()
    reference16=transform(torch.from_numpy(reference24)).numpy()
    mels=melmod.mel_spectrogram(torch.from_numpy(reference24)).transpose(1,2).contiguous()
    row=dict(name=name,audio=path.name,audio_sha256=sha(path),normalise=True,
             pcm24=export(name+'-pcm24',audio),pcm16=export(name+'-pcm16',pcm16),
             reference24=export(name+'-reference24',reference24),reference16=export(name+'-reference16',reference16),
             s3_mels=export(name+'-prompt_feat',mels))
    cases.append(row)
    runtime.append(dict(pcm24=audio,pcm16=pcm16,reference24=reference24,reference16=reference16,mels=mels))
# Real modules are importable without executing heavyweight package constructors.
for name,path in [('chatterbox',source),('chatterbox.models',source/'models'),
                  ('chatterbox.models.voice_encoder',source/'models/voice_encoder'),
                  ('chatterbox.models.s3tokenizer',source/'models/s3tokenizer'),('s3tokenizer',base/'s3tokenizer')]:
    package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
ve_module=importlib.import_module('chatterbox.models.voice_encoder.voice_encoder')
model=ve_module.VoiceEncoder()
model.load_state_dict(load_file(checkpoint/'ve.safetensors'),strict=True)
model.eval()
with torch.inference_mode():
    for row,rt in zip(cases,runtime):
        embedding=model.embeds_from_wavs([rt['pcm16']],sample_rate=16000).mean(axis=0,keepdims=True)
        row['t3_speaker']=export(row['name']+'-speaker',embedding);rt['speaker']=embedding
        print(json.dumps(dict(stage='voice_encoder',name=row['name'])),flush=True)
del model
gc.collect()
camp_module=module_file('deployed_campplus',source/'models/s3gen/xvector.py')
with torch.device('meta'): model=camp_module.CAMPPlus(feat_dim=80,embedding_size=192)
state={}
with safe_open(checkpoint/'s3gen_meanflow.safetensors',framework='pt',device='cpu') as weights:
    for key in weights.keys():
        if key.startswith('speaker_encoder.'):
            value=weights.get_tensor(key)
            state[key.removeprefix('speaker_encoder.')]=value if not value.is_floating_point() else value.float()
model.load_state_dict(state,strict=True,assign=True)
del state
model.eval()
with torch.inference_mode():
    for row,rt in zip(cases,runtime):
        embedding=model.inference(torch.from_numpy(rt['reference16']).unsqueeze(0))
        row['s3_embedding']=export(row['name']+'-embedding',embedding);rt['embedding']=embedding
        print(json.dumps(dict(stage='campplus',name=row['name'])),flush=True)
del model
gc.collect()
codec_module=importlib.import_module('chatterbox.models.s3tokenizer.s3tokenizer')
core=importlib.import_module('s3tokenizer.model_v2')
with torch.device('meta'): model=codec_module.S3Tokenizer()
state={}
with safe_open(checkpoint/'s3gen_meanflow.safetensors',framework='pt',device='cpu') as weights:
    for key in weights.keys():
        if key.startswith('tokenizer.'): state[key.removeprefix('tokenizer.')]=weights.get_tensor(key).float()
state['_mel_filters']=torch.from_numpy(librosa.filters.mel(sr=16000,n_fft=400,n_mels=128))
state['window']=torch.hann_window(400)
model.load_state_dict(state,strict=True,assign=True)
del state
model.encoder.freqs_cis=core.precompute_freqs_cis(64,2048)
model.eval()
with torch.inference_mode():
    for row,rt in zip(cases,runtime):
        raw,lengths=model(torch.from_numpy(rt['reference16']).unsqueeze(0))
        tokens,t3lens=model([rt['pcm16'][:240000]],max_len=375)
        rt['raw_tokens'],rt['raw_lengths'],rt['t3_tokens']=raw,lengths,tokens
        row['t3_tokens']=tokens[0].tolist()
        print(json.dumps(dict(stage='codec',name=row['name'],t3_tokens=len(row['t3_tokens']),s3_tokens=int(lengths[0]))),flush=True)
del model
gc.collect()
class T3Cond(SimpleNamespace):
    def to(self,**kwargs): return self
class Conditionals:
    def __init__(self,t3,gen): self.t3=t3;self.gen=gen
env.update(T3Cond=T3Cond,Conditionals=Conditionals)
for row,rt in zip(cases,runtime):
    def speaker_inference(audio):
        assert np.array_equal(audio.numpy()[0],rt['reference16'])
        return rt['embedding'].clone()
    def codec_inference(audio):
        assert np.array_equal(audio.numpy()[0],rt['reference16'])
        return rt['raw_tokens'].clone(),rt['raw_lengths'].clone()
    def t3_inference(wavs,max_len):
        assert max_len==375 and np.array_equal(wavs[0],rt['pcm16'][:240000])
        return rt['t3_tokens'].clone(),torch.tensor([rt['t3_tokens'].shape[1]])
    def speaker_t3(wavs,sample_rate):
        assert sample_rate==16000 and np.array_equal(wavs[0],rt['pcm16'])
        return rt['speaker'].copy()
    decoder=SimpleNamespace(device=torch.device('cpu'),dtype=torch.float32,
                            mel_extractor=melmod.mel_spectrogram,speaker_encoder=SimpleNamespace(inference=speaker_inference))
    class Codec:
        def __call__(self,audio): return codec_inference(audio)
        def forward(self,wavs,max_len): return t3_inference(wavs,max_len)
    decoder.tokenizer=Codec()
    def embed_reference(audio,rate,device):
        assert rate==24000 and np.array_equal(audio,rt['reference24'])
        return env['embed_ref'](decoder,audio,rate,device=device)
    decoder.embed_ref=embed_reference
    owner=SimpleNamespace(device=torch.device('cpu'),DEC_COND_LEN=240000,ENC_COND_LEN=240000,
                          s3gen=decoder,t3=SimpleNamespace(hp=SimpleNamespace(speech_cond_prompt_len=375)),
                          ve=SimpleNamespace(embeds_from_wavs=speaker_t3))
    owner.norm_loudness=lambda wav,sr:env['norm_loudness'](None,wav,sr)
    env['prepare_conditionals'](owner,output/row['audio'])
    reference=owner.conds.gen
    assert np.array_equal(owner.conds.t3.speaker_emb.numpy(),rt['speaker'])
    assert torch.equal(owner.conds.t3.cond_prompt_speech_tokens,rt['t3_tokens'])
    assert torch.equal(reference['prompt_feat'],rt['mels']) and reference['prompt_feat_len'] is None
    assert torch.equal(reference['embedding'],rt['embedding'])
    row['s3_tokens']=reference['prompt_token'][0].tolist()
    row['prompt_token_len']=int(reference['prompt_token_len'][0])
    row['prompt_feat_shape']=list(reference['prompt_feat'].shape)
    row['prompt_feat_len']=None;row['actual_method_assembly_exact']=True
assert not torch.cuda.is_initialized()
receipt=dict(schema_version=1,analysis_manifest='analysis.json',analysis_manifest_sha256=sha(output/'analysis.json'),
             analysis_cases=analysis_cases,voice_cases=cases,source_hashes={str(p.relative_to(base)):sha(p) for p in
             [s3source,ttssource,source/'models/s3gen/utils/mel.py',base/'torchaudio/functional/functional.py',
              base/'torchaudio/transforms/_transforms.py']},
             checkpoint_hashes={name:sha(checkpoint/name) for name in ['ve.safetensors','s3gen_meanflow.safetensors']},
             retained_voice_preparations=0,cuda_initialized=False,
             method='Actual deployed individual encoder methods; actual embed_ref/prepare_conditionals AST replay uses captured neural outputs with exact PCM input assertions. No full speech model instantiated.')
(output/'oracle.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(passed=True,complete_cases=len(cases),analysis_cases=len(analysis_cases),cuda_initialized=False,retained_voice_preparations=0)),flush=True)
