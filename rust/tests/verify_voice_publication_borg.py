#!/usr/bin/env python3
"""Native publication/HTTP gate. Host stdlib controls processes, never inference."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import struct
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path("/home/muttleydosomething/chatterbox-rust-build-20261030")
output = Path(args.output)
output.mkdir(exist_ok=False)
binary = Path(args.binary)
template = Path("/var/lib/docker/volumes/chatterbox-outputs/_data/rust-prepared-20261030/retained-assets.json")
image = "hailstorm/myth-native:ec98e4b2"
started = time.time_ns() // 1_000_000
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def docker(*args):
    return subprocess.check_output(["sudo", "-n", "docker", *args], text=True).strip()
def state():
    return docker("inspect", "--format", "{{.State.Running}} {{.State.StartedAt}}", "chatterbox-tts")
before = state()
assert before.startswith("true "), before
# The old trial is read-only evidence; create a task-owned isolated listener.
base = ["sudo","-n","docker","run","--rm","--network","none","--read-only",
        "--user",str(os.getuid())+":"+str(os.getgid()),"--cap-drop","ALL",
        "--security-opt","no-new-privileges","--pids-limit","64",
        "--memory","1536m","--memory-swap","1536m","--cpus","2"]
mounts = [
    "/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro",
    "/lib/ld-linux-aarch64.so.1:/lib/ld-linux-aarch64.so.1:ro",
    str(root)+":"+str(root)+":ro",
    str(template.parent)+":"+str(template.parent)+":ro",
    str(output)+":"+str(output)+":rw",
]
for mount in mounts:
    base += ["-v",mount]
base += ["--entrypoint",str(binary),image]
model = root/"preparation-borg.json"
model_sha = sha(model)
template_sha = sha(template)
voice = output/"prepared"
recording = root/"voice-preparation-assets/tone.wav"
def native(arguments, label, success=True):
    result = subprocess.run(base+list(map(str,arguments)),capture_output=True,text=True,timeout=180)
    (output/(label+".stderr")).write_text(result.stderr)
    (output/(label+".stdout")).write_text(result.stdout)
    if success:
        assert result.returncode == 0, (label,result.returncode,result.stderr)
        return json.loads(result.stdout)
    assert result.returncode != 0, label
    return result
receipt = native(["publish-voice",model,model_sha,recording,template,template_sha,voice],"publication")
assert receipt["status"] == "voice_published"
assert receipt["finished_unix_ms"] >= receipt["started_unix_ms"]
pins = receipt["published"]
reference = json.loads(Path(pins["reference"]["path"]).read_text())
assert sha(Path(pins["reference"]["path"])) == pins["reference"]["sha256"]
assert sha(Path(pins["prompt"]["path"])) == pins["prompt"]["sha256"]
# Every portable tensor must retain the earlier native preparation bits.
previous = json.loads((root/"voice-preparation-receipts/tone-voice.json").read_text())["voice"]
assert receipt["reference"] == previous["reference"]
def npy(path):
    raw = path.read_bytes()
    length = int.from_bytes(raw[8:10],"little")
    return raw[10+length:]
assert npy(voice/"mels.npy") == b"".join(struct.pack("<f",v) for v in previous["prompt_feat"])
assert npy(voice/"embedding.npy") == b"".join(struct.pack("<f",v) for v in previous["embedding"])
assert json.loads((voice/"t3-tokens.json").read_text()) == previous["t3_tokens"]
assert json.loads((voice/"s3-tokens.json").read_text()) == previous["prompt_tokens"]
native(["publish-voice",model,model_sha,recording,template,template_sha,voice],"overwrite",False)
assert sha(Path(pins["prompt"]["path"])) == pins["prompt"]["sha256"]
bad = output/"wrong-pin"
native(["publish-voice",model,"0"*64,recording,template,template_sha,bad],"wrong-pin",False)
assert not bad.exists()
# Reuse the catalog's existing pipeline. Only the two voice pins change.
old_catalog = json.loads(Path("/home/muttleydosomething/myth-native-service-20261030/service.json").read_text())
old_voice = old_catalog["voices"]["retained-reference"]
old_pipeline = Path(old_voice["pipeline"])
assert sha(old_pipeline) == old_voice["sha256"]
pipeline = json.loads(old_pipeline.read_text())
pipeline["prompt"] = pins["prompt"]
pipeline["reference"] = pins["reference"]
pipeline_path = output/"pipeline.json"
pipeline_path.write_text(json.dumps(pipeline,indent=2)+"\n")
catalog = dict(old_catalog,voices={"synthetic-publication":{
    "display_name":"synthetic publication","pipeline":str(pipeline_path),"sha256":sha(pipeline_path)}})
catalog_path = output/"service.json"
catalog_path.write_text(json.dumps(catalog,indent=2)+"\n")
name = "myth-voice-publication-trial"
port = 8767
launch = ["sudo","-n","docker","run","-d","--name",name,"--runtime","nvidia",
          "--network","host","--read-only","--cap-drop","ALL","--security-opt","no-new-privileges",
          "--pids-limit","64","--memory","1536m","--memory-swap","1536m","--restart","no"]
for mount in [
    "/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro",
    "/lib/ld-linux-aarch64.so.1:/lib/ld-linux-aarch64.so.1:ro",
    "/usr/local/cuda:/usr/local/cuda:ro",
    "/sbin/ldconfig:/sbin/ldconfig:ro",
    str(root)+":"+str(root)+":ro",
    str(template.parent)+":"+str(template.parent)+":ro",
    "/srv/borg-recovery/chatterbox-optimisation-20261029/tensorrt-unified-native-reference/decode.engine:/srv/borg-recovery/chatterbox-optimisation-20261029/tensorrt-unified-native-reference/decode.engine:ro",
]:
    launch += ["-v",mount]
launch += ["--entrypoint",str(binary),image,"serve",str(catalog_path),sha(catalog_path),"10.200.0.2:"+str(port)]
container = subprocess.check_output(launch,text=True).strip()
(output/"deployment.json").write_text(json.dumps(dict(command=launch,container=container,binary_sha256=sha(binary)),indent=2)+"\n")
def request(method,path,value=None):
    client = http.client.HTTPConnection("10.200.0.2",port,timeout=180)
    client.request(method,path,body=None if value is None else json.dumps(value),
                   headers={} if value is None else {"Content-Type":"application/json"})
    response=client.getresponse()
    result=response.status,dict(response.getheaders()),response.read()
    client.close()
    return result
try:
    deadline=time.monotonic()+10
    while True:
        try:
            status,_,body=request("GET","/health")
            assert status==200,body
            break
        except ConnectionError:
            if time.monotonic()>deadline:
                raise
            time.sleep(.05)
    status,_,body=request("GET","/get_predefined_voices")
    assert status==200 and json.loads(body)==[{"filename":"synthetic-publication","display_name":"synthetic publication"}]
    payload=dict(model="myth",input="Ready, Holmes.",voice="synthetic-publication",seed=42,response_format="wav")
    responses=[]
    for label in ["first","replay"]:
        status,headers,wav=request("POST","/v1/audio/speech",payload)
        assert status==200,(status,wav)
        assert headers["content-type"]=="audio/wav"
        assert wav[:4]==b"RIFF" and wav[8:12]==b"WAVE"
        assert len(wav)==int(headers["x-myth-samples"])*4+56
        assert hashlib.sha256(wav).hexdigest()==headers["x-myth-sha256"]
        assert int(headers["x-myth-finished-unix-ms"])>=int(headers["x-myth-started-unix-ms"])
        (output/(label+".wav")).write_bytes(wav)
        (output/(label+"-headers.json")).write_text(json.dumps(headers,indent=2)+"\n")
        responses.append((headers,wav))
    assert responses[0][1]==responses[1][1],"HTTP replay changed"
    assert state()==before
    process=docker("top",name,"-eo","pid,comm")
    assert "chatterbox" in process and "python" not in process.lower(),process
    finished=time.time_ns()//1_000_000
    result=dict(passed=True,started_unix_ms=started,finished_unix_ms=finished,duration_ms=finished-started,
        binary_sha256=sha(binary),publication_ms=receipt["duration_ms"],
        exact_prior_native_reference=True,exact_published_tensors=True,overwrite_rejected=True,
        wrong_pin_rejected=True,http_voice_listed=True,http_synthesis=True,http_replay_exact=True,
        samples=int(responses[0][0]["x-myth-samples"]),waveform_sha256=hashlib.sha256(responses[0][1]).hexdigest(),
        request_ms=[int(h["x-myth-duration-ms"]) for h,_ in responses],
        python_inference=False,retained_voice_preparations=0,new_synthetic_preparations=2,
        synthesis_reference_preparations=0,production_service_before=before,production_service_after=state(),
        production_service_replaced=False,audible_quality_verified=False,process=process)
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
finally:
    # Retain the task container and its evidence, but release its idle resources.
    docker("stop","--time","15",name)
