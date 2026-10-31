#!/usr/bin/env bash
set -euo pipefail

# Native Myth replaces the inference owner at the established container/port.
# Service Multiplex still starts/stops chatterbox-tts; no proxy or second GPU
# owner is introduced. Python remains an explicit recovery choice only.
case "${MYTH_RUNTIME:-native}" in
  native)
    # Admit precisely the executable, bridge and catalog exercised by the retained
    # token-loop proof. The service independently validates every asset pin.
    sha256sum --check --status <<'PINS'
2eaf1689f02ec64ca317f51c7cae18b1d3b0abd08a46e00eccfca8e5dd4a7119  /home/muttleydosomething/myth-token-loop-20261031/release/chatterbox-rs
e9cd7941e931c63c7e37a7f1ae058bc92b973f161e9aaac8bf57af1513746a8e  /home/muttleydosomething/myth-token-loop-20261031/release/libchatterbox_trt.so
304400bb532ef7a7d1760522284fbbbd440d1f421d8638aaa2122207211fb0d7  /home/muttleydosomething/myth-token-loop-20261031/release/service-production-chatterbox.json
PINS
    exec 'docker' \
      'run' \
      '-d' \
      '--name' \
      'chatterbox-tts' \
      '--runtime' \
      'nvidia' \
      '--network' \
      'host' \
      '--read-only' \
      '--cap-drop' \
      'ALL' \
      '--security-opt' \
      'no-new-privileges' \
      '--pids-limit' \
      '64' \
      '--memory' \
      '4294967296' \
      '--memory-swap' \
      '4294967296' \
      '--restart' \
      'no' \
      '--tmpfs' \
      '/tmp:rw,nosuid,noexec,size=64m' \
      '--mount' \
      'type=bind,src=/usr/lib/aarch64-linux-gnu,dst=/usr/lib/aarch64-linux-gnu,readonly' \
      '--mount' \
      'type=bind,src=/lib/ld-linux-aarch64.so.1,dst=/lib/ld-linux-aarch64.so.1,readonly' \
      '--mount' \
      'type=bind,src=/usr/local/cuda,dst=/usr/local/cuda,readonly' \
      '--mount' \
      'type=bind,src=/sbin/ldconfig,dst=/sbin/ldconfig,readonly' \
      '--mount' \
      'type=bind,src=/home/muttleydosomething/chatterbox-rust-build-20261030,dst=/home/muttleydosomething/chatterbox-rust-build-20261030,readonly' \
      '--mount' \
      'type=bind,src=/var/lib/docker/volumes/chatterbox-outputs/_data/rust-prepared-20261030,dst=/var/lib/docker/volumes/chatterbox-outputs/_data/rust-prepared-20261030,readonly' \
      '--mount' \
      'type=bind,src=/srv/borg-recovery/chatterbox-optimisation-20261029/tensorrt-unified-native-reference/decode.engine,dst=/srv/borg-recovery/chatterbox-optimisation-20261029/tensorrt-unified-native-reference/decode.engine,readonly' \
      '--mount' \
      'type=bind,src=/home/muttleydosomething/myth-token-loop-20261031/release,dst=/myth,readonly' \
      '--mount' \
      'type=bind,src=/home/muttleydosomething/myth-native-stream-20261030/ui,dst=/workshop/ui,readonly' \
      '--mount' \
      'type=bind,src=/home/muttleydosomething/myth-token-loop-20261031/release/state,dst=/home/muttleydosomething/myth-native-import-20261030-ownership-verified/retained-three-owner/state' \
      '--mount' \
      'type=bind,src=/home/muttleydosomething/chatterbox-jetson/patches/native-runtime,dst=/app/patches/native-runtime,readonly' \
      '--mount' \
      'type=bind,src=/home/muttleydosomething/chatterbox-jetson/patches/native-runtime,dst=/home/muttleydosomething/chatterbox-jetson/patches/native-runtime,readonly' \
      '-e' \
      'MYTH_WAVEFORM_BACKEND=cuda' \
      '--user' \
      '2002:2002' \
      '--group-add' \
      '44' \
      '--group-add' \
      '993' \
      'sha256:9506af3b9fd29a504639fdd1ae23f2915a97c3933f85aa7b8791812713d1a505' \
      'serve' \
      '/myth/service-production-chatterbox.json' \
      '304400bb532ef7a7d1760522284fbbbd440d1f421d8638aaa2122207211fb0d7' \
      '0.0.0.0:8004'
    ;;
  python) ;; # Recovery uses the original launcher and retained volumes below.
  *) printf 'MYTH_RUNTIME must be native or python\n' >&2; exit 2 ;;
esac

# Chatterbox TTS for Borg's Jetson runtime. The application source and voices
# remain in the established host location; this launcher is the lifecycle
# authority consumed when the container must be recreated.

chatterbox_root="${CHATTERBOX_ROOT:-/home/muttleydosomething/chatterbox-jetson}"
cublas_host="/usr/local/cuda/targets/aarch64-linux/lib"
cublas_container="/usr/local/cuda/targets/sbsa-linux/lib"
cudnn_host="/lib/aarch64-linux-gnu"
cudnn_container="/usr/lib/aarch64-linux-gnu"

# Opt-in, checksum-pinned trial. Omission selects the original PyTorch model at
# startup; TensorRT runs directly without retaining native transformer weights.
tensorrt_args=()
if [[ -n "${CHATTERBOX_TENSORRT_ENGINE:-}" ]]; then
  : "${CHATTERBOX_TENSORRT_SHA256:?An admitted engine checksum is required}"
  : "${CHATTERBOX_TENSORRT_PYTHON:?Matching Python bindings are required}"
  tensorrt_args=(
    -e CHATTERBOX_TENSORRT_ENGINE=/trt-engine/decode.engine
    -e "CHATTERBOX_TENSORRT_SHA256=${CHATTERBOX_TENSORRT_SHA256}"
    -v "${CHATTERBOX_TENSORRT_ENGINE}:/trt-engine/decode.engine:ro"
    -v "${CHATTERBOX_TENSORRT_PYTHON}:/trt-python:ro"
    -v /usr/lib/aarch64-linux-gnu:/host-trt:ro
    -v /usr/local/cuda/lib64:/host-cuda:ro
  )
fi

docker run -d --name chatterbox-tts \
  --runtime nvidia --network host \
  --restart no \
  -e HF_TOKEN="${HF_TOKEN:-hf_cached}" \
  -e PYTHONPATH=/app/patches \
  -e PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
  "${tensorrt_args[@]}" \
  -v chatterbox-models:/app/hf_cache \
  -v chatterbox-outputs:/app/outputs \
  -v "${chatterbox_root}/config.yaml:/app/config.yaml" \
  -v "${chatterbox_root}/config.py:/app/config.py" \
  -v "${chatterbox_root}/models.py:/app/models.py" \
  -v "${chatterbox_root}/server.py:/app/server.py" \
  -v "${chatterbox_root}/engine.py:/app/engine.py" \
  -v "${chatterbox_root}/voices:/app/voices" \
  -v /mnt/nas/music/_tts:/app/outputs/permanent \
  -v "${chatterbox_root}/patches:/app/patches:ro" \
  -v "${chatterbox_root}/ui/index.html:/app/ui/index.html:ro" \
  -v "${chatterbox_root}/ui/favicon.png:/app/ui/favicon.png:ro" \
  -v "${cublas_host}/libcublas.so.12.6.1.4:${cublas_container}/libcublas.so.12.6.4.1:ro" \
  -v "${cublas_host}/libcublasLt.so.12.6.1.4:${cublas_container}/libcublasLt.so.12.6.4.1:ro" \
  -v "${cublas_host}/libcufft.so.11.2.6.59:${cublas_container}/libcufft.so.11.3.0.4:ro" \
  -v "${cublas_host}/libcufftw.so.11.2.6.59:${cublas_container}/libcufftw.so.11.3.0.4:ro" \
  -v "${cudnn_host}/libcudnn.so.9.3.0:${cudnn_container}/libcudnn.so.9.5.1:ro" \
  -v "${cudnn_host}/libcudnn_adv.so.9.3.0:${cudnn_container}/libcudnn_adv.so.9.5.1:ro" \
  -v "${cudnn_host}/libcudnn_cnn.so.9.3.0:${cudnn_container}/libcudnn_cnn.so.9.5.1:ro" \
  -v "${cudnn_host}/libcudnn_engines_precompiled.so.9.3.0:${cudnn_container}/libcudnn_engines_precompiled.so.9.5.1:ro" \
  -v "${cudnn_host}/libcudnn_engines_runtime_compiled.so.9.3.0:${cudnn_container}/libcudnn_engines_runtime_compiled.so.9.5.1:ro" \
  -v "${cudnn_host}/libcudnn_graph.so.9.3.0:${cudnn_container}/libcudnn_graph.so.9.5.1:ro" \
  -v "${cudnn_host}/libcudnn_heuristic.so.9.3.0:${cudnn_container}/libcudnn_heuristic.so.9.5.1:ro" \
  -v "${cudnn_host}/libcudnn_ops.so.9.3.0:${cudnn_container}/libcudnn_ops.so.9.5.1:ro" \
  chatterbox-tts-jetson
