#!/usr/bin/env bash
# Build only the bridge; never loads a model or submits CUDA work.
set -euo pipefail
[[ $# == 1 && "$1" == /* ]] || { printf 'Usage: bash native/build.sh /absolute/path/libchatterbox_trt.so\n' >&2; exit 2; }
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cuda_root="${CUDA_HOME:-/usr/local/cuda}"
exec "${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -Wall -Wextra \
  -I"${cuda_root}/include" "${source_dir}/tensorrt.cpp" \
  -L"${cuda_root}/lib64" -Wl,-rpath,"${cuda_root}/lib64" \
  -lnvinfer -lnvinfer_plugin -lcudart -pthread -o "$1"
