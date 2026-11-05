#!/usr/bin/env bash
# Project-local research interpreter and caches; never modifies system Python.
set -euo pipefail
lab_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lab_runtime_root="${QWEN_LAB_RUNTIME_ROOT:-$lab_project_dir/.runtime-research}"
mkdir -p "$lab_runtime_root/tmp" "$lab_runtime_root/inductor" "$lab_runtime_root/triton" "$lab_runtime_root/cuda"
export TMPDIR="$lab_runtime_root/tmp"
export TORCHINDUCTOR_CACHE_DIR="$lab_runtime_root/inductor"
export TRITON_CACHE_DIR="$lab_runtime_root/triton"
export CUDA_CACHE_PATH="$lab_runtime_root/cuda"
export PYTHONPATH="$lab_project_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$lab_project_dir/.venv-research/bin/python" "$@"
