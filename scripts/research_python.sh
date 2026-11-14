#!/usr/bin/env bash
# Project-local research interpreter and caches; never modifies system Python.
set -euo pipefail
lab_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lab_runtime_root="${QWEN_LAB_RUNTIME_ROOT:-$lab_project_dir/.runtime-research}"
mkdir -p "$lab_runtime_root/tmp" "$lab_runtime_root/inductor" "$lab_runtime_root/triton" "$lab_runtime_root/cuda" \
  "$lab_runtime_root/xdg" "$lab_runtime_root/huggingface" "$lab_runtime_root/torch" "$lab_runtime_root/extensions"
export TMPDIR="$lab_runtime_root/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="$lab_runtime_root/matplotlib"
export NUMBA_CACHE_DIR="$lab_runtime_root/numba"
export XDG_CONFIG_HOME="$lab_runtime_root/config"
export XDG_DATA_HOME="$lab_runtime_root/data"
export XDG_STATE_HOME="$lab_runtime_root/state"
export XDG_CACHE_HOME="$lab_runtime_root/xdg"
export HF_HOME="$lab_runtime_root/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$lab_runtime_root/torch"
export TORCH_EXTENSIONS_DIR="$lab_runtime_root/extensions"
export PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="$lab_runtime_root/inductor"
export TRITON_CACHE_DIR="$lab_runtime_root/triton"
export CUDA_CACHE_PATH="$lab_runtime_root/cuda"
lab_source_root="${QWEN_LAB_SOURCE_ROOT:-$lab_project_dir}"
export PYTHONPATH="$lab_source_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$lab_project_dir/.venv-research/bin/python" "$@"
