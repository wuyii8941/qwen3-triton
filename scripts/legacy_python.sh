#!/usr/bin/env bash
# Run historical modules from their original relative source layout.
set -euo pipefail
lab_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export QWEN_LAB_SOURCE_ROOT="$lab_project_dir/experiments/legacy"
cd -- "$QWEN_LAB_SOURCE_ROOT"
exec bash "$lab_project_dir/scripts/research_python.sh" "$@"
