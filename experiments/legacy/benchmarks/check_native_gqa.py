"""Reuse identical full-model gates for native GQA (not a Triton speed claim)."""
import json
from pathlib import Path
import sys
import benchmarks.check_tied_embedding as harness
from training.native_gqa import patch_native_gqa


def main():
    harness.patch=patch_native_gqa
    harness.main()
    tag=sys.argv[sys.argv.index('--tag')+1]
    path=Path('artifacts')/tag/'result.json'
    result=json.loads(path.read_text())
    result.update(experiment='native_sdpa_gqa',
                  scope='Real Qwen3 HF projections/norm/RoPE; native SDPA enable_gqa without KV expansion.')
    path.write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
