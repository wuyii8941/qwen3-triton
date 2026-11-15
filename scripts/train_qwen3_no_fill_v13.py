"""Diagnostic common HF/Triton baseline with deterministic fill disabled.

Deterministic algorithms stay enabled; model arithmetic, gradients, AdamW,
data, full validation and complete-step timing remain in the V12 trainer.
This is a separate declared protocol, never relabelled as the original
fill=True experiment. Full original numerical/quality gates are still required.
"""
import argparse
from pathlib import Path
import sys
import torch
from train_qwen3_vocab_v12 import main as train
from qwen3_triton.overwrite_allocation import configure as configure_outputs
from training.fsdp.communication_v13 import configure_communication
from overwrite_fsdp_v13 import configure as configure_buffers


def main():
    p=argparse.ArgumentParser(add_help=False,description=__doc__)
    p.add_argument('--overwrite-triton-outputs',action='store_true')
    p.add_argument('--overwrite-communication-buffers',action='store_true')
    a,remaining=p.parse_known_args()
    if '--help' in remaining or '-h' in remaining:print(p.format_help())
    sys.argv=[sys.argv[0],*remaining]
    configure_outputs(a.overwrite_triton_outputs)
    def setup(model):
        # Applied after construction, before any validation, forward/backward,
        # optimizer state allocation or measured update. No global env change.
        assert torch.are_deterministic_algorithms_enabled()
        torch.utils.deterministic.fill_uninitialized_memory=False
        return configure_buffers(model,None) if a.overwrite_communication_buffers else None
    root=Path(__file__).resolve().parents[1]
    train(model_setup=setup,extra_sources=[Path(__file__).resolve(),root/'scripts/overwrite_fsdp_v13.py',
          root/'qwen3_triton/overwrite_allocation.py'])


if __name__=='__main__':main()
