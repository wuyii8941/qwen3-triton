"""Qwen3-8B communication experiment over the unchanged V12 training loop.

All model arithmetic, gradient comparisons, optimizer updates, data, validation
and timing remain in train_qwen3_vocab_v12.py. New communication configuration
is applied once after FSDP sharding, before optimizer construction.
"""
import argparse
from pathlib import Path
import sys
from training.fsdp.communication_v13 import configure_communication
from train_qwen3_vocab_v12 import main as train
from qwen3_triton.overwrite_allocation import configure as configure_outputs
from overwrite_fsdp_v13 import configure as configure_buffers


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--communication',choices=['native','separate','hierarchical'],default='native')
    p.add_argument('--communication-minimum-bytes',type=int,default=64*2**20)
    p.add_argument('--overwrite-triton-outputs',action='store_true')
    p.add_argument('--overwrite-communication-buffers',action='store_true')
    args,remaining=p.parse_known_args()
    if '--help' in remaining or '-h' in remaining:print(p.format_help())
    if args.communication_minimum_bytes<0:p.error('minimum bytes must be nonnegative')
    sys.argv=[sys.argv[0],*remaining]
    root=Path(__file__).resolve().parents[1]
    configure_outputs(args.overwrite_triton_outputs)
    def setup(model):
        comm=configure_communication(model,args.communication,minimum_bytes=args.communication_minimum_bytes)
        if args.overwrite_communication_buffers:comm=configure_buffers(model,comm)
        return comm
    train(model_setup=setup,extra_sources=[Path(__file__).resolve(),root/'training/fsdp/communication_v13.py',
          root/'scripts/overwrite_fsdp_v13.py',root/'qwen3_triton/overwrite_allocation.py'])


if __name__=='__main__':main()
