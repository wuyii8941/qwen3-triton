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


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--communication',choices=['native','separate','hierarchical'],default='native')
    p.add_argument('--communication-minimum-bytes',type=int,default=64*2**20)
    args,remaining=p.parse_known_args()
    if '--help' in remaining or '-h' in remaining:print(p.format_help())
    if args.communication_minimum_bytes<0:p.error('minimum bytes must be nonnegative')
    sys.argv=[sys.argv[0],*remaining]
    root=Path(__file__).resolve().parents[1]
    train(model_setup=lambda model:configure_communication(model,args.communication,minimum_bytes=args.communication_minimum_bytes),
          extra_sources=[Path(__file__).resolve(),root/'training/fsdp/communication_v13.py'])


if __name__=='__main__':main()
