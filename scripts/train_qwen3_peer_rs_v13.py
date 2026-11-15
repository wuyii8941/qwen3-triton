"""Experimental Qwen3-8B with checked lossless peer transport and FP32 RS.

Original model math, input policy, gradient gates, full validation and complete
optimizer-step timing stay in the V12 trainer. Deterministic fill stays True.
External symmetric storage is conservatively included in the 41.875GiB budget.
"""
from pathlib import Path
from train_qwen3_vocab_v12 import main as train
from fsdp_peer_rs_v13 import configure,EXTERNAL_RESERVE


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    train(model_setup=configure,extra_sources=[Path(__file__).resolve(),root/'scripts/fsdp_peer_rs_v13.py',root/'scripts/symmetric_rs_v13.py'],
          extra_framework=dict(custom_collectives=True,gradient_transport='checked lossless BF16 staging; FP32 reduction and output',
              external_memory_budget_bytes=EXTERNAL_RESERVE,communication_experiment='peer_RS_v13'))
