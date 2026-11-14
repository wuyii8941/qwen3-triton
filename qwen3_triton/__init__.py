"""Complete explicit-Triton Qwen3-8B training math, under qualification.

BF16 strict pointwise equivalence is not qualified. See TRITON_QWEN3.md for
full8B evidence, supported inputs and the separate FP32 mathematical proof.
"""
from .model import PrecisionQwen3ForCausalLM, model_for_variant
TritonQwen3ForCausalLM = PrecisionQwen3ForCausalLM
__all__ = ["TritonQwen3ForCausalLM", "PrecisionQwen3ForCausalLM", "model_for_variant"]
