"""BF16 Qwen3-8B training and inference with Triton kernels."""
from .model import PrecisionQwen3ForCausalLM, model_for_variant
TritonQwen3ForCausalLM = PrecisionQwen3ForCausalLM
__all__ = ["TritonQwen3ForCausalLM", "PrecisionQwen3ForCausalLM", "model_for_variant"]
