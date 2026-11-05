from dataclasses import dataclass


@dataclass
class TinyQwenConfig:
    """Small Qwen-shaped configuration for fast local training.

    The defaults intentionally use a small vocabulary and only two layers.
    Change these only after the training smoke test is stable.
    """

    vocab_size: int = 4096
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int | None = None
    max_position_embeddings: int = 1024
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    rms_norm_impl: str = "torch"
    qk_norm_rope_impl: str = "torch"
    attention_impl: str = "sdpa"
    swiglu_impl: str = "torch"
    packed_gate_up: bool = False
    fused_gate_up: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads when head_dim is omitted")
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even number")
        if self.rms_norm_impl not in {"torch", "triton"}:
            raise ValueError("rms_norm_impl must be 'torch' or 'triton'")
        if self.qk_norm_rope_impl not in {"torch", "fused_triton"}:
            raise ValueError("qk_norm_rope_impl must be 'torch' or 'fused_triton'")
        if self.attention_impl not in {"sdpa", "triton"}:
            raise ValueError("attention_impl must be 'sdpa' or 'triton'")
        if self.swiglu_impl not in {"torch", "triton"}:
            raise ValueError("swiglu_impl must be 'torch' or 'triton'")
        if self.fused_gate_up and self.packed_gate_up:
            raise ValueError("fused_gate_up currently requires separate gate/up weights")
