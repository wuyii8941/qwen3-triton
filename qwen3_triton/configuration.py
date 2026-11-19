"""Options for the supported Qwen3 compute path."""
from dataclasses import dataclass
from ._model import ComputeOptions

@dataclass(frozen=True)
class OptimizationOptions(ComputeOptions):
    linear_optimized: bool=True
    attention_optimized: bool=True
    loss_optimized: bool=True
    norm_rope_fusion: bool=True
    mlp_fusion: bool=True
    precision_options: tuple=()

def options_for_variant(name):
    if name != 'triton_optimized':
        raise ValueError(f'Unsupported training variant: {name}')
    return OptimizationOptions()
