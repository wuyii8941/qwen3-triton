"""Model optimization switches and ablation presets."""
from dataclasses import dataclass, replace
from ._model import ComputeOptions

@dataclass(frozen=True)
class OptimizationOptions(ComputeOptions):
    linear_optimized: bool=True
    attention_optimized: bool=True
    loss_optimized: bool=True
    norm_rope_fusion: bool=True
    mlp_fusion: bool=True
    precision_options: tuple=()

VARIANTS=('triton_base','triton_optimized','ablate_attention','ablate_norm_rope','ablate_mlp','ablate_linear','ablate_loss')

def options_for_variant(name):
    if name=='triton_base':return ComputeOptions()
    fields={'ablate_attention':'attention_optimized','ablate_norm_rope':'norm_rope_fusion','ablate_mlp':'mlp_fusion','ablate_linear':'linear_optimized','ablate_loss':'loss_optimized'}
    if name=='triton_optimized':return OptimizationOptions()
    if name in fields:return replace(OptimizationOptions(),**{fields[name]:False})
    raise ValueError('Unknown Triton variant: '+str(name))
