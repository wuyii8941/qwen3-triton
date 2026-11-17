"""Public Qwen3 training model and variant factory."""
from dataclasses import fields, replace
from ._model import TritonQwen3ForCausalLM
from .configuration import OptimizationOptions, options_for_variant
from .components import Backbone, HFOrderNorm

class PrecisionQwen3ForCausalLM(TritonQwen3ForCausalLM):
    backbone_class=Backbone
    def __init__(self,config,options=None,*,precision_options=None):
        if precision_options is None:
            precision_options=dict(xor_sum=True,lse_fma=True,native_delta=True,native_dq=True,sum_fma=True)
        if set(precision_options)!={'xor_sum','lse_fma','native_delta','native_dq','sum_fma'} or any(type(v) is not bool for v in precision_options.values()):
            raise ValueError('Explicit five boolean precision options required')
        assert config.num_hidden_layers==36 and config.hidden_size==4096 and config.head_dim==128
        if options is None:
            options=OptimizationOptions(**{f.name:False for f in fields(OptimizationOptions) if f.name!='precision_options'})
        elif not isinstance(options,OptimizationOptions):
            options=OptimizationOptions(**{f.name:getattr(options,f.name,False) for f in fields(OptimizationOptions) if f.name!='precision_options'})
        options=replace(options,precision_options=tuple(precision_options.items()))
        super().__init__(config,options)

def model_for_variant(config,name,*,precision_options=None):
    return PrecisionQwen3ForCausalLM(config,options_for_variant(name),precision_options=precision_options)
