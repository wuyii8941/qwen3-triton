"""Independent layer selection and non-reentrant recomputation policies."""
from functools import partial


def checkpoint_layer_indices(layer_count,*,all_layers=False,every=0):
    if layer_count<1 or type(every) is not int or every<0:
        raise ValueError('Positive layer count and nonnegative integer checkpoint interval required')
    if all_layers and every:raise ValueError('Choose all-layer or interval checkpoint, not both')
    return tuple(range(layer_count)) if all_layers else tuple(range(0,layer_count,every)) if every else ()


def canonical_parameter_name(name):
    return name.replace('_checkpoint_wrapped_module.','')


def wrap_decoder_checkpoint(layer,*,mode='full'):
    if mode not in ('full','selective'):raise ValueError('Unknown recomputation policy')
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper,CheckpointImpl
    kwargs=dict(preserve_rng_state=True,checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    if mode=='selective':
        from torch.utils.checkpoint import create_selective_checkpoint_contexts
        from training.selective_checkpoint import policy
        kwargs['context_fn']=partial(create_selective_checkpoint_contexts,policy)
    return checkpoint_wrapper(layer,**kwargs)
