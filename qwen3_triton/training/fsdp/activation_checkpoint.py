"""Independent layer selection and non-reentrant recomputation policies."""


def checkpoint_layer_indices(layer_count,*,all_layers=False,every=0):
    if layer_count<1 or type(every) is not int or every<0:
        raise ValueError('Positive layer count and nonnegative integer checkpoint interval required')
    if all_layers and every:raise ValueError('Choose all-layer or interval checkpoint, not both')
    return tuple(range(layer_count)) if all_layers else tuple(range(0,layer_count,every)) if every else ()


def canonical_parameter_name(name):
    return name.replace('_checkpoint_wrapped_module.','')


def wrap_decoder_checkpoint(layer,*,mode='full'):
    if mode!='full':raise ValueError('Only final full recomputation policy is supported')
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper,CheckpointImpl
    kwargs=dict(preserve_rng_state=True,checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    return checkpoint_wrapper(layer,**kwargs)
