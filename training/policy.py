"""Explicit run policy: no historical backend names encoding retention counts."""


def resolve_policy(backend, keep_full_layers=None, checkpoint_every=0,
                   checkpoint_mode='full', *, layer_count=None, tiny=False):
    if backend not in ('official', 'custom'):
        raise ValueError('Backend must be official or custom')
    if type(checkpoint_every) is not int or checkpoint_every < 0:
        raise ValueError('Nonnegative integer checkpoint interval required')
    if checkpoint_mode != 'full':
        raise ValueError('Current runtime supports full activation checkpointing only')
    if keep_full_layers is not None and (
            backend != 'official' or type(keep_full_layers) is not int or keep_full_layers < 0):
        raise ValueError('Explicit retention requires official backend and nonnegative integer')
    if checkpoint_every and backend != 'official':
        raise ValueError('Checkpointing requires official FSDP')
    requested = keep_full_layers or 0
    if layer_count is not None and requested > layer_count and not tiny:
        raise ValueError('Retention exceeds actual decoder layer count')
    effective = min(requested, layer_count) if layer_count is not None else requested
    return dict(requested_keep_full_layers=requested, keep_full_layers=effective,
                checkpoint_every=checkpoint_every, checkpoint_mode=checkpoint_mode)


def validate_exchange(backend, exchange, protocol):
    if exchange not in ('native', 'torch_decoder'):
        raise ValueError('Unknown gradient transport')
    if exchange != 'native' and (
            backend != 'official' or protocol.get('compute_dtype') != 'bfloat16'
            or protocol.get('master_gradient_optimizer_dtype') != 'float32'
            or protocol.get('global_batch') != protocol.get('world_size', 0) * protocol.get('microbatch_per_rank', 0)):
        raise ValueError('Exchange requires official FSDP, BF16 contributions, FP32 reduction and accumulation=1')
