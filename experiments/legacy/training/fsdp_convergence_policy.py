"""Explicit policy identity shared by long-run workers and their launcher."""

def resolve_policy(backend, keep_full_layers=None, checkpoint_every=0,
                   checkpoint_mode='full', *, layer_count=None, tiny=False):
    if type(checkpoint_every) is not int or checkpoint_every < 0:
        raise ValueError('Nonnegative integer checkpoint interval required')
    if checkpoint_mode not in ('full', 'selective'):
        raise ValueError('Unknown checkpoint mode')
    if checkpoint_mode == 'selective' and not checkpoint_every:
        raise ValueError('Selective mode requires checkpointed layers')
    if keep_full_layers is not None and (
            backend != 'official' or type(keep_full_layers) is not int or keep_full_layers < 0):
        raise ValueError('Explicit retention requires official backend and nonnegative integer')
    if checkpoint_every and not backend.startswith('official'):
        raise ValueError('Checkpointing requires official FSDP')
    requested = {'official_keep8': 8, 'official_keep24': 24}.get(backend, 0)
    if keep_full_layers is not None:
        requested = keep_full_layers
    effective = requested
    if layer_count is not None:
        if requested > layer_count and keep_full_layers is not None and not tiny:
            raise ValueError('Explicit retention exceeds actual decoder layer count')
        effective = min(requested, layer_count)
    return dict(requested_keep_full_layers=requested, keep_full_layers=effective,
                checkpoint_every=checkpoint_every, checkpoint_mode=checkpoint_mode)
