"""PyTorch 2.10 FSDP2 baseline: FP32 masters/reductions, BF16 compute.

No custom collectives. Bottom-up wrapping keeps units comparable to BlockFSDP.
Default native mode retains activations; checkpoint mode trades recomputation
for memory, using PyTorch's supported non-reentrant checkpoint wrapper.
"""
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule


def reshard_official_parameters(model):
    """Restore FP32 DTensor master shards after no-grad retained forward.

    FSDPModule.reshard is not recursive. Visit every unit before enumerating
    parameters for best-state snapshots; temporary BF16 full parameters are
    not a canonical optimizer/master checkpoint representation.
    """
    for unit in model.modules():
        if isinstance(unit,FSDPModule):unit.reshard()


def shard_official_qwen(model, *, checkpoint=False, keep_full_layers=0,
        checkpoint_every=0,checkpoint_mode='full'):
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
    body = model.model
    if not 0<=keep_full_layers<=len(body.layers):
        raise ValueError('keep_full_layers outside decoder layer count')
    from training.fsdp_activation import checkpoint_layer_indices,wrap_decoder_checkpoint
    checkpointed=checkpoint_layer_indices(len(body.layers),all_layers=checkpoint,every=checkpoint_every)
    if checkpoint_mode not in ('full','selective'):raise ValueError('Unknown checkpoint mode')
    if checkpoint_mode=='selective' and not checkpointed:raise ValueError('Selective mode requires checkpointed layers')
    tied = body.embed_tokens.weight is model.lm_head.weight
    # Convert and shard one unit at a time rather than materializing an entire
    # FP32 GPU model. fully_shard moves CPU parameter shards to current CUDA.
    if tied:
        body.embed_tokens.float(); model.lm_head.float()
        fully_shard([body.embed_tokens,model.lm_head],mp_policy=mp,reshard_after_forward=True)
    else:
        fully_shard(body.embed_tokens.float(),mp_policy=mp,reshard_after_forward=True)
        fully_shard(model.lm_head.float(),mp_policy=mp,reshard_after_forward=True)
    for index in range(len(body.layers)):
        layer = body.layers[index].float()
        if index in checkpointed:
            layer=wrap_decoder_checkpoint(layer,mode=checkpoint_mode)
            body.layers[index] = layer
        # Opt-in memory/communication tradeoff: retain the last K decoder
        # units through forward to avoid their backward parameter all-gather.
        retain=index>=len(body.layers)-keep_full_layers
        fully_shard(layer,mp_policy=mp,reshard_after_forward=not retain)
    fully_shard(body.norm.float(),mp_policy=mp,reshard_after_forward=True)
    body.rotary_emb.cuda()
    fully_shard(model,mp_policy=mp,reshard_after_forward=True)
    model.config.use_cache = False
    return model
