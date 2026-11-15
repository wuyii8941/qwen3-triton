"""Explicit model-graph and FSDP integration for experimental vocabulary CE.

This changes the graph, collective schedule and parameter ownership. It is
not a standard FSDP switch. Decoder FSDP and native fused AdamW are shared.
The head's FP32 master is already vocabulary-sharded and its gradient is
already data averaged; root FSDP must ignore it.
"""
import torch
from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from training.fsdp.activation_checkpoint import checkpoint_layer_indices,wrap_decoder_checkpoint
from qwen3_triton.model import PrecisionQwen3ForCausalLM
from qwen3_triton.vocab_parallel import VocabParallelCE


class _LossOnlyGraph:
    def forward(self,input_ids,labels=None,*,position_ids=None,use_cache=False,attention_mask=None,
                past_key_values=None,output_attentions=False,output_hidden_states=False,return_dict=True,
                logits_to_keep=0,diagnostic_logits=False):
        if (labels is None or use_cache or attention_mask is not None or past_key_values is not None
                or output_attentions or output_hidden_states or logits_to_keep):
            raise ValueError('Vocabulary CE requires dense unpadded labeled loss-only training/evaluation')
        if isinstance(self,PrecisionQwen3ForCausalLM):hidden=self.model(input_ids,position_ids)
        else:hidden=self.model(input_ids=input_ids,position_ids=position_ids,use_cache=False,return_dict=True).last_hidden_state
        loss,logits=self.lm_head(hidden,labels,diagnostics=diagnostic_logits)
        if not return_dict:return (loss,logits)
        return CausalLMOutputWithPast(loss=loss,logits=logits)


class VocabLossTritonQwen3(_LossOnlyGraph,PrecisionQwen3ForCausalLM):pass
class VocabLossHFQwen3(_LossOnlyGraph,Qwen3ForCausalLM):pass


def shard_vocab_loss_model(model,*,keep_full_layers,checkpoint_every,skip_embedding_backward_gather):
    if model.config.tie_word_embeddings:raise ValueError('Untied head required')
    if not 0<=keep_full_layers<=len(model.model.layers):raise ValueError('Invalid retention count')
    if isinstance(model,PrecisionQwen3ForCausalLM):model.__class__=VocabLossTritonQwen3
    elif isinstance(model,Qwen3ForCausalLM):model.__class__=VocabLossHFQwen3
    else:raise TypeError('Supported Qwen3 graph required')
    model.lm_head=VocabParallelCE(model.lm_head.weight.detach().cpu())
    body=model.model;mp=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
    fully_shard(body.embed_tokens.float(),mp_policy=mp,reshard_after_forward=True)
    if skip_embedding_backward_gather:body.embed_tokens.set_unshard_in_backward(False)
    checkpointed=checkpoint_layer_indices(len(body.layers),every=checkpoint_every)
    for index,layer in enumerate(body.layers):
        layer=layer.float()
        if index in checkpointed:layer=wrap_decoder_checkpoint(layer,mode='full')
        body.layers[index]=layer
        retain=index>=len(body.layers)-keep_full_layers
        fully_shard(layer,mp_policy=mp,reshard_after_forward=not retain)
    fully_shard(body.norm.float(),mp_policy=mp,reshard_after_forward=True)
    body.rotary_emb.cuda()
    fully_shard(model,mp_policy=mp,reshard_after_forward=True,ignored_params={model.lm_head.weight})
    model.config.use_cache=False
    return model
