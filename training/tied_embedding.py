"""Opt-in shared embedding/head gradient transport for dense Qwen training.

Contract: positive-length input_ids, shared weight, ordinary dense embedding,
no padding_idx/max_norm/frequency scaling, bias-free LM head. Call the complete
model with input_ids (not inputs_embeds or a separately invoked LM head).
Duplicate rows accumulate in FP32 before casting and adding the head gradient.
The reduction rounding order differs from HF; a model-gradient gate is needed.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _merge(G, IDS, ORDER, DY, OUT, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = tl.load(IDS + row)
    previous = tl.load(IDS + row - 1, row > 0, -1)
    if token != previous:
        cols = tl.arange(0, BLOCK)
        acc = tl.full((BLOCK,), 0, tl.float32)
        j = row
        same = True
        while (j < N) & same:
            original = tl.load(ORDER + j)
            acc += tl.load(DY + original * D + cols, cols < D, 0).to(tl.float32)
            j += 1
            nxt = tl.load(IDS + j, j < N, -1)
            same = nxt == token
        dense = tl.load(G + token * D + cols, cols < D, 0)
        combined = dense.to(tl.float32) + acc.to(dense.dtype).to(tl.float32)
        tl.store(OUT + token * D + cols, combined, cols < D)


class TiedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, ids):
        ctx.save_for_backward(ids)
        return torch.nn.functional.embedding(ids, weight), weight.view_as(weight)

    @staticmethod
    def backward(ctx, grad_embeddings, grad_head):
        ids, = ctx.saved_tensors
        ordered, order = torch.sort(ids.reshape(-1), stable=True)
        dy = grad_embeddings.contiguous().reshape(ids.numel(), -1)
        dense = grad_head.contiguous()
        out = dense.clone()
        _merge[(ids.numel(),)](dense, ordered, order, dy, out,
                              ids.numel(), dy.shape[1], triton.next_power_of_2(dy.shape[1]))
        return out, None


def patch_tied_embedding(model, *, consume_proxy=False):
    """Patch only this model and return a rollback callback. Preserve state_dict."""
    embed, head = model.model.embed_tokens, model.lm_head
    if embed.weight is not head.weight: raise ValueError('Embedding/head weights must be tied')
    if (embed.padding_idx is not None or embed.max_norm is not None or
            embed.scale_grad_by_freq or embed.sparse or head.bias is not None):
        raise ValueError('Experiment supports only plain, dense, bias-free tied embedding/head')
    old_embed, old_head = embed.forward, head.forward
    def embed_forward(ids):
        result, proxy = TiedEmbedding.apply(embed.weight, ids)
        head._experimental_tied_proxy = proxy
        return result
    def head_forward(x):
        proxy=head._experimental_tied_proxy
        if consume_proxy:
            del head._experimental_tied_proxy
        return torch.nn.functional.linear(x, proxy)
    embed.forward, head.forward = embed_forward, head_forward
    def undo():
        embed.forward, head.forward = old_embed, old_head
        if hasattr(head, '_experimental_tied_proxy'):
            del head._experimental_tied_proxy
    return undo
