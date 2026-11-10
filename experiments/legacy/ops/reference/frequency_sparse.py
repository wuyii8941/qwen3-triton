"""FASA-inspired decode references; not the authors' implementation.

Paper: https://arxiv.org/html/2602.03152v3, Eq. 4 / Algorithm 1.
Qwen3's frequency chunk is (i, i + D/2), NOT adjacent coordinates.
All functions here are offline FP32 oracles, never timed as fast kernels.
Q: [B,Hq,D], K/V: [B,T,Hkv,D], indices: [B,Hq,budget].
"""
import torch


def expanded_keys(q, k):
    if q.ndim != 3 or k.ndim != 4 or q.shape[0] != k.shape[0]:
        raise ValueError('Expected Q [B,Hq,D] and K [B,T,Hkv,D]')
    if q.shape[-1] != k.shape[-1] or q.shape[-1] % 2 or q.shape[1] % k.shape[2]:
        raise ValueError('Even head_dim and divisible GQA head counts required')
    heads = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k.shape[2])
    # Offline oracle ONLY: the Triton implementation does not expand full KV.
    return k[:, :, heads].permute(0, 2, 1, 3).float()


def dense_scores(q, k):
    return (q.float().unsqueeze(-2) * expanded_keys(q, k)).sum(-1)


def contextual_agreement(q, k, top_k=256):
    """Return per-query-head, per-frequency top-k overlap averaged over batch."""
    d = q.shape[-1] // 2
    keys = expanded_keys(q, k)
    q = q.float()
    parts = (keys[..., :d] * q[..., None, :d]
             + keys[..., d:] * q[..., None, d:]).transpose(-1, -2)
    top_k = min(top_k, k.shape[1])
    full_indices = parts.sum(-2).topk(top_k, -1).indices
    membership = torch.zeros_like(parts[..., 0, :], dtype=torch.bool)
    membership.scatter_(-1, full_indices, True)
    partial_indices = parts.topk(top_k, -1).indices
    return membership.unsqueeze(-2).expand_as(parts).gather(
        -1, partial_indices).float().mean(-1).mean(0)


def proxy_scores(q, k, pairs):
    """pairs [Hq,F]; use BOTH coordinates of every selected frequency chunk."""
    d = q.shape[-1] // 2
    coords = torch.cat((pairs, pairs + d), -1).long()
    selected_q = q.float().gather(-1, coords[None].expand(q.shape[0], -1, -1))
    keys = expanded_keys(q, k)
    selected_k = keys.gather(-1, coords[None, :, None].expand(
        keys.shape[0], -1, keys.shape[2], -1))
    return (selected_k * selected_q.unsqueeze(-2)).sum(-1)


def selected_attention(q, k, v, indices, scale=None):
    """Recompute FULL-DIM scores, then softmax on the selected subset."""
    keys = expanded_keys(q, k)
    values = expanded_keys(q, v)
    gather = indices.long().unsqueeze(-1).expand(-1, -1, -1, q.shape[-1])
    keys, values = keys.gather(-2, gather), values.gather(-2, gather)
    scale = q.shape[-1] ** -.5 if scale is None else scale
    scores = (q.float().unsqueeze(-2) * keys).sum(-1) * scale
    return (scores.softmax(-1).unsqueeze(-1) * values).sum(-2)


def selection_metrics(q, k, v, indices, dense_output, scale=None):
    scale = q.shape[-1] ** -.5 if scale is None else scale
    scores = dense_scores(q, k)
    probs = (scores * scale).softmax(-1)
    retained = probs.gather(-1, indices).sum(-1)
    oracle_idx = scores.topk(indices.shape[-1], -1).indices
    membership = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, oracle_idx, True)
    recall = membership.gather(-1, indices).float().mean(-1)
    sparse = selected_attention(q, k, v, indices, scale)
    rel = (sparse - dense_output.float()).norm() / dense_output.float().norm().clamp_min(1e-12)
    return dict(retained_mass_mean=float(retained.mean()),
                retained_mass_min=float(retained.min()),
                topk_recall=float(recall.mean()), output_relative_l2=float(rel))
