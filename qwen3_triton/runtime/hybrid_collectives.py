"""Topology-aware hybrid collective routing for the four-GPU training experiment.

Only large FP32 FSDP reduce-scatter payloads use the checked Triton peer
transport. NCCL handles all-gather and latency-sensitive/small collectives.
This preserves the historical winning scope instead of claiming that every
collective benefits from a Triton implementation.
"""
import torch
import torch.distributed as dist


_OPS = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
}


class HybridCollectives:
    def __init__(self, peer, group, *, threshold_bytes=64 * 2**20):
        if threshold_bytes <= 0:
            raise ValueError("A positive Triton reduce-scatter threshold is required")
        self.peer = peer
        self.group = group
        self.threshold_bytes = threshold_bytes
        self.calls = {
            "native_all_gather": 0,
            "native_reduce_scatter": 0,
            "triton_reduce_scatter": 0,
            "native_all_reduce": 0,
            "native_all_to_all": 0,
            "native_barrier": 0,
        }

    def all_gather(self, output, source):
        dist.all_gather_into_tensor(output, source, group=self.group)
        self.calls["native_all_gather"] += 1
        return output

    def reduce_scatter(self, output, source, op="avg", *, lossless_bf16=False):
        if op not in _OPS:
            raise ValueError(f"Unsupported reduction: {op}")
        source_bytes = source.numel() * source.element_size()
        use_peer = (
            source.dtype == torch.float32
            and output.dtype == torch.float32
            and source.numel() == 4 * output.numel()
            and source_bytes >= self.threshold_bytes
        )
        if use_peer:
            self.peer.reduce_scatter(output, source, op, lossless_bf16=True)
            self.calls["triton_reduce_scatter"] += 1
        else:
            dist.reduce_scatter_tensor(output, source, op=_OPS[op], group=self.group)
            self.calls["native_reduce_scatter"] += 1
        return output

    def all_reduce(self, tensor, op="sum"):
        if op not in _OPS:
            raise ValueError(f"Unsupported reduction: {op}")
        dist.all_reduce(tensor, op=_OPS[op], group=self.group)
        self.calls["native_all_reduce"] += 1
        return tensor

    def all_to_all(self, output, source):
        dist.all_to_all_single(output, source, group=self.group)
        self.calls["native_all_to_all"] += 1
        return output

    def barrier(self):
        dist.barrier(group=self.group)
        self.calls["native_barrier"] += 1

    def close(self):
        self.peer.close()

    def report(self):
        return {
            "mode": "hybrid",
            "triton_min_reduce_scatter_bytes": self.threshold_bytes,
            "native_collectives": [
                "all_gather",
                "small_reduce_scatter",
                "all_reduce",
                "all_to_all",
                "barrier",
            ],
            "triton_collectives": ["large_fp32_reduce_scatter"],
            "calls": dict(self.calls),
            "peer": self.peer.report(),
        }
