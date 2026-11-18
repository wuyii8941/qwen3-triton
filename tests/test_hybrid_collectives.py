import torch

from qwen3_triton.runtime.hybrid_collectives import HybridCollectives


class Peer:
    def __init__(self):
        self.calls = []

    def reduce_scatter(self, output, source, op, *, lossless_bf16):
        self.calls.append((output, source, op, lossless_bf16))
        return output

    def report(self):
        return {"backend": "triton-peer"}

    def close(self):
        self.calls.append("closed")


def test_routes_only_large_fp32_reduce_scatter_to_triton(monkeypatch):
    native = []
    monkeypatch.setattr(
        "qwen3_triton.runtime.hybrid_collectives.dist.reduce_scatter_tensor",
        lambda output, source, op, group: native.append((output, source, op, group)),
    )
    peer = Peer()
    collective = HybridCollectives(peer, group="nccl", threshold_bytes=64)
    output = torch.empty(4, dtype=torch.float32)

    small = torch.empty(8, dtype=torch.float32)
    assert collective.reduce_scatter(output, small, "avg") is output
    assert len(native) == 1 and not peer.calls

    large = torch.empty(16, dtype=torch.float32)
    assert collective.reduce_scatter(output, large, "avg") is output
    assert len(native) == 1
    assert peer.calls == [(output, large, "avg", True)]


def test_routes_other_collectives_to_native_group(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "qwen3_triton.runtime.hybrid_collectives.dist.all_gather_into_tensor",
        lambda output, source, group: calls.append(("gather", group)),
    )
    monkeypatch.setattr(
        "qwen3_triton.runtime.hybrid_collectives.dist.all_reduce",
        lambda tensor, op, group: calls.append(("reduce", op, group)),
    )
    monkeypatch.setattr(
        "qwen3_triton.runtime.hybrid_collectives.dist.all_to_all_single",
        lambda output, source, group: calls.append(("alltoall", group)),
    )
    monkeypatch.setattr(
        "qwen3_triton.runtime.hybrid_collectives.dist.barrier",
        lambda group: calls.append(("barrier", group)),
    )
    collective = HybridCollectives(Peer(), group="nccl", threshold_bytes=64)
    source = torch.empty(4)
    output = torch.empty(16)

    assert collective.all_gather(output, source) is output
    assert collective.all_reduce(source, "max") is source
    assert collective.all_to_all(output, source) is output
    collective.barrier()

    assert calls == [
        ("gather", "nccl"),
        ("reduce", torch.distributed.ReduceOp.MAX, "nccl"),
        ("alltoall", "nccl"),
        ("barrier", "nccl"),
    ]


def test_report_makes_native_and_triton_scope_explicit():
    collective = HybridCollectives(Peer(), group="nccl", threshold_bytes=64)
    report = collective.report()
    assert report["mode"] == "hybrid"
    assert report["triton_min_reduce_scatter_bytes"] == 64
    assert report["native_collectives"] == ["all_gather", "small_reduce_scatter", "all_reduce", "all_to_all", "barrier"]
    assert report["triton_collectives"] == ["large_fp32_reduce_scatter"]
    collective.close()
    assert collective.peer.calls == ["closed"]
