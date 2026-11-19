from pathlib import Path

import pytest

from qwen3_triton.runtime.dispatch import TritonRuntimeMode


ROOT = Path(__file__).resolve().parents[1]


def test_training_cli_exposes_only_explicit_collective_modes():
    source = (ROOT / "scripts" / "train.py").read_text()
    assert "--collective-mode" in source
    assert "choices=['full','hybrid']" in source
    assert "default='hybrid'" in source
    assert "TrainingRuntime(args.collective_mode" in source


def test_runtime_constructs_a_separate_nccl_group_for_hybrid_mode():
    source = (ROOT / "qwen3_triton" / "runtime" / "manager.py").read_text()
    assert "HybridCollectives" in source
    assert "backend='nccl'" in source
    assert "kind not in ('full','hybrid')" in source
    assert "scatter=320*2**20" in source
    assert "observed={self.observed_extra_bytes}" in source
    assert "gather=4096" in source
    assert "dist.all_reduce(nccl_warm" in source
    assert "del nccl_warm" in source
    assert "peer_buffers_needed_for_validation_all_gather=self.full" in source


def test_training_result_labels_the_selected_runtime():
    source = (ROOT / "scripts" / "train.py").read_text()
    assert "experiment=f'{args.collective_mode}_GPU_runtime'" in source
    assert "runtime=args.collective_mode" in source


def test_runtime_mode_accepts_only_explicit_c10d_collective_schemas():
    allowed = {
        "c10d::barrier",
        "c10d::allreduce_",
        "c10d::_allgather_base_",
        "c10d::_reduce_scatter_base_",
        "c10d::alltoall_base_",
    }
    mode = TritonRuntimeMode(native_collectives=allowed)
    assert mode.native_collectives == frozenset(allowed)
    with pytest.raises(ValueError, match="c10d collective schemas"):
        TritonRuntimeMode(native_collectives={"aten::add"})
    with pytest.raises(ValueError, match="audited c10d collective schemas"):
        TritonRuntimeMode(native_collectives={"c10d::reduce_"})


def test_hybrid_runtime_declares_the_audited_native_collectives():
    source = (ROOT / "qwen3_triton" / "runtime" / "manager.py").read_text()
    for schema in [
        "c10d::barrier",
        "c10d::allreduce_",
        "c10d::_allgather_base_",
        "c10d::_reduce_scatter_base_",
        "c10d::alltoall_base_",
    ]:
        assert schema in source
    assert "TritonRuntimeMode(native_collectives=native_collectives)" in source


def test_external_memory_accounting_subtracts_torch_allocator_growth():
    from qwen3_triton.runtime.manager import external_process_bytes

    assert external_process_bytes(900, 300, 100 * 2**20, 0) == 500 * 2**20
    assert external_process_bytes(300, 300, 0, 0) == 0


def test_phase_peaks_refresh_external_memory_observation():
    source = (ROOT / "qwen3_triton" / "runtime" / "manager.py").read_text()
    assert "def refresh_external_memory" in source
    assert "self.refresh_external_memory()" in source
    assert "observed_external_bytes=self.observed_extra_bytes" in source
