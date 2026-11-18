"""Reference archive utilities shared by final Qwen3 training commands."""
import hashlib
import json
from pathlib import Path

import torch

from qwen3_triton.gradient_archive import load as load_archive
from training.fsdp.activation_checkpoint import canonical_parameter_name


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def compare_first_gradients(
    parameters, ids, logits, loss, reference_dir, rank, identity, *, strict_control=False
):
    """Compare every coordinate of all 399 local FP32 gradients."""
    reference_dir = Path(reference_dir)
    result = json.loads((reference_dir / "result.json").read_text())
    assert result["input_manifest_sha256"] == identity
    assert result["math_policy"]["allow_bf16_reduced_precision_reduction"] is False
    assert result["variant"] == "hf" and result["layers"] == 36
    path = reference_dir / f"raw_rank{rank}.pt.gz"
    expected = result["ranks"][rank]["reference_archive"]
    assert sha(path) == expected["sha256"]
    raw = load_archive(path)["1"]
    assert torch.equal(ids, raw["ids"])
    assert set(raw["gradients"]) == {canonical_parameter_name(name) for name in parameters}
    rows = []
    error2 = reference2 = 0.0
    for name, parameter in parameters.items():
        name = canonical_parameter_name(name)
        got = local(parameter.grad).detach().cpu()
        reference = raw["gradients"][name]
        assert got.shape == reference.shape
        assert got.dtype == reference.dtype == torch.float32
        assert torch.isfinite(got).all() and torch.isfinite(reference).all()
        delta = float((got - reference).norm())
        norm = float(reference.norm())
        threshold = (0.0001 if strict_control else 0.06) * max(norm, 1e-8) + (
            1e-7 if strict_control else 1e-6
        )
        rows.append(
            dict(
                name=name,
                absolute_error_norm=delta,
                reference_norm=norm,
                relative_l2=delta / max(norm, 1e-8),
                threshold=threshold,
                passed=delta < threshold,
            )
        )
        error2 += delta**2
        reference2 += norm**2
    assert len(rows) == 399
    expected_logits = raw["logits"].float()
    got_logits = logits.detach().cpu().float()
    logit_error = float(
        (got_logits - expected_logits).norm() / expected_logits.norm().clamp_min(1e-8)
    )
    loss_error = abs(float(loss.detach().cpu()) - float(raw["local_loss"]))
    passed = all(row["passed"] for row in rows)
    bitwise = torch.equal(logits.detach().cpu(), raw["logits"]) and torch.equal(
        loss.detach().cpu(), raw["local_loss"]
    )
    forward_passed = logit_error < 0.025 and loss_error < 0.01 and (
        bitwise or not strict_control
    )
    return dict(
        complete=True,
        rows=rows,
        passed=passed,
        global_relative_l2=(error2 / max(reference2, 1e-16)) ** 0.5,
        logits_relative_l2=logit_error,
        loss_abs_error=loss_error,
        forward_gate_passed=forward_passed,
        strict_native_control=strict_control,
        forward_bitwise_equal=bitwise,
        numerical_gate_passed=passed and forward_passed,
        reference_dir=str(reference_dir),
        reference_result_sha256=sha(reference_dir / "result.json"),
        scope=(
            "All-coordinate runtime comparison on first update with explicit "
            "BF16 compute and FP32 accumulation."
        ),
    )
