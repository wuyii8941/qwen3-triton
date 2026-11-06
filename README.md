# Qwen / Triton GPU Kernel 优化实践

## 仓库范围与最新训练实验

新增：[7B/8B 级全分片训练与 HF 导出](MINIMAL_FSDP.md)。真实 DeepSeek-R1-0528-Qwen3-8B 已完成四卡全参数训练、标准 HF 权重导出和重新加载验证；不是随机初始化容量测试。NCCL 负责通信，Triton 负责本地 shard cast/复制/梯度平均。本版本仅覆盖基础实现与部署验证，不包含后续官方 FSDP 性能对照和调优实验。

本项目包含 PyTorch reference、自定义 Triton forward/backward、Tiny-Qwen、真实 HF Qwen3 接入、算子与完整训练 benchmark。它是实验研究代码，不是所有配置都更快的通用训练框架。

- `ops/`：reference 与 Triton 实现；`model/`：模型及 HF 接入。
- `training/`：训练、优化策略和恢复逻辑；`benchmarks/`：独立实验入口；`tests/`：数值与集成验证。
- `submission/`、`artifacts/`、数据、权重、checkpoint、虚拟环境和编译缓存均不发布。文档中历史 `artifacts/...` 链接指本地实验记录，不是仓库内可下载文件。
- 最新训练环境为 RTX A6000、Python 3.11、PyTorch 2.10.0+cu128、Triton 3.6.0、Transformers 4.51.0、FlashOptim 0.1.4。下方 Python 3.12 / Torch 2.5.1 说明属于早期推理实验，结果不可跨环境直接比较。
- 最新固定 16 GiB PyTorch allocator 预算、S512、effective batch=8 实验，在双方均启用 expandable segments 后，baseline 最佳 microbatch=2/accumulation=4，优化路径最佳为 4/2，重复测量吞吐提升约 12.7–13.4%。这是短程 steady-state 对照，不是收敛时间提升承诺。
- 相同数据顺序的 128 次真实数据更新验证中，验证 loss 最大绝对差为 0.000511；这只是质量回归检查，不是新的收敛实验。78 项针对性研究测试通过，并非全仓测试总数。
- owned-gradient 与 selective-checkpoint 原型没有采用：前者比现有优化路径更慢，后者真实模型梯度门槛未通过。负结果保留，不能将各项独立 speedup 相加。

复现前先在独立 Conda/venv 中安装上述匹配依赖及本地 Qwen3-1.7B 权重，不修改系统 Python。最新研究入口使用项目内 `.venv-research`：

```bash
# 从仓库根目录运行；需先创建并配置 .venv-research
bash scripts/research_python.sh -m pytest --collect-only -q
bash scripts/research_python.sh training/flashoptim_convergence_job.py --help
bash scripts/research_python.sh benchmarks/training_dataflow_wave2.py --help
```

脚本将临时文件和编译缓存放到项目 `.runtime-research/`，避免依赖系统 `/tmp` 的剩余容量。许多历史 benchmark 保留了本机模型/数据路径，运行前应查看 `--help` 并调整路径；本仓库不自动提供模型和语料。

以下章节保留历史实验及复现记录；其中“queued”等状态是当时记录，不代表当前任务仍在运行。

从独立算子到真实 Qwen3-1.7B 的推理替换原型。已实现 RMSNorm、split-half RoPE、SwiGLU，保留 PyTorch/Flash SDPA 与 GEMM，提供可恢复的 HF 算子替换 context。

- [优化工作总表：31 项策略、正负结果与证据](OPTIMIZATION_SUMMARY.md)
- [LM head / 输入布局 / 整模型 compile 实测](LM_HEAD_COMPILE.md)
- [Graph 后进一步优化：KV 写入融合与可选 W8A16](DECODE_REFINEMENTS.md)

- [性能证据与测量口径](INFERENCE_EVIDENCE.md)
- [真实模型接入与限制](HF_INFERENCE.md)
- [近期论文筛选与新增融合消融](RESEARCH_OPTIMIZATIONS.md)
- [现有 kernel 内部调优与消融结果](KERNEL_TUNING_RESULTS.md)
- [已验证的端到端优化：QK 融合 + compact GQA](E2E_ATTENTION_OPTIMIZATION.md)
- [进一步的端到端实验：Static Cache / CUDA Graph / packed QKV](STATIC_DECODE_OPTIMIZATION.md)
- `model/hf_inference.py`：Transformers 4.51.0 的 inference-only 兼容 kernel 和替换入口。
- `benchmarks/benchmark_inference_evidence.py`：算子 eager/compile/Triton 及 Qwen 结构 block 对照。
- `benchmarks/benchmark_hf_inference.py`：本地预训练 checkpoint、逐层 hidden states、prefill/增量 decode logits 对齐与计时。
- `benchmarks/benchmark_qk_layout.py`：去除布局复制的消融实验。
- `demo_hf_inference.py`：基线和替换后的自然语言贪心生成对照。

```bash
cd /data1/tzh/proj/qwen_operator_lab
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_hf_inference.py --dtype bfloat16 --profile
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_hf_inference.py --dtype float16
```

当前运行环境：Python 3.12、PyTorch 2.5.1+cu121、Triton 3.1.0、Transformers 4.51.0、RTX A6000。复用独立 `.venv`，不更改宿主 Python。真实模型测试只读取本地权重，不自动下载。

以下保留启动与训练实验说明；训练/GEMM/自写 attention 属于独立实验路径，不代表真实 HF inference benchmark 的默认实现。

## 原始 profiling 入口

This first experiment loads the local Qwen3-1.7B checkpoint, fixes one text
input, records the input/output tensor interface of every Transformer block,
and measures end-to-end forward latency and peak CUDA memory.

This profiling script is a baseline and does not replace operators.
The JSON report becomes the reference for later PyTorch and Triton rewrites.

Run from this directory:

```bash
export PATH=/data1/tzh/envs/cs336_basics/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/data1/tzh/proj/systems/.venv
export UV_NO_SYNC=1
uv run --no-sync python profile_qwen.py
```

The startup project intentionally reuses the systems environment so it does
not install a second copy of PyTorch. `UV_NO_SYNC=1` keeps `uv` from changing
that environment; the required Transformers dependencies have already been
installed there.

The default model is `/data1/tzh/models/Qwen/Qwen3-1.7B`. To use another local
checkpoint:

```bash
uv run --no-sync python profile_qwen.py --model /path/to/model
```

The report is written to `artifacts/qwen_baseline.json`. The script uses CUDA
synchronization around each timed forward pass, performs warmup iterations,
and records both allocated and reserved peak memory when CUDA is available.

## Tiny-Qwen training smoke test

The initial trainable model is in `model/qwen_tiny.py`. It is intentionally a
small Qwen-shaped model, not a replacement for the 1.7B checkpoint. It uses
ordinary PyTorch operators so that the training boundary is working before we
replace anything with custom kernels.

Run it with the same shared environment:

```bash
uv run --no-sync python -m training.train_tiny --steps 20
```

To run the same training smoke test with the Triton RMSNorm path:

```bash
uv run --no-sync python -m training.train_tiny \
  --steps 20 \
  --device cuda \
  --rms-norm-impl triton
```

This trains on a deterministic toy next-token pattern, writes a checkpoint to
`artifacts/tiny_qwen_checkpoint.pt`, and writes the loss curve to the matching
JSON file. The TODOs in `model/qwen_tiny.py` are the Qwen-specific extension
points: position-id/mask/cache support, rope scaling, fused GQA attention, and
fused SwiGLU. The baseline now includes explicit `head_dim`, Q/K RMSNorm, and
Qwen3's split-half RoPE convention. The existing CS336 operators are
deliberately used as baseline knowledge rather than reimplemented here.

## Unit tests

The baseline is runnable before any TODO is filled:

```bash
uv run --no-sync pytest -q
```

The tests progress from RMSNorm and RoPE, to GQA attention, to a complete
forward/loss/backward pass, and finally an optimizer step. Replace one TODO at
a time and rerun the smallest relevant test first.

For the complete implementation roadmap, interfaces, TODOs, and milestones,
see [`PROJECT_PLAN.md`](./PROJECT_PLAN.md).

Paper-inspired long-context research experiments (FASA-style frequency selection
and BLASST-style block skipping), including rejected candidates and rollback
decisions, are documented in [`RESEARCH_OPTIMIZATIONS.md`](./RESEARCH_OPTIMIZATIONS.md).
They use separate opt-in runners; neither changes the default dense/Flash path.
Measured latency improvements that fail the numerical gate are not accepted gains.

The Triton RMSNorm path now includes a Triton backward kernel as well, so the
`--rms-norm-impl triton` training smoke test exercises both forward and
backward custom kernels.

Benchmark the standalone RMSNorm forward paths:

```bash
uv run --no-sync python -m benchmarks.benchmark_rms_norm
```

Benchmark the full Tiny-Qwen training step with PyTorch and Triton RMSNorm:

```bash
uv run --no-sync python -m benchmarks.benchmark_tiny_training
```

## Experimental full-model training optimizer

### Completed full-job comparison

Both jobs completed the frozen validation-plateau rule: native 15750 steps,
Triton 17500 steps. Same-quality observed time was 7208→5912 seconds
(1.219×, 18.0% less time); complete process lifetimes were 8320→7682 seconds
(7.7% less time). This is one sequential BF16 Qwen3-1.7B/WikiText job pair,
not original-model pretraining convergence or a compiled-HF comparison.
Only AdamW is replaced; HF forward/backward remains unchanged.
See [full results, evidence and limitations](TRAINING_CONVERGENCE_RESULTS.md).

Independent follow-up is now queued: three S512 data-order seeds with eager HF,
compiled HF and Triton AdamW, plus one S2048 three-way extension. It has actual
save/resume/numerical preflight gates and per-process compilation-inclusive
timing. This is work in progress, not an additional speedup claim:
[replication plan and monitoring](TRAINING_REPLICATION_PLAN.md).

`benchmarks/run_convergence_comparison.py --tag <new>` runs actual native and
Triton save/resume preflights, validation-noise calibration, then sequential
same-GPU training jobs using one frozen protocol. It is NOT a fixed-1000-step
benchmark: the terminal criterion is the declared validation plateau. Each job
has a six-hour safety deadline, which is reported as **not converged**, never
silently treated as convergence. Shared preprocessing/preflight is excluded
from per-job timing. Shared raw-data preprocessing was not timed, so no
raw-text-to-convergence speedup is claimed; preflight/calibration have logs.
Model/data loading, transfers,
training, validation, durable checkpoint writes and final evaluation are included.

The current comparison is under `artifacts/convergence_comparison_v1/`.
Each phase has a process log; jobs additionally write `events.jsonl`, `last.pt`,
`best.pt` and, ONLY on termination, `summary.json`. The runner produces
`comparison.json` only after both jobs terminate. Absence of that file means
there is no full-job comparison result yet. Raw checkpoints are trusted local
artifacts and must not be loaded from an untrusted source.

```bash
cd /data1/tzh/proj/qwen_operator_lab
tail -n 8 artifacts/convergence_comparison_v1/native/events.jsonl
tail -n 8 artifacts/convergence_comparison_v1/triton/events.jsonl
```

These files do not exist until the corresponding phase actually starts.
Use the running session/process as authoritative for liveness; log inactivity
alone is not permission to restart a job. Actual terminal results for v1 have
now been inspected and independently audited in `verified_result.json`.

Read-only audit (also works while jobs are running):

```bash
PYTHONPATH=. /data1/tzh/proj/systems/.venv/bin/python \
  benchmarks/report_convergence_comparison.py \
  --root artifacts/convergence_comparison_v1
```

The job's `elapsed_seconds` starts inside `main`, after Python imports, and
ends before process teardown. `benchmarks/observe_job_processes.py` is a
separate live observer that includes those process-lifetime costs without
changing the frozen training code. Its event times are observation times,
subject to polling and scheduling delay; events preceding observer attachment
do not receive invented timestamps. The audit distinguishes these clocks,
withholds convergence speedups for safety-limited/incomplete jobs, and reports
missing external timestamps as missing. Checkpoint time is separately logged;
validation time is included in total time but was not separately instrumented.
This first comparison is one sequential job pair, not a repeated-trial estimate.

`training.triton_adamw.TritonAdamW` is an opt-in, single-group CUDA AdamW
prototype. Its default `math_mode="rounded"` passed the actual-update tests;
`legacy` is retained only for historical ablations. It keeps parameter and
moment dtype unchanged but uses FP32 scalar arithmetic rather than reproducing
all native double-hyperparameter operations bit-for-bit.

```python
from training.triton_adamw import TritonAdamW
optimizer = TritonAdamW(model.parameters(), lr=1e-5, math_mode="rounded")
# Ordinary loss.backward(), optimizer.step(), zero_grad(), state_dict() flow.
```

Validated work so far uses BF16 parameters and moments, no FP32 master copy.
No AMP gradient-scaling, AMSGrad, maximize, sparse gradients, CUDA Graph capture,
multiple parameter groups, or distributed compatibility is claimed. Do not
replace an existing mixed-precision/distributed optimizer without validation.
Default HF training/inference entry points are unchanged. Full-model benchmark:

```bash
PYTHONPATH=. /data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_full_training.py \
  --variant hf --optimizer triton --adamw-math rounded --seq 512 --timing steady \
  --output artifacts/adamw_new_run.json
```
