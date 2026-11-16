# Inference reference accumulation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export an opt-in Triton accumulation path that addresses the original Qwen3-8B short/decode failures, then qualify its complete GPU path and measure its request performance.

**Architecture:** Keep the unmodified inference default and official BF16 weights/activations. An inference-only module selects measured contiguous Split-K3/5/7 with64-aligned boundaries and forward FP32 reduction, interleaved2x32, or explicit sequential32/16-lane GEMV followed by descending warp reduction, for the exact affected8B shapes. Unselected shapes use the existing Triton implementation. No HF/native GPU fallback.

**Tech Stack:** Existing Python3.11, PyTorch2.10.0+cu128, Triton3.6, Transformers4.51, SM86 RTX A6000.

**Spec:** `INFERENCE_V15.md`, especially original qualification gates and complete native linear schedule recollection; user goal to continue optimizing a complete Triton Qwen3-8B inference path.

## Global constraints

- 不要在/home/tzh写入任何东西
- Work only in the existing `research/inference-v15` worktree; preserve the dirty main worktree.
- Preserve all original gates: relativeL2<=0.02, meanAbs<=0.03, maxAbs<=0.5; literal16-token greedy equality; complete phase GPU audit; clean process monitoring.
- Keep production defaultsFalse for all new options; training entries and weights are unchanged.
- Scope qualification to measured shapes/protocols; no padded/ragged/sampling/serving claim.
- Keep caches/tmp/results under `/data1/tzh`; no saved tensors/checkpoints/logits.
- Freeze executed source during GPU runs; bind historical and current evidence explicitly.

## Review focus

- Noncontiguous input/weight views must retain the existing stride-aware fallback.
- Non-power-of-two tail partitions must cover each K coordinate once and initialize every partial/output before read.
- Unsupported shapes and disabled flags must preserve the existing dispatch.
- CUDA graphs must capture the selected route and match their own eager route on changed inputs.
- Audits must reject missing partial/reduction launches and mismatched route counts, even when all cache appends remain.

---

### Prerequisite: Resolve native GEMV arithmetic

- [x] Run028proved the selected SplitK/interleaved calls but retained one cached decode failure.
- [x] Run029identified32-lane K/V and16-lane gate/up sequential accumulator trees with descending warp reduction.
- [x] Run030explicitly reproduced these trees:2520/2520real-layer local calls are bitwise, all24original numerical gates and both literal16-token continuations pass. Exact-source receipts are in `results/qwen3_triton_inference_v15_reference_order_*_v1.json`.
- [ ] Exported API, complete updated phase audits, full fixed validation and paired performance remain open below. Research qualified=False is retained.

### Task 1: Export inference-only arithmetic and configuration

**Files:** Create `qwen3_triton/inference/accumulation.py`; modify inference `kernels.py`, `model.py`, generation CLI; add `scripts/check_reference_accumulation_config_v15.py`.

**Interfaces:** `TritonQwen3Inference(..., reference_accumulation=False)` and `from_pretrained` retain the option; CLI `--reference-accumulation` forwards it. `accumulation.linear(x,weight)` returns a BF16 output for supported contiguous same-device BF16 CUDA shapes, otherwiseNone. `kernels.linear(...,reference_accumulation=False)` checks it before existing paths and preserves diagnostic `accurate` precedence.

- [x] Write/run CPU failing loader/CLI regression: explicitTrue survives loading, defaultFalse, official fixture state tensors unchanged, prefill tiling independent.
- [x] Implement exact routing: M1N4096K4096/12288 interleaved2x32; M17N1024K4096 interleaved2x32; M1N1024K4096 ordered GEMV32 andM1N12288K4096 ordered GEMV16; M17N12288K4096 split3; M65N4096K4096 split3, N1024K4096 split5, N4096K12288 split7; all global partitions aligned64 and reduced in increasing partition order in FP32 before BF16 materialization.
- [x] Use measured GEMM machinery and raw overwrite allocation; preserve native-precision SwiGLU. M17 gate/up use independently reduced linears plus existing SwiGLU; other MLP shapes retain existing path.
- [x] Check contiguous/shape/type guards, partition-tail coverage and literal disabled fallback; run CPU configuration checks.
- [x] Validate selected kernels on actual36-layer HF operands, including positive-stride views; retain discrepancies, do not infer full-model parity from local checks.
- [x] Commit only after fresh checks, with bounded actual-source evidence.

### Task 2: Complete trace audit and unchanged8B gates

**Files:** Modify `scripts/verify_qwen3_inference_v15.py`, `scripts/check_inference_trace_integrity_v15.py`; add bounded results and update `INFERENCE_V15.md`.

**Interfaces:** Verifier `--reference-accumulation` forwards/records the actual model option. Trace reviewer takes that option and explicit prompt length, with expected counts derived for the M65 and M17 routes.

- [x] Write/run failing CPU trace fixtures for the new complete route, missing partials, missing reductions, wrong route and unmatched launch activity.
- [x] Add only actual new Triton kernel names to the whitelist and require all operation counts. M65 prefill has180split+180reduce,36fusedgate/up,1GEMV; B1decode has108interleaved+144orderedGEMV+1existingGEMV. M17 prefill has108ordinaryGEMM,72fusedinterleaved,72split+72reduce,36SwiGLU,1existingGEMV. Other common attention/cache/norm/residual counts remain unchanged.
- [x] Record complete M17 and M65 phase traces with bounded bootstrap collection; verify all launches have GPU events and reject native compute/device memory fallbacks.
- [x] Run unchanged24cases, literal two16-token continuations and full fixed512xS512 corpus with the actual exported API, pinned checkpoint hashes, source freeze, stable lease and continuous monitoring. Preserve earlier failed reports.
- [x] Declare qualification only if every required gate and clean-monitor condition is proven; record exact limits.

### Task 3: Matched whole-request performance and evidence

**Files:** Update the paired graph benchmark, bounded result records, README/INFERENCE usage.

**Interfaces:** Same GPU, B1S65,8outputs, no earlyEOS, two orders,2warmup+5measured repetitions, HF/default/reference arithmetic graph and eager arms; each graph must equal its own eager path.

- [ ] Capture/select options before graph creation and verify three changed inputs.
- [ ] Measure both orders and retain all eager/graph samples; exclude preparation uniformly and report any absent eager improvement.
- [ ] If a tile choice changes the proven FP32 schedule, revalidate local/full-model arithmetic before using its timings. Do not select a faster path that fails original gates.
- [ ] Obtain independent read-only code review, fix concrete defects with regression evidence, and commit exact-source results/documentation. Leave speculative/unqualified alternatives opt-in.

Execution is native in the existing worktree, continuing the user's authorization. Review subagents remain read-only and never use GPUs or mutate executed source.
