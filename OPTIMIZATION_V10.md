# Qwen3-8B: SM86 optimization experiments

The accepted V9 baseline is GitHub commit
`13dbdf984201a70ca4bba5fb38c72606cd33fbeb` on `main`.
New work lives on `research/ampere-gemm-v10`; the original research workspace
and the qualified V9 kernels are preserved. Operator screening, module checks and
the eight-role whole-step comparison are complete. All numerical, short-quality
and CUDA coverage gates pass. Neither candidate meets the predeclared performance
adoption rule, so V9 remains the default.

The target remains complete Triton model training on official Qwen3-8B,
4×RTX A6000, BF16 compute and FP32 masters/gradients/optimizer state. Keep
the same FSDP2, checkpointing, data, precision and optimizer protocol when
comparing with HF+Flash. The previously published V9 cohort was 1.74–1.87% slower
than HF+Flash under that protocol. The new cohort is reported below.

## Current evidence and priorities

V9 has one optimized GEMM configuration (64×64×32, four warps, three stages)
for all matrix shapes and forward/dX/dW layouts. Disabling optimized Linear
tiling increased whole-step latency by 6.26%/5.16%. Isolated MLP and LM-head
forward-plus-backward remain slower than HF. Those results justify testing
shape/layout-specific configurations before adding more fusion.

The first screen covers five weight shapes × three directions and fused
gate/up/SwiGLU forward: 16 cases using official weights and the hashed real
activations from the accepted full-8B reference capture. Upstreams are seeded
random BF16 tensors, as in the previous isolated module benchmark. This is
not an actual full-model gradient or quality qualification.

Search block sizes, warp counts, software pipeline depth and L2 grouping.
Prune clearly infeasible shared-memory footprints, then compile/measure.
Confirm the three fastest configurations and V9 in both orders. For large-K
cases also test split-K with FP32 scratch and a separate fixed-tree reduction;
include both kernels in timing and report scratch bytes. Preserve BF16 output,
FP32 accumulation, deterministic mode and disabled TF32/reduced-precision
reduction. A raw GEMM error screen is not a replacement for the 399-parameter
training numerical contract.

Raw operator measurements preallocate outputs/workspace. They exclude
allocation, deterministic empty filling, FSDP/NCCL and AdamW. Do not compare
their sum directly with whole-step latency. One selected GPU must be empty
for 120 continuous observations, then remain free of foreign processes.
Other GPUs' owners are recorded and never modified. All worktrees share
the original research admission lock.

## 2026 methods reviewed

- [Helion compile-time heuristics, November 12, 2026](https://pytorch.org/blog/fast-by-default-compile-time-heuristics-for-helion-kernels/):
  resource-aware configuration selection and useful search seeds. Adapt the
  occupancy/shared-memory/warp-count ideas to SM86; do not import the reported
  H100 speedups or replace the pinned environment. Our small search is inspired
  by this approach, not an implementation or reproduction of Helion's IR analysis.
- [Norm/GEMM fusion, July 2026](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/):
  Lazy Pre-Norm and multi-CTA norm fusion are interesting later candidates.
  The reported results use B200; multi-CTA fusion relies on CTA clusters/DSMEM.
  Lazy Pre-Norm is presented for affine-free RMSNorm. Qwen3 has learned norm
  weights and explicit BF16 rounding boundaries, so the algebra cannot simply
  be substituted without changing numerical behavior.
- [Correct but Slow, July 2026](https://arxiv.org/abs/2607.04454):
  use library-relative efficiency as well as correctness to reject slow kernels.
  Native BF16 matmul timings are recorded during the operator screen.
- [Syncopate, OSDI 2026, July camera-ready](https://arxiv.org/html/2601.20595v4):
  chunk-level communication schedules and tile scheduling expose finer overlap
  than separate collective/compute kernels. Its evaluation uses H100/NVLink and
  includes tensor/sequence-parallel GEMM and attention; it does not establish a
  drop-in optimization for our FSDP weight sharding on SM86. This is a follow-up
  candidate motivated by the profiled communication window, not an
  implemented or measured gain. First establish a suitable communication backend
  and the exact parameter-shard dependencies; TMA paths do not apply to A6000.
  Changing native FSDP/NCCL would require a separate controlled comparison.
- [Triton-distributed build guide](https://triton-distributed.readthedocs.io/en/latest/build.html):
  related fine-grained collective/GEMM infrastructure, originating before 2026.
  Its source build replaces the installed Triton and adds NVSHMEM dependencies;
  no such environment change is made during this frozen cohort. Platform/backend
  suitability and integration remain untested here.
- [tritonBLAS, December 2025](https://arxiv.org/abs/2512.04226):
  related resource/topology-aware parameter selection; this is prior-year work,
  not a new 2026 technique.
- [Triton persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
  and [Stream-K, 2023](https://arxiv.org/abs/2301.03598): scheduling/decomposition
  candidates, not newly invented methods. Non-TMA persistent scheduling may be
  tested after tiling; Hopper/Blackwell-only TMA/WGMMA/TMEM paths do not apply
  to A6000. Split-K currently uses a two-kernel deterministic decomposition,
  not a complete Stream-K scheduler.

## Acceptance before adoption

The CPU-only resource screen compiled 135 SM86 configurations across six hot
shapes. All contain Tensor Core and asynchronous-copy instructions. This is
code-generation evidence, not timing, register-spill proof or correctness.
A second audit rejected the hypothesis that rewriting fused gate/up's
accumulation spelling alone eliminates FP32 additions: both forms already
carry their accumulators inside Tensor Core dot operations. The remaining
floating addition in IR belongs to the SiLU denominator. Keep a single
explicit-accumulator negative control, rather than claiming an improvement.

1. Clean monitored operator screen and opposite-order confirmation.
2. Integrated real-8B module forward/backward comparisons with numerical checks.
3. All 399 parameter gradients, actual repeated optimizer updates, full validation
   and no-fallback/actual-kernel evidence under the original training contract.
4. Whole-step comparison against both V9 and HF+Flash in opposite orders with
   memory/resource checks. Preserve negative and contaminated attempts.

Only the final whole-step results can establish an improvement over HF.
Kernel improvements may overlap with communication and need not transfer
one-for-one to training time. The target is measured progress; a failure to beat
HF must be reported without changing precision or comparison protocols.

## Current experiment evidence

The first GPU operator screen completed but was excluded because a foreign
process entered GPU2. Its complete records remain in `gemm_screen_003`, with a
separate resource-rejection receipt. The retry `gemm_screen_004` passed selected
GPU ownership monitoring and completed all 16 cases. Fifteen configurations
passed the opposite-order improvement requirement; all are ordinary tiled GEMM.
Split-K and the explicit-accumulator negative control were not selected.

Two explicit plans were compared:

- Plan 001 applies all 15 selected shape/layout configurations. One configuration
  concerns separate gate/up forward GEMMs and is unused when gate/up fusion is on.
- Plan 002 removes the five attention-only configurations, retaining MLP/LM-head
  tuning. The complete-model comparison did not establish a useful step reduction.

The initial integrated module run (`module_check_001`) passed all seven local VJP
groups and selected-GPU monitoring. Relative to V9, MLP's separate-forward/backward
median sum fell by about 31–32%, and LM head's by about 49%. They remained about
4–7% and 11–13% slower than HF respectively. Attention regressed at layers 17/35,
so a selective plan was added. These are isolated BF16-parameter module results,
not FP32-master FSDP gradient or whole-step claims.

The host dispatcher then changed to check the shape table before querying device
properties and to cache device capabilities. Plans 001/002 were both retested
under this same dispatcher (`module_check_003` / `module_check_002`). The earlier
run's exact source files are archived with its result; its hashes are not changed.
Plan 002 has passed independent checks of 42 timing cases and seven local VJPs.

The whole-step queue has eight fresh four-rank roles: HF, V9, Plan 001, Plan 002,
then the reverse order. Each uses the original 16 quality + 3 warmup + 8 timed
updates, before/after full validation, and all 399 parameter-gradient comparisons.
The original lossless native archive is reused and both fresh HF arms have tighter
native-control thresholds and bitwise forward checks. No dense archives are
regenerated. The trace reviewer additionally verifies GEMM/gate-up launch grids
and warp counts against each plan; an original V9 trace fails the new-plan
geometry contract, as expected. Admission required all four GPUs to be idle.

The unchanged historical `scripts/check_qwen3_triton.py` intentionally verifies
V9 hashes and rejects a modified model source tree. For this branch use
`scripts/check_qwen3_tuned_source.py --plan PLAN --output RESULT`: it verifies the
explicit delta against the published commit, unchanged GEMM/gate-up kernels and
autograd/backward definitions, the unchanged update/timing loop, and all 399
parameter names/shapes in seven full-8B meta variants. This is CPU evidence only.
The root CPU regression also passes 45 tests, with 61 GPU tests skipped.

Controllers and CLI tools:

- `tune_qwen3_ampere_gemm.py`: actual-weight/activation operator screening.
- `build_qwen3_gemm_plan.py`: reject dirty runs and select stable configurations.
- `benchmark_qwen3_tuned_modules.py` / `review_qwen3_tuned_modules.py`: local VJPs,
  HF/V9/candidate timing, numerical arithmetic and ownership checks.
- `run_qwen3_tuned_suite.py`: optional second plan for the eight-role comparison;
  `--dry-run` prints commands without initializing CUDA.
- `review_qwen3_tuned_suite.py`: archived-reference replay, numerical/short-quality
  contracts, timing/memory, and actual CUDA kernel/launch-geometry review.
- `summarize_qwen3_tuned_suite.py`: hash-verified summary and profile-specific
  recommendation under the recorded selection policy, with model defaults unchanged.
- `summarize_qwen3_trace_overlap.py`: reproducible interval-union diagnostics for
  the separate untimed rank-0 forward/backward traces; no transport-cost claim.

All new model paths remain opt-in via `--gemm-plan`. Without a plan the original
configuration choices remain in effect. The source code and comparison results
must be read separately from the accepted V9 reports bundled in this branch.

When the frozen protocol contains relative corpus paths, pass `--input-root`
to bind them to the directory used when the manifest was created. All CLI file
arguments are resolved before changing the worker's directory. The controller
now validates every frozen input byte on CPU before GPU admission and requires
the original native-reference input identity. This preserves the original
manifest and reference archives. A CPU regression checks directory/path binding
before the first CUDA call, and the update/timing-loop AST check still passes.

The first whole-step attempt (`paired_training_001`) stopped in input validation
because its relocated working directory disagreed with the frozen relative
corpus path. It performed no model loading or optimizer updates and supplies no
performance results. Its logs, manifest and changed launcher sources are
preserved. `paired_training_002` restarts all eight roles with explicit input-root
binding; the model math, data identity, quality contract and timing protocol
remain the same.

Before the first measured optimizer update in the restarted cohort, the choice
rule was recorded separately in `selection_policy_001.json`. A candidate needs
passing native/V9 controls and its own numerical, short-quality and CUDA gates,
at least 1% lower median latency than V9 in each order, no more than 0.01 GiB
extra allocated peak, and compliance with the original 41.875 GiB cap. Among
eligible candidates, prefer a median lower than HF in both orders, then the
larger worst-order reduction against V9, then fewer entries for an exact tie.
If neither qualifies, keep V9. These are engineering acceptance rules, not
statistical significance or a global-optimum claim. Operator/module screening
was already available when choosing the two candidate plans.

The restarted cohort completed a clean `hf_r1` (27 updates; all four ranks'
399-parameter strict native controls passed). The first `v9_r1` also completed
all 27 updates and numerical gates, but two non-rank CUDA contexts appeared on
GPU0 during the role. Its entire timing record was rejected under the existing
whole-role resource policy. `recovery_001` preserves those files and keeps the
clean HF prefix unchanged; the remaining seven roles restart in their original
order. This is an interrupted cohort, not an uninterrupted eight-run session.
The original manifest and pre-timing selection policy remain unchanged.

For this cohort the summarizer requires every `recovery_*/manifest.json` via
`--recovery`, verifies the kept/rejected file hashes and remaining role order,
and discloses the recovery in its output. The recovery controller also records
process ancestry after update27 to diagnose any further short CUDA probes;
those extra GPU queries occur after all measured optimizer updates and do not
relax the resource gate.

## Final whole-step results

All eight accepted roles completed, with all 399 parameter comparisons on each
of four ranks passing. Both candidate plans have the same 27 recorded global
losses and before/after validation losses as V9 in both orders. The CPU reviewer
replayed all four original native archives and checked the recorded numerical
arithmetic, short-quality contract, actual full-model CUDA kernel counts and
plan-specific launch geometries. Both fresh HF controls pass the tighter native
thresholds and bitwise forward checks. The earlier resource-rejected V9 attempt
supplies no accepted timing; this remains an interrupted cohort.

| Role | First-order median (s) | Reverse-order median (s) | Reduction against V9, first / reverse |
| --- | ---: | ---: | ---: |
| HF + Flash | 7.017468 | 7.023537 | — |
| V9 | 7.155400 | 7.147384 | — |
| Plan 001: all selected GEMMs | 7.119030 | 7.084660 | 0.508% / 0.878% |
| Plan 002: MLP/head only | 7.153963 | 7.198019 | 0.020% / −0.708% |

Plan 001 remains 1.447% / 0.870% slower than HF. Neither candidate meets the
predeclared 1% reduction against V9 in each order. The selection-rule summarizer
therefore recommends no candidate, and model defaults remain unchanged. These
two orders are descriptive; they do not establish statistical significance or
a global optimum. In the isolated module checks, MLP latency fell about 32% and
LM-head latency about 49% against V9, but this did not yield a large step reduction.

All Triton roles have a 34.140 GiB measured allocated peak. Plan 001's reserved
pool peak is 38.365 GiB versus V9's 40.684 GiB in both orders; this is not a claim
of reduced activation storage. All roles satisfy the 41.875 GiB cap. This work
changes explicit GEMM configurations and their host dispatch, with no additional
checkpointing, FSDP or optimizer changes between the compared arms.

The short-quality gate uses 16 updates and complete 512-block validation before
and after those updates; no convergence or inference-performance claim is made.
Candidate gradient coordinates were compared at runtime; dense candidate vectors
are not archived, so independent replay verifies the reference and recorded gate
arithmetic rather than re-comparing unavailable candidate vectors.

`trace_overlap_diagnostic_003.json` describes the untimed rank-0 forward/backward
profile after update 27. V9's model-kernel interval union is 0.625 s; Plan 001's
is 0.440 s. Their NCCL interval unions are 6.933 s and 6.910 s, respectively,
inside profiled GPU windows of 7.011 s and 6.972 s. Most model computation overlaps
NCCL activity. This supports investigating exposed communication and dependency
waits before expecting more isolated GEMM speedups to yield large whole-step
gains. NCCL kernels can spin on dependencies: these durations do not establish
transport cost or causal attribution. The profile excludes the optimizer and
is separate from the measured step samples. Its categories overlap, and their
durations must not be added to estimate a training step.

Reproduce this diagnostic with the CPU-only invocation below. Hand-calculated
nested/overlapping intervals and an idle gap verify the interval arithmetic;
the resulting real-trace statistics match the initial manual diagnostic exactly.

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/research_python.sh \
  scripts/summarize_qwen3_trace_overlap.py \
  --cohort /data1/tzh/proj/qwen_operator_lab/artifacts/triton_qwen3_v10/paired_training_002 \
  --role v9_r1 --role v10_all_r1 --role v10_r1 \
  --role v9_r2 --role v10_all_r2 --role v10_r2 \
  --destination /data1/tzh/proj/qwen_operator_lab/artifacts/triton_qwen3_v10/trace_overlap_reproduction.json
```

## Checked-in evidence and configuration reuse

- [Evidence index and original-file hashes](results/qwen3_triton_v10_evidence_index_v1.json).
- [Final selection summary](results/qwen3_triton_v10_paired_training_v1.json) and
  [independent training review](results/qwen3_triton_v10_training_review_v1.json).
- [Frozen cohort manifest](results/qwen3_triton_v10_cohort_manifest_v1.json),
  [pre-timing policy](results/qwen3_triton_v10_selection_policy_v1.json) and
  [recovery disclosure](results/qwen3_triton_v10_recovery_v1.json).
- [Raw operator screen](results/qwen3_triton_v10_operator_screen_v1.json) and
  [its ownership monitor](results/qwen3_triton_v10_operator_resource_v1.json).
- [CPU code generation](results/qwen3_triton_v10_cpu_codegen_v1.json) and
  [accumulation-spelling negative control](results/qwen3_triton_v10_accumulation_control_v1.json).
- [All-plan module review](results/qwen3_triton_v10_all_modules_v1.json),
  [selective-plan module review](results/qwen3_triton_v10_selective_modules_v1.json),
  [source/meta checks](results/qwen3_triton_v10_source_meta_v1.json) and
  [trace overlap](results/qwen3_triton_v10_trace_overlap_v1.json).

These are byte-identical copies of the corresponding local evidence. The raw
training logs, traces, rejected attempts, reference gradients and input corpus
remain under the original artifact paths; models, data and caches are not part
of this bundle. Hashes in the reports refer to those original files.

Both plans remain available for explicit experiments:
`--gemm-plan results/qwen3_triton_v10_gemm_plan_001.json` for all selected GEMMs,
or `--gemm-plan results/qwen3_triton_v10_gemm_plan_002.json` for MLP/head only.
Their hashes match the measured plans. Their qualification flags record the
selection-time state; the later numerical and performance decisions are in the
final summary. Missing shapes use the original Triton configuration. These
SM86/BF16 plans were measured at batch 1 per rank and sequence length 512;
they are not a portable performance guarantee for other shapes or GPUs.

The most useful next experiment is to distinguish exposed transfer time from
dependency waits and then test a compatible communication/compute schedule.
Syncopate provides a 2026 design reference, but neither its H100 speedups nor
FSDP integration have been reproduced here. Further isolated GEMM search has
lower priority under this measured profile.
