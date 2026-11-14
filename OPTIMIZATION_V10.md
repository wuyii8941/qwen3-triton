# Qwen3-8B: SM86 optimization experiments

The accepted V9 baseline is GitHub commit
`13dbdf984201a70ca4bba5fb38c72606cd33fbeb` on `main`.
New work lives on `research/ampere-gemm-v10`; the original research workspace
and the qualified V9 kernels are preserved. Operator and module screening has
completed; whole-step qualification is still pending. No new default is adopted.

The target remains complete Triton model training on official Qwen3-8B,
4×RTX A6000, BF16 compute and FP32 masters/gradients/optimizer state. Keep
the same FSDP2, checkpointing, data, precision and optimizer protocol when
comparing with HF+Flash. The V9 optimized path remains 1.74–1.87% slower
than HF+Flash under that protocol.

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

Two explicit plans are being compared:

- Plan 001 applies all 15 selected shape/layout configurations. One configuration
  concerns separate gate/up forward GEMMs and is unused when gate/up fusion is on.
- Plan 002 removes the five attention-only configurations, retaining MLP/LM-head
  tuning. It is a selection candidate, not a proven whole-step winner.

The initial integrated module run (`module_check_001`) passed all seven local VJP
groups and selected-GPU monitoring. Relative to V9, MLP's separate-forward/backward
median sum fell by about 31–32%, and LM head's by about 49%. They remained about
4–7% and 11–13% slower than HF respectively. Attention regressed at layers 17/35,
so a selective plan was added. These are isolated BF16-parameter module results,
not FP32-master FSDP gradient or whole-step claims.

The host dispatcher then changed to check the shape table before querying device
properties and to cache device capabilities. Plans 001/002 are both retested
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
geometry contract, as expected. The queue waits for all four GPUs to be idle.

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

All new model paths remain opt-in via `--gemm-plan`. Without a plan the original
configuration choices remain in effect. The source code and comparison results
must be read separately from the accepted V9 reports bundled in this branch.
