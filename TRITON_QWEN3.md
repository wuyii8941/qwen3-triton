# Complete Triton Qwen3-8B: training implementation and results

The target is official **Qwen/Qwen3-8B**, revision
`b968826d9c46dd6066d109eabc6255188de91218`, with all 36 decoder layers and
399 parameter tensors. All current GPU qualification uses this full model.
Historical toy/1.7B runs are debugging records. The historical tokenizer path
in the frozen input manifest is provenance: all three token streams have
been independently replayed byte for byte with the official 8B tokenizer.

The complete model computation is implemented and passes the recorded
numerical and short-training contracts. The optimized Triton version reduces
whole-step latency against the complete Triton baseline by **3.96% / 4.25%**
in two opposite orders. It remains **1.87% / 1.74% slower than HF + Flash**.
These measurements support a complete trainable implementation and useful
operator improvements; they do not establish an overall speed advantage over HF.

## Compute boundary

`qwen3_triton/model.py` and its kernels implement embedding; every Q/K/V/O,
gate/up/down and LM-head Linear; RMSNorm/QK-Norm/RoPE; causal GQA attention;
SwiGLU; residual addition; causal cross entropy; and their backward computations.
Official checkpoint parameter names and shapes are preserved.

PyTorch supplies parameter/tensor management, autograd scheduling and gradient
accumulation, allocation and layout copies. Native FSDP2/NCCL, full-layer
activation checkpointing and native fused AdamW are common to every arm.
Their settings are held equal and are not credited as operator gains.
All ranks run the no-fallback dispatcher guard during untimed training and
trace execution. Each role also captures an actual rank0 CUDA training trace.
The independent review checks expected Triton kernels and rejects hidden
cuBLAS/Flash model computation and unclassified kernels in Triton variants.

The supported path is dense, unpadded causal training. KV caching/generation,
padding masks, attention dropout, sliding windows, scaled RoPE, attention bias
and extra hidden/attention outputs fail explicitly. These results establish
neither inference speed nor training convergence.

## Variants

| Argument | Model computation |
| --- | --- |
| `hf` | Official HF Qwen3 with forced Flash SDPA |
| `triton_base` | Complete Triton model with baseline tiling and separate operations |
| `triton_optimized` | Candidate tiling, attention stages, Norm/RoPE and gate/up/SwiGLU fusion |
| `ablate_attention` | Optimized model with baseline attention pipeline stages |
| `ablate_norm_rope` | Optimized model with separate norm and rotary operations |
| `ablate_mlp` | Optimized model with separate gate/up and SwiGLU |
| `ablate_linear` | Optimized model with baseline GEMM tiling |
| `ablate_loss` | Optimized model with baseline loss tiling |

Every Triton ablation retains Triton computation. The separate `fusion_only`
control is available in code but is outside the sixteen-role comparison.
The V9 precision choices use native-order norm/attention reductions, accurate
SiLU and explicitly selected BF16 cast/reduction/FMA choices. Attention-stage
ablation preserves those numerical choices. Source relocation and identities
are recorded in `results/qwen3_triton_source_port_v2.json` and the CPU/AST reports.

## End-to-end measurements

Four RTX A6000 GPUs; batch 1/rank, sequence 512, global batch 4; BF16 model
compute and FP32 masters/gradients/moments; TF32 and BF16 reduced-precision
GEMM reduction disabled. All arms use full checkpointing every layer,
keep-full-layers 0, native FSDP2/NCCL and native fused AdamW. Vocabulary
parallelism and custom collectives are disabled. The allocator cap is
41.875 GiB per GPU, with deterministic algorithms and deterministic empty fill.

Each fresh role performs 16 quality updates, full validation before/after16,
3 warmups and 8 measured complete optimizer steps. Timing includes
zero-grad, forward, backward, AdamW and CUDA synchronization, taking the
maximum rank elapsed time. Input copying, barriers, reference comparison,
validation, compilation and the separate trace are excluded.

| Version | Median step seconds, order 1 / 2 | Peak allocated / reserved GiB |
| --- | ---: | ---: |
| HF + Flash | 7.0273 / 7.0304 | 34.202 / 40.748 |
| Complete Triton base | 7.4540 / 7.4705 | 34.139 / 40.684 |
| Complete Triton optimized | 7.1585 / 7.1529 | 34.140 / 40.684 |

Memory values are maxima across ranks during the measured steps and are the
same in both orders for these three arms. Whole-training memory is effectively
unchanged. Opposite-order results are descriptive, not statistical confidence
intervals.

The main independent review passes all sixteen roles:
`artifacts/triton_qwen3_v1/official_8b_paired_quality_ablation_v9_control/independent_review.json`.
The portable curated summary is `results/qwen3_triton_v9_paired_training_v1.json`.
The final data/runtime audit is `results/qwen3_triton_v9_data_acceptance_v1.json`;
it completes the agent-audit field that was pending when the immutable paired
summary was first created.

## Ablation and method selection

Positive values below mean disabling the choice makes the full step slower.

| Disabled choice | Step latency change, order 1 / 2 |
| --- | ---: |
| Attention stages | +0.803% / -0.083% |
| Norm/RoPE fusion | -0.230% / +0.021% |
| MLP fusion | +0.027% / -0.159% |
| Linear tiling | **+6.256% / +5.164%** |
| Loss tiling | -0.024% / +0.078% |

**Linear/GEMM tiling has the clearest whole-step support in this workload.**
Its disabled samples are slower than the enabled samples in both orders.
The other four choices have small effects with inconsistent direction here;
they retain explicit switches and are not advertised as established whole-step
gains. These individual ablations do not prove a globally optimal combination.

## Isolated modules

Fresh activations are captured from a complete official 8B HF forward.
Ninety-six actual cases cover layers 0/17/35 and both orders, with five warmups
and twenty CUDA-event samples per forward/backward part. Every Triton case
first executes an untimed guarded finite VJP. Local parameter gradients are
BF16; FSDP/NCCL/optimizer are excluded.

| Module | Median per-case latency reduction vs HF |
| --- | ---: |
| Full self-attention | +15.06% |
| QK-Norm/RoPE | +59.35% |
| MLP | -53.93% |
| LM head | -120.46% |
| Causal cross entropy | +86.80% |

Negative reductions mean slower execution. Each case uses the sum of its
separate forward/backward medians. Full self-attention includes Q/K/V/O
projections, QK-Norm/RoPE and causal GQA; the separate Norm/RoPE row overlaps.
These elapsed times and savings are not additive.
See `results/qwen3_triton_v9_module_benchmark_v1.json`.

One profiled rank0 forward/backward after27updates gives model-kernel interval
unions of 1.315 s for base and 0.626 s for optimized, while first-to-last GPU
kernel spans are 7.304 s and 7.010 s. NCCL and model intervals overlap heavily.
This diagnostic excludes AdamW and does not identify communication bandwidth,
a critical path or a whole-step speedup bound:
`results/qwen3_triton_v9_profile_screen_v1.json`.
MLP and LM-head GEMMs remain concrete targets for further operator work.

## Numerical and training qualification

All roles complete 27 actual updates with finite FP32 master, gradient and
AdamW moment tensors for every one of the 399 parameters on each rank.
There are 25,536 first-update parameter-shard comparisons across sixteen roles
and four ranks. Candidate comparisons use the unchanged per-shard contract:

`L2_error < 0.06 * max(reference_L2, 1e-8) + 1e-6`.

Native controls use `0.0001 * max(reference_L2, 1e-8) + 1e-7` and bitwise
forward checks. The worst rank-wide gradient relative L2 is 0.007913;
this is distinct from the per-parameter gate and does not claim bitwise
candidate gradients. Initial logits have zero relative L2 error; the largest
initial loss absolute error is 4.768e-7.

Full validation covers all 512 complete blocks and 261,632 shifted tokens
before and after16updates, batch 4/rank. The largest validation loss drift is
9.601e-5. The largest mean training loss drift is 9.173e-4 and the largest
single-step drift is 0.002122. All pass the frozen short-quality thresholds:
mean training drift <0.002, maximum <0.01 and validation drift <0.002.

Every coordinate of every candidate first gradient is compared at runtime.
Only the native gradients are losslessly archived. Independent CPU review
replays those reference archives and checks recorded candidate gate arithmetic;
dense candidate vectors are not retained or independently replayed.
The separate sixteen-role report, CUDA traces and source hashes establish
the stated scope. Sixteen quality updates do not establish convergence.

## Resource and provenance record

The cohort uses the original shared four-GPU lease, 120 continuously idle
observations before admission and strict global ownership monitoring.
The initial reverse-order MLP attempt observed a foreign GPU3 context in
70 samples and was rejected for performance. Its original files remain
byte-identical under the cohort's `recovery_001/rejected_attempt/`.

The first ten clean roles were retained with bound hashes; the rejected MLP
role and remaining five roles ran in their original relative order with
unchanged source/protocol. The final sixteen accepted roles all pass resource
monitoring. This is an explicitly interrupted cohort. Final summaries require
and bind the recovery manifest, and rejected timings remain excluded.
No historical V5/tiny/1.7B timing is mixed into this cohort.

Historical gradient failures, artifact-source controls, earlier exports and
their individual source identities remain preserved in the local evidence.
Current public-entry results supersede their qualification status; old
artifact records are not relabeled as new public GPU experiments.

## Reproduction and source export

Use `scripts/train_qwen3_triton_8b.py` and
`scripts/run_qwen3_triton_suite.py` with verified official 8B inputs.
[QWEN3_TRITON_REPRODUCE.md](QWEN3_TRITON_REPRODUCE.md) provides the pinned
environment, corpus replay, three-arm/ablation, module and review commands.

The 75-file V3 review export passes eleven CLI checks, all seven full8B
CPU/meta variants and physical import checks without original artifact
modules. This uses the existing pinned interpreter; a fresh installation or
exported-tree GPU execution is not claimed. See
`results/qwen3_triton_portable_clone_cpu_v3.json`.

The final source candidate is under
`artifacts/triton_qwen3_v1/opensource_source_candidate_v3/`. Its runtime model,
training and benchmark code is byte-identical to the cold-checked review
export. The packaging helper also runs against the completed evidence; the
final copy adds curated acceptance/results and updated documentation.

Source export excludes weights, corpus bytes, runtime caches, native gradient
archives and original artifact source directories. Third-party notices are
retained under `qwen3_triton/third_party`. Own source licensing and publication
remain pending; no repository has been published by this work.
