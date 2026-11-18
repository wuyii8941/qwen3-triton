# Distributed training experiments

This document defines the only three distributed candidates considered after
the repository cleanup. Experimental code stays on `research/hybrid-parallelism`
until the gates below pass on four idle RTX A6000 GPUs.

## Evidence already available

Historical paired Qwen3-8B B1/S512 runs show:

| Runtime | Order 1 | Order 2 |
| --- | ---: | ---: |
| HF native | 5.2760 s/step | 5.2756 s/step |
| selective Triton peer reduce-scatter | 4.5148 s/step | 4.5176 s/step |
| selective peer RS + Triton AdamW | 4.6249 s/step | 4.6287 s/step |
| every collective/runtime operation in Triton | 5.9268 s/step | 5.9269 s/step |

The selective result passed the historical gradient, short-quality, memory,
lossless-wire and paired-order gates. It is evidence for a narrow hybrid
candidate, not evidence that the cleaned source still has the same result.

The four GPUs form two fast PIX pairs `(0,1)` and `(2,3)`; traffic between the
pairs traverses SYS links. Historical 64 MiB collective probes measured direct
peer and NCCL at about 6.22 ms, while the two-level hierarchical candidate took
about 9.14 ms. The old hierarchical implementation is therefore rejected.

## Experiment A: selective collective routing

Candidate: NCCL for all-gather, small reduce-scatter, all-reduce, all-to-all and
barrier; checked Triton peer transport only for FP32 reduce-scatter inputs of at
least 64 MiB. The threshold matches the historically qualified path.

Adoption gates:

1. Every-coordinate collective checks pass for tails and actual decoder sizes.
2. First-update gradients pass the existing 399-parameter gate.
3. No lossless BF16 staging violation and effective peak memory remains within
   41.875 GiB per rank.
4. Two opposite-order runs each beat the cleaned full mode and HF native by at
   least 3%; otherwise the candidate remains research-only.
5. The result records native and Triton call counts; it must never be labelled
   an all-Triton collective runtime.

## Experiment B: communication/compute overlap

Current peer RS performs a complete source-to-peer staging pass, a system
barrier, remote reduction, and a reuse barrier. Merely returning an async handle
does not remove these dependencies.

FSDP2 allocates a complete FP32 reduce-scatter input before its copy-in. The
largest current input is about 736 MiB, while the external-memory reserve is
512 MiB and historical effective reserved memory reached about 41.87/41.875
GiB. A complete peer-visible FP32 input is therefore rejected under the current
memory contract.

The remaining candidate is two smaller alternating peer buffers to defer the
reuse barrier and pipeline
   staging of chunk N+1 with reduction of chunk N. The total external allocation
   must remain at or below the current 512 MiB reserve.

For each candidate, compare isolated RS latency and full-step traces. Adopt only
if the full-step median improves by at least 3% in both execution orders without
reducing model-kernel/communication overlap or changing gradients. A synthetic
stream benchmark alone is insufficient.

## Experiment C: TP, PP and DP

The current four-rank FSDP run is the DP/sharded baseline. The first and only TP
candidate is `TP=2 x DP=2`:

- TP groups: `(0,1)` and `(2,3)` on PIX links.
- DP groups: `(0,2)` and `(1,3)` across the two replicas.
- Column-parallel Q/K/V and gate/up projections; row-parallel output/down
  projections; vocabulary stays sharded.
- Qwen3 dimensions are divisible by two: hidden 4096, 32 query heads, 8 KV
  heads, intermediate 12288 and vocabulary 151936.

Reject `TP=4` initially because every layer would communicate across SYS links.
Do not implement PP for the current microbatch-one protocol: `PP=2 x DP=2` with
two microbatches has a theoretical 1F1B utilization of only 2/3. PP becomes a
candidate only if gradient accumulation supplies at least eight microbatches,
where the ideal utilization reaches 8/9, and memory or measured throughput then
beats TP2xDP2.

Parallelism adoption requires identical global-batch semantics, first-gradient
and short-quality gates, effective memory accounting, and two-order throughput
improvement of at least 3% over the winning non-TP runtime. No new public mode is
added for an analytical or microbenchmark-only improvement.
