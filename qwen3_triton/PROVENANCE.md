# Source provenance

Earlier model source snapshots were published. V14 has passed its fixed
4×A6000 full-model functional qualification, but formal comparisons found it
slower than both HF+Flash and V13, so it is not adopted as a performance
recommendation. The repository's source license remains to be selected before
an open-source release; the third-party notices below do not license this
project's own source.

`model.py` is the complete 36-layer Qwen3-8B candidate. `_model.py` and
`_model_optimized.py` provide its model graph and optimization switches.
All model arithmetic and its derivatives are explicit handwritten Triton.
PyTorch provides tensor/parameter/autograd bookkeeping. Historical training
callers use common native FSDP2/NCCL/AdamW; V14 explicitly replaces the GPU
runtime data path described below. No FlashAttention or cuBLAS model kernels
are called by the Triton variants.

The current V9 source port is recorded in
`results/qwen3_triton_source_port_v2.json`, including original/generated hashes,
selected attention defaults and every reversible transformation. The V1
manifest and its exported source snapshot remain historical evidence.
`scripts/check_qwen3_triton.py` checks generated source hashes and all
official parameter shapes without original artifact sources. The original
import-transformation AST replay is a separate retained audit, performed by
`scripts/check_qwen3_triton_source.py`. A CPU/meta check is not a GPU qualification.
Original experiment sources and failed numerical gates are preserved under
`artifacts/triton_qwen3_v1/`; they are not package dependencies.

## Algorithm and architecture references

- Official architecture/parameter names: Transformers 4.51.0 Qwen3 and
  `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`. Model weights are not vendored.
- BF16 attention cast locations, reverse key-tile traversal and probability
  reduction order were checked against PyTorch 2.10's FlashAttention
  submodule, commit `979702c87a8713a8e0a5e9fee122b90d2ef13be5`.
  Its original BSD-3-Clause notice is in
  `third_party/flash-attention/LICENSE`. The Triton code neither builds nor
  calls that implementation. Audit:
  [FlashAttention source](https://github.com/Dao-AILab/flash-attention/tree/979702c87a8713a8e0a5e9fee122b90d2ef13be5/csrc/flash_attn).
- `kernels/hf_reduction.py` preserves the PyTorch 2.10 CUDA row-mean
  summation order. `kernels/math_precision.py` evaluates the native SiLU
  opmath formula with explicit BF16 rounding edges. These are new Triton
  kernels, not compiled ATen calls. References:
  [Reduce.cuh](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/Reduce.cuh),
  [ActivationSiluKernel.cu](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/ActivationSiluKernel.cu).
  The installed PyTorch 2.10 source-distribution license is retained in
  `third_party/pytorch/LICENSE`.

Dependency packages retain their own licenses. Frozen WikiText corpus bytes
and official checkpoint files are obtained separately, verified against the
pinned recipe and not included in a source release. See the upstream
[WikiText dataset revision](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1)
and [Qwen3-8B model revision](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218).

## V14 runtime algorithm and API references

- `runtime/adamw.py` is handwritten Triton. FP32 state boundaries and double
  coefficient arithmetic were checked against the installed Torch2.10
  [fused AdamW reference](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/fused_adam_utils.cuh).
  The public Optimizer base supplies host state/group bookkeeping only.
- `runtime/dispatch.py`, `ops.py` and `fsdp.py` route actual FSDP packing,
  splitting, conversion, initialization and autograd accumulation to Triton.
  Their layout/API references are the installed Torch2.10 FSDP sources and
  public custom AllGather/ReduceScatter interfaces. No global Torch dispatch
  function or CUDA API is replaced or patched.
- `runtime/collectives.py` implements exchange, reduction and system-scope
  release/acquire mailbox synchronization in Triton. The ordering reference
  for two qualified vocabulary-CE message sizes is NCCL2.27.5's
  [ring algorithm](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/device/all_reduce.h),
  additionally measured from raw native CUDA traces and independent CPU
  addition trees. Candidate execution calls no native NCCL operation; native
  NCCL is an explicitly labeled diagnostic/control only. Matching addition
  order is fixed to the measured four-A6000 shapes, not a wider parity claim.
- `runtime/peer_memory.py` binds CUDA Driver IPC allocation/mapping APIs from
  Python; Gloo exchanges CPU handles. Payload/mailbox initialization is
  Triton. It avoids Torch2.10 Symmetric Memory allocator's explicit native
  `cudaMemset`, identified in source and actual profiler memory events.
  Reference: [CUDA Driver memory/IPC API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
- CPU process/autograd/FSDP scheduling, driver allocation/mapping/events and
  host-device DMA remain host services. GPU copies, fills, casts, arithmetic
  and synchronization are not exempted with them. Raw phase-wide audits
  inspect kernel events plus native GPU memset and device-copy events.

These references describe algorithms/APIs; runtime source is independently
written. Formal full8B proof, earlier component checks and failed attempts are
preserved in `TRITON_RUNTIME_V14.md` and the linked bounded evidence.
Functional qualification does not establish an overall speed improvement.

## Current qualification boundary

The historical artifact V9 candidate passes full36-layer BF16 forward checks on four
real training inputs with bitwise-equal HF logits. Its actual four-rank FSDP
control passes all399 parameter-shard gradient norm gates in both Triton
arms, completes four AdamW updates and verifies finite FP32 masters, gradients
and moments. Full-training CUDA traces show explicit Triton model arithmetic.
Its relocation and later V9-V13 qualification are documented by their dated
reports; the following describes the original artifact boundary.
The candidate gradient contract is L2 error below 0.06 times the reference
norm plus 1e-6 for every parameter shard; it is not bitwise gradient parity.
Full validation, paired step/module timings and method selection remain
separate requirements. Historical BF16 failures and the supporting FP32/MATH
control are preserved with their source identities.
Full details and supported-input limits are in `TRITON_QWEN3.md`.
