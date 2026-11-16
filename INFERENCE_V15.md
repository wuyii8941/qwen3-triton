# Qwen3 Triton inference V15

Development branch: `research/inference-v15`, based on the qualified V14 training
runtime (`2577c29`). Training entry points and their evidence remain independent.
The small independent attention/cache/graph checks passed (nine numerical cases plus
cache guards, greedy generation and EOS handling). See
[small fixture evidence](results/qwen3_triton_inference_v15_small_fixture_v1.json).
Official8B inference qualification and performance are still pending; code existence
and a small fixture do not establish full-model correctness or speed.

The new `qwen3_triton.inference.TritonQwen3Inference` path loads local official
BF16 safetensors and provides dense prefill, chunked prefill, request-owned KV
cache, single-token decode and greedy generation. GPU embedding, GEMM/GEMV,
normalization/RoPE, attention, cache writes, residual/activation, token selection
and EOS updates are explicit Triton. Runtime dispatch rejects unknown CUDA
operations. Python controls the request/cache lifecycle; CUDA allocation, H2D/D2H
transfers and host synchronization retain the V14 host-service boundary.

Initial scope is one CUDA GPU, BF16 weights/activations, FP32 reductions, equal
unpadded batch lengths and head dimension128. Generation is greedy and accepts
one or multiple EOS token IDs, per-row finished state and padding. It is not a
serving engine: padded/ragged requests, batching scheduler, sampling/top-p,
beam search, quantization and tensor parallel inference are not implemented.
Unsupported inputs fail rather than select an HF GPU compute fallback.

## Usage

Keep runtime caches and outputs under `/data1/tzh` and use the pinned existing
research environment. A local official checkpoint/tokenizer is required:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/research_python.sh \
  scripts/generate_qwen3_triton_v15.py \
  --model /data1/tzh/models/Qwen/Qwen3-8B \
  --chat --prompt 'What is the capital of France?' --max-new-tokens 32
```

`--enable-thinking` controls the tokenizer's Qwen3 chat template. It does not
change the GPU architecture. Tokenizer/chat formatting run on the CPU.

```python
from qwen3_triton.inference import TritonQwen3Inference

model = TritonQwen3Inference.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B')
# ids: contiguous CUDA int64[B,S], no padding, all IDs in vocabulary.
cache = model.new_cache(batch_size=ids.shape[0], max_length=2048)
output = model(ids, past_key_values=cache, logits_to_keep=1)
output = model(next_ids, past_key_values=cache)  # next_ids: CUDA[B,1]
cache.reset()  # reuse allocation; old positions become logically invalid
sequence = model.generate(ids, max_new_tokens=32)
```

RoPE and causal positions use `cache.length`; appending a chunk advances every
layer together. A failed append invalidates the cache until reset, avoiding a
partially updated cache being reused. The cache cannot cross model instances.
Capacity is checked before writing. Cache storage is allocated once per request;
only the initialized prefix is read. Model outputs use the standard
`CausalLMOutputWithPast` container, with this implementation's `KVCache` in
`past_key_values`; it is not interchangeable with HF's cache object.

## Checks and retained output policy

`check_qwen3_inference_v15.py` compares attention to an independent FP32 SDPA
oracle, and a small random Qwen3 graph to HF. It checks chunked/nonzero-offset
cache semantics, reset, capacity/token bounds, model ownership, greedy output,
EOS/padding and finished-row handling. This fixture does not qualify8B.

`verify_qwen3_inference_v15.py` verifies pinned official checkpoint hashes, then
runs the complete36-layer8B graph against HF with cached decoding, chunked
prefill and text generation. Four compressed raw traces cover weight transfer,
prefill, decode and generation; the auditor rejects unclassified kernel names,
native GPU memset and device copies. Each trace is capped at8MiB. Opposite-order
latency comparisons use fresh caches, the same prompt/continuation and2warmup
plus5measured repetitions. No dense logits, gradients, weights or checkpoints
are saved. Reports distinguish numerical qualification and measured performance.

Historical development runs and failures remain local; publish only bounded
JSON summaries and source hashes. Current evidence status is updated after the
actual check, not inferred from the script.


## Measured development status

The complete36-layer8B diagnostic now runs prefill, chunked prefill, cached
single-token decode and raw-text greedy generation. Four raw phase traces pass
the Triton-only kernel and GPU memory-event audit. The non-aligned chunked
prefill NaN bug is fixed: reverse causal scanning now assigns zero softmax mass
to empty tiles instead of evaluating negative-infinity minus negative-infinity.
All four measured chunked-prefill cases pass their original numerical gates.

**HF numerical qualification remains false.** Two of24logit cases exceed the
original gates, and one of two16-token text continuations differs from HF. See
[complete diagnostic](results/qwen3_triton_inference_v15_official_diagnostic_v1.json).
The cases and original thresholds are retained. Diagnostic continuation does
not relax gates or convert failures to success. Pinning the local checkpoint's
missing tokenizer chat-template files is also required for `--chat`; measured
text continuations use raw prompt text.

`CapturedGreedyRequest(model, ids, max_new_tokens=8)` captures the same Triton
GPU work for a fixed batch/prompt shape and output budget. Three changed8B
inputs produce exactly the same tokens as eager Triton. For one65-token prompt
and8generated tokens, two opposite orders measured eager1.2898/1.3006seconds
versus graph0.20365/0.20373seconds, about6.3x faster than this eager candidate.
This is not an HF speed claim or a serving-throughput result. The graph run
used the shared stable GPU lease, but did not continuously monitor foreign
contexts; treat its timings as preliminary until monitored paired confirmation.
See [graph evidence](results/qwen3_triton_inference_v15_graph_8b_v1.json).

Graph limitations: no earlyEOS, fixed dense shapes/token budget, output aliases
request-owned storage until next replay, one object must stay on one stream
and cannot serve concurrent requests. CUDA Graph removes repeated CPU launch
work; its GPU arithmetic remains the existing Triton kernels. HF numerical
parity, matched HF Graph timing and broader request/length coverage remain open.


## Decode attention rounding investigation

The exact HF decode kernel is `Flash_fwd_kernel_traits<128,128,32,4>`:
head-dimension128, query-tile128, **key-tile32**, four warps. The initial
interpretation of the template's first two128 values as M/N was wrong; the
128-key experiment is retained as a rejected diagnostic. Source parameter order:
[Flash kernel traits](https://raw.githubusercontent.com/Dao-AILab/flash-attention/979702c87a8713a8e0a5e9fee122b90d2ef13be5/csrc/flash_attn/src/kernel_traits.h).

Same-Q/K/V probes over all36layers and fourdecode positions show exact BF16
agreement with HF for32-key tiles (both64/128query tiles), while64-key and
split decode have differences. Prefill on the same HF operands agrees exactly
with64-key tiles. The default now selects64-key prefill and32-key single-token
decode, with split decode retained as an explicit experimental control.
See [geometry probe](results/qwen3_triton_inference_v15_attention_geometry_v1.json).

The full-model rerun still fails two original numerical gates and one generated
continuation. All four phase GPU coverage audits pass and monitoring recorded no
foreign processes. The [new full diagnostic](results/qwen3_triton_inference_v15_official_diagnostic_v2.json)
preserves those failures: local attention agreement does not qualify the full
model. Earlier graph speed measurements apply to the earlier split-decode source,
not this revised default; graph speed and output parity must be revalidated for
this source before a performance recommendation. A matched, monitored HF Graph
comparison remains pending.
