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
