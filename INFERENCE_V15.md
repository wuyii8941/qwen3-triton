# Qwen3 Triton inference V15

Development branch: `research/inference-v15`, based on the qualified V14 training
runtime (`2577c29`). Training entry points and their evidence remain independent.
The small independent attention/cache/graph checks passed (nine numerical cases plus
cache guards, greedy generation and EOS handling). See
[small fixture evidence](results/qwen3_triton_inference_v15_small_fixture_v1.json).
The opt-in reference accumulation route now passes the original official8B
qualification protocol, including all512fixedS512blocks and five complete GPU
phase audits. This is finite-protocol acceptance, not universal numerical parity.
Earlier default-route failures remain documented below. Request performance for
the qualified route is measured separately.

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
  --reference-accumulation --prefill-tiling \
  --chat --prompt 'What is the capital of France?' --max-new-tokens 32
```

`--enable-thinking` controls the tokenizer's Qwen3 chat template. It does not
change the GPU architecture. Tokenizer/chat formatting run on the CPU.

```python
from qwen3_triton.inference import TritonQwen3Inference

model = TritonQwen3Inference.from_pretrained(
    '/data1/tzh/models/Qwen/Qwen3-8B',
    reference_accumulation=True, prefill_tiling=True)
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
prefill and text generation. Five compressed raw traces cover weight transfer,
M17/M65 prefill, decode and generation; the auditor requires the actual phase
scope, exact operation counts and GPU activity for every correlated launch.
It rejects unclassified kernel names,
native GPU memset and device copies. Each trace is capped at8MiB. Opposite-order
latency comparisons use fresh caches, the same prompt/continuation and2warmup
plus5measured repetitions. No dense logits, gradients, weights or checkpoints
are saved. Reports distinguish numerical qualification and measured performance.

Historical development runs and failures remain local; publish only bounded
JSON summaries and source hashes. Current evidence status is updated after the
actual check, not inferred from the script.


## Measured development status (historical default route)

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


## Monitored matched Graph comparison

The revised decode source was compared with HF on oneA6000 with continuous
foreign-process monitoring and the shared120s-idle admission. Both paths use
one65-token prompt and8fixed greedy tokens, no earlyEOS. Both Graph paths match
their own eager generation on three changed inputs. No foreign context was seen.
Forward/reverse request medians are:

| Path | Order1(seconds/request) | Order2(seconds/request) |
| --- | ---: | ---: |
| HF eager |0.31627|0.31498|
| HF Graph |0.21889|0.21861|
| Triton eager |1.14893|1.14441|
| Triton Graph |0.20499|0.20480|

Triton Graph request time is6.35%/6.32% lower than HF Graph in this fixed-shape
experiment. **This remains conditional performance evidence, not a quality-
equivalent performance recommendation:** one of the three inputs generates
different tokens from HF, and the original full-model logit gates still fail.
Preparation/capture cost is excluded in all replay timings, both models are
resident simultaneously, and this is not a serving throughput result.
[Full monitored comparison](results/qwen3_triton_inference_v15_paired_graph_v1.json).

On identical officialHF intermediate operands, all measured rotary tables,
fusedQK-Norm/RoPE and SwiGLU outputs match bitwise; remaining local differences
are in linear projection/MLP accumulation. The current separate-GEMV decode
MLP is closer toHF than the experimental tensorcore fusedMLP on those inputs,
so that experimental path was not adopted. [Boundary diagnostics](results/qwen3_triton_inference_v15_boundaries_v1.json).


Linear isolation probes compared identical HF operands at layers0/12/24/31/35
in prefill and fourdecode positions. Changing GEMM Ktile16/32/64/128 did not
change their discrepancy. SplitK2/4/8/16 and GEMV lane widths32/64/128/256/512
also did not remove it. No probe was adopted. Native HF decode uses a mix of
cuBLAS GEMV and TensorCore GEMM kernels, captured in the retained diagnostic
summary. [Linear-order diagnostics](results/qwen3_triton_inference_v15_linear_order_diagnostics_v1.json).
These observations narrow the remaining precision investigation; they do not
establish bitwise parity or justify relaxing the original full-model gates.


## Independent rounding and exploratory quality diagnostics

CPUFP64 matrix products and `math.fsum` agree exactly on the228disagreeing
coordinates sampled from20realHF layer calls. On223of those coordinates the
TritonBF16 result is nearer the fsum dot product, and HF is nearer on5. These
are selected disagreements, not an unbiased whole-model precision comparison.
They show that discrepancy fromHF alone does not identify a candidate bug.
[Independent oracle](results/qwen3_triton_inference_v15_linear_oracle_v1.json).

Eight fixed128-token validation blocks(1024prediction positions,1016shiftedNLL
targets) show HFmeanNLL3.066296 and Triton3.066984, a difference0.000688.
Top1matches1012/1024; meanHF-to-TritonKL0.000925. The earlier random-input
cases match397/398top1positions. The only teacher-forced token difference
in the two16-token text continuations has exactly tied topHFlogits; Triton's
choice also has maximumHFlogit. That tie can diverge subsequent free generation.
These observations neither replace the original gates nor establish universal
quality equivalence. The original two absolute-logit gate failures and literal
greedy mismatch remain false. The later full512-block/S512 comparison is recorded below.
[Raw diagnostic](results/qwen3_triton_inference_v15_quality_diagnostic_v1.json),
[exact execution source](results/qwen3_triton_inference_v15_quality_source_v1.json).


## Complete fixed-corpus S512 comparison

The full512-block validation comparison completed with all512unique block
indices,261632shiftedNLLtargets and262144logit positions. Every block's maximum
and mean absolute logit difference,KL andTV is zero; all top1positions agree.
HF and Triton meanNLL are both2.605893106199801. No foreignGPUprocess was
observed, and the recorded model/source hashes match that run’s executed source.
[Complete raw report](results/qwen3_triton_inference_v15_full_validation_v1.json),
[coverage and source receipt](results/qwen3_triton_inference_v15_full_validation_receipt_v1.json).

This qualifies this recorded S512numerical comparison, not every inference
shape. The two original short/decode logit failures and literal greedy mismatch
remain unqualified. The run's decode profiler audit also fails:585CUDAlaunches
were recorded but only570GPUkernel events; the first15launch correlations lack
GPU activity events, so only35/36cache append kernels were observed. The raw
trace is preserved and is not silently treated as a passing trace. A complete
phase recording must be acquired separately; earlier passing traces do not
replace this failed run's coverage result.


## Complete phase recollection and generation metadata

The stricter trace reviewer now requires exact complete operation counts and
matching GPU activity for every actual-phase CUDA launch. Its CPU regression
accepts a complete decode trace and rejects both a missingGEMV and an unmatched
launch, even if36cache appends remain. A separate monitored8B recording now
passes setup/prefill/decode/generation with no missing correlations, count gaps,
unclassified kernels or native GPU memory events. Collector bootstrap consists
of64Triton position launches separately annotated and retained in each raw trace.
Original full_validation_008 stays failed; the fresh phase evidence is separately
labeled. [New phase report](results/qwen3_triton_inference_v15_phase_reaudit_v1.json),
[execution-source boundary](results/qwen3_triton_inference_v15_phase_reaudit_sources_v1.json).

The local checkpoint lacked tokenizer_config.json. The exact official file from
revisionb968826d9c46dd6066d109eabc6255188de91218was fetched and installed without
changing weights or tokenizer.json. SHA256d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101,
9732bytes; [source receipt](results/qwen3_triton_inference_v15_tokenizer_metadata_v1.json).
Fresh clones need a complete local official checkpoint/tokenizer including this
file. This supplies both thinking/non-thinking CPU chat templates.

The model now loads generation_config.json and defaults to its EOSlist/padding;
explicit generate arguments still override it. The CLI no longer replaces the
model's EOSlist with a single tokenizerEOS. CPU regression verifies loading and
CLI propagation of multipleEOS IDs with a conflicting tokenizer fixture. In the
real8B CLI run, `--chat` answered `The capital of France is Paris.` and stopped
on token151645after8tokens within a32-token budget, without stderr.
[CLI record](results/qwen3_triton_inference_v15_chat_demo_v1.json).
The metadata fix changes CPU stopping configuration, not GPU arithmetic;
phase009's exact pre-metadata source is retained. This smoke check does not
resolve the original short-shape numerical gates or establish broad chat quality.


## Experimental interleaved decode accumulation

`TritonQwen3Inference.from_pretrained(..., sliced_decode=True)` explicitly enables
an inference-only experiment for B1 single-token Q/O/down linear projections
(N4096,K4096/12288). The default remains `False`. Two independent FP32 TensorCore
accumulators consume alternating32-element Ksegments before a final sum, matching
the measured SM86 cuBLAS sliced1x2 order on identical operands. This does not
change prefill, K/V/gate/up, weights, activation dtype or the original gates.

Across all36layers and four decoded positions, this kernel has zero unequal
coordinates against the recorded native Q/O/down results; the old GEMV has4746
unequal coordinates. This is local operand parity, not complete model parity.
[Shape coverage](results/qwen3_triton_inference_v15_sliced_shape_coverage_v1.json),
[single-kernel check](results/qwen3_triton_inference_v15_sliced_kernel_check_v1.json),
[initial interleaving probe](results/qwen3_triton_inference_v15_interleaved_probe_v1.json).

The first full8B diagnostic with this route passes all four phase audits and
both16-token text continuations matchHF. However, one original logit case still
fails: B1decode after67tokens has meanAbs0.0552724 and relativeL2 0.0248562,
versus0.0371610/0.0174549 with default GEMV. The failing-case count drops from2to1,
but the remaining error is larger. The eightS128 corpus diagnostics are unchanged
because prefill is unchanged. This route therefore stays opt-in and unqualified.
[Full diagnostic](results/qwen3_triton_inference_v15_sliced_full_diagnostic_v1.json),
[exact source before the opt-in API change](results/qwen3_triton_inference_v15_sliced_execution_sources_v1.json).

The verifier accepts `--sliced-decode --diagnostic --warmup-collection`; phase
counts require145GEMV+108sliced kernels per decode instead of253GEMV. CPU tests
exercise both routes and reject wrong-route counts, missing compute and missing
GPU activity. `compare_qwen3_graph_v15.py --include-sliced` compares HF, default
and sliced eager/graph requests in both orders with the same resident weights,
prompt, token budget, GPU admission and continuous foreign-process monitoring.

A fresh8B run through the explicit opt-in API reproduces all24case metrics from
the first diagnostic exactly. All four new phase audits pass, including exactly
108sliced and145GEMV launches per decode, with no observed foreignGPUprocesses.
The original numerical failure remains false; both text continuations agree.
[Explicit opt-in verification](results/qwen3_triton_inference_v15_sliced_optin_verification_v1.json).

The same-process six-arm request comparison confirms the sliced route is slower:

| Graph request |Forward order seconds|Reverse order seconds|
|---|---:|---:|
|HF|0.218586|0.218960|
|Triton default|0.204737|0.204710|
|Triton sliced|0.213085|0.213162|

Sliced is4.08%/4.13% slower than default, while matchingHF on all three tested
changed inputs (default matches2/3). Each graph matches its own eager route.
No foreignGPUprocess was observed. B1S65,8generated tokens, no earlyEOS, same
resident weights, preparation/capture excluded. This rejects sliced as a speed
improvement and does not resolve the numerical gates. Default Graph's6.34%/6.51%
lower latency thanHF Graph remains conditional, not quality-equivalent evidence.
[Six-arm raw report](results/qwen3_triton_inference_v15_sliced_paired_graph_v1.json).


## Prefill tile screening (research only)

A synthetic identical-input screen of8configurations x4operator shapes x3M
values (17/65/257) found96/96bitwise equal to each screened operator’s default
tile configuration. The gate/up screen uses the older activation implementation;
its difference from the runtime precision kernel is disclosed below.
Only S65 is selected for the next full-request experiment; other lengths are
not changed. The narrower Kstep32 variants do not provide the expected gain;
the measured faster choices use Kstep64 and shape-specific Ntiles:

|S65 operator|Default ms|Selected ms|M/N/K tile|
|---|---:|---:|---|
|Q/O|0.07837|0.05970|128/64/64|
|K/V|0.04275|0.02804|128/32/64|
|down|0.24796|0.17580|128/64/64|
|gate/up|0.49578|0.31813|128/32/64|

These are synthetic per-kernel graph replay measurements on one A6000, not
whole-model speed results. At S257 several choices are slower, so there is no
blanket switch to larger tiles. No foreignGPUprocess was observed. Production
dispatch is unchanged. The full-request experiment uses a local research-only
route override, then checks all logits and cached continuation against the
unaltered baseline before reporting timings.
[Complete tile screen](results/qwen3_triton_inference_v15_prefill_tiles_v1.json).


The research-only S65 route now has complete-model evidence: eight random
prompts, each with full65-position prefill logits and four cached decode steps,
give40/40bitwise equality comparisons against the unchanged Triton baseline.
Both fresh prefill/decode traces pass. Each graph matches its own eager route;
three changed-input generations agree across baseline/tiled/HF in this run.
This does not replace the independent original short/decode HF gates.

|Graph request|Forward order seconds|Reverse order seconds|
|---|---:|---:|
|HF|0.219219|0.219673|
|Triton default|0.205221|0.205345|
|Triton S65 tiles|0.194885|0.194781|

The tiled request is5.04%/5.14% lower latency than default and11.10%/11.33%
lower thanHF in this B1S65/8-token graph experiment. No foreignGPUprocesses
were observed. Graph setup is excluded, as in all prior comparisons. This
result belongs to the standalone research override; it is not yet an exported
production option. [Complete request spike](results/qwen3_triton_inference_v15_tiled_request_spike_v1.json).


Source review found an important boundary in the preceding spike: its gate/up
candidate uses `kernels/mlp.py`, whereas the unchanged runtime uses
`kernels/mlp_precision.py` (libdevice exp and round-to-nearest division). Thus
the96-case screen compares tile variants within the older gate/up implementation,
and the40 full-model equalities establish only those tested inputs. The spike's
speed gain combines that implementation change with tiling and must not be
attributed solely to tiles. The exported option uses the existing precision
kernel and changes only its configuration; it requires fresh API evidence.
The earlier raw sources/results are retained at commit7497112, unmodified.


## Explicit prefill tiling API

`TritonQwen3Inference.from_pretrained(path, prefill_tiling=True)` and CLI
`--prefill-tiling` select the measured M65 configurations. The default isFalse;
`sliced_decode` remains independent and defaultsFalse. Only flattenedM=B*S=65
and the official8B Q/O/K/V/down plus gate/up shapes are selected; other shapes
use the existing Triton route. Evidence is scoped to B1S65 on SM86. Existing
precision-kernel activation arithmetic, weights and cache handling are retained.
CPU regression verifies loader propagation, unchanged checkpoint tensors,
defaultFalse, independent decode choice and CLI forwarding.

The actual API full-logit/request comparison and unchanged HF gates are recorded
below. Prior spike timings remain assigned to the standalone research override.

Set `prefill_tiling=True` before constructing `CapturedGreedyRequest`: a graph
captures that route at construction. Changing model options later does not
rewrite an existing captured graph; create a new request graph for a new choice.


The fresh precision-API audit retains all24original HF-case metrics exactly,
including the same two failed decode cases (meanAbs0.0364415 after18tokens and
0.0371610 after67tokens) and the raw greedy mismatch. It passes all four complete
phase traces and observes no foreignGPUprocesses. The eightS128 diagnostics are
unchanged. Original qualification remainsFalse.
[Original-gate audit](results/qwen3_triton_inference_v15_prefill_api_HF_gates_v1.json),
[exact pre-layout-guard source](results/qwen3_triton_inference_v15_prefill_api_HF_sources_v1.json).

Read-only review identified a layout boundary: fixed-stride tiled launches must
not consume positive-stride views accepted by the existing matmul helper. The
new eligibility guard requires matching contiguous BF16 CUDA input/weight;
other layouts use the existing stride-aware Triton matmul. This guard does not
change the official contiguous path. Fresh API testing includes stride2 input
and weight views, reproducing the former wrong fixed-stride output before
checking the guarded result against the baseline.


The fresh guarded precision API now reproduces40/40full-logit comparisons
bitwise against the default. Both stride2 regressions reproduce the old wrong
launch (265996/266015unequal coordinates) and pass with the eligibility guard.
Both new complete phase audits pass, and three changed-input generations agree
with default/HF; each graph matches its own eager path. No foreignGPUprocesses
were observed, and every recorded source digest matches the executed file.

|Guarded API graph request|Forward order seconds|Reverse order seconds|
|---|---:|---:|
|HF|0.218980|0.219544|
|Triton default|0.205253|0.205412|
|Triton prefill_tiling=True|0.194780|0.194825|

This actual API lowers request latency by5.10%/5.15% versus default and
11.05%/11.26% versusHF Graph in the measured B1S65/8-token SM86 setting.
Capture/preparation is excluded. Eager timings do not show a consistent gain
and remain slower thanHF; this is a graph-replay optimization result. The
original independent HF short/decode gates remain failed, so the option stays
explicit and is not a generally qualified quality-equivalent recommendation.
[Guarded API raw report](results/qwen3_triton_inference_v15_prefill_api_request_v1.json).

Reproduce with `probe_qwen3_tiled_request_v15.py --production --output ...`.
Historical API attempt020failed from a benchmark argument-name collision;
022was retired while pending so it could not execute a previously imported
unguarded module after the source changed. Both local records are retained;
only complete guarded run023supplies the API performance claim above.


## Original-failure attribution with public-path observation

A reference-only diagnostic now reproduces all recorded metrics for the original
B1S17/S65 prefixes and four cached decode positions (10phases), including both
original failures. Wrappers around the actual public model entry retain only
in-memory intermediate references; every wrapped call matches unwrapped logits
and all initialized KV positions bitwise. Model/default arithmetic is unchanged.
No foreignGPUprocesses were observed; all executed source digests are retained.

On identicalHF operands, all360attention,720input/post-attention normalization,
and360SwiGLU probes match bitwise. Earliest discrepancies appear in linear
accumulation, then propagate through layer residuals and normalization. At the
two failing positions, replacing the candidate LMhead with the nativeHF LMhead
on the candidate hidden state leaves meanAbs essentially unchanged:

|Original failing position|Triton full-path meanAbs|Native head on Triton hidden|
|---|---:|---:|
|after18tokens|0.0364415273|0.0364415757|
|after67tokens|0.0371609703|0.0371609367|

The Triton LMhead on exactHF hidden inputs instead differs by meanAbs only
5.63e-8/1.13e-7 at those positions. Thus this experiment directs investigation
toward accumulated body-linear rounding, not an LMhead or identical-input
attention mismatch. The shared logit offsets are+0.0352378/-0.0351442; original
raw-logit gates remain false. Local equivalence does not imply whole-model
quality equivalence after different inputs have propagated through36layers.
The HF cache is made contiguous only for the isolated attention reference probe;
HF first-append caches may be transposed, unlike the model’s own cache contract.
Attempt024was retired after review caught that diagnostic layout boundary;
only corrected complete run025supplies these results.
[Complete bounded diagnostic](results/qwen3_triton_inference_v15_error_growth_v1.json).


## Controlled accumulation-order probe

The three-term probe uses BF16 weights with+2**30,-2**30and a unit marker.
Exact arithmetic leaves1 (0when a large term overwrites the marker); FP32
accumulation can lose the marker.196comparisons cover seven placements,
M1/17/65/512, and seven Triton schedules. All four recorded native kernel names
match the corresponding real8B Q-projection observations; monitoring is clean.
M1matches interleaved2x32on all seven placements. M17andM512match sequential
Triton accumulation. M65matches neither tested schedule: for a large pair at
K0/K2048 only marker columns2816..4095survive; forK4064/K4095only0..2815survive.
This suggests a different partition/reduction schedule, not a universal Ktile
choice. A contiguous non-power-of-two split is a hypothesis to test next;
marker patterns alone do not prove the complete native algorithm. Earlier
native-name filters only retained GEMM/GEMV names, so any separately named
reduction kernel needs full-event recollection. Production is unchanged.
[Complete sparse-dot probe](results/qwen3_triton_inference_v15_dot_traversal_v1.json).


## Complete native linear schedule recollection

The complete raw native trace confirms a separately named cublasLt
`splitKreduce_kernel`, previously excluded by GEMM/GEMV-name-only filters.
All80scoped launches across16shape configurations have corresponding GPU
activity. The gridZsplit counts and successful random-operand reproductions are:

|M|Operator|Splits|Contiguous Kpartition length|Triton reduction|
|---:|---|---:|---:|---|
|17|gate/up, K4096|3|1408|forwardFP32|
|65|Q/O, K4096|3|1408|forwardFP32|
|65|K/V, K4096|5|832|forwardFP32|
|65|down, K12288|7|1792|forwardFP32|

A64-element partition alignment reproduces all four random operand cases
bitwise. Reversing reduction order leaves discrepancies; earlier power-of-two
split tests did not include the observed3/5/7choices. The sparse-dot2816boundary
corresponds to two1408-element partitions. These measurements identify a
concrete arithmetic mismatch to test on all real layers, not yet full-model
qualification. The production model remains unchanged.
[Complete native schedule report](results/qwen3_triton_inference_v15_native_splitk_v1.json).
The cuBLAS12.8documentation also exposes split count and reduction scheme as
separate algorithm settings: [official cuBLASLt attributes](https://docs.nvidia.com/cuda/archive/12.8.2/cublas/index.html#cublasltmatmulalgoconfigattributes-t).


## Real8B schedule reproduction (research override)

The exact observed contiguous3/5/7partitions and interleaved2x32routes now match
all1188selected realHF layer calls bitwise across36layers and10phases. The
previous implementations differed on176031coordinates in those same selected
calls. Under a process-local research wrapper, B1S17andS65full-prefill logits
become bitwise equal toHF; both16-token text continuations match. The original
B1S257prefill also remains bitwise equal.

One original case still fails: cached decode after67tokens has meanAbs0.0429935
(relativeL2 0.0196509), versus0.0371610in the unmodified baseline. After18tokens
now passes at meanAbs0.0132812. Thus this is not adopted as the default or
qualified as a complete fix. Remaining native GEMV reductions on K/V/gate/up
still differ locally; exact body-prefill parity does not ensure later decoding
parity when those values propagate through36layers.

The underlying verifier was deliberately unchanged: its setup trace passes,
but prefill/decode/generation traces reject the new kernel names/counts. Those
failed raw audits are preserved, and the outer source receipt records the actual
research wrappers while underlying model option flags remainFalse. No foreign
GPUprocesses were observed. Formal API/audit work is planned only after the
remaining numerical gate is resolved.
[Actual-source/local-call receipt](results/qwen3_triton_inference_v15_native_order_local_v1.json),
[unchanged-gate raw report](results/qwen3_triton_inference_v15_native_order_full_v1.json).


## Native GEMV tree identification

Controlled sparse dots now cover18large-term placements and everyKcoordinate
for native K/V and gate/up GEMV. The recorded native kernel identities match
the complete schedule study. Repeated markers are consistent; no foreignGPU
processes were observed. A CPU binary-addition-tree model is checked against
explicit FP32 sparse-dot evaluation before fitting478regular schedules per
operator. The fits include a simple sequential per-lane accumulation followed
by a descending-offset warp tree:32lanes forK/V,16forgate/up.

Nine/eightparameter configurations fit the respective masks exactly; several
are alternative load groupings for the same arithmetic tree. This does not
prove full arithmetic identity on arbitrary operands. The next reference-only
run implements the inferred tree explicitly (rather than relying on an
implementation-dependent generic reduction), preserves K-order while unrolling
loads, and rechecks real operands plus all original full-model gates. The fast
existing LMhead is unchanged. Production defaults remain unchanged.
[Complete GEMV tree probe](results/qwen3_triton_inference_v15_gemv_tree_v1.json).


## Complete arithmetic research prototype: original gates pass

Adding explicit native GEMV trees resolves the remaining original failure in the
research prototype without changing any threshold. All24original numerical
cases pass and both literal16-token text continuations matchHF. All2520realHF
body-linear local checks match bitwise, including2340selected changed calls.
B1S17/S65/S257full-prefill logits are bitwise equal. At the two originally
failing decode positions, meanAbs falls to5.63e-8(after18) and1.13e-7(after67).
Other cases still use the original tolerance: e.g.B1decode after a257-token
prefix reaches meanAbs0.0189731; this is not a universal bitwise claim.
No foreignGPUprocesses were observed.

The actual GEMV implementation uses sequential per-lane FP32FMA,8load unroll,
explicit descending XOR gathers and lane-zero extraction. K/V use32lanes;
gate/up use16. This avoids relying on a generic reduction’s physical layout.
The existing fast LMhead is retained. The prototype changes arithmetic only
inside its process-local research wrappers; production defaults are unchanged.

**This is arithmetic evidence, not exported API/GPU-path qualification.** The
unchanged profiler reviewer still rejects the new kernel names/counts, so setup
passes while prefill/decode/generation audits remainFalse and qualified=False.
The outer receipt includes the actual new kernel/wrapper source. Exporting the
option, testing complete phase counts, full fixed validation and measuring the
correct version’s paired whole-request speed remain required.
[Actual-source/complete arithmetic receipt](results/qwen3_triton_inference_v15_reference_order_complete_v1.json),
[all real-layer checks](results/qwen3_triton_inference_v15_reference_order_locals_v1.json),
[unchanged original gates and retained trace failures](results/qwen3_triton_inference_v15_reference_order_gates_v1.json).


## Exported reference accumulation module

`TritonQwen3Inference.from_pretrained(path, reference_accumulation=True)` and CLI
`--reference-accumulation` now select the measured Triton-only arithmetic.
DefaultFalse preserves the earlier inference path. The fast existing LMhead
remains unchanged. Only the recorded M1/M17/M65official8B shapes are selected;
unsupported shapes/layouts retain the existing Triton route. The diagnostic
`accurate` flag takes precedence; `gemv=False` declines ordered GEMV routes.
Qualification uses default `gemv=True`. Set options before graph construction.

The exported module uses M128/N64tiles for M65split linears and M32/N64forM17,
avoiding repeated weight reads across row tiles while retaining64-aligned K
partitions and forward FP32reduction. Fresh identical-HF-input checks cover
all2520real-layer calls and remain bitwise equal. Positive-stride input/weight
views, unsupportedM and disabledGEMV preference preserve default outputs.
Last-K-only tests for all four global-split shapes pass even when every FP32
partial startsNaN, checking tail coverage and complete overwrite. CPU checks
verify loading/CLI propagation, defaultFalse, independent options and weights.
No foreignGPUprocesses were observed. This is local exported-module evidence;
complete public-model/audit/fixed-corpus qualification and speed follow separately.
[Exported-module operator checks](results/qwen3_triton_inference_v15_reference_api_ops_v1.json).


## Actual API qualification032: before short-row norm correction

Run032 uses the exported API with `reference_accumulation=True`,
`prefill_tiling=True`, default `gemv=True` and `sliced_decode=False`. All24
original cases pass unchanged relativeL2<=0.02, meanAbs<=0.03, maxAbs<=0.5;
maximum case meanAbs is0.027414. Both original raw-text16-token greedy
continuations match HF literally. All512unique fixedS512validation blocks
(261632shifted targets) have bitwise-identical logits and meanNLL
2.605893106199801 on both implementations. Official pinned weights were
hash-verified; executed source hashes match the retained report.

All five raw phase traces pass, including M17's separate split/reduce path,
M65's180split+180reduce launches, and decode's144orderedGEMV+108interleaved
launches per forward. No unclassified/native compute, device-memory fallback,
missing launch activity or foreign GPU process was observed. The five gzip
traces total1199585bytes; each is bounded by8MiB. Only metrics/tokens and raw
traces are retained, with no tensors/checkpoints/logits.

`functional_qualified` covers the original numerical/greedy gates and all five
phase audits with clean monitoring. `qualified` additionally requires the
complete unique512xS512fixed validation set and all its numerical gates. Empty
or partial corpus runs cannot qualify the complete protocol. CPU regressions
reject missing scopes, launches, reductions, activity and incorrect route counts.

Limits: extra teacher-forced diagnostics on the two text prompts retain22
meanAbs failures despite literal token equality. The original24case gate is
unchanged; these diagnostics are not silently counted as passing numerical
checks. Unsupported shapes use the existing Triton route and have no blanket
parity guarantee. New flags remain opt-in/defaultFalse. Earlier failures and
unqualified faster routes are historical evidence, not substitutes for this run.

[Complete API report](results/qwen3_triton_inference_v15_reference_api_qualified_v1.json),
[source/raw-trace receipt](results/qwen3_triton_inference_v15_reference_api_receipt_v1.json).

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/research_python.sh \
  scripts/verify_qwen3_inference_v15.py --physical-device 1 \
  --reference-accumulation --prefill-tiling --warmup-collection \
  --corpus-blocks 512 --corpus-sequence 512 \
  --output /data1/tzh/proj/qwen_operator_lab/artifacts/inference_v15/new_unique_run
```


## Changed-input guard found a short-row norm gap

The first six-arm request attempt033stopped before timing: changed-input case1
matched each graph's own eager path, but reference greedy token4was11whereHF
selected15. Original032finite-protocol acceptance remains historical evidence,
not a claim that this new case passed. The failed attempt retained no speed claim.

Direct real-input diagnostic038locates the first propagated difference at
layer14inputRMSNorm in the third request step. All2016same-HF-input body-linear
checks and288same-QKVattention checks are bitwise. The old32-lane norm mean is
1.1948513984680176, while native uses1.1948515176773071; inverse values differ
0.9148352146148682vs0.9148351550102234, producing35BF16differences. A separate
512-lane Triton probe matches the native mean, inverse and output bitwise. This
is a reduction-order issue; no epsilon/threshold/weight change is used to mask it.

The opt-in route now mirrors local PyTorch2.10Reduce.cuh's short-row block width
for contiguous BF16H4096inputs:512/last_pow2(rows),1<=rows<16. Other rows retain
the previous normalization. DefaultFalse preserves existing norm arithmetic.
The shared runtime's functional `__and__`/`__or__` dispatch also stops choosing
the left operand as an in-place output; a CPU regression reproduces the alias.
These source changes require fresh complete validation and request measurements.
Ordinary and widened random norm fixtures both passed640/640before the fix, so
they are retained as weak checks; actual model-input regression is required.


## Fresh qualification039: short-row norm correction included

The current opt-in reference+prefill-tiling source passes a fresh complete run039.
All24original numerical cases pass (maximummeanAbs0.0258631855); all512unique
fixedS512blocks have bitwise-identical logits and identical meanNLL
2.605893106199801; both original16-token continuations match literally. All five
phase audits pass, including73reference-norm launches per B1decode. Raw gzip
traces total1172877bytes. All executed source hashes match and continuous
monitoring reports no foreign context. No original gate was relaxed.

The original failed changed-input case also passes8HFteacher-forcedsteps, with
576/576real-inputRMSNorm,2016/2016body-linear and288/288same-QKVattention
checks bitwise. All propagated model-body intermediates match; the unchanged
fastLMhead retains a few BF16rounding differences. Synthetic norm checks cover
640cases, but the actual-input RED→GREEN record is the evidence for the fix.

The earlier032source is superseded for current qualification. Extra text-prompt
teacher-forced logits still retain23meanAbs failures despite literal greedy
equality, so this is finite-protocol acceptance, not a universal accuracy claim.
Two new option defaults remainFalse. No training numerical/performance result
is re-certified by this inference run. The shared boolean dispatcher correction
has a targeted CPU alias regression and fresh complete inference validation.

[Current complete report](results/qwen3_triton_inference_v15_reference_api_qualified_v2.json),
[current source/raw-trace receipt](results/qwen3_triton_inference_v15_reference_api_receipt_v2.json),
[real-input norm RED→GREEN](results/qwen3_triton_inference_v15_reference_norm_regression_v1.json),
[preserved failed request](results/qwen3_triton_inference_v15_reference_request_failure_v1.json).

The combined regression/full run can be reproduced under one continuously
monitored stableGPUlease:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/research_python.sh \
  scripts/qualify_qwen3_reference_request_v15.py \
  --output /data1/tzh/proj/qwen_operator_lab/artifacts/inference_v15/new_unique_regression
```


## Qualified-route paired whole-request measurement040

Same physicalA6000, B1S65,8output tokens including prefill+7cacheddecode steps,
fixed budget/no earlyEOS, two opposite orders,2warmup+5measured repetitions.
Both model instances remain resident; model loading, compilation and graph
construction are uniformly excluded. Every arm's complete request is synchronized
and wall-timed. Source/options/raw-trace proof is bound to complete039validation.

|Complete request, ms|Order1 median|Reverse-order median|
|---|---:|---:|
|HF eager|306.705|306.081|
|HF Graph|218.790|219.010|
|Default Triton eager (unqualified route)|1108.736|1122.598|
|Default Triton Graph (unqualified route)|204.735|204.902|
|Reference+tiling Triton eager|1111.911|1111.335|
|Reference+tiling Triton Graph|201.387|201.237|

The measured reference Graph request uses7.95%/8.12%less time than same-order
HF Graph and1.64%/1.79%less than the default Triton Graph. The default route
remains numerically unqualified and is only a timing ablation. Eager has no
consistent improvement versus default and remains about3.63xslower than HF
eager. This is a Graph-only result at the stated finite shape/protocol, not a
serving-throughput or unrestricted quality-equivalence claim. The earlier194.8ms
prefill-only path failed original quality gates and is not promoted here.

All three changed inputs have Graph=eager output equality for all routes;
reference output also matches HF for all three (default only2/3). The original
failed case1nowproduces `[11,220,16,15,15,15,15,15]` in both HF and reference.
No foreign context is observed and source hashes match. All60measured samples
are retained; no median was selected across orders. The original033failure and
its guard remain; no assertion or quality threshold was removed to get timings.

[All paired samples, changed-input tokens and source/proof binding](results/qwen3_triton_inference_v15_reference_paired_request_v1.json).

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/research_python.sh \
  scripts/compare_qwen3_graph_v15.py --include-reference \
  --reference-prefill-tiling \
  --qualification /data1/tzh/proj/qwen_operator_lab/artifacts/inference_v15/reference_norm_qualified_039/full/result.json \
  --output /data1/tzh/proj/qwen_operator_lab/artifacts/inference_v15/new_unique_paired_request
```
