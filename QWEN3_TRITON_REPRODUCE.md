# Reproduce complete Qwen3-8B experiments

The public V9 implementation has completed all sixteen accepted four-rank
official Qwen3-8B roles: HF + Flash, complete Triton base/optimized and five
ablations in opposite orders. The independent review passes all 399-parameter
gradient contracts, 27 updates and FP32 states, full validation before/after16,
and actual training CUDA coverage. Whole-step gains over the Triton baseline
are 3.96% / 4.25%; optimized remains 1.87% / 1.74% slower than HF + Flash.
[TRITON_QWEN3.md](TRITON_QWEN3.md) gives scope, contracts and complete results.

Source identities and relocation are mapped by
`results/qwen3_triton_source_port_v2.json`. The public module entry completes
96 actual8B cases in two orders. The 75-file V3 review export passes eleven
CLI checks, seven full8B CPU/meta variants and physical import checks without
original artifact modules, using the existing pinned interpreter.
Its record is `results/qwen3_triton_portable_clone_cpu_v3.json`; a fresh package
installation or exported-tree GPU experiment is not claimed.

The accepted cohort includes an explicitly documented resource-failure
recovery. The rejected reverse-MLP attempt remains preserved and excluded;
ten clean roles were retained and the remaining roles used identical source
and protocol in their original order. Public fresh-reference reviewer routing
is tested separately on a clearly marked CPU fixture, with no new GPU/timing
experiment claimed. Historical source/export/tiny/1.7B timings remain separate.

## Environment and paths

The measured environment uses Python 3.11.16, torch 2.10.0+cu128,
triton 3.6.0, transformers 4.51.0, safetensors 0.8.0, numpy 2.4.6,
huggingface_hub 0.36.2 and pyarrow 26.0.0. Provide these in `.venv-research`.
The wrapper `scripts/research_python.sh` keeps caches and temporary files
under `.runtime-research`; invoke Python through it. Four RTX A6000 GPUs
(SM86) and a 41.875 GiB allocator budget per GPU are the current target.
Model loading also needs sufficient host RAM for four rank-local loads.

For a fresh Python 3.11 environment, the pinned installation commands are:

```bash
mkdir -p .runtime-research/tmp .runtime-research/pip
export TMPDIR="$PWD/.runtime-research/tmp"
export PIP_CACHE_DIR="$PWD/.runtime-research/pip"
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1
python3.11 -m venv .venv-research
.venv-research/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128'
.venv-research/bin/python -m pip install \
  --index-url https://pypi.org/simple -r requirements-qwen3-triton.txt
```

These specify the already measured package versions; a fresh installation
has not been replayed in a second environment. The existing shared research
environment is kept unchanged during experiments.

Fetch [official Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)
at revision `b968826d9c46dd6066d109eabc6255188de91218` into `weights/Qwen3-8B`.
Fetch the four parquet files from
[the pinned WikiText config](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1)
into `datasets/wikitext-parquet`. Weights and dataset bytes are not distributed
with this source. `results/qwen3_triton_input_recipe_v1.json` contains every
expected shard, config/tokenizer/index, parquet, exported-text and token hash.

Recreate all three token splits with the **8B tokenizer**:

```bash
TOKENIZERS_PARALLELISM=false bash scripts/research_python.sh \
  scripts/prepare_qwen3_triton_corpus.py \
  --model-dir weights/Qwen3-8B \
  --parquet-source datasets/wikitext-parquet \
  --output artifacts/reproduced_wikitext
```

Every split must reproduce its frozen token hash and count. For an already
verified text export, `--text-source <directory>` replaces `--parquet-source`.
The full local text-to-token replay with the 8B tokenizer is complete and
independently checked: 124253184 train, 262228 validation and 298958 test
tokens match all frozen text/token SHA256 values. The small curated report is
`results/qwen3_triton_corpus_replay_v1.json`. The parquet-to-text branch has
also reproduced the pinned validation split. A complete parquet export of
all splits has not been replayed in this environment.

Bind verified weights and corpus to new paths; this verifies every byte
against the pinned recipe and exercises the actual training validator:

```bash
bash scripts/research_python.sh scripts/configure_qwen3_triton_inputs.py \
  --model-dir weights/Qwen3-8B \
  --corpus-dir artifacts/reproduced_wikitext \
  --output artifacts/reproduced_inputs
```

Existing evidence and configuration are never overwritten. Rebinding does
not regenerate data, qualify model numerics or alter the training policy.

## Source and experiment commands

Check generated source hashes and all 399 official parameter shapes for
each of the seven Triton variants without initializing CUDA or reading
the original artifact sources:

```bash
bash scripts/research_python.sh scripts/check_qwen3_triton.py \
  --protocol artifacts/reproduced_inputs/protocol.json \
  --output artifacts/source_cpu_review.json
```

Inspect the complete fresh-process protocol, or execute it by omitting
`--plan`:

```bash
bash scripts/research_python.sh scripts/run_qwen3_triton_suite.py \
  --protocol artifacts/reproduced_inputs/protocol.json \
  --input-manifest artifacts/reproduced_inputs/input_manifest.json \
  --output artifacts/reproduced_8b_suite --plan
```

Execution requires exclusive GPUs. The shared research lease waits for
120 consecutive idle observations; monitoring records foreign contexts and
observation errors. Sixteen fresh processes cover HF, complete Triton base,
optimized and five single-switch ablations in opposite orders. Each runs
16 quality updates, full validation at 0/16 (512 complete blocks, 261632
shifted tokens), three warmup updates, eight measured optimizer steps and
a separate training CUDA trace. BF16 compute, FP32 masters/grads/moments,
disabled TF32/reduced-precision GEMM reduction, checkpointing, FSDP2 and
native fused AdamW are common to every arm.

Only the first HF role exports dense first-update gradients, losslessly
compressed and capped at 6 GiB per rank. Allow at least 34 GiB free storage;
do not commit these archives or traces. Other arms compare every gradient
coordinate at runtime. CPU review replays the reference archive and checks
recorded candidate gate arithmetic; candidate dense gradients are not kept.

After all processes finish the controller runs the independent reviewer.
It can also be invoked separately with
`scripts/review_qwen3_triton_suite.py --output <suite-directory>` through
the wrapper. Inspect numerical gates, short quality, CUDA coverage and paired
step/memory measurements separately. Failed gates remain visible and
`performance_adopted` stays false. Sixteen updates establish a short training
check, not validation convergence or full pretraining quality.

For isolated module forward/backward timings, the public controller captures
fresh real activations from a complete official 36-layer 8B forward, then
measures attention, QK-Norm/RoPE and MLP at layers 0/17/35, plus LM head and
causal loss. It runs 96 cases with two orders, five warmup VJPs and twenty
CUDA-event samples per part. Native HF and each Triton case use the same
inputs/upstreams. An untimed guarded VJP precedes timing; these local BF16
parameter-gradient measurements exclude FSDP/NCCL/optimizer.

The `attention` case covers the full `self_attn`, including Q/K/V/O
projections, QK-Norm and RoPE. The separate `norm_rope` case overlaps with
that measurement. Module timings and gains are not additive.

```bash
bash scripts/research_python.sh scripts/run_qwen3_triton_module_benchmark.py \
  --protocol artifacts/reproduced_inputs/protocol.json \
  --input-manifest artifacts/reproduced_inputs/input_manifest.json \
  --output artifacts/reproduced_8b_modules --plan
```

Omit `--plan` to acquire the shared GPU lease and execute capture, timing
and CPU review. This public entry's CLI/source checks have passed; its actual
GPU module run has completed96 cases with a clean monitor and independent
review in `results/qwen3_triton_v9_module_benchmark_v1.json`. All sixteen accepted public training roles pass the full paired independent
review. Keep the measured whole-step comparison separate from isolated
module timing. Results retain their
source identities. Once both independent reviews have passed, create the
comparison tables using the newly reproduced results:

```bash
bash scripts/research_python.sh scripts/summarize_qwen3_triton_suite.py \
  --suite artifacts/reproduced_8b_suite/independent_review.json \
  --modules artifacts/reproduced_8b_modules/measurements/independent_review.json \
  --output artifacts/reproduced_8b_summary
```

The JSON and Markdown include both orders, measured memory, short-quality
results, all five ablation effects and isolated module summaries. Rank-level
logits/loss/gradient errors, the per-parameter gradient contract and actual
CUDA kernel coverage remain in the JSON; the Markdown includes a numeric
error table. Dense candidate gradients are not archived or independently
replayed. The
summarizer rejects incomplete/failed reviews and changed source evidence.
It does not select a combined configuration or claim statistical significance
from two runs. Source licensing/publication remains pending; notices are in
`qwen3_triton/PROVENANCE.md`.

For an interrupted cohort, pass every retained recovery manifest with
`--recovery <suite-directory>/recovery_001/manifest.json` (repeat the option
for subsequent recoveries). The summarizer rejects undisclosed interruptions,
changed retained measurements and changed rejected-attempt evidence. The
local V9 run required such recovery after a reverse-MLP ownership failure;
that attempt is excluded from performance and remains preserved. A fresh
reproduction uses the normal sixteen-role command above.
