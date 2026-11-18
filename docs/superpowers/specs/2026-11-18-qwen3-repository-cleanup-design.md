# Qwen3 Triton Repository Cleanup Design

## Intent

Turn the current experiment archive into a maintainable Qwen3-8B Triton repository. The current branch mixes v10-v15 implementation files, experiment controllers, probes, manifests, and 305 result JSON files. Git history remains the archive; the working tree should expose only the final runnable training and inference paths plus one result summary.

## Goals

- Preserve the final Qwen3-8B Triton training and inference behavior.
- Make every public entry point unversioned and documented.
- Make the dependency graph explicit: scripts orchestrate; package modules implement.
- Retain input preparation, HF reference export, full training, generation, and inference verification.
- Keep tests that exercise the retained implementation.
- Keep exactly one human-readable result summary.

## Non-goals

- Redesign Triton kernels, distributed algorithms, or numerical policies.
- Claim new performance or convergence results.
- Preserve historical experiment runners in the current tree.
- Push changes to the remote repository.

## Chosen Approach

Use dependency-closure cleanup rather than a rewrite. Start from the final training, generation, verification, and input-preparation entry points; move still-live code out of versioned script names; update imports and documentation; then remove everything outside the retained closure.

The rejected alternative is a clean rewrite of the training/inference front ends. It would create a prettier tree but would unnecessarily risk changing FSDP, vocabulary-parallel, collective, precision, and validation behavior.

## Target Public Layout

```text
README.md
TRAINING.md
INFERENCE.md
pyproject.toml
configs/
  fsdp_8b.json
  gemm_plan.json
docs/
  RESULTS.md
scripts/
  prepare_corpus.py
  configure_inputs.py
  export_training_reference.py
  train.py
  generate.py
  verify_inference.py
qwen3_triton/
  model and configuration modules
  kernels/
  runtime/
  inference/
  training/
  validation/
tests/
```

`benchmarks/`, `examples/`, legacy `model/`, most of `ops/`, old `training/` helpers, historical scripts, controller/reviewer scripts, and result JSON files are removed unless retained by the verified final dependency closure.

## Stable Module Boundaries

### Scripts

Scripts contain argument parsing and orchestration only. They may import package modules, but package modules must not import scripts.

- `scripts/train.py`: the current full runtime v14 implementation, without a version label.
- `scripts/generate.py`: the current v15 generation implementation.
- `scripts/verify_inference.py`: the current v15 end-to-end inference verification.
- `scripts/export_training_reference.py`: HF reference export and the public reference CLI.
- `scripts/prepare_corpus.py` and `scripts/configure_inputs.py`: deterministic input preparation.

### Package

- `qwen3_triton/training/model.py`: vocabulary-parallel/FSDP model assembly currently stored in `scripts/vocab_model_v12.py`.
- `qwen3_triton/validation/training_reference.py`: reusable gradient/reference comparison helpers currently imported from `train_qwen3_vocab_v12.py`.
- Existing `qwen3_triton/kernels`, `runtime`, and `inference` modules retain implementation behavior, but filenames and imports lose version terminology where it remains.
- Native/v13/adam comparison controls are removed from the public training CLI unless they are required for the final full runtime. The final repository presents one production path, not historical comparison arms.

## Configuration and Results

- Move the selected GEMM tuning plan from `results/qwen3_triton_v10_gemm_plan_001.json` to `configs/gemm_plan.json` and update defaults.
- Remove all machine-generated JSON under `results/` after extracting any final facts needed by `docs/RESULTS.md`.
- Keep `docs/RESULTS.md` as the only result artifact. It must clearly distinguish historical measured results from the current refactored code and avoid claiming revalidation that has not occurred.
- Remove version numbers from experiment names, output metadata, documentation, and source paths when they describe the retained final implementation rather than an external schema.

## Deletion Policy

A tracked file is retained only if it is:

1. reachable from a final public entry point through static or verified dynamic imports;
2. required by a retained test;
3. a package/configuration/documentation file needed to install or understand the final system; or
4. the single final result summary.

Historical probes, diagnostic scripts, attempt manifests, reviewers, controllers, intermediate benchmarks, duplicate reference implementations, symlink aliases, and version-specific contract files are deleted. Git history is the recovery path.

## Verification

- No tracked Python filename contains implementation version suffixes such as `v12`, `v14`, or `v15`.
- No retained source/documentation references deleted files or `results/*.json` except `configs/gemm_plan.json`.
- All retained Python files compile.
- Static dependency analysis from every public entry resolves only retained files.
- CPU-safe tests pass.
- CLI `--help` smoke tests pass for all six public scripts.
- GPU/distributed tests are reported separately if the busy shared server prevents safe execution; no unrun GPU test is reported as passing.
- Git diff contains only the cleanup and dependency reorganization, with no generated artifacts or model data.

## Safety

- Work in an isolated Git worktree because the cleanup deletes hundreds of tracked files.
- Do not push automatically.
- Do not modify model weights, datasets, runtime artifacts, or files outside the repository.
- Preserve the original branch untouched until the cleaned tree is reviewed.
