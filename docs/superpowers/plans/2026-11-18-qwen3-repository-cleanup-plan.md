# Qwen3 Triton Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the repository to one unversioned, runnable Qwen3-8B Triton training/inference implementation with explicit dependencies and one result summary.

**Architecture:** Preserve the verified implementation and move live code from versioned scripts into stable package modules. Public scripts become thin orchestration entry points; `qwen3_triton` owns implementation; historical scripts and generated result evidence are removed after the retained dependency closure passes tests.

**Tech Stack:** Python 3.11, PyTorch 2.10 FSDP2, Triton, Transformers Qwen3, pytest.

**Spec:** `docs/superpowers/specs/2026-11-18-qwen3-repository-cleanup-design.md`

## Global Constraints

- Preserve official Qwen3-8B loading, 36-layer/8,190,735,360-parameter checks, BF16 compute, FP32 master/gradient/moment policy, vocabulary-parallel loss, Triton collectives, and Triton AdamW behavior.
- Do not change kernel mathematics or distributed numerical order as part of cleanup.
- Do not modify model weights, datasets, runtime artifacts, or paths outside the repository.
- Do not push automatically.
- Keep only one human-readable result summary; Git history is the experiment archive.
- Work on an isolated branch/worktree and leave the original branch untouched.

## Review Focus

- Dynamic imports formerly relying on `scripts/` being on `sys.path` must resolve after moving helpers into `qwen3_triton`.
- CLI-relative paths must be resolved before `os.chdir()` and before CUDA initialization.
- The full runtime must not retain native/v13/adam comparison imports after those scripts are deleted.
- GEMM plan loading must use `configs/gemm_plan.json` with the same content and identity semantics.
- Documentation must not claim that the refactored tree has been GPU-requalified when only historical results exist.

---

### Task 1: Create an isolated cleanup worktree and record the baseline

**Files:**
- Copy into worktree: `docs/superpowers/specs/2026-11-18-qwen3-repository-cleanup-design.md`
- Copy into worktree: `docs/superpowers/plans/2026-11-18-qwen3-repository-cleanup-plan.md`

**Interfaces:**
- Consumes: clean commit `0a828a0` on `research/inference-v15`.
- Produces: isolated branch `cleanup/final-qwen3-8b` and a baseline test report.

- [ ] **Step 1: Verify the source checkout has no user changes other than the two planning documents**

Run: `git status --short`

Expected: only `docs/superpowers/` is untracked.

- [ ] **Step 2: Create an ignored `.worktrees/cleanup-final-qwen3-8b` worktree on branch `cleanup/final-qwen3-8b`**

Verify `.worktrees/` is ignored before creation; add it to `.gitignore` on the cleanup branch if required.

- [ ] **Step 3: Copy the approved spec and plan into the worktree**

- [ ] **Step 4: Run the CPU-safe baseline**

Run: `pytest -q`

Expected: record exact pass/fail/skip counts. Stop for an unexpected baseline failure rather than attributing it to cleanup.

- [ ] **Step 5: Commit the planning baseline**

Commit: `docs: define final repository cleanup`

---

### Task 2: Move live training helpers into stable package modules

**Files:**
- Create: `qwen3_triton/training/model.py`
- Create: `qwen3_triton/validation/training_reference.py`
- Modify: `qwen3_triton/training/__init__.py`
- Modify: `qwen3_triton/validation/__init__.py`
- Test: `tests/test_training_module_boundaries.py`

**Interfaces:**
- Consumes: `scripts/vocab_model_v12.py`, helper functions from `scripts/train_qwen3_triton_8b.py`, and the vocabulary-parallel/FSDP APIs already used by the final runtime.
- Produces: `shard_vocab_loss_model(...)`, `compare_first_gradients(...)`, `sha(path)`, `local(tensor)`, and any shared reference-evaluation helper under package imports.

- [ ] **Step 1: Add failing import-boundary tests**

Assert that package modules import without adding `scripts/` to `sys.path`, expose the interfaces above, and that no file under `qwen3_triton/` imports a module rooted at `scripts`.

- [ ] **Step 2: Run the boundary test and confirm failure**

Run: `pytest tests/test_training_module_boundaries.py -q`

Expected: failure because the stable modules do not exist.

- [ ] **Step 3: Move model assembly from `scripts/vocab_model_v12.py` to `qwen3_triton/training/model.py` without changing behavior**

- [ ] **Step 4: Move reusable reference/hash/local helpers out of `scripts/train_qwen3_triton_8b.py` into `qwen3_triton/validation/training_reference.py`**

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_training_module_boundaries.py tests/test_hf_parity.py tests/test_training_shared_helpers.py -q`

Expected: all pass.

- [ ] **Step 6: Commit**

Commit: `refactor: stabilize training support modules`

---

### Task 3: Replace versioned launchers with six stable public commands

**Files:**
- Create/rename: `scripts/prepare_corpus.py` from `scripts/prepare_qwen3_triton_corpus.py`
- Create/rename: `scripts/configure_inputs.py` from `scripts/configure_qwen3_triton_inputs.py`
- Create: `scripts/export_training_reference.py` as a real file based on `scripts/train_qwen3_vocab_v12.py`
- Create/rename: `scripts/train.py` from `scripts/train_qwen3_runtime_v14.py`
- Create/rename: `scripts/generate.py` from `scripts/generate_qwen3_triton_v15.py`
- Create/rename: `scripts/verify_inference.py` from `scripts/verify_qwen3_inference_v15.py`
- Keep: `scripts/research_python.sh`
- Modify: `tests/test_triton_input_root.py`
- Create: `tests/test_public_cli.py`

**Interfaces:**
- Consumes: stable package helpers from Task 2.
- Produces: six supported CLI entry points, each with a CPU-safe `--help` path.

- [ ] **Step 1: Add failing CLI smoke tests**

Invoke every public Python script with `--help`; assert exit code 0. Assert `scripts/train.py` binds CLI-relative protocol, manifest, output, and reference paths before changing directory or touching CUDA.

- [ ] **Step 2: Run the CLI tests and confirm failure**

Run: `pytest tests/test_public_cli.py tests/test_triton_input_root.py -q`

- [ ] **Step 3: Rename the preparation, generation, verification, and training entry points**

Update imports to use `qwen3_triton.training.model` and `qwen3_triton.validation.training_reference`. Replace symlinks with regular tracked files.

- [ ] **Step 4: Convert the v12 reference runner into `scripts/export_training_reference.py`**

Keep only reference export behavior required by `TRAINING.md`; remove historical variant/controller behavior not used by the final path.

- [ ] **Step 5: Remove native/v13/adam comparison choices from `scripts/train.py`**

The public training command always uses the complete Triton model/runtime. Remove imports of `fsdp_peer_rs_v13.py` and `symmetric_rs_v13.py`; retain only full-runtime reporting fields that describe the final path.

- [ ] **Step 6: Run focused CLI and import tests**

Run: `pytest tests/test_public_cli.py tests/test_triton_input_root.py tests/test_training_module_boundaries.py -q`

Expected: all pass.

- [ ] **Step 7: Commit**

Commit: `refactor: expose stable Qwen3 commands`

---

### Task 4: Stabilize configuration and eliminate result-file runtime dependencies

**Files:**
- Move: `results/qwen3_triton_v10_gemm_plan_001.json` to `configs/gemm_plan.json`
- Modify: `configs/fsdp_8b.json`
- Modify: `scripts/train.py`
- Modify: retained inference/model configuration references
- Create: `tests/test_final_configuration.py`

**Interfaces:**
- Consumes: selected GEMM plan bytes and current Qwen3-8B protocol.
- Produces: all runtime configuration under `configs/`, with no dependency on `results/` or the old `qwen_operator_lab` path.

- [ ] **Step 1: Add failing configuration tests**

Assert that the training default points to `configs/gemm_plan.json`, the plan identity/content is unchanged, the model is Qwen/Qwen3-8B, and no config value references `qwen_operator_lab` or `results/*.json`.

- [ ] **Step 2: Run the test and confirm failure**

Run: `pytest tests/test_final_configuration.py -q`

- [ ] **Step 3: Move the GEMM plan and update defaults**

- [ ] **Step 4: Normalize `configs/fsdp_8b.json` paths and remove historical status text while preserving the numerical protocol**

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_final_configuration.py tests/test_public_cli.py -q`

- [ ] **Step 6: Commit**

Commit: `refactor: consolidate final runtime configuration`

---

### Task 5: Delete historical code and evidence outside the final dependency closure

**Files:**
- Delete: all tracked `results/*.json`
- Delete: all `scripts/*` except the six public Python commands and `scripts/research_python.sh`
- Delete: `benchmarks/`, `examples/`, and legacy `model/`
- Delete: `TRITON_QWEN3.md` after merging any unique useful statement into `README.md`
- Delete from `training/`: everything except `__init__.py`, `data.py`, `evaluation.py`, `input_manifest.py`, `fsdp/__init__.py`, `fsdp/official.py`, and `fsdp/activation_checkpoint.py`
- Delete: `ops/` if a fresh dependency scan confirms no retained import; otherwise retain only exact imported files and document why.
- Modify/delete tests whose only subject is removed legacy code.
- Modify: `tests/test_repository_layout.py`

**Interfaces:**
- Consumes: verified stable entry points and configuration from Tasks 2-4.
- Produces: a repository whose tracked implementation is exactly the final dependency closure.

- [ ] **Step 1: Add repository-layout assertions before deletion**

Assert that no tracked Python filename contains `_v<digits>` or a historical attempt suffix, exactly six public Python scripts exist, no retained package imports `scripts`, no source references `results/*.json`, and no symlink launcher remains.

- [ ] **Step 2: Run layout tests and confirm they fail against the historical tree**

- [ ] **Step 3: Run the static dependency analyzer from all six public entry points and save the retained closure list in the task log**

- [ ] **Step 4: Delete the files/directories listed above**

Use `git rm` with explicit repository-relative targets. Do not delete ignored artifacts, model weights, datasets, `.runtime-research`, or anything outside the worktree.

- [ ] **Step 5: Remove stale imports, strings, and tests revealed by compilation**

Run: `python -m compileall -q qwen3_triton training scripts tests`

Expected: exit 0.

- [ ] **Step 6: Run layout and CPU-safe tests**

Run: `pytest -q`

Expected: all retained tests pass; GPU-only skips are explicitly counted.

- [ ] **Step 7: Commit**

Commit: `chore: remove historical experiment archive`

---

### Task 6: Rewrite documentation around the final runnable system

**Files:**
- Modify: `README.md`
- Modify: `TRAINING.md`
- Modify: `INFERENCE.md`
- Rewrite: `docs/RESULTS.md`
- Modify: `requirements-qwen3-triton.txt`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: final public commands, configuration paths, and retained historical aggregate measurements.
- Produces: one coherent usage path and one result summary without per-attempt evidence files.

- [ ] **Step 1: Add documentation-link and command-reference tests to `tests/test_repository_layout.py`**

Assert every referenced repository path exists, only the six public script names appear, and no v10-v15 implementation filename is documented.

- [ ] **Step 2: Rewrite the root README as the project map**

Document purpose, supported 8B scope, installation, the six commands, package layout, and explicit limitations.

- [ ] **Step 3: Rewrite training and inference guides using only stable command names**

- [ ] **Step 4: Collapse results into `docs/RESULTS.md`**

Keep aggregate historical training/inference measurements, hardware/protocol, and caveats. State that the reorganized source requires fresh GPU requalification.

- [ ] **Step 5: Align dependency metadata**

Ensure `pyproject.toml` project name/description and the requirements file describe the final repository rather than `qwen-operator-lab` experiments.

- [ ] **Step 6: Run documentation/layout tests**

Run: `pytest tests/test_repository_layout.py tests/test_public_cli.py -q`

- [ ] **Step 7: Commit**

Commit: `docs: document final Qwen3 Triton runtime`

---

### Task 7: Final verification and review

**Files:**
- No intended source changes; fix only defects found by verification.

**Interfaces:**
- Consumes: cleaned worktree.
- Produces: evidence that the final tree is internally consistent and a reviewable cleanup branch.

- [ ] **Step 1: Verify repository shape**

Run:

```bash
git status --short
git ls-files scripts
git ls-files results
git ls-files | grep -E '(^|/).*(v[0-9]+|_[0-9]{3})([^/]*)$'
```

Expected: only intended changes before commit; six Python scripts plus `research_python.sh`; no result JSON; no implementation-version filenames.

- [ ] **Step 2: Compile all retained Python**

Run: `python -m compileall -q qwen3_triton training scripts tests`

Expected: exit 0.

- [ ] **Step 3: Run the complete CPU-safe suite**

Run: `pytest -q`

Expected: all retained tests pass with exact pass/skip counts recorded.

- [ ] **Step 4: Run CLI smoke tests explicitly**

Run each public command with `--help`; expected exit 0 without CUDA initialization or data access.

- [ ] **Step 5: Re-run dependency closure and dead-reference scans**

Assert no missing local import, no package-to-script import, no deleted path reference, and no dependency on `results/`.

- [ ] **Step 6: Review the full Git diff and file statistics**

Run: `git diff --stat origin/main...HEAD` and `git diff --check`.

Expected: no whitespace errors, generated artifacts, weights, datasets, or unrelated changes.

- [ ] **Step 7: Commit any verification-only fixes and report GPU validation status honestly**

Do not run four-GPU qualification while the shared server is overloaded. If it is not run, state that historical aggregate results remain but the cleaned tree still requires fresh GPU qualification.
