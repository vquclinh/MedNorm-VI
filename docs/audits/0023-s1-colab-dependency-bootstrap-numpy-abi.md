# Audit 0023 — S1 Colab Dependency Bootstrap and NumPy ABI Integrity Fix

- **Date:** 2026-07-26
- **Author:** Claude (AI agent), for human review
- **Change type:** Correctness fix for the S1 Colab dependency bootstrap. **No commit. No
  push. No local training, no model/tokenizer/VnCoreNLP downloads, no local GPU, no
  external API. Governed corpus files untouched.**
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1 (S1 training stage, offline/fail-fast,
  reproducibility). No architecture change; backbone, optimizer, and smoke bounds unchanged.
- **Status:** `IMPLEMENTED_UNEXECUTED` — requires a **new fresh Colab GPU runtime** run.

Audit 0022 is committed (`187ffe5 fix: support ViHealthBERT slow-tokenizer alignment`) and is
**not modified** by this milestone.

## 1. Verified Colab failure

The repaired S1 notebook passed corpus, tokenizer, segmentation, alignment, and model-loading
stages on a fresh Colab GPU runtime, then failed constructing the optimizer:

```text
optimizer = torch.optim.AdamW(model.parameters(), lr=limits.learning_rate)
  ...
numpy/random/_pickle.py      from .mtrand import RandomState
numpy/random/mtrand.pyx      in init numpy.random.mtrand
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
            Expected 96 from C header, got 88 from PyObject
```

## 2. Root cause

This is a **NumPy C-ABI/runtime environment failure**, not an AdamW, model, loss, GPU,
HF_TOKEN, or tokenizer-alignment defect.

"Expected 96 from C header, got 88 from PyObject" is the classic NumPy 1.x/2.x ABI mismatch:
a compiled extension built against one NumPy `dtype` struct layout is loaded against a
different NumPy runtime.

**Why the AdamW line exposed it.** Everything before it — corpus JSONL reading, the slow
PhoBERT tokenizer, and the Audit 0022 character-alignment backend — is pure Python and never
touches a compiled NumPy extension. `torch.optim.AdamW` is simply the first statement whose
import chain reaches `numpy.random.mtrand` (a compiled module). The environment was already
broken from the first cell; AdamW only revealed it.

### Contamination path in the committed notebook

Raw-cell inventory of the committed notebook found **three separate pip transactions**, an
install **in the same cell as the scientific imports**, and **no kernel restart anywhere**:

| Cell | Command | Problem |
| --- | --- | --- |
| code #3 | `pip install -q PyYAML>=6.0` | 1st mid-session environment mutation |
| code #7 | `pip install -q transformers==4.44.2 accelerate==0.33.0 sentencepiece==0.2.0` **then** `importlib.import_module("torch")` / `("transformers")` **in the same cell** | 2nd mutation; **torch/NumPy imported immediately afterwards with no restart** |
| code #8 | `pip install -q py_vncorenlp==0.1.4` | 3rd mutation, **after** NumPy/Torch were already loaded |

Nothing constrained NumPy or Torch. Any of those resolutions could rewrite NumPy on disk
(older `transformers`/`accelerate`/`py_vncorenlp` metadata can request `numpy<2` against
Colab's NumPy 2.x image), after which the already-imported or subsequently-imported compiled
extensions no longer matched. Restart mechanisms present in the notebook: **none**
(`os.kill`, `os._exit`, `restart_runtime`, `kernel.restart` all absent).

`pyarrow` was correctly **not** installed by S1.

## 3. New two-phase bootstrap

**PHASE A** (before any scientific import):
- configuration and repository checkout use **stdlib only** (`subprocess` git clone);
- dependency inspection uses `importlib.metadata` — **NumPy is never imported**;
- the Colab baseline versions of numpy/torch/torchvision/torchaudio/transformers/tokenizers/
  huggingface_hub/… are captured;
- `build_pip_constraints()` writes a constraints file pinning the **detected baseline** of
  the protected packages, so pip cannot silently move NumPy/Torch (an incompatible request
  now fails loudly instead of corrupting the ABI);
- **one consolidated install transaction** runs with `--constraint`;
- `pip check` is recorded;
- a **full-fingerprint** marker is written to **`/content/.mednorm_s1_bootstrap.json`**
  (not Drive) — see §3.1;
- the kernel is then **really restarted** via `os.kill(os.getpid(), 9)` — Colab reconnects.

**PHASE B** (after the restart):
- the marker is re-validated against the live runtime; only a **complete** match skips
  installation;
- the inherited stack is verified against the marker's baseline (protected packages must be
  unchanged);
- NumPy/Torch are imported here for the **first time**.

Expected behaviour is documented prominently in the notebook header:
**PASS 1 = install + restart, PASS 2 = Run all again**.

## 3.1 Marker integrity binding

A marker is only evidence that *some* install once happened. Accepting it on
`dependency_contract_version` alone was too weak: the same version string survives an edited
contract, a different Python minor version, a rebuilt Colab image with a different NumPy, or
a changed requirement set — exactly the drift that produces the ABI failure in §1.

The marker now persists and **must match in every field**:

| Field | Source | What it detects |
| --- | --- | --- |
| `install_completed` | writer | a truncated/aborted PASS 1 |
| `dependency_contract_version` | contract | a declared contract revision |
| `dependency_contract_sha256` | SHA-256 of the **exact bytes** of `configs/training/s1_mention_colab_dependencies.yaml` | any edit to the tracked contract, even one that keeps the version string |
| `python_major_minor` | live `sys.version_info` | a runtime image on a different Python (different ABI tag) |
| `protected_baseline_versions` | detected versions of `numpy`, `torch`, `torchvision`, `torchaudio` | the inherited stack moved under the marker |
| `install_requirement_hash` | SHA-256 over the canonical JSON of the sorted install specifiers + sorted protected-package policy + sorted inherited list + sorted `not_installed` list | a changed requirement set or protection policy |

`marker_mismatches()` returns the list of reasons a marker cannot be trusted;
`decide_bootstrap_action()` returns `PROCEED` **only** when that list is empty and
`INSTALL_AND_RESTART` otherwise. A missing field short-circuits (legacy markers are rejected
without ever comparing values), so an old version-only marker can never skip installation.

The hashes are order-independent and deterministic — canonical JSON with sorted keys and
sorted lists — so an identical contract always yields an identical hash across runs and
machines.

**No restart loop:** the marker is written with the *current* fingerprint immediately before
the restart, so PASS 2 matches exactly and yields `PROCEED`; repeated evaluation of a matching
marker is proven idempotent by test. The fingerprint is also re-validated inside the restarted
kernel (PASS 1 wrote it in a different process), and the notebook prints the expected
fingerprint and any mismatch reasons as diagnostics on both passes.

## 4. NumPy / AdamW ABI preflight

Immediately after the restart and **before** Drive mount, corpus gate, VnCoreNLP acquisition,
tokenizer acquisition, model weights, or batch preparation, the notebook executes real
operations:

```python
import numpy as np
from numpy.random import RandomState
rng = RandomState(42); values = rng.rand(4); assert values.shape == (4,)
import torch
dummy_parameter = torch.nn.Parameter(torch.zeros(1))
dummy_optimizer = torch.optim.AdamW([dummy_parameter], lr=1e-3)
dummy_optimizer.zero_grad(set_to_none=True)
```

It also records NumPy version/path/core path/`mtrand` path, Torch version/path, CUDA
availability, Python version, `pip check`, and the NumPy distribution count, then applies
`validate_abi_report()`, which flags failed imports, failed AdamW construction, duplicate
NumPy distributions, a failed `pip check`, and NumPy imported from Drive or the Git checkout.
On failure it prints diagnostics (no credentials) and raises — **no automatic in-process
repair**, and nothing is acquired.

## 5. Dependency contract

`configs/training/s1_mention_colab_dependencies.yaml`
(`dependency_contract_version: s1-colab-deps-v1`).

- **Inherited from the Colab runtime (never installed/upgraded/reinstalled):** `numpy`,
  `torch`, `torchvision`, `torchaudio`, `scipy`, `pandas`, `scikit-learn`.
- **Installed in one constrained transaction:** `transformers>=4.44,<5`,
  `accelerate>=0.33,<2`, `sentencepiece>=0.2,<1`, `py_vncorenlp==0.1.4`, `PyYAML>=6.0`.
- **Not installed:** `pyarrow` (S1 reads governed-corpus JSONL only).
- **Forbidden:** `--force-reinstall` on numpy/torch, blind `numpy<2`, unconstrained
  `--upgrade`, CPU Torch over the Colab CUDA build, any torch reinstall.

Ranges replace the previous brittle exact pins so pip can find a build compatible with the
runtime's NumPy 2.x / Python 3.12; the constraints file guarantees that if none exists, pip
**fails loudly** rather than downgrading NumPy.

**Honest limitation:** `py_vncorenlp==0.1.4`'s dependency metadata could not be inspected
offline in this environment, so whether it requests NumPy is unverified. The constraint file
is the protection: it is installed inside the same constrained transaction, so it cannot move
NumPy/Torch regardless. The real Colab run is what confirms resolution succeeds.

## 6. Notebook execution order (now enforced by tests)

```text
configuration (no scientific imports)
 -> repository checkout (stdlib git)
 -> dependency metadata inspection (importlib.metadata)
 -> ONE consolidated constrained installation
 -> forced kernel restart            [PASS 1 ends here]
 -> post-restart dependency verification
 -> NumPy/AdamW ABI preflight
 -> runtime, GPU, Drive mount
 -> governed-corpus gate
 -> registry / smoke subsets
 -> transformers import (no install)
 -> VnCoreNLP acquisition + verification
 -> slow tokenizer acquisition
 -> tokenizer equivalence preflight
 -> alignment preflight
 -> model acquisition
 -> bounded forward/backward/optimizer smoke
 -> checkpoint save/reload
 -> manifest write
```

Verified mechanically: **1** install transaction, **1** restart call, **no** scientific import
before the ABI preflight, and the real optimizer remains
`optimizer = torch.optim.AdamW(model.parameters(), lr=limits.learning_rate)`.

## 7. Manifest and success contract

`training_manifest.json` gains an `environment` block: `dependency_contract_version`,
**`dependency_contract_sha256`**, **`install_requirement_hash`**, **`python_major_minor`**,
**`protected_baseline_versions`**, **`bootstrap_action`**, **`bootstrap_marker_path`**,
**`bootstrap_marker_mismatches`**, `dependency_restart_completed`,
`numpy_abi_preflight_passed`, `python_version`, `numpy_version`, `numpy_path`,
`numpy_mtrand_path`, `torch_version`, `torch_path`, `transformers_version`,
`tokenizers_version`, `py_vncorenlp_version`, `pip_check_passed`. Every smoke run therefore
records which marker hashes were accepted and which decision they produced.

`full_training_readiness` is computed by the tracked helper
`evaluate_full_training_readiness()` and requires **all** of: production VnCoreNLP
segmentation; ≥1 tokenizer-equivalence example; zero equivalence failures; zero unalignable
examples; dependency restart completed; NumPy ABI preflight passed; finite train loss;
backward completed; optimizer step completed; validation completed; checkpoint saved;
checkpoint reloaded. A failed environment preflight therefore can never produce a success
manifest.

## 8. Tests (no downloads, no environment mutation)

`tests/unit/test_colab_bootstrap.py` (49): contract loads and declares the inherited stack;
protected packages are never installed; pyarrow excluded; malformed contract rejected;
constraints pin the **detected baseline**
and never emit a blind `numpy<2`; constraints require a baseline; install command is
consolidated + constrained; violations rejected (`--force-reinstall`, `--upgrade`, torch
install, missing constraints); ABI report flags each failure mode, duplicate NumPy, failed
`pip check`, and Drive/checkout-shadowed NumPy; readiness true only when everything holds and
false for each individual failure.

Marker integrity is proven field by field:

| Test | Proves |
| --- | --- |
| `test_exact_marker_match_proceeds` | exact marker match → `PROCEED` |
| `test_marker_prevents_restart_loop` | repeated evaluation of an exact marker stays `PROCEED` — **no restart loop** |
| `test_changed_contract_hash_forces_reinstall` | changed `dependency_contract_sha256` → `INSTALL_AND_RESTART` |
| `test_changed_python_major_minor_forces_reinstall` | changed Python major/minor → `INSTALL_AND_RESTART` |
| `test_changed_protected_baseline_forces_reinstall` | changed protected baseline (e.g. NumPy 2.0.2 → 1.26.4) → `INSTALL_AND_RESTART` |
| `test_changed_requirement_hash_forces_reinstall` | changed `install_requirement_hash` → `INSTALL_AND_RESTART` |
| `test_incomplete_marker_forces_reinstall` (parametrized over all 6 required fields) | any missing field → `INSTALL_AND_RESTART` |
| `test_legacy_version_only_marker_is_never_accepted` | the old version-only marker → `INSTALL_AND_RESTART` |
| `test_contract_sha256_is_the_hash_of_the_exact_file_bytes` / `..._changes_when_the_contract_bytes_change` | the hash is over the exact contract bytes |
| `test_install_requirement_hash_is_deterministic_and_order_independent` | the requirement hash is canonical and changes when requirements change |
| `test_fingerprint_carries_every_required_marker_field` | all of `REQUIRED_MARKER_FIELDS` are persisted |

`tests/unit/test_notebook_placeholders.py` (+10 S1 assertions): no NumPy/Torch import before
the ABI preflight; exactly one install and one restart; no `--force-reinstall`/`--upgrade`/
`numpy<2`/torch reinstall; ABI preflight precedes every acquisition step; preflight is
fail-fast (`RandomState(42)`, dummy AdamW, `assert not abi_problems`); real optimizer remains
AdamW; manifest records all environment fields including the marker hashes and decision;
the marker is bound to the full fingerprint (`build_marker_fingerprint`, the marker persists
`**MARKER_FINGERPRINT.as_dict()`, the decision uses `marker_mismatches`, and the fingerprint
is re-validated after the restart); two-pass execution documented.

A pure helper carries the logic so tests never touch the local environment.

## 9. Validation

```text
env PYTHONPATH=src python3 -m pytest -q          -> 748 passed, 1 skipped (pyarrow reader)
ruff check .                                      -> All checks passed!
env PYTHONPATH=src python3 -m mypy                -> Success: no issues found in 222 source files
env PYTHONPATH=src python3 -m compileall -q src   -> clean
git diff --check                                  -> clean
```

No local NumPy/Torch modification, no downloads, no training.

## 10. Exact changed files

Verified with `git status --short`, `git diff --name-status`, and `git diff --stat`.

**Added (4)** — untracked, `??`:
```text
configs/training/s1_mention_colab_dependencies.yaml   (dependency contract, s1-colab-deps-v1)
docs/audits/0023-s1-colab-dependency-bootstrap-numpy-abi.md
src/mednorm_vi/training/colab_bootstrap.py            (pure bootstrap + marker-fingerprint logic)
tests/unit/test_colab_bootstrap.py                    (49 tests)
```

**Modified (4)** — tracked, `M`:
```text
docs/audits/README.md                               (0023 index entry)
docs/notebooks/notebook_execution_integrity.md      (S1 status note)
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb   (two-phase bootstrap, fingerprint marker, restart, ABI preflight, reorder, manifest)
tests/unit/test_notebook_placeholders.py            (bootstrap/ABI/ordering/marker assertions; updated stale order test)
```

**Total changed tracked files: 8** (4 added + 4 modified). The earlier "Added (3)" count in
this audit was wrong — it listed four paths under a heading of three; the audit document
itself is an added file and is counted.

## 11. Honest status

`IMPLEMENTED_UNEXECUTED`. The bootstrap logic is unit-tested and the notebook ordering is
mechanically verified, but **the notebook has not been run**. No smoke success, checkpoint, or
training manifest may be claimed until the user completes a **new fresh Colab GPU runtime**
run — and that run requires **two passes** (install+restart, then Run all again).
