# Audit 0024 - S1 Scoped Dependency Health

- **Date:** 2026-07-26
- **Author:** Codex (AI agent), for human review
- **Change type:** Correctness fix for the S1 first-run Colab smoke dependency gate.
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1, read in full. The PDF was not
  modified, moved, renamed, regenerated, or replaced.
- **Status:** `IMPLEMENTED_UNEXECUTED` - the completed notebook still requires a fresh
  Colab Pro GPU two-pass run by the user. This audit does **not** claim the Colab smoke
  has passed.

Audit 0023 is committed at `8fa4281 fix: harden S1 Colab dependency bootstrap` and was
not rewritten.

## 1. Objective And Scope

Finish the partially implemented dependency-health fix left after Audit 0023 so that
`notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb` can continue when S1 itself is
healthy, even if unrelated preinstalled Colab packages make global `pip check` return
nonzero.

No local training, GPU work, model/tokenizer download, VnCoreNLP download, governed
corpus change, backbone change, organizer inference, `output.zip`, external API, commit,
or push was performed.

## 2. Observed Colab Output

The latest fresh Colab diagnostics were:

```text
returncode: 1

ipython 7.34.0 requires jedi, which is not installed.
gradio 6.20.0 has requirement huggingface-hub<2.0,>=1.2.0, but you have huggingface-hub 0.36.2.
```

S1 does not import or use IPython completion, Jedi, or Gradio.

## 3. Why NumPy/AdamW Was Healthy

The reported global `pip check` failure was not itself a NumPy ABI failure. The S1
health gate now keeps the mandatory direct ABI checks:

- `import numpy`;
- `from numpy.random import RandomState`;
- `RandomState(42).rand(4)` with shape verification;
- `import numpy.random.mtrand`;
- `import torch`;
- real dummy `torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)`;
- exactly one installed NumPy distribution;
- NumPy must not load from Drive or the Git checkout.

Only after these real operations pass can the notebook set `NUMPY_ABI_PREFLIGHT_PASSED`
and continue.

## 4. Why The Old Gate Was Too Broad

Audit 0023 correctly added a two-pass install/restart workflow and a fail-fast
NumPy/AdamW ABI preflight, but the environment report still treated global
`pip_check_passed=false` as a blocking ABI problem. That made the S1 smoke depend on the
health of the entire Colab image, including unrelated packages preinstalled by Colab.

The Gradio/Hugging Face Hub line is a reverse-dependency complaint raised by `gradio`,
not by `huggingface-hub`. S1 imports `huggingface_hub` for model/tokenizer resolution,
but it does not import Gradio. Moving `huggingface-hub` to satisfy Gradio could damage
the S1 Transformers stack and is forbidden by this milestone.

## 5. Final Scoped Policy

`pip check` still runs and its stdout/stderr are preserved, but the pass/fail policy is
scoped:

- compute the S1 dependency closure from governed import roots in
  `configs/training/s1_mention_colab_dependencies.yaml` plus installed distribution
  metadata;
- parse each `pip check` complaint and keep the original message;
- block only when the **complaining distribution** is inside the S1 closure;
- record complaints from outside the closure as non-blocking diagnostics;
- keep unknown pip-check lines, blocking them only when they mention a closure member;
- import every governed S1 module successfully before any training step;
- keep `pip_check_passed` as diagnostic evidence only.

Contract v2 roots are NumPy, Torch, Transformers, Tokenizers, Hugging Face Hub,
Safetensors, SentencePiece, Accelerate, py_vncorenlp, and PyYAML. Gradio, IPython, and
Jedi are deliberately not closure roots or remediation targets.

## 6. Blocking And Non-Blocking Examples

Non-blocking examples covered by tests:

```text
ipython 7.34.0 requires jedi, which is not installed.
gradio 6.20.0 has requirement huggingface-hub<2.0,>=1.2.0, but you have huggingface-hub 0.36.2.
ipython 8.12.3 requires jedi>=0.16, which is not installed.
gradio 5.49.1 has requirement huggingface-hub<1.0,>=0.28.1, but you have huggingface-hub 1.0.1.
gradio 6.20.0 requires huggingface-hub<2.0,>=1.2.0, but you have huggingface-hub 0.36.2 which is incompatible.
```

Blocking examples covered by tests:

```text
transformers 4.44.2 requires tokenizers<0.20,>=0.19, but you have tokenizers 0.21.0 which is incompatible.
transformers 4.44.2 requires tokenizers, which is not installed.
torch 2.8.0 requires filelock, which is not installed.
huggingface-hub 1.0.1 requires requests>=2.0, which is not installed.
```

The validator also blocks failed required imports, failed NumPy import, failed
RandomState, failed dummy AdamW, duplicate NumPy distributions, and Drive/checkout
NumPy shadowing.

## 7. Notebook And Manifest Changes

The notebook order remains:

```text
configuration and checkout
 -> dependency bootstrap pass
 -> forced restart
 -> post-restart verification
 -> scoped dependency health and NumPy/AdamW ABI preflight
 -> Drive and governed corpus
 -> VnCoreNLP
 -> slow tokenizer and alignment/equivalence
 -> model acquisition
 -> bounded forward/backward/AdamW smoke
 -> tiny validation
 -> checkpoint save/reload
 -> truthful manifest
```

The two-pass workflow remains intact: PASS 1 performs one constrained install and
restarts; PASS 2 accepts only a full matching marker and proceeds. No restart loop was
introduced.

Pass-1 marker diagnostics now preserve raw `pip_check_stdout`, raw `pip_check_stderr`,
and combined `pip_check_output` without truncation. Post-restart diagnostics are also
captured in full.

The final manifest records:

- `pip_check_passed`, `pip_check_returncode`, and complete `pip_check_output`;
- `s1_dependency_healthy`;
- `s1_dependency_closure`;
- `blocking_dependency_conflicts`;
- `non_blocking_dependency_conflicts`;
- structured `dependency_conflicts`;
- S1 import versions and failures;
- `s1_dependency_closure_verified`.

`full_training_readiness` now requires `s1_dependency_closure_verified=true`.

## 8. Tests Added Or Updated

Focused dependency-health tests now cover:

- exact latest Colab IPython/Jedi conflict as non-blocking;
- exact latest Colab Gradio/Hugging Face Hub conflict as non-blocking;
- equivalent IPython/Jedi and Gradio/Hugging Face Hub version/wording variants;
- Transformers/Tokenizers incompatibility as blocking;
- missing required S1 dependency as blocking;
- failed required module import as blocking;
- NumPy import, RandomState, dummy AdamW, duplicate NumPy, and shadowed NumPy as blocking;
- global `pip check` failure while scoped S1 health passes;
- readiness false when the S1 dependency closure is not verified;
- complete pip-check diagnostics retained;
- notebook has no executable remediation for Gradio/Jedi/Hugging Face Hub;
- notebook still uses real AdamW;
- two-pass bootstrap and previous tokenizer/alignment checks remain enforced.

## 9. Validation

Commands run locally, without installing or modifying packages:

```text
env PYTHONPATH=src python3 -m pytest -q tests/unit/test_colab_bootstrap.py
-> 79 passed in 0.32s

env PYTHONPATH=src python3 -m pytest -q tests/unit/test_colab_bootstrap.py tests/unit/test_notebook_placeholders.py
-> 116 passed in 0.54s

S1 notebook code-cell syntax parse
-> S1 notebook code cells parsed: 20

env PYTHONPATH=src python3 -m pytest -q
-> 779 passed, 1 skipped in 116.01s

ruff check .
-> All checks passed!

env PYTHONPATH=src python3 -m mypy
-> Success: no issues found in 222 source files

env PYTHONPATH=src python3 -m compileall -q src
-> clean

git diff --check
-> clean
```

The single skip is the existing local pyarrow-absence skip in
`tests/unit/test_vietmed_adapter.py`; S1 deliberately does not install pyarrow.

## 10. Exact Changed Files

Final changed-file inventory for this milestone:

```text
configs/training/s1_mention_colab_dependencies.yaml
docs/audits/0024-s1-scoped-dependency-health.md
docs/audits/README.md
docs/notebooks/notebook_execution_integrity.md
docs/training/first_colab_run.md
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb
src/mednorm_vi/training/colab_bootstrap.py
tests/unit/test_colab_bootstrap.py
tests/unit/test_notebook_placeholders.py
```

## 11. Remaining Risks

- The notebook remains `IMPLEMENTED_UNEXECUTED`; only a fresh Colab Pro GPU run can
  confirm dependency resolution and GPU/model/resource behavior in the live image.
- Future pip versions may introduce new diagnostic wording. Unknown lines are retained
  and fail closed when they mention an S1 closure member.
- Import roots are a governed contract. If S1 later imports new packages, the closure
  must be updated before the notebook can honestly claim readiness.
- VnCoreNLP resources, model weights, and the governed corpus are still Colab/Drive
  runtime concerns and were not downloaded locally.

## 12. Final Git Status Snapshot

Final `git status --short` snapshot:

```text
 M configs/training/s1_mention_colab_dependencies.yaml
 M docs/audits/README.md
 M docs/notebooks/notebook_execution_integrity.md
 M docs/training/first_colab_run.md
 M notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb
 M src/mednorm_vi/training/colab_bootstrap.py
 M tests/unit/test_colab_bootstrap.py
 M tests/unit/test_notebook_placeholders.py
?? docs/audits/0024-s1-scoped-dependency-health.md
```

## 13. Recommended Next Milestone

Run the completed S1 smoke notebook on a fresh Colab Pro GPU runtime using the documented
two-pass workflow. Return only `OUTPUT_DIR` for review: `checkpoint/s1_mention_smoke_model.pt`
and `training_manifest.json`, not base-model cache files.
