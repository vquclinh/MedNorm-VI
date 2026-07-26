# Audit 0030 - S1 Authoritative Preflight and Astral Protection

- **Date:** 2026-07-26
- **Author:** Codex (AI agent), for human review
- **Change type:** Restore Audit 0029 immutability, add a real-component local
  `align-preflight` CLI, protect non-BMP characters across the VnCoreNLP JVM boundary,
  and validate S1 with a project-supported mypy toolchain.
- **Spec:** `docs/MedNorm-VI_Architecture.pdf` v1.1, read in full and not modified.
  The relevant contracts are §4 original-offset preservation and reversible
  normalization, §15 S1 training strategy, and Appendix A fail-fast validation.
- **Status:** `PREFLIGHT_EXECUTED_PASSED - READY_FOR_SMOKE_V5`.

## 1. Objective and scope

Finish the S1 pre-smoke milestone without rewriting committed audit history. The only allowed
state transition is: authoritative full-corpus S1 alignment preflight has now passed with the
real local components; smoke-v5 is the next Colab action.

No model weights were loaded, no local training or fine-tuning ran, no GPU was used, no
organizer inference ran, and no `output.zip` was produced.

## 2. Audit 0029 immutability

Audit 0029 is already committed in `HEAD` and is append-only. Claude had left uncommitted
edits in:

```text
docs/audits/0029-s1-orthographic-equivalence-and-full-corpus-preflight.md
```

Those working-tree edits were restored back to the exact committed `HEAD` version before any
new documentation was added. `git diff -- docs/audits/0029-s1-orthographic-equivalence-and-full-corpus-preflight.md`
is empty after the restore. Audit 0029 remains immutable and unchanged.

## 3. Corrected composition-direction finding

Audit 0029's implementation remains valid, but one causal explanation is superseded here.
The authoritative real-component run showed that RDRSegmenter did **not** compose the observed
decomposed source sequences. The old aligner composed the `original_text` side to NFC and then
compared that composed copy against the segmenter's still-decomposed output, causing the
mismatch.

The general canonical-equivalence fix remains valid because it is direction-agnostic:
composed-vs-decomposed clusters are accepted only when their NFD structure is equivalent under
the explicit Vietnamese mark rules, while offsets still index untouched `original_text`.

## 4. Astral-character JVM boundary defect

The first authoritative real-component preflight exposed one real failure:

```text
privacy_safe_example_id  f892d0fc934943c3
source                   vimedner
split                    train
stage                    subtoken_encoding
reason_code              SEGMENTER_ALTERED_CHARACTERS
```

The source contains `U+1D6C3 MATHEMATICAL BOLD SMALL BETA`, a non-BMP code point. Across the
VnCoreNLP JVM/UTF-16 boundary it returned as `U+D6C3`, i.e. the low 16 bits of the original
code point. That is character corruption, not Vietnamese orthographic equivalence. The
annotation is valid, so governed exclusions remain empty and no dataset-specific exception was
added.

## 5. General sentinel protection

`resolve_segmented_text()` now protects non-BMP characters before calling VnCoreNLP and
restores the exact original code points immediately afterward:

- each astral input code point maps to one BMP private-use sentinel code point (`U+E000`,
  `U+E001`, ...);
- offsets remain indices into untouched `original_text`;
- the aligner sees restored real text, never sentinels;
- lost, duplicated, reordered, or corrupted sentinels raise
  `SEGMENTER_ALTERED_CHARACTERS`;
- text already containing the reserved BMP private-use sentinel range is rejected safely when
  astral protection is needed;
- the truncated `U+D6C3` character is never accepted as an orthographic equivalent;
- there is no example-id or dataset-specific special case.

The CLI also loads optional heavy dependencies (`py_vncorenlp`, `transformers`) dynamically
inside the preflight-only functions, matching the notebook pattern and keeping the base package
free of static heavy imports.

## 6. Authoritative preflight results

The user-provided authoritative passing run used Fedora Linux 43, kernel 7.0.13; CPU only with
`CUDA_VISIBLE_DEVICES=""`; Python 3.14.5; `torch` not installed; `py_vncorenlp==0.1.4`;
`transformers==4.57.6`; `tokenizers==0.22.2`; OpenJDK 25; real VnCoreNLP RDRSegmenter with no
whitespace fallback; real slow `PhobertTokenizer` (`is_fast=False`, `vocab_size=64000`); pinned
model revision `f89e80b461e86f9cfc1c84019bd819830c24b6c5`.

Counters:

```text
examples_considered             24844
aligned_example_count           24844
unalignable_example_count           0
governed_exclusion_count            0
equivalence_checked_count       24844
equivalence_failure_count           0
counters_reconciled              true
alignment_preflight_passed       true
reason_code_counts                 {}
stage_counts                       {}
```

Splits:

```text
train       23799 / 23799
validation   1045 / 1045
```

Sources:

```text
vietmed_ner  9267 / 9267
vimedner     6702 / 6702
vimq         8875 / 8875
```

The user-provided runtime was 36.9 seconds wall, 30.0 seconds in-preflight, peak RSS
approximately 460 MB. The report at
`reports/local_alignment_preflight/alignment_preflight.json` had SHA-256:

```text
b9b3ae6ec4b293737a67c245fa7354f804de711e05d33ffb6881804eb6c2ab69
```

Codex reran the same authoritative local preflight after the final code changes, with
`HF_HUB_OFFLINE=1` to force the cached tokenizer files and avoid downloads. It passed with the
same counters. The final persisted ignored report records `runtime.elapsed_seconds: 29.7`;
`/usr/bin/time -v` recorded 35.28 seconds wall and 445,064 KB max RSS. Its SHA-256 is:

```text
78cbed16eb3ac0c12e66bb32011df4110bee63c31851275f59b8966ab870ba8e
```

The SHA differs from the user-provided report only because the report JSON includes elapsed
runtime.

## 7. Governed corpus verification

Governed corpus files remained ignored/untracked and unchanged. Verified counts and hashes:

```text
public_ner_examples.jsonl  35916 lines  9e8060775d97713485c964df9165f640050a0085f385b2a9ff216c75f9d81588
splits/train.jsonl        33826 lines  892dc22d7e051e05f9c96d90f42dfde7f38083a74bba6fe65b5c1d9dd05e2a4a
splits/validation.jsonl    1045 lines  ed7cdd2d49799cef0a868b6c75a3df4ca1e93ed03223337a7d31afe40f68f103
splits/internal_test.jsonl 1045 lines  e23acde0d87154d1750d25d1e47c92c86e457e42f2f97cb985661253dea7e135
corpus_manifest.json                  a3fd365d523b0ac54aab408ee14d00f88aefa42691fc810c445f8528e89fb8e3
split_manifest.json                   ceb962b54c8d74eb89a016571a350987d1bfd226d883f871ad057b9997c53778
source_contributions.json             9eb82426b24d3f71c61b59674e88f015085ba5797e7c0c210ac39f4131ab1fb8
annotation_coverage.json              b33614a1bf1f2216530a85acfbf61ac66bc9de0e8ccecb59d7382b46067f59f9
label_distribution.json               3e5ed4ccf98281712e17700f65aa2979f92a46f4c595687fc56ef6f97a6dc61c
leakage_summary.json                  aada8b15fc89b838533ee7650b9bfffc5cfb54c6762d0eb2468b4382a148b558
offset_validation.json                17f581b7f49f160288ca709b2993643d02781fb95068f2bd1727894509d2193f
s1_governed_exclusions.yaml           a949ce2101132482e7b409cbb59eaf2578f35cfe9ed2855a54967e6475efe8cf
```

`configs/training/s1_governed_exclusions.yaml` remains `exclusions: []`.

## 8. Mypy root cause and solution

`pyproject.toml` declares `requires-python = ">=3.10"` and mypy target
`python_version = "3.10"`. The Python 3.14 preflight `.venv` contains optional heavy/runtime
packages for the real preflight, including `numpy==2.5.1`, `transformers==4.57.6`, and
`py_vncorenlp==0.1.4`. Static imports from the preflight-only CLI made mypy follow installed
optional package stubs; NumPy's stubs use Python 3.12 type-statement syntax, which is invalid
under the repository's configured 3.10 target.

This was treated as a local development-toolchain incompatibility, not an application type
error. The clean validation toolchain is a separate Python 3.12 env at
`/tmp/mednorm-vi-dev-py312` with only the declared lightweight dependencies:
`PyYAML==6.0.3`, `pytest==9.1.1`, `ruff==0.16.0`, and `mypy==2.3.0`. It intentionally does not
install NumPy, Transformers, tokenizers, torch, or VnCoreNLP.

No mypy configuration was weakened. No NumPy errors were globally ignored. The project Python
target was not changed. Production runtime dependencies were not changed.

## 9. Files created

```text
docs/audits/0030-s1-authoritative-preflight-and-astral-protection.md
```

## 10. Files modified

```text
docs/audits/README.md
docs/notebooks/notebook_execution_integrity.md
docs/training/first_colab_run.md
src/mednorm_vi/training/cli.py
src/mednorm_vi/training/phobert_alignment.py
tests/unit/test_s1_alignment_diagnostics.py
```

Audit 0029 was restored, not modified.

## 11. Tests and validation

```text
env PYTHONPATH=src /tmp/mednorm-vi-dev-py312/bin/python -m pytest -q
  -> 1042 passed, 1 skipped in 112.36s

/tmp/mednorm-vi-dev-py312/bin/python -m ruff check .
  -> All checks passed!

env PYTHONPATH=src /tmp/mednorm-vi-dev-py312/bin/python -m mypy
  -> Success: no issues found in 224 source files

env PYTHONPATH=src /tmp/mednorm-vi-dev-py312/bin/python -m compileall -q src
  -> clean

git diff --check
  -> clean

S1 notebook code-cell AST parse
  -> 4 notebooks, 55 code cells, 0 syntax errors

env PYTHONPATH=src CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 /usr/bin/time -v \
  .venv/bin/python -m mednorm_vi.training.cli align-preflight \
  --revision f89e80b461e86f9cfc1c84019bd819830c24b6c5 \
  --vncorenlp-dir /home/vquclinh/.cache/mednorm-vi/vncorenlp \
  --report reports/local_alignment_preflight/alignment_preflight.json
  -> exit 0, 24,844/24,844 aligned, zero failures, 35.28s wall, 445,064 KB max RSS

governed corpus hash/count verification
  -> hashes and line counts listed in section 7

git ignored/local asset verification
  -> .venv/, reports/local_alignment_preflight/, data/derived/training_corpora/,
     .mypy_cache/, .pytest_cache/, .ruff_cache/, and __pycache__/ are ignored;
     no unignored local cache files; no repo-local model-weight files
```

## 12. Deferred items and next step

Smoke-v5 has not run. Full S1 training has not run. The next Colab step is to run:

```text
notebooks/MedNorm_S1_Mention_FirstRun_Smoke.ipynb
```

Use Colab Pro GPU, `REPO_REF = "main"` after this audit is committed and pushed, run all once
for PASS 1, reconnect after the intentional restart, run all again for PASS 2, and return the
`s1_mention_first_run_smoke_v5` artifact directory plus its checkpoint SHA-256 for validation.

## 13. Git status snapshot

```text
## main...origin/main
 M docs/audits/README.md
 M docs/notebooks/notebook_execution_integrity.md
 M docs/training/first_colab_run.md
 M src/mednorm_vi/training/cli.py
 M src/mednorm_vi/training/phobert_alignment.py
 M tests/unit/test_s1_alignment_diagnostics.py
?? docs/audits/0030-s1-authoritative-preflight-and-astral-protection.md
```
