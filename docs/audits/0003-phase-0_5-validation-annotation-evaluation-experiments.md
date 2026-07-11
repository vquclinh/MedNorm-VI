# Audit 0003 — Phase 0.5: Validation, Annotation, Evaluation & Experiment Foundation

- **Date:** 2026-07-11
- **Author:** Claude (AI agent), for human review
- **Change type:** New deterministic research infrastructure. No model logic, no
  downloads, no KB indices, no network, no heavy ML dependencies.

## 1. Objective and scope

Complete the deterministic research foundation before model-heavy work: (A) a
PROVISIONAL LOCAL EVALUATOR for the published metric on team-owned labeled data,
(B) an annotation + data-quality contract, (C) diagnostic + replay reporting, and
(D) local leaderboard experiment tracking. Phase 0.5 explicitly does **not**
implement L1–L8 model logic and adds no torch/transformers/faiss/vLLM/etc.

## 2. Files created

**Evaluation** (`src/mednorm_vi/evaluation/`): `__init__.py`, `models.py`,
`tokenization.py`, `wer.py`, `jaccard.py`, `scoring.py`, `aggregation.py`,
`diagnostics.py`, `loading.py`, `replay.py`, `reporting_json.py`,
`reporting_html.py`, `cli.py`, and `matching/{__init__,base,exact_position,
exact_text_occurrence,min_cost_bipartite}.py`.

**Annotation** (`src/mednorm_vi/annotation/`): `__init__.py`, `models.py`,
`validation.py`, `agreement.py`, `adjudication.py`, `export.py`.

**Experiments** (`src/mednorm_vi/experiments/`): `__init__.py`, `models.py`,
`hashing.py`, `registry.py`, `leaderboard.py`, `cli.py`.

**Configs**: `configs/evaluation/provisional_v1.yaml`,
`configs/annotation/guideline_v1.yaml`, `configs/experiments/default.yaml`.

**Docs**: `docs/evaluation/{LOCAL_EVALUATOR,METRIC_ASSUMPTIONS}.md`,
`docs/annotation/{ANNOTATION_GUIDELINES,GOLD_SILVER_POLICY,ADJUDICATION_PROTOCOL}.md`,
`docs/experiments/LEADERBOARD_EXPERIMENT_TRACKING.md`,
`reports/README.md`, `experiments/README.md`.

**Fixtures**: `tests/fixtures/evaluation/{gold,predictions}/` (8 synthetic docs
covering C1–C7, duplicates, wrong-type; gold has a manifest),
`tests/fixtures/annotation/gold/` (adjudicated gold + manifest),
`tests/fixtures/experiments/.gitkeep`.

**Tests**: `tests/unit/{test_wer,test_jaccard,test_matching,test_aggregation,
test_diagnostics,test_annotation_validation,test_annotation_agreement,
test_experiment_registry,test_replay}.py`;
`tests/integration/{test_evaluator_cli,test_full_synthetic_evaluation,
test_leaderboard_workflow}.py`.

**Namespaces / markers**: `data/{organizer_test,dev_gold,dev_silver,synthetic,
external_permitted}/.gitkeep`, `reports/.gitkeep`, `experiments/registry/.gitkeep`.

## 3. Files modified

`.gitignore` (reports/ + data namespaces + experiment-registry note),
`README.md` (competition reality, order of operations, Phase 0.5 tooling),
`data/README.md` (trust-level namespaces + competition reality),
`docs/decisions/0001-open-questions-pending-organizer-data.md` (Phase 0.5 note).

## 4. Published metric formulas implemented

- Final: `0.3·text + 0.3·assertions + 0.4·candidates`.
- Text: token-sequence Levenshtein WER, component `1 − WER` (raw preserved;
  optional explicit clipping).
- Assertions / candidates: set Jaccard with empty-set rules (∅/∅→1, ∅/≠∅→0,
  else `|∩|/|∪|`).
- Candidate aggregation weight: `len(GT candidates) + 1`.
- Wrong type: no cross-type credit; within-type-only matching yields the
  double error (one missing + one spurious).

## 5. Confirmed contracts vs provisional assumptions

Confirmed: final weights; WER family; Jaccard family + empty-set behavior;
wrong-type zero-credit; candidate weight expression. Provisional (configurable,
audited): entity matching algorithm, WER tokenization, `1−WER` clipping,
corpus/document aggregation, position's role in matching, candidate ordering/max
length. Documented in `docs/evaluation/METRIC_ASSUMPTIONS.md`. The tool is named
**PROVISIONAL LOCAL EVALUATOR** and never claims official-clone equivalence.

## 6. Organizer test ground truth is unavailable

The organizer does not release ground truth for the competition test set. The
evaluator scores team-owned labeled data only (GOLD/SILVER/SYNTHETIC/
ORGANIZER_PUBLISHED_EXAMPLE/EXTERNAL_PERMITTED). `ORGANIZER_TEST` is input-only
and can never be a labeled provenance; it is rejected in loading, annotation
validation, and manifest validation. Organizer inputs and labeled data use
separate `data/` namespaces.

## 7. Matching strategies

`exact-position` (same type + identical `[start,end)`), `exact-text-occurrence`
(same type + exact text, occurrence order, position tie-break; default),
`min-cost-bipartite` (deterministic pure-Python Hungarian assignment within a
type, configurable cost, `max_matching_cost` rejection). No numerical dependency
added. Wrong types never enter the same cost matrix.

## 8. Tokenization strategies

`whitespace`, `whitespace-punctuation` (default), `character-diagnostic`. No
normalization of organizer text. Default rationale documented (punctuation-dense
clinical text). Edge cases tested (empty/empty, empty/non-empty, WER>1,
Vietnamese Unicode, punctuation, repeated whitespace).

## 9. Aggregation policies

`provisional-v1` (default; micro over scoring slots, candidate-weighted — the
only one approximating the published score), `macro-document`,
`micro-entity-diagnostic` (diagnostic only). Every report shows matching,
tokenization, aggregation, clipping, evaluator version, and data provenance.

## 10. Annotation schema & review workflow

`AnnotationEntity` (organizer type, offsets, assertions/candidates where allowed,
annotator id, timestamp, provenance, review status, notes, source dataset id,
guideline version, route/section). Review lifecycle
`DRAFT→SINGLE_REVIEWED→DOUBLE_REVIEWED→ADJUDICATED` (or `REJECTED`); only
`ADJUDICATED` (or approved GOLD) counts as gold. Agreement utilities
(span/type/assertion/candidate + corpus) and adjudication records with integrity
validation. Export to evaluator GT files + manifest.

## 11. Gold/silver/synthetic provenance policy

`docs/annotation/GOLD_SILVER_POLICY.md`: GOLD (adjudicated, model selection),
SILVER (weak/pseudo, warned on scoring), SYNTHETIC (known-by-construction, tests),
EXTERNAL_PERMITTED. Pseudo-labels are never called ground truth. Dataset
manifests carry `provenance` + `labeled`; organizer-inputs manifests must set
`labeled: false`.

## 12. Cue-lexicon extensibility policy

`docs/annotation/ANNOTATION_GUIDELINES.md` §cues + `configs/annotation/
guideline_v1.yaml`: broad, **versioned** historical/family/negation inventories as
evidence sources (not rules); local discourse overrides section priors; aliases /
no-diacritic / abbreviations / misspellings / code-switching supported; fuzzy +
classifier complements; every added cue evaluated for recall AND false positives.
No exhaustive lexicon built in Phase 0.5 — only the extensible contract + fixtures.

## 13. Experiment & leaderboard tracking

Local, diff-friendly JSON registry (`EXP-NNNN.json`, sorted keys). Records code/
config/model/adapter/data/KB versions, seed, thresholds, output ZIP + per-file
hashes, and **separate** local gold/silver/synthetic scores vs a **manually**
entered leaderboard score. Deterministic id allocation; overwrite refused. CLI:
`create`, `attach-output`, `record-local`, `record-leaderboard`, `show`,
`compare`. No network / no scraping (asserted by a test).

## 14. Reports generated

Per run: `summary.json`, `per_document.json`, `entity_pairs.jsonl`,
`unmatched_entities.jsonl`, `diagnostics.jsonl`, `config_snapshot.json`,
`replay_manifest.json`, and a self-contained `report.html` (no external
JS/CSS/fonts/CDN; all user text HTML-escaped; PROVISIONAL banner; scores; by-type
and by-case tables; worst documents; error categories; matched / unmatched
tables). Data reports are deterministic; only the replay manifest + HTML banner
carry a timestamp.

## 15. Tests and exact results

```
python3 -m pytest -q   ->  159 passed in ~1.5s
ruff check .           ->  All checks passed!
python3 -m mypy        ->  Success: no issues found in 59 source files
```

Coverage includes: exact match → perfect score; both Jaccard sets empty → 1;
GT∅/pred≠∅ → 0; WER sub/ins/del; WER>1 preserved; explicit clipping; Vietnamese
Unicode; punctuation tokenization; duplicate candidate/assertion diagnostics;
same text at two positions distinct; wrong type no-match / double error; missing
+ spurious; candidate weighting; the three matchers + determinism; deterministic
JSON reports + replay manifest; HTML escaping; UTF-8 in JSON/HTML; end-exclusive
offsets; malformed prediction/GT failure; organizer-test-not-labeled; full
synthetic 100-document run; annotation valid/invalid/agreement/adjudication;
experiment create/ids/attach/record/compare/no-overwrite/git-state/no-network.

## 16. CLI smoke-test commands and outputs

**1. Evaluator** (`--ground-truth tests/fixtures/evaluation/gold
--predictions tests/fixtures/evaluation/predictions
--config configs/evaluation/provisional_v1.yaml --report-dir …`): exit 0.

```
PROVISIONAL LOCAL EVALUATOR
final provisional score : 0.689819
  text score            : 0.823529
  assertion score       : 0.730769
  candidate score       : 0.558824
matching strategy       : exact-text-occurrence
tokenization strategy   : whitespace-punctuation
aggregation strategy    : provisional-v1
data provenance         : SYNTHETIC
```
All 8 report files written.

**2. Annotation validation** (`--dataset tests/fixtures/annotation/gold
--manifest …/manifest.yaml`): `OK: annotations valid`, exit 0.

**3. Experiment workflow**: created EXP-0001/EXP-0002; attached synthetic
`output.zip` (sha256 recorded); recorded local gold 0.689819 and leaderboard
0.71234 (kept separate); `compare` printed both with the "leaderboard never
replaces local error analysis" note. Exit 0.

**4. Negative test** (manifest `provenance: ORGANIZER_TEST`, `labeled: true`):

```
error: manifest.organizer_test_labeled: manifest declares provenance
ORGANIZER_TEST with labeled: true; ... can never be labeled ...
FAILED: 1 error(s)
```
Exit 1 (correctly rejected).

## 17. Negative organizer-test-label test result

PASS — both `validate_manifest` and the annotation CLI reject
`ORGANIZER_TEST + labeled: true` with code `manifest.organizer_test_labeled`
(nonzero exit). Loading + annotation-entity validation reject `ORGANIZER_TEST`
provenance as well.

## 18. Items intentionally deferred

L1–L8 model logic; any model/dataset/KB download; ICD/RxNorm index building;
KB-membership validation of candidate codes; exact organizer matching/tokenization
/clipping/aggregation (surfaced as provisional switches); a web annotation UI.

## 19. Risks and open questions

Provisional metric may diverge from the organizer's private evaluator — mitigated
by explicit, audited switches and the PROVISIONAL naming. Open questions from ADR
0001 (matching mode, WER tokenization, candidate ordering/limit, position role)
remain open. Silver/synthetic scores must not be mistaken for gold — enforced by
separate provenance + warnings.

## 20. No models/datasets/KB indices downloaded

Confirmed. `grep` for `torch|transformers|faiss|vllm|requests|urllib.request|
sklearn|numpy|pandas` over `src/` returns nothing. The only runtime dependency
remains `PyYAML` (`pyproject.toml`). No network calls anywhere.

## 21. PDF confirmation

Both PDFs are **unmodified** (`git status --short -- '*.pdf'` empty). The
Vietnamese PDF remains **ignored/untracked** (`git check-ignore` →
`MedNorm-VI_Architecture_Vietnamese.pdf`). The English PDF was not modified.

## 22. `git status --short`

Snapshot after the post-audit housekeeping move (see §25); this audit now lives at
`docs/audits/0003-phase-0_5-validation-annotation-evaluation-experiments.md`:

```
 M .gitignore
 M AGENTS.md
 M CLAUDE.md
 M README.md
 D audits/0001-repository-bootstrap.md
 D audits/0002-organizer-contract-corrections.md
 M data/README.md
 M docs/decisions/0001-open-questions-pending-organizer-data.md
?? configs/annotation/
?? configs/evaluation/
?? configs/experiments/
?? docs/annotation/
?? docs/audits/
?? docs/evaluation/
?? docs/experiments/
?? experiments/
?? reports/
?? src/mednorm_vi/annotation/
?? src/mednorm_vi/evaluation/
?? src/mednorm_vi/experiments/
?? tests/fixtures/annotation/
?? tests/fixtures/evaluation/
?? tests/fixtures/experiments/
?? tests/integration/test_evaluator_cli.py
?? tests/integration/test_full_synthetic_evaluation.py
?? tests/integration/test_leaderboard_workflow.py
?? tests/unit/test_aggregation.py
?? tests/unit/test_annotation_agreement.py
?? tests/unit/test_annotation_validation.py
?? tests/unit/test_diagnostics.py
?? tests/unit/test_experiment_registry.py
?? tests/unit/test_jaccard.py
?? tests/unit/test_matching.py
?? tests/unit/test_replay.py
?? tests/unit/test_wer.py
```
The `D audits/0001…` / `D audits/0002…` deletions plus the new `?? docs/audits/`
directory are the audit-directory relocation. Everything remains **uncommitted**,
left for human review.

## 23. Is Phase 0.5 complete?

**Yes.** All four foundations (provisional evaluator, annotation/data-quality,
diagnostic+replay reporting, experiment tracking) are implemented, documented,
and tested; pytest/ruff/mypy are green; all four smoke tests pass; no models/
datasets/indices/network were introduced; both PDFs are untouched.

## 24. Recommended next milestone

**Phase 1 — L1 Document Intelligence:** reversible normalization, character
alignment, section parsing, sentence/list segmentation, and `DocumentGraph`
construction — now measurable against the provisional evaluator.

## 25. Post-audit housekeeping (path-only correction)

The audit directory was relocated from the root-level `audits/` to
**`docs/audits/`**. This is a path-only housekeeping change — no findings above
were altered. Audits 0001–0003 now live under `docs/audits/`; a new
`docs/audits/README.md` documents the numbering, no-overwrite, required-sections,
human-review, and no-auto-commit conventions. Active references were updated in
`CLAUDE.md`, `AGENTS.md`, `README.md`, and `.gitignore`
(`docs/audits/**/*.log|json` are ignored; numbered `.md` audits are tracked). The
binding rule now reads: “Every meaningful change must create the next numbered
audit Markdown file under `docs/audits/`.” Per the housekeeping instruction, **no
Audit 0004 was created** for this move; it is recorded here instead. `git status
--short` in §22 reflects the state after the move.
