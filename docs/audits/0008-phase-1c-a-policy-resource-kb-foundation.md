# Audit 0008 — Phase 1C-A: Organizer Policy, Resource Governance, KB Forensics, and Deterministic Resolver Foundation

- **Date:** 2026-07-15
- **Author:** Claude (AI agent), for human review
- **Change type:** Foundation for external-resource + KB use, submission-position
  policy, and the first deterministic L4 resolver. No models, no datasets, no KB
  content, no network, no final predictions.

## 1. Objective and scope

Build everything required *before* downloading, ingesting, comparing, or using
external datasets and medical knowledge bases, while honouring the organizer's
partially disclosed and intentionally hidden conventions. Delivered: a versioned
organizer-policy registry; a submission-position policy framework + forensics; an
external-resource governance layer; local-only RxNorm and Vietnamese ICD-10
snapshot interfaces with forensics; a public-NER manifest + label-mapping
foundation; and a deterministic Boundary & Type Resolver foundation over Phase 1B
proposals — all wired into one offline `doctor` CLI.

**Explicitly out of scope (not implemented / not run):** neural NER, model
training, continued pretraining, LoRA, LLM inference, dense embeddings, FAISS,
full RxNorm/ICD/NER downloads, final ICD/RxNorm prediction, organizer
`output.zip`, leaderboard submission, assertion classification, Phase 2 encoders.
No network calls. No datasets/KB/weights/indexes/outputs committed.

## 2. Git state before implementation

Branch `main`, clean working tree, latest commit
`4cfa83e feat: implement deterministic medical specialists` (Phase 1B, hardened).

## 3. Organizer-confirmed facts encoded

`configs/organizer/confirmed_facts_v1.yaml` (13 facts, each with a confirming
source): external datasets usable for training subject to license; the team may
need to provide data/generation process; RxNorm is a 2026 version; ICD-10 is
Vietnamese; the organizer will not provide the KB files; TEST_NAME/TEST_RESULT
may occur independently; TEST_RESULT need not be numeric; pairing not required in
output; repeated symptoms are distinct; assertions are multi-label; plus the
previously confirmed offset example, submission layout, and label set. Loaded by
`mednorm_vi.organizer_policy`; a hypothesis is never presented as a confirmed rule
(`PolicyHypothesis.is_confirmed` is always `False`).

## 4. Unresolved policies encoded

`unresolved_policies_v1.yaml` (20 policies, all open) covers position coordinate
space, char-vs-byte, line-ending, interval, matching mode; RxNorm release,
distribution, active/suppressed handling, remap, TTY priority, missing-strength,
range-strength, combination-drug; ICD source/version, dotted-vs-undotted,
specificity; descriptive TEST_RESULT boundary, procedure handling, symptom
boundary, isHistorical. Each carries status, supporting observation,
contradicting evidence, confidence, test method, a `leaderboard_experiment_id`
slot, and an `internal_default`. Extra hypotheses live in
`rxnorm_decoding_hypotheses_v1.yaml`, `icd_format_hypotheses_v1.yaml`, and
`historical_policies_v1.yaml` (36 hypotheses total). See ADR 0002.

## 5. PositionPolicy design

`mednorm_vi.position`: `PositionPolicy`, `PositionEncodingResult`,
`PositionProvenance`, `PositionPolicyRegistry`. The internal invariant
`original_text[start:end] == text` is unchanged and still enforced by L1/proposal/
resolver validators. An encoder re-expresses a raw code-point span into a
submission coordinate with reversible provenance and **never edits entity text**.
Registered policies: `raw-codepoint-half-open` (default, internal + diagnostic),
`utf8-byte-half-open`, `canonical-lf-codepoint`, `canonical-crlf-codepoint`,
`separator-reconstruction-codepoint`, `normalized-view-codepoint` (only with a
reversible `OffsetAlignment`, else fails clearly). Validation checks: registered
policy, exact raw invariant, deterministic re-encode, and reversible round-trip.
See `docs/position/POSITION_POLICY.md`.

## 6. Position-forensics design

`position.forensics.analyze` compares every registered policy against a
user-provided sample input/output and reports exact/near matches, systematic
(modal) deltas, per-line cumulative shift, max offset, and byte-vs-code-point /
line-ending / interval evidence. It contains no organizer sample text; tests use
synthetic fixtures. CLI: `position-forensics --input --output`.

## 7. Resource manifest schema

`mednorm_vi.resources`: `ResourceManifest`, `LicenseRecord`, `SourceRecord`,
`SnapshotIdentity`, `ChecksumRecord`, `RedistributionPolicy`, `AcquisitionRecord`,
`TransformationRecord`, `DerivedArtifactRecord` (+ NER `NerDatasetManifest`,
`LabelMapping`, `LabelMappingSet`). Tracked templates under `configs/resources/`
and `schemas/resources/RESOURCE_MANIFEST.md`; raw/derived data under `data/**`
(git-ignored) with tracked READMEs and `data/manifests/*.yaml`.

## 8. License / provenance policy

License status ∈ `LICENSE_UNKNOWN`, `REVIEW_REQUIRED`, `PERMITTED_INTERNAL_USE`,
`REDISTRIBUTION_RESTRICTED`, `REJECTED`. `validate_manifest` enforces governance
completeness — it fails a resource used without a version, source, checksum,
license review status, or intended-use — and makes **no automatic legal
conclusion** (a human sets the status). A resource is *usable* only under
`PERMITTED_INTERNAL_USE` / `REDISTRIBUTION_RESTRICTED`. CLI:
`validate-resource-manifest [--ner]`. See `docs/resources/OVERVIEW.md`.

## 9. RxNorm local snapshot interface

`mednorm_vi.kb.rxnorm`: RRF discovery + readers (RXNCONSO/RXNREL/RXNSAT/RXNSTY),
`RxnormSnapshot` (atoms/relations/attributes/semantic types indexed by RXCUI with
preferred string, TTYs, suppression, remap targets), content-derived snapshot id,
and integrity validation. Local-only; tiny synthetic RRF fixtures; no real RxNorm
content; no final decode.

## 10. RxNorm comparison and compatibility logic

`diff_snapshots` reports concepts in both, label/TTY changes, active↔suppressed
transitions, replaced/remapped RXCUIs, and concepts unique to one snapshot;
`resolve_current` follows remap relations to a stable current RXCUI (provisional;
`UNRES-RXNORM-REMAP`). Decoding-policy hypotheses are recorded, not selected. CLI:
`inspect-rxnorm-snapshot`, `compare-rxnorm-snapshots`. See `docs/kb/rxnorm/`.

## 11. ICD-10 Vietnamese local snapshot interface

`mednorm_vi.kb.icd10`: a **configurable** tabular loader (`ColumnMap`, not a
hard-coded source), `IcdConcept` with reversible dotted/undotted forms, Vietnamese/
English labels, aliases, parent/children, chapter; `normalization` for
dotted↔undotted and specificity; integrity validation. No WHO-vs-CM assumption
unless a manifest declares it (`UNRES-ICD-SOURCE`).

## 12. ICD comparison and formatting logic

`diff_snapshots` reports missing codes, format (dotted vs undotted supplied),
label, hierarchy, alias, and specificity differences. No output dotted/undotted
policy is chosen (`UNRES-ICD-DOTTED`, `UNRES-ICD-SPECIFICITY`). CLI:
`inspect-icd-snapshot`, `compare-icd-snapshots`. See `docs/kb/icd10/`.

## 13. Public NER dataset manifest foundation

`NerDatasetManifest` records original labels, proposed MedNorm-VI mappings,
incompatible labels, boundary-convention differences, assertion/normalization
availability, redistribution + privacy constraints, split provenance, leakage
risk, and permitted use. Templates for ViMedNer / ViMQ / PhoNER_COVID19 under
`configs/resources/ner/` — **no license terms are claimed** (status
`LICENSE_UNKNOWN`/`REVIEW_REQUIRED`). No dataset downloaded; no training pipeline.

## 14. Label-mapping foundation

`LabelMapping` (source → canonical intermediate → MedNorm target or `DROP`/
`REVIEW`) and `LabelMappingSet` (versioned, deterministic `config_hash`).
`validate_ner_manifest` rejects a mapping target outside the MedNorm entity types
∪ {DROP, REVIEW}.

## 15. Deterministic resolver foundation

`mednorm_vi.resolution`: consumes Phase 1B proposals + relations and emits
`EntityHypothesis` objects with a chosen boundary, retained alternatives,
boundary/type/overlap evidence, rejection reason, retained `has_result`
pair-group ids, deterministic score, and status (accepted/rejected/unresolved).
Only `THUỐC`/`TÊN_XÉT_NGHIỆM`/`KẾT_QUẢ_XÉT_NGHIỆM` are resolved; no CHẨN_ĐOÁN/
TRIỆU_CHỨNG is invented. Boundary policy is configurable
(`configs/resolution/resolver_v1.yaml`); medication default is the **full** span
(grounded in the confirmed golden example), test-result default `value_only`.
Guarantees: exact offsets; never dedup by text; repeated occurrences distinct;
same-type overlaps resolved deterministically (cross-type kept); abstention
available; **no ontology candidate emitted** (validated). See
`docs/resolution/RESOLVER.md`.

## 16. TEST_NAME / TEST_RESULT independence

Regression tests confirm: an unpaired TEST_NAME survives; an unpaired TEST_RESULT
survives; a `has_result` relation is optional internal evidence; the resolver
deletes neither an unpaired name nor an unpaired result; and result kinds
numeric / qualitative / short-descriptive are all handled. Long descriptive
result boundaries are explicitly **deferred** to later span/boundary models.

## 17. Synthetic fixtures

`tests/fixtures/kb/rxnorm/{snapshot_a,snapshot_b}` (RRF: ingredient + SCD + a
suppressed then remapped legacy CUI + a new brand);
`tests/fixtures/kb/icd10/{snapshot_a.csv,snapshot_b.csv,columns.yaml}` (3-/4-char
codes, dotted/undotted supplied forms, parent-child, Vietnamese aliases, label
changes). All synthetic; no real KB/organizer content.

## 18. Tests and exact outputs

```
python3 -m pytest -q   ->  433 passed in ~8s
ruff check .           ->  All checks passed!
python3 -m mypy        ->  Success: no issues found in 150 source files
git diff --check       ->  clean
```

New suites: `test_organizer_policy`, `test_position_policy` (mapping, CRLF/LF
shift, UTF-8 byte vs code-point, reversibility, forensics), `test_resource_
governance` (invalid manifest, unknown license, NER + label mapping),
`test_kb_rxnorm` (RRF ingestion, comparison, legacy remap), `test_kb_icd10`
(ingestion, dotted/undotted, comparison), `test_resolution` (name/result
independence, repeated occurrence, offsets, no ontology, boundary/overlap), and
`test_phase1c_doctor` (determinism, no-network, missing-KB flags).

## 19. Doctor CLI output

```
MedNorm-VI Phase 1C-A — doctor
==============================================
organizer registry     : 13 facts, 20/20 open policies, 36 hypotheses
position policies      : 6 registered (default raw-codepoint-half-open)
resource manifests     : 0 tracked, 2 templates
public NER manifests   : 3
RxNorm snapshot        : MISSING (local)
ICD-10 VI snapshot     : MISSING (local)
resolver               : ready
network access         : none
missing local resources:
  - RxNorm snapshot under data/external/rxnorm/ (RXNCONSO.RRF)
  - ICD-10 VI table under data/external/icd10_vi/ (*.csv)
determinism hash       : 23a6d2fba8e3c6252a136a1c8e946c1d018b675ff1c8eb295d5500b5cbd67a96
```

## 20. Determinism result

`doctor --json` twice → **byte-identical**. `resolve-phase1b-debug-output` twice
on the synthetic document → **byte-identical**. The organizer registry, position
registry, and resolver config all expose deterministic `config_hash`es; snapshot
ids are content-derived.

## 21. No-network confirmation

`grep -rE "requests|urllib|http|socket|aiohttp"` over all new packages returns
nothing. No command downloads anything; all operate on explicit local paths. The
doctor report carries `no_network: true`.

## 22. Items intentionally deferred

Neural NER + training + LoRA + LLM inference + embeddings/FAISS; full RxNorm/ICD/
NER downloads; assertion classification; final ICD/RxNorm prediction; organizer
`output.zip`; long descriptive TEST_RESULT boundaries; snapshot indexing for large
real KBs; leaderboard probes that would resolve the open policies.

## 23. Known limitations

- No policy is resolved: every hypothesis awaits leaderboard evidence.
- KB lookups iterate the snapshot (fine for synthetic sizes; real releases need
  indexing — documented).
- The RxNorm remap RELA set and the ICD parent inference are documented
  approximations.
- Resolver scores are deterministic heuristics, not probabilities; the overlap
  winner rule is heuristic (abstention available).

## 24. No data / KB / models / outputs / indexes tracked

Confirmed. Raw RxNorm/ICD/NER payloads and derived artifacts stay under `data/**`
(git-ignored); only READMEs, `.gitkeep`, and `data/manifests/*.yaml` track. No
`models/`, `indices/`, `outputs/`, `reports/` content added. Synthetic KB
fixtures live under `tests/fixtures/`, not `data/`.

## 25. Assistant-specific files untracked

`.claude/` remains git-ignored (`git check-ignore .claude/` confirms). No
`CLAUDE.md` / `AGENTS.md` tracked.

## 26. Architecture PDFs unchanged

Neither `docs/MedNorm-VI_Architecture.pdf` nor the Vietnamese copy was modified or
moved (`git status --short -- '*.pdf'` empty).

## 27. `git status --short`

```
 M .gitignore
 M docs/deterministic_baseline/PHASE1B.md
?? configs/organizer/
?? configs/resolution/
?? configs/resources/
?? data/derived/          (README only)
?? data/external/         (READMEs only; raw payloads ignored)
?? data/manifests/        (README only)
?? docs/decisions/0002-organizer-facts-and-unresolved-policies.md
?? docs/kb/  docs/position/  docs/resolution/  docs/resources/
?? schemas/resources/
?? src/mednorm_vi/{organizer_policy,position,resources,kb,resolution,phase1c_foundation}/
?? tests/fixtures/kb/
?? tests/unit/test_{organizer_policy,position_policy,resource_governance,kb_rxnorm,kb_icd10,resolution,phase1c_doctor}.py
```
(Everything uncommitted, left for human review.)

## 28. Is Phase 1C-A ready for review?

**Yes.** Organizer policy registry, position framework + forensics, resource
governance, RxNorm + ICD local snapshot interfaces + forensics, NER manifest +
label-mapping foundation, and the deterministic resolver are implemented and
wired into the doctor CLI; pytest/ruff/mypy/`git diff --check` are green; doctor
and resolve are byte-deterministic; no network, data, KB, models, indexes, or
outputs are tracked; PDFs untouched; assistant files ignored.

## 29. Recommended manual acquisition steps

Follow `docs/resources/ACQUISITION.md`: (1) acquire a licensed RxNorm RRF release
into `data/external/rxnorm/` and a Vietnamese ICD-10 table into
`data/external/icd10_vi/`; (2) copy a template from `configs/resources/` to
`data/manifests/`, fill version/source/SHA-256/license/intended-use, and have a
human set the license status; (3) `validate-resource-manifest`; (4)
`inspect-*`/`compare-*` to characterise the snapshot. Exact external resources
still missing: **the RxNorm 2026 RRF release and the Vietnamese ICD-10 table**
(both undisclosed; both must be acquired manually and licensed).

## 30. Recommended next milestone

**Phase 1C-B — deterministic linking baseline over frozen permitted snapshots:**
exact/fuzzy RxNorm + ICD-10 candidate retrieval driven by the resolver's
hypotheses and the decoding-policy hypotheses, plus the first leaderboard probes
to begin resolving the open position / RxNorm / ICD policies — still no neural
models and no final submission.
