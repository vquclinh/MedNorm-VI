# BTC Updated Task Contract Review — turn2 / vong1

- **Date:** 2026-07-23
- **Scope:** Review the updated BTC task description against the repository's
  confirmed output contract (`src/mednorm_vi/schemas/constants.py`,
  `schemas/prediction.py`, `schemas/spans.py`, `inference/serialization.py`,
  `validator/`), the Architecture PDF, and existing tests. **Descriptive only —
  no model training, no thresholds tuned, no output generated.**

The updated input (`input_turn2_vong1.zip`) changed the clinical text of all 100
documents (Audit 0015) but did **not** change the output contract. Every confirmed
field below is already implemented and enforced by the repository.

## Confirmed output structure

- One JSON file per input file: `1.txt → 1.json … 100.txt → 100.json`.
- Each JSON file is one **list**; each list item is one concept dictionary with
  `text`, `position`, `type`, `assertions`, `candidates` (per-type field subset
  applies — see below).
- Entity types (exact Vietnamese labels, `ORGANIZER_LABEL_BY_TYPE`):
  `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, `THUỐC`.
- Assertions (`ASSERTION_LABELS`, multi-label): `isNegated`, `isFamily`, `isHistorical`.
- Candidates: `CHẨN_ĐOÁN → ICD-10`, `THUỐC → RxNorm` (`CANDIDATE_ONTOLOGY_BY_TYPE`;
  cross-links forbidden).

## Requirement classification

| Requirement | Classification | Notes / repository evidence |
| --- | --- | --- |
| One JSON per TXT | **UNCHANGED** | `N_SUBMISSION_DOCUMENTS = 100`; serializer/packaging emit `N.json`. |
| JSON list structure | **UNCHANGED** | `to_submission_json` emits a JSON array. |
| Entity dict fields | **UNCHANGED** | `EntityPrediction.to_submission_dict`. |
| Five entity labels | **UNCHANGED** | `ORGANIZER_LABEL_BY_TYPE` (exact VN strings). |
| Assertion scope (symptom/diagnosis/medication only) | **UNCHANGED** | `ORGANIZER_FIELDS_BY_TYPE`: assertions emitted only for `SYMPTOM`/`DIAGNOSIS`/`MEDICATION`. |
| Multi-label assertions | **UNCHANGED** | `assertions` is a subset tuple of `ASSERTION_LABELS`. |
| ICD candidates (CHẨN_ĐOÁN) | **UNCHANGED** | `CANDIDATE_ONTOLOGY_BY_TYPE["DIAGNOSIS"] == "ICD10"`. |
| RxNorm candidates (THUỐC) | **UNCHANGED** | `CANDIDATE_ONTOLOGY_BY_TYPE["MEDICATION"] == "RXNORM"`. |
| Patient information | **UNSUPPORTED_BY_OUTPUT_SCHEMA** | No patient-info type/field. See below. |
| Inter-concept relations | **UNSUPPORTED_BY_OUTPUT_SCHEMA** (secondary: AMBIGUOUS) | No `relations` field in the schema. See below. |
| Position convention | **CLARIFIED** (residual AMBIGUOUS) | Repo convention is half-open `[start, end)` end-exclusive, confirmed by a prior organizer example (`original_text[start:end] == text`). BTC "0…n-1" wording does not, alone, re-distinguish inclusive vs exclusive; the repo evidence resolves it to end-exclusive and is preserved. See below. |
| Laboratory-result boundaries | **CLARIFIED** | Supports value-only, value+unit, qualitative, and comparison; normal/abnormal terms (`bình thường`, `tăng nhẹ`) added to the lab lexicon. Units are not required. |
| Input file count/layout | **UNCHANGED** | 100 flat `.txt` documents, same names. |
| Output filename convention | **UNCHANGED** | `N.txt → N.json`. |
| Additional training-data wording | **AMBIGUOUS** | No labeled training data was provided in this drop; the input is unlabeled test input. No ground truth is present. |

## Patient information (§9)

The overview mentions patient information, but the output schema provides **no**
patient-information entity type or field (no age/gender/name/address/phone label).
Therefore patient information may be used **internally** as document context (e.g.
to support `isFamily`/temporality reasoning) but must **not** be serialized. Do not
invent `THÔNG_TIN_BỆNH_NHÂN` or `patient_information`, and do not add unsupported
JSON fields. Status: **UNSUPPORTED_BY_OUTPUT_SCHEMA** until BTC provides an explicit
type and output contract. (Enforced: `test_organizer_output_contract.py`.)

## Inter-concept relations (§10)

The overview mentions ontological reasoning and relationships between concepts, but
the output schema defines no `relations` field, source/target entity, relation
labels, or scoring. Do not invent a relation schema. Output remains entity-local
(`isNegated`, `isFamily`, `isHistorical`). Status: **UNSUPPORTED_BY_OUTPUT_SCHEMA /
AMBIGUOUS**.

## Position convention (§11)

Canonical internal convention (preserved, not changed): half-open `[start, end)`,
end-exclusive, over Unicode code points, always derived from absolute coordinates
(`SpanCoordinates.output_position`), with the invariant `original_text[start:end] ==
text` enforced by the validator. `POSITION_IS_END_EXCLUSIVE = True`, "CONFIRMED by
the organizer's published example."

Honest ambiguity: the updated BTC wording ("indexed 0 to n-1") does not, by itself,
carry a concrete JSON example that re-distinguishes `[start, end_inclusive]` from
`[start, end_exclusive)`. The prior organizer example resolves it to end-exclusive,
so the repository preserves that. **If** BTC later confirms inclusive output, the
change is localized: `output_end = internal_end_exclusive - 1` (a serializer policy
toggle), and no internal spans need to change. Boundary tests cover one-character
entities, start-at-zero, end-at-last-character, Vietnamese Unicode, line breaks,
empty spans, and invalid spans.

## Laboratory-result boundaries (§12)

`KẾT_QUẢ_XÉT_NGHIỆM` supports numeric value only, numeric value + unit, qualitative
results, and comparison results; units are **not** required universally. Decimal
comma (`14,43`) and decimal point (`7.2`) both handled. Normal/abnormal descriptors
`bình thường` and `tăng nhẹ` were added to the qualitative lexicon (coverage
expansion, **not** threshold tuning). Covered by `test_organizer_output_contract.py`
and `test_laboratory.py`.

## Conclusion

The updated task's **output contract is materially unchanged**. The only genuinely
open items are (a) patient-information output (unsupported), (b) inter-concept
relations (unsupported), and (c) the residual position end-inclusive/exclusive
wording ambiguity (resolved to end-exclusive by prior repository-local evidence).
No conclusion is drawn that models require retraining; the input change (Audit 0015)
implies later re-examination of L1/L2 coverage once BTC confirms the task, but that
is deferred.
