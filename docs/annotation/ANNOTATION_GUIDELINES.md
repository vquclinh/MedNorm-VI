# MedNorm-VI internal annotation guidelines (v1)

Guidance for producing **team-owned** development labels. Some rules are not yet
fixed by the organizer; those are marked **INTERNAL PROVISIONAL ANNOTATION RULE**
and must be revisited when the organizer publishes more examples. Machine-readable
companion: `configs/annotation/guideline_v1.yaml`.

> These guidelines apply to team data only. The organizer competition test set
> has no labels and must never be annotated (see `GOLD_SILVER_POLICY.md`).

## Offsets & text (confirmed)

- `text` is the **exact original substring**: `original_text[start:end] == text`.
- Offsets are **end-exclusive** `[start, end)` over **Unicode code points**.
- Never trim whitespace, re-case, or rewrite punctuation before storing a span.

## Entity types (confirmed — organizer Vietnamese labels)

`THUỐC` (medication), `CHẨN_ĐOÁN` (diagnosis), `TRIỆU_CHỨNG` (symptom),
`TÊN_XÉT_NGHIỆM` (test name), `KẾT_QUẢ_XÉT_NGHIỆM` (test result).

## Assertions (confirmed) — multi-label

`isNegated`, `isHistorical`, `isFamily`. Allowed on `THUỐC`, `CHẨN_ĐOÁN`,
`TRIỆU_CHỨNG` only. A cue only contributes **evidence**; scope/section/discourse
decide which entity actually receives an assertion — never label every entity
just because a cue is present.

## Per-type fields (confirmed)

- `THUỐC`, `CHẨN_ĐOÁN`: text, type, candidates, assertions, position.
- `TRIỆU_CHỨNG`: text, type, assertions, position.
- `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`: text, type, position.

## Duplicate, overlapping, repeated mentions

- The **same surface form at different positions** is **distinct** — annotate both
  (e.g. `táo bón` at `[397,404]` and `[443,450]`).
- An exact duplicate at the **same** position is rejected.
- **Nested/overlapping spans** — *INTERNAL PROVISIONAL ANNOTATION RULE*: allow
  only if organizer data supports them; otherwise prefer a single best span.

## Medication span boundaries — *INTERNAL PROVISIONAL ANNOTATION RULE*

Whether `strength`, `route`, `frequency`, `PRN`, and `release form` belong inside
the `THUỐC` span is **not yet fixed** by the organizer. Provisional default:
annotate the **drug name span** and record strength/route/frequency as separate
evidence during boundary study; generate multiple boundary candidates and let
error analysis on published examples decide. Do not hard-code a convention.

## Diagnosis vs symptom — *INTERNAL PROVISIONAL ANNOTATION RULE*

Prefer `CHẨN_ĐOÁN` when the phrase names a diagnosed condition (often after
"chẩn đoán"/"diagnosis" or with an ICD-mappable term); prefer `TRIỆU_CHỨNG` for
patient-reported findings. Ambiguous items go to adjudication.

## Test name vs test result — *INTERNAL PROVISIONAL ANNOTATION RULE*

`TÊN_XÉT_NGHIỆM` is the assay/measurement name (`WBC`); `KẾT_QUẢ_XÉT_NGHIỆM` is
its value/flag (`14.43`, `H`). In `name: value` rows, split at the delimiter and
keep both absolute spans.

## Lists & sections

- Split list items while retaining separators and absolute offsets.
- Sections are **evidence, not absolute rules**: a "pre-admission medication list"
  gives an `isHistorical` prior, but a local "started today" overrides it.

## Assertion scope (Vietnamese) — *INTERNAL PROVISIONAL*

- Negation (`không`, `chưa`, ...) typically stops at contrast markers (`nhưng`)
  unless context says otherwise.
- Historical cues may scope over a whole noun phrase, list, or section.
- Family experiencer may be far from the entity — use discourse, not proximity.

## Abbreviations & noise

Handle `THA`, `DTD`, `qhs`, unaccented variants, and VN-EN code switching via the
cue lexicons (below) plus fuzzy/contextual methods — not exact strings alone.

## Cue & section lexicon policy (high recall, versioned)

Maintain **broad, versioned** cue inventories; treat them as evidence sources, not
rules. Examples (non-exhaustive, versioned in `configs/annotation/guideline_v1.yaml`):

- **historical:** tiền sử · bệnh sử trước đây · từng mắc · đã từng · trước đây ·
  trước nhập viện · thuốc tại nhà · dùng từ trước · prior history · past medical
  history · PMH
- **family:** gia đình · tiền sử gia đình · bố · cha · mẹ · anh · chị · em ·
  họ hàng · người thân
- **negation:** không · chưa · không ghi nhận · phủ nhận · không thấy · âm tính
  với · loại trừ · chưa phát hiện

Rules: local discourse may override section priors; version every phrase;
support aliases, no-diacritic variants, abbreviations, misspellings, and code
switching; complement lexicons with fuzzy matching and classifiers; evaluate
every added cue for **both** recall and false positives. **Do not build the final
exhaustive lexicon in Phase 0.5** — only the extensible, versioned contract and
representative fixtures.

## Uncertain cases & adjudication

Mark uncertain items and route them through `ADJUDICATION_PROTOCOL.md`. Only
**ADJUDICATED** (or explicitly approved GOLD) records become development gold.

## Examples

All worked examples must derive only from **permitted or synthetic** data — never
from organizer test inputs. See `tests/fixtures/annotation/`.
