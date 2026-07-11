# Adjudication protocol

How the team resolves annotation disagreements into GOLD.

## Review lifecycle

`DRAFT → SINGLE_REVIEWED → DOUBLE_REVIEWED → ADJUDICATED` (or `REJECTED`).
Only **ADJUDICATED** (or explicitly approved GOLD) records are used as final
development gold by default (`ACCEPTED_GOLD_STATUSES`).

## Double annotation

1. Two annotators (A, B) label the same document independently under
   `ANNOTATION_GUIDELINES.md` (record `guideline_version`).
2. Compute agreement with `mednorm_vi.annotation.agreement.document_agreement`:
   - exact-span agreement,
   - type agreement (over span-agreeing pairs),
   - assertion-set agreement,
   - candidate-set agreement.
3. Items where A and B agree on span **and** type advance to
   `DOUBLE_REVIEWED`; assertion/candidate mismatches still route to adjudication.

## Adjudication

For each disagreement, create an `AdjudicationRecord`
(`mednorm_vi.annotation.adjudication.make_adjudication`) preserving:

- annotator A, annotator B,
- disagreement reason,
- adjudicated result (a valid annotation, forced to `ADJUDICATED`),
- adjudicator id,
- guideline version,
- notes.

`validate_adjudication` enforces: adjudicator present, reason present, guideline
version present, result status is `ADJUDICATED`, and A ≠ B.

## Promotion to GOLD

Adjudicated records are exported to evaluator ground-truth files with
`mednorm_vi.annotation.export.export_gold_dataset`, which writes per-document
organizer-shaped JSON (plus optional `route_case`/`section` metadata) and a
`manifest.yaml` (`provenance: GOLD`, `labeled: true`). The result is then usable
by the provisional local evaluator.

## Escalation

Systematic disagreements feed back into the guidelines: update the relevant
**INTERNAL PROVISIONAL ANNOTATION RULE**, bump `guideline_version`, and re-run
regression checks so recall gains do not silently cost precision.
