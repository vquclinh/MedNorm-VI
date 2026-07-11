# Metric assumptions — confirmed vs provisional

The evaluator separates what the organizer has **confirmed** from what remains
**provisional**. Provisional choices are configurable in
`configs/evaluation/provisional_v1.yaml` and recorded in every report.

## Confirmed (published)

- **Final weights:** `final = 0.3·text + 0.3·assertions + 0.4·candidates`.
- **Text component:** Word Error Rate on the `text` field; component uses
  `1 − WER`.
- **Assertions component:** Jaccard similarity.
- **Candidates component:** Jaccard similarity.
- **Candidate aggregation weight:** `len(ground_truth_candidates) + 1`.
- **Jaccard empty-set rules:** GT∅ & pred∅ → 1; GT∅ & pred≠∅ → 0; else
  `|∩| / |∪|`.
- **Wrong type:** no partial cross-type credit; a correct-text/wrong-type
  prediction counts as one missing correct entity **and** one spurious wrong-type
  entity. Matching never aligns entities across different organizer types.

## Provisional (organizer implementation not published)

- **Entity matching algorithm.** We provide `exact-position`,
  `exact-text-occurrence` (default), and `min-cost-bipartite`. Default rationale:
  it relies only on confirmed text+type, keeps repeated mentions at different
  positions distinct, and assumes nothing about whether position participates in
  matching (itself an open question — ADR 0001).
- **WER tokenization.** `whitespace`, `whitespace-punctuation` (default),
  `character-diagnostic`. Default rationale: clinical text is punctuation-dense
  (`10 mg`, `325-650`, `WBC:14.43`); tokenizing punctuation separately prevents
  one punctuation slip from dominating a short entity's WER, without altering the
  organizer-facing text.
- **Clipping of `1 − WER`.** `clip_text_score` (default `false`). When `WER > 1`,
  `1 − WER` is negative; we never silently clamp. Raw values are always preserved
  in reports; clipping, if enabled, is a separate recorded field.
- **Corpus/document aggregation.** `provisional-v1` (default; micro over scoring
  slots, candidate-weighted), `macro-document`, `micro-entity-diagnostic`. Only
  `provisional-v1` approximates the published score; the others are diagnostic.
  Convention: a component with no eligible slots scores 1.0 (nothing to get
  wrong).
- **Position's role in matching.** Unknown; encoded via matcher choice.
- **Candidate ordering and maximum list length.** Unknown; not enforced.

## Never claimed

This is **not** an "official evaluator clone". Treat scores as a relative,
auditable signal. All provisional switches appear in `summary.json`,
`config_snapshot.json`, and `report.html`.
