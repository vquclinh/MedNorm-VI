# L2 Case Router — Known Limitations (Phase 1B)

- **Line-level routing.** Routing runs at the L1 LINE level; sub-line routing
  (per sentence/cell) is deferred. Multi-label at the line captures most cases.
- **Heuristic weights.** Signal weights are hand-tuned deterministic evidence,
  not learned/calibrated. A future learned segment classifier is planned (spec).
- **C3 is a hook only.** Narrative routing activates a placeholder; Phase 1B does
  **not** produce narrative NER (no fake output).
- **C4 emits evidence, not assertions.** Negation/history/family cues are anchors
  only; final assertion labels are assigned by a later layer.
- **C6/C7 are derived + coarse.** They summarize specialist ambiguity/overlap;
  they are evidence, never ICD/RxNorm codes or a final dedup decision.
- **Abbreviation handling (C5) is detection only.** Abbreviations are flagged,
  not expanded to a single final meaning.
- **Cue inventories are representative.** They are versioned seeds, not the final
  exhaustive clinical lexicon; extend via `signals_v1.yaml` + regression tests.

None of these violate the offset invariant or emit final predictions.
