# Medication Grammar — Known Limitations (Phase 1B)

- **Single-token names.** The grammar treats an ingredient/brand as one token; a
  brand written as multiple words is not fully joined yet (salt is captured
  separately when present).
- **Unknown-name recall vs. precision.** The unknown-name heuristic (word before a
  strength/dose/route) can over-propose when the router is bypassed; in the
  pipeline, C1/C5 routing gates this. Warnings mark unknown names.
- **Hard-negative list is representative.** Ages, vitals, dates/times, and bare
  units are covered; unusual formats may need new patterns
  (`grammar_v1.yaml::hard_negatives`).
- **No final selection.** Multiple boundary candidates are emitted; none is the
  final entity (L4 decides). Scores are heuristic evidence, not probabilities.
- **No linking.** No RxNorm codes, term types, or ingredient normalization beyond
  a lowercase `normalized_form`.
- **Conjunction handling is basic.** `A and B` → two parses; more complex
  multi-ingredient products (shared strength lists) are approximated.

None of these violate the offset invariant or emit final entities/codes.
