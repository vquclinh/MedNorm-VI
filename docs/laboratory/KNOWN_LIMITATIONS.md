# Laboratory Parser — Known Limitations (Phase 1B)

- **Aligned columns without tabs** are labelled `table_like` by L1 but not split
  into per-column cells; only tab-split and `key: value` structures yield cells.
- **Narrative pairing is conservative.** It requires a known lexicon test name and
  a numeric value + unit; vague mentions (`glucose tăng`) intentionally produce no
  result (no fabrication).
- **Reference-range vs. value.** A parenthesized range is treated as the reference
  and excluded from the value; unparenthesized ranges as values may be ambiguous.
- **Flags are single-token.** `H`/`L`/`↑`/`↓`/`cao`/`thấp`; composite flag phrases
  are not fully modeled.
- **Blood pressure** is not treated as a lab test unless explicitly configured.
- **Hard-negative list is representative** (dates/ages/times/phones/med doses);
  unusual formats may need new patterns.
- **No linking / no final entities.** No LOINC/ontology codes; proposals and the
  internal `has_result` relation are evidence for later layers only.

None of these violate the offset invariant or emit final entities.
