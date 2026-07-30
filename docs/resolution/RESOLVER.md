# Deterministic L4 Resolver Foundation (Phase 1C-A)

The first Boundary & Type Resolver foundation over Phase 1B proposals. It chooses
a boundary per logical mention, keeps alternatives, assigns a type, resolves
same-type overlaps, and marks each hypothesis accepted / rejected / unresolved.
It performs **no ontology linking** and emits **no final organizer entities**.

## Inputs → outputs

- **In:** `DocumentGraph` (for `original_text`), Phase 1B `SpanProposal`s and
  `RelationProposal`s.
- **Out:** `EntityHypothesis` objects with chosen boundary, retained
  alternatives, boundary/type evidence, overlap decision, rejection reason,
  retained `has_result` pair-group ids, a deterministic score, and status.

Only the proposal types Phase 1B produces are resolved: `THUỐC`,
`TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`. It never invents `CHẨN_ĐOÁN` /
`TRIỆU_CHỨNG` without a proposal source.

## Grouping and boundary policy

Boundary alternatives are grouped by `boundary_group_id` (else the proposal id),
so medication `name_only…full` alternatives group together, a lab result's
value-only/value+unit alternatives group together, and each lab test name is its
own singleton. A **configurable** boundary policy
(`configs/resolution/boundary_type_resolver_v1.yaml`) then picks one:

- **medication:** default `full` — grounded in the confirmed golden example
  (`amlodipine 10 mg po daily` → `[58, 83]`). Also `name_only`/`name_strength`/…
- **test result:** default `value_only` (the descriptive/numeric boundary is
  unresolved, `UNRES-TESTRESULT-BOUNDARY`).

Unchosen alternatives are retained on the hypothesis.

## Guarantees

- Exact offsets: `original_text[start:end] == text` (validated).
- Never deduplicates by text; repeated occurrences at different positions stay
  distinct hypotheses.
- TEST_NAME and TEST_RESULT are independent: an unpaired name or result is
  retained; a `has_result` relation is optional internal evidence, not required.
- Same-type overlaps are resolved deterministically (score, width, position, id);
  cross-type overlaps are both kept. `abstain_on_conflict` marks a tied overlap
  unresolved instead of picking.
- No ontology candidate is emitted (validated).

## Result kinds

Numeric, qualitative, and short descriptive TEST_RESULTs are all handled at their
proposed boundary. **Long descriptive result boundaries** (e.g. multi-sentence
imaging findings) are deliberately **deferred** to later span/boundary models —
Phase 1C-A keeps the conservative `value_only` boundary.

## Limitations

- Scores are deterministic heuristics, not calibrated probabilities.
- No ontology linking, assertions, or organizer output — later phases.
- The overlap winner rule is heuristic; genuinely ambiguous cases can be routed
  to `unresolved` via `abstain_on_conflict`.

## Command

    python -m mednorm_vi.phase1c_foundation.cli resolve-phase1b-debug-output \
        --input DOC.txt [--output out.json]
