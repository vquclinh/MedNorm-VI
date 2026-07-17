# Local RxNorm Snapshot Interface (Phase 1C-A)

A **local-only**, offline interface for a manually acquired RxNorm RRF release.
No network, no real RxNorm content in the repo (tiny synthetic RRF fixtures only),
and **no final medication decode**.

## Not organizer-exact

The organizer describes RxNorm as a "2026 version" but the exact release, the
full-vs-prescribable distribution, active/suppressed/historical handling, legacy
remapping, and TTY priority are all **undisclosed**
(`UNRES-RXNORM-*`). No snapshot is therefore treated as the organizer's KB. We
ingest whatever release we acquired, and use snapshot *forensics* to reason about
differences between candidate releases.

## Model

`RxnormSnapshot` holds atoms (RXNCONSO), relations (RXNREL), attributes (RXNSAT),
and semantic types (RXNSTY), indexed by RXCUI. It exposes preferred string, TTYs,
suppression, relations, and remap targets — all deterministic.

## Comparison / compatibility

`compare-rxnorm-snapshots` reports concepts in both, label/TTY changes,
active↔suppressed transitions, replaced/remapped RXCUIs, and concepts unique to
one snapshot. `resolve_current` follows remap relations to a stable current RXCUI
(a provisional heuristic — whether the organizer expects remapping is
`UNRES-RXNORM-REMAP`, unresolved).

## Provisional decoding hypotheses

`configs/organizer/rxnorm_decoding_hypotheses_v1.yaml` lists candidate strategies
(ingredient-safe, most-specific-supported, SCD-first, generic-first,
legacy-compatible, lower-bound-range, multi-candidate). These are **experiment
hypotheses**; no global winner is selected in Phase 1C-A.

## Commands

    python -m mednorm_vi.phase1c_foundation.cli inspect-rxnorm-snapshot --dir DIR
    python -m mednorm_vi.phase1c_foundation.cli compare-rxnorm-snapshots --left A --right B

## Limitations

- Lookups iterate the snapshot (fine for these sizes; a real release would want
  indexing — a documented follow-up).
- The remap RELA set is a documented approximation; the real retirement mechanism
  is release-specific.
