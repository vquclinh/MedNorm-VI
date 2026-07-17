# Local Vietnamese ICD-10 Snapshot Interface (Phase 1C-A)

A **local-only**, offline interface for a manually acquired Vietnamese ICD-10
table. No network, no real ICD content in the repo (synthetic fixtures only), and
**no final diagnosis code**.

## Not organizer-exact; configurable loader

The organizer uses a "Vietnamese version" of ICD-10, but the exact source,
version, and format are **undisclosed** (`UNRES-ICD-SOURCE`). We do **not** assume
WHO vs ICD-10-CM vs a Ministry-of-Health edition unless a manifest declares it.
Because the columns vary by source, the loader is driven by a `ColumnMap` (code /
label_vi / label_en / parent / aliases / chapter) rather than a hard-coded file.

## Model + normalization

`IcdConcept` stores the code exactly as supplied plus reversible `dotted`
(`A09.9`) and `undotted` (`A099`) forms, Vietnamese/English labels, aliases,
parent/children, and chapter. `normalization` converts dotted↔undotted and
computes specificity. Whether the organizer expects dotted or undotted output,
and what specificity is required, are unresolved (`UNRES-ICD-DOTTED`,
`UNRES-ICD-SPECIFICITY`); the interface picks no output policy.

## Comparison

`compare-icd-snapshots` reports missing codes, format differences (dotted vs
undotted supplied), label differences, hierarchy (parent) changes, alias
differences, and specificity differences between two local snapshots.

## Commands

    python -m mednorm_vi.phase1c_foundation.cli inspect-icd-snapshot \
        --table TABLE.csv --columns COLUMNS.yaml --source SRC --version VER
    python -m mednorm_vi.phase1c_foundation.cli compare-icd-snapshots \
        --left A.csv --right B.csv --columns COLUMNS.yaml

## Limitations

- Hierarchy is inferred from code structure when a parent column is absent.
- Lookups iterate the snapshot; indexing is a documented follow-up.
