"""Competition RxNorm searchable view and endpoint closure (Audit 0058 §5, §6).

Audit 0057 reported "2,010,514 relation rows with missing endpoints" in the
prescribable subset and listed that as a blocker. Milestone 3B re-measured it against
the same local RRF files and found the count is an artifact of the check, not a defect
in the data.

``RXNREL.RRF`` carries relations at two levels. A ``CUI``-level row populates
``RXCUI1``/``RXCUI2``; an ``AUI``-level row leaves those columns empty **by design**
and identifies its endpoints in ``RXAUI1``/``RXAUI2`` instead. The 3A builder read
only the RXCUI columns, so every atom-level row looked like a relation with two
missing endpoints. Resolving ``RXAUI -> RXCUI`` through ``RXNCONSO.RRF`` shows the
true picture, verified on the local snapshot:

    2,563,978 prescribable relation rows
      553,464 CUI-level, 0 with a missing endpoint
    2,010,514 AUI-level, all 2,010,514 resolvable, 0 unresolvable atoms

So the prescribable relation set is already endpoint-complete and needs **no** closure
from Full RxNorm. What Full does supply is *reach*: relations that RxNorm records
outside the prescribable extract. This module retains those and adds only the concepts
they require.

The retained-relation policy is deliberately narrow. Only
:data:`COMPETITION_STRUCTURE_RELAS` — the ingredient/component/dose-form/tradename
relations that spec §10.2's ``IN -> SCDC -> SCD -> SBD`` walk actually traverses — are
kept. ``has_inactive_ingredient`` alone is 1.5M rows and cannot identify a product;
carrying it would inflate the snapshot and the closure for no candidate benefit.

Two flags stay separate throughout, because collapsing them is how a closure node
leaks into a submission:

    searchable          may be retrieved and may become a final candidate
    graph_endpoint      may be traversed

A prescribable concept is both. A closure-only concept is **only** the second:
:func:`concept_may_emit_final_candidate` returns ``False`` for it regardless of how
the traversal reached it.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..indexing.normalization import normalize_text
from .policy import COMPETITION_RUNTIME_ELIGIBLE, RETRIEVAL_ONLY

RXNORM_COMPETITION_VIEW_VERSION = "competition-rxnorm-view-v1"
RXNORM_COMPETITION_VIEW_SCHEMA = "competition-rxnorm-view-schema-v1"

#: Relations the candidate-generation walk traverses. Everything else is excluded from
#: the retained set with a recorded count, never silently dropped.
COMPETITION_STRUCTURE_RELAS: frozenset[str] = frozenset(
    {
        "has_ingredient",
        "ingredient_of",
        "has_precise_ingredient",
        "precise_ingredient_of",
        "consists_of",
        "constitutes",
        "has_dose_form",
        "dose_form_of",
        "has_form",
        "form_of",
        "has_tradename",
        "tradename_of",
        "has_quantified_form",
        "quantified_form_of",
    }
)

#: Membership classes for a concept in the competition view.
CONCEPT_SEARCHABLE = "searchable"
CONCEPT_CLOSURE_ONLY = "closure_only"

_RXN_CONSO = {
    "RXCUI": 0,
    "LAT": 1,
    "ISPREF": 6,
    "RXAUI": 7,
    "SAB": 11,
    "TTY": 12,
    "CODE": 13,
    "STR": 14,
    "SUPPRESS": 16,
}
_RXN_REL = {
    "RXCUI1": 0,
    "RXAUI1": 1,
    "STYPE1": 2,
    "REL": 3,
    "RXCUI2": 4,
    "RXAUI2": 5,
    "STYPE2": 6,
    "RELA": 7,
    "RUI": 8,
    "SAB": 10,
    "DIR": 13,
    "SUPPRESS": 14,
}

CONCEPT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "rxcui",
    "preferred_name",
    "normalized_name",
    "tty_values",
    "source_vocabularies",
    "suppress_values",
    "atom_count",
    "membership",
    "searchable",
    "graph_endpoint",
    "runtime_role",
    "source_file",
    "provenance_id",
)

RELATION_FIELDS: tuple[str, ...] = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "source_rxcui",
    "target_rxcui",
    "rel",
    "rela",
    "direction",
    "source_vocabulary",
    "rui",
    "source_level",
    "origin_snapshot",
    "source_endpoint_class",
    "target_endpoint_class",
    "endpoint_status",
)


def concept_may_emit_final_candidate(membership: str) -> bool:
    """Whether a concept with this membership may be emitted to the organizer.

    ``False`` for :data:`CONCEPT_CLOSURE_ONLY`. Traversal reaching a closure node is
    graph evidence; it is not candidacy.
    """
    return membership == CONCEPT_SEARCHABLE


@dataclass(slots=True)
class _ConceptAggregate:
    rxcui: str
    preferred_name: str = ""
    fallback_name: str = ""
    tty_values: set[str] = field(default_factory=set)
    source_vocabularies: set[str] = field(default_factory=set)
    suppress_values: set[str] = field(default_factory=set)
    atom_count: int = 0


@dataclass(frozen=True, slots=True)
class RxNormClosureReport:
    """Counts for §6's required before/after closure report."""

    searchable_concepts: int
    closure_only_concepts: int
    concepts_after_closure: int
    prescribable_relation_rows: int
    prescribable_cui_level_rows: int
    prescribable_aui_level_rows: int
    prescribable_aui_unresolvable: int
    prescribable_missing_endpoint_cui_check: int
    prescribable_missing_endpoint_after_resolution: int
    full_structure_rows_scanned: int
    retained_relations: int
    retained_from_prescribable: int
    retained_added_from_full: int
    excluded_non_structure_rows: int
    excluded_no_searchable_endpoint: int
    duplicate_relation_rows_collapsed: int
    missing_endpoints_after_closure: int
    tty_distribution: Mapping[str, int]
    relation_label_distribution: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "view_version": RXNORM_COMPETITION_VIEW_VERSION,
            "searchable_concepts": self.searchable_concepts,
            "closure_only_concepts": self.closure_only_concepts,
            "concepts_after_closure": self.concepts_after_closure,
            "prescribable_relation_rows": self.prescribable_relation_rows,
            "prescribable_cui_level_rows": self.prescribable_cui_level_rows,
            "prescribable_aui_level_rows": self.prescribable_aui_level_rows,
            "prescribable_aui_unresolvable": self.prescribable_aui_unresolvable,
            "prescribable_missing_endpoint_cui_check": (
                self.prescribable_missing_endpoint_cui_check
            ),
            "prescribable_missing_endpoint_after_resolution": (
                self.prescribable_missing_endpoint_after_resolution
            ),
            "full_structure_rows_scanned": self.full_structure_rows_scanned,
            "retained_relations": self.retained_relations,
            "retained_from_prescribable": self.retained_from_prescribable,
            "retained_added_from_full": self.retained_added_from_full,
            "excluded_non_structure_rows": self.excluded_non_structure_rows,
            "excluded_no_searchable_endpoint": self.excluded_no_searchable_endpoint,
            "duplicate_relation_rows_collapsed": self.duplicate_relation_rows_collapsed,
            "missing_endpoints_after_closure": self.missing_endpoints_after_closure,
            "tty_distribution": dict(self.tty_distribution),
            "relation_label_distribution": dict(self.relation_label_distribution),
        }


def _rrf_rows(path: Path) -> Iterator[list[str]]:
    """Stream one RRF file. Line by line; nothing is accumulated here."""
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            cols = raw.rstrip("\n").split("|")
            if cols and cols[-1] == "":
                cols = cols[:-1]
            yield cols


def _get(cols: list[str], index: int) -> str:
    return cols[index] if index < len(cols) else ""


def _numeric_key(value: str) -> tuple[int, int, str]:
    return (0, int(value), "") if value.isdigit() else (1, 0, value)


def _ingest_concepts(
    conso: Path, *, wanted: frozenset[str] | None = None
) -> tuple[dict[str, _ConceptAggregate], dict[str, str]]:
    """Aggregate concepts and the atom-to-concept map from one RRF snapshot.

    ``wanted`` restricts which concepts are aggregated. The atom-to-concept map is
    always complete, because relation resolution needs every atom; but Full RxNorm has
    1.2M concepts and a closure needs a small fraction of them, so aggregating all of
    them would cost hundreds of megabytes to then discard. Passing ``wanted`` after a
    first map-only pass keeps the peak bounded.
    """
    concepts: dict[str, _ConceptAggregate] = {}
    aui_to_cui: dict[str, str] = {}
    for cols in _rrf_rows(conso):
        rxcui = _get(cols, _RXN_CONSO["RXCUI"])
        rxaui = _get(cols, _RXN_CONSO["RXAUI"])
        surface = _get(cols, _RXN_CONSO["STR"])
        if not rxcui or not rxaui or not surface:
            continue
        aui_to_cui[rxaui] = rxcui
        if wanted is not None and rxcui not in wanted:
            continue
        agg = concepts.get(rxcui)
        if agg is None:
            agg = _ConceptAggregate(rxcui=rxcui)
            concepts[rxcui] = agg
        agg.atom_count += 1
        agg.tty_values.add(_get(cols, _RXN_CONSO["TTY"]))
        agg.source_vocabularies.add(_get(cols, _RXN_CONSO["SAB"]))
        agg.suppress_values.add(_get(cols, _RXN_CONSO["SUPPRESS"]) or "N")
        if _get(cols, _RXN_CONSO["ISPREF"]) == "Y" and not agg.preferred_name:
            agg.preferred_name = surface
        if not agg.fallback_name or surface < agg.fallback_name:
            agg.fallback_name = surface
    return concepts, aui_to_cui


def _resolve_endpoints(cols: list[str], aui_to_cui: Mapping[str, str]) -> tuple[str, str, str]:
    """Concept endpoints for a relation row, plus the level it was recorded at.

    This is the correction at the heart of §6: an ``AUI``-level row is not an
    endpoint-less relation, it is a relation recorded between atoms.
    """
    if _get(cols, _RXN_REL["STYPE1"]) == "CUI":
        return _get(cols, _RXN_REL["RXCUI1"]), _get(cols, _RXN_REL["RXCUI2"]), "CUI"
    source = aui_to_cui.get(_get(cols, _RXN_REL["RXAUI1"]), "")
    target = aui_to_cui.get(_get(cols, _RXN_REL["RXAUI2"]), "")
    return source, target, "AUI"


def build_rxnorm_competition_view(
    *,
    prescribable_rrf: str | Path,
    full_rrf: str | Path,
    output_dir: str | Path,
    snapshot_id: str,
) -> RxNormClosureReport:
    """Build the searchable view, retained relations and minimum endpoint closure.

    Memory is bounded by design: concept aggregates and the atom-to-concept map are
    the only large structures, relation files are streamed, and the retained-relation
    map holds one compact entry per surviving relation rather than per source row.
    """
    pre_root = Path(prescribable_rrf)
    full_root = Path(full_rrf)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- searchable concepts, from the prescribable snapshot --------------------
    searchable, pre_aui = _ingest_concepts(pre_root / "RXNCONSO.RRF")
    searchable_ids = frozenset(searchable)

    tty_distribution: dict[str, int] = {}
    for agg in searchable.values():
        for tty in agg.tty_values:
            tty_distribution[tty] = tty_distribution.get(tty, 0) + 1

    # --- retained relations from the prescribable snapshot ----------------------
    # value: rel|rela|sab|rui|level|origin, keyed by "source|target|rel|rela"
    retained: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    pre_rows = pre_cui = pre_aui_rows = pre_unresolvable = 0
    pre_missing_cui_check = pre_missing_after = 0
    excluded_non_structure = duplicates = 0

    for cols in _rrf_rows(pre_root / "RXNREL.RRF"):
        if len(cols) < 9:
            continue
        pre_rows += 1
        level = _get(cols, _RXN_REL["STYPE1"])
        if level == "CUI":
            pre_cui += 1
            if (
                _get(cols, _RXN_REL["RXCUI1"]) not in searchable_ids
                or _get(cols, _RXN_REL["RXCUI2"]) not in searchable_ids
            ):
                pre_missing_cui_check += 1
        else:
            pre_aui_rows += 1
            # What the 3A check counted as a missing endpoint.
            pre_missing_cui_check += 1
        source, target, resolved_level = _resolve_endpoints(cols, pre_aui)
        if not source or not target:
            pre_unresolvable += 1
            pre_missing_after += 1
            continue
        if source not in searchable_ids or target not in searchable_ids:
            pre_missing_after += 1
        rela = _get(cols, _RXN_REL["RELA"])
        if rela not in COMPETITION_STRUCTURE_RELAS:
            excluded_non_structure += 1
            continue
        rel = _get(cols, _RXN_REL["REL"])
        key = f"{source}|{target}|{rel}|{rela}"
        if key in retained:
            duplicates += 1
            continue
        retained[key] = (
            f"{_get(cols, _RXN_REL['SAB'])}|{_get(cols, _RXN_REL['RUI'])}|"
            f"{resolved_level}|prescribable"
        )
        label_counts[rela] = label_counts.get(rela, 0) + 1
    retained_from_prescribable = len(retained)
    del pre_aui

    # --- extend reach using Full, and close only the endpoints that requires ----
    # Two passes over Full's concept file on purpose: the first needs only the
    # atom-to-concept map, and the second aggregates the closure set once it is known.
    _unused, full_aui = _ingest_concepts(full_root / "RXNCONSO.RRF", wanted=frozenset())
    del _unused
    closure_ids: set[str] = set()
    full_scanned = excluded_no_searchable = 0

    for cols in _rrf_rows(full_root / "RXNREL.RRF"):
        if len(cols) < 9:
            continue
        rela = _get(cols, _RXN_REL["RELA"])
        if rela not in COMPETITION_STRUCTURE_RELAS:
            continue
        full_scanned += 1
        source, target, resolved_level = _resolve_endpoints(cols, full_aui)
        if not source or not target:
            continue
        source_searchable = source in searchable_ids
        target_searchable = target in searchable_ids
        if not source_searchable and not target_searchable:
            # Neither end can ever be offered, so this relation cannot contribute to
            # a candidate. Excluding it is what keeps the closure minimal.
            excluded_no_searchable += 1
            continue
        rel = _get(cols, _RXN_REL["REL"])
        key = f"{source}|{target}|{rel}|{rela}"
        if key in retained:
            duplicates += 1
            continue
        retained[key] = (
            f"{_get(cols, _RXN_REL['SAB'])}|{_get(cols, _RXN_REL['RUI'])}|{resolved_level}|full"
        )
        label_counts[rela] = label_counts.get(rela, 0) + 1
        # The minimum closure: exactly the endpoints a retained relation requires.
        if not source_searchable:
            closure_ids.add(source)
        if not target_searchable:
            closure_ids.add(target)
    del full_aui
    full_concepts, _map = _ingest_concepts(
        full_root / "RXNCONSO.RRF", wanted=frozenset(closure_ids)
    )
    del _map

    # --- write concepts ---------------------------------------------------------
    concept_path = out / "rxnorm_competition_concepts_v1.csv"
    written_concepts = 0
    with concept_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CONCEPT_FIELDS)
        writer.writeheader()
        for rxcui in sorted(searchable_ids | closure_ids, key=_numeric_key):
            if rxcui in searchable_ids:
                agg = searchable[rxcui]
                membership, source_file = CONCEPT_SEARCHABLE, "prescribable/RXNCONSO.RRF"
            else:
                agg = full_concepts[rxcui]
                membership, source_file = CONCEPT_CLOSURE_ONLY, "full/RXNCONSO.RRF"
            name = agg.preferred_name or agg.fallback_name
            writer.writerow(
                {
                    "schema_version": RXNORM_COMPETITION_VIEW_SCHEMA,
                    "ontology": "rxnorm",
                    "snapshot_id": snapshot_id,
                    "rxcui": rxcui,
                    "preferred_name": name,
                    "normalized_name": normalize_text(name, strip_accents=True),
                    "tty_values": "|".join(sorted(t for t in agg.tty_values if t)),
                    "source_vocabularies": "|".join(
                        sorted(s for s in agg.source_vocabularies if s)
                    ),
                    "suppress_values": "|".join(sorted(s for s in agg.suppress_values if s)),
                    "atom_count": str(agg.atom_count),
                    "membership": membership,
                    "searchable": "true" if membership == CONCEPT_SEARCHABLE else "false",
                    # Both classes are traversable; only one is retrievable.
                    "graph_endpoint": "true",
                    "runtime_role": (
                        COMPETITION_RUNTIME_ELIGIBLE
                        if membership == CONCEPT_SEARCHABLE
                        else RETRIEVAL_ONLY
                    ),
                    "source_file": source_file,
                    "provenance_id": f"rxnorm:{membership}:{rxcui}",
                }
            )
            written_concepts += 1
    del full_concepts

    # --- write relations --------------------------------------------------------
    all_ids = searchable_ids | closure_ids
    relation_path = out / "rxnorm_competition_relations_v1.csv"
    missing_after = 0
    with relation_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RELATION_FIELDS)
        writer.writeheader()
        for key in sorted(
            retained,
            key=lambda k: (
                _numeric_key(k.split("|")[0]),
                _numeric_key(k.split("|")[1]),
                k.split("|")[2],
                k.split("|")[3],
            ),
        ):
            source, target, rel, rela = key.split("|")
            sab, rui, level, origin = retained[key].split("|")
            source_ok = source in all_ids
            target_ok = target in all_ids
            if not source_ok or not target_ok:
                missing_after += 1
            writer.writerow(
                {
                    "schema_version": RXNORM_COMPETITION_VIEW_SCHEMA,
                    "ontology": "rxnorm",
                    "snapshot_id": snapshot_id,
                    "source_rxcui": source,
                    "target_rxcui": target,
                    "rel": rel,
                    "rela": rela,
                    "direction": "source_to_target",
                    "source_vocabulary": sab,
                    "rui": rui,
                    "source_level": level,
                    "origin_snapshot": origin,
                    "source_endpoint_class": (
                        CONCEPT_SEARCHABLE if source in searchable_ids else CONCEPT_CLOSURE_ONLY
                    ),
                    "target_endpoint_class": (
                        CONCEPT_SEARCHABLE if target in searchable_ids else CONCEPT_CLOSURE_ONLY
                    ),
                    "endpoint_status": "ok" if source_ok and target_ok else "missing",
                }
            )

    return RxNormClosureReport(
        searchable_concepts=len(searchable_ids),
        closure_only_concepts=len(closure_ids),
        concepts_after_closure=written_concepts,
        prescribable_relation_rows=pre_rows,
        prescribable_cui_level_rows=pre_cui,
        prescribable_aui_level_rows=pre_aui_rows,
        prescribable_aui_unresolvable=pre_unresolvable,
        prescribable_missing_endpoint_cui_check=pre_missing_cui_check,
        prescribable_missing_endpoint_after_resolution=pre_missing_after,
        full_structure_rows_scanned=full_scanned,
        retained_relations=len(retained),
        retained_from_prescribable=retained_from_prescribable,
        retained_added_from_full=len(retained) - retained_from_prescribable,
        excluded_non_structure_rows=excluded_non_structure,
        excluded_no_searchable_endpoint=excluded_no_searchable,
        duplicate_relation_rows_collapsed=duplicates,
        missing_endpoints_after_closure=missing_after,
        tty_distribution=dict(sorted(tty_distribution.items())),
        relation_label_distribution=dict(sorted(label_counts.items())),
    )


__all__ = [
    "COMPETITION_STRUCTURE_RELAS",
    "CONCEPT_CLOSURE_ONLY",
    "CONCEPT_FIELDS",
    "CONCEPT_SEARCHABLE",
    "RELATION_FIELDS",
    "RXNORM_COMPETITION_VIEW_SCHEMA",
    "RXNORM_COMPETITION_VIEW_VERSION",
    "RxNormClosureReport",
    "build_rxnorm_competition_view",
    "concept_may_emit_final_candidate",
]
