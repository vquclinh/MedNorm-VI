"""Competition ICD-10 view and advisory hierarchy (Audit 0058 §4).

Milestone 3A gated ICD concepts on *name quality*: 3,470 of 15,308 names looked like
PDF fragments, so 3,668 concepts were marked non-authoritative and the whole snapshot
was held back pending human review of thousands of labels. For a clinical deployment
that is right. For the organizer's metric it is self-defeating, because a concept the
runtime refuses to index can never be retrieved, and a suspect *name* says nothing
about whether the *code* is correct.

This module implements the competition policy instead:

**Everything structurally valid is searchable.** Validity is a property of the code
(``A00``, ``A00.1`` — a letter, two digits, an optional subdivision), not of the
prose next to it. A concept whose name is a fragment still indexes under whatever
text it does have, plus its own code.

**Suspect names are a ranking penalty, not an exclusion.** ``name_quality`` travels
with every record and :func:`name_quality_penalty` turns it into a small negative
ranking term. A fragment-named concept can still win when the mention matches it
exactly; it simply starts behind a clean-named rival. Nothing is repaired — the
original source name is preserved verbatim and no canonical name is invented.

**Hierarchy is advisory.** The 12,968 parent edges are derived from code prefixes,
not from source structure, and 3A recorded exactly that. They are kept, marked
:data:`HIERARCHY_ADVISORY`, and permitted to contribute a weak ranking feature — but
:func:`advisory_hierarchy_may_offer` is hard-wired to ``False``, so an advisory edge
can never introduce a code that lexical evidence did not already reach. Exact lexical
evidence dominates: see :data:`EXACT_MATCH_WEIGHT` against
:data:`ADVISORY_HIERARCHY_WEIGHT`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..indexing.normalization import normalize_text
from .policy import COMPETITION_RUNTIME_ELIGIBLE, EXCLUDED

ICD10_COMPETITION_VIEW_VERSION = "competition-icd10-view-v1"
ICD10_COMPETITION_VIEW_SCHEMA = "competition-icd10-view-schema-v1"

#: ICD-10 code shape: one letter, two digits, optional further subdivision.
_ICD_CODE = re.compile(r"^[A-Z][0-9]{2}[0-9A-Z]*$")

#: Name-quality bands. Ordinal, and only ever used as a ranking penalty.
NAME_CLEAN = "clean"
NAME_SUSPECT = "suspect"
NAME_MISSING = "missing"
NAME_QUALITY_BANDS: tuple[str, ...] = (NAME_CLEAN, NAME_SUSPECT, NAME_MISSING)

#: Hierarchy evidence classes.
HIERARCHY_SOURCE_SUPPORTED = "source_supported"
HIERARCHY_ADVISORY = "advisory_prefix_inference"

#: Ranking weights. Exact lexical evidence must dominate hierarchy expansion, so the
#: gap between these two is deliberately three orders of magnitude, not a tuned knob.
EXACT_MATCH_WEIGHT = 1000.0
ADVISORY_HIERARCHY_WEIGHT = 1.0
#: Penalty applied to a suspect or missing name. Small enough that an exact match on a
#: fragment name still outranks a weak match on a clean one.
SUSPECT_NAME_PENALTY = 25.0
MISSING_NAME_PENALTY = 50.0

ICD_CONCEPT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "code",
    "display_code",
    "lookup_code",
    "source_name",
    "normalized_name",
    "name_quality",
    "quality_flags",
    "structurally_valid",
    "searchable",
    "runtime_role",
    "provenance_id",
    "source_file",
    "source_record_id",
    "exclusion_reason",
)

ICD_HIERARCHY_FIELDS: tuple[str, ...] = (
    "schema_version",
    "ontology",
    "snapshot_id",
    "parent_code",
    "child_code",
    "relation_label",
    "direction",
    "evidence_class",
    "source_authoritative",
    "may_offer_candidate",
    "ranking_weight",
    "provenance_id",
)


def is_structurally_valid_code(code: str) -> bool:
    """Whether a code has ICD-10 shape. A property of the code, not of its label."""
    return bool(_ICD_CODE.match(code.strip().upper()))


def name_quality_of(source_name: str, quality_flags: Sequence[str]) -> str:
    """Band a concept's name without repairing it."""
    if not source_name.strip():
        return NAME_MISSING
    if any(flag.startswith("suspect_") or flag == "missing_source_name" for flag in quality_flags):
        return NAME_SUSPECT
    return NAME_CLEAN


def name_quality_penalty(name_quality: str) -> float:
    """Negative ranking contribution for a suspect or missing name.

    A penalty, never a filter: policy point 5 requires suspect names to stay
    retrievable, because a fragment label does not make its code wrong.
    """
    if name_quality == NAME_SUSPECT:
        return SUSPECT_NAME_PENALTY
    if name_quality == NAME_MISSING:
        return MISSING_NAME_PENALTY
    return 0.0


def advisory_hierarchy_may_offer(evidence_class: str) -> bool:
    """Whether a hierarchy edge may *introduce* a candidate code on its own.

    ``False`` for advisory edges, always. An advisory edge may re-rank a code that
    lexical retrieval already reached; it may not put a new code on the table.
    """
    return evidence_class == HIERARCHY_SOURCE_SUPPORTED


@dataclass(frozen=True, slots=True)
class Icd10CompetitionConcept:
    """One ICD concept as the competition runtime sees it."""

    code: str
    display_code: str
    lookup_code: str
    source_name: str
    normalized_name: str
    name_quality: str
    quality_flags: tuple[str, ...]
    structurally_valid: bool
    searchable: bool
    runtime_role: str
    provenance_id: str
    source_file: str
    source_record_id: str
    exclusion_reason: str = ""

    @property
    def ranking_penalty(self) -> float:
        return name_quality_penalty(self.name_quality)

    def as_row(self, *, snapshot_id: str) -> dict[str, str]:
        return {
            "schema_version": ICD10_COMPETITION_VIEW_SCHEMA,
            "ontology": "icd10_vi",
            "snapshot_id": snapshot_id,
            "code": self.code,
            "display_code": self.display_code,
            "lookup_code": self.lookup_code,
            "source_name": self.source_name,
            "normalized_name": self.normalized_name,
            "name_quality": self.name_quality,
            "quality_flags": "|".join(self.quality_flags),
            "structurally_valid": "true" if self.structurally_valid else "false",
            "searchable": "true" if self.searchable else "false",
            "runtime_role": self.runtime_role,
            "provenance_id": self.provenance_id,
            "source_file": self.source_file,
            "source_record_id": self.source_record_id,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True, slots=True)
class Icd10AdvisoryEdge:
    """One hierarchy edge, explicitly labelled with what evidence supports it."""

    parent_code: str
    child_code: str
    evidence_class: str
    provenance_id: str
    relation_label: str = "parent_of"
    direction: str = "parent_to_child"

    @property
    def source_authoritative(self) -> bool:
        return self.evidence_class == HIERARCHY_SOURCE_SUPPORTED

    @property
    def may_offer_candidate(self) -> bool:
        return advisory_hierarchy_may_offer(self.evidence_class)

    @property
    def ranking_weight(self) -> float:
        return EXACT_MATCH_WEIGHT if self.source_authoritative else ADVISORY_HIERARCHY_WEIGHT

    def as_row(self, *, snapshot_id: str) -> dict[str, str]:
        return {
            "schema_version": ICD10_COMPETITION_VIEW_SCHEMA,
            "ontology": "icd10_vi",
            "snapshot_id": snapshot_id,
            "parent_code": self.parent_code,
            "child_code": self.child_code,
            "relation_label": self.relation_label,
            "direction": self.direction,
            "evidence_class": self.evidence_class,
            "source_authoritative": "true" if self.source_authoritative else "false",
            "may_offer_candidate": "true" if self.may_offer_candidate else "false",
            "ranking_weight": f"{self.ranking_weight:g}",
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True, slots=True)
class Icd10CompetitionView:
    """The built ICD competition view plus its census."""

    concepts: tuple[Icd10CompetitionConcept, ...]
    edges: tuple[Icd10AdvisoryEdge, ...]
    counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def searchable(self) -> tuple[Icd10CompetitionConcept, ...]:
        return tuple(c for c in self.concepts if c.searchable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "view_version": ICD10_COMPETITION_VIEW_VERSION,
            "counts": dict(self.counts),
        }


def build_icd10_competition_view(
    concept_rows: Iterable[Mapping[str, str]],
    edge_rows: Iterable[Mapping[str, str]],
) -> Icd10CompetitionView:
    """Derive the competition ICD view from 3A's frozen concept and hierarchy tables.

    Nothing is recomputed from the PDF and no name is repaired: this is a *policy*
    re-reading of the same frozen rows, which is why it is cheap and why its inputs
    remain the audited 3A artifacts.
    """
    concepts: list[Icd10CompetitionConcept] = []
    for row in concept_rows:
        code = str(row.get("code", ""))
        flags = tuple(f for f in str(row.get("quality_flags", "")).split("|") if f)
        source_name = str(row.get("original_source_name", "")) or str(
            row.get("canonical_display_name", "")
        )
        valid = is_structurally_valid_code(code)
        quality = name_quality_of(source_name, flags)
        concepts.append(
            Icd10CompetitionConcept(
                code=code,
                display_code=str(row.get("display_code", "")),
                lookup_code=str(row.get("lookup_code", "")) or normalize_text(code),
                source_name=source_name,
                normalized_name=normalize_text(source_name, strip_accents=True),
                name_quality=quality,
                quality_flags=flags,
                structurally_valid=valid,
                # Policy point 1: structural validity alone decides searchability.
                searchable=valid,
                runtime_role=COMPETITION_RUNTIME_ELIGIBLE if valid else EXCLUDED,
                provenance_id=str(row.get("provenance_id", "")),
                source_file=str(row.get("source_file", "")),
                source_record_id=str(row.get("source_record_id", "")),
                exclusion_reason="" if valid else "code_does_not_match_icd10_structure",
            )
        )
    concepts.sort(key=lambda c: c.code)

    known = {c.code for c in concepts if c.searchable}
    edges: list[Icd10AdvisoryEdge] = []
    for row in edge_rows:
        parent = str(row.get("parent_code", ""))
        child = str(row.get("child_code", ""))
        if parent not in known or child not in known:
            # An edge to a non-searchable endpoint cannot help ranking and must not
            # be carried as if it described the runtime graph.
            continue
        source_evidence = str(row.get("source_evidence", ""))
        edges.append(
            Icd10AdvisoryEdge(
                parent_code=parent,
                child_code=child,
                evidence_class=(
                    HIERARCHY_SOURCE_SUPPORTED
                    if source_evidence and source_evidence != "legacy_code_prefix_inference"
                    else HIERARCHY_ADVISORY
                ),
                provenance_id=str(row.get("provenance_id", "")),
            )
        )
    edges.sort(key=lambda e: (e.parent_code, e.child_code))

    counts = {
        "concepts_total": len(concepts),
        "structurally_valid": sum(1 for c in concepts if c.structurally_valid),
        "searchable": sum(1 for c in concepts if c.searchable),
        "excluded": sum(1 for c in concepts if not c.searchable),
        "name_clean": sum(1 for c in concepts if c.name_quality == NAME_CLEAN),
        "name_suspect": sum(1 for c in concepts if c.name_quality == NAME_SUSPECT),
        "name_missing": sum(1 for c in concepts if c.name_quality == NAME_MISSING),
        "hierarchy_edges": len(edges),
        "hierarchy_advisory": sum(1 for e in edges if e.evidence_class == HIERARCHY_ADVISORY),
        "hierarchy_source_supported": sum(
            1 for e in edges if e.evidence_class == HIERARCHY_SOURCE_SUPPORTED
        ),
        "hierarchy_may_offer_candidate": sum(1 for e in edges if e.may_offer_candidate),
    }
    return Icd10CompetitionView(tuple(concepts), tuple(edges), counts)


__all__ = [
    "ADVISORY_HIERARCHY_WEIGHT",
    "EXACT_MATCH_WEIGHT",
    "HIERARCHY_ADVISORY",
    "HIERARCHY_SOURCE_SUPPORTED",
    "ICD10_COMPETITION_VIEW_SCHEMA",
    "ICD_CONCEPT_FIELDS",
    "ICD_HIERARCHY_FIELDS",
    "ICD10_COMPETITION_VIEW_VERSION",
    "MISSING_NAME_PENALTY",
    "NAME_CLEAN",
    "NAME_MISSING",
    "NAME_QUALITY_BANDS",
    "NAME_SUSPECT",
    "SUSPECT_NAME_PENALTY",
    "Icd10AdvisoryEdge",
    "Icd10CompetitionConcept",
    "Icd10CompetitionView",
    "advisory_hierarchy_may_offer",
    "build_icd10_competition_view",
    "is_structurally_valid_code",
    "name_quality_of",
    "name_quality_penalty",
]
