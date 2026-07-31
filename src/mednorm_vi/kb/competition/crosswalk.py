"""Canonical INN-RxNorm bridge normalization (Audit 0058 §3).

Milestone 3A emitted 46 rows into ``inn_rxnorm_crosswalk_v1.csv`` and Audit 0057
counted them as 46 mappings requiring 46 clinical decisions. They are not 46
mappings. They are 46 **atom-level evidence rows** supporting 12 distinct
local-surface-to-RxCUI bridges, verified against the frozen snapshot:

    46 physical rows -> 12 normalized local surfaces -> 11 distinct RxCUIs
                     -> 12 canonical (normalized_local_surface, rxcui) bridges

The inflation has one cause: the 3A builder emitted one row per *matching atom*, so
``adrenaline`` produced six rows because the RxNorm concept 3992 carries the strings
``EPINEPHRINE``, ``EPINEPHrine``, ``Epinephrine`` and ``epinephrine`` across the term
types ``IN``, ``SU`` and ``TMSY``. Different casing and different atom term types for
the *same* RxCUI are corroborating evidence for one bridge, not six competing claims.

Two consequences follow, and both are load-bearing.

**A term-type collection is not a concept type.** 3A wrote the pipe-joined string
``IN|SU|TMSY`` into a field named ``rxnorm_tty``, which reads as though RxNorm has a
term type of that name. It does not. :class:`BridgeEvidence` parses the field into a
typed ``tuple`` of the term types actually observed, so downstream code compares
against real TTYs (``IN``, ``PIN``, ``MIN``, ``SCD``, …) and never against a
concatenation.

**Two surfaces may legitimately share one RxCUI.** ``co-trimoxazole`` and
``trimethoprim-sulfamethoxazole`` both name RxCUI 10831
(``sulfamethoxazole / trimethoprim``). They stay two bridges, because they are two
different local surfaces the runtime must be able to normalize; collapsing them would
lose one of the two lookup keys. The uniqueness invariant is therefore on the
**pair**, not on either half:

    no duplicate (normalized_local_surface, rxcui)

Every bridge produced here is :data:`~.policy.RETRIEVAL_ONLY`. It may normalize a
Vietnamese/INN surface into the USAN name the snapshot indexes, and it may widen
retrieval. It may **not** hand its ``target_rxcui`` to the organizer as a final
candidate: that code must still be reached through the searchable concept index and
justified by mention evidence (name, strength, dose form, TTY, context). See
:func:`bridge_may_emit_final_candidate`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..indexing.normalization import normalize_text
from .policy import (
    COMPETITION_POLICY_VERSION,
    CONFIDENCE_LEXICAL_EQUIVALENCE,
    CONFIDENCE_NONE,
    EXCLUDED,
    RETRIEVAL_ONLY,
    UNMAPPED,
    assert_automated_decision,
    may_emit_final_candidate,
)

CANONICAL_CROSSWALK_SCHEMA_VERSION = "canonical-crosswalk-bridge-schema-v1"
CANONICAL_CROSSWALK_BUILDER_VERSION = "canonical-crosswalk-builder-v1"

#: Term types that name a *concept* rather than one of its strings. A bridge whose
#: evidence includes one of these is naming an ingredient concept directly.
CONCEPT_DEFINING_TTYS: frozenset[str] = frozenset({"IN", "PIN", "MIN"})
#: Term types denoting a multi-ingredient concept.
COMBINATION_TTYS: frozenset[str] = frozenset({"MIN"})

#: Ambiguity flags. Recorded, never silently resolved.
FLAG_MULTIPLE_RXCUIS = "surface_maps_to_multiple_rxcuis"
FLAG_SHARED_RXCUI = "rxcui_shared_with_another_surface"
FLAG_NO_CONCEPT_TTY = "no_concept_defining_tty_in_evidence"
FLAG_COMBO_SINGLE_MISMATCH = "combination_single_ingredient_mismatch"
FLAG_NO_TARGET = "no_rxcui_in_local_snapshot"

#: Local surfaces whose INN name denotes a combination product. Derived from the
#: surface itself (an explicit multi-ingredient name), not from clinical knowledge.
_COMBINATION_SURFACE_MARKERS: tuple[str, ...] = ("-", "/", "+", " and ", " with ")

BRIDGE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "snapshot_id",
    "local_surface",
    "normalized_local_surface",
    "target_rxcui",
    "preferred_target_name",
    "observed_tty",
    "supporting_atom_count",
    "supporting_provenance_ids",
    "source_evidence",
    "competing_rxcuis",
    "ambiguity_flags",
    "automated_decision",
    "decision_reason",
    "confidence",
    "runtime_role",
)


def _split_tty(raw: str) -> tuple[str, ...]:
    """Parse a term-type field into the collection of term types it observed.

    3A wrote either a single term type (``IN``) or a pipe-joined aggregate
    (``IN|SU|TMSY``). Both are collections; only the second *looks* like one.
    """
    return tuple(sorted({part.strip() for part in raw.split("|") if part.strip()}))


def _is_combination_surface(normalized_surface: str) -> bool:
    return any(marker in normalized_surface for marker in _COMBINATION_SURFACE_MARKERS)


@dataclass(frozen=True, slots=True)
class BridgeEvidence:
    """One atom-level source row supporting a canonical bridge.

    This is the unit 3A mistook for a mapping. It carries the observed term types as
    a typed collection rather than as the concatenated string the CSV held.
    """

    local_surface: str
    normalized_local_surface: str
    target_rxcui: str
    rxnorm_name: str
    target_name: str
    observed_tty: tuple[str, ...]
    provenance_id: str
    evidence_source: str
    source_row_index: int

    @property
    def names_concept(self) -> bool:
        """Whether this atom names the concept itself rather than one of its strings."""
        return bool(CONCEPT_DEFINING_TTYS & set(self.observed_tty))

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_surface": self.local_surface,
            "normalized_local_surface": self.normalized_local_surface,
            "target_rxcui": self.target_rxcui,
            "rxnorm_name": self.rxnorm_name,
            "target_name": self.target_name,
            "observed_tty": list(self.observed_tty),
            "provenance_id": self.provenance_id,
            "evidence_source": self.evidence_source,
            "source_row_index": self.source_row_index,
        }


@dataclass(frozen=True, slots=True)
class CanonicalBridge:
    """Exactly one governed bridge from a normalized local surface to one RxCUI."""

    normalized_local_surface: str
    target_rxcui: str
    local_surface: str
    preferred_target_name: str
    observed_tty: tuple[str, ...]
    supporting_atom_count: int
    supporting_provenance_ids: tuple[str, ...]
    source_evidence: tuple[str, ...]
    competing_rxcuis: tuple[str, ...]
    ambiguity_flags: tuple[str, ...]
    automated_decision: str
    decision_reason: str
    confidence: str
    runtime_role: str
    evidence: tuple[BridgeEvidence, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, str]:
        """The uniqueness key. Two surfaces may share an RxCUI; a pair may not repeat."""
        return (self.normalized_local_surface, self.target_rxcui)

    @property
    def may_emit_final_candidate(self) -> bool:
        return may_emit_final_candidate(self.runtime_role)

    def as_row(self, *, snapshot_id: str) -> dict[str, str]:
        """Flat CSV row. Collections are pipe-joined for storage, never for semantics."""
        return {
            "schema_version": CANONICAL_CROSSWALK_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "local_surface": self.local_surface,
            "normalized_local_surface": self.normalized_local_surface,
            "target_rxcui": self.target_rxcui,
            "preferred_target_name": self.preferred_target_name,
            "observed_tty": "|".join(self.observed_tty),
            "supporting_atom_count": str(self.supporting_atom_count),
            "supporting_provenance_ids": "|".join(self.supporting_provenance_ids),
            "source_evidence": "|".join(self.source_evidence),
            "competing_rxcuis": "|".join(self.competing_rxcuis),
            "ambiguity_flags": "|".join(self.ambiguity_flags),
            "automated_decision": self.automated_decision,
            "decision_reason": self.decision_reason,
            "confidence": self.confidence,
            "runtime_role": self.runtime_role,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized_local_surface": self.normalized_local_surface,
            "target_rxcui": self.target_rxcui,
            "local_surface": self.local_surface,
            "preferred_target_name": self.preferred_target_name,
            "observed_tty": list(self.observed_tty),
            "supporting_atom_count": self.supporting_atom_count,
            "supporting_provenance_ids": list(self.supporting_provenance_ids),
            "source_evidence": list(self.source_evidence),
            "competing_rxcuis": list(self.competing_rxcuis),
            "ambiguity_flags": list(self.ambiguity_flags),
            "automated_decision": self.automated_decision,
            "decision_reason": self.decision_reason,
            "confidence": self.confidence,
            "runtime_role": self.runtime_role,
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class CrosswalkAdjudication:
    """The full normalization result: bridges, counts and a content hash."""

    bridges: tuple[CanonicalBridge, ...]
    source_row_count: int
    unique_normalized_surfaces: int
    unique_rxcuis: int
    decision_counts: Mapping[str, int]
    runtime_role_counts: Mapping[str, int]
    canonical_content_sha256: str

    @property
    def bridge_count(self) -> int:
        return len(self.bridges)

    def bridge_for(self, surface: str) -> tuple[CanonicalBridge, ...]:
        """Every bridge whose normalized local surface matches ``surface``."""
        key = normalize_text(surface, strip_accents=True)
        return tuple(b for b in self.bridges if b.normalized_local_surface == key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "builder_version": CANONICAL_CROSSWALK_BUILDER_VERSION,
            "policy_version": COMPETITION_POLICY_VERSION,
            "schema_version": CANONICAL_CROSSWALK_SCHEMA_VERSION,
            "source_row_count": self.source_row_count,
            "canonical_bridge_count": self.bridge_count,
            "unique_normalized_surfaces": self.unique_normalized_surfaces,
            "unique_rxcuis": self.unique_rxcuis,
            "decision_counts": dict(self.decision_counts),
            "runtime_role_counts": dict(self.runtime_role_counts),
            "canonical_content_sha256": self.canonical_content_sha256,
            "bridges": [bridge.as_dict() for bridge in self.bridges],
        }


def bridge_may_emit_final_candidate(bridge: CanonicalBridge) -> bool:
    """Whether this bridge alone justifies emitting its RxCUI to the organizer.

    Always ``False`` for a :data:`~.policy.RETRIEVAL_ONLY` bridge. The bridge widens
    retrieval; the searchable concept index and the mention's own evidence decide
    candidacy.
    """
    return bridge.may_emit_final_candidate


def read_crosswalk_evidence(rows: Iterable[Mapping[str, str]]) -> tuple[BridgeEvidence, ...]:
    """Parse 3A crosswalk rows into typed atom-level evidence."""
    evidence: list[BridgeEvidence] = []
    for position, row in enumerate(rows):
        local_surface = str(row.get("local_inn_surface", ""))
        normalized = str(row.get("normalized_local_surface", "")) or normalize_text(
            local_surface, strip_accents=True
        )
        evidence.append(
            BridgeEvidence(
                local_surface=local_surface,
                normalized_local_surface=normalized,
                target_rxcui=str(row.get("rxcui", "")),
                rxnorm_name=str(row.get("rxnorm_name", "")),
                target_name=str(row.get("rxnorm_canonical_name", "")),
                observed_tty=_split_tty(str(row.get("rxnorm_tty", ""))),
                provenance_id=str(row.get("provenance_ids", "")),
                evidence_source=str(row.get("evidence_source", "")),
                source_row_index=position,
            )
        )
    return tuple(evidence)


def _preferred_target_name(evidence: Sequence[BridgeEvidence]) -> str:
    """Deterministic canonical name for a bridge target.

    Ranked so the *concept's* own name wins over an incidental synonym string:

    1. atoms whose term types include a concept-defining TTY (``IN``/``PIN``/``MIN``);
    2. among those, the widest observed term-type collection — the concept-level
       aggregate row rather than one single-term atom;
    3. then lexicographic order, so the result is stable across rebuilds.
    """
    naming = [item for item in evidence if item.names_concept] or list(evidence)
    ranked = sorted(naming, key=lambda item: (-len(item.observed_tty), item.target_name))
    return ranked[0].target_name if ranked else ""


def _decide(
    *,
    normalized_surface: str,
    target_rxcui: str,
    observed_tty: Sequence[str],
    flags: Sequence[str],
) -> tuple[str, str, str]:
    """Automated decision, reason and confidence band for one bridge.

    Never returns :data:`~.policy.CLINICALLY_APPROVED` — the caller routes every
    value through :func:`~.policy.assert_automated_decision`.
    """
    if not target_rxcui:
        return (
            UNMAPPED,
            "no RxCUI matched this surface in the local snapshot",
            CONFIDENCE_NONE,
        )
    if FLAG_COMBO_SINGLE_MISMATCH in flags:
        return (
            EXCLUDED,
            "combination surface resolved to a single-ingredient concept, "
            "or a single-ingredient surface resolved to a combination concept",
            CONFIDENCE_NONE,
        )
    if not set(CONCEPT_DEFINING_TTYS) & set(observed_tty):
        return (
            RETRIEVAL_ONLY,
            "surface matched only non-ingredient source strings; usable to normalize "
            "a lookup but not to establish a concept",
            CONFIDENCE_LEXICAL_EQUIVALENCE,
        )
    return (
        RETRIEVAL_ONLY,
        "local naming variant of an RxNorm ingredient concept, established by exact "
        "source-string match and not clinically adjudicated",
        CONFIDENCE_LEXICAL_EQUIVALENCE,
    )


def _canonical_content_hash(bridges: Sequence[CanonicalBridge]) -> str:
    """Content hash over the canonical bridge set, independent of file layout."""
    digest = hashlib.sha256()
    for bridge in bridges:
        payload = {
            "normalized_local_surface": bridge.normalized_local_surface,
            "target_rxcui": bridge.target_rxcui,
            "preferred_target_name": bridge.preferred_target_name,
            "observed_tty": list(bridge.observed_tty),
            "supporting_atom_count": bridge.supporting_atom_count,
            "automated_decision": bridge.automated_decision,
            "runtime_role": bridge.runtime_role,
            "ambiguity_flags": list(bridge.ambiguity_flags),
            "competing_rxcuis": list(bridge.competing_rxcuis),
        }
        digest.update(
            (
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def normalize_crosswalk(rows: Iterable[Mapping[str, str]]) -> CrosswalkAdjudication:
    """Collapse atom-level crosswalk rows into canonical bridges.

    Grouping is by ``(normalized_local_surface, rxcui)``. Casing, source atom and
    term-type differences inside a group are corroboration, so they raise the
    supporting-atom count and widen the observed term-type collection without ever
    producing a second bridge.
    """
    evidence = read_crosswalk_evidence(rows)
    grouped: dict[tuple[str, str], list[BridgeEvidence]] = {}
    for item in evidence:
        grouped.setdefault((item.normalized_local_surface, item.target_rxcui), []).append(item)

    rxcuis_by_surface: dict[str, set[str]] = {}
    surfaces_by_rxcui: dict[str, set[str]] = {}
    for surface, rxcui in grouped:
        rxcuis_by_surface.setdefault(surface, set()).add(rxcui)
        if rxcui:
            surfaces_by_rxcui.setdefault(rxcui, set()).add(surface)

    bridges: list[CanonicalBridge] = []
    for (surface, rxcui), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item.source_row_index)
        observed_tty = tuple(sorted({tty for item in ordered for tty in item.observed_tty}))

        flags: list[str] = []
        competing = tuple(sorted(rxcuis_by_surface[surface] - {rxcui}))
        if competing:
            # Preserved, never silently resolved: picking one target without source
            # evidence is exactly the failure this governance layer exists to stop.
            flags.append(FLAG_MULTIPLE_RXCUIS)
        if rxcui and len(surfaces_by_rxcui.get(rxcui, set())) > 1:
            # Legitimate: two local surfaces naming the same concept stay two bridges.
            flags.append(FLAG_SHARED_RXCUI)
        if not rxcui:
            flags.append(FLAG_NO_TARGET)
        elif not CONCEPT_DEFINING_TTYS & set(observed_tty):
            flags.append(FLAG_NO_CONCEPT_TTY)
        target_is_combination = bool(COMBINATION_TTYS & set(observed_tty))
        if rxcui and target_is_combination != _is_combination_surface(surface):
            flags.append(FLAG_COMBO_SINGLE_MISMATCH)

        decision, reason, confidence = _decide(
            normalized_surface=surface,
            target_rxcui=rxcui,
            observed_tty=observed_tty,
            flags=flags,
        )
        bridges.append(
            CanonicalBridge(
                normalized_local_surface=surface,
                target_rxcui=rxcui,
                local_surface=ordered[0].local_surface,
                preferred_target_name=_preferred_target_name(ordered),
                observed_tty=observed_tty,
                supporting_atom_count=len(ordered),
                supporting_provenance_ids=tuple(
                    sorted({item.provenance_id for item in ordered if item.provenance_id})
                ),
                source_evidence=tuple(
                    sorted({item.evidence_source for item in ordered if item.evidence_source})
                ),
                competing_rxcuis=competing,
                ambiguity_flags=tuple(sorted(set(flags))),
                automated_decision=assert_automated_decision(decision),
                decision_reason=reason,
                confidence=confidence,
                # A retrieval aid is never a candidate source; the role mirrors the
                # decision exactly so the two can never drift apart.
                runtime_role=decision,
                evidence=tuple(ordered),
            )
        )

    keys = [bridge.key for bridge in bridges]
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"duplicate canonical crosswalk bridge keys: {duplicates}")

    decision_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for bridge in bridges:
        decision_counts[bridge.automated_decision] = (
            decision_counts.get(bridge.automated_decision, 0) + 1
        )
        role_counts[bridge.runtime_role] = role_counts.get(bridge.runtime_role, 0) + 1

    return CrosswalkAdjudication(
        bridges=tuple(bridges),
        source_row_count=len(evidence),
        unique_normalized_surfaces=len(rxcuis_by_surface),
        unique_rxcuis=len({rxcui for _surface, rxcui in grouped if rxcui}),
        decision_counts=decision_counts,
        runtime_role_counts=role_counts,
        canonical_content_sha256=_canonical_content_hash(bridges),
    )


def retrieval_expansion_names(adjudication: CrosswalkAdjudication, surface: str) -> tuple[str, ...]:
    """RxNorm-side names a surface may additionally be *looked up* by.

    This is the one thing a ``RETRIEVAL_ONLY`` bridge is allowed to do. It returns
    names, never RxCUIs, precisely so a caller cannot mistake the result for a
    candidate code.
    """
    names: list[str] = []
    for bridge in adjudication.bridge_for(surface):
        if bridge.automated_decision != RETRIEVAL_ONLY:
            continue
        if bridge.preferred_target_name:
            names.append(bridge.preferred_target_name)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        key = normalize_text(name, strip_accents=True)
        if key and key not in seen:
            seen.add(key)
            ordered.append(name)
    return tuple(ordered)


__all__ = [
    "BRIDGE_FIELDS",
    "CANONICAL_CROSSWALK_BUILDER_VERSION",
    "CANONICAL_CROSSWALK_SCHEMA_VERSION",
    "COMBINATION_TTYS",
    "CONCEPT_DEFINING_TTYS",
    "FLAG_COMBO_SINGLE_MISMATCH",
    "FLAG_MULTIPLE_RXCUIS",
    "FLAG_NO_CONCEPT_TTY",
    "FLAG_NO_TARGET",
    "FLAG_SHARED_RXCUI",
    "BridgeEvidence",
    "CanonicalBridge",
    "CrosswalkAdjudication",
    "bridge_may_emit_final_candidate",
    "normalize_crosswalk",
    "read_crosswalk_evidence",
    "retrieval_expansion_names",
]
