"""RxNorm Super Linker: structured retrieval + graph reasoning (spec §10.1, §10.2).

Before Audit 0054 this module did one thing: ``search_index(index, hypothesis.text)``.
Retrieving on the whole surface — `amlodipine besylate 10 mg po daily` — is why the
old linker produced n-gram noise: no RxNorm concept is named that way, so the exact
and normalized channels never fired and the result was whatever shared trigrams. E1's
structured decomposition had been preserved through the lattice and L4 since Audit
0052 precisely so this could stop, and Audit 0053 recorded that it had not.

What this module now does, in order:

1. build spec §10.1's structured mention from E1's ComponentSpans;
2. retrieve on the **ingredient** (and brand, and salt-qualified name), not only the
   whole surface;
3. walk the locked graph IN -> SCDC -> SCD -> SBD by TTY role (see ``rxnorm_graph``
   for why the walk is TTY-based and not relation-based);
4. test every reached concept against the mention's *stated* structured evidence;
5. suppress hard negatives — conflicting strength, unit, concentration, dose form,
   release, route;
6. fall back to the ingredient alone when only the name is justified;
7. order deterministically and record, per candidate, exactly why it survived.

Two rules hold throughout. **Nothing is invented**: every emitted code is a concept
retrieval or traversal actually reached in the locked snapshot, and membership is
re-checked before emission. **Missing evidence never counts against a candidate**: a
mention stating no dose form cannot conflict on dose form, and the decision records
``NOT_STATED`` rather than a silent pass.

Candidate *quality* is not measured here or anywhere. The governed corpus carries no
RxCUIs, so Recall@K and candidate Jaccard are UNMEASURABLE_WITHOUT_CODE_BEARING_GOLD
(see ``evaluation.code_linking`` for the contract a future human-checked set must
satisfy). What is measured is behaviour: compatibility counts, suppression counts and
determinism.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..kb.indexing.retrieval import CandidateHit, LocalIndex, search_index
from ..resolution.models import EntityHypothesis
from .models import LinkedCandidate, LinkerResult
from .rxnorm_graph import (
    RXNORM_GRAPH_VERSION,
    TTY_BRAND_NAME,
    TTY_BRANDED_DRUG,
    TTY_CLINICAL_DRUG,
    TTY_COMPONENT,
    TTY_GROUPER,
    TTY_INGREDIENT,
    TTY_PRIORITY,
    TTY_TERMINAL,
    GraphPath,
    aliases_of,
    attached_brand_names,
    attached_ingredients,
    is_closure_only,
    is_suppressed,
    name_of,
    traverse_ingredient_chain,
    tty_of,
)
from .structured_medication import (
    INN_BRIDGE_STATUS,
    StructuredMedicationMention,
    bridged_ingredient_names,
    build_structured_mention,
    canonical_unit,
    normalize_surface,
    strength_in_mg,
    unit_family,
)

RXNORM_LINKER_VERSION = "rxnorm-structured-linker-v1"

# Per-field compatibility verdicts. `NOT_STATED` is a first-class outcome, distinct
# from `MATCH`: it means the mention is silent, so the field may not influence the
# decision in either direction.
MATCH = "MATCH"
CONFLICT = "CONFLICT"
NOT_STATED = "NOT_STATED"
NOT_COMPARABLE = "NOT_COMPARABLE"

# Reasons a candidate is retained.
KEEP_EXACT_INGREDIENT = "KEEP_EXACT_INGREDIENT"
KEEP_STRUCTURED_MATCH = "KEEP_STRUCTURED_MATCH"
KEEP_INGREDIENT_FALLBACK = "KEEP_INGREDIENT_FALLBACK"
KEEP_LEXICAL_ONLY = "KEEP_LEXICAL_ONLY"

# Reasons a candidate is suppressed. Each is a hard negative in spec §10.2's sense.
DROP_STRENGTH_CONFLICT = "DROP_STRENGTH_CONFLICT"
DROP_UNIT_CONFLICT = "DROP_UNIT_CONFLICT"
DROP_CONCENTRATION_CONFLICT = "DROP_CONCENTRATION_CONFLICT"
DROP_DOSE_FORM_CONFLICT = "DROP_DOSE_FORM_CONFLICT"
DROP_RELEASE_CONFLICT = "DROP_RELEASE_CONFLICT"
DROP_ROUTE_CONFLICT = "DROP_ROUTE_CONFLICT"
DROP_SUPPRESSED_CONCEPT = "DROP_SUPPRESSED_CONCEPT"
DROP_NOT_IN_SNAPSHOT = "DROP_NOT_IN_SNAPSHOT"
# Audit 0058: a v3 closure-only endpoint. Traversal may stand on it; nothing may
# emit it. Distinct from DROP_NOT_IN_SNAPSHOT because the concept *is* in the
# snapshot — it simply is not a searchable candidate.
DROP_CLOSURE_ONLY = "DROP_CLOSURE_ONLY"
DROP_NON_TERMINAL_TTY = "DROP_NON_TERMINAL_TTY"
DROP_INGREDIENT_MISMATCH = "DROP_INGREDIENT_MISMATCH"
DROP_BUDGET = "DROP_BUDGET"

# Confidence tiers, ordered. These are evidence tiers, not calibrated probabilities —
# no stage below L4 produces a calibrated probability (Audit 0053 §19).
TIER_STRUCTURED_EXACT = "structured_exact"
TIER_STRUCTURED_PARTIAL = "structured_partial"
TIER_INGREDIENT_ONLY = "ingredient_only"
TIER_LEXICAL = "lexical"
TIER_ORDER: tuple[str, ...] = (
    TIER_STRUCTURED_EXACT,
    TIER_STRUCTURED_PARTIAL,
    TIER_INGREDIENT_ONLY,
    TIER_LEXICAL,
)

# Safety bound on emitted candidates. A bound, not the selection rule (Audit 0053 §9).
CANDIDATE_BOUND = 20
# Retrieval bound per query term.
RETRIEVAL_LIMIT = 24

# Strength inside an SCDC/SCD name, e.g. "aspirin 162.5 MG", "... 500 mg / ...".
_NAME_STRENGTH = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|ug|µg|ml|l|unit|units|iu|ui|meq|mmol|%)\b", re.IGNORECASE
)
# Concentration inside a name, e.g. "500 MG in 5 mL", "250 mg/5 mL".
_NAME_CONCENTRATION = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|unit|units)\s*(?:/|in)\s*(\d+(?:[.,]\d+)?)?\s*(ml|l)\b",
    re.IGNORECASE,
)

# Dose-form vocabulary as the snapshot writes it, mapped to the surface forms E1 emits
# in Vietnamese clinical text. Only conservative, unambiguous pairs.
_DOSE_FORM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "tablet": ("tablet", "tab", "vien nen", "vien"),
    "capsule": ("capsule", "cap", "vien nang", "nang"),
    "suspension": ("suspension", "hon dich"),
    "solution": ("solution", "dung dich"),
    "syrup": ("syrup", "siro"),
    "injection": ("injection", "injectable", "thuoc tiem", "tiem"),
    "cream": ("cream", "kem"),
    "ointment": ("ointment", "thuoc mo"),
    "suppository": ("suppository", "vien dat"),
    "patch": ("patch", "cao dan", "mieng dan"),
    "drops": ("drops", "drop", "nho giot"),
    "powder": ("powder", "bot"),
    "inhalant": ("inhalant", "inhalation"),
}

# Release-form vocabulary. RxNorm writes these into the SCD name.
_RELEASE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "extended_release": (
        "extended release",
        "phong thich keo dai",
        "24 hr",
        "12 hr",
        "er",
        "xr",
        "xl",
        "sr",
    ),
    "delayed_release": ("delayed release", "enteric coated", "dr", "ec"),
    "immediate_release": ("immediate release", "ir"),
}

# Route vocabulary. RxNorm expresses route inside the dose form ("Oral Tablet").
_ROUTE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "oral": ("oral", "by mouth", "duong uong", "uong", "po"),
    "intravenous": ("intravenous", "tinh mach", "iv"),
    "intramuscular": ("intramuscular", "tiem bap", "im"),
    "subcutaneous": ("subcutaneous", "duoi da", "sc", "sq"),
    "topical": ("topical", "ngoai da", "boi"),
    "rectal": ("rectal", "dat hau mon"),
    "ophthalmic": ("ophthalmic", "nho mat", "eye"),
    "nasal": ("nasal", "mui"),
    "inhalation": ("inhalation",),
}


def _canonical_from_synonyms(surface: str, table: dict[str, tuple[str, ...]]) -> str | None:
    """Canonical concept for a surface form, or ``None`` when unrecognised.

    Longest synonym wins, so `extended release` is not reduced to `er` and `vien nang`
    is not shadowed by `vien`.
    """
    key = normalize_surface(surface)
    if not key:
        return None
    best: tuple[int, str] | None = None
    for canonical, synonyms in table.items():
        for synonym in synonyms:
            if re.search(rf"(?<![a-z0-9]){re.escape(synonym)}(?![a-z0-9])", key):
                if best is None or len(synonym) > best[0]:
                    best = (len(synonym), canonical)
    return best[1] if best else None


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """One structured field, compared against one candidate concept."""

    field_name: str
    verdict: str
    mention_value: str = ""
    candidate_value: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "verdict": self.verdict,
            "mention": self.mention_value,
            "candidate": self.candidate_value,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class RxNormCandidateDecision:
    """Everything spec §10.2 asks to be recorded about one candidate decision."""

    concept_id: str
    tty: str
    retained: bool
    reason: str
    tier: str
    score: float
    retrieval_sources: tuple[str, ...] = field(default_factory=tuple)
    graph_path: str = ""
    graph_depth: int = 0
    matched_fields: tuple[str, ...] = field(default_factory=tuple)
    conflicting_fields: tuple[str, ...] = field(default_factory=tuple)
    not_stated_fields: tuple[str, ...] = field(default_factory=tuple)
    comparisons: tuple[FieldComparison, ...] = field(default_factory=tuple)
    snapshot_id: str = ""
    is_combination: bool = False
    brand_names: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "tty": self.tty,
            "retained": self.retained,
            "reason": self.reason,
            "tier": self.tier,
            "score": round(self.score, 4),
            "retrieval_sources": list(self.retrieval_sources),
            "graph_path": self.graph_path,
            "graph_depth": self.graph_depth,
            "matched_fields": list(self.matched_fields),
            "conflicting_fields": list(self.conflicting_fields),
            "not_stated_fields": list(self.not_stated_fields),
            "comparisons": [c.as_dict() for c in self.comparisons],
            "snapshot_id": self.snapshot_id,
            "is_combination": self.is_combination,
            "brand_names": list(self.brand_names),
        }


@dataclass(frozen=True, slots=True)
class RxNormLinkReport:
    """Per-mention linking report. Consumed by L6/L7/L8 and summarised in manifests."""

    hypothesis_id: str
    structured: StructuredMedicationMention
    decisions: tuple[RxNormCandidateDecision, ...] = field(default_factory=tuple)
    traversal_notes: tuple[str, ...] = field(default_factory=tuple)
    snapshot_id: str = ""
    linker_version: str = RXNORM_LINKER_VERSION
    graph_version: str = RXNORM_GRAPH_VERSION

    @property
    def retained(self) -> tuple[RxNormCandidateDecision, ...]:
        return tuple(d for d in self.decisions if d.retained)

    @property
    def suppressed(self) -> tuple[RxNormCandidateDecision, ...]:
        return tuple(d for d in self.decisions if not d.retained)

    @property
    def best_tier(self) -> str:
        for tier in TIER_ORDER:
            if any(d.tier == tier for d in self.retained):
                return tier
        return TIER_LEXICAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "linker_version": self.linker_version,
            "graph_version": self.graph_version,
            "hypothesis_id": self.hypothesis_id,
            "snapshot_id": self.snapshot_id,
            "best_tier": self.best_tier,
            "retained": len(self.retained),
            "suppressed": len(self.suppressed),
            "traversal_notes": list(self.traversal_notes),
            "decisions": [d.as_dict() for d in self.decisions],
        }


def _candidate_strengths(text: str) -> tuple[tuple[float, str], ...]:
    """Every `value unit` strength stated in a candidate's name."""
    found: list[tuple[float, str]] = []
    for raw_value, raw_unit in _NAME_STRENGTH.findall(text):
        unit = canonical_unit(raw_unit)
        try:
            value = float(raw_value.replace(",", "."))
        except ValueError:
            continue
        if unit:
            found.append((value, unit))
    return tuple(found)


def _compare_strength(
    mention: StructuredMedicationMention, candidate_name: str, tty: str
) -> FieldComparison:
    """Strength compatibility, including cross-unit mass comparison."""
    if mention.strength_value is None or mention.strength_value.value is None:
        return FieldComparison("strength", NOT_STATED)
    mention_unit = mention.strength_unit.unit if mention.strength_unit else None
    if mention.strength_unit is not None and mention_unit is None:
        return FieldComparison(
            "strength",
            NOT_COMPARABLE,
            mention.strength_value.text,
            note="mention unit not recognised by the conservative normalizer",
        )
    candidate_strengths = _candidate_strengths(candidate_name)
    if not candidate_strengths:
        # An ingredient or brand-name concept states no strength. That is not a
        # conflict — it is a less specific concept, which the tier system handles.
        note = (
            "candidate states no strength"
            if tty in (TTY_INGREDIENT | TTY_BRAND_NAME)
            else "no parsable strength in candidate name"
        )
        return FieldComparison("strength", NOT_COMPARABLE, mention.strength_value.text, note=note)

    mention_value = mention.strength_value.value
    mention_mg = strength_in_mg(mention_value, mention_unit)
    for value, unit in candidate_strengths:
        if mention_unit is not None and unit_family(unit) != unit_family(mention_unit):
            continue
        if mention_unit is not None and unit == mention_unit and value == mention_value:
            return FieldComparison(
                "strength", MATCH, f"{mention_value:g} {mention_unit}", f"{value:g} {unit}"
            )
        candidate_mg = strength_in_mg(value, unit)
        if (
            mention_mg is not None
            and candidate_mg is not None
            and (abs(candidate_mg - mention_mg) < 1e-9)
        ):
            return FieldComparison(
                "strength",
                MATCH,
                f"{mention_value:g} {mention_unit}",
                f"{value:g} {unit}",
                note="matched after mass conversion",
            )
    stated = ", ".join(f"{v:g} {u}" for v, u in candidate_strengths[:4])
    return FieldComparison(
        "strength",
        CONFLICT,
        f"{mention_value:g} {mention_unit or '?'}",
        stated,
        note="candidate states a different strength for a comparable unit",
    )


def _compare_unit(mention: StructuredMedicationMention, candidate_name: str) -> FieldComparison:
    """Unit-family compatibility, checked independently of the numeric value."""
    if mention.strength_unit is None:
        return FieldComparison("unit", NOT_STATED)
    mention_unit = mention.strength_unit.unit
    if mention_unit is None:
        return FieldComparison(
            "unit",
            NOT_COMPARABLE,
            mention.strength_unit.text,
            note="unit not in the conservative normalization table",
        )
    mention_family = unit_family(mention_unit)
    candidate_units = {unit for _value, unit in _candidate_strengths(candidate_name)}
    if not candidate_units:
        return FieldComparison("unit", NOT_COMPARABLE, mention_unit)
    families = {unit_family(u) for u in candidate_units}
    if mention_family in families:
        return FieldComparison("unit", MATCH, mention_unit, ",".join(sorted(candidate_units)))
    return FieldComparison(
        "unit",
        CONFLICT,
        mention_unit,
        ",".join(sorted(candidate_units)),
        note=f"unit family {mention_family} absent from candidate",
    )


def _ratio_mg_per_ml(match: re.Match[str]) -> float | None:
    numerator = canonical_unit(match.group(2))
    denominator_unit = canonical_unit(match.group(4))
    try:
        amount = float(match.group(1).replace(",", "."))
        volume = float((match.group(3) or "1").replace(",", "."))
    except ValueError:
        return None
    amount_mg = strength_in_mg(amount, numerator)
    if amount_mg is None or not volume:
        return None
    if denominator_unit == "l":
        volume *= 1000.0
    return amount_mg / volume


def _compare_concentration(
    mention: StructuredMedicationMention, candidate_name: str
) -> FieldComparison:
    """Concentration compatibility, e.g. mention `250 mg/5 mL` vs `500 MG in 5 mL`."""
    if mention.concentration is None:
        return FieldComparison("concentration", NOT_STATED)
    mention_match = _NAME_CONCENTRATION.search(mention.concentration.text)
    candidate_match = _NAME_CONCENTRATION.search(candidate_name)
    if mention_match is None:
        return FieldComparison(
            "concentration",
            NOT_COMPARABLE,
            mention.concentration.text,
            note="mention concentration not parsable",
        )
    if candidate_match is None:
        return FieldComparison(
            "concentration",
            NOT_COMPARABLE,
            mention.concentration.text,
            note="candidate states no concentration",
        )
    mention_ratio = _ratio_mg_per_ml(mention_match)
    candidate_ratio = _ratio_mg_per_ml(candidate_match)
    if mention_ratio is None or candidate_ratio is None:
        return FieldComparison(
            "concentration",
            NOT_COMPARABLE,
            mention.concentration.text,
            note="units outside the conservative conversion table",
        )
    if abs(mention_ratio - candidate_ratio) < 1e-9:
        return FieldComparison(
            "concentration", MATCH, mention.concentration.text, candidate_match.group(0)
        )
    return FieldComparison(
        "concentration",
        CONFLICT,
        mention.concentration.text,
        candidate_match.group(0),
        note="different mg-per-mL concentration",
    )


def _compare_vocabulary(
    field_name: str,
    mention_surface: str | None,
    candidate_name: str,
    table: dict[str, tuple[str, ...]],
) -> FieldComparison:
    """Shared comparison for dose form, release and route."""
    if mention_surface is None:
        return FieldComparison(field_name, NOT_STATED)
    mention_concept = _canonical_from_synonyms(mention_surface, table)
    if mention_concept is None:
        return FieldComparison(
            field_name,
            NOT_COMPARABLE,
            mention_surface,
            note="surface form not in the conservative vocabulary",
        )
    candidate_concept = _canonical_from_synonyms(candidate_name, table)
    if candidate_concept is None:
        return FieldComparison(
            field_name,
            NOT_COMPARABLE,
            mention_concept,
            note="candidate name states no comparable value",
        )
    if candidate_concept == mention_concept:
        return FieldComparison(field_name, MATCH, mention_concept, candidate_concept)
    return FieldComparison(
        field_name,
        CONFLICT,
        mention_concept,
        candidate_concept,
        note="candidate states a different value",
    )


_CONFLICT_TO_REASON = {
    "strength": DROP_STRENGTH_CONFLICT,
    "unit": DROP_UNIT_CONFLICT,
    "concentration": DROP_CONCENTRATION_CONFLICT,
    "dose_form": DROP_DOSE_FORM_CONFLICT,
    "release": DROP_RELEASE_CONFLICT,
    "route": DROP_ROUTE_CONFLICT,
}


def compare_candidate(
    mention: StructuredMedicationMention,
    index: LocalIndex,
    concept_id: str,
) -> tuple[FieldComparison, ...]:
    """Compare every stated structured field against one candidate concept."""
    name = name_of(index, concept_id)
    searchable = " ; ".join((name, *aliases_of(index, concept_id)[:8]))
    tty = tty_of(index, concept_id)
    return (
        _compare_strength(mention, name, tty),
        _compare_unit(mention, name),
        _compare_concentration(mention, name),
        _compare_vocabulary(
            "dose_form",
            mention.dose_form.text if mention.dose_form else None,
            searchable,
            _DOSE_FORM_SYNONYMS,
        ),
        _compare_vocabulary(
            "release",
            mention.release.text if mention.release else None,
            searchable,
            _RELEASE_SYNONYMS,
        ),
        _compare_vocabulary(
            "route", mention.route.text if mention.route else None, searchable, _ROUTE_SYNONYMS
        ),
    )


def _ingredient_needles(mention: StructuredMedicationMention) -> tuple[str, ...]:
    """Surfaces that count as naming this mention's ingredient.

    Governed crosswalk names can extend this set only after approval. The legacy
    INN bridge is review inventory and does not widen runtime retrieval in 3A.
    """
    if mention.ingredient is None:
        return ()
    surface = mention.ingredient.text
    needles = [normalize_surface(surface)]
    needles.extend(normalize_surface(n) for n in bridged_ingredient_names(surface))
    return tuple(n for n in needles if n)


def _ingredient_supported(needles: Sequence[str], index: LocalIndex, concept_id: str) -> bool:
    """Whether one of the ingredient surfaces appears in the candidate's names.

    Checked against canonical name, aliases and adjacent ingredient names, because
    `aspirin` is an alias of the concept whose canonical name is `ACETYLSALICYLIC
    ACID` — a live example from the locked snapshot.
    """
    if not needles:
        return True
    haystacks = (
        name_of(index, concept_id),
        *aliases_of(index, concept_id),
        *attached_ingredients(index, concept_id),
    )
    normalized = [normalize_surface(h) for h in haystacks if h]
    return any(needle in hay for needle in needles for hay in normalized)


def _tier_for(tty: str, matched: Sequence[str], comparisons: Sequence[FieldComparison]) -> str:
    stated = [c for c in comparisons if c.verdict in {MATCH, CONFLICT}]
    if (
        stated
        and all(c.verdict == MATCH for c in stated)
        and tty in (TTY_CLINICAL_DRUG | TTY_BRANDED_DRUG | TTY_COMPONENT)
    ):
        return TIER_STRUCTURED_EXACT
    if matched:
        return TIER_STRUCTURED_PARTIAL
    if tty in TTY_INGREDIENT:
        return TIER_INGREDIENT_ONLY
    return TIER_LEXICAL


def _score(tier: str, depth: int, retrieval_score: float, tty: str) -> float:
    """Deterministic ordering score. Not a probability, and never presented as one."""
    tier_weight = float(len(TIER_ORDER) - TIER_ORDER.index(tier)) * 1000.0
    tty_rank = TTY_PRIORITY.index(tty) if tty in TTY_PRIORITY else len(TTY_PRIORITY)
    tty_weight = float(len(TTY_PRIORITY) - tty_rank) * 10.0
    return tier_weight + tty_weight - float(depth) + min(retrieval_score, 200.0) / 1000.0


def _concept_key(concept_id: str) -> tuple[int, int, str]:
    return (0, int(concept_id), "") if concept_id.isdigit() else (1, 0, concept_id)


def _query_label(structured: StructuredMedicationMention, term: str, is_last: bool) -> str:
    key = normalize_surface(term)
    if structured.ingredient is not None and key == structured.ingredient.normalized:
        return "query:ingredient"
    if structured.brand is not None and key == structured.brand.normalized:
        return "query:brand"
    return "query:full_surface" if is_last else "query:qualified_name"


def link_rxnorm_structured(
    hypothesis: EntityHypothesis,
    index: LocalIndex,
    *,
    limit: int = CANDIDATE_BOUND,
) -> RxNormLinkReport:
    """Structured RxNorm linking for one medication hypothesis."""
    structured = build_structured_mention(hypothesis)
    if index.index_type != "rxnorm":
        return RxNormLinkReport(
            hypothesis.hypothesis_id,
            structured,
            traversal_notes=("wrong_index_type",),
            snapshot_id=index.source_snapshot_id,
        )

    # --- retrieval, on structured terms rather than only the whole surface ------
    hits: dict[str, CandidateHit] = {}
    sources: dict[str, set[str]] = {}
    terms = structured.query_terms()
    for rank, term in enumerate(terms):
        label = _query_label(structured, term, rank == len(terms) - 1)
        for hit in search_index(index, term, limit=RETRIEVAL_LIMIT):
            previous = hits.get(hit.concept_id)
            if previous is None or hit.score > previous.score:
                hits[hit.concept_id] = hit
            bucket = sources.setdefault(hit.concept_id, set())
            bucket.update(hit.channels)
            bucket.add(label)

    # --- traversal, from ingredient-level seeds --------------------------------
    seeds = tuple(
        sorted(
            (cid for cid in hits if tty_of(index, cid) in TTY_INGREDIENT),
            key=lambda cid: (-hits[cid].score, _concept_key(cid)),
        )
    )
    paths, notes = traverse_ingredient_chain(index, seeds or tuple(sorted(hits)))
    path_by_id: dict[str, GraphPath] = {p.concept_id: p for p in paths}

    needles = _ingredient_needles(structured)
    if structured.ingredient is not None:
        bridged = bridged_ingredient_names(structured.ingredient.text)
        if bridged:
            # Recorded, not hidden: a reader must be able to see that retrieval used a
            # curated INN equivalence rather than the name the clinician wrote.
            notes = notes + tuple(
                f"inn_bridge:{normalize_surface(structured.ingredient.text)}->"
                f"{normalize_surface(name)}:{INN_BRIDGE_STATUS}"
                for name in bridged
            )
        if not any(tty_of(index, cid) in TTY_INGREDIENT for cid in hits):
            notes = notes + ("ingredient_absent_from_snapshot",)

    decisions: list[RxNormCandidateDecision] = []
    for concept_id in sorted(set(hits) | set(path_by_id), key=_concept_key):
        tty = tty_of(index, concept_id)
        path = path_by_id.get(concept_id)
        retrieval_score = hits[concept_id].score if concept_id in hits else 0.0
        base: dict[str, Any] = dict(
            concept_id=concept_id,
            tty=tty,
            retrieval_sources=tuple(sorted(sources.get(concept_id, set()))),
            graph_path=path.as_text() if path else "",
            graph_depth=path.depth if path else 0,
            snapshot_id=index.source_snapshot_id,
        )

        if not index.exists(concept_id):
            decisions.append(
                RxNormCandidateDecision(
                    retained=False,
                    reason=DROP_NOT_IN_SNAPSHOT,
                    tier=TIER_LEXICAL,
                    score=0.0,
                    **base,
                )
            )
            continue
        if is_suppressed(index, concept_id):
            decisions.append(
                RxNormCandidateDecision(
                    retained=False,
                    reason=DROP_SUPPRESSED_CONCEPT,
                    tier=TIER_LEXICAL,
                    score=0.0,
                    **base,
                )
            )
            continue
        if is_closure_only(index, concept_id):
            # Reached by traversal, which is exactly what the closure is for. Emitting
            # it would defeat the point of building the closure separately.
            decisions.append(
                RxNormCandidateDecision(
                    retained=False,
                    reason=DROP_CLOSURE_ONLY,
                    tier=TIER_LEXICAL,
                    score=0.0,
                    **base,
                )
            )
            continue
        if tty not in TTY_TERMINAL:
            # Groupers and synonyms are evidence, never emitted codes.
            note = (
                "grouper/synonym TTY is evidence, not a prescribable concept"
                if tty in TTY_GROUPER
                else "non-terminal TTY"
            )
            decisions.append(
                RxNormCandidateDecision(
                    retained=False,
                    reason=DROP_NON_TERMINAL_TTY,
                    tier=TIER_LEXICAL,
                    score=0.0,
                    comparisons=(FieldComparison("tty", NOT_COMPARABLE, "", tty, note),),
                    **base,
                )
            )
            continue
        if not _ingredient_supported(needles, index, concept_id):
            decisions.append(
                RxNormCandidateDecision(
                    retained=False,
                    reason=DROP_INGREDIENT_MISMATCH,
                    tier=TIER_LEXICAL,
                    score=0.0,
                    **base,
                )
            )
            continue

        comparisons = compare_candidate(structured, index, concept_id)
        matched = tuple(c.field_name for c in comparisons if c.verdict == MATCH)
        conflicting = tuple(c.field_name for c in comparisons if c.verdict == CONFLICT)
        not_stated = tuple(c.field_name for c in comparisons if c.verdict == NOT_STATED)
        ingredients = attached_ingredients(index, concept_id)
        detail: dict[str, Any] = dict(
            comparisons=comparisons,
            matched_fields=matched,
            conflicting_fields=conflicting,
            not_stated_fields=not_stated,
            is_combination=len(ingredients) > 1,
            brand_names=attached_brand_names(index, concept_id)[:4],
            **base,
        )

        if conflicting:
            # Hard-negative suppression (spec §10.2). The first conflicting field in
            # declaration order names the reason, so the code is stable across runs.
            reason = next(
                _CONFLICT_TO_REASON[c.field_name]
                for c in comparisons
                if c.verdict == CONFLICT and c.field_name in _CONFLICT_TO_REASON
            )
            decisions.append(
                RxNormCandidateDecision(
                    retained=False, reason=reason, tier=TIER_LEXICAL, score=0.0, **detail
                )
            )
            continue

        tier = _tier_for(tty, matched, comparisons)
        if tier == TIER_STRUCTURED_EXACT:
            reason = KEEP_STRUCTURED_MATCH
        elif tier == TIER_INGREDIENT_ONLY:
            reason = (
                KEEP_EXACT_INGREDIENT
                if "exact" in sources.get(concept_id, set())
                else KEEP_INGREDIENT_FALLBACK
            )
        elif matched:
            reason = KEEP_STRUCTURED_MATCH
        else:
            reason = KEEP_LEXICAL_ONLY
        decisions.append(
            RxNormCandidateDecision(
                retained=True,
                reason=reason,
                tier=tier,
                score=_score(tier, base["graph_depth"], retrieval_score, tty),
                **detail,
            )
        )

    # Deterministic ordering, then the safety bound. Anything cut by the bound is
    # recorded as DROP_BUDGET, so "20 candidates" never silently means "only 20 exist".
    retained = sorted(
        (d for d in decisions if d.retained),
        key=lambda d: (TIER_ORDER.index(d.tier), -d.score, _concept_key(d.concept_id)),
    )
    final: list[RxNormCandidateDecision] = list(retained[:limit])
    final.extend(
        RxNormCandidateDecision(**{**_decision_fields(d), "retained": False, "reason": DROP_BUDGET})
        for d in retained[limit:]
    )
    final.extend(d for d in decisions if not d.retained)
    return RxNormLinkReport(
        hypothesis.hypothesis_id, structured, tuple(final), notes, index.source_snapshot_id
    )


def _decision_fields(decision: RxNormCandidateDecision) -> dict[str, Any]:
    return {
        "concept_id": decision.concept_id,
        "tty": decision.tty,
        "retained": decision.retained,
        "reason": decision.reason,
        "tier": decision.tier,
        "score": decision.score,
        "retrieval_sources": decision.retrieval_sources,
        "graph_path": decision.graph_path,
        "graph_depth": decision.graph_depth,
        "matched_fields": decision.matched_fields,
        "conflicting_fields": decision.conflicting_fields,
        "not_stated_fields": decision.not_stated_fields,
        "comparisons": decision.comparisons,
        "snapshot_id": decision.snapshot_id,
        "is_combination": decision.is_combination,
        "brand_names": decision.brand_names,
    }


def link_rxnorm(
    hypothesis: EntityHypothesis, index: LocalIndex, *, limit: int = CANDIDATE_BOUND
) -> LinkerResult:
    """Public L5 entry point. Same signature as before; structured underneath.

    ``LinkerResult`` is the contract L6/L7/L8/L9 already consume, so the structured
    report travels as evidence strings rather than by changing every downstream
    signature. ``link_rxnorm_structured`` returns the full report for callers that
    want the decisions themselves.
    """
    report = link_rxnorm_structured(hypothesis, index, limit=limit)
    if "wrong_index_type" in report.traversal_notes:
        return LinkerResult(hypothesis.hypothesis_id, (), ("wrong_index_type",))
    candidates = tuple(
        LinkedCandidate(
            code=decision.concept_id,
            score=decision.score,
            channels=tuple(c for c in decision.retrieval_sources if not c.startswith("query:")),
            snapshot_id=decision.snapshot_id,
            evidence=(
                f"tty:{decision.tty}",
                f"tier:{decision.tier}",
                f"reason:{decision.reason}",
                *(f"matched:{name}" for name in decision.matched_fields),
                *((f"graph:{decision.graph_path}",) if decision.graph_path else ()),
            ),
        )
        for decision in report.retained
    )
    suppressions = tuple(
        f"suppressed:{d.reason}:{d.concept_id}"
        for d in report.suppressed
        if d.reason in set(_CONFLICT_TO_REASON.values())
    )[:8]
    return LinkerResult(
        hypothesis.hypothesis_id, candidates, tuple(report.traversal_notes) + suppressions
    )


__all__ = [
    "CANDIDATE_BOUND",
    "CONFLICT",
    "DROP_BUDGET",
    "DROP_CLOSURE_ONLY",
    "DROP_CONCENTRATION_CONFLICT",
    "DROP_DOSE_FORM_CONFLICT",
    "DROP_INGREDIENT_MISMATCH",
    "DROP_NON_TERMINAL_TTY",
    "DROP_NOT_IN_SNAPSHOT",
    "DROP_RELEASE_CONFLICT",
    "DROP_ROUTE_CONFLICT",
    "DROP_STRENGTH_CONFLICT",
    "DROP_SUPPRESSED_CONCEPT",
    "DROP_UNIT_CONFLICT",
    "KEEP_EXACT_INGREDIENT",
    "KEEP_INGREDIENT_FALLBACK",
    "KEEP_LEXICAL_ONLY",
    "KEEP_STRUCTURED_MATCH",
    "MATCH",
    "NOT_COMPARABLE",
    "NOT_STATED",
    "RXNORM_LINKER_VERSION",
    "TIER_INGREDIENT_ONLY",
    "TIER_LEXICAL",
    "TIER_ORDER",
    "TIER_STRUCTURED_EXACT",
    "TIER_STRUCTURED_PARTIAL",
    "FieldComparison",
    "RxNormCandidateDecision",
    "RxNormLinkReport",
    "compare_candidate",
    "link_rxnorm",
    "link_rxnorm_structured",
]
