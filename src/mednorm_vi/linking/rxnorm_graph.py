"""TTY-aware traversal of the locked RxNorm graph (spec §10.2).

**The single most important fact about this module, established by inspecting the real
snapshot rather than the spec:** the generated RxNorm index stores its graph as

    graph: dict[concept_id, list[concept_id]]

with **no relation labels and no direction**. The builder saw 2,563,978 relations
(``metadata.parameters.relations_seen``) and flattened them into a symmetric adjacency
list — a bounded sample found 0 of 21,353 sampled edges lacking a reverse. So spec
§10.2's "traverse ingredient → component → clinical drug → branded drug" cannot be
done by reading relation names, because the relation names are not in the artifact.

What *is* in the artifact is ``metadata.tty`` on every record. Direction is therefore
reconstructed from the **TTY of the endpoints**: a step is legal when the neighbour's
TTY is the next role in the chain. This was verified against the live snapshot before
being relied on — an ingredient's neighbours really do include SCDC, an SCDC's really
do include SCD, and an SCD's really do include SBD and DF:

    IN 1191  ->  SCDC "aspirin 162.5 MG"  ->  SCD "24 HR ASA 162.5 MG Extended
                 Release Oral Capsule"    ->  SBD "... [Durlaza]"

The consequence, stated plainly so nobody over-reads this module: these are
**TTY-role transitions inferred from an unlabeled graph**, not RxNorm's named
relations (``has_ingredient``, ``consists_of``, ``tradename_of``). Every path this
module returns records the TTY sequence it walked, so a reader can see exactly what
was inferred. Rebuilding the index with labeled edges would let this module tighten
from "a neighbour with the right TTY" to "a neighbour across the right relation", and
that is recorded as a KB-intake improvement rather than hidden here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..kb.indexing.retrieval import LocalIndex

RXNORM_GRAPH_VERSION = "rxnorm-tty-traversal-v1"

# TTY roles, as they actually occur in the locked prescribable snapshot. Counts from
# the live index are in Audit 0054 §3; they are not repeated here because a count is
# a property of a snapshot, not of this code.
TTY_INGREDIENT: frozenset[str] = frozenset({"IN", "PIN", "MIN"})
TTY_COMPONENT: frozenset[str] = frozenset({"SCDC", "SBDC"})
TTY_CLINICAL_DRUG: frozenset[str] = frozenset({"SCD"})
TTY_BRANDED_DRUG: frozenset[str] = frozenset({"SBD"})
TTY_BRAND_NAME: frozenset[str] = frozenset({"BN"})
TTY_DOSE_FORM: frozenset[str] = frozenset({"DF", "DFG"})
TTY_PACK: frozenset[str] = frozenset({"GPCK", "BPCK"})
# Grouper and form-specific TTYs. Useful as evidence, never as a final candidate:
# a grouper is not a prescribable product.
TTY_GROUPER: frozenset[str] = frozenset(
    {"SCDG", "SBDG", "SCDF", "SBDF", "SCDGP", "SCDFP", "SBDFP", "DFG"})

# Preference order for candidate ordering. A more specific product outranks a less
# specific one only when the mention's evidence justifies the specificity — the
# linker decides that; this is the tie-break skeleton.
TTY_PRIORITY: tuple[str, ...] = (
    "SCD", "SBD", "SCDC", "SBDC", "IN", "PIN", "MIN", "BN", "GPCK", "BPCK",
)

# Roles a traversal may end on. Groupers, synonyms and dose forms are evidence.
TTY_TERMINAL: frozenset[str] = (
    TTY_INGREDIENT | TTY_COMPONENT | TTY_CLINICAL_DRUG | TTY_BRANDED_DRUG | TTY_PACK
)

# The chain spec §10.2 names, expressed in the vocabulary the artifact actually has.
INGREDIENT_TO_BRANDED: tuple[frozenset[str], ...] = (
    TTY_INGREDIENT, TTY_COMPONENT, TTY_CLINICAL_DRUG, TTY_BRANDED_DRUG,
)

# Traversal is bounded so a 6,643-degree hub cannot turn one mention into a walk of
# the whole snapshot. These are safety bounds, not selection rules: exceeding one is
# recorded as a truncation, never presented as "no more candidates exist".
MAX_NEIGHBOURS_PER_STEP = 64
MAX_NODES_PER_STEP = 48
MAX_PATHS = 256


def tty_of(index: LocalIndex, concept_id: str) -> str:
    """TTY for a concept, or ``""`` when the record carries none."""
    record = index.records.get(concept_id)
    if not isinstance(record, dict):
        return ""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("tty", ""))


def name_of(index: LocalIndex, concept_id: str) -> str:
    record = index.records.get(concept_id)
    return str(record.get("canonical_name", "")) if isinstance(record, dict) else ""


def aliases_of(index: LocalIndex, concept_id: str) -> tuple[str, ...]:
    record = index.records.get(concept_id)
    if not isinstance(record, dict):
        return ()
    aliases = record.get("aliases")
    return tuple(str(a) for a in aliases) if isinstance(aliases, list) else ()


def membership_of(index: LocalIndex, concept_id: str) -> str:
    """Membership class for a concept, or ``""`` when the index does not state one.

    The competition v3 index carries ``searchable`` or ``closure_only`` here. The
    legacy index carries nothing, and an absent value must mean "this index draws no
    such distinction" rather than "closure-only" — treating silence as exclusion would
    empty the candidate list on the legacy snapshot.
    """
    record = index.records.get(concept_id)
    if not isinstance(record, dict):
        return ""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("membership", ""))


def is_closure_only(index: LocalIndex, concept_id: str) -> bool:
    """Whether traversal may reach this concept but the runtime may not emit it.

    Closure-only concepts exist so the ingredient/product walk has somewhere to land
    (see ``kb/competition/rxnorm_view.py``). They are not prescribable candidates, and
    the walk reaches them by construction, so this check is what stops the leak.
    """
    return membership_of(index, concept_id) == "closure_only"


def is_suppressed(index: LocalIndex, concept_id: str) -> bool:
    """True when the snapshot marks the concept suppressed or obsoleted.

    ``suppress`` is present on every record in the locked snapshot; ``RXN_OBSOLETED``
    on a minority. A suppressed concept exists in the snapshot, so it passes
    membership validation — it simply must not be offered as a candidate.
    """
    record = index.records.get(concept_id)
    if not isinstance(record, dict):
        return False
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return False
    if str(metadata.get("suppress", "N")).upper() not in {"N", "", "NONE"}:
        return True
    return "RXN_OBSOLETED" in metadata


@dataclass(frozen=True, slots=True)
class GraphStep:
    """One TTY-role transition actually walked in the locked graph."""

    from_id: str
    from_tty: str
    to_id: str
    to_tty: str

    def as_text(self) -> str:
        return f"{self.from_tty}:{self.from_id}->{self.to_tty}:{self.to_id}"


@dataclass(frozen=True, slots=True)
class GraphPath:
    """A concept reached by traversal, with the exact path that reached it."""

    concept_id: str
    tty: str
    steps: tuple[GraphStep, ...] = field(default_factory=tuple)
    truncated: bool = False

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def origin_id(self) -> str:
        return self.steps[0].from_id if self.steps else self.concept_id

    def as_text(self) -> str:
        if not self.steps:
            return f"{self.tty}:{self.concept_id}"
        return " | ".join(step.as_text() for step in self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id, "tty": self.tty, "depth": self.depth,
            "path": self.as_text(), "truncated": self.truncated,
        }


def neighbours(index: LocalIndex, concept_id: str) -> tuple[str, ...]:
    """Adjacency for a concept. Undirected and unlabeled — see the module docstring."""
    raw = index.graph.get(concept_id)
    return tuple(str(n) for n in raw) if isinstance(raw, list) else ()


def step_to_role(
    index: LocalIndex,
    origins: Sequence[GraphPath],
    target_ttys: frozenset[str],
) -> tuple[tuple[GraphPath, ...], bool]:
    """Advance every origin one hop to neighbours whose TTY is in ``target_ttys``.

    Returns the reached paths and whether any bound truncated the walk. Truncation is
    surfaced rather than swallowed: a caller that reports "these are the candidates"
    must be able to say whether it saw all of them.
    """
    reached: dict[str, GraphPath] = {}
    truncated = False
    for origin in origins:
        adjacency = neighbours(index, origin.concept_id)
        if len(adjacency) > MAX_NEIGHBOURS_PER_STEP:
            adjacency = adjacency[:MAX_NEIGHBOURS_PER_STEP]
            truncated = True
        for neighbour in adjacency:
            neighbour_tty = tty_of(index, neighbour)
            if neighbour_tty not in target_ttys:
                continue
            if neighbour in reached:
                continue
            step = GraphStep(origin.concept_id, origin.tty, neighbour, neighbour_tty)
            reached[neighbour] = GraphPath(
                concept_id=neighbour, tty=neighbour_tty,
                steps=origin.steps + (step,), truncated=origin.truncated or truncated)
    # Deterministic ordering: shallowest first, then TTY preference, then id. Concept
    # ids are numeric strings in RxNorm, so they are compared numerically where
    # possible to keep ordering stable and human-predictable.
    ordered = sorted(reached.values(), key=lambda p: (
        p.depth, _tty_rank(p.tty), _numeric_key(p.concept_id)))
    if len(ordered) > MAX_NODES_PER_STEP:
        ordered = ordered[:MAX_NODES_PER_STEP]
        truncated = True
    return tuple(ordered), truncated


def _tty_rank(tty: str) -> int:
    return TTY_PRIORITY.index(tty) if tty in TTY_PRIORITY else len(TTY_PRIORITY)


def _numeric_key(concept_id: str) -> tuple[int, int | str]:
    return (0, int(concept_id)) if concept_id.isdigit() else (1, concept_id)


def traverse_ingredient_chain(
    index: LocalIndex,
    seed_ids: Sequence[str],
) -> tuple[tuple[GraphPath, ...], tuple[str, ...]]:
    """Walk spec §10.2's chain from ingredient-level seeds.

    Returns every concept reached at any role in the chain — the ingredient seeds
    themselves included, because an ingredient-only candidate is the correct
    conservative answer when a mention states nothing but a name.
    """
    notes: list[str] = []
    collected: dict[str, GraphPath] = {}
    frontier: list[GraphPath] = []
    for seed in seed_ids:
        tty = tty_of(index, seed)
        if not tty:
            notes.append(f"seed_without_tty:{seed}")
            continue
        path = GraphPath(concept_id=seed, tty=tty)
        collected[seed] = path
        if tty in TTY_INGREDIENT:
            frontier.append(path)
    if not frontier:
        notes.append("no_ingredient_level_seed")

    for role in INGREDIENT_TO_BRANDED[1:]:
        if not frontier:
            break
        reached, truncated = step_to_role(index, frontier, role)
        if truncated:
            notes.append(f"traversal_truncated_at:{sorted(role)[0]}")
        for path in reached:
            collected.setdefault(path.concept_id, path)
        frontier = list(reached)

    # Dose form and brand name are evidence attached to a reached product, not
    # candidates. They are collected separately so the linker can compare them
    # without their ever being emitted as a code.
    if len(collected) > MAX_PATHS:
        notes.append(f"path_budget_exceeded:{len(collected)}")
    ordered = sorted(collected.values(), key=lambda p: (
        p.depth, _tty_rank(p.tty), _numeric_key(p.concept_id)))
    return tuple(ordered[:MAX_PATHS]), tuple(notes)


def attached_dose_forms(index: LocalIndex, concept_id: str) -> tuple[str, ...]:
    """Dose-form names adjacent to a product, for dose-form compatibility."""
    return tuple(sorted(
        name_of(index, n) for n in neighbours(index, concept_id)
        if tty_of(index, n) in TTY_DOSE_FORM))


def attached_brand_names(index: LocalIndex, concept_id: str) -> tuple[str, ...]:
    """Brand names adjacent to a product, for generic-versus-brand evidence."""
    return tuple(sorted(
        name_of(index, n) for n in neighbours(index, concept_id)
        if tty_of(index, n) in TTY_BRAND_NAME))


def attached_ingredients(index: LocalIndex, concept_id: str) -> tuple[str, ...]:
    """Ingredient names adjacent to a concept, for combination-product detection."""
    return tuple(sorted(
        name_of(index, n) for n in neighbours(index, concept_id)
        if tty_of(index, n) in TTY_INGREDIENT))


def graph_health(index: LocalIndex) -> Mapping[str, int]:
    """Bounded structural census, for the audit and for the run manifest.

    Deliberately cheap and deliberately not a quality claim: it counts what the
    artifact contains, which is the only honest thing to say about a graph whose
    relation labels were discarded at build time.
    """
    records = set(index.records)
    graph_nodes = set(index.graph)
    isolated = sum(1 for node, adj in index.graph.items() if not adj)
    return {
        "records": len(records),
        "graph_nodes": len(graph_nodes),
        "records_absent_from_graph": len(records - graph_nodes),
        "graph_nodes_absent_from_records": len(graph_nodes - records),
        "isolated_graph_nodes": isolated,
    }


__all__ = [
    "INGREDIENT_TO_BRANDED",
    "MAX_NEIGHBOURS_PER_STEP",
    "MAX_NODES_PER_STEP",
    "MAX_PATHS",
    "RXNORM_GRAPH_VERSION",
    "TTY_BRANDED_DRUG",
    "TTY_BRAND_NAME",
    "TTY_CLINICAL_DRUG",
    "TTY_COMPONENT",
    "TTY_DOSE_FORM",
    "TTY_GROUPER",
    "TTY_INGREDIENT",
    "TTY_PACK",
    "TTY_PRIORITY",
    "TTY_TERMINAL",
    "GraphPath",
    "GraphStep",
    "aliases_of",
    "attached_brand_names",
    "attached_dose_forms",
    "attached_ingredients",
    "graph_health",
    "is_closure_only",
    "is_suppressed",
    "membership_of",
    "name_of",
    "neighbours",
    "step_to_role",
    "traverse_ingredient_chain",
    "tty_of",
]
