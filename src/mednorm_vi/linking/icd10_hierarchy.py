"""ICD-10 hierarchy traversal and the specificity controller (spec §9, §9.3).

The hierarchy graph has been materialized and loaded on every run since Audit 0014.
Audits 0051-0053 each recorded that ``linking/icd10.py`` never read it. This module
reads it.

**What the artifact actually is** — established by inspecting the locked TT06-2026
index, not by reading the spec:

```text
graph            dict[concept_id, list[concept_id]]   14,534 nodes
edge labels      NONE
direction        NOT EXPLICIT — the adjacency is symmetric (0 asymmetric edges)
records          15,308, of which 774 are absent from the graph entirely
aliases          0 records have any alias; retrieval is canonical_name + n-gram + sparse
metadata         block, chapter, dotted_code, specificity ("0" | "1" | "2")
code lengths     3 chars 2,011  |  4 chars 9,790  |  5 chars 3,507
exact-name collisions  1,340 normalized names map to more than one code
```

Two consequences drive the design.

**Direction is reconstructed from code length**, because the graph does not carry it: a
neighbour whose code is shorter is an ancestor, longer is a descendant. This was
verified rather than assumed — every one of the 12,968 parent edges has a matching
child edge, and no neighbour pair shares a length. It is also why ``metadata.specificity``
is used only as a cross-check: it equals ``len(code) - 3`` for every record, so it is
**structural depth, not semantic specificity**, and treating it as clinical precision
would be exactly the mistake spec §9.3 warns against.

**The hierarchy is incomplete, and the code must survive that.** 410 four-character
codes have no three-character parent record (`A390` exists, `A39` does not), 454
three-character codes have no children, and 774 records are not in the graph at all —
including `J189` "Viêm phổi, không xác định", whose parent `J18` is absent from the
snapshot. So "walk up to the parent" is a request that legitimately fails, and a
missing ancestor is a fact to report, never an error to raise.

**The specificity rule, stated once.** A descendant may outrank its ancestor **only
when the mention's own text expresses the detail the descendant adds**. The added
detail is computed as the token difference between the descendant's canonical name and
the ancestor's; if those tokens are absent from the mention, the descendant is
suppressed with the missing tokens recorded and the ancestor is preferred. Depth alone
never wins. This is the whole of §9.3 in one deterministic rule, and it is auditable
because the missing tokens are on the decision.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..kb.indexing.retrieval import LocalIndex
from .structured_medication import normalize_surface

ICD10_HIERARCHY_VERSION = "icd10-hierarchy-v1"

# Relationship of a candidate to the best lexical hit.
REL_SELF = "self"
REL_ANCESTOR = "ancestor"
REL_DESCENDANT = "descendant"
REL_SIBLING = "sibling"
REL_UNRELATED = "unrelated"

# Traversal bounds. Bounds, not selection rules: exceeding one is recorded.
MAX_ANCESTOR_DEPTH = 3
MAX_DESCENDANTS = 24
MAX_SIBLINGS = 8

# Vietnamese function words that carry no diagnostic detail. A descendant whose only
# added tokens are these adds no specificity to require in the text.
_STOPWORDS: frozenset[str] = frozenset({
    "va", "hoac", "khong", "co", "cua", "voi", "trong", "do", "cac", "nhung", "mot",
    "la", "den", "tu", "ve", "cho", "tai", "khac", "xac", "dinh", "phan", "loai",
    "the", "dang", "benh", "hoi", "chung", "roi", "nhiem", "su", "bi", "bao", "gom",
})

_TOKEN = re.compile(r"[0-9a-z]+")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(normalize_surface(text)))


def content_tokens(text: str) -> frozenset[str]:
    """Diagnostic tokens: normalized, stopword-free, length >= 2."""
    return frozenset(t for t in _tokens(text) if len(t) > 1 and t not in _STOPWORDS)


def record_of(index: LocalIndex, code: str) -> Mapping[str, Any]:
    record = index.records.get(code)
    return record if isinstance(record, dict) else {}


def metadata_of(index: LocalIndex, code: str) -> Mapping[str, Any]:
    metadata = record_of(index, code).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def canonical_name(index: LocalIndex, code: str) -> str:
    return str(record_of(index, code).get("canonical_name", ""))


def dotted_code(index: LocalIndex, code: str) -> str:
    return str(metadata_of(index, code).get("dotted_code", code))


def declared_specificity(index: LocalIndex, code: str) -> int | None:
    """``metadata.specificity`` as an int, or ``None`` when absent.

    Reported for provenance. **Not** used as a ranking signal: it equals
    ``len(code) - 3`` throughout the locked snapshot, so it is depth wearing a
    misleading name.
    """
    raw = metadata_of(index, code).get("specificity")
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def graph_depth(code: str) -> int:
    """Structural depth from the code itself: 3 chars = 0, 4 = 1, 5 = 2."""
    return max(0, len(code) - 3)


def neighbours(index: LocalIndex, code: str) -> tuple[str, ...]:
    raw = index.graph.get(code)
    return tuple(str(n) for n in raw) if isinstance(raw, list) else ()


def parents(index: LocalIndex, code: str) -> tuple[str, ...]:
    """Immediate ancestors: neighbours with a shorter code (see module docstring)."""
    return tuple(sorted(n for n in neighbours(index, code) if len(n) < len(code)))


def children(index: LocalIndex, code: str) -> tuple[str, ...]:
    """Immediate descendants: neighbours with a longer code."""
    return tuple(sorted(n for n in neighbours(index, code) if len(n) > len(code)))


def ancestor_path(index: LocalIndex, code: str) -> tuple[str, ...]:
    """Ancestors from nearest to furthest, bounded.

    Returns ``()`` when the code has no ancestor in the snapshot — which happens for
    410 four-character codes whose three-character prefix was never published as a
    record. That is a snapshot property, not a failure.
    """
    path: list[str] = []
    current = code
    for _ in range(MAX_ANCESTOR_DEPTH):
        immediate = parents(index, current)
        if not immediate:
            break
        # Deterministic: the longest (nearest) parent, then lexicographic.
        nearest = sorted(immediate, key=lambda c: (-len(c), c))[0]
        path.append(nearest)
        current = nearest
    return tuple(path)


def descendants(index: LocalIndex, code: str) -> tuple[str, ...]:
    """Descendants reachable downward, bounded and deterministic."""
    seen: list[str] = []
    frontier = [code]
    while frontier and len(seen) < MAX_DESCENDANTS:
        nxt: list[str] = []
        for node in frontier:
            for child in children(index, node):
                if child not in seen:
                    seen.append(child)
                    nxt.append(child)
                if len(seen) >= MAX_DESCENDANTS:
                    break
        frontier = nxt
    return tuple(seen[:MAX_DESCENDANTS])


def siblings(index: LocalIndex, code: str) -> tuple[str, ...]:
    """Codes sharing this code's nearest ancestor, excluding itself.

    Bounded and used only as *competition* evidence: a mention that matches several
    siblings equally well is ambiguous, and that ambiguity is worth recording rather
    than resolving by coin flip.
    """
    immediate = parents(index, code)
    if not immediate:
        return ()
    parent = sorted(immediate, key=lambda c: (-len(c), c))[0]
    return tuple(c for c in children(index, parent) if c != code)[:MAX_SIBLINGS]


def relationship(index: LocalIndex, code: str, anchor: str) -> str:
    """How ``code`` relates to ``anchor`` in the locked hierarchy."""
    if code == anchor:
        return REL_SELF
    if code in ancestor_path(index, anchor):
        return REL_ANCESTOR
    if anchor in ancestor_path(index, code):
        return REL_DESCENDANT
    if code in siblings(index, anchor):
        return REL_SIBLING
    return REL_UNRELATED


@dataclass(frozen=True, slots=True)
class SpecificityAssessment:
    """Whether a mention's text justifies a descendant over its ancestor (spec §9.3)."""

    code: str
    ancestor: str
    added_tokens: tuple[str, ...]
    supported_tokens: tuple[str, ...]
    missing_tokens: tuple[str, ...]

    @property
    def justified(self) -> bool:
        """True when every token the descendant adds is present in the mention.

        Vacuously true when the descendant adds no *content* token — some ICD children
        differ from their parent only by wording like "không xác định", which asserts
        no extra clinical detail to look for.
        """
        return not self.missing_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "ancestor": self.ancestor,
            "added_tokens": list(self.added_tokens),
            "supported_tokens": list(self.supported_tokens),
            "missing_tokens": list(self.missing_tokens),
            "justified": self.justified,
        }


def assess_specificity(
    index: LocalIndex, code: str, ancestor: str, mention_text: str
) -> SpecificityAssessment:
    """Does the mention express the detail ``code`` adds over ``ancestor``?"""
    child_tokens = content_tokens(canonical_name(index, code))
    parent_tokens = content_tokens(canonical_name(index, ancestor))
    added = child_tokens - parent_tokens
    mention_tokens = content_tokens(mention_text)
    supported = added & mention_tokens
    return SpecificityAssessment(
        code=code, ancestor=ancestor,
        added_tokens=tuple(sorted(added)),
        supported_tokens=tuple(sorted(supported)),
        missing_tokens=tuple(sorted(added - mention_tokens)))


@dataclass(frozen=True, slots=True)
class HierarchyContext:
    """Everything the specificity controller knows about one candidate code."""

    code: str
    dotted: str
    depth: int
    declared_specificity: int | None
    relationship: str
    ancestor_path: tuple[str, ...] = field(default_factory=tuple)
    child_count: int = 0
    sibling_count: int = 0
    in_graph: bool = True
    specificity: SpecificityAssessment | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "dotted_code": self.dotted, "graph_depth": self.depth,
            "declared_specificity": self.declared_specificity,
            "relationship": self.relationship,
            "ancestor_path": list(self.ancestor_path),
            "child_count": self.child_count, "sibling_count": self.sibling_count,
            "in_graph": self.in_graph,
            "specificity": self.specificity.as_dict() if self.specificity else None,
        }


def hierarchy_context(
    index: LocalIndex, code: str, anchor: str, mention_text: str
) -> HierarchyContext:
    """Build the hierarchy view of one candidate relative to the lexical anchor."""
    path = ancestor_path(index, code)
    relation = relationship(index, code, anchor)
    assessment: SpecificityAssessment | None = None
    if path:
        assessment = assess_specificity(index, code, path[0], mention_text)
    return HierarchyContext(
        code=code, dotted=dotted_code(index, code), depth=graph_depth(code),
        declared_specificity=declared_specificity(index, code),
        relationship=relation, ancestor_path=path,
        child_count=len(children(index, code)),
        sibling_count=len(siblings(index, code)),
        in_graph=code in index.graph, specificity=assessment)


def expand_hierarchy(
    index: LocalIndex, anchors: Sequence[str], mention_text: str
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Expand lexical anchors upward, downward and sideways.

    Returns ``((code, origin_relationship), notes)``. Sibling expansion is deliberately
    the narrowest: siblings are only added for an anchor whose own added detail is
    **unsupported** by the mention, because that is the case where the lexical match
    may simply have landed on the wrong child.
    """
    reached: dict[str, str] = {}
    notes: list[str] = []
    for anchor in anchors:
        reached.setdefault(anchor, REL_SELF)
        if anchor not in index.graph:
            notes.append(f"anchor_absent_from_graph:{anchor}")
        path = ancestor_path(index, anchor)
        if not path and graph_depth(anchor) > 0:
            notes.append(f"no_ancestor_in_snapshot:{anchor}")
        for ancestor in path:
            reached.setdefault(ancestor, REL_ANCESTOR)
        for descendant in descendants(index, anchor):
            reached.setdefault(descendant, REL_DESCENDANT)
        if path:
            assessment = assess_specificity(index, anchor, path[0], mention_text)
            if not assessment.justified:
                for sibling in siblings(index, anchor):
                    reached.setdefault(sibling, REL_SIBLING)
                notes.append(f"sibling_competition_opened:{anchor}")
    ordered = tuple(sorted(reached.items(), key=lambda item: (
        graph_depth(item[0]), item[0])))
    return ordered, tuple(notes)


def hierarchy_health(index: LocalIndex) -> Mapping[str, int]:
    """Bounded structural census of the locked hierarchy, for audits and manifests."""
    codes = set(index.records)
    graph_nodes = set(index.graph)
    orphans = sum(
        1 for code in codes if len(code) >= 4 and code[:3] not in codes)
    childless = sum(1 for code in codes if len(code) == 3 and not children(index, code))
    return {
        "records": len(codes),
        "graph_nodes": len(graph_nodes),
        "records_absent_from_graph": len(codes - graph_nodes),
        "codes_without_parent_record": orphans,
        "three_char_codes_without_children": childless,
    }


__all__ = [
    "ICD10_HIERARCHY_VERSION",
    "MAX_ANCESTOR_DEPTH",
    "MAX_DESCENDANTS",
    "MAX_SIBLINGS",
    "REL_ANCESTOR",
    "REL_DESCENDANT",
    "REL_SELF",
    "REL_SIBLING",
    "REL_UNRELATED",
    "HierarchyContext",
    "SpecificityAssessment",
    "ancestor_path",
    "assess_specificity",
    "canonical_name",
    "children",
    "content_tokens",
    "declared_specificity",
    "descendants",
    "dotted_code",
    "expand_hierarchy",
    "graph_depth",
    "hierarchy_context",
    "hierarchy_health",
    "metadata_of",
    "neighbours",
    "parents",
    "record_of",
    "relationship",
    "siblings",
]
