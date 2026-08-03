"""Per-mention record, option ordering and confidence policy (0082).

Three decisions live here because all three must be reproducible without re-running Qwen:

**Context extraction** turns a verified 0081 entity into the previous/current/next sentence
plus the section heading it sits under. Sentences are cut from the source; nothing is
rewritten.

**Option ordering** deliberately destroys retrieval order. If the options were listed by
rank, position would encode the retrievers' opinion and the model would inherit it. Instead
the concept ids are sorted - a stable order with no relation to any score - and then rotated
by a hash of the mention, so the same mention always produces the same arrangement while no
single concept permanently owns the first slot.

**Confidence** is decided from evidence the pipeline already has: whether the chosen concept
was found by more than one independent retriever, by more than one query view, or as an exact
governed alias. No threshold here was tuned against any answer key, and every input to the
policy is stored per mention so a different policy can be evaluated later without another
Qwen pass.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ..extraction.alignment import DocumentView
from ..extraction.assertions import Section, detect_sections, section_for
from .reformulation import MentionContext
from .retrieval import Candidate

VARIANT_ALLNULL = "allnull"
VARIANT_HIGH = "ontofusion_high"
VARIANT_BROAD = "ontofusion_broad"
VARIANTS: tuple[str, ...] = (VARIANT_ALLNULL, VARIANT_HIGH, VARIANT_BROAD)

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n+")


def sentences(source: str) -> list[tuple[int, int]]:
    """Sentence spans as offsets into the source. The text itself is never copied."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(source):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(source):
        spans.append((start, len(source)))
    return spans


def mention_context(
    document: DocumentView,
    *,
    name: str,
    mention: str,
    entity_type: str,
    start: int,
    end: int,
    sections: list[Section] | None = None,
) -> MentionContext:
    """Section heading plus the three sentences around the mention."""
    spans = sentences(document.source)
    index = next(
        (i for i, (left, right) in enumerate(spans) if left <= start < right), -1
    )
    def text(position: int) -> str:
        if 0 <= position < len(spans):
            left, right = spans[position]
            return document.source[left:right]
        return ""

    found = section_for(sections if sections is not None else detect_sections(document), start)
    return MentionContext(
        document=name,
        mention=mention,
        entity_type=entity_type,
        section=found.heading if found else "",
        previous_sentence=text(index - 1) if index > 0 else "",
        sentence=text(index) if index >= 0 else document.source[start:end],
        next_sentence=text(index + 1),
    )


def ordered_option_ids(concept_ids: list[str], mention: str) -> list[str]:
    """A stable arrangement that carries no retrieval information.

    Sorting alone would give low concept ids a permanent positional advantage, so the sorted
    list is rotated by a digest of the mention: deterministic for a given mention, unrelated
    to any score, and different across mentions.
    """
    ordered = sorted(set(concept_ids))
    if len(ordered) < 2:
        return ordered
    digest = hashlib.sha256(mention.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % len(ordered)
    return ordered[offset:] + ordered[:offset]


@dataclass
class MentionRecord:
    """Everything about one mention that a later analysis could need. No chain-of-thought."""

    document: str
    mention: str
    entity_type: str
    position: tuple[int, int]
    ontology: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    reformulation: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    candidates_before_pruning: int = 0
    candidates_after_pruning: int = 0
    prune_reasons: dict[str, int] = field(default_factory=dict)
    option_order: list[str] = field(default_factory=list)
    selection: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    high_confidence: bool = False
    support: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_id(self) -> str:
        return str(self.selection.get("concept_id", ""))

    def candidates_for(self, variant: str) -> list[str]:
        """The emitted candidate list for one variant. Never more than one concept."""
        if variant == VARIANT_ALLNULL or not self.selected_id:
            return []
        if variant == VARIANT_HIGH and not self.high_confidence:
            return []
        return [self.selected_id]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "mention": self.mention,
            "type": self.entity_type,
            "position": list(self.position),
            "ontology": self.ontology,
            "context": self.context,
            "reformulation": self.reformulation,
            "candidates": self.candidates,
            "candidates_before_pruning": self.candidates_before_pruning,
            "candidates_after_pruning": self.candidates_after_pruning,
            "prune_reasons": self.prune_reasons,
            "option_order": list(self.option_order),
            "selection": self.selection,
            "facts": self.facts,
            "high_confidence": self.high_confidence,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    """What counts as independent support for a set-wise selection.

    Deliberately generic: it asks whether more than one *independent* thing found the chosen
    concept, which is knowable from the pipeline alone. It is not calibrated against any
    answer key, and every input is recorded so another policy can be applied offline.
    """

    min_retrievers: int = 2
    min_views: int = 2
    exact_alias_is_sufficient: bool = True

    def evaluate(self, candidate: Candidate | None) -> tuple[bool, dict[str, Any]]:
        if candidate is None:
            return False, {"reason": "selected_concept_not_in_candidate_set"}
        support = {
            "retriever_count": candidate.retriever_count,
            "view_count": candidate.view_count,
            "exact_alias": candidate.exact_alias,
            "supporting_retrievers": sorted(candidate.supporting),
        }
        if self.exact_alias_is_sufficient and candidate.exact_alias:
            support["reason"] = "exact_governed_alias"
            return True, support
        if candidate.retriever_count >= self.min_retrievers:
            support["reason"] = "independent_retriever_agreement"
            return True, support
        if candidate.view_count >= self.min_views:
            support["reason"] = "independent_query_view_agreement"
            return True, support
        support["reason"] = "single_source_only"
        return False, support

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_retrievers": self.min_retrievers, "min_views": self.min_views,
            "exact_alias_is_sufficient": self.exact_alias_is_sufficient,
        }


def apply_candidates(
    entities: list[dict[str, Any]],
    records: list[MentionRecord],
    variant: str,
) -> list[dict[str, Any]]:
    """Attach candidates to a copy of the 0081 entities. Spans, types, assertions untouched.

    The entity rows come from 0081 and are reproduced field for field; the only difference
    between variants is the `candidates` list on the two candidate-bearing types.
    """
    by_position = {
        (r.position[0], r.position[1]): r for r in records if r.document
    }
    out: list[dict[str, Any]] = []
    for entity in entities:
        row = dict(entity)
        position = entity.get("position") or []
        record = (
            by_position.get((int(position[0]), int(position[1])))
            if len(position) == 2
            else None
        )
        if record is not None:
            row["candidates"] = record.candidates_for(variant)
        out.append(row)
    return out


__all__ = [
    "VARIANTS",
    "VARIANT_ALLNULL",
    "VARIANT_BROAD",
    "VARIANT_HIGH",
    "ConfidencePolicy",
    "MentionRecord",
    "apply_candidates",
    "mention_context",
    "ordered_option_ids",
    "sentences",
]
