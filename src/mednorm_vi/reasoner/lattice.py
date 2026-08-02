"""Finite span lattice for boundary correction (experiment 0077).

The 0076 GPU smoke kept every proposal but retyped nine of them, and the retypes were wrong:
`G6PD` CHẨN_ĐOÁN -> TÊN_XÉT_NGHIỆM, `viêm tuyến mồ hôi` CHẨN_ĐOÁN -> TÊN_XÉT_NGHIỆM,
`thẳng` TRIỆU_CHỨNG -> CHẨN_ĐOÁN. Unconstrained semantic retyping is not trustworthy, so here
**type is frozen**: it is inherited from E3 and no decision can change it.

The same smoke showed the more interesting failure. Document 1 carried `MEN`, `G6PD`,
`Thiếu`, `Thiếu men` - fragments of one clinical phrase, extracted as four entities. WER is
~82.72 on the public best, so boundary fragmentation plausibly costs more than candidate
linking ever did.

This module builds, in code and before any model runs, a small finite set of span options for
each cluster of same-type proposals. **Every option is a literal slice of the original note.**
The model may only choose ids from that set, so it cannot invent text, offsets or types - not
as a matter of filtering, but as a matter of what it is able to express.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

LATTICE_VERSION = "span-lattice-v1"

KEEP_ORIGINALS = "KEEP_ORIGINALS"

#: Two same-type proposals join a cluster when they overlap, one contains the other, or only
#: this many characters separate them. Deliberately small: `Thiếu` / `Thiếu men` / `G6PD` sit
#: within a few characters, while two genuinely distinct diagnoses in one sentence do not.
MAX_GAP_CHARS = 16
#: How far a boundary may be pushed outward, in whitespace tokens, on each side.
EXPANSION_TOKEN_RADIUS = 2
#: Hard cap per cluster so the lattice stays finite and the prompt stays short.
MAX_CANDIDATES_PER_CLUSTER = 8

_SENTENCE = re.compile(r"[.;\n]+")
_TOKEN = re.compile(r"\S+")
#: Clause punctuation between two proposals blocks clustering. A comma is how Vietnamese
#: clinical notes separate distinct findings, and the first lattice happily merged
#: `G6PD` with `viêm tuyến mồ hôi` across `bẩm sinh,` - two unrelated concepts.
_CLAUSE_BREAK = re.compile(r"[,;:()\[\]/]|\bvà\b|\bhoặc\b")

PROVENANCE_ORIGINAL = "e3_original"
PROVENANCE_UNION = "union_of_cluster"
PROVENANCE_UNION_SNAPPED = "union_snapped_to_token_boundaries"
PROVENANCE_EXPANDED = "token_radius_expansion"


@dataclass(frozen=True, slots=True)
class SpanCandidate:
    """One selectable span. `text` is always `source[start:end]`, never synthesised."""

    candidate_id: str
    text: str
    start: int
    end: int
    entity_type: str
    provenance: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class Cluster:
    """Same-type proposals that may describe one phrase, plus their span options."""

    cluster_id: int
    entity_type: str
    proposals: tuple[dict[str, Any], ...]
    candidates: tuple[SpanCandidate, ...]

    @property
    def multiple(self) -> bool:
        return len(self.proposals) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "type": self.entity_type,
            "proposals": [
                {"text": p["text"], "position": p["position"], "type": p["type"]}
                for p in self.proposals
            ],
            "candidates": [c.as_dict() for c in self.candidates],
        }


def sentence_bounds(source: str) -> list[tuple[int, int]]:
    """Character ranges of coarse sentences. Clustering never crosses one."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE.finditer(source):
        if match.start() > start:
            bounds.append((start, match.start()))
        start = match.end()
    if start < len(source):
        bounds.append((start, len(source)))
    return bounds or [(0, len(source))]


def _sentence_of(position: int, bounds: list[tuple[int, int]]) -> int:
    for index, (start, end) in enumerate(bounds):
        if start <= position < end:
            return index
    return -1


def _snap(source: str, start: int, end: int) -> tuple[int, int]:
    """Widen to the enclosing whitespace-token boundaries. Never crosses whitespace inward."""
    while start > 0 and not source[start - 1].isspace():
        start -= 1
    while end < len(source) and not source[end].isspace():
        end += 1
    return start, end


def build_clusters(source: str, proposals: list[dict[str, Any]]) -> list[Cluster]:
    """Group same-type proposals and enumerate their deterministic span options.

    Proposals of different organizer types are never grouped, however close they sit: the
    0076 smoke showed that mixing types is exactly where the model goes wrong.
    """
    bounds = sentence_bounds(source)
    indexed = sorted(
        ({**p, "_start": int(p["position"][0]), "_end": int(p["position"][1])} for p in proposals),
        key=lambda p: (p["type"], p["_start"], p["_end"]),
    )

    groups: list[list[dict[str, Any]]] = []
    for proposal in indexed:
        placed = False
        if groups:
            last = groups[-1][-1]
            same_type = last["type"] == proposal["type"]
            same_sentence = _sentence_of(last["_start"], bounds) == _sentence_of(
                proposal["_start"], bounds
            )
            overlaps = proposal["_start"] <= last["_end"]
            gap = source[max(last["_end"], 0) : proposal["_start"]]
            near = proposal["_start"] - last["_end"] <= MAX_GAP_CHARS and not _CLAUSE_BREAK.search(
                gap
            )
            if same_type and same_sentence and (overlaps or near):
                groups[-1].append(proposal)
                placed = True
        if not placed:
            groups.append([proposal])

    clusters: list[Cluster] = []
    for cluster_id, group in enumerate(groups):
        clusters.append(_build_cluster(source, cluster_id, group, bounds))
    return clusters


def _add_candidate(
    accumulator: dict[tuple[int, int], SpanCandidate],
    source: str,
    cluster_id: int,
    entity_type: str,
    start: int,
    end: int,
    provenance: str,
) -> None:
    """Record one span option. Module-level so loop variables are bound explicitly."""
    start, end = max(0, start), min(len(source), end)
    if end <= start or (start, end) in accumulator:
        return
    if len(accumulator) >= MAX_CANDIDATES_PER_CLUSTER:
        return
    accumulator[(start, end)] = SpanCandidate(
        candidate_id=f"c{cluster_id}_{len(accumulator)}",
        text=source[start:end],  # literal slice: never synthesised
        start=start,
        end=end,
        entity_type=entity_type,
        provenance=provenance,
    )


def _build_cluster(
    source: str, cluster_id: int, group: list[dict[str, Any]], bounds: list[tuple[int, int]]
) -> Cluster:
    entity_type = group[0]["type"]
    seen: dict[tuple[int, int], SpanCandidate] = {}

    def add(start: int, end: int, provenance: str) -> None:
        _add_candidate(seen, source, cluster_id, entity_type, start, end, provenance)

    for proposal in group:
        add(proposal["_start"], proposal["_end"], PROVENANCE_ORIGINAL)

    low = min(p["_start"] for p in group)
    high = max(p["_end"] for p in group)
    if len(group) > 1:
        add(low, high, PROVENANCE_UNION)
    add(*_snap(source, low, high), PROVENANCE_UNION_SNAPPED)

    # Expansion never crosses clause punctuation: a span spilling over a comma joins two
    # findings into one entity, which is the opposite of fixing fragmentation.
    sentence = bounds[max(0, _sentence_of(low, bounds))]
    tokens = [
        (a + sentence[0], b + sentence[0])
        for a, b in (m.span() for m in _TOKEN.finditer(source[sentence[0] : sentence[1]]))
    ]
    left = [t for t in tokens if t[1] <= low]
    right = [t for t in tokens if t[0] >= high]
    for radius in range(1, EXPANSION_TOKEN_RADIUS + 1):
        if len(left) >= radius:
            start = left[-radius][0]
            if not _CLAUSE_BREAK.search(source[start:low]):
                add(start, high, PROVENANCE_EXPANDED)
        if len(right) >= radius:
            end = right[radius - 1][1]
            if not _CLAUSE_BREAK.search(source[high:end]):
                add(low, end, PROVENANCE_EXPANDED)

    return Cluster(
        cluster_id=cluster_id,
        entity_type=entity_type,
        proposals=tuple({k: v for k, v in p.items() if not k.startswith("_")} for p in group),
        candidates=tuple(seen.values()),
    )


def apply_selection(
    cluster: Cluster, selected_ids: list[str] | None, source: str
) -> list[dict[str, Any]]:
    """Replace a cluster's proposals with the chosen spans, or keep the originals.

    Anything unrecognised - an unknown id, an empty choice, a malformed reply - resolves to
    KEEP_ORIGINALS. Boundary correction is the experiment; pruning is not.
    """
    by_id = {c.candidate_id: c for c in cluster.candidates}
    chosen = [by_id[i] for i in (selected_ids or []) if i in by_id]
    if not chosen:
        return [dict(p) for p in cluster.proposals]

    ordered = sorted({(c.start, c.end): c for c in chosen}.values(), key=lambda c: c.start)
    out: list[dict[str, Any]] = []
    for candidate in ordered:
        assert source[candidate.start : candidate.end] == candidate.text
        row: dict[str, Any] = {
            "text": candidate.text,
            "type": cluster.entity_type,  # FROZEN: inherited from E3, never model-chosen
            "position": [candidate.start, candidate.end],
        }
        out.append(row)
    return out


__all__ = [
    "EXPANSION_TOKEN_RADIUS",
    "KEEP_ORIGINALS",
    "LATTICE_VERSION",
    "MAX_CANDIDATES_PER_CLUSTER",
    "MAX_GAP_CHARS",
    "PROVENANCE_EXPANDED",
    "PROVENANCE_ORIGINAL",
    "PROVENANCE_UNION",
    "PROVENANCE_UNION_SNAPPED",
    "Cluster",
    "SpanCandidate",
    "apply_selection",
    "build_clusters",
    "sentence_bounds",
]
