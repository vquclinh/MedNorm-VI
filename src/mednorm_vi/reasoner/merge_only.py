"""Strict merge-only span lattice (experiment 0078).

The 0077 full run changed 152 clusters, but only 32 were the fragment merges the experiment
was testing - **120 were single-proposal boundary expansions**, and several were clearly
wrong: `'22' -> '22 tuần'`, `'3' -> '3 ngày'`, `'máu đông' -> 'có cục máu đông'`,
`'Bệnh phổi kẽ' -> '- Bệnh phổi kẽ'`. The hypothesis under test was drowned out by an
operation nobody wanted.

0078 removes every operation except one. A cluster is eligible only when it holds **two or
more** original E3 proposals of the same type, and the only replacement available is the exact
union `source[min(start):max(end)]`. There is no token-radius expansion, no single-proposal
expansion, no left or right growth, and no punctuation prefix - not disabled by a flag, but
absent from the action space. `'22' -> '22 tuần'` cannot be produced because a one-proposal
cluster is never offered to the model at all.

Assertions are frozen. A cluster whose proposals disagree on assertions is not eligible: it is
kept as-is rather than merged under a guessed flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .lattice import (
    _CLAUSE_BREAK,
    MAX_GAP_CHARS,
    _sentence_of,
    sentence_bounds,
)

MERGE_ONLY_VERSION = "merge-only-lattice-v1"

KEEP_ORIGINALS = "KEEP_ORIGINALS"
MERGE_TO_UNION = "MERGE_TO_UNION"
CHOICES: tuple[str, ...] = (KEEP_ORIGINALS, MERGE_TO_UNION)

PROVENANCE_STRICT_UNION = "strict_union_of_same_type_fragments"


@dataclass(frozen=True, slots=True)
class MergeCluster:
    """Two or more same-type proposals and the single exact union that may replace them."""

    cluster_id: int
    entity_type: str
    proposals: tuple[dict[str, Any], ...]
    union_start: int
    union_end: int
    union_text: str
    assertions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "type": self.entity_type,
            "original_texts": [p["text"] for p in self.proposals],
            "original_offsets": [list(p["position"]) for p in self.proposals],
            "union_text": self.union_text,
            "union_offsets": [self.union_start, self.union_end],
            "provenance": PROVENANCE_STRICT_UNION,
        }


def _assertions_of(proposal: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(proposal.get("assertions") or []))


def build_merge_clusters(
    source: str, proposals: list[dict[str, Any]]
) -> tuple[list[MergeCluster], list[dict[str, Any]]]:
    """``(eligible clusters, untouchable proposals)``.

    A proposal reaches the second list - and is emitted byte-identically - unless it joins a
    cluster of two or more same-type, same-clause, same-assertion proposals.
    """
    bounds = sentence_bounds(source)
    indexed = sorted(
        ({**p, "_start": int(p["position"][0]), "_end": int(p["position"][1])} for p in proposals),
        key=lambda p: (p["type"], p["_start"], p["_end"]),
    )

    groups: list[list[dict[str, Any]]] = []
    for proposal in indexed:
        joined = False
        if groups:
            last = groups[-1][-1]
            gap = source[max(last["_end"], 0) : proposal["_start"]]
            if (
                last["type"] == proposal["type"]
                and _sentence_of(last["_start"], bounds) == _sentence_of(proposal["_start"], bounds)
                and (
                    proposal["_start"] <= last["_end"]
                    or (
                        proposal["_start"] - last["_end"] <= MAX_GAP_CHARS
                        and not _CLAUSE_BREAK.search(gap)
                    )
                )
            ):
                groups[-1].append(proposal)
                joined = True
        if not joined:
            groups.append([proposal])

    clusters: list[MergeCluster] = []
    untouched: list[dict[str, Any]] = []
    cluster_id = 0
    for group in groups:
        plain = [{k: v for k, v in p.items() if not k.startswith("_")} for p in group]
        if len(group) < 2:
            untouched.extend(plain)  # a lone proposal is never offered to the model
            continue
        flags = {_assertions_of(p) for p in group}
        if len(flags) > 1:
            # Disagreeing assertions: keep originals rather than merge under a guess.
            untouched.extend(plain)
            continue
        start = min(p["_start"] for p in group)
        end = max(p["_end"] for p in group)
        clusters.append(
            MergeCluster(
                cluster_id=cluster_id,
                entity_type=group[0]["type"],
                proposals=tuple(plain),
                union_start=start,
                union_end=end,
                union_text=source[start:end],  # literal slice; the ONLY replacement offered
                assertions=next(iter(flags)),
            )
        )
        cluster_id += 1
    return clusters, untouched


def apply_merge(
    cluster: MergeCluster, choice: str | None, source: str
) -> tuple[list[dict[str, Any]], bool]:
    """``(entities, merged)``. Anything other than an explicit MERGE_TO_UNION keeps originals."""
    if choice != MERGE_TO_UNION:
        return [dict(p) for p in cluster.proposals], False
    assert source[cluster.union_start : cluster.union_end] == cluster.union_text
    merged: dict[str, Any] = {
        "text": cluster.union_text,
        "type": cluster.entity_type,  # frozen from E3
        "position": [cluster.union_start, cluster.union_end],
    }
    return [merged], True


__all__ = [
    "CHOICES",
    "KEEP_ORIGINALS",
    "MERGE_ONLY_VERSION",
    "MERGE_TO_UNION",
    "PROVENANCE_STRICT_UNION",
    "MergeCluster",
    "apply_merge",
    "build_merge_clusters",
]
