"""RxNorm Super Linker retrieval/reranking contract."""

from __future__ import annotations

from ..kb.indexing.retrieval import LocalIndex, search_index
from ..resolution.models import EntityHypothesis
from .models import LinkedCandidate, LinkerResult

TTY_PRIORITY = ("IN", "PIN", "SCDC", "SCDF", "SCD", "SBD", "BN")


def _tty_bonus(record: dict[str, object]) -> float:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return 0.0
    tty = str(metadata.get("tty", ""))
    if tty not in TTY_PRIORITY:
        return 0.0
    return float(max(0, len(TTY_PRIORITY) - TTY_PRIORITY.index(tty)))


def link_rxnorm(
    hypothesis: EntityHypothesis, index: LocalIndex, *, limit: int = 20
) -> LinkerResult:
    """Retrieve RxCUIs; every emitted candidate must exist in the declared snapshot."""
    if index.index_type != "rxnorm":
        return LinkerResult(hypothesis.hypothesis_id, (), ("wrong_index_type",))
    hits = search_index(index, hypothesis.text, limit=limit)
    candidates = tuple(
        LinkedCandidate(
            code=hit.concept_id,
            score=hit.score + _tty_bonus(index.records[hit.concept_id]),
            channels=hit.channels,
            snapshot_id=index.source_snapshot_id,
            evidence=(f"rxnorm:{hit.concept_id}",),
        )
        for hit in hits
        if index.exists(hit.concept_id)
    )
    return LinkerResult(
        hypothesis.hypothesis_id,
        tuple(sorted(candidates, key=lambda c: (-c.score, c.code))[:limit]),
    )


__all__ = ["link_rxnorm"]
