"""ICD-10 Super Linker retrieval/reranking contract."""

from __future__ import annotations

from ..kb.indexing.retrieval import LocalIndex, search_index
from ..resolution.models import EntityHypothesis
from .models import LinkedCandidate, LinkerResult


def link_icd10(
    hypothesis: EntityHypothesis,
    index: LocalIndex,
    *,
    limit: int = 20,
    dotted_output: bool = False,
) -> LinkerResult:
    """Retrieve ICD candidates; every emitted candidate must exist in the index."""
    if index.index_type != "icd10_vi":
        return LinkerResult(hypothesis.hypothesis_id, (), ("wrong_index_type",))
    hits = search_index(index, hypothesis.text, limit=limit)
    candidates: list[LinkedCandidate] = []
    for hit in hits:
        record = index.records[hit.concept_id]
        code = str(record.get("metadata", {}).get("dotted_code", hit.concept_id))
        if not dotted_output:
            code = hit.concept_id
        if index.exists(hit.concept_id):
            candidates.append(
                LinkedCandidate(
                    code=code,
                    score=hit.score,
                    channels=hit.channels,
                    snapshot_id=index.source_snapshot_id,
                    evidence=(f"icd10:{hit.concept_id}",),
                )
            )
    return LinkerResult(hypothesis.hypothesis_id, tuple(candidates))


__all__ = ["link_icd10"]
