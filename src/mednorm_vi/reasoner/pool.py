"""Parameter-free high-recall governed candidate pool (sprint 0075).

The 8B reasoner replaces the Audit-0073 embedding/reranker pair, which cannot co-deploy with
it under the 9B cap. So candidate generation must be **parameter-free**: lexical, normalized
and accent-insensitive retrieval over the governed indices, plus RxNorm structured fields
recovered in Audit 0074.

Recall is the objective, not lexical top-1 precision - the model does the choosing, and it can
only choose from what it is given. A concept missing from this pool is unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..kb.indexing.normalization import normalize_text
from ..kb.indexing.retrieval import LocalIndex, search_index
from ..kb.rxnorm.structured import StructuredDrug

#: Deliberately generous. Cost is prompt tokens, not parameters.
DEFAULT_POOL_LIMIT = 25


@dataclass(frozen=True, slots=True)
class Candidate:
    concept_id: str
    display_code: str
    name: str
    detail: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)

    def as_prompt_line(self) -> str:
        line = f"{self.display_code} | {self.name}"
        return f"{line} | {self.detail}" if self.detail else line


def _variants(mention: str) -> list[str]:
    """Surface forms worth querying. Deterministic and bounded."""
    base = (mention or "").strip()
    out = [base]
    for extra in (normalize_text(base), normalize_text(base, strip_accents=True)):
        if extra and extra not in out:
            out.append(extra)
    # Head noun: many Vietnamese mentions carry a trailing qualifier the lexicon lacks.
    words = base.split()
    if len(words) > 2:
        head = " ".join(words[:2])
        if head not in out:
            out.append(head)
    return out


def build_pool(
    mention: str,
    indexes: list[tuple[str, LocalIndex]],
    *,
    limit: int = DEFAULT_POOL_LIMIT,
    structured: dict[str, StructuredDrug] | None = None,
    dotted: bool = False,
) -> list[Candidate]:
    """Union lexical retrieval across every supplied index and surface variant."""
    merged: dict[str, list[str]] = {}
    for label, index in indexes:
        for variant in _variants(mention):
            for hit in search_index(index, variant, limit=limit):
                merged.setdefault(hit.concept_id, [])
                if label not in merged[hit.concept_id]:
                    merged[hit.concept_id].append(label)

    records: dict[str, dict[str, Any]] = {}
    for _, index in indexes:
        for concept_id in merged:
            if concept_id not in records and index.exists(concept_id):
                records[concept_id] = index.records[concept_id]

    out: list[Candidate] = []
    for concept_id, sources in merged.items():
        record = records.get(concept_id)
        if record is None:
            continue  # never offer a concept the governed KB does not contain
        metadata = record.get("metadata") or {}
        display = str(metadata.get("dotted_code", concept_id)) if dotted else concept_id
        detail = ""
        drug = (structured or {}).get(concept_id)
        if drug is not None:
            bits = []
            if drug.tty:
                bits.append(drug.tty)
            if drug.ingredients:
                bits.append("ingredient " + "/".join(drug.ingredients[:2]))
            if drug.strengths:
                bits.append("strength " + (drug.strengths[0].raw or drug.strengths[0].key))
            elif drug.available_strength:
                bits.append("strength " + drug.available_strength)
            if drug.dose_forms:
                bits.append("form " + drug.dose_forms[0])
            elif drug.rxterm_form:
                bits.append("form " + drug.rxterm_form)
            if drug.brands:
                bits.append("brand " + drug.brands[0])
            detail = "; ".join(bits)
        elif metadata.get("tty"):
            detail = str(metadata["tty"])
        out.append(
            Candidate(
                concept_id=concept_id,
                display_code=display,
                name=str(record.get("canonical_name", "")),
                detail=detail,
                sources=tuple(sources),
            )
        )
    out.sort(key=lambda c: (-len(c.sources), c.concept_id))
    return out[:limit]


__all__ = ["DEFAULT_POOL_LIMIT", "Candidate", "build_pool"]
