"""The MedNorm-VI production pipeline: one document in, organizer JSON out.

This is the whole system. There is no second inference path.

    clinical document
      -> proposal sources         E3 seed + Qwen contextual passes + governed alias support
      -> exact-source alignment   offsets are computed here, never supplied by a model
      -> finite span/type lattice a closed set of alternatives
      -> Qwen finite verifier     accept / reject / retype, by local index only
      -> deterministic resolution span and type decided by fixed rules
      -> contextual assertions    isNegated / isFamily / isHistorical
      -> verified entities
      -> Qwen query reformulation two extra retrieval views, never entities
      -> multi-view retrieval     sparse character n-gram + SapBERT + ClinLinker (diagnosis)
      -> governed candidate union deduplicated by concept id, capped deterministically
      -> compatibility pruning    type gate plus structured conflicts
      -> Qwen set-wise reranking  one option letter, or NONE
      -> organizer JSON           one governed ICD-10 / RxCUI per mention, or none

Two properties hold by construction rather than by care. A model never produces a character
offset, so every span is a literal slice of the source; and a model never produces a concept
id, so every emitted code came from the governed knowledge base.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import RuntimeConfig
from ..extraction.aliases import AliasLexicon
from ..extraction.runtime import ContextualConfig, DocumentOutcome
from ..extraction.runtime import process_document as extract_document
from ..extraction.runtime import serialize as serialize_entities
from ..kb.indexing.retrieval import LocalIndex
from ..kb.rxnorm.structured import StructuredDrug
from ..linking.pipeline import MentionRecord, apply_candidates
from ..linking.runtime import OntoFusionConfig, RunOutcome
from ..linking.runtime import run as link_documents
from ..models.qwen import log
from ..models.registry import RetrieverSpec

#: The single production entity policy. `contextual_high` is the verified contextual set:
#: E3 plus contextual mentions carrying independent evidence. There is no variant family.
ENTITY_POLICY = "contextual_high"

#: The single production candidate policy: emit the set-wise selection when it carries
#: independent deterministic support, otherwise emit nothing for that mention.
CANDIDATE_POLICY = "ontofusion_high"


@dataclass
class PipelineOutcome:
    """Everything one run produced: entities, links and the evidence for both."""

    documents: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    extraction: list[DocumentOutcome] = field(default_factory=list)
    linking: RunOutcome | None = None
    seconds: float = 0.0

    @property
    def entity_count(self) -> int:
        return sum(len(rows) for rows in self.documents.values())

    @property
    def code_count(self) -> int:
        return sum(
            len(row.get("candidates") or [])
            for rows in self.documents.values()
            for row in rows
        )


def run(
    notes: dict[str, str],
    seed_entities: dict[str, list[dict[str, Any]]],
    *,
    specs: Sequence[RetrieverSpec],
    kb_documents: list[Any],
    runtime: RuntimeConfig,
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
    structured: dict[str, StructuredDrug],
    qwen: Any,
    qwen_revision: str,
    lexicon: AliasLexicon | None = None,
    extraction_config: ContextualConfig | None = None,
    linking_config: OntoFusionConfig | None = None,
    encoder_factory: Any = None,
) -> PipelineOutcome:
    """Extraction then linking, in one pass, for the whole corpus.

    The two halves are sequential rather than interleaved because they need the model in
    different shapes: extraction asks for text, linking asks for a choice among governed
    candidates, and the retrievers must be resident for neither.
    """
    started = time.time()
    outcome = PipelineOutcome()

    log(f"extraction: {len(notes)} documents")
    extracted: list[DocumentOutcome] = []
    for name, source in notes.items():
        extracted.append(
            extract_document(
                name, source, seed_entities.get(name, []),
                qwen=qwen, lexicon=lexicon, config=extraction_config,
            )
        )
    outcome.extraction = extracted

    entities = {
        document.document: serialize_entities(document, ENTITY_POLICY)
        for document in extracted
    }
    log(f"extraction complete: {sum(len(v) for v in entities.values())} verified entities")

    kwargs: dict[str, Any] = {}
    if encoder_factory is not None:
        kwargs["encoder_factory"] = encoder_factory
    linked = link_documents(
        notes, entities, specs, kb_documents, runtime,
        icd_index=icd_index, rxnorm_index=rxnorm_index, structured=structured,
        qwen=qwen, qwen_revision=qwen_revision, config=linking_config, **kwargs,
    )
    outcome.linking = linked

    by_document: dict[str, list[MentionRecord]] = {}
    for record in linked.records:
        by_document.setdefault(record.document, []).append(record)
    outcome.documents = {
        name: apply_candidates(rows, by_document.get(name, []), CANDIDATE_POLICY)
        for name, rows in entities.items()
    }
    outcome.seconds = time.time() - started
    log(
        f"linking complete: {outcome.code_count} governed codes on "
        f"{outcome.entity_count} entities in {outcome.seconds:.1f}s"
    )
    return outcome


def write_output(run_dir: Path, outcome: PipelineOutcome) -> Path:
    """Organizer JSON, one file per document, plus the evidence records."""
    out_dir = Path(run_dir) / "output_raw_dotless"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outcome.documents.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )

    if outcome.linking is not None:
        (Path(run_dir) / "mention-records.jsonl").write_text(
            "".join(
                json.dumps(r.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for r in outcome.linking.records
            ),
            encoding="utf-8",
        )
    (Path(run_dir) / "proposal-records.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "document": document.document,
                    "proposals": [p.as_dict() for p in document.pool.sorted()],
                    "entities": [
                        {
                            **entity.as_dict(),
                            "assertions": list(
                                document.assertions.get((entity.start, entity.end), ())
                            ),
                        }
                        for entity in document.entities
                    ],
                    "counters": document.counters,
                },
                ensure_ascii=False, sort_keys=True,
            )
            + "\n"
            for document in outcome.extraction
        ),
        encoding="utf-8",
    )
    return out_dir


def diagnostics(outcome: PipelineOutcome, *, manifest_total: int = 0) -> dict[str, Any]:
    """One summary covering both halves, plus the certified deployed total."""
    extraction: dict[str, int] = {}
    for document in outcome.extraction:
        for key, value in document.counters.items():
            extraction[key] = extraction.get(key, 0) + value
    reports = [r.as_dict() for d in outcome.extraction for r in d.reports]

    linking = dict(sorted(outcome.linking.counters.items())) if outcome.linking else {}
    return {
        "documents": len(outcome.documents),
        "entities": outcome.entity_count,
        "governed_codes": outcome.code_count,
        "runtime_seconds": round(outcome.seconds, 1),
        "peak_vram_gib": round(outcome.linking.peak_vram_gib, 2) if outcome.linking else 0.0,
        "canonical_manifest_total_parameters": manifest_total,
        "entity_policy": ENTITY_POLICY,
        "candidate_policy": CANDIDATE_POLICY,
        "extraction_counters": dict(sorted(extraction.items())),
        "extraction_generation_reports": reports,
        "linking_counters": linking,
    }


__all__ = [
    "CANDIDATE_POLICY",
    "ENTITY_POLICY",
    "PipelineOutcome",
    "diagnostics",
    "run",
    "write_output",
]
