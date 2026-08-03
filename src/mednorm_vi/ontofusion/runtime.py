"""OntoFusion runtime: the three-phase lifecycle (0082).

The phase order exists because of memory, and it is not an optimization detail - it is what
lets the whole pipeline run on a 22.5 GiB L4:

    1. load Qwen  -> reformulate every unique mention -> UNLOAD Qwen
    2. load each retriever in turn -> retrieve -> UNLOAD it
    3. load Qwen  -> set-wise rerank every mention   -> UNLOAD Qwen

Qwen is loaded twice rather than held resident across retrieval, because holding an 8B model
and a semantic encoder together is what would not fit. Peak VRAM is therefore roughly the
largest single model, and it is measured rather than assumed.

Nothing here mutates GraphCENT: the KB loader, the dense caches, the encoders, the ontology
facts and the model manifest are all reused as they are. 0082 is a second linker, so
`0080 = E3 + GraphCENT`, `0081 = contextual + GraphCENT` and `0082 = contextual + OntoFusion`
remain three clean ablations of one variable each.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contextual.assertions import detect_sections
from ..contextual.document import DocumentView
from ..graphcent.encoders import build_encoder
from ..graphcent.models import RetrieverSpec
from ..graphcent.ontology import ONTOLOGY_ICD, IcdFacts, RxNormFacts, icd_facts, rxnorm_facts
from ..graphcent.runtime import RuntimeConfig, exact_alias_ids, load_index
from ..kb.indexing.retrieval import LocalIndex
from ..kb.rxnorm.structured import StructuredDrug
from .pipeline import (
    VARIANTS,
    ConfidencePolicy,
    MentionRecord,
    apply_candidates,
    mention_context,
    ordered_option_ids,
)
from .pruning import linkable, ontology_for, prune
from .reformulation import MentionContext, ReformulationCache
from .reranking import build_options, build_rerank_prompt, parse_selection
from .retrieval import (
    CandidateSet,
    RetrievalPolicy,
    assert_role_allowed,
    retrieve_dense,
    retrieve_sparse,
    role_for_ontology,
)
from .sparse import SparseSettings, SurfaceMemo


def log(message: str) -> None:
    print(f"[0082] {message}", flush=True)


def peak_vram_gib() -> float:
    """Peak allocated device memory since the process started, or 0 without CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 2**30
    except ImportError:  # pragma: no cover - torch present on any run host
        pass
    return 0.0


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class OntoFusionConfig:
    """0082 settings. Model paths and caches come from the GraphCENT runtime config."""

    retrieval: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    sparse: SparseSettings = field(default_factory=SparseSettings)
    confidence: ConfidencePolicy = field(default_factory=ConfidencePolicy)
    use_reformulation: bool = True
    allow_multi_code: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval.as_dict(),
            "sparse": self.sparse.as_dict(),
            "confidence": self.confidence.as_dict(),
            "use_reformulation": self.use_reformulation,
            "allow_multi_code": self.allow_multi_code,
        }


@dataclass
class RunOutcome:
    records: list[MentionRecord] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    peak_vram_gib: float = 0.0
    seconds: float = 0.0

    def count(self, reason: str, amount: int = 1) -> None:
        if amount:
            self.counters[reason] = self.counters.get(reason, 0) + amount

    def merge(self, counters: dict[str, int]) -> None:
        for key, value in counters.items():
            self.count(key, value)


# ------------------------------------------------------------------ phase 1: reformulate


@dataclass(frozen=True, slots=True)
class LinkableMention:
    document: str
    mention: str
    entity_type: str
    start: int
    end: int
    ontology: str


def collect_mentions(
    notes: dict[str, str], entities: dict[str, list[dict[str, Any]]]
) -> list[LinkableMention]:
    """Only the two candidate-bearing organizer types ever reach the linker."""
    out: list[LinkableMention] = []
    for document, rows in entities.items():
        source = notes.get(document, "")
        for row in rows:
            entity_type = str(row.get("type", ""))
            if not linkable(entity_type):
                continue
            position = row.get("position") or []
            if len(position) != 2:
                continue
            start, end = int(position[0]), int(position[1])
            out.append(
                LinkableMention(
                    document=document, mention=source[start:end], entity_type=entity_type,
                    start=start, end=end, ontology=ontology_for(entity_type),
                )
            )
    return out


def reformulate_all(
    notes: dict[str, str],
    mentions: Sequence[LinkableMention],
    *,
    qwen: Any,
    revision: str,
    config: OntoFusionConfig,
    outcome: RunOutcome,
) -> tuple[dict[str, MentionContext], ReformulationCache]:
    """Phase 1. Qwen is resident here and nowhere else in this function's caller."""
    views = {name: DocumentView.build(text) for name, text in notes.items()}
    sections = {name: detect_sections(view) for name, view in views.items()}
    cache = ReformulationCache(
        revision=revision,
        document_checksums={name: checksum(text) for name, text in notes.items()},
    )

    contexts: dict[str, MentionContext] = {}
    for mention in mentions:
        view = views.get(mention.document)
        if view is None:
            continue
        context = mention_context(
            view, name=mention.document, mention=mention.mention,
            entity_type=mention.entity_type, start=mention.start, end=mention.end,
            sections=sections.get(mention.document, []),
        )
        contexts[_mention_key(mention)] = context
        cache.reformulate(context, qwen if config.use_reformulation else None)

    outcome.count("unique_reformulations", len(cache.entries))
    outcome.count("reformulation_qwen_calls", cache.calls)
    outcome.merge(cache.counters)
    for entry in cache.entries.values():
        outcome.count("reformulation_parse_failures", int(entry.parse_failed))
        outcome.count("canonical_vi_non_empty", int(bool(entry.canonical_vi)))
        outcome.count("canonical_en_non_empty", int(bool(entry.canonical_en)))
    return contexts, cache


def _mention_key(mention: LinkableMention) -> str:
    return f"{mention.document}\x00{mention.start}\x00{mention.end}"


# ------------------------------------------------------------------ phase 2: retrieval


def dense_hits(
    mentions: Sequence[LinkableMention],
    contexts: dict[str, MentionContext],
    cache: ReformulationCache,
    specs: Sequence[RetrieverSpec],
    documents: list[Any],
    graph_config: RuntimeConfig,
    config: OntoFusionConfig,
    *,
    encoder_factory: Any = build_encoder,
    outcome: RunOutcome | None = None,
) -> dict[str, dict[str, list[tuple[str, str, int]]]]:
    """One load/unload per retriever for the whole corpus. Returns hits per mention key.

    A retriever only sees a mention whose role it declares, which is how ClinLinker stays
    out of medication retrieval - the constraint comes from the model registry, and is
    asserted here rather than restated.
    """
    queries: list[tuple[str, str, str, str]] = []   # (key, view, text, ontology)
    for mention in mentions:
        key = _mention_key(mention)
        context = contexts.get(key)
        if context is None:
            continue
        reformulation = cache.entries.get(
            context.cache_key(
                revision=cache.revision,
                document_checksum=cache.document_checksums.get(mention.document, ""),
            )
        )
        texts = reformulation.views if reformulation else (mention.mention,)
        names = ("raw", "canonical_vi", "canonical_en")
        for position, text in enumerate(texts):
            queries.append((key, names[min(position, 2)], text, mention.ontology))

    hits: dict[str, dict[str, list[tuple[str, str, int]]]] = {}
    if not queries:
        return hits

    for spec in specs:
        if not spec.enabled:
            continue
        relevant = [
            q for q in queries
            if role_for_ontology(q[3]) in spec.role and q[1] in config.retrieval.dense_views
        ]
        if not relevant:
            log(f"{spec.key}: no queries for its declared roles {list(spec.role)}")
            continue
        for _, _, _, ontology in relevant:
            assert_role_allowed(spec.role, ontology, spec.key)

        index = load_index(spec, documents, graph_config)
        encoder = encoder_factory(spec, graph_config.model_root, graph_config.device)
        vectors = encoder.encode(
            [text for _, _, text, _ in relevant], batch_size=graph_config.embed_batch_size
        )
        encoder.unload()
        for position, (key, view, _text, ontology) in enumerate(relevant):
            found = index.search(
                vectors[position], ontology=ontology, top_k=config.retrieval.dense_top_k
            )
            bucket = hits.setdefault(key, {}).setdefault(spec.key, [])
            for concept_id, rank in found:
                bucket.append((view, concept_id, rank))
        if outcome is not None:
            outcome.count(f"dense_queries_{spec.key}", len(relevant))
        log(f"{spec.key}: {len(relevant)} queries retrieved, encoder unloaded")
    return hits


# ------------------------------------------------------------------ phase 3: rerank


def _facts_for(
    concept_id: str,
    ontology: str,
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
    structured: dict[str, StructuredDrug],
    mention: str,
) -> IcdFacts | RxNormFacts | None:
    if ontology == ONTOLOGY_ICD:
        return icd_facts(icd_index, concept_id) if icd_index else None
    return rxnorm_facts(rxnorm_index, concept_id, mention, structured) if rxnorm_index else None


def link_mentions(
    notes: dict[str, str],
    mentions: Sequence[LinkableMention],
    contexts: dict[str, MentionContext],
    cache: ReformulationCache,
    dense: dict[str, dict[str, list[tuple[str, str, int]]]],
    *,
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
    structured: dict[str, StructuredDrug],
    qwen: Any,
    config: OntoFusionConfig,
    outcome: RunOutcome,
) -> list[MentionRecord]:
    """Union, prune, order, rerank. Qwen is resident here; the retrievers are not."""
    governed_icd = frozenset(icd_index.records) if icd_index else frozenset()
    governed_rx = frozenset(rxnorm_index.records) if rxnorm_index else frozenset()
    memo_icd = SurfaceMemo(icd_index, config.sparse) if icd_index else None
    memo_rx = SurfaceMemo(rxnorm_index, config.sparse) if rxnorm_index else None

    records: list[MentionRecord] = []
    for mention in mentions:
        key = _mention_key(mention)
        context = contexts.get(key)
        if context is None:
            continue
        reformulation = cache.entries.get(
            context.cache_key(
                revision=cache.revision,
                document_checksum=cache.document_checksums.get(mention.document, ""),
            )
        )
        views_map: dict[str, str] = {"raw": mention.mention}
        if reformulation:
            views_map["canonical_vi"] = reformulation.canonical_vi
            views_map["canonical_en"] = reformulation.canonical_en

        is_icd = mention.ontology == ONTOLOGY_ICD
        index = icd_index if is_icd else rxnorm_index
        governed = governed_icd if is_icd else governed_rx

        candidates = CandidateSet(ontology=mention.ontology)
        retrieve_sparse(
            candidates, index, views_map, config.retrieval,
            memo=memo_icd if is_icd else memo_rx, settings=config.sparse,
        )
        retrieve_dense(candidates, dense.get(key, {}), config.retrieval)
        for concept_id in exact_alias_ids(index, mention.mention):
            if concept_id in candidates.candidates:
                candidates.mark_exact(concept_id)
        outcome.merge(candidates.counters)
        outcome.count("union_size", candidates.size)

        before = candidates.capped(config.retrieval.candidate_cap)
        pruned = prune(
            before, entity_type=mention.entity_type, mention=mention.mention,
            governed=governed, structured=structured,
        )
        outcome.merge(pruned.reasons)

        facts_by_id = {
            candidate.concept_id: _facts_for(
                candidate.concept_id, mention.ontology, icd_index, rxnorm_index,
                structured, mention.mention,
            )
            for candidate in pruned.kept
        }
        order = ordered_option_ids(
            [c.concept_id for c in pruned.kept], mention.mention
        )
        options = build_options(order, facts_by_id)

        selection_dict: dict[str, Any] = {"decision": "NONE", "concept_id": ""}
        if options and qwen is not None:
            reply = qwen.ask(
                build_rerank_prompt(
                    mention.mention, mention.entity_type, options,
                    section=context.section,
                    previous_sentence=context.previous_sentence,
                    sentence=context.sentence,
                    next_sentence=context.next_sentence,
                )
            )
            selection = parse_selection(reply, options)
            outcome.merge(selection.counters)
            selection_dict = selection.as_dict()
            outcome.count("select" if selection.selected else "none")
        elif not options:
            outcome.count("no_candidates")

        chosen = next(
            (c for c in pruned.kept if c.concept_id == selection_dict.get("concept_id")),
            None,
        )
        high, support = config.confidence.evaluate(chosen)
        high = high and bool(selection_dict.get("concept_id"))

        record = MentionRecord(
            document=mention.document, mention=mention.mention,
            entity_type=mention.entity_type, position=(mention.start, mention.end),
            ontology=mention.ontology,
            context=context.as_dict(),
            reformulation=reformulation.as_dict() if reformulation else {},
            candidates=[c.as_dict() for c in pruned.kept],
            candidates_before_pruning=len(before),
            candidates_after_pruning=len(pruned.kept),
            prune_reasons=dict(pruned.reasons),
            option_order=order,
            selection=selection_dict,
            facts={
                cid: (f.as_dict() if f else None)
                for cid, f in facts_by_id.items()
                if cid == selection_dict.get("concept_id")
            },
            high_confidence=high,
            support=support,
        )
        if record.selected_id:
            outcome.count(
                "diagnosis_selections" if is_icd else "drug_selections"
            )
            outcome.count("high_selections", int(high))
        records.append(record)
    return records


# ------------------------------------------------------------------ whole run


def run(
    notes: dict[str, str],
    entities: dict[str, list[dict[str, Any]]],
    specs: Sequence[RetrieverSpec],
    documents: list[Any],
    graph_config: RuntimeConfig,
    *,
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
    structured: dict[str, StructuredDrug],
    qwen: Any,
    qwen_revision: str,
    config: OntoFusionConfig | None = None,
    encoder_factory: Any = build_encoder,
) -> RunOutcome:
    """The three phases, in order, with an unload between each."""
    settings = config or OntoFusionConfig()
    started = time.time()
    outcome = RunOutcome()

    mentions = collect_mentions(notes, entities)
    outcome.count("documents", len(notes))
    outcome.count("verified_linkable_mentions", len(mentions))
    log(f"linkable mentions {len(mentions):,} across {len(notes)} documents")

    log("phase 1: Qwen resident, reformulating")
    contexts, cache = reformulate_all(
        notes, mentions, qwen=qwen, revision=qwen_revision, config=settings,
        outcome=outcome,
    )
    if qwen is not None and hasattr(qwen, "unload"):
        qwen.unload()
        log("phase 1 complete: Qwen unloaded")

    log("phase 2: retrievers resident one at a time")
    dense = dense_hits(
        mentions, contexts, cache, specs, documents, graph_config, settings,
        encoder_factory=encoder_factory, outcome=outcome,
    )
    log("phase 2 complete: all retrievers unloaded")

    log("phase 3: Qwen resident, set-wise reranking")
    records = link_mentions(
        notes, mentions, contexts, cache, dense,
        icd_index=icd_index, rxnorm_index=rxnorm_index, structured=structured,
        qwen=qwen, config=settings, outcome=outcome,
    )
    if qwen is not None and hasattr(qwen, "unload"):
        qwen.unload()
        log("phase 3 complete: Qwen unloaded")

    outcome.records = records
    outcome.peak_vram_gib = peak_vram_gib()
    outcome.seconds = time.time() - started
    outcome.count(
        "multi_code_selections",
        sum(1 for r in records if len(r.candidates_for("ontofusion_broad")) > 1),
    )
    return outcome


def write_variants(
    run_dir: Path,
    entities: dict[str, list[dict[str, Any]]],
    outcome: RunOutcome,
) -> dict[str, int]:
    """Three variants from one pass. Only the candidate lists differ."""
    by_document: dict[str, list[MentionRecord]] = {}
    for record in outcome.records:
        by_document.setdefault(record.document, []).append(record)

    counts: dict[str, int] = {}
    for variant in VARIANTS:
        out_dir = Path(run_dir) / f"output_{variant}_raw_dotless"
        out_dir.mkdir(parents=True, exist_ok=True)
        emitted = 0
        for document, rows in entities.items():
            serialized = apply_candidates(rows, by_document.get(document, []), variant)
            emitted += sum(len(r.get("candidates") or []) for r in serialized)
            (out_dir / f"{document}.json").write_text(
                json.dumps(serialized, ensure_ascii=False), encoding="utf-8"
            )
        counts[variant] = emitted
        log(f"variant {variant:18s} codes emitted {emitted}")

    (Path(run_dir) / "mention-records.jsonl").write_text(
        "".join(
            json.dumps(r.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for r in outcome.records
        ),
        encoding="utf-8",
    )
    return counts


def diagnostics(
    outcome: RunOutcome, counts: dict[str, int], config: OntoFusionConfig,
    *, manifest_total: int = 0,
) -> dict[str, Any]:
    return {
        "config": config.as_dict(),
        "variant_code_counts": counts,
        "peak_vram_gib": round(outcome.peak_vram_gib, 2),
        "runtime_seconds": round(outcome.seconds, 1),
        "canonical_manifest_total_parameters": manifest_total,
        "union_size_mean": (
            round(outcome.counters.get("union_size", 0) / len(outcome.records), 2)
            if outcome.records else 0.0
        ),
        "counters": dict(sorted(outcome.counters.items())),
    }


__all__ = [
    "LinkableMention",
    "OntoFusionConfig",
    "RunOutcome",
    "checksum",
    "collect_mentions",
    "dense_hits",
    "diagnostics",
    "link_mentions",
    "peak_vram_gib",
    "reformulate_all",
    "run",
    "write_variants",
]
