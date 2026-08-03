"""Complete GraphCENT runtime: index build, retrieval, disambiguation, variants (0080).

Sequential by design. The index build holds one retriever at a time and unloads it before the
next; all retrieval evidence for the whole corpus is precomputed and the retrievers are
released before Qwen3-8B is loaded at all. Peak VRAM is therefore roughly the largest single
model, not their sum - which is what lets this run on an L4 as well as an A100.

That is a *memory* story only. The competition budget sums every enabled deployed model
regardless of when it is resident, and `certify_budget` does exactly that.

One inference pass writes one neutral record per mention holding the evidence and the model's
decision. The four submission variants are deterministic re-serializations of those records,
so Qwen runs once, not four times.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..kb.indexing.normalization import normalize_text
from ..kb.indexing.retrieval import LocalIndex, search_index
from ..kb.rxnorm.structured import StructuredDrug, parse_strengths
from ..reasoner.budget import QWEN3_8B_REVISION
from ..reasoner.validator import CANDIDATE_TYPES
from .acquisition import file_digest
from .disambiguation import (
    NULL_DECISION,
    VARIANTS,
    Decision,
    TierPolicy,
    build_prompt,
    parse_decision,
)
from .encoders import build_encoder
from .models import (
    ROLE_DIAGNOSIS,
    ROLE_DRUG,
    DeployedModel,
    ModelManifest,
    RetrieverSpec,
    RevisionNotPinned,
    is_hub_revision,
)
from .ontology import (
    ONTOLOGY_ICD,
    ONTOLOGY_RXNORM,
    IcdFacts,
    RxNormFacts,
    build_kb_documents,
    icd_facts,
    rxnorm_facts,
)
from .pipeline import ONTOLOGY_FOR_TYPE, MentionResult, linkable, serialize_document
from .retrieval import (
    MAX_CANDIDATE_CONTEXT,
    TOP_K_PER_RETRIEVER,
    CachedIndex,
    CacheManifest,
    KbDocument,
    checksum,
    fuse,
    write_cache,
)
from .spans import alternatives_for

RETRIEVER_ROLE_FOR_TYPE: dict[str, str] = {
    "CHẨN_ĐOÁN": ROLE_DIAGNOSIS,
    "THUỐC": ROLE_DRUG,
}


def log(message: str) -> None:
    print(f"[0080] {message}", flush=True)


def parameters_from_safetensors(root: Path) -> int:
    """Sum tensor elements from safetensors headers alone. 0 when there is nothing to read.

    Each shard begins with an 8-byte little-endian header length followed by a JSON header
    mapping tensor name to dtype/shape/offsets. Reading shapes needs no weights in memory,
    and tied weights are stored once, exactly as `model.parameters()` counts them once.
    """
    total = 0
    shards = sorted(Path(root).glob("*.safetensors")) if Path(root).is_dir() else []
    for shard in shards:
        with shard.open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            if length <= 0:
                continue
            header = json.loads(handle.read(length).decode("utf-8"))
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            count = 1
            for dim in meta.get("shape") or []:
                count *= int(dim)
            total += count if (meta.get("shape") or []) else 0
    return total


class Disambiguator(Protocol):
    """Anything that answers one prompt. Injected so tests need no weights."""

    def ask(self, prompt: str) -> str: ...

    def unload(self) -> None: ...


@dataclass
class QwenDisambiguator:
    """The real local Qwen3-8B. Greedy, bounded, loaded lazily."""

    model_root: Path
    device: str = "cuda"
    max_new_tokens: int = 256
    _model: Any = None
    _tokenizer: Any = None
    system_prompt: str = (
        "You are a precise clinical entity linker. You return STRICT JSON and nothing else. "
        "You select candidate indices only; you never write a medical code."
    )

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = str(self.model_root)
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._model = (
            AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        config = self._model.generation_config
        config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(config, name):
                setattr(config, name, None)
        log("Qwen3-8B loaded (greedy, sampling parameters cleared)")

    def parameter_count(self) -> int:
        """Exact count, preferably without materializing 8B parameters.

        The safetensors header carries every tensor's shape, so the budget can be certified
        on a host that could not hold the model - which matters because `download-models`
        only requires enough memory for a retriever. Loading is the fallback, not the path.
        """
        counted = parameters_from_safetensors(self.model_root)
        if counted:
            return counted
        self.load()
        return sum(p.numel() for p in self._model.parameters())

    def ask_detailed(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """`ask`, plus the shape of what was generated. Added, never substituted.

        0080 calls `ask` and is unaffected. 0081 needs to know whether generation stopped
        because it ran out of budget: a structured extraction that silently hits its token
        limit produces an unclosed JSON object, which is indistinguishable from a bad model
        unless the caller is told. That is what hid the first proposer failure.
        """
        reply, stats = self._generate(prompt)
        return reply, stats

    def ask(self, prompt: str) -> str:
        return self._generate(prompt)[0]

    def _generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        import torch

        self.load()
        text = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        encoded = self._tokenizer([text], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        produced = generated[0][encoded["input_ids"].shape[1] :]
        reply = self._tokenizer.decode(produced, skip_special_tokens=True)
        return str(reply), {
            "generated_tokens": int(produced.shape[0]),
            "hit_token_limit": int(produced.shape[0]) >= int(self.max_new_tokens),
            "max_new_tokens": int(self.max_new_tokens),
        }

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Everything the runtime needs, resolved from the profile."""

    model_root: Path
    cache_root: Path
    qwen_model_root: Path
    device: str = "cuda"
    top_k_per_retriever: int = TOP_K_PER_RETRIEVER
    max_candidate_context: int = MAX_CANDIDATE_CONTEXT
    embed_batch_size: int = 256
    max_new_tokens: int = 256
    joint_safe_span: bool = False
    apply_spans: bool = False
    tier_policy: TierPolicy = field(default_factory=TierPolicy)


# ----------------------------------------------------------------------- index build


def cache_path(cache_root: Path, key: str) -> Path:
    return cache_root / f"{key}_kb"


def build_index_for(
    spec: RetrieverSpec,
    documents: list[KbDocument],
    config: RuntimeConfig,
    *,
    encoder_factory: Any = build_encoder,
) -> CachedIndex:
    """Embed the governed corpus with one retriever, then release it."""
    if not is_hub_revision(spec.revision):
        raise RevisionNotPinned(
            f"{spec.key}: refusing to build a cache against revision "
            f"{spec.revision or 'empty'}. An index whose model identity is unknown "
            "cannot be validated on reload; run download-models first so the pinned "
            "revision is recorded."
        )
    encoder = encoder_factory(spec, config.model_root, config.device)
    started = time.time()
    log(f"{spec.key}: embedding {len(documents):,} governed documents")
    vectors = encoder.encode(
        [d.text for d in documents], batch_size=config.embed_batch_size
    )
    revision = spec.revision
    dim = int(vectors.shape[1]) if len(vectors) else 0
    encoder.unload()
    log(f"{spec.key}: done in {time.time() - started:.1f}s, dim {dim}")

    index = CachedIndex(
        spec=spec, documents=documents, vectors=vectors,
        manifest=CacheManifest(
            model_repo=spec.repo_id, revision=revision, pooling=spec.pooling,
            normalized=True,
            document_checksum=checksum([d.text for d in documents]),
            row_order_checksum=checksum([d.concept_id for d in documents]),
            rows=len(documents), dim=dim, dtype=str(vectors.dtype),
        ),
    )
    index.validate()  # alignment proved before anything is written
    write_cache(cache_path(config.cache_root, spec.key), index)
    return index


def load_index(
    spec: RetrieverSpec, documents: list[KbDocument], config: RuntimeConfig
) -> CachedIndex:
    """Restore a cache and refuse it unless it describes these exact documents."""
    import torch

    base = cache_path(config.cache_root, spec.key)
    manifest_path = base.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{spec.key}: no cache manifest at {manifest_path}")
    index = CachedIndex(
        spec=spec, documents=documents,
        vectors=torch.load(base.with_suffix(".pt"), map_location="cpu"),
        manifest=CacheManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        ),
    )
    index.validate()
    return index


# --------------------------------------------------------------------- retrieval


@dataclass
class RetrievalEvidence:
    """Per-span retrieval results, keyed by span text. Collected before Qwen loads."""

    per_span: dict[str, dict[str, list[tuple[str, int]]]] = field(default_factory=dict)

    def record(self, span: str, key: str, hits: list[tuple[str, int]]) -> None:
        self.per_span.setdefault(span, {})[key] = hits

    def for_span(self, span: str) -> dict[str, list[tuple[str, int]]]:
        return self.per_span.get(span, {})


def precompute_retrieval(
    spans: Sequence[tuple[str, str, str]],
    specs: Sequence[RetrieverSpec],
    documents: list[KbDocument],
    config: RuntimeConfig,
    *,
    encoder_factory: Any = build_encoder,
) -> RetrievalEvidence:
    """Embed every mention once per retriever, then unload before disambiguation.

    ``spans`` is ``(text, ontology, retriever_role)``. Batching all mentions per model
    keeps the number of load/unload cycles at one per retriever for the entire run, while
    preserving the registry contract that a retriever only contributes to declared roles.
    """
    evidence = RetrievalEvidence()
    unique = sorted({(text, ontology, role) for text, ontology, role in spans})
    if not unique:
        return evidence

    for spec in specs:
        relevant = [(text, ontology, role) for text, ontology, role in unique if role in spec.role]
        if not relevant:
            continue
        texts = [text for text, _, _ in relevant]
        index = load_index(spec, documents, config)
        encoder = encoder_factory(spec, config.model_root, config.device)
        vectors = encoder.encode(texts, batch_size=config.embed_batch_size)
        encoder.unload()
        for position, (text, ontology, _role) in enumerate(relevant):
            hits = index.search(
                vectors[position], ontology=ontology, top_k=config.top_k_per_retriever
            )
            evidence.record(f"{ontology}\x00{text}", spec.key, hits)
        del index
        log(f"{spec.key}: retrieved for {len(relevant):,} unique spans, model unloaded")
    return evidence


def lexical_hits(
    index: LocalIndex | None, mention: str, limit: int
) -> list[tuple[str, int]]:
    if index is None:
        return []
    return [
        (hit.concept_id, rank)
        for rank, hit in enumerate(search_index(index, mention, limit=limit), start=1)
    ]


def exact_alias_ids(index: LocalIndex | None, mention: str) -> frozenset[str]:
    """Concepts whose governed label or alias equals the mention after normalization."""
    if index is None:
        return frozenset()
    key = normalize_text(mention)
    out = set(index.exact.get(key, []))
    for posting in ("exact_canonical", "exact_alias", "exact_synonym"):
        out.update(getattr(index, posting, {}).get(key, []))
    return frozenset(out)


# ------------------------------------------------------------------ disambiguation


def candidate_lines(
    concept_id: str,
    ontology: str,
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
    structured: dict[str, StructuredDrug],
    mention: str,
) -> tuple[list[str], IcdFacts | RxNormFacts | None]:
    if ontology == ONTOLOGY_ICD:
        facts = icd_facts(icd_index, concept_id) if icd_index else None
        return (facts.as_prompt_lines() if facts else [concept_id]), facts
    drug = rxnorm_facts(rxnorm_index, concept_id, mention, structured) if rxnorm_index else None
    return (drug.as_prompt_lines() if drug else [concept_id]), drug


def _facts_lookup(
    facts_by_id: dict[str, IcdFacts | RxNormFacts | None],
) -> Any:
    """Bind the mapping explicitly; a closure over the loop variable would rebind."""

    def lookup(concept_id: str) -> IcdFacts | RxNormFacts | None:
        return facts_by_id.get(concept_id)

    return lookup


def local_context(source: str, start: int, end: int, radius: int = 120) -> str:
    left = max(0, start - radius)
    right = min(len(source), end + radius)
    return " ".join(source[left:right].split())


# ------------------------------------------------------------------ budget


def certify_budget(
    specs: Sequence[RetrieverSpec],
    config: RuntimeConfig,
    *,
    e3_checkpoint: Path | None,
    qwen: QwenDisambiguator | None,
    licence_overrides: Sequence[str] = (),
    encoder_factory: Any = build_encoder,
) -> ModelManifest:
    """Measure every ENABLED deployed model and enforce the 9B sum.

    Sequential loading reduces peak VRAM; it does not reduce the number of models the
    submission deploys, so every enabled retriever is counted here whether or not it is
    resident at any given moment.
    """
    manifest = ModelManifest(licence_overrides_accepted=tuple(licence_overrides))
    resolved: list[RetrieverSpec] = []
    for spec in specs:
        if not spec.enabled:
            resolved.append(spec)
            continue
        if not is_hub_revision(spec.revision):
            raise RevisionNotPinned(
                f"{spec.key}: refusing to certify a retriever without an immutable hub "
                f"commit ({spec.revision or 'empty'}). Run download-models first."
            )
        encoder = encoder_factory(spec, config.model_root, config.device)
        count = encoder.parameter_count()
        # The acquisition pin is authoritative: it is what the weights were downloaded at.
        revision = spec.revision
        dim = getattr(encoder, "embedding_dim", 0)
        encoder.unload()
        import dataclasses

        resolved.append(
            dataclasses.replace(
                spec, revision=revision, parameter_count=count, embedding_dim=dim
            )
        )
        manifest.deployed.append(
            DeployedModel(spec.repo_id, count, revision, spec.licence, "retriever")
        )
    manifest.retrievers = resolved

    if e3_checkpoint is not None:
        import torch

        obj = torch.load(e3_checkpoint, map_location="cpu", weights_only=False)
        raw = obj.get("model_state_dict", obj.get("state_dict", obj)) if isinstance(
            obj, dict
        ) else {}
        state: dict[str, Any] = raw if isinstance(raw, dict) else {}
        manifest.deployed.append(
            DeployedModel(
                "ViHealthBERT E3",
                sum(v.numel() for v in state.values() if hasattr(v, "numel")),
                # A local checkpoint has no hub commit. Its file digest is the only
                # immutable identity it has, and it is labelled so it cannot be mistaken
                # for one.
                revision=file_digest(e3_checkpoint),
                role="mention expert",
            )
        )
    if qwen is not None:
        manifest.deployed.append(
            DeployedModel(
                "Qwen/Qwen3-8B",
                qwen.parameter_count(),
                # The pin the reasoner budget module already treats as authoritative for
                # this repo, and the revision the runbook downloads.
                revision=QWEN3_8B_REVISION,
                role="disambiguator",
            )
        )

    manifest.assert_licences_cleared()
    manifest.resolved_revisions = {
        spec.key: spec.revision for spec in manifest.retrievers if spec.enabled
    }
    manifest.assert_revisions_pinned()
    total = manifest.assert_within_cap()
    log("DEPLOYED PARAMETER TABLE")
    for model in manifest.deployed:
        log(f"  {model.name:48s} {model.parameter_count:>15,}  {model.revision or '-'}")
    log(f"  {'TOTAL':48s} {total:>15,}  (cap 9,000,000,000)")
    return manifest


# ------------------------------------------------------------------ the run


@dataclass
class RunOutcome:
    records: list[MentionResult] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def run_documents(
    notes: dict[str, str],
    seeds: dict[str, list[dict[str, Any]]],
    specs: Sequence[RetrieverSpec],
    documents: list[KbDocument],
    config: RuntimeConfig,
    *,
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
    structured: dict[str, StructuredDrug],
    disambiguator: Disambiguator,
    encoder_factory: Any = build_encoder,
) -> RunOutcome:
    """One neutral pass: spans -> retrieval -> ontology -> Qwen -> tiers."""
    from .pipeline import resolve_tiers

    plans: list[tuple[str, dict[str, Any], Any]] = []
    span_keys: list[tuple[str, str, str]] = []
    for document, seed in seeds.items():
        source = notes[document]
        for entity in seed:
            if not linkable(entity.get("type", "")):
                continue
            ontology = ONTOLOGY_FOR_TYPE[entity["type"]]
            role = RETRIEVER_ROLE_FOR_TYPE[entity["type"]]
            for option in alternatives_for(
                source, entity, joint_safe_span=config.joint_safe_span
            ):
                plans.append((document, entity, option))
                span_keys.append((option.text, ontology, role))

    log(f"linkable mentions {len(plans):,} across {len(seeds)} documents")
    evidence = precompute_retrieval(
        span_keys, [s for s in specs if s.enabled], documents, config,
        encoder_factory=encoder_factory,
    )

    governed_icd = frozenset(icd_index.records) if icd_index else frozenset()
    governed_rx = frozenset(rxnorm_index.records) if rxnorm_index else frozenset()

    outcome = RunOutcome()
    counters: dict[str, int] = {
        "null": 0, "select": 0, "parse_fallbacks": 0, "invalid_ids": 0,
        "invalid_indices": 0, "selection_conflicts": 0,
        "multi_code": 0, "conflicts": 0, "spans_expanded": 0,
    }
    for document, entity, option in plans:
        ontology = ONTOLOGY_FOR_TYPE[entity["type"]]
        governed = governed_icd if ontology == ONTOLOGY_ICD else governed_rx
        base_index = icd_index if ontology == ONTOLOGY_ICD else rxnorm_index
        per_retriever = evidence.for_span(f"{ontology}\x00{option.text}")
        fused = fuse(
            lexical_hits(base_index, option.text, config.top_k_per_retriever * 2),
            {k: v for k, v in per_retriever.items()},
            exact_alias_ids=exact_alias_ids(base_index, option.text),
            governed_ids=governed,
            limit=config.max_candidate_context,
        )
        facts_by_id: dict[str, IcdFacts | RxNormFacts | None] = {}
        prompt_candidates: list[tuple[str, list[str]]] = []
        for candidate in fused:
            lines, facts = candidate_lines(
                candidate.concept_id, ontology, icd_index, rxnorm_index,
                structured, option.text,
            )
            facts_by_id[candidate.concept_id] = facts
            prompt_candidates.append((candidate.concept_id, lines))
            if isinstance(facts, RxNormFacts) and facts.has_conflict:
                counters["conflicts"] += 1

        decision: Decision = NULL_DECISION
        if prompt_candidates:
            reply = disambiguator.ask(
                build_prompt(
                    option.text, entity["type"],
                    local_context(
                        notes[document], option.start, option.end
                    ),
                    prompt_candidates,
                )
            )
            decision = parse_decision(reply, [c for c, _ in prompt_candidates])
        counters["parse_fallbacks"] += int(decision.parse_failed)
        counters["invalid_ids"] += len(decision.invalid_ids)
        counters["invalid_indices"] += len(decision.invalid_indices)
        counters["selection_conflicts"] += int(decision.conflict)
        counters["select" if decision.selected else "null"] += 1
        counters["multi_code"] += int(len(decision.candidate_ids) > 1)
        if option.provenance != "e3_original":
            counters["spans_expanded"] += 1

        result = MentionResult(
            document=document, text=option.text, entity_type=entity["type"],
            position=(option.start, option.end), span_provenance=option.provenance,
            candidates=fused, decision=decision,
            facts={
                k: (v.as_dict() if v else None)
                for k, v in facts_by_id.items()
                if k in decision.candidate_ids
            },
        )
        resolve_tiers(
            result, _facts_lookup(facts_by_id),
            bool(parse_strengths(option.text)), config.tier_policy,
        )
        outcome.records.append(result)

    tier_counts: dict[str, int] = {}
    for record in outcome.records:
        for tier in record.tiers.values():
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
    outcome.diagnostics = {
        "documents": len(seeds),
        "linkable_mentions": len(plans),
        "diagnosis_mentions": sum(
            1 for _, e, _ in plans if e["type"] == "CHẨN_ĐOÁN"
        ),
        "drug_mentions": sum(1 for _, e, _ in plans if e["type"] == "THUỐC"),
        "avg_candidates_per_mention": (
            round(sum(len(r.candidates) for r in outcome.records) / len(outcome.records), 2)
            if outcome.records else 0.0
        ),
        "lexically_backed": sum(
            1 for r in outcome.records for c in r.candidates if c.lexically_backed
        ),
        "multi_retriever_backed": sum(
            1 for r in outcome.records for c in r.candidates if c.supporting_retrievers >= 2
        ),
        "qwen": dict(counters),
        "tiers": tier_counts,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return outcome


def write_variants(
    run_dir: Path,
    seeds: dict[str, list[dict[str, Any]]],
    outcome: RunOutcome,
    config: RuntimeConfig,
) -> dict[str, int]:
    """Four submission variants from the single neutral record set. Qwen is not re-run."""
    by_document: dict[str, list[MentionResult]] = {}
    for record in outcome.records:
        by_document.setdefault(record.document, []).append(record)

    emitted: dict[str, int] = {}
    for variant in VARIANTS:
        out_dir = run_dir / f"output_{variant}_raw_dotless"
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for document, seed in seeds.items():
            rows = serialize_document(
                seed, by_document.get(document, []), variant,
                apply_spans=config.apply_spans,
            )
            total += sum(
                len(r.get("candidates") or [])
                for r in rows if r["type"] in CANDIDATE_TYPES
            )
            (out_dir / f"{document}.json").write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8"
            )
        emitted[variant] = total
        log(f"variant {variant:9s} codes emitted {total}")
    (run_dir / "mention-records.jsonl").write_text(
        "".join(
            json.dumps(r.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for r in outcome.records
        ),
        encoding="utf-8",
    )
    return emitted


def governed_kb(
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
    structured: dict[str, StructuredDrug] | None,
) -> list[KbDocument]:
    return build_kb_documents(icd_index, rxnorm_index, structured)


__all__ = [
    "Disambiguator", "QwenDisambiguator", "RetrievalEvidence", "RunOutcome",
    "RuntimeConfig", "build_index_for", "cache_path", "candidate_lines",
    "certify_budget", "exact_alias_ids", "governed_kb", "lexical_hits", "load_index",
    "local_context", "parameters_from_safetensors", "precompute_retrieval",
    "run_documents", "write_variants",
    "ONTOLOGY_ICD", "ONTOLOGY_RXNORM",
]
