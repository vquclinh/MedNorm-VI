"""Qwen3 cross-encoder reranking (Audit 0072 §2).

Audit 0071 shipped an evaluator that measured dense retrieval and nothing else, and the Colab
handoff would have selected an architecture on that number. Dense recall is not a system
result: Audit 0070 already showed that widening the candidate pool without matching
discrimination *lowers* J_candidates, so the discriminator is the part that has to be
measured.

This module implements the reranker for real. Qwen3-Reranker is not an embedding model and
its score is not a cosine similarity - it is a causal LM asked a yes/no question, and the
score is the probability it assigns to `yes`:

    <|im_start|>system
    Judge whether the Document meets the requirements based on the Query and the Instruct
    provided. Note that the answer can only be "yes" or "no".<|im_end|>
    <|im_start|>user
    <Instruct>: …
    <Query>: …
    <Document>: …<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

Taking the logits at the final position, restricted to the `yes` and `no` token ids, and
softmaxing them yields a calibrated-ish probability in [0, 1]. Substituting embedding
similarity here would silently reproduce the Audit-0071 defect, so `RerankerBackend` exists to
make the contract explicit and `MockReranker` lets every test run without weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

RERANKER_PROTOCOL_VERSION = "qwen3-reranker-yes-no-v1"

#: Official Qwen3-Reranker framing. Reproduced verbatim: the model was trained with it, and
#: paraphrasing it changes the score distribution.
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)

#: Task instruction. Mirrors the retrieval instruction in `representation.py` and is likewise
#: negative about inference: the reranker judges compatibility, it does not add specificity.
TASK_INSTRUCTION = (
    "Given a Vietnamese clinical mention with its context, judge whether the candidate "
    "ICD-10 or RxNorm concept is the correct normalization. Answer no if the candidate "
    "asserts a disease subtype, organism, drug strength, dose form, or brand that the "
    "clinical text does not state."
)

#: Hard ceiling on one pair. Long enough for a mention with context plus a bounded candidate
#: document, short enough that batches stay predictable on a 40 GB card.
MAX_PAIR_TOKENS = 1024


def format_pair(query: str, document: str, *, instruction: str = TASK_INSTRUCTION) -> str:
    """The user-turn body for one (query, candidate) pair."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """One candidate with everything needed to audit why it moved."""

    concept_id: str
    rank_before: int
    rank_after: int
    reranker_score: float
    normalized_score: float
    sources: tuple[str, ...] = field(default_factory=tuple)
    dense_score: float | None = None
    lexical_score: float | None = None
    evidence_tier: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "reranker_score": round(self.reranker_score, 6),
            "normalized_score": round(self.normalized_score, 6),
            "sources": list(self.sources),
            "dense_score": self.dense_score,
            "lexical_score": self.lexical_score,
            "evidence_tier": self.evidence_tier,
        }


class RerankerBackend(Protocol):
    """Scores (query, document) pairs. Higher is more compatible."""

    def score(self, query: str, documents: list[str]) -> list[float]: ...

    def unload(self) -> None: ...


class MockReranker:
    """Deterministic, weight-free stand-in used by every unit test.

    Scores by normalized token overlap, which is *not* a claim about quality - it exists so
    the plumbing (ordering, provenance, gate interaction) is testable without a GPU. It is
    never used in an evaluation path that reports a system metric.
    """

    protocol_version = RERANKER_PROTOCOL_VERSION
    model_id = "mock-reranker"
    revision = "mock"

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_tokens = {t for t in query.casefold().split() if len(t) > 1}
        out = []
        for document in documents:
            tokens = {t for t in document.casefold().split() if len(t) > 1}
            overlap = len(query_tokens & tokens)
            out.append(overlap / len(query_tokens) if query_tokens else 0.0)
        return out

    def unload(self) -> None:  # pragma: no cover - nothing to release
        return None


class Qwen3Reranker:
    """The real backend. Weights load on first `score`, never at import."""

    protocol_version = RERANKER_PROTOCOL_VERSION

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str = "cuda",
        batch_size: int = 8,
        max_tokens: int = MAX_PAIR_TOKENS,
        local_path: str | None = None,
    ) -> None:
        if len(revision) < 40 and revision != "mock":
            raise ValueError(
                f"revision must be a full 40-character immutable commit sha, got {revision!r}. "
                "Abbreviated revisions and branch names are not reproducible (Audit 0072 §9)."
            )
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.local_path = local_path
        self._tokenizer: Any = None
        self._model: Any = None
        self._yes_id: int = -1
        self._no_id: int = -1
        self._prefix: str = ""
        self._suffix: str = ""

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = self.local_path or self.model_id
        kwargs: dict[str, Any] = {} if self.local_path else {"revision": self.revision}
        self._tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left", **kwargs)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        try:
            self._model = (
                AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, **kwargs)
                .to(self.device)
                .eval()
            )
        except RuntimeError as error:  # pragma: no cover - depends on host memory
            if "out of memory" in str(error).lower():
                raise RuntimeError(
                    f"CUDA OOM loading {self.model_id}. Free VRAM first: unload the embedding "
                    f"model and call torch.cuda.empty_cache(), or lower --rerank-batch-size "
                    f"(currently {self.batch_size}). Audit 0072 §10 requires the embedding and "
                    "reranker stages to be resident one at a time."
                ) from error
            raise
        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        self._prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        self._suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def score(self, query: str, documents: list[str]) -> list[float]:
        """P(yes) per document, in the order given. Deterministic: no sampling anywhere."""
        if not documents:
            return []
        self._load()
        import torch

        texts = [self._prefix + format_pair(query, d) + self._suffix for d in documents]
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                ).to(self.device)
                try:
                    logits = self._model(**encoded).logits[:, -1, :]
                except RuntimeError as error:  # pragma: no cover - depends on host memory
                    if "out of memory" in str(error).lower():
                        raise RuntimeError(
                            f"CUDA OOM scoring {len(chunk)} pairs with {self.model_id}. "
                            f"Lower --rerank-batch-size below {self.batch_size}."
                        ) from error
                    raise
                pair = torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=1)
                probabilities = torch.nn.functional.softmax(pair.float(), dim=1)[:, 1]
                scores.extend(probabilities.cpu().tolist())
        return scores

    def unload(self) -> None:
        """Release the model so the next system can fit (Audit 0072 §10)."""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch always present on the GPU host
            pass


def rerank(
    backend: RerankerBackend,
    query: str,
    candidates: list[dict[str, Any]],
) -> list[RerankedCandidate]:
    """Score and reorder ``candidates``, preserving the pre-rerank rank of each.

    ``candidates`` carry ``concept_id`` and ``document`` plus optional provenance. Ordering is
    total - ties fall back to the incoming rank and then the concept id - so a repeat run
    cannot reorder.
    """
    if not candidates:
        return []
    scores = backend.score(query, [str(c["document"]) for c in candidates])
    highest = max(scores) if scores else 0.0
    lowest = min(scores) if scores else 0.0
    span = highest - lowest

    scored = [
        RerankedCandidate(
            concept_id=str(candidate["concept_id"]),
            rank_before=index + 1,
            rank_after=0,
            reranker_score=score,
            normalized_score=(score - lowest) / span if span > 0 else 1.0,
            sources=tuple(candidate.get("sources", ())),
            dense_score=candidate.get("dense_score"),
            lexical_score=candidate.get("lexical_score"),
            evidence_tier=str(candidate.get("evidence_tier", "")),
        )
        for index, (candidate, score) in enumerate(zip(candidates, scores, strict=True))
    ]
    scored.sort(key=lambda c: (-c.reranker_score, c.rank_before, c.concept_id))
    return [
        RerankedCandidate(
            concept_id=c.concept_id,
            rank_before=c.rank_before,
            rank_after=position + 1,
            reranker_score=c.reranker_score,
            normalized_score=c.normalized_score,
            sources=c.sources,
            dense_score=c.dense_score,
            lexical_score=c.lexical_score,
            evidence_tier=c.evidence_tier,
        )
        for position, c in enumerate(scored)
    ]


__all__ = [
    "MAX_PAIR_TOKENS",
    "RERANKER_PROTOCOL_VERSION",
    "SYSTEM_PROMPT",
    "TASK_INSTRUCTION",
    "MockReranker",
    "Qwen3Reranker",
    "RerankedCandidate",
    "RerankerBackend",
    "format_pair",
    "rerank",
]
