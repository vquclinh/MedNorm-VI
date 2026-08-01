"""Production runtime for the frozen S1 zero-shot semantic ICD linker (Audit 0073).

Audit 0072 selected S1 and froze its revisions. This module is the only new thing production
needs: a `LinkerResult`-shaped wrapper around the *already implemented* 0072 components. It
duplicates none of them - `build_pool`, `run_hybrid`, `Qwen3Reranker` and the semantic
representation are imported, not reimplemented.

Scope is **both ontologies**, because that is what Audit 0072 actually measured. Its 2,000-query
benchmark was ~197 ICD10 and ~1,803 RXNORM, so the frozen S1 numbers (R@1 0.5115, R@10 0.7440,
MRR 0.5985) are roughly 90% RxNorm-driven. Wiring the semantic backend to ICD only would
reproduce neither the selected system nor its evidence.

The two ontologies use different pipelines, matching 0072 exactly:

    ICD10   v3 Top-20  +  v4.1 Top-20  +  dense Top-50  ->  rerank  ->  H2
    RXNORM  lexical Top-20             +  dense Top-50  ->  rerank  (NO hierarchy)

H2 is an ICD within-family rule and is never applied to RxNorm: `run_hybrid` gates it on
`ontology == "ICD10"`, so the separation is structural rather than a convention.

E1/E2/E3, L4, assertions, spans, types and the output schema are untouched, and the NULL gate
stays in shadow. The one variable against the scored 11.9188 baseline is the linker backend.

Everything fails closed. A missing model directory, a missing dense index, or a revision that
does not match what Audit 0072 froze stops the run with a diagnostic naming the exact file -
never a silent fallback to lexical, which would make the ablation a lie.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...kb.indexing.retrieval import LocalIndex
from ..models import LinkedCandidate, LinkerResult
from .hybrid import (
    DEFAULT_DENSE_TOPK,
    DEFAULT_V3_TOPK,
    DEFAULT_V41_TOPK,
    NULL_MODE_SHADOW,
    build_pool,
    run_hybrid,
)
from .representation import SemanticQuery
from .reranker import Qwen3Reranker

#: Backend identifiers accepted by a governed profile. Anything else is refused.
BACKEND_LEXICAL_V3 = "lexical_v3"
BACKEND_SEMANTIC_S1 = "semantic_s1_0072"
LINKER_BACKENDS: tuple[str, ...] = (BACKEND_LEXICAL_V3, BACKEND_SEMANTIC_S1)

#: Environment override for the model root, so no user-specific Drive path is ever a
#: repository default. Precedence: CLI > config > environment > unset (which fails closed).
MODEL_ROOT_ENV = "MEDNORM_SEMANTIC_MODEL_ROOT"

#: Frozen by Audit 0072. Full 40-character immutable shas; abbreviated or moving refs are
#: refused by `Qwen3Reranker` at construction and by `verify()` here.
S1_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
S1_EMBEDDING_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
S1_EMBEDDING_DIRNAME = "emb-4b"
S1_RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"
S1_RERANKER_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
S1_RERANKER_DIRNAME = "rr-4b"

#: The document corpus the frozen S1 dense index was built over. A mismatch means the cache
#: belongs to a different corpus and every retrieved row would be the wrong concept.
EXPECTED_DOCUMENT_COUNT = 227_257
EXPECTED_ICD_ROWS = 15_308
EXPECTED_RXNORM_ROWS = 211_949

#: How a dense cache's rows map onto documents. This is a CONTRACT, not a detail.
#:
#: `documents.jsonl` is written ICD-first (rows 0-15,307) then RxNorm; the Audit-0072
#: orchestrator encoded `sorted(documents)`, which puts every numeric RxCUI before every
#: letter-prefixed ICD code and therefore yields RxNorm-first. Reading the frozen cache with
#: file-order positions - which this runtime did until Audit 0074 - indexes ICD mentions
#: against RxNorm vectors and vice versa. The candidate ids still come back looking valid,
#: so the failure is silent.
#: CORRECT and the only default. The cache's row i is `sorted(concept_id)[i]`.
ROW_ORDER_SORTED_CONCEPT_ID = "sorted_concept_id"

#: A coherent contract for a cache genuinely BUILT in `documents.jsonl` order. Nothing ships
#: such a cache today; it exists so a future builder can declare its order honestly.
ROW_ORDER_DOCUMENTS_FILE = "documents_file_order"

#: HISTORICAL REPRODUCTION ONLY - reproduces the Audit-0073 defect on purpose.
#:
#: The scored 12.4757 submission read the frozen `sorted(concept_id)` cache using
#: `documents.jsonl` file positions. That is not an ordering choice, it is a *mismatch*: ICD
#: mentions were scored against RxNorm vectors and vice versa, while the returned ids still
#: looked valid. This mode exists so that run can be re-executed exactly, and for no other
#: reason. It is never a default, never inferred, and selecting it logs a warning.
#:
#: It shares a position mapping with `ROW_ORDER_DOCUMENTS_FILE` but means something different:
#: that constant asserts "the cache really is in file order", this one asserts "the cache is in
#: sorted order and we are deliberately misreading it".
ROW_ORDER_LEGACY_0073_MISALIGNED = "legacy_0073_misaligned_file_positions"

ROW_ORDERS: tuple[str, ...] = (
    ROW_ORDER_SORTED_CONCEPT_ID,
    ROW_ORDER_DOCUMENTS_FILE,
    ROW_ORDER_LEGACY_0073_MISALIGNED,
)

#: Orders that must never be chosen by accident.
HISTORICAL_ROW_ORDERS: frozenset[str] = frozenset({ROW_ORDER_LEGACY_0073_MISALIGNED})

#: The frozen Audit-0072 cache carries no manifest, so its order is recorded here.
FROZEN_0072_ROW_ORDER = ROW_ORDER_SORTED_CONCEPT_ID

#: Measured, not estimated. E3 from its state dict, the Qwen models from the HF API.
E3_PARAMETERS = 135_002_117
S1_EMBEDDING_PARAMETERS = 4_021_774_336
S1_RERANKER_PARAMETERS = 4_021_784_576
PARAMETER_CAP = 9_000_000_000


def s1_deployed_parameters() -> int:
    return E3_PARAMETERS + S1_EMBEDDING_PARAMETERS + S1_RERANKER_PARAMETERS


class SemanticLinkerNotReady(RuntimeError):
    """Raised when the semantic backend cannot run. Never downgraded to a warning."""


@dataclass(frozen=True, slots=True)
class SemanticLinkerSettings:
    """Everything the S1 backend needs, all overridable, none user-specific by default."""

    model_root: str = ""
    dense_index: str = ""
    documents: str = ""
    device: str = "cuda"
    embed_batch_size: int = 16
    rerank_batch_size: int = 8
    v3_topk: int = DEFAULT_V3_TOPK
    v41_topk: int = DEFAULT_V41_TOPK
    dense_topk: int = DEFAULT_DENSE_TOPK
    candidate_limit: int = 20
    null_mode: str = NULL_MODE_SHADOW
    icd_v41_index: str = ""
    rxnorm_index: str = ""
    rxnorm_topk: int = 20
    #: Row-order contract of `dense_index`. Defaults to the frozen 0072 cache's order.
    row_order: str = FROZEN_0072_ROW_ORDER

    @staticmethod
    def from_mapping(doc: dict[str, Any] | None) -> SemanticLinkerSettings:
        doc = doc or {}
        root = str(doc.get("model_root", "") or os.environ.get(MODEL_ROOT_ENV, ""))
        return SemanticLinkerSettings(
            model_root=root,
            dense_index=str(doc.get("dense_index", "")),
            documents=str(doc.get("documents", "")),
            device=str(doc.get("device", "cuda")),
            embed_batch_size=int(doc.get("embed_batch_size", 16)),
            rerank_batch_size=int(doc.get("rerank_batch_size", 8)),
            v3_topk=int(doc.get("v3_topk", DEFAULT_V3_TOPK)),
            v41_topk=int(doc.get("v41_topk", DEFAULT_V41_TOPK)),
            dense_topk=int(doc.get("dense_topk", DEFAULT_DENSE_TOPK)),
            candidate_limit=int(doc.get("candidate_limit", 20)),
            null_mode=str(doc.get("null_mode", NULL_MODE_SHADOW)),
            icd_v41_index=str(doc.get("icd_v41_index", "")),
            rxnorm_index=str(doc.get("rxnorm_index", "")),
            rxnorm_topk=int(doc.get("rxnorm_topk", 20)),
            row_order=str(doc.get("row_order", FROZEN_0072_ROW_ORDER)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_root": self.model_root,
            "dense_index": self.dense_index,
            "documents": self.documents,
            "device": self.device,
            "embed_batch_size": self.embed_batch_size,
            "rerank_batch_size": self.rerank_batch_size,
            "v3_topk": self.v3_topk,
            "v41_topk": self.v41_topk,
            "dense_topk": self.dense_topk,
            "candidate_limit": self.candidate_limit,
            "null_mode": self.null_mode,
            "icd_v41_index": self.icd_v41_index,
            "rxnorm_index": self.rxnorm_index,
            "rxnorm_topk": self.rxnorm_topk,
            "row_order": self.row_order,
        }


def model_manifest() -> dict[str, Any]:
    """The exact learned components this backend deploys, for the run manifest."""
    return {
        "linker_backend": BACKEND_SEMANTIC_S1,
        "selected_by": "Audit 0072 frozen selection",
        "training_performed": False,
        "null_mode": NULL_MODE_SHADOW,
        "components": [
            {
                "model": S1_EMBEDDING_MODEL,
                "revision": S1_EMBEDDING_REVISION,
                "role": "dense retrieval",
                "parameter_count": S1_EMBEDDING_PARAMETERS,
                "deployed": True,
            },
            {
                "model": S1_RERANKER_MODEL,
                "revision": S1_RERANKER_REVISION,
                "role": "cross-encoder rerank",
                "parameter_count": S1_RERANKER_PARAMETERS,
                "deployed": True,
            },
            {
                "model": "ViHealthBERT E3 span/type head",
                "revision": "f89e80b461e86f9cfc1c84019bd819830c24b6c5",
                "role": "mention expert (unchanged from baseline)",
                "parameter_count": E3_PARAMETERS,
                "deployed": True,
            },
        ],
        "total_deployed_parameters": s1_deployed_parameters(),
        "parameter_cap": PARAMETER_CAP,
        "under_cap": s1_deployed_parameters() < PARAMETER_CAP,
    }


def missing_requirements(settings: SemanticLinkerSettings) -> list[str]:
    """Every reason the S1 backend cannot run, named precisely. Empty means ready."""
    problems: list[str] = []
    if not settings.model_root:
        problems.append(
            f"semantic_linker.model_root is unset; set it in the profile, pass "
            f"--semantic-model-root, or export {MODEL_ROOT_ENV}"
        )
    else:
        root = Path(settings.model_root)
        if not root.is_dir():
            problems.append(f"semantic model root not found: {root}")
        else:
            for dirname, model in (
                (S1_EMBEDDING_DIRNAME, S1_EMBEDDING_MODEL),
                (S1_RERANKER_DIRNAME, S1_RERANKER_MODEL),
            ):
                config_file = root / dirname / "config.json"
                if not config_file.is_file():
                    problems.append(f"missing {model} weights: {config_file} not found")
    if not settings.documents:
        problems.append("semantic_linker.documents is unset (semantic document JSONL)")
    elif not Path(settings.documents).is_file():
        problems.append(f"semantic documents not found: {settings.documents}")
    if not settings.dense_index:
        problems.append("semantic_linker.dense_index is unset (precomputed document vectors)")
    elif not Path(settings.dense_index).is_file():
        problems.append(
            f"dense index not found: {settings.dense_index}. Build it on a GPU host first; "
            "inference never encodes 227k documents on the fly."
        )
    if not settings.icd_v41_index:
        problems.append("semantic_linker.icd_v41_index is unset (ICD coverage source)")
    elif not Path(settings.icd_v41_index).is_file():
        problems.append(f"ICD competition-v4.1 index not found: {settings.icd_v41_index}")
    if not settings.rxnorm_index:
        problems.append("semantic_linker.rxnorm_index is unset (governed RxNorm KB)")
    elif not Path(settings.rxnorm_index).is_file():
        problems.append(f"RxNorm competition index not found: {settings.rxnorm_index}")
    if settings.row_order not in ROW_ORDERS:
        problems.append(
            f"semantic_linker.row_order must be one of {list(ROW_ORDERS)}, got "
            f"{settings.row_order!r}"
        )
    if settings.null_mode != NULL_MODE_SHADOW:
        problems.append(
            f"null_mode must be {NULL_MODE_SHADOW!r} in this milestone, got "
            f"{settings.null_mode!r} (Audit 0073 forbids activating the NULL gate)"
        )
    return problems


@dataclass
class SemanticLinkerRuntime:
    """Frozen S1 for BOTH ontologies. ICD gets H2; RxNorm never does."""

    settings: SemanticLinkerSettings
    icd_v3_index: LocalIndex
    icd_v41_index: LocalIndex
    rxnorm_index: LocalIndex
    _documents: dict[str, str] = field(default_factory=dict)
    _rows_by_ontology: dict[str, list[int]] = field(default_factory=dict)
    _ids_by_ontology: dict[str, list[str]] = field(default_factory=dict)
    _vectors: Any = None
    _backend: Any = None
    _encoder: Any = None
    _shadow: list[dict[str, Any]] = field(default_factory=list)

    def ensure_ready(self) -> None:
        problems = missing_requirements(self.settings)
        if problems:
            raise SemanticLinkerNotReady(
                "semantic_s1_0072 backend is not ready:\n  - " + "\n  - ".join(problems)
            )

    def load(self) -> None:
        """Load documents, the frozen dense index and the reranker. Idempotent."""
        if self._backend is not None:
            return
        self.ensure_ready()
        import torch

        rows = [
            json.loads(line)
            for line in Path(self.settings.documents).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != EXPECTED_DOCUMENT_COUNT:
            raise SemanticLinkerNotReady(
                f"documents.jsonl has {len(rows):,} rows, expected "
                f"{EXPECTED_DOCUMENT_COUNT:,}. The dense index was built over the frozen "
                "corpus; a different corpus would map every vector to the wrong concept."
            )
        vectors = torch.load(self.settings.dense_index, map_location="cpu")
        if int(vectors.shape[0]) != len(rows):
            raise SemanticLinkerNotReady(
                f"dense index has {int(vectors.shape[0]):,} vectors but documents.jsonl has "
                f"{len(rows):,} rows; the cached index does not belong to this corpus"
            )

        # Row order IS the contract between documents.jsonl and the cached vectors, and the
        # two are NOT the same order. Positions must be derived the way the cache was built.
        self._documents = {r["concept_id"]: r["text"] for r in rows}
        ontology_of = {r["concept_id"]: r.get("ontology", "") for r in rows}
        if self.settings.row_order == ROW_ORDER_SORTED_CONCEPT_ID:
            ordered_ids = sorted(ontology_of)
        elif self.settings.row_order == ROW_ORDER_LEGACY_0073_MISALIGNED:
            # Deliberate reproduction of the Audit-0073 defect. Loud, because a silent
            # misalignment is exactly what made the original failure hard to see.
            print(
                "[semantic] WARNING: row_order="
                f"{ROW_ORDER_LEGACY_0073_MISALIGNED!r} reproduces the Audit-0073 dense "
                "misalignment on purpose (ICD mentions score against RxNorm vectors). "
                "Use only to re-execute that historical run; never for new experiments.",
                file=sys.stderr,
                flush=True,
            )
            ordered_ids = [r["concept_id"] for r in rows]
        else:
            ordered_ids = [r["concept_id"] for r in rows]

        self._rows_by_ontology = {}
        self._ids_by_ontology = {}
        for position, concept_id in enumerate(ordered_ids):
            ontology = ontology_of[concept_id]
            self._rows_by_ontology.setdefault(ontology, []).append(position)
            self._ids_by_ontology.setdefault(ontology, []).append(concept_id)
        icd_rows = len(self._rows_by_ontology.get("ICD10", []))
        rx_rows = len(self._rows_by_ontology.get("RXNORM", []))
        if icd_rows != EXPECTED_ICD_ROWS or rx_rows != EXPECTED_RXNORM_ROWS:
            raise SemanticLinkerNotReady(
                f"corpus has {icd_rows:,} ICD and {rx_rows:,} RxNorm rows, expected "
                f"{EXPECTED_ICD_ROWS:,} and {EXPECTED_RXNORM_ROWS:,}"
            )
        self._vectors = vectors

        root = Path(self.settings.model_root)
        self._backend = Qwen3Reranker(
            S1_RERANKER_MODEL,
            S1_RERANKER_REVISION,
            device=self.settings.device,
            batch_size=self.settings.rerank_batch_size,
            local_path=str(root / S1_RERANKER_DIRNAME),
        )

    def _encode(self, text: str) -> Any:
        import torch
        from transformers import AutoModel, AutoTokenizer

        root = Path(self.settings.model_root) / S1_EMBEDDING_DIRNAME
        if self._encoder is None:
            tokenizer = AutoTokenizer.from_pretrained(str(root))
            dtype = torch.bfloat16 if self.settings.device.startswith("cuda") else torch.float32
            model = (
                AutoModel.from_pretrained(str(root), torch_dtype=dtype)
                .to(self.settings.device)
                .eval()
            )
            self._encoder = (tokenizer, model)
        tokenizer, model = self._encoder
        with torch.no_grad():
            enc = tokenizer(
                [text], padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.settings.device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, dim=-1).float().cpu()[0]

    def _dense_hits(self, mention: str, ontology: str) -> list[tuple[str, float]]:
        """Top-K within one ontology's slice of the frozen index."""
        import torch

        positions = self._rows_by_ontology.get(ontology, [])
        if not positions:
            return []
        query = SemanticQuery(mention, "", "", "", ontology).render()
        vector = self._encode(query)
        subset = self._vectors[positions]
        scores = vector @ subset.T
        top = torch.topk(scores, min(self.settings.dense_topk, len(positions)))
        ids = self._ids_by_ontology[ontology]
        return [
            (ids[i], float(s))
            for i, s in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        ]

    def _run(self, hypothesis: Any, ontology: str) -> LinkerResult:
        self.load()
        mention = hypothesis.text
        if ontology == "ICD10":
            pool = build_pool(
                mention,
                v3_index=self.icd_v3_index,
                v41_index=self.icd_v41_index,
                dense_hits=self._dense_hits(mention, ontology),
                governed_index=self.icd_v41_index,
                v3_topk=self.settings.v3_topk,
                v41_topk=self.settings.v41_topk,
                dense_topk=self.settings.dense_topk,
            )
            governed = self.icd_v41_index
            entity_type = "CHẨN_ĐOÁN"
        else:
            # RxNorm: one lexical source plus dense. No v4.1, no hierarchy - there is no
            # RxNorm equivalent of the ICD family arbitration and inventing one here would
            # be new work this milestone forbids.
            pool = build_pool(
                mention,
                v3_index=self.rxnorm_index,
                v41_index=None,
                dense_hits=self._dense_hits(mention, ontology),
                governed_index=self.rxnorm_index,
                v3_topk=self.settings.rxnorm_topk,
                dense_topk=self.settings.dense_topk,
            )
            governed = self.rxnorm_index
            entity_type = "THUỐC"

        query_text = SemanticQuery(mention, entity_type, "", "", ontology).render(
            with_instruction=False
        )
        result = run_hybrid(
            mention,
            query_text,
            pool,
            {c.concept_id: self._documents.get(c.concept_id, c.concept_id) for c in pool},
            self._backend,
            ontology=ontology,
            icd_index=self.icd_v41_index if ontology == "ICD10" else None,
            null_mode=self.settings.null_mode,
            limit=self.settings.candidate_limit,
        )
        self._shadow.append(
            {
                "mention_id": hypothesis.hypothesis_id,
                "ontology": ontology,
                "null_decision": result.null_decision,
                "pool_size": result.pool_size,
                "hierarchy_applied": result.hierarchy_applied,
            }
        )
        by_id = {row["concept_id"]: row for row in result.reranked}
        return LinkerResult(
            mention_id=hypothesis.hypothesis_id,
            candidates=tuple(
                LinkedCandidate(
                    code=code,
                    score=float(by_id.get(code, {}).get("reranker_score", 0.0)),
                    channels=tuple(by_id.get(code, {}).get("sources", ())),
                    snapshot_id=governed.source_snapshot_id,
                    evidence=(f"backend:{BACKEND_SEMANTIC_S1}", f"ontology:{ontology}"),
                )
                for code in result.codes
            ),
            warnings=(),
        )

    def link_icd10(self, hypothesis: Any) -> LinkerResult:
        return self._run(hypothesis, "ICD10")

    def link_rxnorm(self, hypothesis: Any) -> LinkerResult:
        return self._run(hypothesis, "RXNORM")

    @property
    def shadow_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._shadow)

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
        self._encoder = None
        self._backend = None


__all__ = [
    "BACKEND_LEXICAL_V3",
    "FROZEN_0072_ROW_ORDER",
    "HISTORICAL_ROW_ORDERS",
    "ROW_ORDERS",
    "ROW_ORDER_DOCUMENTS_FILE",
    "ROW_ORDER_LEGACY_0073_MISALIGNED",
    "ROW_ORDER_SORTED_CONCEPT_ID",
    "BACKEND_SEMANTIC_S1",
    "E3_PARAMETERS",
    "LINKER_BACKENDS",
    "MODEL_ROOT_ENV",
    "PARAMETER_CAP",
    "S1_EMBEDDING_MODEL",
    "S1_EMBEDDING_PARAMETERS",
    "S1_EMBEDDING_REVISION",
    "S1_RERANKER_MODEL",
    "S1_RERANKER_PARAMETERS",
    "S1_RERANKER_REVISION",
    "EXPECTED_DOCUMENT_COUNT",
    "SemanticLinkerNotReady",
    "SemanticLinkerRuntime",
    "SemanticLinkerSettings",
    "missing_requirements",
    "model_manifest",
    "s1_deployed_parameters",
]
