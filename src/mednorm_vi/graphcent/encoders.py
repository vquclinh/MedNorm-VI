"""Real per-model encoder adapters (GraphCENT 0080).

Deliberately three classes rather than one parameterised encoder. The pooling difference
between these checkpoints is not a configuration detail: cross-lingual SapBERT is trained to
be read at the CLS position *before* the pooler, and the BioBERT sentence model is a
SentenceTransformer whose configured pooling is attention-masked mean. Sharing one
implementation is how that distinction quietly disappears, so each model gets its own adapter
whose name states the contract.

Weights load on `.load()`, never at import, and `.unload()` releases them so the index build
can hold one model at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .models import POOLING_CLS, POOLING_MEAN, RetrieverSpec

DEFAULT_MAX_LENGTH = 64
DEFAULT_BATCH_SIZE = 256


@dataclass
class _TransformerEncoder:
    """Shared plumbing. Subclasses supply the pooling, which is the part that differs."""

    #: Declared by every concrete adapter and checked against the registry in
    #: `build_encoder`. An adapter that does not declare its pooling cannot be built.
    POOLING: ClassVar[str] = ""

    spec: RetrieverSpec
    model_root: Path
    device: str = "cuda"
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    _model: Any = None
    _tokenizer: Any = None
    _resolved_revision: str = ""
    _dim: int = 0
    _meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def resolved_revision(self) -> str:
        return self._resolved_revision

    def _source(self) -> str:
        local = self.model_root / self.spec.key
        return str(local) if local.exists() else self.spec.repo_id

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        source = self._source()
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model = AutoModel.from_pretrained(source, torch_dtype=dtype)
        self._model = model.to(self.device).eval()
        config = AutoConfig.from_pretrained(source)
        self._dim = int(config.hidden_size)
        self._resolved_revision = str(getattr(config, "_commit_hash", "") or "")
        self._meta = {"parameters": sum(p.numel() for p in model.parameters())}

    def parameter_count(self) -> int:
        if self._model is None:
            self.load()
        return int(self._meta.get("parameters", 0))

    def unload(self) -> None:
        """Release GPU memory so the next model can be resident alone."""
        self._model = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch present on any run host
            pass

    def _pool(self, hidden_state: Any, attention_mask: Any) -> Any:  # pragma: no cover
        # Every concrete adapter defines pooling; `build_encoder` only ever returns one of
        # them, so this base is unreachable. It fails loudly rather than pooling by guess.
        raise TypeError(f"{type(self).__name__} defines no pooling; use a concrete adapter")

    def encode(self, texts: Sequence[str], *, batch_size: int | None = None) -> Any:
        """L2-normalized embeddings, one row per input, in input order."""
        import torch

        self.load()
        size = batch_size or self.batch_size
        chunks: list[Any] = []
        with torch.inference_mode():
            for start in range(0, len(texts), size):
                batch = list(texts[start : start + size])
                encoded = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                hidden = self._model(**encoded).last_hidden_state
                pooled = self._pool(hidden, encoded["attention_mask"])
                chunks.append(
                    torch.nn.functional.normalize(pooled.float(), p=2, dim=-1).cpu()
                )
        if not chunks:
            return torch.empty(0, self._dim or 1)
        return torch.cat(chunks)


class SapBertEncoder(_TransformerEncoder):
    """Cross-lingual SapBERT: CLS of the last hidden state, before the pooler."""

    POOLING: ClassVar[str] = POOLING_CLS

    def _pool(self, hidden_state: Any, attention_mask: Any) -> Any:
        return hidden_state[:, 0, :]


class ClinLinkerEncoder(_TransformerEncoder):
    """ClinLinker-KB-GP: CLS representation, per the model card."""

    POOLING: ClassVar[str] = POOLING_CLS

    def _pool(self, hidden_state: Any, attention_mask: Any) -> Any:
        return hidden_state[:, 0, :]


class SentenceMeanEncoder(_TransformerEncoder):
    """SentenceTransformer-compatible: attention-masked mean over tokens."""

    POOLING: ClassVar[str] = POOLING_MEAN

    def _pool(self, hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        return (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


#: Key -> adapter. A new retriever must name its adapter here; there is no generic fallback,
#: so an unknown model cannot silently inherit the wrong pooling.
ENCODER_BY_KEY: dict[str, type[_TransformerEncoder]] = {
    "sapbert_xlmr": SapBertEncoder,
    "clinlinker_kb_gp": ClinLinkerEncoder,
    "biobert_mnli": SentenceMeanEncoder,
}


def build_encoder(
    spec: RetrieverSpec, model_root: Path, device: str = "cuda", **kwargs: Any
) -> _TransformerEncoder:
    encoder_type = ENCODER_BY_KEY.get(spec.key)
    if encoder_type is None:
        raise KeyError(
            f"no encoder adapter registered for {spec.key!r}. Pooling is model-specific; "
            "add an explicit adapter rather than reusing another model's."
        )
    encoder = encoder_type(spec=spec, model_root=model_root, device=device, **kwargs)
    declared = getattr(encoder_type, "POOLING", "")
    if declared != spec.pooling:
        raise ValueError(
            f"{spec.key}: registry declares pooling {spec.pooling!r} but the adapter "
            f"implements {declared!r}"
        )
    return encoder


__all__ = [
    "DEFAULT_BATCH_SIZE", "DEFAULT_MAX_LENGTH", "ENCODER_BY_KEY",
    "ClinLinkerEncoder", "SapBertEncoder", "SentenceMeanEncoder", "build_encoder",
]
