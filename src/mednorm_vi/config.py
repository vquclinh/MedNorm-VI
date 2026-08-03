"""Runtime configuration: where the assets are and how much of the device to use.

A leaf module on purpose. It is imported by the knowledge base, the models and the
pipeline alike, so it must depend on none of them - the retired inference package
became unusable precisely because its package init pulled the whole system in.

Model roots, cache roots, device and the retrieval shape. Everything here is resolved from
the single production profile; nothing is discovered at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .kb.semantic_cache import MAX_CANDIDATE_CONTEXT, TOP_K_PER_RETRIEVER


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



__all__ = ["RuntimeConfig"]
