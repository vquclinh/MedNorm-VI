"""The deployed model stack: registry, pinned acquisition, encoders and the budget."""

from .budget import QWEN3_8B_REVISION
from .registry import (
    DEFAULT_RETRIEVERS,
    PARAMETER_CAP,
    ModelManifest,
    RetrieverSpec,
)

__all__ = [
    "DEFAULT_RETRIEVERS",
    "PARAMETER_CAP",
    "QWEN3_8B_REVISION",
    "ModelManifest",
    "RetrieverSpec",
]
