"""E7 Qwen3-1.7B proposer interface contract.

This milestone intentionally does not download, train, execute, or simulate
Qwen. The class below exists so L3 profiles can declare and audit the proposer
without allowing text-only or free-generating spans into the lattice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from ..lattice.models import EXPERT_QWEN_PROPOSER

QWEN_PROPOSER_CONFIG_VERSION = "qwen3-1.7b-proposer-interface-v1"
QWEN_PROPOSER_STATUS = "UNAVAILABLE_UNTRAINED"


class QwenProposerUnavailableError(RuntimeError):
    """Raised whenever E7 proposal execution is requested in this milestone."""


@dataclass(frozen=True, slots=True)
class QwenProposerConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    model_revision: str = ""
    checkpoint_path: str = ""
    expected_checkpoint_sha256: str = ""
    enabled: bool = False
    config_version: str = QWEN_PROPOSER_CONFIG_VERSION


def config_sha256(config: QwenProposerConfig) -> str:
    payload = {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "config_version": config.config_version,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class QwenProposerStatus:
    expert_id: str
    status: str
    reason: str
    config_sha256: str


class QwenProposerInterface:
    """Strict feature-flag boundary for the future L3/L7 LLM proposer."""

    def __init__(self, config: QwenProposerConfig) -> None:
        self.config = config

    def status(self) -> QwenProposerStatus:
        if not self.config.enabled:
            return QwenProposerStatus(
                EXPERT_QWEN_PROPOSER,
                QWEN_PROPOSER_STATUS,
                "disabled_by_profile",
                config_sha256(self.config),
            )
        if not self.config.checkpoint_path:
            return QwenProposerStatus(
                EXPERT_QWEN_PROPOSER,
                QWEN_PROPOSER_STATUS,
                "enabled_without_local_checkpoint",
                config_sha256(self.config),
            )
        if not Path(self.config.checkpoint_path).exists():
            return QwenProposerStatus(
                EXPERT_QWEN_PROPOSER,
                QWEN_PROPOSER_STATUS,
                "local_checkpoint_missing",
                config_sha256(self.config),
            )
        return QwenProposerStatus(
            EXPERT_QWEN_PROPOSER,
            QWEN_PROPOSER_STATUS,
            "execution_deferred_to_later_confidence_cascade_milestone",
            config_sha256(self.config),
        )

    def propose(self, document_id: str, original_text: str) -> NoReturn:
        raise QwenProposerUnavailableError(
            "E7 Qwen proposer execution is outside this milestone; "
            f"document {document_id!r} length {len(original_text)} was not processed"
        )


__all__ = [
    "EXPERT_QWEN_PROPOSER",
    "QWEN_PROPOSER_CONFIG_VERSION",
    "QWEN_PROPOSER_STATUS",
    "QwenProposerConfig",
    "QwenProposerInterface",
    "QwenProposerStatus",
    "QwenProposerUnavailableError",
    "config_sha256",
]
