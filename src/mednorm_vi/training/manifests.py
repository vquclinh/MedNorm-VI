"""Immutable run-manifest contracts for trainable Phase-2 components."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


class RunManifestError(ValueError):
    """Raised when a training/evaluation manifest omits required custody fields."""


@dataclass(frozen=True, slots=True)
class RunManifest:
    stage: str
    expert: str
    config_sha256: str
    data_sha256: str
    corpus_sha256: str
    model_revision: str
    seed: int
    git_commit: str
    checkpoint_sha256: str
    parameter_count: int
    train_split_id: str
    validation_split_id: str
    internal_test_accessed: bool

    def validate(self) -> None:
        required = {
            "stage": self.stage,
            "expert": self.expert,
            "config_sha256": self.config_sha256,
            "data_sha256": self.data_sha256,
            "corpus_sha256": self.corpus_sha256,
            "model_revision": self.model_revision,
            "git_commit": self.git_commit,
            "checkpoint_sha256": self.checkpoint_sha256,
            "train_split_id": self.train_split_id,
            "validation_split_id": self.validation_split_id,
        }
        missing = tuple(sorted(key for key, value in required.items() if not value))
        if missing:
            raise RunManifestError("run manifest missing required fields: " + ",".join(missing))
        if self.internal_test_accessed and self.stage in {"training", "model_selection"}:
            raise RunManifestError(
                "training/model-selection manifests may not access internal_test"
            )
        if self.parameter_count < 0:
            raise RunManifestError("parameter_count must be non-negative")

    def stable_sha256(self) -> str:
        self.validate()
        raw = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def write_json(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = ["RunManifest", "RunManifestError"]
