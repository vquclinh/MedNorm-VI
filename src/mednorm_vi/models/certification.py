"""Deployed-parameter certification.

The budget sums every model the submission DEPLOYS, not the ones resident at any moment.
Sequential loading reduces peak memory; it does not reduce what was deployed, and counting
only what happens to be in memory would be a way of not answering the question.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import RuntimeConfig
from .acquisition import file_digest
from .budget import QWEN3_8B_REVISION
from .encoders import build_encoder
from .qwen import QwenDisambiguator, log
from .registry import (
    DeployedModel,
    ModelManifest,
    RetrieverSpec,
    RevisionNotPinned,
    is_hub_revision,
)


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


__all__ = ["certify_budget"]
