"""The learned-L4 backend seam (spec §7). Attach a checkpoint, not a runner edit.

Before Audit 0056a ``enable_l4_learned_v2`` was **inert**: the canonical runner
called the deterministic L4 unconditionally and read no L4 flag at all, so setting
the flag ``true`` changed nothing and raised nothing. ``resolution/learned_v2.py``
was reachable only from the trainer and the ablation harness. A flag that silently
does nothing is the untruthful-fail-closed pattern this project forbids, and it is
worse than an absent flag because an operator believes they configured something.

This module is the seam that gives it real semantics, without loading a model:

* :class:`LearnedL4Backend` is the contract. One method, ``resolve``, returning the
  **same** ``ResolutionResult`` of ``EntityHypothesis`` the deterministic L4 returns.
  There is still one canonical output contract and one canonical runner;
* :class:`CheckpointLearnedL4Backend` is the real implementation, wrapping
  ``learned_v2`` and adapting its ``TypedHypothesis`` output onto that contract;
* :func:`build_learned_l4_backend` is the factory the runner calls. With no
  checkpoint it raises :class:`LearnedL4Unavailable` — it **never** returns ``None``
  and never lets the caller quietly continue on the deterministic path.

The no-silent-fallback rule is structural rather than documented. The runner asks
for a backend only when the flag is on; the factory either returns a usable backend
or raises; and :func:`build_learned_l4_backend` has no code path that yields a
deterministic resolver. A future trained checkpoint is attached by pointing
``configs/resolution/learned_l4_v2.yaml`` at it — no edit to
``inference/pipeline.py`` and no new L4 entry point.

Learned output is **not trusted**. :func:`_adapt_result` re-verifies every span
against ``original_text`` and rejects the whole document on the first violation, so
a mis-trained or corrupted checkpoint fails closed instead of emitting coordinates
that L9 would have to catch later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..lattice.models import SpanLattice
from ..mention_factory.models import RelationProposal
from ..schemas.constants import ORGANIZER_LABEL_BY_TYPE
from .learned_v2 import (
    RESOLVER_V2_VERSION,
    ResolverV2Checkpoint,
    ResolverV2Config,
    ResolverV2Error,
    ResolverV2Result,
    ResolverV2UnavailableError,
    validate_runtime_checkpoint,
)
from .learned_v2 import resolve_lattice as resolve_lattice_learned
from .models import (
    STATUS_ACCEPTED,
    STATUS_UNRESOLVED,
    BoundaryEvidence,
    EntityHypothesis,
    ResolutionResult,
    TypeEvidence,
)

LEARNED_L4_DISPATCH_VERSION = "l4-learned-dispatch-v1"

#: Config key under which a profile points at the learned-L4 declaration.
DEFAULT_LEARNED_L4_CONFIG = "configs/resolution/learned_l4_v2.yaml"


class LearnedL4Unavailable(RuntimeError):
    """Raised when learned L4 is ENABLED but cannot execute.

    Deliberately a hard error rather than a fallback. Spec §7 lets the resolver
    abstain on a *hypothesis*; it does not let the pipeline silently substitute a
    different resolver for the one the profile selected. An operator who enabled
    the learned route and received deterministic output would have no way to know.
    """

    def __init__(self, reason: str, *, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        suffix = f" ({detail})" if detail else ""
        super().__init__(
            f"learned L4 v2 is enabled but cannot run: {reason}{suffix}. "
            "The deterministic L4 is NOT substituted — a profile that selects the "
            "learned route must either supply a valid checkpoint or disable the flag"
        )


class LearnedL4OutputRejected(LearnedL4Unavailable):
    """Raised when a learned backend returns output that violates the L4 contract."""


@runtime_checkable
class LearnedL4Backend(Protocol):
    """What a learned-L4 implementation must provide.

    ``resolve`` returns the canonical ``ResolutionResult``. There is no method here
    that can emit organizer JSON or a final entity, so a learned resolver is held to
    exactly the same structural limits as the deterministic one.
    """

    @property
    def contract_version(self) -> str:
        """Immutable contract version advertised by this backend."""
        ...

    def describe(self) -> Mapping[str, Any]:
        """Provenance for the run manifest. Must not load a model."""
        ...

    def resolve(
        self, lattice: SpanLattice, *, relations: Sequence[RelationProposal] = ()
    ) -> ResolutionResult:
        """Resolve one lattice into typed hypotheses."""
        ...


def _adapt_result(
    lattice: SpanLattice, inner: ResolverV2Result, *, backend_version: str
) -> ResolutionResult:
    """Adapt learned output onto the canonical contract, verifying every span.

    Learned output is treated as untrusted input. A span whose text does not slice
    out of ``original_text``, or whose coordinates are out of range, rejects the
    **document** rather than being dropped quietly: a resolver that produced one bad
    coordinate has no claim to the others.
    """
    text = lattice.original_text
    # The learned decisions carry the coordinates each hypothesis STARTED from, which
    # is how a hypothesis is traced back to the lattice node whose evidence produced
    # it. Matching on the final coordinates instead would lose the provenance of any
    # hypothesis the backend moved.
    origin_by_hypothesis = {
        decision.hypothesis_id: (decision.original_start, decision.original_end)
        for decision in inner.decisions
    }
    proposal_by_origin = {
        (proposal.start, proposal.end): proposal for proposal in lattice.proposals
    }
    hypotheses: list[EntityHypothesis] = []
    for ordinal, hypothesis in enumerate(inner.hypotheses, start=1):
        start, end = hypothesis.position
        if not isinstance(start, int) or not isinstance(end, int):
            raise LearnedL4OutputRejected(
                "the backend returned non-integer coordinates",
                detail=f"hypothesis {hypothesis.hypothesis_id}",
            )
        abstained = bool(hypothesis.abstained)
        if not abstained:
            if start < 0 or end < start or end > len(text):
                raise LearnedL4OutputRejected(
                    "the backend returned coordinates outside the document",
                    detail=(
                        f"hypothesis {hypothesis.hypothesis_id} "
                        f"[{start}, {end}) against text length {len(text)}"
                    ),
                )
            if text[start:end] != hypothesis.text:
                raise LearnedL4OutputRejected(
                    "the backend returned text that is not an exact original_text slice (spec §4)",
                    detail=f"hypothesis {hypothesis.hypothesis_id} [{start}, {end})",
                )
        internal_type = hypothesis.argmax_type() or ""
        if not abstained and not internal_type:
            raise LearnedL4OutputRejected(
                "the backend accepted a hypothesis with no type",
                detail=f"hypothesis {hypothesis.hypothesis_id}",
            )
        organizer_type = ORGANIZER_LABEL_BY_TYPE.get(internal_type, internal_type)

        origin = origin_by_hypothesis.get(hypothesis.hypothesis_id)
        proposal = proposal_by_origin.get(origin) if origin is not None else None
        sources = tuple(proposal.sources) if proposal is not None else ()
        components = proposal.components() if proposal is not None else ()
        relation_refs = proposal.relation_refs() if proposal is not None else ()
        chosen = hypothesis.source_proposal_ids[0] if hypothesis.source_proposal_ids else ""

        hypotheses.append(
            EntityHypothesis(
                hypothesis_id=hypothesis.hypothesis_id,
                document_id=lattice.document_id,
                start=start,
                end=end,
                text=hypothesis.text,
                entity_type=organizer_type,
                status=STATUS_UNRESOLVED if abstained else STATUS_ACCEPTED,
                chosen_proposal_id=chosen,
                source_proposal_ids=tuple(hypothesis.source_proposal_ids),
                boundary_evidence=BoundaryEvidence(
                    policy=backend_version,
                    chosen_kind="learned",
                    chosen_proposal_id=chosen,
                    considered_kinds=(),
                    note=f"learned_l4={RESOLVER_V2_VERSION} ordinal={ordinal}",
                ),
                type_evidence=TypeEvidence(
                    entity_type=organizer_type,
                    source_specialist="",
                    proposed_types=tuple(
                        ORGANIZER_LABEL_BY_TYPE.get(name, name)
                        for name in sorted(hypothesis.type_distribution)
                    ),
                    note="learned_l4_v2",
                ),
                rejection_reason=None if not abstained else "learned_l4_abstained",
                has_result_pair_group_ids=relation_refs,
                score=float(hypothesis.calibrated_score),
                features={k: float(v) for k, v in hypothesis.features.items()},
                expert_ids=tuple(sorted({source.expert_id for source in sources})),
                components=components,
            )
        )

    return ResolutionResult(
        document_id=lattice.document_id,
        hypotheses=tuple(hypotheses),
        config_version=f"{backend_version}+{RESOLVER_V2_VERSION}",
        warnings=tuple(lattice.warnings),
    )


@dataclass(frozen=True, slots=True)
class CheckpointLearnedL4Backend:
    """The real learned-L4 backend: a verified local checkpoint, read-only."""

    checkpoint: ResolverV2Checkpoint
    config: ResolverV2Config
    contract_version: str = LEARNED_L4_DISPATCH_VERSION

    def describe(self) -> Mapping[str, Any]:
        metadata = self.checkpoint.metadata
        return {
            "contract_version": self.contract_version,
            "resolver_version": RESOLVER_V2_VERSION,
            "checkpoint_sha256": metadata.checkpoint_sha256,
            "model_revision": metadata.model_revision,
            "parameter_count": metadata.parameter_count,
            "train_split_id": metadata.train_split_id,
            "validation_split_id": metadata.validation_split_id,
        }

    def resolve(
        self, lattice: SpanLattice, *, relations: Sequence[RelationProposal] = ()
    ) -> ResolutionResult:
        del relations  # learned v2 scores the lattice; pairing is preserved by adaptation
        try:
            inner = resolve_lattice_learned(lattice, self.checkpoint, self.config)
        except (ResolverV2Error, ResolverV2UnavailableError) as exc:
            raise LearnedL4OutputRejected(
                "the learned resolver raised during inference", detail=str(exc)
            ) from exc
        return _adapt_result(lattice, inner, backend_version=self.contract_version)


def load_learned_l4_config(path: str | Path = DEFAULT_LEARNED_L4_CONFIG) -> ResolverV2Config:
    """Read the learned-L4 declaration. Reads no checkpoint and loads no model."""
    import yaml

    target = Path(path)
    if not target.is_file():
        raise LearnedL4Unavailable(
            "the learned-L4 configuration file is missing", detail=str(target)
        )
    doc: dict[str, Any] = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    checkpoint = doc.get("checkpoint", {})
    if not isinstance(checkpoint, Mapping):
        checkpoint = {}
    thresholds = doc.get("thresholds", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    targets = doc.get("targets", {})
    if not isinstance(targets, Mapping):
        targets = {}
    sha = str(checkpoint.get("sha256", "") or "")
    # `UNAVAILABLE_UNTRAINED` is a documented placeholder, not a digest. Treating it
    # as one would compare a real checkpoint against a sentence and fail confusingly.
    if not _looks_like_sha256(sha):
        sha = ""
    return ResolverV2Config(
        enabled=bool(doc.get("enabled", False)),
        checkpoint_path=str(checkpoint.get("path", "") or ""),
        expected_checkpoint_sha256=sha,
        max_boundary_delta=int(targets.get("max_boundary_delta", 12)),
        keep_threshold=float(thresholds.get("keep_threshold", 0.5)),
        abstain_margin=float(thresholds.get("abstain_margin", 0.05)),
        wrong_type_risk_threshold=float(thresholds.get("wrong_type_risk_threshold", 0.35)),
        config_version=str(doc.get("version", RESOLVER_V2_VERSION)),
    )


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def learned_l4_blockers(config: ResolverV2Config) -> tuple[str, ...]:
    """Why learned L4 could not run, as readiness strings. Loads nothing.

    Empty means the route is executable. Called by readiness so an operator learns
    the route is unusable **before** a run starts, not on the first document.
    """
    blockers: list[str] = []
    if not config.checkpoint_path:
        blockers.append(
            "learned_l4_no_checkpoint_configured:"
            "configs/resolution/learned_l4_v2.yaml declares no checkpoint.path; "
            "the learned L4 slot has never been trained"
        )
        return tuple(blockers)
    path = Path(config.checkpoint_path)
    if not path.is_file():
        blockers.append(f"learned_l4_checkpoint_missing:{path}")
        return tuple(blockers)
    if not config.expected_checkpoint_sha256:
        blockers.append(
            "learned_l4_no_expected_sha256:"
            "a learned checkpoint must declare its SHA-256 so a substituted file "
            "fails closed"
        )
    return tuple(blockers)


def build_learned_l4_backend(
    config: ResolverV2Config, *, backend: LearnedL4Backend | None = None
) -> LearnedL4Backend:
    """Return an executable learned-L4 backend, or raise. Never returns ``None``.

    ``backend`` is the injection point tests use to exercise dispatch without a
    model. Production passes nothing and gets the checkpoint-backed backend.
    """
    if backend is not None:
        if not isinstance(backend, LearnedL4Backend):
            raise LearnedL4Unavailable(
                "the injected backend does not satisfy the LearnedL4Backend contract"
            )
        return backend
    blockers = learned_l4_blockers(config)
    if blockers:
        raise LearnedL4Unavailable("no usable learned checkpoint", detail="; ".join(blockers))
    runtime_config = ResolverV2Config(
        enabled=True,
        checkpoint_path=config.checkpoint_path,
        expected_checkpoint_sha256=config.expected_checkpoint_sha256,
        max_boundary_delta=config.max_boundary_delta,
        keep_threshold=config.keep_threshold,
        abstain_margin=config.abstain_margin,
        wrong_type_risk_threshold=config.wrong_type_risk_threshold,
        config_version=config.config_version,
    )
    try:
        checkpoint = validate_runtime_checkpoint(runtime_config)
    except (ResolverV2UnavailableError, ResolverV2Error) as exc:
        raise LearnedL4Unavailable(
            "the learned checkpoint failed validation", detail=str(exc)
        ) from exc
    return CheckpointLearnedL4Backend(checkpoint=checkpoint, config=runtime_config)


__all__ = [
    "DEFAULT_LEARNED_L4_CONFIG",
    "LEARNED_L4_DISPATCH_VERSION",
    "CheckpointLearnedL4Backend",
    "LearnedL4Backend",
    "LearnedL4OutputRejected",
    "LearnedL4Unavailable",
    "build_learned_l4_backend",
    "learned_l4_blockers",
    "load_learned_l4_config",
]
