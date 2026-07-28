"""ZS0 arm execution and metrics (Audit 0049).

The missing piece. Audit 0048's Stage 3 was a placeholder that set
``ARM_RESULTS = []`` and printed a note asking the reader to populate it, so
Stage 4's guard fired correctly on an empty list. This module is what Stage 3
now calls.

Two behaviours matter more than the metrics:

* **an arm that predicts nothing still returns an ``ArmResult``** with zero
  metrics. Zero predictions is a measurement; a missing result is a bug, and the
  two must never look the same;
* **a failing arm raises**, naming the arm and the exact prerequisite. Catching
  per-arm errors and continuing would turn a missing model into a reported score
  of 0.0, which is the worst possible outcome — a broken environment presented
  as evidence.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..lattice.models import ExpertSpanProposal
from ..schemas.constants import ENTITY_TYPES
from .profile import ZS0Arm
from .proposals import RejectionLedger, convert_gliner_spans, qwen_proposals
from .resolver import ResolverThresholds, emitted, resolve
from .submission import ArmResult

RUNNER_CONTRACT_VERSION = "zs0-runner-v1"


class ArmExecutionError(RuntimeError):
    """Raised when an arm cannot run. Names the arm and the prerequisite."""

    def __init__(self, arm: str, prerequisite: str, detail: str = "") -> None:
        self.arm = arm
        self.prerequisite = prerequisite
        self.detail = detail
        super().__init__(
            f"ZS0 arm {arm} could not run: {prerequisite}"
            + (f" ({detail})" if detail else ""))


@dataclass(frozen=True, slots=True)
class Document:
    """One governed document reduced to what an arm needs."""

    document_id: str
    text: str
    gold: tuple[tuple[int, int, str], ...] = ()


@dataclass
class ArmSources:
    """The callables one arm uses. Missing ones are refused, never skipped."""

    deterministic: Callable[[Document], Sequence[ExpertSpanProposal]]
    gliner: Callable[[str], Sequence[Mapping[str, Any]]] | None = None
    qwen: Callable[[str], str] | None = None
    gliner_revision: str = ""
    gliner_sha256: str = ""
    qwen_revision: str = ""
    qwen_sha256: str = ""
    # Segments are routed to Qwen only when the cheaper experts left the
    # document with no confident mention. ZS0-C is a cascade, not a second pass.
    uncertain_when_fewer_than: int = 1


@dataclass
class ArmCounters:
    """Aggregate counts. No clinical text is retained."""

    gold_total: int = 0
    predicted_total: int = 0
    true_positives: int = 0
    wrong_type: int = 0
    offset_violations: int = 0
    per_type_gold: dict[str, int] = field(default_factory=dict)
    per_type_predicted: dict[str, int] = field(default_factory=dict)
    per_type_true_positive: dict[str, int] = field(default_factory=dict)

    def per_type_metrics(self) -> dict[str, dict[str, float]]:
        report: dict[str, dict[str, float]] = {}
        for entity_type in sorted(ENTITY_TYPES):
            gold = self.per_type_gold.get(entity_type, 0)
            predicted = self.per_type_predicted.get(entity_type, 0)
            hit = self.per_type_true_positive.get(entity_type, 0)
            precision = hit / predicted if predicted else 0.0
            recall = hit / gold if gold else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            report[entity_type] = {
                "gold": float(gold), "predicted": float(predicted),
                "true_positive": float(hit), "precision": precision,
                "recall": recall, "f1": f1}
        return report


def _peak_vram_gib() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 1024 ** 3)


def _reset_vram_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def run_arm(
    arm: ZS0Arm,
    documents: Sequence[Document],
    sources: ArmSources,
    *,
    thresholds: ResolverThresholds | None = None,
    active_parameters: int = 0,
) -> ArmResult:
    """Execute one arm over every document. Always returns exactly one result.

    Raises :class:`ArmExecutionError` when a prerequisite the arm declares is
    absent — an arm that cannot run has no score, and inventing 0.0 for it would
    be indistinguishable from a genuine zero.
    """
    if arm.uses_gliner and sources.gliner is None:
        raise ArmExecutionError(
            arm.name, "the pretrained GLiNER backend is not loaded",
            "load it in Stage 1 before running this arm")
    if arm.uses_qwen_proposer and sources.qwen is None:
        raise ArmExecutionError(
            arm.name, "the pretrained Qwen backend is not loaded",
            "load it in Stage 1 before running this arm")
    if not documents:
        raise ArmExecutionError(
            arm.name, "no governed validation document was supplied")

    counters = ArmCounters()
    ledger = RejectionLedger()
    _reset_vram_peak()
    started = time.monotonic()

    for document in documents:
        proposals: list[ExpertSpanProposal] = []

        # E1 + E2 always run. A failure here is a prerequisite failure, not a
        # zero score, so it propagates with the arm named.
        try:
            proposals.extend(sources.deterministic(document))
        except Exception as error:  # noqa: BLE001 - re-raised, never swallowed
            raise ArmExecutionError(
                arm.name, "the deterministic E1/E2 path failed",
                f"{document.document_id}: {type(error).__name__}: {error}") from error

        if arm.uses_gliner and sources.gliner is not None:
            try:
                raw = sources.gliner(document.text)
            except Exception as error:  # noqa: BLE001 - re-raised, never swallowed
                raise ArmExecutionError(
                    arm.name, "GLiNER inference failed",
                    f"{document.document_id}: {type(error).__name__}: {error}"
                ) from error
            accepted, ledger = convert_gliner_spans(
                document_id=document.document_id, original_text=document.text,
                raw_spans=raw, model_revision=sources.gliner_revision,
                checkpoint_sha256=sources.gliner_sha256, ledger=ledger)
            proposals.extend(accepted)

        # ZS0-C: Qwen is a cascade step, consulted only where the cheaper
        # experts produced nothing confident.
        if (arm.uses_qwen_proposer and sources.qwen is not None
                and len(proposals) < sources.uncertain_when_fewer_than):
            try:
                completion = sources.qwen(document.text)
            except Exception as error:  # noqa: BLE001 - re-raised, never swallowed
                raise ArmExecutionError(
                    arm.name, "Qwen inference failed",
                    f"{document.document_id}: {type(error).__name__}: {error}"
                ) from error
            accepted, ledger = qwen_proposals(
                document_id=document.document_id, original_text=document.text,
                segment_start=0, segment_text=document.text, payload=completion,
                model_revision=sources.qwen_revision,
                checkpoint_sha256=sources.qwen_sha256, ledger=ledger)
            proposals.extend(accepted)

        resolved = emitted(resolve(proposals, thresholds=thresholds))

        gold = set(document.gold)
        gold_by_span = {(s, e): t for s, e, t in document.gold}
        predicted = {(m.start, m.end, m.entity_type) for m in resolved}

        counters.gold_total += len(gold)
        counters.predicted_total += len(predicted)
        counters.true_positives += len(gold & predicted)
        for _s, _e, entity_type in gold:
            counters.per_type_gold[entity_type] = (
                counters.per_type_gold.get(entity_type, 0) + 1)
        for start, end, entity_type in predicted:
            counters.per_type_predicted[entity_type] = (
                counters.per_type_predicted.get(entity_type, 0) + 1)
            if (start, end, entity_type) in gold:
                counters.per_type_true_positive[entity_type] = (
                    counters.per_type_true_positive.get(entity_type, 0) + 1)
            elif gold_by_span.get((start, end)) not in (None, entity_type):
                # Right span, wrong type — the error the metric cares about most.
                counters.wrong_type += 1

        for mention in resolved:
            if document.text[mention.start:mention.end] != mention.text:
                counters.offset_violations += 1

    elapsed = time.monotonic() - started
    precision = (counters.true_positives / counters.predicted_total
                 if counters.predicted_total else 0.0)
    recall = counters.true_positives / counters.gold_total if counters.gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)

    # A zero-prediction arm reaches here and returns a real result with zeros.
    return ArmResult(
        arm=arm.name, exact_f1=f1, exact_precision=precision, exact_recall=recall,
        wrong_type_count=counters.wrong_type,
        malformed_proposal_count=ledger.total,
        offset_violations=counters.offset_violations,
        runtime_seconds=elapsed, peak_vram_gib=_peak_vram_gib(),
        active_parameters=active_parameters)


def run_all_arms(
    arms: Sequence[ZS0Arm],
    documents: Sequence[Document],
    sources_by_arm: Mapping[str, ArmSources],
    *,
    thresholds: ResolverThresholds | None = None,
    active_parameters_by_arm: Mapping[str, int] | None = None,
) -> list[ArmResult]:
    """Run every arm, in order. Exactly one result each, or a loud failure.

    There is no ``try/except`` around an arm here on purpose: a broken arm must
    stop the run and say which one and why, rather than leaving a short
    ``ARM_RESULTS`` for Stage 4 to trip over — which is precisely how the
    Audit-0048 placeholder presented itself.
    """
    budgets = dict(active_parameters_by_arm or {})
    results: list[ArmResult] = []
    for arm in arms:
        sources = sources_by_arm.get(arm.name)
        if sources is None:
            raise ArmExecutionError(
                arm.name, "no ArmSources were supplied for this arm")
        results.append(run_arm(
            arm, documents, sources, thresholds=thresholds,
            active_parameters=int(budgets.get(arm.name, 0))))
    if len(results) != len(arms):
        raise ArmExecutionError(
            "ALL", f"expected {len(arms)} results, produced {len(results)}")
    return results


def assert_stage3_complete(results: Sequence[ArmResult], arms: Sequence[ZS0Arm]) -> None:
    """Stage 3's postcondition, checked where it is produced."""
    expected = {arm.name for arm in arms}
    produced = {result.arm for result in results}
    if len(results) != len(expected) or produced != expected:
        raise ArmExecutionError(
            "ALL",
            f"Stage 3 produced {sorted(produced)}, expected {sorted(expected)}")


__all__ = [
    "RUNNER_CONTRACT_VERSION",
    "ArmCounters",
    "ArmExecutionError",
    "ArmSources",
    "Document",
    "assert_stage3_complete",
    "run_all_arms",
    "run_arm",
]
