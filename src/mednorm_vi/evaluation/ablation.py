"""The exact four-way ablation over the governed internal-test split (spec §18.2).

Four arms, all scored by the **same** Audit 0033 exact character-offset evaluator
so the numbers are directly comparable:

    Arm 1  ViHealthBERT neural only              (E3 alone — the 0.7103 baseline)
    Arm 2  deterministic medication/laboratory   (E1 + E2 alone)
    Arm 3  unified L3 span lattice, before L4    (every proposal, untyped contest)
    Arm 4  unified lattice after L4 resolver v1  (typed, shaped, deduplicated)

Nothing here trains, and nothing here loads a model: the neural spans are supplied
by the caller (from :mod:`mednorm_vi.mention_factory.neural.runtime`, or from a
cache, or from a test double), which keeps the whole ablation testable offline.

Unfavourable results are reported exactly as measured. An arm that scores below
Arm 1 is reported below Arm 1.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..deterministic_baseline.models import Phase1BConfig, Phase1BResult
from ..deterministic_baseline.pipeline import run_phase1b
from ..document_intelligence import analyze_text
from ..lattice.builder import build_span_lattice
from ..lattice.models import (
    FAMILY_DETERMINISTIC,
    SpanLattice,
)
from ..mention_factory.neural.decoding import NeuralSpan
from ..resolution.config_v1 import ResolverV1Config
from ..resolution.resolver_v1 import resolve_lattice
from .exact_mention import (
    BOTH_BOUNDARY,
    LEFT_BOUNDARY,
    MISSED,
    PROVENANCE_DETERMINISTIC,
    PROVENANCE_HYBRID,
    PROVENANCE_NEURAL,
    RIGHT_BOUNDARY,
    SPURIOUS,
    WRONG_TYPE,
    ExactMentionEvaluator,
    Mention,
    privacy_safe_example_id,
)
from .symptom_attribution import (
    SymptomContext,
    attribute_all,
    summarize,
    treatment_cue_before,
)

ARM_NEURAL_ONLY = "arm1_vihealthbert_neural_only"
ARM_DETERMINISTIC_ONLY = "arm2_deterministic_medication_laboratory_only"
ARM_LATTICE_BEFORE_L4 = "arm3_unified_l3_lattice_before_l4"
ARM_LATTICE_AFTER_L4 = "arm4_unified_lattice_after_l4_resolver_v1"

ARMS: tuple[str, ...] = (
    ARM_NEURAL_ONLY, ARM_DETERMINISTIC_ONLY, ARM_LATTICE_BEFORE_L4, ARM_LATTICE_AFTER_L4)

NeuralSpanProvider = Callable[[str], Sequence[NeuralSpan]]


@dataclass
class _ArmAccumulator:
    evaluator: ExactMentionEvaluator = field(default_factory=ExactMentionEvaluator)
    proposal_count: int = 0
    final_count: int = 0
    abstained: int = 0
    suppressed: int = 0
    boundary_changed: int = 0


@dataclass(frozen=True, slots=True)
class DocumentArtifacts:
    """Per-example intermediate state, kept so every arm sees the same inputs."""

    example_id: str
    text: str
    phase1b: Phase1BResult
    lattice: SpanLattice
    neural_spans: tuple[NeuralSpan, ...]


def _argmax_type(distribution: Mapping[str, float]) -> str:
    if not distribution:
        return ""
    return min(distribution.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def build_artifacts(
    example: Mapping[str, Any],
    phase1b_config: Phase1BConfig,
    neural_spans: Sequence[NeuralSpan],
) -> DocumentArtifacts:
    """Run L1 -> L2 -> E1/E2 and merge with E3 into one lattice for this example."""
    example_id = str(example.get("example_id", ""))
    text = str(example.get("text", ""))
    graph = analyze_text(text, source_path=f"governed://{example_id}")
    phase1b = run_phase1b(graph, phase1b_config)
    lattice = build_span_lattice(
        example_id, text,
        routings=phase1b.routings,
        specialist_proposals=phase1b.proposals,
        neural_spans=neural_spans)
    return DocumentArtifacts(
        example_id=example_id, text=text, phase1b=phase1b,
        lattice=lattice, neural_spans=tuple(neural_spans))


def arm_mentions(
    arm: str, artifacts: DocumentArtifacts, resolver_config: ResolverV1Config,
) -> tuple[tuple[Mention, ...], dict[str, int]]:
    """Predicted mentions for one arm, plus that arm's own counters."""
    counters = {"proposals": 0, "abstained": 0, "suppressed": 0, "boundary_changed": 0}

    if arm == ARM_NEURAL_ONLY:
        mentions = tuple(
            Mention(span.start, span.end, span.entity_type, span.text,
                    provenance=PROVENANCE_NEURAL)
            for span in artifacts.neural_spans)
        counters["proposals"] = len(artifacts.neural_spans)
        return mentions, counters

    if arm == ARM_DETERMINISTIC_ONLY:
        nodes = artifacts.lattice.by_family(FAMILY_DETERMINISTIC)
        mentions = tuple(
            Mention(node.start, node.end, node.best_type(), node.text,
                    route="+".join(node.routes), section=node.section,
                    provenance=PROVENANCE_DETERMINISTIC)
            for node in nodes if node.best_type())
        counters["proposals"] = len(nodes)
        return mentions, counters

    if arm == ARM_LATTICE_BEFORE_L4:
        nodes = artifacts.lattice.proposals
        mentions = tuple(
            Mention(node.start, node.end, node.best_type(), node.text,
                    route="+".join(node.routes), section=node.section,
                    provenance=(PROVENANCE_HYBRID if len(node.expert_ids) > 1
                                else PROVENANCE_NEURAL if node.families == ("neural",)
                                else PROVENANCE_DETERMINISTIC))
            for node in nodes if node.best_type())
        counters["proposals"] = len(nodes)
        return mentions, counters

    if arm == ARM_LATTICE_AFTER_L4:
        result = resolve_lattice(
            artifacts.lattice, resolver_config, relations=artifacts.phase1b.relations)
        counts = result.counts()
        counters.update({
            "proposals": len(artifacts.lattice.proposals),
            "abstained": counts.get("abstained", 0),
            "suppressed": counts.get("suppressed", 0),
            "boundary_changed": counts.get("boundary_changed", 0),
        })
        by_id = {d.hypothesis_id: d for d in result.decisions}
        mentions = tuple(
            Mention(hypothesis.coords.absolute.start, hypothesis.coords.absolute.end,
                    _argmax_type(hypothesis.type_distribution), hypothesis.text,
                    route="+".join(by_id[hypothesis.hypothesis_id].routes),
                    section=by_id[hypothesis.hypothesis_id].section,
                    provenance=(PROVENANCE_HYBRID
                                if len(by_id[hypothesis.hypothesis_id].expert_ids) > 1
                                else PROVENANCE_NEURAL))
            for hypothesis in result.hypotheses)
        return tuple(m for m in mentions if m.entity_type), counters

    raise ValueError(f"unknown ablation arm {arm!r}")


def symptom_contexts(
    example: Mapping[str, Any], artifacts: DocumentArtifacts,
    gold: Sequence[Mention], predicted: Sequence[Mention],
) -> list[SymptomContext]:
    """Build the L2/L3-grounded context for every gold SYMPTOM failure."""
    handle = privacy_safe_example_id(str(example.get("example_id", "")))
    predicted_keys = {m.key for m in predicted}
    contexts: list[SymptomContext] = []

    for reference in gold:
        if reference.entity_type != "SYMPTOM" or reference.key in predicted_keys:
            continue
        overlapping_predictions = [
            m for m in predicted if m.overlaps(reference)]
        exact_span_match = next(
            (m for m in overlapping_predictions
             if (m.start, m.end) == (reference.start, reference.end)), None)
        same_type_overlap = next(
            (m for m in overlapping_predictions
             if m.entity_type == "SYMPTOM"), None)

        if exact_span_match is not None:
            category, prediction = WRONG_TYPE, exact_span_match
        elif same_type_overlap is not None:
            left = same_type_overlap.start != reference.start
            right = same_type_overlap.end != reference.end
            category = (BOTH_BOUNDARY if left and right
                        else LEFT_BOUNDARY if left else RIGHT_BOUNDARY)
            prediction = same_type_overlap
        else:
            category, prediction = MISSED, None

        covering = [
            node for node in artifacts.lattice.proposals
            if node.start < reference.end and reference.start < node.end
        ]
        experts = sorted({expert for node in covering for expert in node.expert_ids})
        confidence = max(
            (node.score_for("SYMPTOM") for node in covering), default=0.0)
        routes = tuple(sorted({route for node in covering for route in node.routes}))
        sections = sorted({node.section for node in covering if node.section})

        contexts.append(SymptomContext(
            privacy_safe_example_id=handle,
            gold_span=(reference.start, reference.end),
            evaluator_category=category,
            predicted_span=((prediction.start, prediction.end)
                            if prediction is not None else None),
            predicted_type=prediction.entity_type if prediction is not None else "",
            route_tags=routes,
            section=sections[0] if sections else "",
            proposing_experts=tuple(experts),
            neural_confidence=confidence,
            overlapping_competitors=len(covering),
            treatment_cue_nearby=treatment_cue_before(artifacts.text, reference.start),
            lattice_covered=bool(covering),
            symptom_proposed=any("SYMPTOM" in node.type_scores for node in covering)))

    for prediction in predicted:
        if prediction.entity_type != "SYMPTOM":
            continue
        if any(prediction.overlaps(g) for g in gold):
            continue
        covering = [
            node for node in artifacts.lattice.proposals
            if node.start < prediction.end and prediction.start < node.end
        ]
        contexts.append(SymptomContext(
            privacy_safe_example_id=handle,
            gold_span=(prediction.start, prediction.end),
            evaluator_category=SPURIOUS,
            predicted_span=(prediction.start, prediction.end),
            predicted_type="SYMPTOM",
            route_tags=tuple(sorted({r for node in covering for r in node.routes})),
            neural_confidence=max(
                (node.score_for("SYMPTOM") for node in covering), default=0.0),
            overlapping_competitors=len(covering),
            treatment_cue_nearby=treatment_cue_before(artifacts.text, prediction.start),
            lattice_covered=bool(covering),
            symptom_proposed=True))
    return contexts


def run_ablation(
    examples: Sequence[Mapping[str, Any]],
    phase1b_config: Phase1BConfig,
    resolver_config: ResolverV1Config,
    neural_spans_for: NeuralSpanProvider,
    *,
    attribution_arm: str = ARM_NEURAL_ONLY,
) -> dict[str, Any]:
    """Score every arm on the same examples and return one comparable report."""
    accumulators = {arm: _ArmAccumulator() for arm in ARMS}
    contexts: list[SymptomContext] = []

    for example in examples:
        example_id = str(example.get("example_id", ""))
        text = str(example.get("text", ""))
        source = str(example.get("source_dataset", ""))
        gold = [
            Mention(int(entity["start"]), int(entity["end"]),
                    str(entity["target_type"]), str(entity["text"]))
            for entity in example.get("entities", []) or []
        ]
        artifacts = build_artifacts(example, phase1b_config, neural_spans_for(example_id))

        for arm in ARMS:
            mentions, counters = arm_mentions(arm, artifacts, resolver_config)
            accumulator = accumulators[arm]
            accumulator.evaluator.update(
                text, gold, mentions, example_id=example_id, source=source)
            accumulator.proposal_count += counters["proposals"]
            accumulator.final_count += len(mentions)
            accumulator.abstained += counters["abstained"]
            accumulator.suppressed += counters["suppressed"]
            accumulator.boundary_changed += counters["boundary_changed"]
            if arm == attribution_arm:
                contexts.extend(symptom_contexts(example, artifacts, gold, mentions))

    arms: dict[str, Any] = {}
    for arm in ARMS:
        accumulator = accumulators[arm]
        report = accumulator.evaluator.report(
            config_hash=resolver_config.config_sha256, label=arm)
        report.pop("errors", None)
        report["proposal_count"] = accumulator.proposal_count
        report["final_count"] = accumulator.final_count
        report["resolver_abstained"] = accumulator.abstained
        report["resolver_suppressed"] = accumulator.suppressed
        report["resolver_boundary_changed"] = accumulator.boundary_changed
        report["boundary_error_total"] = sum(
            report["error_categories"].get(name, 0)
            for name in (LEFT_BOUNDARY, RIGHT_BOUNDARY, BOTH_BOUNDARY))
        arms[arm] = report

    baseline = arms[ARM_NEURAL_ONLY]
    for report in arms.values():
        report["delta_vs_arm1"] = _delta(baseline, report)

    attributions = attribute_all(contexts)
    return {
        "ablation": "exact_four_way_v1",
        "evaluator": "exact_mention_v1",
        "scope": "mention span and type only",
        "reproduces_complete_organizer_score": False,
        "resolver_config_sha256": resolver_config.config_sha256,
        "resolver_config_version": resolver_config.config_version,
        "examples": baseline["examples"],
        "gold_mentions": baseline["gold_mentions"],
        "arms": arms,
        "symptom_attribution": summarize(attributions),
        "symptom_attribution_arm": attribution_arm,
        "symptom_attributions": [a.as_dict() for a in attributions],
    }


def _delta(baseline: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    def gap(path: str, key: str) -> float:
        return round(float(report[path][key]) - float(baseline[path][key]), 6)

    categories = report["error_categories"]
    base_categories = baseline["error_categories"]
    return {
        "precision": gap("micro", "precision"),
        "recall": gap("micro", "recall"),
        "f1": gap("micro", "f1"),
        "exact_matches": categories.get("exact_match", 0) - base_categories.get(
            "exact_match", 0),
        "missed": categories.get(MISSED, 0) - base_categories.get(MISSED, 0),
        "spurious": categories.get(SPURIOUS, 0) - base_categories.get(SPURIOUS, 0),
        "wrong_type": categories.get(WRONG_TYPE, 0) - base_categories.get(WRONG_TYPE, 0),
        "boundary_errors": (
            int(report["boundary_error_total"]) - int(baseline["boundary_error_total"])),
        "final_count": int(report["final_count"]) - int(baseline["final_count"]),
        "by_type_f1": {
            name: round(float(values["f1"])
                        - float(baseline["by_type"][name]["f1"]), 6)
            for name, values in report["by_type"].items()
        },
    }


__all__ = [
    "ARMS",
    "ARM_DETERMINISTIC_ONLY",
    "ARM_LATTICE_AFTER_L4",
    "ARM_LATTICE_BEFORE_L4",
    "ARM_NEURAL_ONLY",
    "DocumentArtifacts",
    "arm_mentions",
    "build_artifacts",
    "run_ablation",
    "symptom_contexts",
]
