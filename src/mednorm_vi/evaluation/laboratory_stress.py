"""Execute the synthetic laboratory stress suite end to end and score it.

Path exercised: L1 DocumentGraph -> L2 route tags -> E2 laboratory parser ->
unified L3 span lattice -> L4 Boundary & Type Resolver v1 -> exact evaluator.

**Synthetic only.** The governed corpus has no TEST_NAME / TEST_RESULT
supervision, so nothing produced here may be reported as real laboratory
performance. Every payload this module emits carries ``synthetic: true`` and a
``real_performance_claim: false`` marker so the two can never be conflated in a
report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..deterministic_baseline.models import Phase1BConfig
from ..deterministic_baseline.pipeline import run_phase1b
from ..document_intelligence import analyze_text
from ..lattice.builder import build_span_lattice
from ..mention_factory.laboratory.synthetic import (
    RESULT_GOLD_BOUNDARY_POLICY,
    SyntheticLabCase,
    build_cases,
    suite_summary,
)
from ..mention_factory.models import HAS_RESULT
from ..resolution.config_v1 import ResolverV1Config
from ..resolution.resolver_v1 import resolve_lattice
from .exact_mention import ExactMentionEvaluator, Mention

SUITE_LABEL = "SYNTHETIC laboratory stress suite (not real performance)"


@dataclass(frozen=True, slots=True)
class PairingScore:
    """How many gold ``has_result`` edges the deterministic pairing recovered."""

    gold: int = 0
    predicted: int = 0
    correct: int = 0

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"gold": self.gold, "predicted": self.predicted, "correct": self.correct,
                "precision": round(self.precision, 6), "recall": round(self.recall, 6),
                "f1": round(self.f1, 6)}


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """One synthetic case's result, with the offsets that were actually emitted."""

    case_id: str
    family: str
    gold: tuple[tuple[int, int, str], ...]
    predicted: tuple[tuple[int, int, str], ...]
    exact_matches: int
    pairing: PairingScore
    lattice_proposals: int = 0
    resolver_abstentions: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "family": self.family,
            "gold": [list(g) for g in self.gold],
            "predicted": [list(p) for p in self.predicted],
            "exact_matches": self.exact_matches,
            "pairing": self.pairing.as_dict(),
            "lattice_proposals": self.lattice_proposals,
            "resolver_abstentions": self.resolver_abstentions,
            "notes": list(self.notes),
        }


def _predicted_pairs(
    phase1b: Any, text: str,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Distinct logical ``has_result`` pairs as ``((name span), (result span))``."""
    by_id = {p.proposal_id: p for p in phase1b.proposals}
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for relation in phase1b.relations:
        if relation.relation_type != HAS_RESULT or not relation.is_primary:
            continue
        name = by_id.get(relation.source_proposal_id)
        result = by_id.get(relation.target_proposal_id)
        if name is None or result is None:
            continue
        if text[name.start:name.end] != name.text:  # pragma: no cover - invariant
            raise ValueError("relation endpoint violates the offset invariant")
        pairs.add(((name.start, name.end), (result.start, result.end)))
    return tuple(sorted(pairs))


def run_case(
    case: SyntheticLabCase,
    phase1b_config: Phase1BConfig,
    resolver_config: ResolverV1Config,
    *,
    apply_resolver: bool = True,
) -> tuple[CaseOutcome, list[Mention], list[Mention]]:
    """Run one synthetic case through L1 -> L2 -> E2 -> L3 -> (L4) and score it."""
    case.validate()
    graph = analyze_text(case.text, source_path=f"synthetic://{case.case_id}")
    phase1b = run_phase1b(graph, phase1b_config)
    lattice = build_span_lattice(
        case.case_id, case.text,
        routings=phase1b.routings, specialist_proposals=phase1b.proposals)

    abstentions = 0
    if apply_resolver:
        result = resolve_lattice(lattice, resolver_config, relations=phase1b.relations)
        abstentions = sum(1 for d in result.decisions if d.status != "accepted")
        predicted = [
            Mention(h.coords.absolute.start, h.coords.absolute.end,
                    _argmax_type(h.type_distribution), h.text,
                    provenance="deterministic")
            for h in result.hypotheses
        ]
    else:
        predicted = [
            Mention(p.start, p.end, p.best_type(), p.text, provenance="deterministic")
            for p in lattice.proposals
        ]
    predicted = [m for m in predicted if m.entity_type]

    gold = [
        Mention(e.start, e.end, e.entity_type, e.text, provenance="deterministic")
        for e in case.entities
    ]
    gold_keys = {m.key for m in gold}
    exact = sum(1 for m in predicted if m.key in gold_keys)

    gold_pairs = set(case.gold_pairs())
    predicted_pairs = set(_predicted_pairs(phase1b, case.text))
    pairing = PairingScore(
        gold=len(gold_pairs), predicted=len(predicted_pairs),
        correct=len(gold_pairs & predicted_pairs))

    outcome = CaseOutcome(
        case_id=case.case_id, family=case.family,
        gold=tuple(sorted(m.key for m in gold)),
        predicted=tuple(sorted(m.key for m in predicted)),
        exact_matches=exact, pairing=pairing,
        lattice_proposals=len(lattice.proposals),
        resolver_abstentions=abstentions,
        notes=(case.note,) if case.note else ())
    return outcome, gold, predicted


def _argmax_type(distribution: dict[str, float]) -> str:
    if not distribution:
        return ""
    return min(distribution.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def run_suite(
    phase1b_config: Phase1BConfig,
    resolver_config: ResolverV1Config,
    *,
    cases: Sequence[SyntheticLabCase] | None = None,
    apply_resolver: bool = True,
) -> dict[str, Any]:
    """Execute every synthetic case and return one clearly-labelled report."""
    suite = tuple(cases) if cases is not None else build_cases()
    evaluator = ExactMentionEvaluator()
    outcomes: list[CaseOutcome] = []
    pairing_total = PairingScore()

    for case in suite:
        outcome, gold, predicted = run_case(
            case, phase1b_config, resolver_config, apply_resolver=apply_resolver)
        evaluator.update(case.text, gold, predicted,
                         example_id=case.case_id, source=case.family)
        outcomes.append(outcome)
        pairing_total = PairingScore(
            gold=pairing_total.gold + outcome.pairing.gold,
            predicted=pairing_total.predicted + outcome.pairing.predicted,
            correct=pairing_total.correct + outcome.pairing.correct)

    report = evaluator.report(
        config_hash=resolver_config.config_sha256, label=SUITE_LABEL)
    report.update({
        "synthetic": True,
        "real_performance_claim": False,
        "warning": (
            "Synthetic cases only. The governed corpus contains NO TEST_NAME or "
            "TEST_RESULT supervision, so these numbers say nothing about real "
            "laboratory performance."),
        "result_gold_boundary_policy": RESULT_GOLD_BOUNDARY_POLICY,
        "resolver_applied": apply_resolver,
        "suite": suite_summary(suite),
        "has_result_pairing": pairing_total.as_dict(),
        "cases": [outcome.as_dict() for outcome in outcomes],
        "by_family": _family_rollup(outcomes),
    })
    return report


def _family_rollup(outcomes: Sequence[CaseOutcome]) -> dict[str, dict[str, int]]:
    rollup: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        bucket = rollup.setdefault(
            outcome.family,
            {"cases": 0, "gold": 0, "predicted": 0, "exact": 0,
             "pair_gold": 0, "pair_correct": 0})
        bucket["cases"] += 1
        bucket["gold"] += len(outcome.gold)
        bucket["predicted"] += len(outcome.predicted)
        bucket["exact"] += outcome.exact_matches
        bucket["pair_gold"] += outcome.pairing.gold
        bucket["pair_correct"] += outcome.pairing.correct
    return dict(sorted(rollup.items()))


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the synthetic suite and write its clearly-labelled report."""
    import argparse
    import json
    from pathlib import Path

    from ..resolution.config_v1 import DEFAULT_CONFIG_PATH, load_resolver_v1_config

    parser = argparse.ArgumentParser(
        description="MedNorm-VI SYNTHETIC laboratory stress suite")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--router-config", default="configs/case_router/base.yaml")
    parser.add_argument("--medication-config", default="configs/medication/grammar_v1.yaml")
    parser.add_argument("--laboratory-config", default="configs/laboratory/parser_v1.yaml")
    parser.add_argument("--resolver-config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    phase1b = Phase1BConfig.load(
        args.router_config, args.medication_config, args.laboratory_config)
    resolver = load_resolver_v1_config(args.resolver_config)
    report = run_suite(phase1b, resolver)
    report["without_resolver"] = run_suite(phase1b, resolver, apply_resolver=False)["micro"]

    directory = Path(args.report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "laboratory_stress_synthetic.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "synthetic": True,
        "real_performance_claim": False,
        "micro": report["micro"],
        "has_result_pairing": report["has_result_pairing"],
        "suite": report["suite"],
    }, indent=2, sort_keys=True))
    return 0


__all__ = [
    "SUITE_LABEL",
    "CaseOutcome",
    "PairingScore",
    "main",
    "run_case",
    "run_suite",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
