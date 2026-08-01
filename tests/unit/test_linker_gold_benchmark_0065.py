"""The linker benchmark and its refusal to report too early (Audit 0065 §8-§9).

The most important behaviour here is a refusal. Audit 0063 had to report that Recall@k was
*undefined* because no gold codes existed anywhere in the project. The failure mode that
replaces is subtler and worse: reporting a confident Recall@10 from a dozen annotations,
which looks like evidence and is not. So the evaluator declines below 50 reviewed ICD and
30 reviewed RxNorm records, and these tests hold that line.

They also hold the two separations the benchmark depends on: silver never counts as gold,
and ICD and RxNorm are never averaged together.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "evaluate_linker_gold.py"


def _module():
    spec = importlib.util.spec_from_file_location("evaluate_linker_gold", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _module()


def record(
    ontology: str,
    *,
    gold: list[str],
    ranked: list[str],
    status: str = "HUMAN_GOLD",
    no_valid: bool = False,
    stratum: str = "s",
    confidence: str = "high",
) -> dict:
    """A minimal record shaped for the metric functions; display order is shuffled."""
    candidates = [
        {
            "code": code,
            "canonical_name": code,
            "aliases": [],
            "linker_rank": position,
            "score": 10.0 - position,
            "retrieval_channels": ["exact"],
            "structure": (
                {"ontology": "ICD10", "specificity_class": "specific", "parent_code": None}
                if ontology == "ICD10"
                else {
                    "ontology": "RXNORM",
                    "tty": "SCD",
                    "ingredient": "x",
                    "strength": None,
                    "dose_form": None,
                }
            ),
        }
        for position, code in enumerate(ranked, start=1)
    ]
    # Deliberately reversed, to prove the metrics use linker_rank and not list order.
    candidates.reverse()
    return {
        "annotation_id": f"lg1-{abs(hash((ontology, tuple(gold), tuple(ranked)))):016x}"[:20],
        "ontology": ontology,
        "adjudication_status": status,
        "selected_codes": gold,
        "no_valid_candidate": no_valid,
        "offered_candidates": candidates,
        "reviewer_confidence": confidence,
        "provenance": {
            "stratum": stratum,
            "surface_words": 2,
            "exact_alias_present": True,
            "facets": [stratum],
        },
    }


# --- the refusal ---------------------------------------------------------------


def test_the_evaluator_refuses_below_the_minimum_reviewed_counts(tmp_path: Path) -> None:
    pack = tmp_path / "pack.jsonl"
    rows = [record("ICD10", gold=["I269"], ranked=["I269"]) for _ in range(10)]
    pack.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    output = tmp_path / "out"
    assert MODULE.main(["--pack", str(pack), "--output", str(output)]) == 0

    payload = json.loads((output / "linker_benchmark.json").read_text(encoding="utf-8"))
    assert payload["reportable"] is False
    assert payload["shortfall"] == {"ICD10": 40, "RXNORM": 30}
    assert "icd10" not in payload, "no metrics may be emitted while below the minimum"
    assert "rxnorm" not in payload
    assert payload["decision"]["next_intervention"] == "MORE_HUMAN_ADJUDICATION_REQUIRED"


def test_the_minimums_are_the_ones_the_brief_specifies() -> None:
    assert MODULE.MIN_REVIEWED == {"ICD10": 50, "RXNORM": 30}


def test_silver_records_do_not_count_toward_the_minimum(tmp_path: Path) -> None:
    """An automatic exact-alias match must never unlock the benchmark."""
    pack = tmp_path / "pack.jsonl"
    rows = [
        record("ICD10", gold=["I269"], ranked=["I269"], status="SILVER_UNIQUE_EXACT")
        for _ in range(80)
    ] + [
        record("RXNORM", gold=["161"], ranked=["161"], status="SILVER_UNIQUE_EXACT")
        for _ in range(80)
    ]
    pack.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    output = tmp_path / "out"
    MODULE.main(["--pack", str(pack), "--output", str(output)])
    payload = json.loads((output / "linker_benchmark.json").read_text(encoding="utf-8"))
    assert payload["reportable"] is False
    assert payload["reviewed_by_ontology"] == {"ICD10": 0, "RXNORM": 0}


def test_metrics_are_reported_once_both_minimums_are_met(tmp_path: Path) -> None:
    pack = tmp_path / "pack.jsonl"
    rows = [record("ICD10", gold=["I269"], ranked=["I269", "I50"]) for _ in range(50)] + [
        record("RXNORM", gold=["161"], ranked=["161"]) for _ in range(30)
    ]
    pack.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    output = tmp_path / "out"
    MODULE.main(["--pack", str(pack), "--output", str(output)])
    payload = json.loads((output / "linker_benchmark.json").read_text(encoding="utf-8"))
    assert payload["reportable"] is True
    assert payload["icd10"]["recall_at_1"] == 1.0
    assert payload["rxnorm"]["recall_at_1"] == 1.0
    # ICD and RxNorm are reported separately and never merged.
    assert "icd10" in payload and "rxnorm" in payload
    assert "combined" not in payload


# --- metric correctness --------------------------------------------------------


def test_rank_is_taken_from_the_linker_not_the_display_order() -> None:
    """The pack shuffles candidates for the reviewer; the metric must not be fooled."""
    row = record("ICD10", gold=["I50"], ranked=["I269", "I509", "I50"])
    assert [c["code"] for c in row["offered_candidates"]] == ["I50", "I509", "I269"]
    assert MODULE.best_rank(row) == 3, "rank 3 by linker_rank, despite being displayed first"


def test_recall_at_k_and_mrr_use_the_best_correct_rank() -> None:
    rows = [
        record("ICD10", gold=["A"], ranked=["A", "B", "C"]),  # rank 1
        record("ICD10", gold=["C"], ranked=["A", "B", "C"]),  # rank 3
        record("ICD10", gold=["Z"], ranked=["A", "B", "C"]),  # not found
    ]
    metrics = MODULE.subset_metrics(rows)
    assert metrics["recall_at_1"] == 1 / 3
    assert metrics["recall_at_5"] == 2 / 3
    assert metrics["kb_coverage"] == 2 / 3
    assert abs(metrics["mrr"] - (1.0 + 1 / 3) / 3) < 1e-9
    assert metrics["top1_exact_accuracy"] == 1 / 3


def test_no_valid_candidate_records_are_excluded_from_recall_but_counted() -> None:
    """There is nothing to retrieve, so they must not depress recall - but the rate matters."""
    rows = [
        record("ICD10", gold=["A"], ranked=["A"]),
        record("ICD10", gold=[], ranked=["B"], no_valid=True),
    ]
    metrics = MODULE.subset_metrics(rows)
    assert metrics["answerable"] == 1
    assert metrics["recall_at_1"] == 1.0
    assert metrics["no_valid_candidate"] == 1
    assert metrics["no_valid_candidate_rate"] == 0.5


def test_an_empty_offered_set_is_recorded_as_a_finding() -> None:
    rows = [record("ICD10", gold=[], ranked=[], no_valid=True)]
    metrics = MODULE.subset_metrics(rows)
    assert metrics["empty_offered_set"] == 1
    assert metrics["empty_offered_set_rate"] == 1.0


# --- the decision gate ---------------------------------------------------------


def test_low_icd_recall_selects_retrieval_repair() -> None:
    decision, _reason = MODULE.decide(
        {"recall_at_10": 0.40, "top1_exact_accuracy": 0.30},
        {"recall_at_10": 0.95, "top1_exact_accuracy": 0.80},
        ready=True,
    )
    assert decision == "ICD_SYNONYM_OR_RETRIEVAL_REPAIR"


def test_high_recall_and_low_top1_on_both_selects_a_shared_reranker() -> None:
    decision, _reason = MODULE.decide(
        {"recall_at_10": 0.90, "top1_exact_accuracy": 0.30},
        {"recall_at_10": 0.95, "top1_exact_accuracy": 0.25},
        ready=True,
    )
    assert decision == "SHARED_LEARNED_RERANKER"


def test_rxnorm_ranking_failure_alone_selects_the_structured_reranker() -> None:
    decision, _reason = MODULE.decide(
        {"recall_at_10": 0.90, "top1_exact_accuracy": 0.80},
        {"recall_at_10": 0.95, "top1_exact_accuracy": 0.20},
        ready=True,
    )
    assert decision == "RXNORM_STRUCTURED_RERANKER"


def test_an_unready_benchmark_always_asks_for_more_adjudication() -> None:
    decision, reason = MODULE.decide({}, {}, ready=False)
    assert decision == "MORE_HUMAN_ADJUDICATION_REQUIRED"
    assert "below the minimum" in reason
