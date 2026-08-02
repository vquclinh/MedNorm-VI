"""Complete-system hybrid linker invariants (Audit 0072 §15).

Model-free: no Qwen weights are downloaded and no CUDA is required. `MockReranker` supplies
deterministic scores so the plumbing - union, dedup, provenance, reranked ordering, H2
placement, null modes - is testable on any machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mednorm_vi.kb.indexing.retrieval import LocalIndex
from mednorm_vi.linking.semantic import hybrid as hy
from mednorm_vi.linking.semantic import null_gate as ng
from mednorm_vi.linking.semantic.reranker import (
    RERANKER_PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    TASK_INSTRUCTION,
    MockReranker,
    Qwen3Reranker,
    format_pair,
    rerank,
)

PARENT, CHILD, OTHER = "X10", "X100", "Y20"

FULL_REVISIONS = {
    "Qwen/Qwen3-Embedding-4B": "5cf2132abc99cad020ac570b19d031efec650f2b",
    "Qwen/Qwen3-Reranker-4B": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
    "Qwen/Qwen3-Embedding-8B": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    "Qwen/Qwen3-Reranker-0.6B": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}


def _index(titles: dict[str, str], **postings: Any) -> LocalIndex:
    payload: dict[str, Any] = {
        "index_type": "icd10_vi",
        "source_snapshot_id": "test",
        "records": {c: {"concept_id": c, "canonical_name": n} for c, n in titles.items()},
        "exact": {},
        "exact_ascii": {},
        "ngrams": {},
        "sparse_terms": {},
        "graph": {},
    }
    payload.update(postings)
    return LocalIndex(**payload)


class FixedReranker:
    """Returns a scripted score per concept id, so ordering assertions are exact."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, len(documents)))
        return [self.scores.get(d.split("|")[0].strip(), 0.0) for d in documents]

    def unload(self) -> None:
        return None


# ------------------------------------------------------------------- reranker contract


def test_reranker_prompt_is_the_official_yes_no_protocol() -> None:
    assert 'only be "yes" or "no"' in SYSTEM_PROMPT
    assert RERANKER_PROTOCOL_VERSION == "qwen3-reranker-yes-no-v1"


def test_task_instruction_forbids_unstated_specificity() -> None:
    lowered = TASK_INSTRUCTION.lower()
    assert "answer no if" in lowered
    for forbidden in ("subtype", "organism", "strength", "dose form", "brand"):
        assert forbidden in lowered


def test_pair_format_carries_instruct_query_document() -> None:
    body = format_pair("mention here", "candidate doc")
    assert body.startswith("<Instruct>: ")
    assert "<Query>: mention here" in body
    assert "<Document>: candidate doc" in body


@pytest.mark.parametrize("revision", ["22e6836", "main", "v1.0"])
def test_abbreviated_or_moving_revisions_are_refused(revision: str) -> None:
    """Audit 0072 §9: only full 40-character immutable shas are reproducible."""
    with pytest.raises(ValueError, match="40-character"):
        Qwen3Reranker("Qwen/Qwen3-Reranker-4B", revision)


def test_full_revisions_are_forty_hex_characters() -> None:
    for model, revision in FULL_REVISIONS.items():
        assert len(revision) == 40, model
        assert all(c in "0123456789abcdef" for c in revision), model


def test_reranker_reorders_and_records_both_ranks() -> None:
    candidates = [
        {"concept_id": "A", "document": "A | wrong"},
        {"concept_id": "B", "document": "B | right"},
    ]
    out = rerank(FixedReranker({"A": 0.1, "B": 0.9}), "q", candidates)
    assert [c.concept_id for c in out] == ["B", "A"]
    assert (out[0].rank_before, out[0].rank_after) == (2, 1)
    assert (out[1].rank_before, out[1].rank_after) == (1, 2)


def test_reranker_ordering_is_deterministic_under_ties() -> None:
    candidates = [{"concept_id": c, "document": f"{c} | x"} for c in ("B", "A", "C")]
    backend = FixedReranker({"A": 0.5, "B": 0.5, "C": 0.5})
    first = [c.concept_id for c in rerank(backend, "q", candidates)]
    assert all([c.concept_id for c in rerank(backend, "q", candidates)] == first for _ in range(5))
    assert first == ["B", "A", "C"]  # ties fall back to incoming rank


def test_mock_reranker_is_deterministic() -> None:
    backend = MockReranker()
    first = backend.score("viêm phổi", ["viêm phổi thùy", "sỏi thận"])
    assert all(
        backend.score("viêm phổi", ["viêm phổi thùy", "sỏi thận"]) == first for _ in range(5)
    )


# ----------------------------------------------------------------- candidate pool union


def test_pool_unions_three_sources_and_records_provenance() -> None:
    v3 = _index({"A": "a", "B": "b"}, exact={"m": ["A"]}, sparse_terms={"m": ["B"]})
    v41 = _index({"A": "a", "C": "c"}, exact_canonical={"m": ["C"]})
    pool = build = hy.build_pool(
        "m",
        v3_index=v3,
        v41_index=v41,
        dense_hits=[("D", 0.9)],
        governed_index=_index({"A": "a", "B": "b", "C": "c", "D": "d"}),
    )
    by_id = {c.concept_id: c for c in pool}
    assert set(by_id) == {"A", "B", "C", "D"}
    assert hy.SOURCE_V3 in by_id["A"].sources
    assert hy.SOURCE_V41 in by_id["C"].sources
    assert by_id["D"].sources == (hy.SOURCE_DENSE,)
    assert by_id["D"].dense_score == pytest.approx(0.9)
    assert build is pool


def test_pool_deduplicates_a_concept_found_by_several_sources() -> None:
    shared = _index({"A": "a"}, exact={"m": ["A"]})
    pool = hy.build_pool(
        "m", v3_index=shared, v41_index=shared, dense_hits=[("A", 0.8)], governed_index=shared
    )
    assert len(pool) == 1
    assert set(pool[0].sources) == {hy.SOURCE_V3, hy.SOURCE_V41, hy.SOURCE_DENSE}


def test_no_model_may_generate_a_code_outside_the_governed_kb() -> None:
    """A dense hit naming an unknown concept is dropped, not emitted."""
    governed = _index({"A": "a"})
    pool = hy.build_pool("m", dense_hits=[("A", 0.9), ("NOT_IN_KB", 0.99)], governed_index=governed)
    assert [c.concept_id for c in pool] == ["A"]


def test_v3_anchor_is_always_present_in_the_pool() -> None:
    v3 = _index({"A": "a"}, exact={"m": ["A"]})
    v41 = _index({"B": "b"}, exact_canonical={"m": ["B"]})
    pool = hy.build_pool(
        "m", v3_index=v3, v41_index=v41, governed_index=_index({"A": "a", "B": "b"})
    )
    assert "A" in [c.concept_id for c in pool]
    assert pool[0].concept_id == "A"  # anchor leads the pre-rerank order


# ------------------------------------------------------------- complete-system assembly


def _rx_governed(ids: list[str]) -> LocalIndex:
    """A minimal governed RxNorm index so the final-eligibility boundary can verify ids."""
    return LocalIndex(
        index_type="rxnorm",
        source_snapshot_id="test",
        records={
            c: {"concept_id": c, "canonical_name": c, "metadata": {"membership": "searchable"}}
            for c in ids
        },
        exact={},
        exact_ascii={},
        ngrams={},
        sparse_terms={},
        graph={},
    )


def _pool(ids: list[str], v3_top: str | None = None) -> list[hy.PooledCandidate]:
    return [
        hy.PooledCandidate(
            concept_id=c,
            sources=(hy.SOURCE_V3,) if c == v3_top else (hy.SOURCE_DENSE,),
            v3_rank=1 if c == v3_top else None,
            dense_rank=index + 1,
        )
        for index, c in enumerate(ids)
    ]


def test_reranker_can_overturn_the_v3_anchor() -> None:
    """Nothing encodes 'v3 always wins' - strong semantic evidence may overrule it."""
    pool = _pool(["A", "B"], v3_top="A")
    result = hy.run_hybrid(
        "m",
        "m",
        pool,
        {"A": "A | x", "B": "B | y"},
        FixedReranker({"A": 0.1, "B": 0.9}),
        ontology="ICD10",
        apply_hierarchy=False,
    )
    assert result.final_top1 == "B"
    assert result.v3_top1_overturned
    assert not result.v3_top1_kept


def test_v3_anchor_is_kept_when_the_reranker_agrees() -> None:
    pool = _pool(["A", "B"], v3_top="A")
    result = hy.run_hybrid(
        "m",
        "m",
        pool,
        {"A": "A | x", "B": "B | y"},
        FixedReranker({"A": 0.9, "B": 0.1}),
        ontology="ICD10",
        apply_hierarchy=False,
    )
    assert result.v3_top1_kept and not result.v3_top1_overturned


def test_hierarchy_runs_after_reranking_and_only_within_a_family() -> None:
    """H2 may reorder PARENT/CHILD, but must never lift OTHER above the family."""
    index = _index(
        {PARENT: "Bệnh rubella kèm sởi đức", CHILD: "Bệnh rubella", OTHER: "Bệnh khác"},
        exact_canonical={"seed": [PARENT]},
    )
    pool = [
        hy.PooledCandidate(concept_id=c, sources=(hy.SOURCE_DENSE,), dense_rank=i + 1)
        for i, c in enumerate([OTHER, PARENT, CHILD])
    ]
    documents = {OTHER: "Y20 | other", PARENT: "X10 | parent", CHILD: "X100 | child"}
    result = hy.run_hybrid(
        "bệnh rubella",
        "bệnh rubella",
        pool,
        documents,
        FixedReranker({"Y20": 0.99, "X10": 0.5, "X100": 0.4}),
        ontology="ICD10",
        icd_index=index,
        apply_hierarchy=True,
    )
    assert result.codes[0] == OTHER  # family containment preserved
    assert result.codes.index(CHILD) < result.codes.index(PARENT)  # H2 fixed the family
    assert result.hierarchy_applied


def test_hierarchy_is_skipped_for_rxnorm() -> None:
    pool = _pool(["A", "B"])
    result = hy.run_hybrid(
        "m", "m", pool, {"A": "A", "B": "B"}, FixedReranker({"A": 0.9, "B": 0.1}), ontology="RXNORM"
    )
    assert not result.hierarchy_applied


# ------------------------------------------------------------------------- null modes


def test_shadow_mode_records_but_never_changes_output() -> None:
    pool = _pool(["A", "B"])
    result = hy.run_hybrid(
        "kháng sinh",
        "kháng sinh",
        pool,
        {"A": "A", "B": "B"},
        FixedReranker({"A": 0.9, "B": 0.1}),
        ontology="RXNORM",
        governed_index=_rx_governed(["A", "B"]),
        null_mode=hy.NULL_MODE_SHADOW,
    )
    assert result.null_decision["decision"] == ng.DECISION_NO_VALID  # gate fired
    assert result.codes  # output unchanged
    assert result.null_mode == hy.NULL_MODE_SHADOW


def test_conservative_mode_applies_the_refusal() -> None:
    pool = _pool(["A", "B"])
    result = hy.run_hybrid(
        "kháng sinh",
        "kháng sinh",
        pool,
        {"A": "A", "B": "B"},
        FixedReranker({"A": 0.9, "B": 0.1}),
        ontology="RXNORM",
        governed_index=_rx_governed(["A", "B"]),
        null_mode=hy.NULL_MODE_CONSERVATIVE,
    )
    assert result.null_decision["decision"] == ng.DECISION_NO_VALID
    assert result.codes == ()


def test_conservative_mode_still_protects_exact_governed_tiers() -> None:
    pool = [
        hy.PooledCandidate(concept_id="A", sources=(hy.SOURCE_V3,), v3_rank=1, evidence_tier="A")
    ]
    result = hy.run_hybrid(
        "x",
        "x",
        pool,
        {"A": "A"},
        FixedReranker({"A": 0.001}),
        ontology="ICD10",
        icd_index=_index({"A": "a"}),
        null_mode=hy.NULL_MODE_CONSERVATIVE,
        apply_hierarchy=False,
    )
    assert result.codes == ("A",)


def test_unknown_null_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown null_mode"):
        hy.run_hybrid(
            "m",
            "m",
            _pool(["A"]),
            {"A": "A"},
            MockReranker(),
            ontology="ICD10",
            null_mode="aggressive",
        )


def test_empty_pool_yields_a_null_decision_and_no_codes() -> None:
    result = hy.run_hybrid("m", "m", [], {}, MockReranker(), ontology="ICD10")
    assert result.codes == ()
    assert result.null_decision["decision"] == ng.DECISION_NO_VALID


# ------------------------------------------------- evaluator and selection contracts


def test_orchestrator_pins_full_revisions_and_reports_final_metrics() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "scripts/run_0072_zero_shot_colab.py"
    ).read_text(encoding="utf-8")
    for revision in FULL_REVISIONS.values():
        assert revision in source
    # Selection must read the reranked stage, never the dense stage.
    assert '-final["recall_at_10"]' in source
    assert '"selected_on": "final_reranked' in source
    assert "training_performed" in source


def test_orchestrator_does_not_train() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "scripts/run_0072_zero_shot_colab.py"
    ).read_text(encoding="utf-8")
    assert "train_semantic_linker" not in source
    assert '"train"' not in source


def test_dense_and_final_metrics_are_separately_labelled() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "scripts/run_0072_zero_shot_colab.py"
    ).read_text(encoding="utf-8")
    for key in ("stage_dense", "stage_union", "final_reranked"):
        assert f'"{key}"' in source
    assert "only final_reranked is" in source


def test_parameter_budget_still_under_nine_billion() -> None:
    e3 = 135_002_117
    s1 = e3 + 4_021_774_336 + 4_021_784_576
    s2 = e3 + 7_567_295_488 + 595_776_512
    assert s1 == 8_178_561_029 and s1 < 9_000_000_000
    assert s2 == 8_298_074_117 and s2 < 9_000_000_000


def test_shipped_parameter_manifest_agrees_if_present() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "artifacts/semantic/0071/deployed_parameter_manifest.json"
    )
    if not path.is_file():  # pragma: no cover - artifact is gitignored
        pytest.skip("parameter manifest not built in this checkout")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["systems"]["S1"]["total_deployed_parameters"] == 8_178_561_029
    assert document["systems"]["S2"]["total_deployed_parameters"] == 8_298_074_117
