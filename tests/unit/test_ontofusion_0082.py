"""Milestone 0082 guards: pretrained-only OntoFusion linking.

No weights, no network. Qwen is an injected fake for both passes; the encoders are injected
fakes; the KB is a handful of governed records.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

torch = pytest.importorskip("torch")

from mednorm_vi.graphcent import models as gm  # noqa: E402
from mednorm_vi.graphcent.ontology import ONTOLOGY_ICD, ONTOLOGY_RXNORM  # noqa: E402
from mednorm_vi.kb.indexing.retrieval import LocalIndex  # noqa: E402
from mednorm_vi.kb.rxnorm.structured import Strength, StructuredDrug  # noqa: E402
from mednorm_vi.ontofusion import pipeline as op  # noqa: E402
from mednorm_vi.ontofusion import pruning as opr  # noqa: E402
from mednorm_vi.ontofusion import reformulation as orf  # noqa: E402
from mednorm_vi.ontofusion import reranking as orr  # noqa: E402
from mednorm_vi.ontofusion import retrieval as ort  # noqa: E402
from mednorm_vi.ontofusion import runtime as orun  # noqa: E402
from mednorm_vi.ontofusion import sparse as osp  # noqa: E402

CLI_PATH = ROOT / "scripts/ontofusion_0082.py"
_spec = importlib.util.spec_from_file_location("ontofusion_cli", CLI_PATH)
assert _spec and _spec.loader
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

TYPE_DIAGNOSIS = "CHẨN_ĐOÁN"
TYPE_DRUG = "THUỐC"
TYPE_SYMPTOM = "TRIỆU_CHỨNG"
TYPE_TEST = "TÊN_XÉT_NGHIỆM"

#: The certified deployed total this milestone must not change.
CANONICAL_TOTAL = 8_729_759_237


# ------------------------------------------------------------------ fixtures


def icd_index() -> LocalIndex:
    return LocalIndex(
        index_type="icd10_vi", source_snapshot_id="t",
        records={
            "J189": {"concept_id": "J189", "canonical_name": "Viêm phổi",
                     "aliases": ["viêm phổi không xác định"]},
            "J180": {"concept_id": "J180", "canonical_name": "Viêm phế quản phổi",
                     "aliases": []},
            "I10": {"concept_id": "I10", "canonical_name": "Tăng huyết áp vô căn",
                    "aliases": ["tăng huyết áp"]},
        },
        exact={"viêm phổi": ["J189"], "tăng huyết áp": ["I10"]},
        exact_ascii={}, ngrams={}, sparse_terms={"viem": ["J189", "J180"]}, graph={},
    )


def rxnorm_index() -> LocalIndex:
    return LocalIndex(
        index_type="rxnorm", source_snapshot_id="t",
        records={
            "1191": {"concept_id": "1191", "canonical_name": "aspirin",
                     "metadata": {"tty": "IN"}},
            "2001": {"concept_id": "2001", "canonical_name": "aspirin 500 MG Oral Tablet",
                     "metadata": {"tty": "SCD"}},
            "3001": {"concept_id": "3001", "canonical_name": "aspirin 100 MG Oral Tablet",
                     "metadata": {"tty": "SCD"}},
        },
        exact={"aspirin": ["1191"]}, exact_ascii={}, ngrams={},
        sparse_terms={"aspirin": ["1191", "2001", "3001"]}, graph={},
    )


def structured() -> dict[str, StructuredDrug]:
    return {
        "2001": StructuredDrug(
            rxcui="2001", name="aspirin 500 MG Oral Tablet", ingredients=("aspirin",),
            strengths=(Strength(value="500", unit="mg", raw="500 MG"),),
            dose_forms=("Oral Tablet",),
        ),
        "3001": StructuredDrug(
            rxcui="3001", name="aspirin 100 MG Oral Tablet", ingredients=("aspirin",),
            strengths=(Strength(value="100", unit="mg", raw="100 MG"),),
            dose_forms=("Oral Tablet",),
        ),
        # 1191 deliberately has NO structured record: an unspecific concept, not a conflict.
    }


def candidate(concept_id: str, ontology: str, *, retrievers: int = 1, views: int = 1,
              exact: bool = False) -> ort.Candidate:
    made = ort.Candidate(concept_id=concept_id, ontology=ontology, exact_alias=exact)
    names = ("sparse_char_ngram", "sapbert_xlmr", "clinlinker_kb_gp")
    view_names = ("raw", "canonical_vi", "canonical_en")
    for r in range(retrievers):
        for v in range(views):
            made.record(names[r % 3], view_names[v % 3], r + v + 1)
    return made


class FakeQwen:
    """Answers the GenQR prompt and the set-wise prompt from canned replies."""

    def __init__(self, genqr: str = '{"canonical_vi": "", "canonical_en": ""}',
                 answer: str = "NONE") -> None:
        self.genqr, self.answer = genqr, answer
        self.prompts: list[str] = []
        self.unloads = 0

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer if "CANDIDATES:" in prompt else self.genqr

    def unload(self) -> None:
        self.unloads += 1

    @property
    def genqr_prompts(self) -> list[str]:
        return [p for p in self.prompts if "MENTION:" in p and "CANDIDATES:" not in p]

    @property
    def rerank_prompts(self) -> list[str]:
        return [p for p in self.prompts if "CANDIDATES:" in p]


# ------------------------------------------------------------------ GenQR contract


def test_genqr_prompt_asks_for_queries_and_forbids_codes() -> None:
    context = orf.MentionContext(
        document="1", mention="viêm phổi", entity_type=TYPE_DIAGNOSIS,
        section="Khám", previous_sentence="Bệnh nhân sốt.", sentence="Chẩn đoán viêm phổi.",
        next_sentence="Điều trị kháng sinh.",
    )
    prompt = orf.build_genqr_prompt(context)
    assert "canonical_vi" in prompt and "canonical_en" in prompt
    assert "NEVER return an ICD-10 code" in prompt
    assert "SEARCH QUERIES ONLY" in prompt
    assert "(previous)" in prompt and "(current)" in prompt and "(next)" in prompt
    assert "SECTION: Khám" in prompt


def test_genqr_cannot_inject_a_governed_id_into_a_query_view() -> None:
    result, counters = orf.parse_reformulation(
        '{"canonical_vi": "J18.9", "canonical_en": "1191"}', "viêm phổi"
    )
    assert result.canonical_vi == "" and result.canonical_en == ""
    assert counters[orf.DROPPED_CODE] == 2
    assert result.views == ("viêm phổi",)      # only the raw mention survives


def test_malformed_genqr_falls_back_to_the_raw_mention() -> None:
    result, counters = orf.parse_reformulation("I think it is pneumonia.", "viêm phổi")
    assert result.raw == "viêm phổi" and result.parse_failed
    assert result.views == ("viêm phổi",)
    assert counters[orf.PARSE_FAILED] == 1


def test_the_raw_mention_is_always_a_retrieval_view() -> None:
    for reply in (
        '{"canonical_vi": "viêm phổi mắc phải cộng đồng", "canonical_en": "pneumonia"}',
        '{"canonical_vi": "", "canonical_en": ""}',
        "nonsense",
    ):
        result, _ = orf.parse_reformulation(reply, "viêm phổi")
        assert result.views[0] == "viêm phổi"


def test_repeated_mentions_reuse_one_reformulation() -> None:
    cache = orf.ReformulationCache(revision="rev", document_checksums={"1": "abc"})
    qwen = FakeQwen('{"canonical_vi": "viêm phổi", "canonical_en": "pneumonia"}')
    context = orf.MentionContext(
        document="1", mention="viêm phổi", entity_type=TYPE_DIAGNOSIS, sentence="s"
    )
    first = cache.reformulate(context, qwen)
    second = cache.reformulate(context, qwen)
    assert first == second
    assert cache.calls == 1
    assert cache.counters["reformulation_cache_hits"] == 1


def test_a_reformulation_cache_is_keyed_by_model_prompt_and_document() -> None:
    context = orf.MentionContext(document="1", mention="m", entity_type=TYPE_DIAGNOSIS)
    a = context.cache_key(revision="rev-a", document_checksum="doc-1")
    b = context.cache_key(revision="rev-b", document_checksum="doc-1")
    c = context.cache_key(revision="rev-a", document_checksum="doc-2")
    assert len({a, b, c}) == 3
    assert orf.PROMPT_VERSION in orf.ReformulationCache(revision="r").as_dict().values()


# ------------------------------------------------------------------ sparse view


def test_sparse_reuses_the_existing_inverted_index_and_reranks_by_character_similarity() -> None:
    index = icd_index()
    hits = osp.sparse_search(index, "viêm phổi", top_k=4)
    assert hits, "the existing inverted index must supply the shortlist"
    assert hits[0].concept_id == "J189"       # closer character overlap than J180
    assert 0.0 < hits[0].similarity <= 1.0
    assert hits[0].similarity >= hits[-1].similarity


def test_sparse_is_deterministic_and_bounded_by_the_shortlist() -> None:
    index = icd_index()
    settings = osp.SparseSettings(shortlist=5)
    first = osp.sparse_search(index, "viem phoi", top_k=3, settings=settings)
    second = osp.sparse_search(index, "viem phoi", top_k=3, settings=settings)
    assert [h.concept_id for h in first] == [h.concept_id for h in second]

    memo = osp.SurfaceMemo(index, settings)
    osp.sparse_search(index, "viem phoi", top_k=3, memo=memo, settings=settings)
    assert memo.size <= len(index.records)     # never more than the shortlist touched


def test_sparse_provenance_fails_closed_on_a_different_knowledge_base(tmp_path: Path) -> None:
    mine = osp.SparseProvenance(
        settings=osp.SparseSettings(), icd_snapshot="a", rxnorm_snapshot="b",
        document_checksum="c", row_order_checksum="d", rows=3,
    )
    path = tmp_path / "sparse-provenance.json"
    osp.write_provenance(path, mine)
    restored = osp.read_provenance(path)
    assert restored is not None
    mine.assert_compatible(restored)

    moved = dataclasses.replace(mine, row_order_checksum="different")
    with pytest.raises(osp.SparseProvenanceMismatch, match="row_order_checksum"):
        mine.assert_compatible(moved)
    assert osp.read_provenance(tmp_path / "absent.json") is None


# ------------------------------------------------------------------ retrieval union


def test_the_union_deduplicates_by_governed_concept_id() -> None:
    candidates = ort.CandidateSet(ontology=ONTOLOGY_ICD)
    candidates.add("J189", ort.RETRIEVER_SPARSE, "raw", 1)
    candidates.add("J189", "sapbert_xlmr", "canonical_vi", 2)
    candidates.add("J180", "sapbert_xlmr", "raw", 3)
    assert candidates.size == 2
    assert candidates.candidates["J189"].retriever_count == 2
    assert candidates.candidates["J189"].view_count == 2


def test_the_candidate_cap_is_deterministic_and_evidence_ordered() -> None:
    candidates = ort.CandidateSet(ontology=ONTOLOGY_ICD)
    for index in range(20):
        candidates.add(f"X{index:03d}", ort.RETRIEVER_SPARSE, "raw", index + 1)
    candidates.add("X019", "sapbert_xlmr", "canonical_vi", 1)   # two retrievers
    capped = [c.concept_id for c in candidates.capped(10)]
    assert len(capped) == 10
    assert capped[0] == "X019"                                   # agreement wins
    assert capped == [c.concept_id for c in candidates.capped(10)]


def test_clinlinker_never_contributes_to_a_medication() -> None:
    clinlinker = next(s for s in gm.DEFAULT_RETRIEVERS if s.key == "clinlinker_kb_gp")
    assert gm.ROLE_DRUG not in clinlinker.role
    assert ort.role_for_ontology(ONTOLOGY_ICD) == gm.ROLE_DIAGNOSIS
    assert ort.role_for_ontology(ONTOLOGY_RXNORM) == gm.ROLE_DRUG
    with pytest.raises(ValueError, match="not declared for role"):
        ort.assert_role_allowed(clinlinker.role, ONTOLOGY_RXNORM, clinlinker.key)
    ort.assert_role_allowed(clinlinker.role, ONTOLOGY_ICD, clinlinker.key)

    sapbert = next(s for s in gm.DEFAULT_RETRIEVERS if s.key == "sapbert_xlmr")
    ort.assert_role_allowed(sapbert.role, ONTOLOGY_RXNORM, sapbert.key)


def test_dense_hits_respect_the_view_and_top_k_policy() -> None:
    candidates = ort.CandidateSet(ontology=ONTOLOGY_ICD)
    policy = ort.RetrievalPolicy(dense_top_k=2, dense_views=("raw",))
    ort.retrieve_dense(
        candidates,
        {"sapbert_xlmr": [("raw", "J189", 1), ("raw", "J180", 3),
                          ("canonical_en", "I10", 1)]},
        policy,
    )
    assert set(candidates.candidates) == {"J189"}     # rank 3 over top_k, wrong view dropped


# ------------------------------------------------------------------ pruning


def test_non_linkable_organizer_types_never_enter_the_linker() -> None:
    assert opr.linkable(TYPE_DIAGNOSIS) and opr.linkable(TYPE_DRUG)
    for entity_type in (TYPE_SYMPTOM, TYPE_TEST, "KẾT_QUẢ_XÉT_NGHIỆM"):
        assert not opr.linkable(entity_type)
        assert opr.ontology_for(entity_type) == ""

    notes = {"1": "bệnh nhân sốt cao"}
    entities = {"1": [{"text": "sốt cao", "type": TYPE_SYMPTOM, "position": [10, 17]}]}
    assert orun.collect_mentions(notes, entities) == []


def test_a_diagnosis_can_never_be_linked_to_a_drug_concept() -> None:
    outcome = opr.prune(
        [candidate("1191", ONTOLOGY_RXNORM)], entity_type=TYPE_DIAGNOSIS,
        mention="viêm phổi", governed=frozenset({"1191"}),
    )
    assert outcome.kept == []
    assert outcome.reasons[opr.PRUNE_WRONG_ONTOLOGY] == 1


def test_an_explicit_strength_conflict_is_pruned_when_structured_evidence_supports_it() -> None:
    drugs = structured()
    outcome = opr.prune(
        [candidate("2001", ONTOLOGY_RXNORM), candidate("3001", ONTOLOGY_RXNORM)],
        entity_type=TYPE_DRUG, mention="aspirin 500mg",
        governed=frozenset({"2001", "3001"}), structured=drugs,
    )
    assert [c.concept_id for c in outcome.kept] == ["2001"]
    assert outcome.reasons[opr.PRUNE_STRENGTH_CONFLICT] == 1


def test_missing_structured_metadata_never_causes_speculative_pruning() -> None:
    drugs = structured()
    # 1191 has no structured record at all: unspecific, not contradictory.
    outcome = opr.prune(
        [candidate("1191", ONTOLOGY_RXNORM)], entity_type=TYPE_DRUG,
        mention="aspirin 500mg", governed=frozenset({"1191"}), structured=drugs,
    )
    assert [c.concept_id for c in outcome.kept] == ["1191"]
    assert outcome.reasons == {}

    # and a mention that states no strength never conflicts with one that records a strength
    kept = opr.prune(
        [candidate("2001", ONTOLOGY_RXNORM)], entity_type=TYPE_DRUG, mention="aspirin",
        governed=frozenset({"2001"}), structured=drugs,
    )
    assert [c.concept_id for c in kept.kept] == ["2001"]


def test_dose_form_conflict_needs_both_sides_to_be_explicit() -> None:
    drugs = {
        "2001": dataclasses.replace(structured()["2001"], dose_forms=("Injection",)),
    }
    assert opr.form_conflict("aspirin viên nén 500mg", drugs["2001"])
    assert not opr.form_conflict("aspirin 500mg", drugs["2001"])
    assert not opr.form_conflict("aspirin viên nén", None)


def test_an_ungoverned_candidate_can_never_survive_pruning() -> None:
    outcome = opr.prune(
        [candidate("J999", ONTOLOGY_ICD)], entity_type=TYPE_DIAGNOSIS,
        mention="viêm phổi", governed=frozenset({"J189"}),
    )
    assert outcome.kept == []
    assert outcome.reasons[opr.PRUNE_NOT_GOVERNED] == 1


# ------------------------------------------------------------------ set-wise reranker


def options_for(ids: list[str]) -> list[orr.Option]:
    return orr.build_options(ids, {})


def test_the_final_prompt_never_shows_a_score_a_rank_or_a_retriever() -> None:
    prompt = orr.build_rerank_prompt(
        "viêm phổi", TYPE_DIAGNOSIS, options_for(["J189", "J180"]),
        section="Khám", sentence="Chẩn đoán viêm phổi.",
    )
    for leaked in (
        "similarity", "score", "sapbert", "clinlinker", "supporting_retrievers",
        "retriever", "top_k", "rank 1", "rank:", "rank=",
    ):
        assert leaked not in prompt.casefold(), leaked
    # the only occurrence of the word is the instruction that order means nothing
    assert prompt.casefold().count("rank") == 1
    assert "not a ranking" in prompt
    assert "[A]" in prompt and "[B]" in prompt and "[NONE]" in prompt
    assert "The order of the options carries no meaning" in prompt
    assert "Prefer a BROADER supported concept" in prompt
    assert "Choose NONE if every option" in prompt


def test_an_option_maps_to_exactly_the_offered_governed_concept() -> None:
    options = options_for(["J189", "J180"])
    assert orr.parse_selection("A", options).concept_id == "J189"
    assert orr.parse_selection("B", options).concept_id == "J180"
    assert orr.parse_selection(" [B] ", options).concept_id == "J180"


def test_none_is_an_explicit_answer_and_selects_nothing() -> None:
    selection = orr.parse_selection("NONE", options_for(["J189"]))
    assert selection.decision == orr.NONE
    assert selection.concept_id == "" and not selection.selected


@pytest.mark.parametrize(
    "reply", ["Z", "A and B", "A, B", "", "I would choose the first one", "[]", "1"]
)
def test_invalid_multiple_or_out_of_range_options_fail_closed_to_none(reply: str) -> None:
    selection = orr.parse_selection(reply, options_for(["J189", "J180"]))
    assert selection.decision == orr.NONE
    assert selection.concept_id == ""


def test_a_literal_concept_id_is_refused_rather_than_honoured() -> None:
    options = options_for(["J189", "J180"])
    for reply in ("J189", "J18.9", "1191", "The answer is J189"):
        selection = orr.parse_selection(reply, options)
        assert selection.concept_id == ""
        assert selection.counters.get(orr.LITERAL_CODE) == 1


def test_an_option_letter_outside_the_offered_list_fails_closed() -> None:
    selection = orr.parse_selection("D", options_for(["J189", "J180"]))
    assert selection.decision == orr.NONE
    assert selection.counters[orr.INVALID_OPTION] == 1


# ------------------------------------------------------------------ option order


def test_option_order_is_independent_of_retrieval_rank_and_deterministic() -> None:
    by_rank = ["J180", "J189", "I10"]
    reversed_rank = ["I10", "J189", "J180"]
    assert op.ordered_option_ids(by_rank, "viêm phổi") == op.ordered_option_ids(
        reversed_rank, "viêm phổi"
    )
    assert op.ordered_option_ids(by_rank, "viêm phổi") == op.ordered_option_ids(
        by_rank, "viêm phổi"
    )
    assert sorted(op.ordered_option_ids(by_rank, "x")) == sorted(set(by_rank))


def test_option_order_does_not_always_favour_the_lowest_concept_id() -> None:
    ids = ["A1", "B2", "C3", "D4", "E5"]
    firsts = {op.ordered_option_ids(ids, f"mention {i}")[0] for i in range(25)}
    assert len(firsts) > 1, "a hash rotation must move the first slot between mentions"


# ------------------------------------------------------------------ confidence variants


def test_high_requires_independent_support_and_broad_does_not() -> None:
    policy = op.ConfidencePolicy()
    lone = candidate("J189", ONTOLOGY_ICD, retrievers=1, views=1)
    agreed = candidate("J189", ONTOLOGY_ICD, retrievers=2, views=1)
    aliased = candidate("J189", ONTOLOGY_ICD, retrievers=1, views=1, exact=True)

    assert policy.evaluate(lone)[0] is False
    assert policy.evaluate(agreed)[0] is True
    assert policy.evaluate(aliased)[0] is True
    assert policy.evaluate(None)[0] is False


def test_variants_differ_only_in_candidates() -> None:
    record = op.MentionRecord(
        document="1", mention="viêm phổi", entity_type=TYPE_DIAGNOSIS, position=(0, 9),
        selection={"decision": "SELECT", "concept_id": "J189"}, high_confidence=False,
    )
    entities = [
        {"position": [0, 9], "text": "viêm phổi", "type": TYPE_DIAGNOSIS,
         "assertions": ["isHistorical"]},
        {"position": [10, 17], "text": "sốt cao", "type": TYPE_SYMPTOM, "assertions": []},
    ]
    rows = {
        variant: op.apply_candidates(entities, [record], variant) for variant in op.VARIANTS
    }
    for variant, serialized in rows.items():
        assert [r["position"] for r in serialized] == [[0, 9], [10, 17]]
        assert [r["type"] for r in serialized] == [TYPE_DIAGNOSIS, TYPE_SYMPTOM]
        assert serialized[0]["assertions"] == ["isHistorical"]
        assert "candidates" not in serialized[1], variant   # symptom untouched

    assert rows[op.VARIANT_ALLNULL][0]["candidates"] == []
    assert rows[op.VARIANT_HIGH][0]["candidates"] == []       # not high-confidence
    assert rows[op.VARIANT_BROAD][0]["candidates"] == ["J189"]


def test_a_none_selection_produces_an_empty_candidate_list() -> None:
    record = op.MentionRecord(
        document="1", mention="viêm phổi", entity_type=TYPE_DIAGNOSIS, position=(0, 9),
        selection={"decision": "NONE", "concept_id": ""}, high_confidence=True,
    )
    for variant in op.VARIANTS:
        assert record.candidates_for(variant) == []


def test_the_default_never_emits_more_than_one_concept() -> None:
    record = op.MentionRecord(
        document="1", mention="viêm phổi", entity_type=TYPE_DIAGNOSIS, position=(0, 9),
        selection={"decision": "SELECT", "concept_id": "J189"}, high_confidence=True,
    )
    for variant in op.VARIANTS:
        assert len(record.candidates_for(variant)) <= 1


# ------------------------------------------------------------------ end to end


NOTE = "Khám: bệnh nhân viêm phổi.\nXử trí: aspirin 500mg."


def entity_rows() -> dict[str, list[dict[str, Any]]]:
    diagnosis = NOTE.index("viêm phổi")
    drug = NOTE.index("aspirin 500mg")
    return {
        "1": [
            {"position": [diagnosis, diagnosis + 9], "text": "viêm phổi",
             "type": TYPE_DIAGNOSIS, "assertions": []},
            {"position": [drug, drug + 13], "text": "aspirin 500mg",
             "type": TYPE_DRUG, "assertions": []},
            {"position": [6, 15], "text": "bệnh nhân", "type": TYPE_SYMPTOM,
             "assertions": []},
        ]
    }


def run_pipeline(qwen: FakeQwen, **overrides: Any) -> orun.RunOutcome:
    config = orun.OntoFusionConfig(**overrides) if overrides else orun.OntoFusionConfig()
    return orun.run(
        {"1": NOTE}, entity_rows(), [], [],
        _graph_config(),
        icd_index=icd_index(), rxnorm_index=rxnorm_index(), structured=structured(),
        qwen=qwen, qwen_revision="rev", config=config,
    )


def _graph_config() -> Any:
    from mednorm_vi.graphcent.runtime import RuntimeConfig

    return RuntimeConfig(
        model_root=Path("/nonexistent"), cache_root=Path("/nonexistent"),
        qwen_model_root=Path("/nonexistent"), device="cpu",
    )


def test_only_linkable_mentions_reach_the_linker_end_to_end() -> None:
    qwen = FakeQwen(answer="NONE")
    outcome = run_pipeline(qwen)
    assert outcome.counters["verified_linkable_mentions"] == 2   # the symptom is excluded
    assert {r.entity_type for r in outcome.records} == {TYPE_DIAGNOSIS, TYPE_DRUG}


def test_the_qwen_lifecycle_is_load_reformulate_unload_retrieve_load_rerank_unload() -> None:
    qwen = FakeQwen(answer="NONE")
    outcome = run_pipeline(qwen)
    assert qwen.unloads == 2                    # once after GenQR, once after reranking
    assert qwen.genqr_prompts, "phase 1 must run"
    assert qwen.rerank_prompts, "phase 3 must run"
    assert outcome.counters["reformulation_qwen_calls"] == len(qwen.genqr_prompts)


def test_a_selected_concept_is_always_one_this_code_offered() -> None:
    qwen = FakeQwen(
        genqr='{"canonical_vi": "viêm phổi", "canonical_en": "pneumonia"}', answer="A"
    )
    outcome = run_pipeline(qwen)
    for record in outcome.records:
        if record.selected_id:
            assert record.selected_id in record.option_order
            assert record.selected_id in {c["concept_id"] for c in record.candidates}


def test_a_model_supplied_code_never_becomes_a_candidate_end_to_end() -> None:
    qwen = FakeQwen(answer="J999")
    outcome = run_pipeline(qwen)
    assert all(not r.selected_id for r in outcome.records)
    assert outcome.counters.get(orr.LITERAL_CODE, 0) >= 1


def test_variants_write_only_candidate_differences(tmp_path: Path) -> None:
    qwen = FakeQwen(answer="A")
    outcome = run_pipeline(qwen)
    entities = entity_rows()
    counts = orun.write_variants(tmp_path, entities, outcome)
    assert set(counts) == set(op.VARIANTS)
    assert counts[op.VARIANT_ALLNULL] == 0
    assert counts[op.VARIANT_BROAD] >= counts[op.VARIANT_HIGH]

    baseline = None
    for variant in op.VARIANTS:
        rows = json.loads(
            (tmp_path / f"output_{variant}_raw_dotless" / "1.json").read_text(
                encoding="utf-8"
            )
        )
        stripped = [
            {k: v for k, v in row.items() if k != "candidates"} for row in rows
        ]
        if baseline is None:
            baseline = stripped
        assert stripped == baseline, f"{variant} changed something other than candidates"
        for row in rows:
            assert NOTE[row["position"][0] : row["position"][1]] == row["text"]
    assert (tmp_path / "mention-records.jsonl").is_file()


def test_diagnostics_record_everything_needed_to_rethreshold_offline(tmp_path: Path) -> None:
    qwen = FakeQwen(answer="A")
    outcome = run_pipeline(qwen)
    counts = orun.write_variants(tmp_path, entity_rows(), outcome)
    summary = orun.diagnostics(
        outcome, counts, orun.OntoFusionConfig(), manifest_total=CANONICAL_TOTAL
    )
    assert summary["canonical_manifest_total_parameters"] == CANONICAL_TOTAL
    assert "peak_vram_gib" in summary and "runtime_seconds" in summary
    for key in ("documents", "verified_linkable_mentions", "unique_reformulations"):
        assert key in summary["counters"]

    record = json.loads(
        (tmp_path / "mention-records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    for key in ("mention", "context", "reformulation", "candidates", "option_order",
                "selection", "support", "high_confidence", "prune_reasons"):
        assert key in record
    assert "chain_of_thought" not in record and "reasoning" not in record


def test_reformulation_can_be_disabled_and_the_raw_view_still_retrieves() -> None:
    qwen = FakeQwen(answer="NONE")
    outcome = run_pipeline(qwen, use_reformulation=False)
    assert qwen.genqr_prompts == []
    assert outcome.counters["verified_linkable_mentions"] == 2
    for record in outcome.records:
        assert record.reformulation["raw"] == record.mention


# ------------------------------------------------------------------ budget and isolation


def test_the_canonical_deployed_total_is_unchanged(tmp_path: Path) -> None:
    from mednorm_vi.graphcent.acquisition import MANIFEST_NAME

    root = tmp_path / "models"
    root.mkdir()
    (root / MANIFEST_NAME).write_text(json.dumps({
        "deployed": [
            {"name": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
             "parameter_count": 278_043_648, "revision": "a" * 40},
            {"name": "ICB-UMA/ClinLinker-KB-GP", "parameter_count": 125_978_112,
             "revision": "b" * 40},
            {"name": "ViHealthBERT E3", "parameter_count": 135_002_117,
             "revision": "sha256:" + "c" * 64},
            {"name": "Qwen/Qwen3-8B", "parameter_count": 8_190_735_360,
             "revision": "d" * 40},
        ],
        "total_deployed_parameters": CANONICAL_TOTAL,
    }), encoding="utf-8")

    assert (
        278_043_648 + 125_978_112 + 135_002_117 + 8_190_735_360 == CANONICAL_TOTAL
    )
    assert CANONICAL_TOTAL < gm.PARAMETER_CAP

    args = cli.build_parser().parse_args(
        ["budget", "--model-root", str(root), "--expect-total", str(CANONICAL_TOTAL)]
    )
    assert args.func(args) == 0

    wrong = cli.build_parser().parse_args(
        ["budget", "--model-root", str(root), "--expect-total", str(CANONICAL_TOTAL + 1)]
    )
    assert wrong.func(wrong) == 3


def test_0082_adds_no_model_and_trains_nothing() -> None:
    sources = [
        *sorted((ROOT / "src/mednorm_vi/ontofusion").glob("*.py")),
        CLI_PATH,
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for banned in (
            "from_pretrained", "snapshot_download", "LoraConfig", "peft",
            "get_peft_model", "Trainer(", "optimizer", "backward()", "requires_grad",
        ):
            assert banned not in text, f"{path.name}: {banned}"
        # no repository is referenced except through the certified registry: a model name
        # may be discussed in prose, but no hub id may appear as a string.
        for hub_id in (
            "microsoft/", "cambridgeltl/", "ICB-UMA/", "pritamdeka/", "FremyCompany/",
        ):
            assert hub_id not in text, f"{path.name}: {hub_id}"


def test_0082_does_not_mutate_graphcent_or_contextual() -> None:
    """0082 must be additive: the other two milestones' modules are untouched by it."""
    for module in ("disambiguation", "runtime", "retrieval", "ontology"):
        text = (ROOT / f"src/mednorm_vi/graphcent/{module}.py").read_text(encoding="utf-8")
        assert "ontofusion" not in text, module
    for module in ("runtime", "resolver", "verifier"):
        text = (ROOT / f"src/mednorm_vi/contextual/{module}.py").read_text(encoding="utf-8")
        assert "ontofusion" not in text, module


def test_graphcent_tier_variants_are_still_present() -> None:
    from mednorm_vi.graphcent.disambiguation import VARIANTS as GRAPHCENT_VARIANTS

    assert GRAPHCENT_VARIANTS == ("allnull", "tierA", "tierAB", "tierABC")
    assert op.VARIANTS == ("allnull", "ontofusion_high", "ontofusion_broad")


def test_the_dense_cache_contract_is_untouched_by_0082() -> None:
    """0082 reads GraphCENT caches through GraphCENT; it never writes or redefines one."""
    from mednorm_vi.graphcent.retrieval import CACHE_MANIFEST_VERSION, CacheManifest

    manifest = CacheManifest(
        model_repo="r", revision="a" * 40, pooling="cls", normalized=True,
        document_checksum="d", row_order_checksum="o", rows=3, dim=2, dtype="torch.float16",
    )
    assert manifest.version == CACHE_MANIFEST_VERSION
    assert "row_order_checksum" in manifest.as_dict()

    text = (ROOT / "src/mednorm_vi/ontofusion/runtime.py").read_text(encoding="utf-8")
    assert "write_cache" not in text and "build_index_for" not in text
    assert "load_index" in text          # reads the existing cache, does not rebuild it


def test_no_stub_markers_in_the_0082_production_path() -> None:
    import re

    banned = re.compile(r"NotImplementedError|TODO|FIXME|implement on Colab|placeholder")
    for path in [*sorted((ROOT / "src/mednorm_vi/ontofusion").glob("*.py")), CLI_PATH]:
        assert not banned.search(path.read_text(encoding="utf-8")), path.name


def test_the_profile_declares_small_diverse_retrieval() -> None:
    import yaml

    config = yaml.safe_load(
        (ROOT / "configs/pipeline/ontofusion_0082.yaml").read_text(encoding="utf-8")
    )
    retrieval = config["retrieval"]
    assert retrieval["sparse_top_k"] <= 6 and retrieval["dense_top_k"] <= 6
    assert 8 <= retrieval["candidate_cap"] <= 12
    assert "raw" in retrieval["sparse_views"] and "raw" in retrieval["dense_views"]
    assert config["allow_multi_code"] is False

    settings = cli.ontofusion_config(config)
    assert settings.retrieval.candidate_cap == retrieval["candidate_cap"]
    assert settings.allow_multi_code is False
