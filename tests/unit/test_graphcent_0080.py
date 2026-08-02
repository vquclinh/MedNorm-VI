"""GraphCENT 0080 guards. No model weights, no network, no CUDA."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from mednorm_vi.graphcent import acquisition as ga  # noqa: E402
from mednorm_vi.graphcent import models as gm  # noqa: E402
from mednorm_vi.graphcent import ontology as go  # noqa: E402
from mednorm_vi.graphcent import spans as gs  # noqa: E402
from mednorm_vi.graphcent.disambiguation import (  # noqa: E402
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_NONE,
    Decision,
    TierPolicy,
    build_prompt,
    classify,
    emit_for_variant,
    parse_decision,
)
from mednorm_vi.graphcent.pipeline import (  # noqa: E402
    MentionResult,
    linkable,
    resolve_tiers,
    serialize_document,
)
from mednorm_vi.graphcent.retrieval import (  # noqa: E402
    CachedIndex,
    CacheManifest,
    CacheMismatch,
    CandidateEvidence,
    KbDocument,
    checksum,
    fuse,
    pool,
)

ROOT = Path(__file__).resolve().parents[2]


class FakeEncoder:
    """Deterministic stand-in so retrieval is testable without weights."""

    def __init__(self, key: str, table: dict[str, list[float]]) -> None:
        self.key, self.table = key, table

    def encode(self, texts: list[str], *, batch_size: int = 32) -> Any:
        rows = [self.table.get(t, [0.0, 0.0]) for t in texts]
        return torch.nn.functional.normalize(torch.tensor(rows), dim=-1)


# --------------------------------------------------------------------- pooling contract


def test_cls_and_mean_pooling_differ_and_are_both_normalized() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 4.0]]])
    mask = torch.tensor([[1, 1]])
    cls = pool(hidden, mask, gm.POOLING_CLS)
    mean = pool(hidden, mask, gm.POOLING_MEAN)
    assert not torch.allclose(cls, mean)
    assert torch.allclose(cls.norm(dim=-1), torch.ones(1))
    assert torch.allclose(mean.norm(dim=-1), torch.ones(1))


def test_mean_pooling_respects_the_attention_mask() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [99.0, 99.0]]])
    masked = pool(hidden, torch.tensor([[1, 0]]), gm.POOLING_MEAN)
    assert torch.allclose(masked, torch.tensor([[1.0, 0.0]]))


def test_registry_declares_the_official_pooling_per_model() -> None:
    by_key = {s.key: s for s in gm.DEFAULT_RETRIEVERS}
    assert by_key["sapbert_xlmr"].pooling == gm.POOLING_CLS
    assert by_key["biobert_mnli"].pooling == gm.POOLING_MEAN
    assert by_key["clinlinker_kb_gp"].pooling == gm.POOLING_CLS


# ------------------------------------------------------------- budget and licensing


def test_parameter_budget_fails_closed_over_the_cap() -> None:
    manifest = gm.ModelManifest(deployed=[gm.DeployedModel("big", 9_000_000_000)])
    with pytest.raises(gm.ParameterBudgetExceeded):
        manifest.assert_within_cap()


def test_budget_refuses_unmeasured_parameter_counts() -> None:
    manifest = gm.ModelManifest(deployed=[gm.DeployedModel("unknown", 0)])
    with pytest.raises(gm.ParameterBudgetExceeded, match="not measured"):
        manifest.assert_within_cap()


def test_non_commercial_licence_blocks_until_explicitly_cleared() -> None:
    # The shipped default already clears the gate, because the only non-commercial model
    # is disabled. Enabling it is what re-arms the gate.
    gm.ModelManifest(retrievers=list(gm.DEFAULT_RETRIEVERS)).assert_licences_cleared()

    import dataclasses as _dc

    enabled = [
        _dc.replace(s, enabled=True) if s.key == "biobert_mnli" else s
        for s in gm.DEFAULT_RETRIEVERS
    ]
    with pytest.raises(gm.LicenceReviewRequired):
        gm.ModelManifest(retrievers=enabled).assert_licences_cleared()
    gm.ModelManifest(
        retrievers=enabled, licence_overrides_accepted=("biobert_mnli",)
    ).assert_licences_cleared()


def test_disabling_the_model_also_clears_the_licence_gate() -> None:
    import dataclasses

    specs = [
        dataclasses.replace(spec, enabled=spec.key != "biobert_mnli")
        for spec in gm.DEFAULT_RETRIEVERS
    ]
    gm.ModelManifest(retrievers=specs).assert_licences_cleared()


# -------------------------------------------------------------------- row alignment


def _index(documents: list[KbDocument], vectors: Any, spec: gm.RetrieverSpec) -> CachedIndex:
    return CachedIndex(
        spec=spec,
        documents=documents,
        vectors=vectors,
        manifest=CacheManifest(
            model_repo=spec.repo_id,
            revision=spec.revision,
            pooling=spec.pooling,
            normalized=True,
            document_checksum=checksum([d.text for d in documents]),
            row_order_checksum=checksum([d.concept_id for d in documents]),
            rows=len(documents),
            dim=int(vectors.shape[1]),
            dtype=str(vectors.dtype),
        ),
    )


DOCS = [
    KbDocument("A00", "ICD10", "cholera"),
    KbDocument("B05", "ICD10", "measles"),
    KbDocument("1191", "RXNORM", "aspirin"),
]
VECS = torch.eye(3)[:, :2]
PIN_A = "a" * 40
PIN_B = "b" * 40
SPEC = dataclasses.replace(gm.DEFAULT_RETRIEVERS[0], revision=PIN_A)


def test_valid_cache_passes_and_retrieval_is_ontology_scoped() -> None:
    index = _index(DOCS, VECS, SPEC)
    index.validate()
    hits = index.search(torch.tensor([1.0, 0.0]), ontology="ICD10", top_k=5)
    assert all(concept_id in {"A00", "B05"} for concept_id, _ in hits)


def test_reordered_documents_are_refused() -> None:
    """The Audit-0073 failure mode, caught before a single query runs."""
    index = _index(DOCS, VECS, SPEC)
    index.documents = [DOCS[2], DOCS[0], DOCS[1]]
    with pytest.raises(CacheMismatch, match="row order changed"):
        index.validate()


def test_changed_document_text_is_refused() -> None:
    index = _index(DOCS, VECS, SPEC)
    index.documents = [KbDocument("A00", "ICD10", "different"), *DOCS[1:]]
    with pytest.raises(CacheMismatch, match="document text changed"):
        index.validate()


def test_wrong_pooling_in_the_cache_is_refused() -> None:
    index = _index(DOCS, VECS, SPEC)

    index.manifest = dataclasses.replace(index.manifest, pooling=gm.POOLING_MEAN)
    with pytest.raises(CacheMismatch, match="pooled with"):
        index.validate()


def test_wrong_model_repo_in_the_cache_is_refused() -> None:
    index = _index(DOCS, VECS, SPEC)
    index.manifest = dataclasses.replace(index.manifest, model_repo="other/repo")
    with pytest.raises(CacheMismatch, match="cache built from"):
        index.validate()


def test_wrong_cache_revision_is_refused() -> None:
    index = _index(DOCS, VECS, SPEC)
    index.manifest = dataclasses.replace(index.manifest, revision=PIN_B)
    with pytest.raises(CacheMismatch, match="cache built at revision"):
        index.validate()


def test_unpinned_cache_revision_is_refused() -> None:
    index = _index(DOCS, VECS, SPEC)
    index.manifest = dataclasses.replace(index.manifest, revision="main")
    with pytest.raises(CacheMismatch, match="cache revision"):
        index.validate()


def test_unpinned_spec_revision_is_refused() -> None:
    index = _index(DOCS, VECS, dataclasses.replace(SPEC, revision=""))
    with pytest.raises(CacheMismatch, match="spec revision"):
        index.validate()


def test_cache_shape_dtype_and_normalization_are_verified() -> None:
    index = _index(DOCS, VECS, SPEC)
    index.manifest = dataclasses.replace(index.manifest, dim=3)
    with pytest.raises(CacheMismatch, match="tensor has dim"):
        index.validate()

    index = _index(DOCS, VECS, SPEC)
    index.manifest = dataclasses.replace(index.manifest, dtype="torch.float16")
    with pytest.raises(CacheMismatch, match="tensor dtype"):
        index.validate()

    index = _index(DOCS, VECS, SPEC)
    index.manifest = dataclasses.replace(index.manifest, normalized=False)
    with pytest.raises(CacheMismatch, match="normalized"):
        index.validate()


def test_row_count_mismatch_is_refused() -> None:
    index = _index(DOCS, torch.eye(4)[:, :2], SPEC)
    with pytest.raises(CacheMismatch):
        index.validate()


# ------------------------------------------------------------------- candidate fusion


def test_fusion_dedupes_and_caps_the_context() -> None:
    lexical = [(f"C{i}", i + 1) for i in range(12)]
    fused = fuse(
        lexical,
        {"sapbert_xlmr": [("C0", 1), ("C99", 2)]},
        governed_ids=frozenset(f"C{i}" for i in range(12)),
        limit=10,
    )
    assert len(fused) == 10
    assert len({e.concept_id for e in fused}) == 10
    assert "C99" not in {e.concept_id for e in fused}  # ungoverned id dropped


def test_fusion_records_provenance_per_retriever() -> None:
    fused = fuse(
        [("A00", 1)],
        {"sapbert_xlmr": [("A00", 2)], "biobert_mnli": [("A00", 3)]},
        exact_alias_ids=frozenset({"A00"}),
    )
    evidence = fused[0]
    assert evidence.lexical_rank == 1
    assert evidence.retriever_ranks == {"sapbert_xlmr": 2, "biobert_mnli": 3}
    assert evidence.supporting_retrievers == 2
    assert evidence.exact_alias_match


def test_top_k_per_retriever_default_is_small() -> None:
    from mednorm_vi.graphcent.retrieval import MAX_CANDIDATE_CONTEXT, TOP_K_PER_RETRIEVER

    assert TOP_K_PER_RETRIEVER == 5
    assert MAX_CANDIDATE_CONTEXT == 10


# ------------------------------------------------------------------------ type gate


@pytest.mark.parametrize(
    "entity_type,expected",
    [
        ("CHẨN_ĐOÁN", True),
        ("THUỐC", True),
        ("TRIỆU_CHỨNG", False),
        ("TÊN_XÉT_NGHIỆM", False),
        ("KẾT_QUẢ_XÉT_NGHIỆM", False),
    ],
)
def test_only_diagnosis_and_drug_are_linked(entity_type: str, expected: bool) -> None:
    assert linkable(entity_type) is expected


# ---------------------------------------------------------------- LLM output parsing


def test_invented_id_is_rejected_and_valid_one_survives() -> None:
    decision = parse_decision(
        '{"decision":"SELECT","candidate_ids":["J18.9","INVENTED"]}', ["J18.9"]
    )
    assert decision.candidate_ids == ("J18.9",)
    assert decision.invalid_ids == ("INVENTED",)


@pytest.mark.parametrize(
    "reply", ["", "no json", "{bad", '{"decision":"MAYBE"}', '{"decision":"SELECT"}']
)
def test_parse_failure_yields_null(reply: str) -> None:
    decision = parse_decision(reply, ["J18.9"])
    assert decision.decision == "NULL" and not decision.selected


def test_only_invented_ids_collapse_to_null() -> None:
    decision = parse_decision('{"decision":"SELECT","candidate_ids":["X"]}', ["J18.9"])
    assert not decision.selected and decision.invalid_ids == ("X",)


def test_reason_codes_come_from_a_closed_enum() -> None:
    decision = parse_decision(
        '{"decision":"SELECT","candidate_ids":["J18.9"],"reason_codes":["alias_match","made_up"]}',
        ["J18.9"],
    )
    assert decision.reason_codes == ("alias_match",)


def test_prompt_is_null_first_and_forbids_writing_codes() -> None:
    prompt = build_prompt("viêm phổi", "CHẨN_ĐOÁN", "ctx", [("J18.9", ["code J18.9: x"])])
    assert "DEFAULT ANSWER IS NULL" in prompt
    assert "You cannot write a code" in prompt
    assert "broader, narrower, parent, sibling" in prompt
    assert '{"decision":"NULL"}' in prompt


# --------------------------------------------------------------------- tier gates


def _rx(conflicts: tuple[str, ...] = (), ingredients: tuple[str, ...] = ("aspirin",)):
    return go.RxNormFacts("1191", "aspirin", "IN", ingredients, "500 mg", "tablet", conflicts)


def test_tier_a_requires_exact_alias() -> None:
    evidence = CandidateEvidence("A00", exact_alias_match=True)
    assert classify(evidence, None, False, TierPolicy()) == TIER_A


def test_tier_b_requires_two_independent_retrievers() -> None:
    two = CandidateEvidence("A00", retriever_ranks={"a": 1, "b": 2})
    one = CandidateEvidence("A00", retriever_ranks={"a": 1})
    assert classify(two, None, False, TierPolicy()) == TIER_B
    assert classify(one, None, False, TierPolicy()) == TIER_NONE


def test_structured_conflict_always_demotes_to_none() -> None:
    evidence = CandidateEvidence("1191", exact_alias_match=True, retriever_ranks={"a": 1, "b": 1})
    assert (
        classify(evidence, _rx(conflicts=("strength_conflict",)), True, TierPolicy()) == TIER_NONE
    )


def test_tier_c_needs_structured_agreement() -> None:
    evidence = CandidateEvidence("1191", retriever_ranks={"a": 1})
    assert classify(evidence, _rx(), True, TierPolicy()) == TIER_C


@pytest.mark.parametrize(
    "variant,tier,expected",
    [
        ("allnull", TIER_A, False),
        ("allnull", TIER_C, False),
        ("tierA", TIER_A, True),
        ("tierA", TIER_B, False),
        ("tierAB", TIER_B, True),
        ("tierAB", TIER_C, False),
        ("tierABC", TIER_C, True),
        ("tierABC", TIER_NONE, False),
    ],
)
def test_variant_emission_matrix(variant: str, tier: str, expected: bool) -> None:
    assert emit_for_variant(tier, variant) is expected


# ------------------------------------------------------------------- serialization


SEED = [
    {"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3], "assertions": ["isNegated"]},
    {"text": "viêm phổi", "type": "CHẨN_ĐOÁN", "position": [5, 14], "assertions": []},
    {"text": "glucose", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [16, 23]},
]


def _result() -> MentionResult:
    result = MentionResult(
        document="1",
        text="viêm phổi",
        entity_type="CHẨN_ĐOÁN",
        position=(5, 14),
        span_provenance=gs.SPAN_ORIGINAL,
        candidates=[CandidateEvidence("J189", exact_alias_match=True)],
        decision=Decision("SELECT", ("J189",)),
    )
    resolve_tiers(result, lambda _: None, False, TierPolicy())
    return result


def test_allnull_variant_emits_nothing_but_keeps_everything_else() -> None:
    rows = serialize_document(SEED, [_result()], "allnull", apply_spans=False)
    diagnosis = next(r for r in rows if r["type"] == "CHẨN_ĐOÁN")
    assert diagnosis["candidates"] == []
    assert rows[0]["assertions"] == ["isNegated"]  # assertions untouched


def test_tier_a_variant_emits_the_selected_code() -> None:
    rows = serialize_document(SEED, [_result()], "tierA", apply_spans=False)
    assert next(r for r in rows if r["type"] == "CHẨN_ĐOÁN")["candidates"] == ["J189"]


def test_types_and_assertions_are_never_mutated() -> None:
    rows = serialize_document(SEED, [_result()], "tierABC", apply_spans=False)
    assert [r["type"] for r in rows] == [e["type"] for e in SEED]
    assert next(r for r in rows if r["type"] == "TRIỆU_CHỨNG")["assertions"] == ["isNegated"]


def test_lab_types_carry_neither_assertions_nor_candidates() -> None:
    rows = serialize_document(SEED, [_result()], "tierA", apply_spans=False)
    lab = next(r for r in rows if r["type"] == "KẾT_QUẢ_XÉT_NGHIỆM")
    assert "assertions" not in lab and "candidates" not in lab


# ------------------------------------------------------------------ span alternatives


DRUG_NOTE = "BN dùng amlodipine 10 mg po daily, paracetamol 500mg. 2. metformin 850 mg điều trị ĐTĐ"


def _entity(text: str, entity_type: str) -> dict[str, Any]:
    start = DRUG_NOTE.index(text)
    return {"text": text, "type": entity_type, "position": [start, start + len(text)]}


def test_organizer_medication_example_is_reachable() -> None:
    """The organizer's own example: `amlodipine 10 mg po daily`, not `amlodipine`."""
    alternative = gs.drug_regimen_alternative(DRUG_NOTE, _entity("amlodipine", "THUỐC"))
    assert alternative is not None
    assert alternative.text == "amlodipine 10 mg po daily"
    assert DRUG_NOTE[alternative.start : alternative.end] == alternative.text


def test_regimen_expansion_stops_at_clause_punctuation() -> None:
    alternative = gs.drug_regimen_alternative(DRUG_NOTE, _entity("paracetamol", "THUỐC"))
    assert alternative is not None and alternative.text == "paracetamol 500mg"


def test_regimen_expansion_stops_before_an_indication_phrase() -> None:
    alternative = gs.drug_regimen_alternative(DRUG_NOTE, _entity("metformin", "THUỐC"))
    assert alternative is not None and alternative.text == "metformin 850 mg"


def test_candidate_only_mode_offers_the_e3_span_alone() -> None:
    options = gs.alternatives_for(DRUG_NOTE, _entity("amlodipine", "THUỐC"), joint_safe_span=False)
    assert [o.text for o in options] == ["amlodipine"]
    assert options[0].provenance == gs.SPAN_ORIGINAL


def test_every_alternative_is_an_exact_source_slice() -> None:
    for option in gs.alternatives_for(
        DRUG_NOTE, _entity("amlodipine", "THUỐC"), joint_safe_span=True
    ):
        assert DRUG_NOTE[option.start : option.end] == option.text


def test_diagnosis_bridge_reuses_the_validated_0079_rule() -> None:
    note = "Trẻ Thiếu men G6PD bẩm sinh gây suy tim"

    def entity(text: str) -> dict[str, Any]:
        start = note.index(text)
        return {"text": text, "type": "CHẨN_ĐOÁN", "position": [start, start + len(text)]}

    good = gs.safe_bridge_alternative(note, [entity("Thiếu"), entity("G6PD")])
    assert good is not None and good.text == "Thiếu men G6PD"
    bad = gs.safe_bridge_alternative(note, [entity("bẩm sinh"), entity("suy tim")])
    assert bad is None  # `gây` is a connector, rejected by the 0079 rule


# ------------------------------------------------------------------ ontology facts


def test_rxnorm_strength_conflict_is_detected() -> None:
    from mednorm_vi.kb.indexing.retrieval import LocalIndex
    from mednorm_vi.kb.rxnorm.structured import StructuredDrug, parse_strengths

    index = LocalIndex(
        index_type="rxnorm",
        source_snapshot_id="t",
        records={
            "1191": {
                "concept_id": "1191",
                "canonical_name": "aspirin 250 mg",
                "metadata": {"tty": "SCD"},
            }
        },
        exact={},
        exact_ascii={},
        ngrams={},
        sparse_terms={},
        graph={},
    )
    structured = {
        "1191": StructuredDrug(
            rxcui="1191",
            tty="SCD",
            name="aspirin 250 mg",
            ingredients=("aspirin",),
            strengths=parse_strengths("250 MG"),
        )
    }
    facts = go.rxnorm_facts(index, "1191", "aspirin 500 mg", structured)
    assert facts is not None and go.CONFLICT_STRENGTH in facts.conflicts


def test_kb_documents_cover_only_governed_ids() -> None:
    from mednorm_vi.kb.indexing.retrieval import LocalIndex

    icd = LocalIndex(
        index_type="icd10_vi",
        source_snapshot_id="t",
        records={"A00": {"concept_id": "A00", "canonical_name": "Bệnh tả", "aliases": []}},
        exact={},
        exact_ascii={},
        ngrams={},
        sparse_terms={},
        graph={},
    )
    documents = go.build_kb_documents(icd, None)
    assert [d.concept_id for d in documents] == ["A00"]
    assert documents[0].ontology == go.ONTOLOGY_ICD


# ------------------------------------------------------------------ local safety


def test_cli_refuses_downloads_without_explicit_confirmation() -> None:
    source = (ROOT / "scripts/graphcent_0080.py").read_text(encoding="utf-8")
    assert "--allow-download" in source
    assert "MIN_VRAM_FOR_QWEN_GIB" in source
    assert "MIN_VRAM_FOR_RETRIEVER_GIB" in source
    assert "development machine" in source


def test_no_module_calls_from_pretrained_at_import_time() -> None:
    for name in ("models", "retrieval", "ontology", "disambiguation", "pipeline", "spans"):
        source = (ROOT / f"src/mednorm_vi/graphcent/{name}.py").read_text(encoding="utf-8")
        assert "from_pretrained" not in source, name


def test_config_declares_the_licence_gate_and_pooling_warning() -> None:
    config = (ROOT / "configs/pipeline/graphcent_0080.yaml").read_text(encoding="utf-8")
    assert "licence_overrides_accepted: []" in config
    assert "cc-by-nc-3.0" in config
    assert "enabled: false" in config
    assert "freeze_e3_types: true" in config
    assert "preserve_e3_assertions: true" in config


# ============================ completed runtime (fakes, no weights) ============================

import importlib.util  # noqa: E402

from mednorm_vi.graphcent import encoders as ge  # noqa: E402
from mednorm_vi.graphcent import runtime as gr  # noqa: E402
from mednorm_vi.kb.indexing.retrieval import LocalIndex  # noqa: E402

CLI = ROOT / "scripts/graphcent_0080.py"
_cli_spec = importlib.util.spec_from_file_location("gc_cli", CLI)
assert _cli_spec and _cli_spec.loader
cli = importlib.util.module_from_spec(_cli_spec)
_cli_spec.loader.exec_module(cli)


class FakeRetrieverEncoder:
    """Records load/unload so lifecycle is observable without weights."""

    events: list[str] = []

    def __init__(self, spec: gm.RetrieverSpec, model_root: Path, device: str = "cuda",
                 **kwargs: Any) -> None:
        self.spec, self.resolved_revision, self.embedding_dim = spec, fake_sha(spec.key), 2
        self.loaded = False

    def encode(self, texts: Any, batch_size: int | None = None) -> Any:
        self.loaded = True
        FakeRetrieverEncoder.events.append(f"load:{self.spec.key}")
        rows = [[float(len(t) % 3), float(len(t) % 5)] for t in texts]
        return torch.nn.functional.normalize(torch.tensor(rows) + 0.1, dim=-1)

    def parameter_count(self) -> int:
        return 111_000_000

    def unload(self) -> None:
        self.loaded = False
        FakeRetrieverEncoder.events.append(f"unload:{self.spec.key}")


def fake_sha(key: str) -> str:
    """A deterministic 40-hex stand-in for a hub commit, so fakes are pinned like the real
    thing. Production refuses an unpinned spec, and tests must not be exempt from that."""
    import hashlib

    return hashlib.sha1(key.encode()).hexdigest()


def pinned(specs: Sequence[gm.RetrieverSpec]) -> list[gm.RetrieverSpec]:
    return [dataclasses.replace(s, revision=fake_sha(s.key)) for s in specs]


def enabled_specs() -> list[gm.RetrieverSpec]:
    return pinned([s for s in gm.DEFAULT_RETRIEVERS if s.enabled])


class FakeQwen:
    def __init__(self, reply: str = '{"decision":"NULL"}') -> None:
        self.reply, self.prompts, self.unloads = reply, [], 0

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply

    def parameter_count(self) -> int:
        from mednorm_vi.reasoner.budget import QWEN3_8B_PARAMETERS

        return QWEN3_8B_PARAMETERS

    def unload(self) -> None:
        self.unloads += 1


def _tiny_kb() -> tuple[LocalIndex, LocalIndex, list[KbDocument]]:
    icd = LocalIndex(
        index_type="icd10_vi", source_snapshot_id="t",
        records={
            "J189": {"concept_id": "J189", "canonical_name": "Viêm phổi", "aliases": []},
            "J180": {"concept_id": "J180", "canonical_name": "Viêm phế quản phổi",
                     "aliases": []},
        },
        exact={"viêm phổi": ["J189"]}, exact_ascii={}, ngrams={},
        sparse_terms={"viem": ["J189", "J180"]}, graph={},
    )
    rx = LocalIndex(
        index_type="rxnorm", source_snapshot_id="t",
        records={"1191": {"concept_id": "1191", "canonical_name": "aspirin",
                          "metadata": {"tty": "IN", "membership": "searchable"}}},
        exact={"aspirin": ["1191"]}, exact_ascii={}, ngrams={},
        sparse_terms={"aspirin": ["1191"]}, graph={},
    )
    return icd, rx, go.build_kb_documents(icd, rx, {})


def _config(tmp_path: Path) -> gr.RuntimeConfig:
    return gr.RuntimeConfig(
        model_root=tmp_path / "models", cache_root=tmp_path / "cache",
        qwen_model_root=tmp_path / "qwen", device="cpu", embed_batch_size=4,
    )


def test_production_commands_are_implemented_not_stubs() -> None:
    source = CLI.read_text(encoding="utf-8")
    for banned in ("NotImplementedError", "TODO", "implement on Colab",
                   "runs on the GPU host; see"):
        assert banned not in source, banned
    for command in ("cmd_download_models", "cmd_build_index", "cmd_smoke", "cmd_run",
                    "cmd_package"):
        assert f"def {command}(" in source
    # each command body does real work, not logging
    assert "acquire(" in source                    # resolves + downloads at a pinned SHA
    assert "snapshot_download(" in (
        ROOT / "src/mednorm_vi/graphcent/acquisition.py"
    ).read_text(encoding="utf-8")
    assert "build_index_for(" in source
    assert "run_documents(" in source
    assert "write_variants(" in source
    assert "package_output_zip(" in source


def test_encoder_adapters_are_distinct_per_model() -> None:
    assert ge.ENCODER_BY_KEY["sapbert_xlmr"] is ge.SapBertEncoder
    assert ge.ENCODER_BY_KEY["clinlinker_kb_gp"] is ge.ClinLinkerEncoder
    assert ge.ENCODER_BY_KEY["biobert_mnli"] is ge.SentenceMeanEncoder
    with pytest.raises(KeyError, match="no encoder adapter"):
        ge.build_encoder(
            dataclasses.replace(gm.DEFAULT_RETRIEVERS[0], key="unknown"), Path("/x")
        )


def test_adapter_pooling_must_match_the_registry() -> None:
    mislabelled = dataclasses.replace(gm.DEFAULT_RETRIEVERS[0], pooling=gm.POOLING_MEAN)
    with pytest.raises(ValueError, match="registry declares pooling"):
        ge.build_encoder(mislabelled, Path("/x"))


def test_index_build_writes_a_validated_cache_and_unloads(tmp_path: Path) -> None:
    FakeRetrieverEncoder.events = []
    _, _, documents = _tiny_kb()
    config = _config(tmp_path)
    index = gr.build_index_for(
        pinned([gm.DEFAULT_RETRIEVERS[0]])[0], documents, config,
        encoder_factory=FakeRetrieverEncoder,
    )
    index.validate()
    assert gr.cache_path(config.cache_root, "sapbert_xlmr").with_suffix(".pt").is_file()
    assert "unload:sapbert_xlmr" in FakeRetrieverEncoder.events
    restored = gr.load_index(pinned([gm.DEFAULT_RETRIEVERS[0]])[0], documents, config)
    assert restored.manifest.row_order_checksum == index.manifest.row_order_checksum


def test_reordered_documents_break_cache_reload(tmp_path: Path) -> None:
    _, _, documents = _tiny_kb()
    config = _config(tmp_path)
    spec = pinned([gm.DEFAULT_RETRIEVERS[0]])[0]
    gr.build_index_for(spec, documents, config, encoder_factory=FakeRetrieverEncoder)
    with pytest.raises(CacheMismatch):
        gr.load_index(spec, list(reversed(documents)), config)


def test_disabled_retriever_is_never_loaded(tmp_path: Path) -> None:
    FakeRetrieverEncoder.events = []
    icd, rx, documents = _tiny_kb()
    config = _config(tmp_path)
    specs = enabled_specs()  # biobert disabled by default
    for spec in specs:
        gr.build_index_for(spec, documents, config, encoder_factory=FakeRetrieverEncoder)
    assert not any("biobert_mnli" in event for event in FakeRetrieverEncoder.events)
    assert "biobert_mnli" not in {s.key for s in specs}


def test_retrieval_precompute_loads_each_model_once_then_unloads(tmp_path: Path) -> None:
    FakeRetrieverEncoder.events = []
    icd, rx, documents = _tiny_kb()
    config = _config(tmp_path)
    specs = enabled_specs()
    for spec in specs:
        gr.build_index_for(spec, documents, config, encoder_factory=FakeRetrieverEncoder)
    FakeRetrieverEncoder.events = []
    evidence = gr.precompute_retrieval(
        [("viêm phổi", go.ONTOLOGY_ICD)], specs, documents, config,
        encoder_factory=FakeRetrieverEncoder,
    )
    for spec in specs:
        assert FakeRetrieverEncoder.events.count(f"unload:{spec.key}") >= 1
    assert evidence.for_span(f"{go.ONTOLOGY_ICD}\x00viêm phổi")


def test_one_pass_derives_all_four_variants_without_rerunning_qwen(tmp_path: Path) -> None:
    icd, rx, documents = _tiny_kb()
    config = _config(tmp_path)
    for spec in enabled_specs():
        gr.build_index_for(spec, documents, config, encoder_factory=FakeRetrieverEncoder)

    note = "Bệnh nhân viêm phổi, dùng aspirin."
    seeds = {"1": [
        {"text": "viêm phổi", "type": "CHẨN_ĐOÁN",
         "position": [note.index("viêm phổi"), note.index("viêm phổi") + 9], "assertions": []},
        {"text": "aspirin", "type": "THUỐC",
         "position": [note.index("aspirin"), note.index("aspirin") + 7], "assertions": []},
        {"text": "Bệnh nhân", "type": "TRIỆU_CHỨNG", "position": [0, 9], "assertions": []},
    ]}
    qwen = FakeQwen('{"decision":"SELECT","candidate_ids":["J189"]}')
    outcome = gr.run_documents(
        {"1": note}, seeds, enabled_specs(),
        documents, config, icd_index=icd, rxnorm_index=rx, structured={},
        disambiguator=qwen, encoder_factory=FakeRetrieverEncoder,
    )
    assert len(qwen.prompts) == 2      # only the diagnosis and the drug, never the symptom
    emitted = gr.write_variants(tmp_path / "run", seeds, outcome, config)
    assert emitted["allnull"] == 0
    assert emitted["tierA"] >= 1       # exact alias match on "viêm phổi"
    assert len(qwen.prompts) == 2      # variants derived, Qwen NOT re-run
    assert (tmp_path / "run" / "mention-records.jsonl").is_file()


def test_qwen_only_ever_sees_governed_candidates(tmp_path: Path) -> None:
    icd, rx, documents = _tiny_kb()
    config = _config(tmp_path)
    for spec in enabled_specs():
        gr.build_index_for(spec, documents, config, encoder_factory=FakeRetrieverEncoder)
    note = "viêm phổi"
    seeds = {"1": [{"text": "viêm phổi", "type": "CHẨN_ĐOÁN", "position": [0, 9],
                    "assertions": []}]}
    qwen = FakeQwen()
    gr.run_documents(
        {"1": note}, seeds, enabled_specs(), documents,
        config, icd_index=icd, rxnorm_index=rx, structured={}, disambiguator=qwen,
        encoder_factory=FakeRetrieverEncoder,
    )
    prompt = qwen.prompts[0]
    offered = {token for token in ("J189", "J180") if token in prompt}
    assert offered, "the prompt must carry governed ICD candidates"
    assert "1191" not in prompt        # RxNorm never offered for a diagnosis
    assert "DEFAULT ANSWER IS NULL" in prompt

    # every code-shaped token in the candidate block is a governed ICD concept
    import re as _re

    governed = {document.concept_id for document in documents}
    block = prompt.split("CANDIDATES:", 1)[1]
    for token in _re.findall(r"\b[A-Z]\d{2,5}\b", block):
        assert token in governed, token


def test_budget_counts_every_enabled_model_even_though_loading_is_sequential(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = gr.certify_budget(
        enabled_specs(), config,
        e3_checkpoint=None, qwen=None, encoder_factory=FakeRetrieverEncoder,
    )
    assert len(manifest.deployed) == 2          # both enabled retrievers counted
    assert manifest.total_parameters == 222_000_000
    assert manifest.assert_within_cap() < gm.PARAMETER_CAP


def test_budget_refuses_an_unpinned_retriever(tmp_path: Path) -> None:
    with pytest.raises(gm.RevisionNotPinned, match="without an immutable hub commit"):
        gr.certify_budget(
            [gm.DEFAULT_RETRIEVERS[0]], _config(tmp_path),
            e3_checkpoint=None, qwen=None, encoder_factory=FakeRetrieverEncoder,
        )


def test_runtime_manifest_records_resolved_revisions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = gr.certify_budget(
        pinned([gm.DEFAULT_RETRIEVERS[0]]), config, e3_checkpoint=None, qwen=None,
        encoder_factory=FakeRetrieverEncoder,
    )
    sha = fake_sha("sapbert_xlmr")
    assert manifest.retrievers[0].revision == sha
    assert manifest.as_dict()["retrievers"][0]["revision"] == sha
    assert manifest.as_dict()["resolved_revisions"] == {"sapbert_xlmr": sha}
    assert manifest.deployed[0].revision == sha
    manifest.assert_revisions_pinned()


def test_manifest_records_e3_digest_and_qwen_revision(tmp_path: Path) -> None:
    from mednorm_vi.reasoner.budget import QWEN3_8B_PARAMETERS, QWEN3_8B_REVISION

    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state_dict": {"w": torch.ones(2, 3)}}, checkpoint)

    manifest = gr.certify_budget(
        [], _config(tmp_path), e3_checkpoint=checkpoint, qwen=FakeQwen(),
        encoder_factory=FakeRetrieverEncoder,
    )
    by_name = {row["name"]: row for row in manifest.as_dict()["deployed"]}
    assert by_name["ViHealthBERT E3"]["parameter_count"] == 6
    assert by_name["ViHealthBERT E3"]["revision"] == ga.file_digest(checkpoint)
    assert by_name["Qwen/Qwen3-8B"]["parameter_count"] == QWEN3_8B_PARAMETERS
    assert by_name["Qwen/Qwen3-8B"]["revision"] == QWEN3_8B_REVISION
    manifest.assert_revisions_pinned()


def test_default_profile_disables_the_non_commercial_model() -> None:
    import yaml

    default = yaml.safe_load(
        (ROOT / "configs/pipeline/graphcent_0080.yaml").read_text(encoding="utf-8")
    )
    assert default["retrievers"]["biobert_mnli"]["enabled"] is False
    assert default["retrievers"]["clinlinker_kb_gp"]["enabled"] is True
    assert default["licence_overrides_accepted"] == []

    research = yaml.safe_load(
        (ROOT / "configs/pipeline/graphcent_0080_full_research.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert research["retrievers"]["biobert_mnli"]["enabled"] is True
    assert research["licence_overrides_accepted"] == ["biobert_mnli"]


def test_corrected_licence_metadata() -> None:
    by_key = {s.key: s for s in gm.DEFAULT_RETRIEVERS}
    assert by_key["sapbert_xlmr"].licence == "mit"
    assert by_key["biobert_mnli"].licence == "cc-by-nc-3.0"
    assert by_key["clinlinker_kb_gp"].licence == "apache-2.0"
    assert not by_key["clinlinker_kb_gp"].licence_needs_review
    assert by_key["biobert_mnli"].licence_needs_review
    for spec in gm.DEFAULT_RETRIEVERS:
        assert spec.licence_source.startswith("https://huggingface.co/")


def test_host_check_is_capability_based_not_a_fixed_threshold() -> None:
    source = CLI.read_text(encoding="utf-8")
    assert "MIN_VRAM_FOR_QWEN_GIB" in source
    assert "MIN_VRAM_FOR_RETRIEVER_GIB" in source
    assert "RECOMMENDED_VRAM_GIB" in source
    assert cli.MIN_VRAM_FOR_RETRIEVER_GIB < cli.MIN_VRAM_FOR_QWEN_GIB
    assert cli.MIN_VRAM_FOR_QWEN_GIB < 22.5   # an L4 must not be rejected


def test_package_validates_the_full_organizer_contract() -> None:
    source = CLI.read_text(encoding="utf-8")
    for check in ("unsupported field", "ungoverned ICD code", "ungoverned RxCUI",
                  "offsets do not match the source text", "assertions on",
                  "candidates on", "not dotted-serializable"):
        assert check in source


def test_package_rejects_a_bad_output_directory(tmp_path: Path) -> None:
    bad = tmp_path / "out"
    bad.mkdir()
    (bad / "1.json").write_text(
        json.dumps([{"text": "x", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [0, 1],
                     "assertions": []}]), encoding="utf-8"
    )
    issues = cli.validate_output(bad, {"1": "x"}, {"ICD10": frozenset()}, 1)
    assert any("assertions on" in issue for issue in issues)


def test_no_stub_markers_anywhere_in_the_production_path() -> None:
    import re as _re

    banned = _re.compile(r"NotImplementedError|TODO|FIXME|implement on Colab|placeholder")
    paths = [*sorted((ROOT / "src/mednorm_vi/graphcent").glob("*.py")),
             ROOT / "scripts/graphcent_0080.py"]
    for path in paths:
        assert not banned.search(path.read_text(encoding="utf-8")), path.name


def test_qwen_parameter_count_is_read_from_headers_not_by_loading(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    root = tmp_path / "qwen"
    root.mkdir()
    tensors = {"a": torch.zeros(3, 5), "b": torch.zeros(7)}
    safetensors.save_file(tensors, str(root / "model-00001-of-00002.safetensors"))
    safetensors.save_file({"c": torch.zeros(2, 2, 2)},
                          str(root / "model-00002-of-00002.safetensors"))
    assert gr.parameters_from_safetensors(root) == 3 * 5 + 7 + 8

    qwen = gr.QwenDisambiguator(root, device="cpu")
    assert qwen.parameter_count() == 30      # never loaded; no weights exist to load
    assert gr.parameters_from_safetensors(tmp_path / "missing") == 0


# ==================== real config resolution (regression: the Colab TypeError) ================

DEFAULT_PROFILE = ROOT / "configs/pipeline/graphcent_0080.yaml"
RESEARCH_PROFILE = ROOT / "configs/pipeline/graphcent_0080_full_research.yaml"

#: Provenance a profile must never be able to change. Comparing these field by field after
#: resolution is what catches a constructor that silently drops a newly required field.
PROVENANCE_FIELDS = (
    "key", "repo_id", "licence", "licence_source", "pooling", "intended_domain",
    "expected_disk_gib",
)


def test_default_profile_resolves_through_the_real_cli_path() -> None:
    specs = cli.resolve_specs(cli.load_config(DEFAULT_PROFILE))
    by_key = {s.key: s for s in specs}
    assert set(by_key) == {"sapbert_xlmr", "biobert_mnli", "clinlinker_kb_gp"}
    assert by_key["sapbert_xlmr"].enabled
    assert by_key["clinlinker_kb_gp"].enabled
    assert not by_key["biobert_mnli"].enabled
    for spec in specs:
        assert spec.licence_source, f"{spec.key} lost its licence_source"
        assert spec.licence_source.startswith("https://huggingface.co/")

    # the disabled model is not an enabled downloadable model, and needs no override
    enabled = [s for s in specs if s.enabled]
    assert "biobert_mnli" not in {s.key for s in enabled}
    manifest = gm.ModelManifest(
        retrievers=specs,
        licence_overrides_accepted=tuple(
            cli.load_config(DEFAULT_PROFILE).get("licence_overrides_accepted") or ()
        ),
    )
    manifest.assert_licences_cleared()


def test_research_profile_resolves_with_its_recorded_override() -> None:
    config = cli.load_config(RESEARCH_PROFILE)
    specs = cli.resolve_specs(config)
    by_key = {s.key: s for s in specs}
    assert by_key["biobert_mnli"].enabled
    assert by_key["biobert_mnli"].licence_needs_review
    for spec in specs:
        assert spec.licence_source
    overrides = tuple(config.get("licence_overrides_accepted") or ())
    assert overrides == ("biobert_mnli",)
    gm.ModelManifest(
        retrievers=specs, licence_overrides_accepted=overrides
    ).assert_licences_cleared()


def test_enabling_biobert_without_the_override_still_fails_closed() -> None:
    config = cli.load_config(RESEARCH_PROFILE)
    config["licence_overrides_accepted"] = []          # the profile's override removed
    specs = cli.resolve_specs(config)
    assert {s.key for s in specs if s.enabled} == {
        "sapbert_xlmr", "biobert_mnli", "clinlinker_kb_gp"
    }
    with pytest.raises(gm.LicenceReviewRequired):
        gm.ModelManifest(retrievers=specs, licence_overrides_accepted=()).assert_licences_cleared()


def test_resolution_preserves_every_registry_field_it_does_not_own() -> None:
    """Generic guard: a profile may change `enabled` and `roles`, nothing else.

    This is the shape of the bug that broke the first Colab run - a resolver that rebuilt
    the spec field by field and dropped whatever it had not been taught about.
    """
    registry = {s.key: s for s in gm.DEFAULT_RETRIEVERS}
    for profile in (DEFAULT_PROFILE, RESEARCH_PROFILE):
        for spec in cli.resolve_specs(cli.load_config(profile)):
            source = registry[spec.key]
            for name in PROVENANCE_FIELDS:
                assert getattr(spec, name) == getattr(source, name), f"{spec.key}.{name}"


def test_every_required_spec_field_is_populated_after_resolution() -> None:
    """No required field may come back empty, whatever the registry grows next."""
    for profile in (DEFAULT_PROFILE, RESEARCH_PROFILE):
        for spec in cli.resolve_specs(cli.load_config(profile)):
            for f in dataclasses.fields(spec):
                if f.default is not dataclasses.MISSING:
                    continue                       # optional / resolved-at-runtime
                value = getattr(spec, f.name)
                assert value not in ("", None, ()), f"{spec.key}.{f.name} is empty"


def test_profile_may_not_override_provenance() -> None:
    config = {"retrievers": {"sapbert_xlmr": {"licence": "mit", "enabled": True}}}
    with pytest.raises(SystemExit, match="may not override"):
        cli.resolve_specs(config)


def test_profile_roles_override_is_still_honoured() -> None:
    specs = {s.key: s for s in cli.resolve_specs(cli.load_config(DEFAULT_PROFILE))}
    assert specs["clinlinker_kb_gp"].role == ("diagnosis",)


# ==================== CPU-safe preflight of the real download-models command ==================


class PreflightEncoder(FakeRetrieverEncoder):
    """Adapter-shaped fake: declares pooling, counts parameters without weights."""

    POOLING = ""

    def __init__(self, spec: gm.RetrieverSpec, model_root: Path, device: str = "cuda",
                 **kwargs: Any) -> None:
        super().__init__(spec, model_root, device, **kwargs)


def _preflight_adapters() -> dict[str, Any]:
    """One fake class per registry key, each declaring that key's registry pooling."""
    adapters: dict[str, Any] = {}
    for spec in gm.DEFAULT_RETRIEVERS:
        adapters[spec.key] = type(
            f"Preflight_{spec.key}", (PreflightEncoder,), {"POOLING": spec.pooling}
        )
    return adapters


def _run_download_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: Path
) -> dict[str, Any]:
    """Execute the real `download-models` body up to the download/GPU boundary.

    The hub is mocked at both call sites - resolution and download - so no network is
    touched and the SHA that acquisition resolves can be followed all the way to disk.
    """
    fetched: list[tuple[str, str]] = []

    def fake_snapshot_download(repo_id: str, revision: str = "",
                               local_dir: str = "", **kwargs: Any) -> str:
        fetched.append((repo_id, revision))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return local_dir

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(huggingface_hub, "model_info", fake_model_info)
    monkeypatch.setattr(cli, "require_host", lambda *a, **k: 24.0)
    monkeypatch.setattr(ge, "ENCODER_BY_KEY", _preflight_adapters())

    model_root = tmp_path / "models"
    code = cli.main([
        "--config", str(profile), "download-models", "--allow-download",
        "--model-root", str(model_root),
    ])
    assert code == 0
    manifest = json.loads((model_root / "model-manifest.json").read_text(encoding="utf-8"))
    manifest["_fetched"] = fetched
    return manifest


def test_preflight_download_models_default_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _run_download_models(monkeypatch, tmp_path, DEFAULT_PROFILE)

    # exactly the enabled models were fetched; the non-commercial one was never touched
    assert manifest["_fetched"] == [
        ("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
         hub_sha("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR")),
        ("ICB-UMA/ClinLinker-KB-GP", hub_sha("ICB-UMA/ClinLinker-KB-GP")),
    ]
    keys = {r["key"]: r for r in manifest["retrievers"]}
    deployed = {m["name"]: m for m in manifest["deployed"]}
    for key in ("sapbert_xlmr", "clinlinker_kb_gp"):
        sha = hub_sha(keys[key]["repo_id"])
        assert keys[key]["revision"] == sha
        assert manifest["resolved_revisions"][key] == sha
        assert deployed[keys[key]["repo_id"]]["revision"] == sha
    assert all(r["licence_source"] for r in keys.values())
    assert keys["biobert_mnli"]["enabled"] is False
    assert keys["biobert_mnli"]["parameter_count"] is None      # never measured, never loaded
    assert manifest["total_deployed_parameters"] == 222_000_000
    assert manifest["under_cap"] is True
    assert manifest["parameter_cap"] == gm.PARAMETER_CAP
    assert manifest["training_performed"] is False
    assert {m["name"] for m in manifest["deployed"]} == {
        "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR", "ICB-UMA/ClinLinker-KB-GP"
    }


def test_preflight_download_models_research_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _run_download_models(monkeypatch, tmp_path, RESEARCH_PROFILE)
    assert len(manifest["_fetched"]) == 3
    assert all(gm.is_pinned_revision(revision) for _, revision in manifest["_fetched"])
    assert manifest["licence_overrides_accepted"] == ["biobert_mnli"]
    assert all(r["licence_source"] for r in manifest["retrievers"])


def test_preflight_download_models_fails_closed_without_the_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = tmp_path / "no_override.yaml"
    import yaml

    config = yaml.safe_load(RESEARCH_PROFILE.read_text(encoding="utf-8"))
    config["licence_overrides_accepted"] = []
    profile.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(gm.LicenceReviewRequired):
        _run_download_models(monkeypatch, tmp_path, profile)


#: The exact command line that failed on the first real Colab run.
COLAB_DOWNLOAD_ARGV = [
    "download-models", "--allow-download",
    "--model-root", "/content/graphcent_models",
    "--e3-checkpoint", "checkpoint/e3_boundary_refinement_0062/best.pt",
    "--qwen-root", "/content/qwen3-8b-local",
]


def test_the_exact_colab_command_line_reaches_the_download_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Everything before the first network/GPU call must work on a laptop."""
    args = cli.build_parser().parse_args(["--config", str(DEFAULT_PROFILE), *COLAB_DOWNLOAD_ARGV])
    config = cli.load_config(args.config)
    specs = cli.resolve_specs(config)                 # this is where the TypeError was raised
    assert [s.key for s in specs if s.enabled] == ["sapbert_xlmr", "clinlinker_kb_gp"]
    runtime = cli._runtime_config(config, args)
    assert runtime.model_root == Path("/content/graphcent_models")
    assert args.func is cli.cmd_download_models


@pytest.mark.parametrize(
    "argv",
    [
        COLAB_DOWNLOAD_ARGV,
        ["build-index", "--cache-root", "/content/graphcent_cache"],
        ["smoke", "--input-dir", "data/organizer_test/input", "--run-dir", "runs/x",
         "--documents", "1,10,100"],
        ["run", "--input-dir", "data/organizer_test/input", "--run-dir", "runs/x"],
        ["package", "--run-dir", "runs/x", "--expected-documents", "100"],
    ],
)
def test_every_subcommand_resolves_its_prologue_on_cpu(argv: list[str]) -> None:
    """The shared prologue - parse, load profile, resolve specs, build runtime config -
    is what broke in Colab. Exercise it for every subcommand and both profiles."""
    for profile in (DEFAULT_PROFILE, RESEARCH_PROFILE):
        args = cli.build_parser().parse_args(["--config", str(profile), *argv])
        config = cli.load_config(args.config)
        specs = cli.resolve_specs(config)
        assert all(s.licence_source for s in specs)
        runtime = cli._runtime_config(config, args)
        assert runtime.top_k_per_retriever > 0
        assert runtime.max_candidate_context >= runtime.top_k_per_retriever


# ========================== revision pinning (regression: UNRESOLVED manifest) ================


def hub_sha(repo_id: str) -> str:
    """A deterministic stand-in for the SHA the hub would return for a repo."""
    import hashlib

    return hashlib.sha1(f"hub::{repo_id}".encode()).hexdigest()


class FakeInfo:
    def __init__(self, sha: str) -> None:
        self.sha = sha


def fake_model_info(repo_id: str, revision: str | None = None) -> FakeInfo:
    return FakeInfo(hub_sha(repo_id))


def test_revision_resolution_returns_an_immutable_commit_sha() -> None:
    spec = gm.DEFAULT_RETRIEVERS[0]
    sha = ga.resolve_revision(spec, info_fn=fake_model_info)
    assert sha == hub_sha(spec.repo_id)
    assert gm.is_pinned_revision(sha)
    assert len(sha) == 40


@pytest.mark.parametrize("returned", ["", "main", "refs/heads/main", "abc123", None])
def test_a_floating_or_empty_revision_is_refused(returned: str | None) -> None:
    def info(repo_id: str, revision: str | None = None) -> Any:
        return FakeInfo(returned)  # type: ignore[arg-type]

    with pytest.raises(gm.RevisionNotPinned, match="immutable commit SHA"):
        ga.resolve_revision(gm.DEFAULT_RETRIEVERS[0], info_fn=info)


def test_a_hub_failure_is_a_refusal_not_a_fallback() -> None:
    def info(repo_id: str, revision: str | None = None) -> Any:
        raise OSError("no network")

    with pytest.raises(gm.RevisionNotPinned, match="could not resolve"):
        ga.resolve_revision(gm.DEFAULT_RETRIEVERS[0], info_fn=info)


def test_an_already_pinned_revision_is_reused_and_the_hub_is_not_asked() -> None:
    asked: list[str] = []

    def info(repo_id: str, revision: str | None = None) -> Any:
        asked.append(repo_id)
        return FakeInfo(hub_sha(repo_id))

    existing = fake_sha("sapbert_xlmr")
    assert ga.resolve_revision(
        gm.DEFAULT_RETRIEVERS[0], cached=existing, info_fn=info
    ) == existing
    assert asked == []                      # no floating HEAD lookup on a repeat run


def test_pinned_revisions_are_read_back_from_an_existing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    (root / ga.MANIFEST_NAME).write_text(json.dumps({
        "retrievers": [
            {"key": "sapbert_xlmr", "revision": fake_sha("sapbert_xlmr")},
            {"key": "biobert_mnli", "revision": ""},            # not pinned -> ignored
            {"key": "clinlinker_kb_gp", "revision": "main"},    # floating -> ignored
        ],
        "resolved_revisions": {"clinlinker_kb_gp": fake_sha("clinlinker_kb_gp")},
    }), encoding="utf-8")
    recorded = ga.read_pinned_revisions(root)
    assert recorded == {
        "sapbert_xlmr": fake_sha("sapbert_xlmr"),
        "clinlinker_kb_gp": fake_sha("clinlinker_kb_gp"),
    }
    assert ga.read_pinned_revisions(tmp_path / "absent") == {}


def test_conflicting_existing_pins_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    (root / ga.MANIFEST_NAME).write_text(json.dumps({
        "retrievers": [{"key": "sapbert_xlmr", "revision": PIN_A}],
        "resolved_revisions": {"sapbert_xlmr": PIN_B},
    }), encoding="utf-8")
    with pytest.raises(gm.RevisionNotPinned, match="conflicting pinned revisions"):
        ga.read_pinned_revisions(root)


def test_acquire_downloads_at_the_resolved_sha_and_skips_disabled_models(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, str]] = []

    def download(repo_id: str, revision: str, local_dir: str) -> str:
        calls.append({"repo_id": repo_id, "revision": revision, "local_dir": local_dir})
        return local_dir

    specs = ga.acquire(
        list(gm.DEFAULT_RETRIEVERS), tmp_path / "models",
        info_fn=fake_model_info, download_fn=download,
    )
    by_key = {s.key: s for s in specs}
    assert by_key["sapbert_xlmr"].revision == hub_sha(by_key["sapbert_xlmr"].repo_id)
    assert by_key["clinlinker_kb_gp"].revision == hub_sha(by_key["clinlinker_kb_gp"].repo_id)
    assert by_key["biobert_mnli"].revision == ""        # disabled: never resolved or fetched
    assert [c["repo_id"] for c in calls] == [
        by_key["sapbert_xlmr"].repo_id, by_key["clinlinker_kb_gp"].repo_id
    ]
    for call in calls:
        assert gm.is_pinned_revision(call["revision"])


def test_acquire_reuses_the_recorded_pin_on_a_second_run(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    (root / ga.MANIFEST_NAME).write_text(json.dumps({
        "resolved_revisions": {"sapbert_xlmr": fake_sha("sapbert_xlmr")},
    }), encoding="utf-8")
    asked: list[str] = []

    def info(repo_id: str, revision: str | None = None) -> Any:
        asked.append(repo_id)
        return FakeInfo(hub_sha(repo_id))

    specs = ga.acquire(
        [gm.DEFAULT_RETRIEVERS[0]], root, info_fn=info, download_fn=lambda **k: "",
    )
    assert specs[0].revision == fake_sha("sapbert_xlmr")
    assert asked == []


def test_manifest_refuses_to_be_written_without_pinned_revisions() -> None:
    unpinned = dataclasses.replace(gm.DEFAULT_RETRIEVERS[0], revision="")
    with pytest.raises(gm.RevisionNotPinned, match="no immutable revision"):
        gm.ModelManifest(retrievers=[unpinned]).assert_revisions_pinned()

    floating = dataclasses.replace(gm.DEFAULT_RETRIEVERS[0], revision="main")
    with pytest.raises(gm.RevisionNotPinned):
        gm.ModelManifest(retrievers=[floating]).assert_revisions_pinned()

    # a disabled retriever was never acquired, so it has nothing to pin
    gm.ModelManifest(
        retrievers=[dataclasses.replace(unpinned, enabled=False)]
    ).assert_revisions_pinned()

    with pytest.raises(gm.RevisionNotPinned):
        gm.ModelManifest(
            deployed=[gm.DeployedModel("Qwen/Qwen3-8B", 8_190_735_360)]
        ).assert_revisions_pinned()


def test_local_checkpoint_identity_is_a_labelled_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    digest = ga.file_digest(checkpoint)
    assert digest.startswith(gm.LOCAL_DIGEST_PREFIX)
    assert gm.is_pinned_revision(digest)
    assert gm.is_local_checkpoint_revision(digest)
    assert not gm.is_pinned_revision(gm.LOCAL_DIGEST_PREFIX)      # prefix alone proves nothing
    assert not gm.is_pinned_revision(f"{gm.LOCAL_DIGEST_PREFIX}{'g' * 64}")
    assert not gm.is_pinned_revision(f"{gm.LOCAL_DIGEST_PREFIX}{'a' * 63}")
    assert not gm.is_local_checkpoint_revision(PIN_A)             # hub SHA != local digest


def test_cache_manifest_records_the_same_sha_as_the_deployed_model(tmp_path: Path) -> None:
    _, _, documents = _tiny_kb()
    config = _config(tmp_path)
    spec = pinned([gm.DEFAULT_RETRIEVERS[0]])[0]
    index = gr.build_index_for(spec, documents, config, encoder_factory=FakeRetrieverEncoder)
    assert index.manifest.revision == spec.revision
    assert index.manifest.model_repo == spec.repo_id

    written = json.loads(
        gr.cache_path(config.cache_root, spec.key)
        .with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert written["revision"] == spec.revision


def test_an_index_built_by_other_weights_is_refused(tmp_path: Path) -> None:
    _, _, documents = _tiny_kb()
    config = _config(tmp_path)
    spec = pinned([gm.DEFAULT_RETRIEVERS[0]])[0]
    gr.build_index_for(spec, documents, config, encoder_factory=FakeRetrieverEncoder)
    moved = dataclasses.replace(spec, revision=fake_sha("some other snapshot"))
    with pytest.raises(CacheMismatch, match="cache built at revision"):
        gr.load_index(moved, documents, config)


class UnpinnedEncoder(FakeRetrieverEncoder):
    """Reports no revision, exactly like a real encoder loaded from a local snapshot dir."""

    def __init__(self, spec: gm.RetrieverSpec, model_root: Path, device: str = "cuda",
                 **kwargs: Any) -> None:
        super().__init__(spec, model_root, device, **kwargs)
        self.resolved_revision = ""


def test_an_unpinned_spec_cannot_build_an_index(tmp_path: Path) -> None:
    """The exact Colab situation: weights on disk, no revision anyone can name."""
    _, _, documents = _tiny_kb()
    with pytest.raises(gm.RevisionNotPinned, match="refusing to build a cache"):
        gr.build_index_for(
            gm.DEFAULT_RETRIEVERS[0], documents, _config(tmp_path),
            encoder_factory=UnpinnedEncoder,
        )


def test_downstream_stages_refuse_to_run_before_download_models(tmp_path: Path) -> None:
    with pytest.raises(gm.RevisionNotPinned, match="Run download-models first"):
        cli.pin_specs(cli.resolve_specs(cli.load_config(DEFAULT_PROFILE)), tmp_path)


def test_downstream_stages_adopt_the_recorded_pin(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    (root / ga.MANIFEST_NAME).write_text(json.dumps({
        "resolved_revisions": {
            "sapbert_xlmr": fake_sha("sapbert_xlmr"),
            "clinlinker_kb_gp": fake_sha("clinlinker_kb_gp"),
        },
    }), encoding="utf-8")
    specs = {s.key: s for s in cli.pin_specs(
        cli.resolve_specs(cli.load_config(DEFAULT_PROFILE)), root
    )}
    assert specs["sapbert_xlmr"].revision == fake_sha("sapbert_xlmr")
    assert specs["clinlinker_kb_gp"].revision == fake_sha("clinlinker_kb_gp")
    assert specs["biobert_mnli"].revision == ""       # disabled, nothing to pin


def test_no_unresolved_placeholder_survives_anywhere_in_graphcent() -> None:
    paths = [
        *sorted((ROOT / "src/mednorm_vi/graphcent").glob("*.py")),
        ROOT / "scripts/graphcent_0080.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "UNRESOLVED" not in text, path.name
        assert 'revision="main"' not in text, path.name
        assert "revision='main'" not in text, path.name
