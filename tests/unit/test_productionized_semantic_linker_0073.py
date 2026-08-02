"""Production wiring of the frozen S1 semantic linker (Audit 0073 §8).

Model-free: no Qwen weights, no CUDA, no network. The semantic backend is exercised through a
scripted reranker so the production contract - governed codes only, dotted-at-serialization,
shadow NULL, deterministic order, recorded revisions - is testable on any machine.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from mednorm_vi.inference.config import PipelineConfig, _linker_backend, evaluate_readiness
from mednorm_vi.kb.indexing.retrieval import LocalIndex
from mednorm_vi.linking.semantic import hybrid as hy
from mednorm_vi.linking.semantic import runtime as rt

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "configs/pipeline/full_v1.yaml"
ABLATION = ROOT / "configs/pipeline/semantic_s1_0073.yaml"


def _index(titles: dict[str, str], **postings: Any) -> LocalIndex:
    payload: dict[str, Any] = {
        "index_type": "icd10_vi",
        "source_snapshot_id": "test-snapshot",
        "records": {c: {"concept_id": c, "canonical_name": n} for c, n in titles.items()},
        "exact": {},
        "exact_ascii": {},
        "ngrams": {},
        "sparse_terms": {},
        "graph": {},
    }
    payload.update(postings)
    return LocalIndex(**payload)


class ScriptedReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [self.scores.get(d.split("|")[0].strip(), 0.0) for d in documents]

    def unload(self) -> None:
        return None


# ------------------------------------------------------------------- backend selection


def test_baseline_profile_still_resolves_to_the_lexical_linker() -> None:
    """The 11.9188 path must be unaffected by this milestone."""
    config = PipelineConfig.load(BASELINE)
    assert config.linker_backend == rt.BACKEND_LEXICAL_V3
    assert config.semantic_linker == {}


def test_semantic_backend_can_be_selected() -> None:
    config = PipelineConfig.load(ABLATION)
    assert config.linker_backend == rt.BACKEND_SEMANTIC_S1


def test_ablation_differs_from_baseline_in_exactly_two_fields() -> None:
    baseline = PipelineConfig.load(BASELINE)
    ablation = PipelineConfig.load(ABLATION)
    differing = {
        f.name
        for f in dataclasses.fields(ablation)
        if getattr(ablation, f.name) != getattr(baseline, f.name)
    }
    assert differing == {"linker_backend", "semantic_linker"}
    # The v3 lexical anchor is retained, not swapped for v4.1.
    assert ablation.icd_index == baseline.icd_index
    assert "competition-v3" in ablation.icd_index


@pytest.mark.parametrize("value", ["bogus", "semantic", "lexical", "S1", ""])
def test_unknown_backend_is_refused_not_defaulted(value: str) -> None:
    with pytest.raises(ValueError, match="unknown linker_backend"):
        _linker_backend({"linker_backend": value})


def test_baseline_yaml_declares_no_backend_and_stays_the_rollback() -> None:
    assert "linker_backend" not in BASELINE.read_text(encoding="utf-8")


# ------------------------------------------------------------------- readiness contract


def test_semantic_backend_fails_closed_when_models_are_missing() -> None:
    """A run that claims semantic and quietly used lexical would invalidate the ablation."""
    config = PipelineConfig.load(ABLATION)
    readiness = evaluate_readiness(config, mode="specialist")
    assert readiness.errors
    assert any("semantic_linker_not_ready" in e for e in readiness.errors)


def test_missing_requirements_names_every_missing_piece() -> None:
    problems = rt.missing_requirements(rt.SemanticLinkerSettings())
    joined = " ".join(problems)
    for expected in ("model_root", "documents", "dense_index", "icd_v41_index"):
        assert expected in joined


def test_baseline_readiness_is_unaffected() -> None:
    readiness = evaluate_readiness(PipelineConfig.load(BASELINE), mode="specialist")
    assert not [e for e in readiness.errors if "semantic" in e]


def test_null_mode_accepts_shadow_and_conservative_and_refuses_anything_else() -> None:
    """Audit 0073 pinned this to shadow; the 0075 sprint enables the conservative gate.

    Both are governed modes. An unrecognised value is still refused, so a profile can never
    silently run a policy it does not name.
    """
    for mode in ("shadow", "conservative"):
        settings = rt.SemanticLinkerSettings(null_mode=mode)
        assert not [p for p in rt.missing_requirements(settings) if "null_mode" in p], mode
    bogus = rt.SemanticLinkerSettings(null_mode="aggressive")
    assert [p for p in rt.missing_requirements(bogus) if "null_mode" in p]


def test_model_root_env_override_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rt.MODEL_ROOT_ENV, "/somewhere/models")
    assert rt.SemanticLinkerSettings.from_mapping({}).model_root == "/somewhere/models"


def test_no_user_specific_path_is_a_repository_default() -> None:
    text = ABLATION.read_text(encoding="utf-8")
    assert 'model_root: ""' in text
    for leaked in ("/content/drive/MyDrive/MedNorm-VI/0072/models", "/home/vquclinh"):
        assert leaked not in rt.SemanticLinkerSettings().model_root


# --------------------------------------------------------------- governed output contract


def _rx_index(names: dict[str, str], **postings: Any) -> LocalIndex:
    payload: dict[str, Any] = {
        "index_type": "rxnorm",
        "source_snapshot_id": "rx-snapshot",
        "records": {
            c: {"concept_id": c, "canonical_name": n, "metadata": {"membership": "searchable"}}
            for c, n in names.items()
        },
        "exact": {},
        "exact_ascii": {},
        "ngrams": {},
        "sparse_terms": {},
        "graph": {},
    }
    payload.update(postings)
    return LocalIndex(**payload)


def _runtime(
    scores: dict[str, float],
    icd_titles: dict[str, str],
    rx_names: dict[str, str] | None = None,
) -> rt.SemanticLinkerRuntime:
    rx_names = rx_names or {"1191": "aspirin"}
    icd = _index(icd_titles, exact={"m": [next(iter(icd_titles))]})
    rx = _rx_index(rx_names, exact={"m": [next(iter(rx_names))]})
    runtime = rt.SemanticLinkerRuntime(
        settings=rt.SemanticLinkerSettings(null_mode=hy.NULL_MODE_SHADOW),
        icd_v3_index=icd,
        icd_v41_index=icd,
        rxnorm_index=rx,
    )
    runtime._documents = {
        **{c: f"{c} | {t}" for c, t in icd_titles.items()},
        **{c: f"{c} | {n}" for c, n in rx_names.items()},
    }
    runtime._ids_by_ontology = {"ICD10": list(icd_titles), "RXNORM": list(rx_names)}
    runtime._rows_by_ontology = {
        "ICD10": list(range(len(icd_titles))),
        "RXNORM": list(range(len(icd_titles), len(icd_titles) + len(rx_names))),
    }
    runtime._backend = ScriptedReranker(scores)
    return runtime


def _linker(scores: dict[str, float], titles: dict[str, str]) -> rt.SemanticLinkerRuntime:
    return _runtime(scores, titles)


class _Hypothesis:
    hypothesis_id = "h1"
    text = "m"


def test_only_governed_concept_ids_are_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    linker = _linker({"A00": 0.9, "A01": 0.5}, {"A00": "a", "A01": "b"})
    monkeypatch.setattr(
        rt.SemanticLinkerRuntime,
        "_dense_hits",
        lambda self, m, o: [("A00", 0.9), ("NOT_IN_KB", 0.99)],
    )
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "load", lambda self: None)
    codes = [c.code for c in linker.link_icd10(_Hypothesis()).candidates]
    assert "NOT_IN_KB" not in codes
    assert set(codes) <= {"A00", "A01"}


def test_emitted_codes_are_dotless_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dotted form is applied only by derive_submission at serialization."""
    linker = _linker({"A000": 0.9}, {"A000": "a"})
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "_dense_hits", lambda self, m, o: [("A000", 0.9)])
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "load", lambda self: None)
    for candidate in linker.link_icd10(_Hypothesis()).candidates:
        assert "." not in candidate.code


def test_null_shadow_does_not_alter_the_emitted_candidate_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linker = _linker({"A00": 0.001, "A01": 0.0005}, {"A00": "a", "A01": "b"})
    monkeypatch.setattr(
        rt.SemanticLinkerRuntime, "_dense_hits", lambda self, m, o: [("A00", 0.1), ("A01", 0.05)]
    )
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "load", lambda self: None)
    result = linker.link_icd10(_Hypothesis())
    assert result.candidates  # gate may fire, output is unchanged
    assert linker.shadow_records[0]["null_decision"]["decision"]


def test_ordering_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rt.SemanticLinkerRuntime, "_dense_hits", lambda self, m, o: [("A00", 0.9), ("A01", 0.9)]
    )
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "load", lambda self: None)
    runs = []
    for _ in range(5):
        linker = _linker({"A00": 0.5, "A01": 0.5}, {"A00": "a", "A01": "b"})
        runs.append([c.code for c in linker.link_icd10(_Hypothesis()).candidates])
    assert all(r == runs[0] for r in runs)


def test_result_records_the_backend_and_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    linker = _linker({"A00": 0.9}, {"A00": "a"})
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "_dense_hits", lambda self, m, o: [("A00", 0.9)])
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "load", lambda self: None)
    candidate = linker.link_icd10(_Hypothesis()).candidates[0]
    assert candidate.snapshot_id == "test-snapshot"
    assert f"backend:{rt.BACKEND_SEMANTIC_S1}" in candidate.evidence


# ------------------------------------------------------- manifest, budget, isolation


def test_manifest_records_exact_frozen_revisions() -> None:
    manifest = rt.model_manifest()
    revisions = {c["model"]: c["revision"] for c in manifest["components"]}
    assert revisions["Qwen/Qwen3-Embedding-4B"] == "5cf2132abc99cad020ac570b19d031efec650f2b"
    assert revisions["Qwen/Qwen3-Reranker-4B"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    for revision in revisions.values():
        assert len(revision) == 40 and re.fullmatch(r"[0-9a-f]{40}", revision)
    assert manifest["training_performed"] is False
    assert manifest["null_mode"] == hy.NULL_MODE_SHADOW


def test_deployed_parameters_stay_under_nine_billion() -> None:
    total = rt.s1_deployed_parameters()
    assert total == 8_178_561_029
    assert total < rt.PARAMETER_CAP
    assert rt.model_manifest()["under_cap"] is True


def test_semantic_backend_routes_both_ontologies() -> None:
    """Frozen S1 was a TWO-ontology system - its benchmark was ~197 ICD / ~1,803 RxNorm."""
    source = (ROOT / "src/mednorm_vi/inference/pipeline.py").read_text(encoding="utf-8")
    assert 'h.entity_type == "CHẨN_ĐOÁN" and semantic is not None' in source
    assert "semantic.link_icd10(h)" in source
    assert 'h.entity_type == "THUỐC" and semantic is not None' in source
    assert "semantic.link_rxnorm(h)" in source


def test_lexical_backend_leaves_both_ontology_linkers_unchanged() -> None:
    source = (ROOT / "src/mednorm_vi/inference/pipeline.py").read_text(encoding="utf-8")
    assert "link_icd10(h, icd_index, hierarchy_policy=hierarchy_policy)" in source
    assert "link_rxnorm(h, rxnorm_index)" in source
    baseline = PipelineConfig.load(BASELINE)
    from mednorm_vi.inference.pipeline import semantic_linker_runtime

    assert semantic_linker_runtime(baseline, None, None) is None


def test_semantic_rxnorm_is_governed_by_the_frozen_rxnorm_kb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime({"1191": 0.9, "161": 0.4}, {"A00": "a"}, {"1191": "aspirin", "161": "x"})
    monkeypatch.setattr(
        rt.SemanticLinkerRuntime, "_dense_hits", lambda self, m, o: [("1191", 0.9), ("161", 0.5)]
    )
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "load", lambda self: None)
    result = runtime.link_rxnorm(_Hypothesis())
    assert [c.code for c in result.candidates] == ["1191", "161"]
    assert result.candidates[0].snapshot_id == "rx-snapshot"
    assert "ontology:RXNORM" in result.candidates[0].evidence


def test_no_out_of_kb_rxcui_can_be_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime({"1191": 0.5, "999999": 0.99}, {"A00": "a"}, {"1191": "aspirin"})
    monkeypatch.setattr(
        rt.SemanticLinkerRuntime,
        "_dense_hits",
        lambda self, m, o: [("1191", 0.5), ("999999", 0.99)],
    )
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "load", lambda self: None)
    codes = [c.code for c in runtime.link_rxnorm(_Hypothesis()).candidates]
    assert codes == ["1191"]


def test_h2_is_never_applied_to_rxnorm(monkeypatch: pytest.MonkeyPatch) -> None:
    """H2 is an ICD family rule; `run_hybrid` gates it on ontology, so this is structural."""
    runtime = _runtime({"1191": 0.9, "161": 0.4}, {"A00": "a"}, {"1191": "aspirin", "161": "x"})
    monkeypatch.setattr(
        rt.SemanticLinkerRuntime, "_dense_hits", lambda self, m, o: [("1191", 0.9), ("161", 0.4)]
    )
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "load", lambda self: None)
    runtime.link_rxnorm(_Hypothesis())
    assert runtime.shadow_records[-1]["hierarchy_applied"] is False
    assert runtime.shadow_records[-1]["ontology"] == "RXNORM"


def test_production_wrapper_matches_the_0072_hybrid_for_both_ontologies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper must not implement a different algorithm from the frozen 0072 runtime."""
    scores = {"A00": 0.4, "A01": 0.8, "1191": 0.9, "161": 0.3}
    icd_titles = {"A00": "a", "A01": "b"}
    rx_names = {"1191": "aspirin", "161": "x"}
    icd_dense = [("A00", 0.7), ("A01", 0.6)]
    rx_dense = [("1191", 0.9), ("161", 0.5)]

    runtime = _runtime(scores, icd_titles, rx_names)
    monkeypatch.setattr(
        rt.SemanticLinkerRuntime,
        "_dense_hits",
        lambda self, m, o: icd_dense if o == "ICD10" else rx_dense,
    )
    monkeypatch.setattr(rt.SemanticLinkerRuntime, "load", lambda self: None)

    for ontology, dense, governed, produced in (
        ("ICD10", icd_dense, runtime.icd_v41_index, runtime.link_icd10(_Hypothesis())),
        ("RXNORM", rx_dense, runtime.rxnorm_index, runtime.link_rxnorm(_Hypothesis())),
    ):
        # Same inputs, driven directly through the 0072 implementation.
        pool = hy.build_pool(
            "m",
            v3_index=runtime.icd_v3_index if ontology == "ICD10" else runtime.rxnorm_index,
            v41_index=runtime.icd_v41_index if ontology == "ICD10" else None,
            dense_hits=dense,
            governed_index=governed,
            v3_topk=20 if ontology == "ICD10" else runtime.settings.rxnorm_topk,
            v41_topk=20,
            dense_topk=50,
        )
        expected = hy.run_hybrid(
            "m",
            "m",
            pool,
            {c.concept_id: runtime._documents[c.concept_id] for c in pool},
            ScriptedReranker(scores),
            ontology=ontology,
            icd_index=runtime.icd_v41_index if ontology == "ICD10" else None,
            governed_index=governed,
            null_mode=hy.NULL_MODE_SHADOW,
            limit=runtime.settings.candidate_limit,
        )
        assert [c.code for c in produced.candidates] == list(expected.codes), ontology


def test_no_network_call_in_the_inference_path() -> None:
    for relative in (
        "src/mednorm_vi/linking/semantic/runtime.py",
        "src/mednorm_vi/linking/semantic/hybrid.py",
        "src/mednorm_vi/linking/semantic/reranker.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        for banned in ("requests.", "urllib.request", "httpx.", "socket.", "http://", "https://"):
            assert banned not in text, f"{relative} references {banned}"


def test_public_inference_script_does_not_submit() -> None:
    text = (ROOT / "scripts/run_public_inference_0073.py").read_text(encoding="utf-8")
    assert "NOT submitted" in text
    assert "specialist" in text  # the scored baseline's mode
    for banned in ("upload", "submit_to", "post("):
        assert banned not in text.lower().replace("not submitted", "")


def test_shipped_parameter_manifest_agrees_if_present() -> None:
    path = ROOT / "artifacts/semantic/0072/deployed_parameter_manifest.json"
    if not path.is_file():  # pragma: no cover - artifact is gitignored
        pytest.skip("parameter manifest not built in this checkout")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["systems"]["S1"]["total_deployed_parameters"] == rt.s1_deployed_parameters()
