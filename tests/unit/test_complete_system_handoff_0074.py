"""GPU-handoff invariants for the Audit-0074 complete-system A/B.

No Qwen weights, no CUDA, no network. The encoder is monkeypatched so the row-order,
slice-identity and atomicity contracts are exercised end to end on CPU with tiny tensors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from mednorm_vi.linking.semantic import runtime as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_0074_complete_system_colab.py"

spec = importlib.util.spec_from_file_location("handoff_0074", SCRIPT)
assert spec and spec.loader
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)

DIM = 4


def _fake_index() -> Any:
    from mednorm_vi.kb.indexing.retrieval import LocalIndex

    return LocalIndex(
        index_type="rxnorm",
        source_snapshot_id="t",
        records={},
        exact={},
        exact_ascii={},
        ngrams={},
        sparse_terms={},
        graph={},
    )


def corpus(icd: int = 3, rx: int = 5) -> list[dict[str, Any]]:
    """ICD-first file order, exactly like the frozen corpus."""
    rows = [
        {"concept_id": f"A{i:02d}", "ontology": "ICD10", "text": f"icd {i}"} for i in range(icd)
    ]
    rows += [
        {"concept_id": str(1000 + i), "ontology": "RXNORM", "text": f"rx {i}"} for i in range(rx)
    ]
    return rows


def write(tmp: Path, rows: list[dict[str, Any]], name: str) -> Path:
    path = tmp / name
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def make_args(tmp: Path, frozen_docs: Path, frozen_vectors: Path, **kwargs: Any) -> Any:
    base = {
        "frozen_documents": frozen_docs,
        "frozen_vectors": frozen_vectors,
        "out": tmp / "out",
        "structured": tmp / "structured.jsonl",
        "row_order": rt.FROZEN_0072_ROW_ORDER,
        "device": "cpu",
        "embed_batch_size": 2,
        "overwrite": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


# ------------------------------------------------------------------- row-order contract


def test_frozen_row_order_is_sorted_concept_id_not_file_order() -> None:
    """The defect Audit 0074 found: the two orders differ and are both contiguous."""
    rows = corpus()
    file_order = handoff.ordered_ids(rows, rt.ROW_ORDER_DOCUMENTS_FILE)
    sorted_order = handoff.ordered_ids(rows, rt.FROZEN_0072_ROW_ORDER)
    assert file_order != sorted_order
    assert file_order[0].startswith("A")  # file order is ICD-first
    assert sorted_order[0].isdigit()  # sorted order is RxNorm-first


def test_runtime_uses_the_declared_row_order() -> None:
    assert rt.FROZEN_0072_ROW_ORDER == rt.ROW_ORDER_SORTED_CONCEPT_ID
    assert rt.SemanticLinkerSettings().row_order == rt.FROZEN_0072_ROW_ORDER


def test_unknown_row_order_is_refused() -> None:
    problems = rt.missing_requirements(rt.SemanticLinkerSettings(row_order="whatever"))
    assert any("row_order" in p for p in problems)


# ------------------------------------------------- historical 0073 reproduction (opt-in)


def _row_map(rows: list[dict[str, Any]], row_order: str) -> dict[str, list[int]]:
    """The ontology -> vector-row mapping the runtime derives for a given order."""
    ontology_of = {r["concept_id"]: r["ontology"] for r in rows}
    if row_order == rt.ROW_ORDER_SORTED_CONCEPT_ID:
        ordered = sorted(ontology_of)
    else:
        ordered = [r["concept_id"] for r in rows]
    out: dict[str, list[int]] = {}
    for position, concept_id in enumerate(ordered):
        out.setdefault(ontology_of[concept_id], []).append(position)
    return out


def test_default_path_uses_sorted_concept_id_order() -> None:
    """The corrected order is the only default, everywhere it can be set."""
    assert rt.SemanticLinkerSettings().row_order == rt.ROW_ORDER_SORTED_CONCEPT_ID
    assert rt.SemanticLinkerSettings.from_mapping({}).row_order == rt.ROW_ORDER_SORTED_CONCEPT_ID
    assert rt.FROZEN_0072_ROW_ORDER == rt.ROW_ORDER_SORTED_CONCEPT_ID
    rows = corpus()
    assert _row_map(rows, rt.SemanticLinkerSettings().row_order)["ICD10"][0] == len(
        [r for r in rows if r["ontology"] == "RXNORM"]
    )


def test_legacy_mode_reproduces_the_old_misaligned_mapping() -> None:
    """The Audit-0073 defect: ICD positions 0..N-1 into a RxNorm-first cache."""
    rows = corpus()
    legacy = _row_map(rows, rt.ROW_ORDER_LEGACY_0073_MISALIGNED)
    icd_count = len([r for r in rows if r["ontology"] == "ICD10"])
    # Exactly what the pre-fix code produced: file-order positions.
    assert legacy["ICD10"] == list(range(icd_count))
    assert legacy["RXNORM"] == list(range(icd_count, len(rows)))
    # And it genuinely differs from the corrected mapping.
    assert legacy != _row_map(rows, rt.ROW_ORDER_SORTED_CONCEPT_ID)


def test_legacy_mode_is_never_selected_silently() -> None:
    assert rt.ROW_ORDER_LEGACY_0073_MISALIGNED in rt.HISTORICAL_ROW_ORDERS
    assert rt.SemanticLinkerSettings().row_order not in rt.HISTORICAL_ROW_ORDERS
    # Only an explicit value reaches it; an empty profile never does.
    assert rt.SemanticLinkerSettings.from_mapping({}).row_order not in rt.HISTORICAL_ROW_ORDERS
    explicit = rt.SemanticLinkerSettings.from_mapping(
        {"row_order": rt.ROW_ORDER_LEGACY_0073_MISALIGNED}
    )
    assert explicit.row_order == rt.ROW_ORDER_LEGACY_0073_MISALIGNED
    assert not [p for p in rt.missing_requirements(explicit) if "row_order" in p]


def test_legacy_mode_warns_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = corpus(icd=rt.EXPECTED_ICD_ROWS, rx=rt.EXPECTED_RXNORM_ROWS)
    documents = write(tmp_path, rows, "documents.jsonl")
    vectors_path = tmp_path / "v.pt"
    torch.save(torch.zeros(len(rows), DIM), vectors_path)
    settings = rt.SemanticLinkerSettings(
        model_root=str(tmp_path),
        documents=str(documents),
        dense_index=str(vectors_path),
        icd_v41_index=str(documents),
        rxnorm_index=str(documents),
        device="cpu",
        row_order=rt.ROW_ORDER_LEGACY_0073_MISALIGNED,
    )
    for name in ("emb-4b", "rr-4b"):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / "config.json").write_text("{}", encoding="utf-8")
    index = _fake_index()
    runtime = rt.SemanticLinkerRuntime(
        settings=settings, icd_v3_index=index, icd_v41_index=index, rxnorm_index=index
    )
    monkeypatch.setattr(rt, "Qwen3Reranker", lambda *a, **k: object())
    runtime.load()
    assert "reproduces the Audit-0073 dense misalignment" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["sorted", "file", "", "LEGACY", "documents_file"])
def test_unknown_row_order_fails_closed(value: str) -> None:
    problems = rt.missing_requirements(rt.SemanticLinkerSettings(row_order=value))
    assert any("row_order" in p for p in problems)


# --------------------------------------------------------- rebuild: identity + atomicity


def _prepare(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, torch.Tensor, list[str]]:
    rows = corpus()
    frozen_docs = write(tmp, rows, "documents.jsonl")
    order = handoff.ordered_ids(rows, rt.FROZEN_0072_ROW_ORDER)
    frozen = torch.arange(len(order) * DIM, dtype=torch.float32).reshape(len(order), DIM)
    frozen_vectors = tmp / "S1_doc_vectors.pt"
    torch.save(frozen, frozen_vectors)

    structured = [
        {
            "rxcui": r["concept_id"],
            "tty": "SCD",
            "name": f"drug {r['concept_id']}",
            "ingredients": ["amoxicillin"],
            "strengths": [{"value": "500", "unit": "mg", "raw": "500 MG", "key": "500mg"}],
            "dose_forms": ["Oral Tablet"],
            "brands": [],
            "precise_ingredients": [],
            "available_strength": "",
            "rxterm_form": "",
            "quantity": "",
            "human_drug": True,
            "provenance": [],
        }
        for r in rows
        if r["ontology"] == "RXNORM"
    ]
    (tmp / "structured.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in structured), encoding="utf-8"
    )
    args = make_args(tmp, frozen_docs, frozen_vectors)
    args.out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        handoff,
        "encode",
        lambda texts, a, label: torch.full((len(texts), DIM), -7.0, dtype=torch.float32),
    )
    return args, frozen, order


def test_only_the_rxnorm_slice_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, frozen, order = _prepare(tmp_path, monkeypatch)
    handoff.build_docs(args)
    target = handoff.rebuild_rxnorm_vectors(args)
    rebuilt = torch.load(target, map_location="cpu")

    rows = handoff.read_documents(args.frozen_documents)
    ontology_of = {r["concept_id"]: r["ontology"] for r in rows}
    icd_positions = [i for i, c in enumerate(order) if ontology_of[c] == "ICD10"]
    rx_positions = [i for i, c in enumerate(order) if ontology_of[c] == "RXNORM"]

    assert torch.equal(rebuilt[icd_positions], frozen[icd_positions])  # ICD identical
    assert torch.all(rebuilt[rx_positions] == -7.0)  # RxNorm replaced
    assert rebuilt.shape == frozen.shape


def test_icd_document_text_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _ = _prepare(tmp_path, monkeypatch)
    out = handoff.build_docs(args)
    before = {r["concept_id"]: r for r in handoff.read_documents(args.frozen_documents)}
    after = {r["concept_id"]: r for r in handoff.read_documents(out)}
    for concept_id, row in before.items():
        if row["ontology"] == "ICD10":
            assert after[concept_id] == row
        else:
            assert after[concept_id]["text"] != row["text"]


def test_frozen_cache_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _, _ = _prepare(tmp_path, monkeypatch)
    frozen_before = args.frozen_vectors.read_bytes()
    handoff.build_docs(args)
    target = handoff.rebuild_rxnorm_vectors(args)
    assert target != args.frozen_vectors
    assert target.name == "S1_structured_rxnorm_doc_vectors.pt"
    assert args.frozen_vectors.read_bytes() == frozen_before


def test_finalization_is_atomic_and_leaves_no_temporaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _ = _prepare(tmp_path, monkeypatch)
    handoff.build_docs(args)
    target = handoff.rebuild_rxnorm_vectors(args)
    assert target.is_file()
    assert target.with_suffix(".manifest.json").is_file()
    assert not list(target.parent.glob("*.tmp"))


def test_existing_cache_is_not_replaced_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _ = _prepare(tmp_path, monkeypatch)
    handoff.build_docs(args)
    handoff.rebuild_rxnorm_vectors(args)
    with pytest.raises(SystemExit, match="already exists"):
        handoff.rebuild_rxnorm_vectors(args)


def test_manifest_records_the_full_integrity_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _ = _prepare(tmp_path, monkeypatch)
    handoff.build_docs(args)
    target = handoff.rebuild_rxnorm_vectors(args)
    manifest = json.loads(target.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    for key in (
        "row_order",
        "document_count",
        "icd_rows",
        "rxnorm_rows",
        "frozen_documents_sha256",
        "structured_documents_sha256",
        "frozen_vectors_sha256",
        "frozen_icd_slice_sha256",
        "rebuilt_rxnorm_slice_sha256",
        "final_tensor_sha256",
        "embedding_model",
        "embedding_revision",
        "dimensions",
        "dtype",
    ):
        assert manifest.get(key) not in (None, ""), key
    assert manifest["icd_slice_unchanged"] is True
    assert manifest["training_performed"] is False
    assert manifest["null_mode"] == "shadow"


# ------------------------------------------------------------------ fail-closed checks


def test_stale_manifest_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _, _ = _prepare(tmp_path, monkeypatch)
    documents = handoff.build_docs(args)
    target = handoff.rebuild_rxnorm_vectors(args)
    assert handoff.verify_cache(target, documents, args.frozen_documents) == []
    extra = '{"concept_id":"X","ontology":"RXNORM","text":"x"}\n'
    documents.write_text(documents.read_text(encoding="utf-8") + extra, encoding="utf-8")
    problems = handoff.verify_cache(target, documents, args.frozen_documents)
    assert any("structured documents changed" in p for p in problems)


def test_cache_without_a_manifest_is_never_valid(tmp_path: Path) -> None:
    orphan = tmp_path / "orphan.pt"
    torch.save(torch.zeros(2, 2), orphan)
    problems = handoff.verify_cache(orphan, tmp_path / "d.jsonl", tmp_path / "f.jsonl")
    assert any("no manifest" in p for p in problems)


def test_preflight_reports_every_missing_input(tmp_path: Path) -> None:
    args = make_args(
        tmp_path,
        tmp_path / "missing.jsonl",
        tmp_path / "missing.pt",
        model_root=tmp_path / "models",
        rxnorm_index=tmp_path / "rx.json",
        icd_v3_index=tmp_path / "i3.json",
        icd_v41_index=tmp_path / "i41.json",
    )
    problems = handoff.preflight(args)
    assert len(problems) >= 4
    assert any("Qwen3-Embedding-4B" in p for p in problems)


# ---------------------------------------------------------------- A/B and pipeline shape


def test_ab_uses_one_frozen_case_set_for_both_arms() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "cases = json.loads(args.cases.read_text" in source
    assert 'evaluate_arm("A", args, args.frozen_documents, args.frozen_vectors, cases)' in source
    assert 'evaluate_arm("B", args, structured_documents, rebuilt, cases)' in source


def test_real_reranker_is_used_not_a_stub() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Qwen3Reranker(" in source
    assert "S1_RERANKER_REVISION" in source
    for stub in ("MockReranker", "FixedReranker"):
        assert stub not in source


def test_null_stays_shadow_and_nothing_is_trained() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "NULL_MODE_SHADOW" in source
    assert "null_mode=NULL_MODE_SHADOW" in source
    for banned in ("optimizer", "loss.backward", "train(", "fine_tune"):
        assert banned not in source


def test_models_are_loaded_sequentially_and_unloaded() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "del model, tokenizer" in source
    assert "torch.cuda.empty_cache()" in source
    assert "backend.unload()" in source


def test_no_public_test_clinical_text_is_referenced() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for banned in ("organizer_test", "private_linker_gold", "output.zip"):
        assert banned not in source
