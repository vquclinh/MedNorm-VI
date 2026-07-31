"""Milestone 3A KB freeze contracts and builders."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mednorm_vi.governance.e4_retirement import RETIRED_FROM_ACTIVE_ARCHITECTURE
from mednorm_vi.kb.freeze import (
    APPROVED,
    CANDIDATE_REPRESENTATION_SCHEMA_VERSION,
    GOVERNED_SYNONYM_SCHEMA_VERSION,
    ICD10_KB_SCHEMA_VERSION,
    INN_RXNORM_CROSSWALK_SCHEMA_VERSION,
    KB_PROVENANCE_SCHEMA_VERSION,
    KB_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    ONTOLOGY_ICD10,
    ONTOLOGY_RXNORM,
    REQUIRES_REVIEW,
    RXNORM_KB_SCHEMA_VERSION,
    build_icd10_freeze,
    build_rxnorm_freeze,
)
from mednorm_vi.linking.structured_medication import (
    INN_TO_RXNORM_NAME,
    bridged_ingredient_names,
)

REPO = Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_icd_normalized_csv(path: Path) -> None:
    fields = [
        "supplied_code",
        "dotted_code",
        "undotted_code",
        "vietnamese_label",
        "english_label",
        "aliases",
        "chapter",
        "block",
        "parent",
        "children",
        "specificity",
        "source_page",
        "source_row",
        "source_document_sha256",
        "status",
        "flags",
    ]
    rows = [
        {
            "supplied_code": "A02",
            "dotted_code": "A02",
            "undotted_code": "A02",
            "vietnamese_label": "Nhiễm Salmonella khác",
            "english_label": "",
            "aliases": "",
            "chapter": "I",
            "block": "A00-A09",
            "parent": "",
            "children": "A021",
            "specificity": "0",
            "source_page": "1",
            "source_row": "11",
            "source_document_sha256": "0" * 64,
            "status": "active",
            "flags": "",
        },
        {
            "supplied_code": "A02.1",
            "dotted_code": "A02.1",
            "undotted_code": "A021",
            "vietnamese_label": "Nhiễm trùng hệ thống do",
            "english_label": "",
            "aliases": "",
            "chapter": "I",
            "block": "A00-A09",
            "parent": "A02",
            "children": "",
            "specificity": "1",
            "source_page": "1",
            "source_row": "13",
            "source_document_sha256": "0" * 64,
            "status": "active",
            "flags": "duplicate_code_rows_collapsed",
        },
        {
            "supplied_code": "A02.8",
            "dotted_code": "A02.8",
            "undotted_code": "A028",
            "vietnamese_label": "+ bệnh kẽ ống",
            "english_label": "",
            "aliases": "",
            "chapter": "I",
            "block": "A00-A09",
            "parent": "A02",
            "children": "",
            "specificity": "1",
            "source_page": "1",
            "source_row": "15",
            "source_document_sha256": "0" * 64,
            "status": "active",
            "flags": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_kb_schema_files_are_versioned() -> None:
    expected = {
        "icd10-kb-schema-v1.json": ICD10_KB_SCHEMA_VERSION,
        "rxnorm-kb-schema-v1.json": RXNORM_KB_SCHEMA_VERSION,
        "kb-provenance-schema-v1.json": KB_PROVENANCE_SCHEMA_VERSION,
        "governed-synonym-schema-v1.json": GOVERNED_SYNONYM_SCHEMA_VERSION,
        "inn-rxnorm-crosswalk-schema-v1.json": INN_RXNORM_CROSSWALK_SCHEMA_VERSION,
        "candidate-representation-schema-v1.json": CANDIDATE_REPRESENTATION_SCHEMA_VERSION,
        "kb-snapshot-manifest-schema-v1.json": KB_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    }
    for filename, schema_id in expected.items():
        payload = json.loads((REPO / "schemas" / "kb" / filename).read_text(encoding="utf-8"))
        assert payload["$id"] == schema_id


def test_icd_freeze_flags_fragment_names_and_inferred_parents(tmp_path: Path) -> None:
    source = tmp_path / "icd_normalized.csv"
    _write_icd_normalized_csv(source)
    result = build_icd10_freeze(
        normalized_csv=source,
        source_manifest=REPO / "data" / "manifests" / "icd10-vi-tt06-2026-official.yaml",
        output_dir=tmp_path / "icd_freeze",
    )

    concepts = _read_csv(result.output_dir / "icd10_concepts_v1.csv")
    edges = _read_csv(result.output_dir / "icd10_hierarchy_edges_v1.csv")
    review = _read_csv(result.output_dir / "icd10_review_queue_v1.csv")

    assert result.ontology == ONTOLOGY_ICD10
    assert any(
        row["code"] == "A028" and row["review_status"] == "SUSPECT_TRUNCATION" for row in concepts
    )
    assert {row["source_evidence"] for row in edges} == {"legacy_code_prefix_inference"}
    assert {row["runtime_authoritative"] for row in edges} == {"false"}
    assert any(row["issue_code"] == "inferred_parent_requires_source_review" for row in review)


def test_rxnorm_freeze_preserves_labeled_directed_relations(tmp_path: Path) -> None:
    result = build_rxnorm_freeze(
        rrf_root=REPO / "tests" / "fixtures" / "kb" / "rxnorm" / "snapshot_a",
        source_manifest=REPO / "data" / "manifests" / "rxnorm-prescribable-2026-07-06.yaml",
        output_dir=tmp_path / "rxnorm_freeze",
        source_version="fixture",
        snapshot_label="fixture",
    )

    concepts = _read_csv(result.output_dir / "rxnorm_concepts_v1.csv")
    relations = _read_csv(result.output_dir / "rxnorm_relations_v1.csv")
    synonyms = _read_csv(result.output_dir / "rxnorm_synonyms_v1.csv")

    assert result.ontology == ONTOLOGY_RXNORM
    assert {row["rxcui"] for row in concepts} == {"100001", "100002", "100003"}
    assert relations == [
        {
            **relations[0],
            "source_rxcui": "100001",
            "target_rxcui": "100002",
            "rel": "RN",
            "rela": "has_ingredient",
            "direction": "source_to_target",
            "endpoint_status": "ok",
        }
    ]
    assert {row["term_type"] for row in synonyms} >= {"IN", "SCD"}


def test_crosswalk_rows_remain_review_required_not_authoritative(tmp_path: Path) -> None:
    result = build_rxnorm_freeze(
        rrf_root=REPO / "tests" / "fixtures" / "kb" / "rxnorm" / "snapshot_a",
        source_manifest=REPO / "data" / "manifests" / "rxnorm-prescribable-2026-07-06.yaml",
        output_dir=tmp_path / "rxnorm_freeze",
        source_version="fixture",
        snapshot_label="fixture",
    )

    crosswalk = _read_csv(result.output_dir / "inn_rxnorm_crosswalk_v1.csv")

    assert crosswalk
    assert {row["review_decision"] for row in crosswalk} == {REQUIRES_REVIEW}
    assert {row["runtime_authoritative"] for row in crosswalk} == {"false"}
    assert not any(row["review_decision"] == APPROVED for row in crosswalk)


def test_the_inn_bridge_widens_lookup_but_never_emits_a_code() -> None:
    """Superseded by Audit 0058: 3A disabled the bridge outright; 3B governs it.

    3A had one state for two questions and answered both with "no": an unreviewed
    bridge could not widen retrieval *or* emit a code. Audit 0058 splits them. The 12
    canonical bridges are ``RETRIEVAL_ONLY`` — an exact source-string equivalence, not
    a clinical approval — so lookup may widen while emission stays gated by the
    searchable index and the mention's own evidence.
    """
    assert INN_TO_RXNORM_NAME["paracetamol"] == "acetaminophen"
    expanded = bridged_ingredient_names("paracetamol")
    assert expanded == ("acetaminophen",), "a RETRIEVAL_ONLY bridge may widen lookup"
    # Names, never codes: the return type makes emission structurally impossible.
    assert not any(name.isdigit() for name in expanded)


def test_e4_retirement_is_unchanged_by_kb_freeze_contracts() -> None:
    assert RETIRED_FROM_ACTIVE_ARCHITECTURE == "RETIRED_FROM_ACTIVE_ARCHITECTURE"
