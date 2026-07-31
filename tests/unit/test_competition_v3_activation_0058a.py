"""Milestone 3B.1: competition KB v3 runtime index activation (Audit 0058A).

Audit 0058 built and verified the v3 indices but left the active config pointing at
the legacy ones, because switching them changes the inputs of a
`FRAMEWORK_RUNTIME_FROZEN` stack. This module is the evidence that the switch is
sound: it asserts what the canonical config now opens, what those indices contain,
and — the part that actually matters — that the three governance boundaries still
hold when the runtime is reading v3 rather than a snapshot that had no closure
population to leak in the first place.

The index fixtures here are the **real activated artifacts**, not miniatures. A
miniature cannot answer "did activation change what the runtime emits?", which is the
only question this milestone exists to answer. The heavy loads are module-scoped and
shared, so the suite pays for each index once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mednorm_vi.inference.config import PipelineConfig, evaluate_readiness
from mednorm_vi.kb.competition.topk import MAX_CANDIDATES
from mednorm_vi.kb.indexing.retrieval import LocalIndex, load_index, search_index
from mednorm_vi.linking.icd10 import link_icd10
from mednorm_vi.linking.rxnorm import DROP_CLOSURE_ONLY, link_rxnorm, link_rxnorm_structured
from mednorm_vi.linking.rxnorm_graph import is_closure_only, name_of
from mednorm_vi.linking.structured_medication import (
    GOVERNED_INN_BRIDGE_RETRIEVAL_ENABLED,
    bridged_ingredient_names,
)
from mednorm_vi.metric_decoder import decode_entities
from mednorm_vi.resolution.models import (
    STATUS_ACCEPTED,
    BoundaryEvidence,
    EntityHypothesis,
    TypeEvidence,
)

REPO = Path(__file__).resolve().parents[2]
PIPELINE_CFG = REPO / "configs" / "pipeline" / "full_v1.yaml"

ACTIVE_ICD = REPO / "indices" / "candidate" / "icd10_vi" / "competition-v3" / "index.json"
ACTIVE_RXNORM = REPO / "indices" / "candidate" / "rxnorm" / "competition-v3" / "index.json"
ROLLBACK_ICD = REPO / "indices" / "icd10_vi" / "tt06-2026" / "index.json"
ROLLBACK_RXNORM = REPO / "indices" / "rxnorm" / "prescribable-2026-07-06" / "index.json"

ICD_INDEX_HASH = "b050a2d5ead90c8410101d257b08b347c129f5a8faa713d7e5f2757c7931a975"
RXNORM_INDEX_HASH = "d3c45f349bad1532135b70214c48e54c270ede6ae6988e1444a9949887d92ac3"
V3_SNAPSHOT_ID = "competition-kb-v3-candidate-4d206538e7d24c32"

# Recorded before activation (Audit 0057 §19, re-verified in 0058A §1). A legacy index
# that changed would mean the "rollback is immediate" claim is false.
LEGACY_HASHES = {
    ROLLBACK_ICD: "94a1378b0f355afc6da23bb0da4d1e015dbbd4f51a1f2c0508f06e7b1727a768",
    ROLLBACK_RXNORM: "4e02ecbfc61e586e041a424177fa682f048618e8e0431e8204f6d2afc2692aec",
}

_needs_indices = pytest.mark.skipif(
    not (ACTIVE_ICD.is_file() and ACTIVE_RXNORM.is_file()),
    reason="competition v3 runtime indices not built locally",
)


@pytest.fixture(scope="module")
def rxnorm_index() -> LocalIndex:
    return load_index(ACTIVE_RXNORM)


@pytest.fixture(scope="module")
def icd_index() -> LocalIndex:
    return load_index(ACTIVE_ICD)


def _hypothesis(
    text: str,
    entity_type: str = "THUỐC",
    *,
    components: tuple[dict[str, object], ...] = (),
) -> EntityHypothesis:
    return EntityHypothesis(
        hypothesis_id="h1",
        document_id="d1",
        start=0,
        end=len(text),
        text=text,
        entity_type=entity_type,
        status=STATUS_ACCEPTED,
        chosen_proposal_id="p1",
        source_proposal_ids=("p1",),
        boundary_evidence=BoundaryEvidence("policy", "full", "p1", ()),
        type_evidence=TypeEvidence(entity_type, "E1", (entity_type,)),
        components=components,
        score=0.9,
    )


def _component(role: str, text: str, start: int = 0) -> dict[str, object]:
    return {
        "role": role,
        "text": text,
        "start": start,
        "end": start + len(text),
        "normalized": text,
        "detail": "",
    }


# --- 1. the active config resolves to competition-v3 -------------------------


def test_active_config_resolves_to_competition_v3_paths() -> None:
    config = PipelineConfig.load(PIPELINE_CFG)
    assert config.icd_index == "indices/candidate/icd10_vi/competition-v3/index.json"
    assert config.rxnorm_index == "indices/candidate/rxnorm/competition-v3/index.json"
    assert (REPO / config.icd_index).is_file()
    assert (REPO / config.rxnorm_index).is_file()


@_needs_indices
def test_readiness_reports_no_index_errors_after_activation() -> None:
    config = PipelineConfig.load(PIPELINE_CFG)
    for mode in ("deterministic", "specialist"):
        report = evaluate_readiness(config, mode=mode)
        assert not [e for e in report.errors if "index" in e], report.errors
    # `full` stays NOT_READY for missing checkpoints, which activation must not fix
    # or appear to fix — but not for an index reason.
    full = evaluate_readiness(config, mode="full")
    assert not [e for e in full.errors if "index" in e], full.errors


# --- 2-5. index contents ------------------------------------------------------


@_needs_indices
def test_icd_index_exposes_15308_searchable_records(icd_index: LocalIndex) -> None:
    assert icd_index.index_type == "icd10_vi"
    assert len(icd_index.records) == 15308
    assert icd_index.source_snapshot_id == V3_SNAPSHOT_ID
    meta = json.loads(ACTIVE_ICD.read_text(encoding="utf-8"))["metadata"]
    assert meta["deterministic_index_hash"] == ICD_INDEX_HASH
    assert meta["parameters"]["suspect_names_indexed"] is True


@_needs_indices
def test_rxnorm_index_partitions_into_searchable_and_closure_only(
    rxnorm_index: LocalIndex,
) -> None:
    assert rxnorm_index.index_type == "rxnorm"
    assert len(rxnorm_index.records) == 211949
    assert rxnorm_index.source_snapshot_id == V3_SNAPSHOT_ID

    membership: dict[str, int] = {}
    for record in rxnorm_index.records.values():
        key = str(record.get("metadata", {}).get("membership", "<unstated>"))
        membership[key] = membership.get(key, 0) + 1
    assert membership == {"searchable": 82429, "closure_only": 129520}

    meta = json.loads(ACTIVE_RXNORM.read_text(encoding="utf-8"))["metadata"]
    assert meta["deterministic_index_hash"] == RXNORM_INDEX_HASH
    assert meta["parameters"]["closure_only_in_postings"] is False


# --- 6. closure-only records cannot become final candidates -------------------


@_needs_indices
def test_closure_only_records_are_not_retrievable(rxnorm_index: LocalIndex) -> None:
    closure = [c for c in list(rxnorm_index.records)[:20000] if is_closure_only(rxnorm_index, c)]
    assert closure, "the activated index must contain closure-only records"
    for concept_id in closure[:10]:
        hits = search_index(rxnorm_index, name_of(rxnorm_index, concept_id), limit=25)
        assert concept_id not in {h.concept_id for h in hits}


@_needs_indices
def test_the_linker_drops_a_closure_node_traversal_reaches(rxnorm_index: LocalIndex) -> None:
    """Retrieval cannot reach a closure node; traversal can, and must not keep it."""
    report = link_rxnorm_structured(
        _hypothesis("mesna", components=(_component("name", "mesna"),)), rxnorm_index
    )
    dropped = [d.concept_id for d in report.suppressed if d.reason == DROP_CLOSURE_ONLY]
    assert dropped, "this mention is known to traverse into the closure population"
    assert report.retained, "dropping closure nodes must not empty the candidate list"
    assert not any(is_closure_only(rxnorm_index, d.concept_id) for d in report.retained)


# --- 7. advisory ICD hierarchy cannot independently offer a candidate ---------


@_needs_indices
def test_advisory_icd_hierarchy_is_not_source_authoritative() -> None:
    from mednorm_vi.kb.competition.icd10_view import (
        HIERARCHY_ADVISORY,
        advisory_hierarchy_may_offer,
    )

    assert not advisory_hierarchy_may_offer(HIERARCHY_ADVISORY)
    rows = (
        REPO
        / "data"
        / "derived"
        / "kb_freeze"
        / "0058_competition_candidate_v3"
        / "icd10_advisory_hierarchy_v1.csv"
    )
    if rows.is_file():
        import csv

        with rows.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                assert row["may_offer_candidate"] == "false"
                assert row["source_authoritative"] == "false"


@_needs_indices
def test_icd_retrieval_works_against_the_activated_index(icd_index: LocalIndex) -> None:
    hits = search_index(icd_index, "Bệnh tả", limit=5)
    assert hits, "exact ICD retrieval must work on the activated index"
    result = link_icd10(_hypothesis("Bệnh tả", "CHẨN_ĐOÁN"), icd_index)
    assert result.candidates, "ICD linking must produce candidates"
    assert all(icd_index.exists(c.code) for c in result.candidates)


# --- 8-9. governed RETRIEVAL_ONLY bridge --------------------------------------


def test_the_bridge_expands_lookup_and_returns_names_not_codes() -> None:
    assert GOVERNED_INN_BRIDGE_RETRIEVAL_ENABLED
    expanded = bridged_ingredient_names("paracetamol")
    assert expanded == ("acetaminophen",)
    assert not any(name.isdigit() for name in expanded)


@_needs_indices
def test_paracetamol_reaches_rxcui_161_through_governed_retrieval(
    rxnorm_index: LocalIndex,
) -> None:
    """The coverage hole Audit 0058 closed must stay closed after activation."""
    report = link_rxnorm_structured(
        _hypothesis("paracetamol", components=(_component("name", "paracetamol"),)),
        rxnorm_index,
    )
    assert report.retained
    assert report.retained[0].concept_id == "161"
    assert any(note.startswith("inn_bridge:paracetamol") for note in report.traversal_notes)
    # Reached through the index, not handed over by the bridge.
    assert "exact" in report.retained[0].retrieval_sources
    assert not any(is_closure_only(rxnorm_index, d.concept_id) for d in report.retained)


# --- 10-12. decoding and validation invariants under the activated indices ----


@_needs_indices
def test_candidate_ordering_is_deterministic_on_the_activated_index(
    rxnorm_index: LocalIndex,
) -> None:
    hypothesis = _hypothesis(
        "amlodipine 10 mg",
        components=(
            _component("name", "amlodipine"),
            _component("strength_value", "10", 11),
            _component("strength_unit", "mg", 14),
        ),
    )
    runs = {
        tuple(c.code for c in link_rxnorm(hypothesis, rxnorm_index).candidates) for _ in range(3)
    }
    assert len(runs) == 1, f"ordering drifted across runs: {runs}"


@_needs_indices
def test_l8_emits_at_most_topk_and_only_offered_codes(rxnorm_index: LocalIndex) -> None:
    from mednorm_vi.confidence_cascade.cascade import CascadeDecision

    hypothesis = _hypothesis(
        "amlodipine 10 mg",
        components=(
            _component("name", "amlodipine"),
            _component("strength_value", "10", 11),
            _component("strength_unit", "mg", 14),
        ),
    )
    link = link_rxnorm(hypothesis, rxnorm_index)
    offered = {c.code for c in link.candidates}
    assert offered, "the fixture mention must offer something to narrow"
    decoded = decode_entities((hypothesis,), (CascadeDecision("h1", True, 0.9, ()),), (), (link,))
    emitted = set(decoded[0].candidates)
    assert emitted <= offered, "L8 cannot introduce a code outside the offered set"
    assert len(emitted) <= MAX_CANDIDATES


@_needs_indices
def test_l9_refuses_a_closure_only_code_against_the_activated_snapshot(
    rxnorm_index: LocalIndex,
) -> None:
    from mednorm_vi.validator.kb_membership import (
        LockedSnapshots,
        validate_document_candidates,
    )

    closure = next(
        c for c in list(rxnorm_index.records)[:20000] if is_closure_only(rxnorm_index, c)
    )
    searchable = next(
        c for c in list(rxnorm_index.records)[:20000] if not is_closure_only(rxnorm_index, c)
    )

    def _ok(code: str) -> bool:
        entities = [{"text": "x", "type": "THUỐC", "candidates": [code], "position": [0, 1]}]
        return validate_document_candidates(
            entities,
            LockedSnapshots(rxnorm=rxnorm_index),
            document_id="1.json",
            offered_codes_by_index={0: [code]},
        ).ok

    assert _ok(searchable), "a searchable concept must pass L9"
    assert not _ok(closure), "a closure-only endpoint must be refused by L9"


# --- 13-14. rollback remains available, legacy indices untouched --------------


def test_rollback_indices_still_exist_and_are_unchanged() -> None:
    import hashlib

    for path, expected in LEGACY_HASHES.items():
        assert path.is_file(), f"rollback index missing: {path}"
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected, f"legacy index changed: {path}"


def test_rollback_is_declared_in_the_active_config() -> None:
    doc = yaml.safe_load(PIPELINE_CFG.read_text(encoding="utf-8"))
    rollback = doc["rxnorm_selection"]["rollback"]
    assert rollback["icd_index"] == "indices/icd10_vi/tt06-2026/index.json"
    assert rollback["rxnorm_index"] == "indices/rxnorm/prescribable-2026-07-06/index.json"
    # The active paths must not equal the rollback paths, or "rollback" is a fiction.
    assert doc["icd_index"] != rollback["icd_index"]
    assert doc["rxnorm_index"] != rollback["rxnorm_index"]
