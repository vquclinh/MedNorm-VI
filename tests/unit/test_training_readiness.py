"""Governed corpus build, label mappings, and annotation-coverage masks
(Audit 0017). Uses tracked config + synthetic fixtures; the real-data corpus
build is a determinism/leakage smoke test guarded on local data presence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mednorm_vi.data_engine.annotation_coverage import (
    SourceCoverage,
    coverage_from_manifest_fields,
    example_loss_mask,
)
from mednorm_vi.data_engine.corpus_build import build_governed_corpus, concrete_type_mapping

REPO = Path(__file__).resolve().parents[2]
LM = REPO / "configs" / "resources" / "label_mappings" / "public_ner"
PUBLIC_NER = REPO / "data" / "external" / "public_ner"
_HAVE_DATA = (PUBLIC_NER / "vimedner" / "data" / "train.txt").is_file()


# --- versioned label mappings ------------------------------------------------

def test_concrete_mapping_translates_vietnamese_to_internal_enum() -> None:
    vm = concrete_type_mapping(LM / "vimedner_v1.yaml")
    assert vm == {"ten_benh": "DIAGNOSIS", "trieu_chung_benh": "SYMPTOM"}
    vq = concrete_type_mapping(LM / "vimq_v1.yaml")
    assert vq == {"drug": "MEDICATION"}


def test_governance_gate_uses_training_use_not_source_license() -> None:
    from mednorm_vi.data_engine import require_training_allowed
    from mednorm_vi.resources.ner import load_ner_manifest
    for name in ("phoner-covid19", "vimedner", "vimq", "vietmed-ner"):
        m = load_ner_manifest(
            REPO / "data" / "manifests" / f"{name}-public-ner.yaml")
        assert m.license.status == "REVIEW_REQUIRED"           # source fact preserved
        assert m.training_use["status"] == "USER_ATTESTED_PERMISSION"
        require_training_allowed(m)                            # gate passes via training_use


def test_merged_label_not_mapped_to_a_concrete_type() -> None:
    # PhoNER's only clinical label is merged symptom+disease -> no concrete type.
    assert concrete_type_mapping(LM / "phoner_covid19_v1.yaml") == {}
    # ViMQ's merged label must not become a concrete type either.
    assert "SYMPTOM_AND_DISEASE" not in concrete_type_mapping(LM / "vimq_v1.yaml")


# --- annotation coverage / masked loss ---------------------------------------

def test_source_coverage_loss_mask() -> None:
    cov = SourceCoverage("x", span=True, entity_type=True)  # no assertions/candidates
    m = cov.loss_mask()
    assert m["span"] and m["entity_type"]
    assert not m["assertions"] and not m["icd_candidates"] and not m["rxnorm_candidates"]


def test_missing_annotation_is_masked_not_negative() -> None:
    cov = coverage_from_manifest_fields("vimedner", has_typed_entities=True,
                                        assertion_availability="none",
                                        normalization_availability="none")
    example = {"entities": [{"target_type": "DIAGNOSIS", "assertions": [], "candidates": []}]}
    mask = example_loss_mask(example, cov)
    assert mask["span"] and mask["entity_type"]
    # No assertion/candidate annotation -> those objectives are masked out.
    assert not mask["assertions"]
    assert not mask["icd_candidates"] and not mask["rxnorm_candidates"]


def test_present_assertion_is_supervised_when_source_covers_it() -> None:
    cov = SourceCoverage("y", span=True, entity_type=True, assertions=True)
    example = {"entities": [
        {"target_type": "SYMPTOM", "assertions": ["isNegated"], "candidates": []}]}
    assert example_loss_mask(example, cov)["assertions"] is True


# --- governed corpus build (real data; determinism + leakage) ----------------

@pytest.mark.skipif(not _HAVE_DATA, reason="public NER data not present locally")
def test_governed_corpus_deterministic_and_leakage_free(tmp_path: Path) -> None:
    a = build_governed_corpus(out_dir=tmp_path / "a", seed=20260723)
    b = build_governed_corpus(out_dir=tmp_path / "b", seed=20260723)
    assert a["offset_invalid"] == 0
    assert a["cross_split_family_leakage"] == 0            # no duplicate family crosses splits
    assert a["eval_map_approximate_entities"] == 0        # val/internal_test are MAP_EXACT only
    assert a["split_sha256"] == b["split_sha256"]         # deterministic
    assert set(a["type_distribution"]) <= {"DIAGNOSIS", "SYMPTOM", "MEDICATION",
                                           "TEST_NAME", "TEST_RESULT"}
    assert "TEST_NAME" in a["target_types_missing"]
    assert "TEST_RESULT" in a["target_types_missing"]


def test_duplicate_families_cluster_variants() -> None:
    from mednorm_vi.data_engine.corpus_build import duplicate_families
    texts = [
        "Bệnh nhân ho nhiều",
        "benh nhan ho nhieu",          # accent/case variant -> same family
        "Bệnh  nhân   ho nhiều",       # whitespace-only variant -> same family
        "Hoàn toàn khác biệt ở đây",   # unrelated
    ]
    fam = duplicate_families(texts)
    fids = fam["family_id"]
    assert fids[0] == fids[1] == fids[2]                  # variants collapse
    assert fids[3] != fids[0]                             # unrelated stays separate


# --- adversarial four-band SimHash LSH (complete recall for Hamming <= 3) -----

def _flip(h: int, *bits: int) -> int:
    for b in bits:
        h ^= (1 << b)
    return h


def test_simhash_bands_partition_64_bits() -> None:
    from mednorm_vi.data_engine.corpus_build import simhash_bands
    h = 0x0123456789ABCDEF
    bands = simhash_bands(h)
    assert len(bands) == 4
    assert bands == (0xCDEF, 0x89AB, 0x4567, 0x0123)     # bits 0-15,16-31,32-47,48-63


def test_near_dup_hamming1_each_band_is_candidate_and_accepted() -> None:
    from mednorm_vi.data_engine.corpus_build import near_duplicate_pairs
    base = 0
    # a Hamming-1 difference in EVERY band must still be caught (3 bands stay equal)
    for bit in (0, 15, 16, 31, 32, 47, 48, 63):
        pairs = near_duplicate_pairs([base, _flip(base, bit)], threshold=3)
        assert pairs == [(0, 1)], f"missed Hamming-1 at bit {bit}"


def test_near_dup_hamming3_across_three_bands_accepted() -> None:
    from mednorm_vi.data_engine.corpus_build import near_duplicate_pairs
    other = _flip(0, 3, 20, 40)                          # one bit in bands 0,1,2 -> band 3 equal
    assert near_duplicate_pairs([0, other], threshold=3) == [(0, 1)]


def test_near_dup_hamming4_one_bit_per_band_rejected() -> None:
    from mednorm_vi.data_engine.corpus_build import near_duplicate_pairs
    other = _flip(0, 3, 20, 40, 60)                      # one bit in every band -> Hamming 4
    # It IS a candidate (bands differ) but the final Hamming check rejects it.
    assert near_duplicate_pairs([0, other], threshold=3) == []


def test_near_dup_identical_and_determinism() -> None:
    from mednorm_vi.data_engine.corpus_build import duplicate_families, near_duplicate_pairs
    assert near_duplicate_pairs([42, 42], threshold=3) == [(0, 1)]
    texts = ["alpha beta gamma", "alpha beta gamma", "alpha  beta  gamma", "totally other"]
    a = duplicate_families(texts)["family_id"]
    b = duplicate_families(texts)["family_id"]
    assert a == b                                        # byte-identical family assignment
