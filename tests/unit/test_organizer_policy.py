"""Organizer-policy registry: confirmed facts vs unresolved hypotheses (1C-A)."""

from __future__ import annotations

from pathlib import Path

from mednorm_vi.organizer_policy import load_organizer_registry

REPO = Path(__file__).resolve().parents[2]
REG = load_organizer_registry(REPO / "configs" / "organizer")


def test_confirmed_facts_present() -> None:
    ids = {f.fact_id for f in REG.confirmed_facts}
    assert "FACT-RXNORM-2026" in ids
    assert "FACT-TESTNAME-RESULT-INDEPENDENT" in ids
    assert "FACT-ASSERTIONS-MULTILABEL" in ids


def test_hypotheses_are_never_confirmed() -> None:
    for p in REG.unresolved_policies:
        assert not p.is_confirmed  # a hypothesis is never a confirmed rule


def test_key_unresolved_policies_exist_and_open() -> None:
    for pid in ("UNRES-POS-COORD-SPACE", "UNRES-MATCH-MODE", "UNRES-RXNORM-RELEASE",
                "UNRES-ICD-SOURCE", "UNRES-ISHISTORICAL"):
        p = REG.policy(pid)
        assert p is not None and p.is_open


def test_rxnorm_and_icd_hypotheses_loaded() -> None:
    assert any(h.policy_id == "RXDEC-INGREDIENT-SAFE" for h in REG.rxnorm_decoding_hypotheses)
    assert any(h.policy_id == "ICDF-DOTTED-OUTPUT" for h in REG.icd_format_hypotheses)
    assert REG.hypothesis_count >= 20


def test_position_default_is_raw_codepoint() -> None:
    assert REG.default_position_policy == "raw-codepoint-half-open"
    assert "raw-codepoint-half-open" in REG.position_policy_ids


def test_registry_hash_deterministic() -> None:
    again = load_organizer_registry(REPO / "configs" / "organizer")
    assert again.config_hash == REG.config_hash
