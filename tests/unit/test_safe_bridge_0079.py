"""Safe-bridge filter guards (experiment 0079). CPU only, no model, no network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mednorm_vi.reasoner.safe_bridge import (
    BRIDGE_DENYLIST,
    MAX_BRIDGE_TOKENS,
    REJECT_CAPITALISED,
    REJECT_CONNECTOR,
    REJECT_PUNCTUATION,
    REJECT_TOO_LONG,
    evaluate_merge,
    normalize,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/filter_merge_only_0079.py"
spec = importlib.util.spec_from_file_location("f0079", SCRIPT)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def verdict(source: str, fragments: list[str]):
    offsets, cursor = [], 0
    for fragment in fragments:
        index = source.index(fragment, cursor)
        offsets.append([index, index + len(fragment)])
        cursor = index + len(fragment)
    return evaluate_merge(source, offsets)


#: The merges 0078 produced that must be rejected, from the real run.
SPURIOUS = [
    ("hẹp tắc mạch vành gây suy tim", ["hẹp tắc mạch vành", "suy tim"]),
    ("Kawasaki là bệnh viêm mạch máu", ["Kawasaki", "bệnh viêm mạch máu"]),
    ("chảy máu chân răng kèm hôi miệng", ["chảy máu chân răng", "hôi miệng"]),
    ("suy hô hấp kèm tăng huyết áp", ["suy hô hấp", "tăng huyết áp"]),
    ("nghẹt ngực kèm sốt", ["nghẹt ngực", "sốt"]),
    ("yếu cơ dẫn đến ngã", ["yếu cơ", "ngã"]),
    ("nôn thức ăn lẫn máu kèm đi ngoài phân đen", ["nôn thức ăn lẫn máu", "đi ngoài phân đen"]),
    ("đau đầu kéo dài kèm theo mờ mắt", ["đau đầu kéo dài", "mờ mắt"]),
    ("răng hay chảy máu chân răng", ["răng", "chảy máu chân răng"]),
    ("Viêm phổi là tình trạng viêm phế quản", ["Viêm phổi", "viêm phế quản"]),
]

#: The fragmentation repairs that must survive.
GENUINE = [
    ("Trẻ Thiếu men G6PD bẩm sinh", ["Thiếu", "G6PD"]),
    ("Thiếu men G6PD", ["Thiếu men", "G6PD"]),
    ("tăng sản tuyến tiền liệt", ["tăng sản tuyến", "liệt"]),
    ("khó thở khi gắng sức", ["khó thở khi", "sức"]),
    ("viêm gan vàng da ứ mật", ["viêm gan vàng da ứ", "mật"]),
    ("phù nhẹ 2 chi dưới", ["phù", "2 chi dưới"]),
]


@pytest.mark.parametrize(("source", "fragments"), SPURIOUS)
def test_every_observed_spurious_merge_is_rejected(source: str, fragments: list[str]) -> None:
    assert not verdict(source, fragments).accepted


@pytest.mark.parametrize(("source", "fragments"), GENUINE)
def test_every_genuine_fragmentation_repair_is_accepted(source: str, fragments: list[str]) -> None:
    assert verdict(source, fragments).accepted


def test_capitalised_bridge_word_is_rejected() -> None:
    """'ho' + 'nhĩ' -> 'ho Rung nhĩ': a different concept intruding, not connective tissue."""
    result = verdict("ho Rung nhĩ", ["ho", "nhĩ"])
    assert not result.accepted and result.reason == REJECT_CAPITALISED


def test_acronym_that_is_itself_a_fragment_is_unaffected() -> None:
    """The capitalisation rule reads ONLY the bridge, never the fragments."""
    assert verdict("Thiếu men G6PD", ["Thiếu", "G6PD"]).accepted
    assert verdict("nhiễm HIV mạn", ["nhiễm", "mạn"]).accepted is False  # bridge is HIV
    assert verdict("xét nghiệm HIV", ["xét nghiệm", "HIV"]).accepted  # HIV is a fragment


def test_punctuation_in_the_bridge_is_rejected() -> None:
    result = verdict("sốt cao, ho nhiều", ["sốt cao", "ho nhiều"])
    assert not result.accepted and result.reason == REJECT_PUNCTUATION


def test_two_token_bridge_is_rejected() -> None:
    result = verdict("đau bụng rất nhiều buồn nôn", ["đau bụng", "buồn nôn"])
    assert not result.accepted and result.reason == REJECT_TOO_LONG
    assert MAX_BRIDGE_TOKENS == 1


def test_connector_matching_is_case_and_whitespace_normalized() -> None:
    result = verdict("sốt  KÈM   ho", ["sốt", "ho"])
    assert not result.accepted and result.reason == REJECT_CONNECTOR
    assert normalize("  KÈM  THEO ") == "kèm theo"


def test_denylist_covers_the_required_vietnamese_connectors() -> None:
    for connector in (
        "và",
        "hoặc",
        "hay",
        "kèm",
        "kèm theo",
        "gây",
        "do",
        "là",
        "với",
        "dẫn",
        "dẫn đến",
        "sau",
        "sau khi",
        "trước",
        "trước khi",
        "khi",
        "thì",
        "xuất hiện",
        "gồm",
        "bao gồm",
        "trở lại",
        "cùng",
        "đồng thời",
        "nhưng",
        "rồi",
    ):
        assert connector in BRIDGE_DENYLIST


def test_overlapping_fragments_are_safe() -> None:
    """Explicit offsets: the sequential helper cannot express an overlap."""
    source = "Thiếu men G6PD"
    overlapping = [[0, len("Thiếu men")], [source.index("men"), len(source)]]
    assert evaluate_merge(source, overlapping).accepted


def test_verdict_records_gaps_and_normalized_tokens() -> None:
    result = verdict("Trẻ Thiếu men G6PD", ["Thiếu", "G6PD"])
    assert result.gaps == (" men ",)
    assert result.normalized_tokens == ("men",)


# ------------------------------------------------------------------ runner invariants


def test_runner_rebuilds_from_the_seed_not_from_0078_output() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Rebuild FROM THE SEED" in source
    assert "--seed-entities" in source
    assert "consumed" in source


def test_runner_never_invents_a_merge_and_pins_the_zero_counters() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for field in (
        '"boundary_expansions": 0',
        '"single_proposal_changes": 0',
        '"type_changes": 0',
        '"assertion_changes": 0',
        '"candidate_policy": "forced_all_null"',
    ):
        assert field in source
    assert '"gpu_used": False' in source
    assert "build_merge_clusters" not in source  # cannot create new clusters


def test_runner_asserts_accepted_union_is_byte_identical() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'assert note[start:end] == merge["union_text"]' in source


def test_runner_refuses_a_truncated_merge_artifact() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "merges_truncated" in source
    assert "complete decision artifact is required" in source


def test_runner_emits_all_required_diagnostic_fields() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for field in (
        "0078_selected_merges",
        "0079_accepted_merges",
        "0079_rejected_merges",
        "entities_in_e3",
        "entities_out_0079",
        "fragments_removed_0079",
        "accepted_by_reason",
        "rejected_by_reason",
        "merge_records",
    ):
        assert field in source
