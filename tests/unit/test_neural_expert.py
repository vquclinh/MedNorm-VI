"""E3 character-offset decoding and the forward-only execution boundary (0034)."""

from __future__ import annotations

import ast
import unicodedata
from pathlib import Path

import pytest

from mednorm_vi.evaluation.exact_mention import PROVENANCE_NEURAL, validate_mention
from mednorm_vi.mention_factory.neural import runtime
from mednorm_vi.mention_factory.neural.decoding import (
    DECODING_POLICY,
    EXPERT_ID,
    NeuralSpan,
    decode_type_runs,
    neural_spans_to_mentions,
    validate_decoded_spans,
)
from mednorm_vi.mention_factory.neural.runtime import (
    CheckpointIntegrityError,
    encode_for_inference,
    fingerprint_checkpoint,
)

TYPES = ("DIAGNOSIS", "MEDICATION", "SYMPTOM", "TEST_NAME", "TEST_RESULT")
REPO = Path(__file__).resolve().parents[2]


def _offsets(*pairs: tuple[int, int]) -> list[tuple[int, int]]:
    return [(0, 0), *pairs, (0, 0)]  # CLS ... SEP


def _positives(rows: list[str]) -> list[list[int]]:
    """Each row is a type name or '' for a negative token; specials are all-zero."""
    out = [[0] * len(TYPES)]
    for row in rows:
        vector = [0] * len(TYPES)
        if row:
            vector[TYPES.index(row)] = 1
        out.append(vector)
    out.append([0] * len(TYPES))
    return out


# -- decoding -------------------------------------------------------------------


def test_contiguous_positive_run_becomes_one_mention() -> None:
    text = "bệnh nhân suy tim nặng"
    spans = decode_type_runs(
        text, _offsets((0, 4), (5, 9), (10, 13), (14, 17), (18, 22)),
        _positives(["", "", "DIAGNOSIS", "DIAGNOSIS", ""]), TYPES)
    assert [(s.start, s.end, s.entity_type, s.text) for s in spans] == [
        (10, 17, "DIAGNOSIS", "suy tim")]


def test_offsets_are_end_exclusive_and_slice_back_exactly() -> None:
    text = "suy tim"
    spans = decode_type_runs(
        text, _offsets((0, 3), (4, 7)), _positives(["DIAGNOSIS", "DIAGNOSIS"]), TYPES)
    span = spans[0]
    assert (span.start, span.end) == (0, 7)
    assert text[span.start:span.end] == span.text
    validate_decoded_spans(spans, text)


def test_two_runs_separated_by_a_negative_token_stay_two_mentions() -> None:
    text = "táo bón và táo bón"
    spans = decode_type_runs(
        text, _offsets((0, 3), (4, 7), (8, 10), (11, 14), (15, 18)),
        _positives(["SYMPTOM", "SYMPTOM", "", "SYMPTOM", "SYMPTOM"]), TYPES)
    assert [(s.start, s.end) for s in spans] == [(0, 7), (11, 18)]
    assert spans[0].text == spans[1].text  # same surface, different offsets


def test_two_types_on_the_same_token_decode_to_two_mentions() -> None:
    text = "sốt"
    row = [0] * len(TYPES)
    row[TYPES.index("SYMPTOM")] = 1
    row[TYPES.index("DIAGNOSIS")] = 1
    spans = decode_type_runs(
        text, _offsets((0, 3)), [[0] * len(TYPES), row, [0] * len(TYPES)], TYPES)
    assert {s.entity_type for s in spans} == {"SYMPTOM", "DIAGNOSIS"}
    assert all(s.start == 0 and s.end == 3 for s in spans)


def test_special_tokens_neither_extend_nor_break_a_run() -> None:
    text = "suy tim"
    offsets = [(0, 0), (0, 3), (0, 0), (4, 7), (0, 0)]
    positives = [[0] * len(TYPES)] * 5
    positives = [
        [0] * len(TYPES),
        [1 if t == "DIAGNOSIS" else 0 for t in TYPES],
        [0] * len(TYPES),
        [1 if t == "DIAGNOSIS" else 0 for t in TYPES],
        [0] * len(TYPES),
    ]
    spans = decode_type_runs(text, offsets, positives, TYPES)
    assert [(s.start, s.end) for s in spans] == [(0, 7)]


def test_decomposed_unicode_decodes_to_exact_offsets() -> None:
    text = unicodedata.normalize("NFD", "sốt cao")
    boundary = text.index(" ")
    spans = decode_type_runs(
        text, _offsets((0, boundary), (boundary + 1, len(text))),
        _positives(["SYMPTOM", "SYMPTOM"]), TYPES)
    assert text[spans[0].start:spans[0].end] == spans[0].text == text


def test_repeated_whitespace_inside_a_run_is_included_but_not_outside() -> None:
    text = "  ho   khan  "
    spans = decode_type_runs(
        text, _offsets((2, 4), (7, 11)), _positives(["SYMPTOM", "SYMPTOM"]), TYPES)
    assert (spans[0].start, spans[0].end) == (2, 11)
    assert spans[0].text == "ho   khan"


def test_score_is_the_mean_probability_over_the_run() -> None:
    text = "suy tim"
    probabilities = [
        [0.0] * len(TYPES),
        [0.8 if t == "DIAGNOSIS" else 0.0 for t in TYPES],
        [0.6 if t == "DIAGNOSIS" else 0.0 for t in TYPES],
        [0.0] * len(TYPES),
    ]
    spans = decode_type_runs(
        text, _offsets((0, 3), (4, 7)), _positives(["DIAGNOSIS", "DIAGNOSIS"]),
        TYPES, probabilities=probabilities)
    assert spans[0].score == pytest.approx(0.7)
    assert spans[0].token_count == 2


def test_mismatched_sequence_lengths_fail_loudly() -> None:
    with pytest.raises(ValueError):
        decode_type_runs("abc", [(0, 3)], [[0] * len(TYPES), [0] * len(TYPES)], TYPES)


def test_unsupported_type_order_fails_loudly() -> None:
    with pytest.raises(ValueError):
        decode_type_runs("abc", [(0, 3)], [[1]], ("NOT_A_TYPE",))


def test_a_decoded_span_that_does_not_slice_back_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_decoded_spans([NeuralSpan(0, 3, "SYMPTOM", "XXX", 0.9, 1)], "abc")


def test_mentions_carry_neural_provenance_and_pass_the_evaluator_invariant() -> None:
    text = "suy tim"
    mentions = neural_spans_to_mentions(
        [NeuralSpan(0, 7, "DIAGNOSIS", "suy tim", 0.9, 2)], text)
    assert mentions[0].provenance == PROVENANCE_NEURAL
    validate_mention(mentions[0], text, role="predicted")


def test_decoding_policy_and_expert_id_are_stable_identifiers() -> None:
    assert DECODING_POLICY == "maximal_contiguous_positive_subtoken_runs_per_type_v1"
    assert EXPERT_ID == "E3_vihealthbert_span_type"


# -- inference-only encoding ----------------------------------------------------


class _FakeSlowTokenizer:
    """Whitespace tokenizer with the slow-tokenizer surface the aligner needs."""

    is_fast = False
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1

    def tokenize(self, text: str) -> list[str]:
        return [text]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [10 + index for index, _ in enumerate(tokens)]


def test_inference_encoding_attaches_original_character_spans() -> None:
    text = "suy tim nặng"
    encoded = encode_for_inference(
        text, _FakeSlowTokenizer(), max_length=32, segmenter=None, example_id="x")
    body = [pair for pair in encoded.offsets if pair != (0, 0)]
    assert body == [(0, 3), (4, 7), (8, 12)]
    for start, end in body:
        assert text[start:end] in text
    assert encoded.example_id == "x"


def test_inference_encoding_builds_no_labels() -> None:
    encoded = encode_for_inference(
        "suy tim", _FakeSlowTokenizer(), max_length=32, segmenter=None)
    assert not hasattr(encoded, "labels")
    assert not hasattr(encoded, "label_mask")


# -- checkpoint custody ---------------------------------------------------------


def test_fingerprint_detects_a_digest_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "fake.pt"
    target.write_bytes(b"weights")
    with pytest.raises(CheckpointIntegrityError):
        fingerprint_checkpoint(target, expected_sha256="0" * 64)


def test_fingerprint_round_trip_is_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "fake.pt"
    target.write_bytes(b"weights")
    before = fingerprint_checkpoint(target)
    before.assert_unchanged(fingerprint_checkpoint(target))


def test_a_mutated_checkpoint_is_detected(tmp_path: Path) -> None:
    target = tmp_path / "fake.pt"
    target.write_bytes(b"weights")
    before = fingerprint_checkpoint(target)
    target.write_bytes(b"tampered")
    with pytest.raises(CheckpointIntegrityError):
        before.assert_unchanged(fingerprint_checkpoint(target))


def test_missing_checkpoint_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(CheckpointIntegrityError):
        fingerprint_checkpoint(tmp_path / "absent.pt")


# -- the execution boundary: no training, ever ----------------------------------

_FORBIDDEN_CALLS = ("backward", "step", "zero_grad", "train")
_FORBIDDEN_NAMES = ("AdamW", "Adam", "SGD", "get_linear_schedule_with_warmup")


def _module_sources() -> list[tuple[str, str]]:
    root = Path(runtime.__file__).parent
    return [(str(path), path.read_text(encoding="utf-8"))
            for path in sorted(root.glob("*.py"))]


def test_no_backward_optimizer_or_scheduler_anywhere_in_the_neural_expert() -> None:
    for path, source in _module_sources():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in _FORBIDDEN_CALLS, f"{path}: .{node.attr}()"
            if isinstance(node, ast.Name):
                assert node.id not in _FORBIDDEN_NAMES, f"{path}: {node.id}"


def test_runtime_declares_forward_only_execution() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "torch.no_grad()" in source
    assert "model.eval()" in source or ".eval()" in source
    assert "requires_grad_(False)" in source


def test_repository_tracks_no_model_weights() -> None:
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    for line in tracked.splitlines():
        assert not line.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".bin"))
