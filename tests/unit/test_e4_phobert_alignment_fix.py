"""Regression tests for the E4 PhoBERT W2NER Colab alignment fix."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from mednorm_vi.mention_factory.w2ner import EntitySpan
from mednorm_vi.training.phase2.common import sha256_file
from mednorm_vi.training.phase2.e4_w2ner_training import (
    E4TrainingContractError,
    build_w2ner_batch_contract_from_segmented_words,
    prepare_phobert_word_inputs,
    resolve_governed_split_by_sha256,
    validate_phobert_encoder_load_report,
)
from mednorm_vi.training.phobert_alignment import map_segmented_words, segmented_text_to_words

REPO = Path(__file__).resolve().parents[2]
E4_NOTEBOOK = REPO / "notebooks" / "MedNorm_E4_PhoBERT_W2NER_Training.ipynb"


class FakeSlowPhoBERTTokenizer:
    """Slow PhoBERT-like tokenizer with no offset mapping API."""

    is_fast = False
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1

    def tokenize(self, text: str) -> list[str]:
        if text in {"kháng_sinh", "suy_tim"}:
            return [text.split("_", 1)[0], "@@" + text.split("_", 1)[1]]
        return [text]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        vocab = {
            "suy": 10,
            "@@tim": 11,
            "kháng": 12,
            "@@sinh": 13,
            "nặng": 14,
            "ho": 15,
            "cao": 16,
            "sốt": 17,
        }
        return [vocab.get(token, 100 + index) for index, token in enumerate(tokens)]

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [self.cls_token_id, *token_ids, self.sep_token_id]

    def get_special_tokens_mask(
        self,
        token_ids: list[int],
        *,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        assert already_has_special_tokens is False
        return [1, *([0] * len(token_ids)), 1]

    def __call__(self, *_args: Any, **kwargs: Any) -> dict[str, list[int]]:
        if kwargs.get("return_offsets_mapping"):
            raise AssertionError("slow PhoBERT path must never request offset_mapping")
        return {"input_ids": []}


class FakeFastPhoBERTTokenizer:
    is_fast = True
    pad_token_id = 1

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int] | list[tuple[int, int]]]:
        self.last_kwargs = dict(kwargs)
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["truncation"] is False
        assert text == "suy tim"
        return {
            "input_ids": [0, 20, 21, 2],
            "attention_mask": [1, 1, 1, 1],
            "offset_mapping": [(0, 0), (0, 3), (4, 7), (0, 0)],
        }


def _segmented(original: str, segmented: str):
    return map_segmented_words(original, segmented_text_to_words(segmented))


def test_slow_phobert_contract_uses_per_word_subtokens_without_offsets() -> None:
    text = "suy tim nặng"
    words = _segmented(text, "suy_tim nặng")
    contract = build_w2ner_batch_contract_from_segmented_words(
        "doc",
        text,
        (EntitySpan(0, 7, "DIAGNOSIS", "suy tim"),),
        words,
        max_words=8,
    )
    encoding = prepare_phobert_word_inputs(
        FakeSlowPhoBERTTokenizer(),
        contract.segmented_words,
        max_length=8,
    )
    assert encoding.tokenizer_is_fast is False
    assert encoding.consumed_offset_mapping is False
    assert "offset_mapping" not in encoding.model_inputs
    assert encoding.subword_indices_by_word[0] == (1, 2)
    assert encoding.subword_indices_by_word[1] == (3,)
    assert encoding.word_ids[0] is None
    assert encoding.word_ids[4] is None
    assert encoding.word_ids[-1] is None
    assert encoding.model_inputs["attention_mask"][-1] == 0


def test_segmented_underscore_word_decodes_to_exact_original_slice() -> None:
    text = "Bệnh nhân suy tim"
    words = _segmented(text, "Bệnh_nhân suy_tim")
    contract = build_w2ner_batch_contract_from_segmented_words(
        "doc",
        text,
        (EntitySpan(10, 17, "DIAGNOSIS", "suy tim"),),
        words,
        max_words=8,
    )
    assert contract.segmented_words[1].model_text == "suy_tim"
    assert contract.grid.words[1].text == "suy tim"
    assert text[contract.grid.words[1].start:contract.grid.words[1].end] == "suy tim"


def test_repeated_whitespace_newlines_and_decomposed_unicode_offsets() -> None:
    symptom = unicodedata.normalize("NFD", "sốt")
    text = symptom + "   cao\nho"
    words = _segmented(text, text)
    contract = build_w2ner_batch_contract_from_segmented_words(
        "doc",
        text,
        (EntitySpan(0, len(symptom), "SYMPTOM", symptom),),
        words,
        max_words=8,
    )
    assert text[contract.grid.words[0].start:contract.grid.words[0].end] == symptom
    assert text[contract.grid.words[1].start:contract.grid.words[1].end] == "cao"
    assert text[contract.grid.words[2].start:contract.grid.words[2].end] == "ho"


def test_truncation_fails_loudly_before_partial_grid_targets() -> None:
    text = "suy tim nặng"
    words = _segmented(text, "suy_tim nặng")
    contract = build_w2ner_batch_contract_from_segmented_words(
        "doc",
        text,
        (EntitySpan(0, 7, "DIAGNOSIS", "suy tim"),),
        words,
        max_words=8,
    )
    with pytest.raises(E4TrainingContractError, match="refuses silent truncation"):
        prepare_phobert_word_inputs(
            FakeSlowPhoBERTTokenizer(),
            contract.segmented_words,
            max_length=3,
        )


def test_fast_tokenizer_branch_requests_and_removes_offset_mapping() -> None:
    text = "suy tim"
    words = _segmented(text, "suy tim")
    contract = build_w2ner_batch_contract_from_segmented_words(
        "doc",
        text,
        (EntitySpan(0, 7, "DIAGNOSIS", "suy tim"),),
        words,
        max_words=8,
    )
    tokenizer = FakeFastPhoBERTTokenizer()
    encoding = prepare_phobert_word_inputs(tokenizer, contract.segmented_words)
    assert tokenizer.last_kwargs["return_offsets_mapping"] is True
    assert encoding.tokenizer_is_fast is True
    assert encoding.consumed_offset_mapping is True
    assert "offset_mapping" not in encoding.model_inputs
    assert encoding.subword_indices_by_word == ((1,), (2,))


def test_governed_corpus_resolution_uses_hash_not_stale_filename(tmp_path: Path) -> None:
    governed = tmp_path / "arbitrary_governed_split_name.jsonl"
    governed.write_text(json.dumps({"text": "suy tim", "entities": []}) + "\n", encoding="utf-8")
    digest = sha256_file(governed)
    resolved = resolve_governed_split_by_sha256(
        split="train",
        expected_sha256=digest,
        search_roots=(tmp_path,),
    )
    assert resolved.path == governed
    assert resolved.sha256 == digest


def test_phobert_encoder_load_report_accepts_only_expected_mlm_head_keys() -> None:
    report = validate_phobert_encoder_load_report(
        missing_keys=(),
        unexpected_keys=("lm_head.decoder.weight", "lm_head.layer_norm.bias"),
    )
    assert report["w2ner_head_expected_from_base"] is False
    with pytest.raises(E4TrainingContractError):
        validate_phobert_encoder_load_report(
            missing_keys=("encoder.layer.0.attention.self.query.weight",),
            unexpected_keys=(),
        )
    with pytest.raises(E4TrainingContractError):
        validate_phobert_encoder_load_report(
            missing_keys=(),
            unexpected_keys=("encoder.layer.0.output.dense.weight",),
        )


def _e4_notebook_code() -> str:
    doc = json.loads(E4_NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in doc["cells"]
        if cell.get("cell_type") == "code"
    )


def test_e4_notebook_bootstrap_orders_repo_import_before_project_modules() -> None:
    code = _e4_notebook_code()
    assert code.index("drive.mount") < code.index("[\"git\", \"clone\"")
    assert code.index("[\"git\", \"clone\"") < code.index("sys.path.insert")
    assert code.index("sys.path.insert") < code.index("import mednorm_vi")
    assert code.index("import mednorm_vi") < code.index("from mednorm_vi.training.phase2")
    assert code.index("OUTPUT_DIR.mkdir") > code.index("drive.mount")


def test_e4_notebook_hash_revision_and_offset_contracts_are_static() -> None:
    code = _e4_notebook_code()
    assert "public_ner_train.jsonl" not in code
    assert "public_ner_validation.jsonl" not in code
    assert "resolve_e4_governed_splits" in code
    assert code.index("PINNED_MODEL_REVISION = resolve_hf_revision") < code.index(
        "require_resolved_revision(PINNED_MODEL_REVISION"
    )
    assert code.index("require_resolved_revision(PINNED_MODEL_REVISION") < code.index(
        "AutoTokenizer.from_pretrained"
    )
    assert "prepare_phobert_word_inputs(tokenizer" in code
    assert "offset_mapping leaked into encoder inputs" in code
