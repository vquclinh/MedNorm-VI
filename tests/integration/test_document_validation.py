"""Integration: end-to-end deterministic validation of a small document.

Exercises the composed ``validate_document`` guard over a realistic-looking
(hand-built) Vietnamese clinical snippet with repeated surface forms. No models
are involved — this only checks the Phase-0 deterministic foundation.
"""

from __future__ import annotations

import json
from pathlib import Path

from mednorm_vi.validator import validate_document

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_document.json"


def test_fixture_document_is_internally_consistent() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    text = data["original_text"]
    entities = data["entities"]

    result = validate_document(text, entities, document_id=data["document_id"])
    assert result.ok, [i.message for i in result.errors]

    # Every entity's offsets slice back to its text (the mandatory invariant).
    for e in entities:
        start, end = e["position"]
        assert text[start:end] == e["text"]


def test_fixture_has_a_repeated_distinct_mention() -> None:
    # The fixture intentionally contains the same surface form twice at
    # different offsets; both must survive as distinct entities.
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    entities = data["entities"]
    hos = [e for e in entities if e["text"] == "ho"]
    assert len(hos) == 2
    assert hos[0]["position"] != hos[1]["position"]
