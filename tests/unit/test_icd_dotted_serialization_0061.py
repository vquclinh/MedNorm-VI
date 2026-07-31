"""Dotted ICD-10 output is canonical for competition submissions (Audit 0061).

`UNRES-ICD-DOTTED` was resolved by an isolated single-variable leaderboard probe. The
dotless Profile A scored 9.1625; the dotted probe, differing **only** by the reversible
dot insertion, scored 9.5736 — with WER and J_assertion unchanged to four decimals and
J_candidates rising 1.2119 -> 2.2396. A zero delta on the two components the probe did
not touch is what makes the attribution airtight.

These tests hold the resolution in place. The distinction they protect is narrow and
easy to erode: **dotted is an output serialization, dotless remains the KB concept
identity**. A future change that started storing dots in the index, or that let the dot
leak into offered-set membership or ranking, would break the L9 offered-set contract
while still looking correct in a submission file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mednorm_vi.inference.code_format import classify_icd_code, to_dotless, to_dotted

REPO = Path(__file__).resolve().parents[2]
UNRESOLVED = REPO / "configs" / "organizer" / "unresolved_policies_v1.yaml"
HYPOTHESES = REPO / "configs" / "organizer" / "icd_format_hypotheses_v1.yaml"
ICD_INDEX = REPO / "indices" / "candidate" / "icd10_vi" / "competition-v3" / "index.json"

# The exact leaderboard evidence that resolved the policy.
DOTLESS_SCORE, DOTTED_SCORE = 9.1625, 9.5736
DOTLESS_JCAND, DOTTED_JCAND = 1.2119, 2.2396
SHARED_WER, SHARED_JASSERT = 87.0976, 16.0234


def test_the_governed_policy_records_dotted_as_resolved() -> None:
    policies = yaml.safe_load(UNRESOLVED.read_text(encoding="utf-8"))["policies"]
    dotted = next(p for p in policies if p["policy_id"] == "UNRES-ICD-DOTTED")
    assert dotted["status"] == "resolved"
    assert dotted["resolution"] == "dotted"
    assert dotted["internal_default"] == "dotted"
    assert dotted["confidence"] == "high"
    assert dotted["leaderboard_experiment_id"] == "EXP-ICD-DOTTED-0060"


def test_the_hypotheses_registry_confirms_dotted_and_rejects_dotless_output() -> None:
    entries = yaml.safe_load(HYPOTHESES.read_text(encoding="utf-8"))["hypotheses"]
    by_id = {e["policy_id"]: e for e in entries}
    assert by_id["ICDF-DOTTED-OUTPUT"]["status"] == "CONFIRMED"
    assert by_id["ICDF-UNDOTTED-OUTPUT"]["status"] == "REJECTED_FOR_OUTPUT"
    # Rejected for OUTPUT only — dotless is still the internal identity.
    assert "internal" in by_id["ICDF-UNDOTTED-OUTPUT"]["note"].lower()


def test_the_probe_arithmetic_reproduces_both_published_scores() -> None:
    """The metric is `0.3·(100−WER) + 0.3·J_assert + 0.4·J_cand`, confirmed twice."""

    def score(wer: float, assertion: float, candidates: float) -> float:
        return 0.3 * (100 - wer) + 0.3 * assertion + 0.4 * candidates

    assert round(score(SHARED_WER, SHARED_JASSERT, DOTLESS_JCAND), 4) == DOTLESS_SCORE
    assert round(score(SHARED_WER, SHARED_JASSERT, DOTTED_JCAND), 4) == DOTTED_SCORE
    # The entire gain is attributable to the candidate component and nothing else.
    delta = round(DOTTED_SCORE - DOTLESS_SCORE, 4)
    assert delta == round(0.4 * (DOTTED_JCAND - DOTLESS_JCAND), 4) == 0.4111


def test_dotted_output_is_reversible_to_the_internal_dotless_identity() -> None:
    """Requirement 8: the submission form must map back to the KB key exactly."""
    for internal in ("I269", "I509", "I248", "I7000", "K650", "N03", "A00", "M4802"):
        emitted = to_dotted(internal)
        assert to_dotless(emitted) == internal, "round-trip must recover the KB identity"


def test_three_character_categories_are_emitted_unchanged() -> None:
    """Requirement 5. A category has no subdivision to separate."""
    for code in ("N03", "B01", "B36", "A00", "Z00"):
        assert to_dotted(code) == code


@pytest.mark.skipif(not ICD_INDEX.is_file(), reason="competition v3 ICD index not built locally")
def test_the_kb_identity_remains_dotless() -> None:
    """Requirement 2 and 4: the index key is unchanged by the serialization decision."""
    codes = [r["concept_id"] for r in json.loads(ICD_INDEX.read_text(encoding="utf-8"))["records"]]
    assert codes
    assert all("." not in code for code in codes), "KB concept identity must stay dotless"
    assert all(classify_icd_code(code) == "dotless" for code in codes)
    # And every one of them is emittable in dotted form without loss.
    assert all(to_dotless(to_dotted(code)) == code for code in codes)


def test_rxnorm_candidates_are_never_dotted() -> None:
    """Requirement 6. RxCUIs are numeric identifiers, not ICD codes."""
    import tempfile

    from mednorm_vi.inference.derive_submission import DERIVE_ICD_DOTTED, derive_submission

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "src"
        src.mkdir()
        (src / "1.json").write_text(
            json.dumps(
                [
                    {
                        "text": "x",
                        "type": "THUỐC",
                        "candidates": ["161", "1043578"],
                        "assertions": [],
                        "position": [0, 1],
                    },
                    {
                        "text": "y",
                        "type": "CHẨN_ĐOÁN",
                        "candidates": ["I269"],
                        "assertions": [],
                        "position": [1, 2],
                    },
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        out = root / "d" / "output"
        derive_submission(
            source_dir=src, output_dir=out, derivation=DERIVE_ICD_DOTTED, expected_documents=1
        )
        got = json.loads((out / "1.json").read_text(encoding="utf-8"))
        assert got[0]["candidates"] == ["161", "1043578"], "RxNorm must be untouched"
        assert got[1]["candidates"] == ["I26.9"]


def test_the_historical_dotless_baselines_are_unmodified() -> None:
    """Requirement 7: the scored baselines stay read-only historical evidence."""
    expected = {
        "runs/baseline_v0_profile_a/output.zip":
            "bbf12cd5d5c293bfd8904ca9217e987e570604aa411acfe0db4fb33d91f30615",
        "runs/baseline_v0_profile_b/output.zip":
            "f3f63af9c5ee6afe59ec5909587c7b6d0288f68a4fd67dc0e71d0523e6532c3e",
        "runs/baseline_v0_profile_c/output.zip":
            "29ef6670a6cf01d7a98567a11601fa59c4e3e8ad191887d0ec8b54871fbebe3d",
    }  # fmt: skip
    import hashlib

    for rel, digest in expected.items():
        path = REPO / rel
        if not path.is_file():
            pytest.skip(f"{rel} not present locally")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, f"{rel} was modified; scored artifacts are immutable"
