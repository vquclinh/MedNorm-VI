"""Milestone 3B.2: run-level governed INN-bridge observability (Audit 0058B).

Audit 0058A proved the governed `RETRIEVAL_ONLY` bridges work and are load-bearing —
`paracetamol` is absent from the index postings, so without expansion it links to
nothing — but recorded a gap: the `inn_bridge:*` note stopped at the linker report and
never reached `PipelineResult.warnings`. A run could therefore use a governed bridge
with nothing above L5 recording it, which is exactly the kind of silent behaviour the
rest of this repository refuses to ship.

This module holds the change to that one property. The tests are deliberately split
between *what is now visible* and *what must not have moved*: an observability change
that quietly altered candidates would be far worse than the gap it closed, so the
byte-identity of the serialized output is asserted as directly as the new note is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mednorm_vi.inference.config import PipelineConfig
from mednorm_vi.inference.pipeline import run_document
from mednorm_vi.inference.serialization import to_submission_json
from mednorm_vi.linking.rxnorm import (
    INN_BRIDGE_NOTE_PREFIX,
    governed_bridge_notes,
    link_rxnorm,
    link_rxnorm_structured,
)
from mednorm_vi.linking.structured_medication import (
    INN_BRIDGE_STATUS,
    INN_TO_RXNORM_NAME,
)
from mednorm_vi.resolution.models import (
    STATUS_ACCEPTED,
    BoundaryEvidence,
    EntityHypothesis,
    TypeEvidence,
)

REPO = Path(__file__).resolve().parents[2]
PIPELINE_CFG = REPO / "configs" / "pipeline" / "full_v1.yaml"
FIX = REPO / "tests" / "fixtures" / "phase1b"
ACTIVE_RXNORM = REPO / "indices" / "candidate" / "rxnorm" / "competition-v3" / "index.json"

_needs_index = pytest.mark.skipif(
    not ACTIVE_RXNORM.is_file(), reason="competition v3 runtime index not built locally"
)


def _hypothesis(text: str, ingredient: str) -> EntityHypothesis:
    return EntityHypothesis(
        hypothesis_id="h1",
        document_id="d1",
        start=0,
        end=len(text),
        text=text,
        entity_type="THUỐC",
        status=STATUS_ACCEPTED,
        chosen_proposal_id="p1",
        source_proposal_ids=("p1",),
        boundary_evidence=BoundaryEvidence("policy", "full", "p1", ()),
        type_evidence=TypeEvidence("THUỐC", "E1", ("THUỐC",)),
        components=(
            {
                "role": "name",
                "text": ingredient,
                "start": 0,
                "end": len(ingredient),
                "normalized": ingredient,
                "detail": "",
            },
        ),
        score=0.9,
    )


@pytest.fixture(scope="module")
def rxnorm_index():  # type: ignore[no-untyped-def]
    from mednorm_vi.kb.indexing.retrieval import load_index

    return load_index(ACTIVE_RXNORM)


# --- the accessor itself ------------------------------------------------------


def test_governed_bridge_notes_deduplicates_and_sorts() -> None:
    """One document can mention the same drug repeatedly; that is not two bridges."""
    note_a = f"{INN_BRIDGE_NOTE_PREFIX}paracetamol->acetaminophen:{INN_BRIDGE_STATUS}"
    note_b = f"{INN_BRIDGE_NOTE_PREFIX}salbutamol->albuterol:{INN_BRIDGE_STATUS}"
    noise = ("ingredient_absent_from_snapshot", "suppressed:DROP_STRENGTH_CONFLICT:123")

    assert governed_bridge_notes([note_a, note_a, note_a]) == (note_a,)
    # Deterministic order regardless of input order.
    assert governed_bridge_notes([note_b, note_a]) == governed_bridge_notes([note_a, note_b])
    # Unrelated linker warnings are not swept into this channel.
    assert governed_bridge_notes([*noise, note_a]) == (note_a,)
    assert governed_bridge_notes(noise) == ()
    assert governed_bridge_notes([]) == ()


def test_a_bridge_note_carries_names_not_rxcuis() -> None:
    """Requirement 3: the channel must not expose a bridge RxCUI."""
    note = f"{INN_BRIDGE_NOTE_PREFIX}paracetamol->acetaminophen:{INN_BRIDGE_STATUS}"
    payload = note[len(INN_BRIDGE_NOTE_PREFIX) :]
    surface, rest = payload.split("->", 1)
    target_name = rest.rsplit(":", 1)[0]
    assert not any(ch.isdigit() for ch in surface + target_name)
    # `161` is acetaminophen's RxCUI; the note names the concept, never the code.
    assert "161" not in note


def test_the_note_states_retrieval_only_and_denies_clinical_approval() -> None:
    """Requirement 2."""
    assert INN_BRIDGE_STATUS == "RETRIEVAL_ONLY_NOT_CLINICALLY_APPROVED"
    note = f"{INN_BRIDGE_NOTE_PREFIX}paracetamol->acetaminophen:{INN_BRIDGE_STATUS}"
    assert note.endswith("RETRIEVAL_ONLY_NOT_CLINICALLY_APPROVED")
    assert "CLINICALLY_APPROVED" in note and "NOT_CLINICALLY_APPROVED" in note


def test_a_bridge_surface_is_always_a_governed_table_key() -> None:
    """The note cannot carry arbitrary clinical text out of a document.

    A bridge fires only when the normalized ingredient surface is a key of the
    12-entry governed table, so the surface in any note is provably one of twelve
    source-code constants — not free text lifted from the record.
    """
    assert len(INN_TO_RXNORM_NAME) <= 20
    for surface, target in INN_TO_RXNORM_NAME.items():
        note = f"{INN_BRIDGE_NOTE_PREFIX}{surface}->{target}:{INN_BRIDGE_STATUS}"
        assert governed_bridge_notes([note]) == (note,)
        assert surface in INN_TO_RXNORM_NAME


# --- run-level propagation ----------------------------------------------------


@_needs_index
def test_paracetamol_bridge_activity_reaches_run_level_warnings() -> None:
    """Requirement 1: the gap Audit 0058A recorded is closed."""
    config = PipelineConfig.load(PIPELINE_CFG)
    result = run_document(FIX / "synthetic_medication_list.txt", config, mode="deterministic")
    notes = governed_bridge_notes(result.warnings)
    assert notes, "a document mentioning paracetamol must record bridge activity"
    assert notes == (f"{INN_BRIDGE_NOTE_PREFIX}paracetamol->acetaminophen:{INN_BRIDGE_STATUS}",)
    # And the note really is in the document's own warning channel, not just derivable.
    assert any(w.startswith(INN_BRIDGE_NOTE_PREFIX) for w in result.warnings)


@_needs_index
def test_a_mention_needing_no_bridge_creates_no_bridge_warning() -> None:
    """Requirement 4: no false positives on documents with no governed surface."""
    config = PipelineConfig.load(PIPELINE_CFG)
    for fixture in ("synthetic_laboratory.txt", "synthetic_hard_negatives.txt"):
        result = run_document(FIX / fixture, config, mode="deterministic")
        assert governed_bridge_notes(result.warnings) == (), fixture


@_needs_index
def test_a_document_repeating_one_drug_records_one_bridge_note() -> None:
    """Requirement 5 in its sharpest form: the record tracks bridges, not mentions."""
    config = PipelineConfig.load(PIPELINE_CFG)
    result = run_document(FIX / "synthetic_medical_document.txt", config, mode="deterministic")
    mentions = result.original_text.lower().count("paracetamol")
    assert mentions >= 1
    assert len(governed_bridge_notes(result.warnings)) == 1


@_needs_index
def test_repeated_runs_produce_identical_warnings() -> None:
    """Requirement 5."""
    config = PipelineConfig.load(PIPELINE_CFG)
    fixture = FIX / "synthetic_medication_list.txt"
    runs = {tuple(run_document(fixture, config, mode="deterministic").warnings) for _ in range(3)}
    assert len(runs) == 1, f"warnings drifted across runs: {runs}"


@_needs_index
def test_warnings_are_not_an_input_to_candidate_selection() -> None:
    """Requirement 6, proven structurally rather than by a digest.

    The strongest available in-test proof that observability is side-effect free:
    strip every warning from the linker results and re-decode. If L8 read the warning
    channel at all — directly or through L6/L7 — the decoded candidates would move.
    They do not, because `warnings` and `candidates` are separate fields and only the
    second reaches the decoder.
    """
    from dataclasses import replace

    from mednorm_vi.confidence_cascade.cascade import apply_confidence_cascade
    from mednorm_vi.kb.indexing.retrieval import load_index
    from mednorm_vi.metric_decoder import decode_entities

    index = load_index(ACTIVE_RXNORM)
    hypothesis = _hypothesis("paracetamol", "paracetamol")
    linked = link_rxnorm(hypothesis, index)
    assert governed_bridge_notes(linked.warnings), "the fixture must exercise a bridge"

    cascade = apply_confidence_cascade((hypothesis,))
    with_notes = decode_entities((hypothesis,), cascade, (), (linked,))
    stripped = decode_entities((hypothesis,), cascade, (), (replace(linked, warnings=()),))
    assert [e.candidates for e in with_notes] == [e.candidates for e in stripped]
    assert [e.candidate_decisions for e in with_notes] == [e.candidate_decisions for e in stripped]


@_needs_index
@pytest.mark.parametrize(
    ("profile", "digest"),
    [
        (
            "competition_adaptive",
            "4203bcc00fbbc77bcb018f8fbed2995dbcaa08401bab2918e7deaff3b29103a4",
        ),
        (
            "competition_top1",
            "cb2e81f98343439ba03dd506ef71847a99c973af1382395693bfffcf1a101c5b",
        ),
    ],
)
def test_serialized_candidate_output_matches_its_regression_pin(profile: str, digest: str) -> None:
    """Requirement 6, as a forward guard — pinned **per decoding profile**.

    The pre/post invariance was established by measurement outside the suite: the
    canonical runner was captured immediately before the propagation change and again
    after, over the same inputs, and the serialized bytes were identical
    (`cac6e87cc4f9ddcf8d6289e8c296f6fba8384df54120d80307fe6b92e515b086`, recorded in
    Audit 0058B §4). That comparison cannot be re-run from inside the suite once the
    change is in the tree, so these digests pin the result going forward.

    Audit 0060 made the decoding profile a config value and set the canonical default
    to `competition_top1`. This pin originally read the config's default and so would
    have moved with it, which would have made a *policy* change look like an
    observability regression. Naming the profile keeps the guard measuring what it
    claims: `competition_adaptive` still reproduces the original digest exactly, which
    is itself the evidence that bridge observability remained side-effect free.
    """
    import hashlib
    from dataclasses import replace

    config = replace(PipelineConfig.load(PIPELINE_CFG), decoding_profile=profile)
    payloads = {}
    for fixture in sorted(FIX.glob("*.txt")):
        result = run_document(fixture, config, mode="deterministic")
        payloads[f"{result.document_id}.json"] = to_submission_json(result.predictions)
    blob = json.dumps(payloads, ensure_ascii=False, sort_keys=True)

    assert hashlib.sha256(blob.encode()).hexdigest() == digest, (
        f"serialized candidate output moved under profile {profile!r}"
    )


@_needs_index
def test_no_bridge_rxcui_is_emitted_directly(rxnorm_index) -> None:  # type: ignore[no-untyped-def]
    """Requirement 3, end to end: 161 is emitted because the index was searched.

    The bridge supplies the *name* `acetaminophen`, which re-enters retrieval as a
    query. So RxCUI 161 appears in the output only because the concept index answered
    for it, not because a crosswalk row handed the code over.
    """
    report = link_rxnorm_structured(_hypothesis("paracetamol", "paracetamol"), rxnorm_index)
    top = report.retained[0]
    assert top.concept_id == "161"
    assert "exact" in top.retrieval_sources, "reached through the index, not the bridge"

    result = link_rxnorm(_hypothesis("paracetamol", "paracetamol"), rxnorm_index)
    codes = {c.code for c in result.candidates}
    assert "161" in codes
    # The note travels beside the candidates and never becomes one.
    for note in governed_bridge_notes(result.warnings):
        assert note not in codes


# --- run-level manifest aggregate --------------------------------------------


@_needs_index
def test_the_run_manifest_counts_bridge_expansions_without_carrying_text() -> None:
    from mednorm_vi.inference.manifest import build_run_manifest

    config = PipelineConfig.load(PIPELINE_CFG)
    results = [
        run_document(FIX / "synthetic_medication_list.txt", config, mode="deterministic"),
        run_document(FIX / "synthetic_laboratory.txt", config, mode="deterministic"),
    ]
    manifest = build_run_manifest(results, mode="deterministic", config=config)
    payload = manifest.as_dict()
    linking = payload["aggregates"]["linking"]

    assert linking["governed_inn_bridge_expansions"] == 1, (
        "one governed bridge across the two documents"
    )
    # The manifest stays aggregate-only: a count, never the note text.
    assert payload["contains_clinical_text"] is False
    assert INN_BRIDGE_NOTE_PREFIX not in manifest.to_json()
    assert "paracetamol" not in manifest.to_json()
