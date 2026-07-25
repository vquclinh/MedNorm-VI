"""Annotation-coverage model and masked-loss collation contract (Audit 0017).

Different source corpora annotate different things. The trainer must NOT penalize
a model for an objective the source never annotated (e.g. assertions on a corpus
without assertion labels), and must NOT treat unannotated entity types as true
negatives. This module records, per source, which supervision signals are
reliable, and derives per-example loss masks so training stages S1/S2/S3 collate
only supervised objectives.

Pure and offline — no model is loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The trainable objectives a loss mask can gate.
OBJECTIVES = ("span", "entity_type", "assertions", "icd_candidates", "rxnorm_candidates")


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """Which supervision a source dataset reliably provides."""

    source_dataset: str
    span: bool = False
    entity_type: bool = False
    assertions: bool = False
    icd_candidates: bool = False
    rxnorm_candidates: bool = False
    test_result_pairing: bool = False
    patient_context: bool = False
    relations: bool = False

    def loss_mask(self) -> dict[str, bool]:
        """Objectives that MAY contribute to the loss for this source.

        Absent supervision => masked out (never penalized, never treated as a
        true negative)."""
        return {
            "span": self.span,
            "entity_type": self.entity_type,
            "assertions": self.assertions,
            "icd_candidates": self.icd_candidates,
            "rxnorm_candidates": self.rxnorm_candidates,
        }


def coverage_from_manifest_fields(
    source_dataset: str,
    *,
    has_typed_entities: bool,
    assertion_availability: str = "none",
    normalization_availability: str = "none",
) -> SourceCoverage:
    """Build a SourceCoverage from NER-manifest-style fields."""
    return SourceCoverage(
        source_dataset=source_dataset,
        span=has_typed_entities,
        entity_type=has_typed_entities,
        assertions=assertion_availability != "none",
        icd_candidates=normalization_availability in ("icd", "both"),
        rxnorm_candidates=normalization_availability in ("rxnorm", "both"),
    )


def example_loss_mask(example: dict[str, Any], coverage: SourceCoverage) -> dict[str, bool]:
    """Per-example loss mask.

    Span/type are supervised only if the source provides typed entities AND the
    example actually carries at least one entity (an empty entity list on a
    typed-coverage source is a real 'no entities here' signal only when the source
    annotates that type exhaustively; we conservatively keep span/type supervised
    when coverage.span is True). Assertion/candidate objectives follow coverage and
    are additionally gated on the presence of that annotation on the example."""
    mask = coverage.loss_mask()
    # Assertions/candidates require the example to actually carry them; otherwise
    # masked (missing != negative).
    has_assert = any(e.get("assertions") for e in example.get("entities", []))
    has_icd = any(
        e.get("candidates") and e.get("target_type") == "DIAGNOSIS"
        for e in example.get("entities", [])
    )
    has_rxn = any(
        e.get("candidates") and e.get("target_type") == "MEDICATION"
        for e in example.get("entities", [])
    )
    mask["assertions"] = mask["assertions"] and has_assert
    mask["icd_candidates"] = mask["icd_candidates"] and has_icd
    mask["rxnorm_candidates"] = mask["rxnorm_candidates"] and has_rxn
    return mask


__all__ = ["OBJECTIVES", "SourceCoverage", "coverage_from_manifest_fields", "example_loss_mask"]
