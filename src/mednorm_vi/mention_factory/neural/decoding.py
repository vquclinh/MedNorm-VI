"""Character-offset decoding for the E3 ViHealthBERT span/type head.

The S1 head is a **multi-label token classifier**: for each subtoken it emits one
independent logit per entity type. Training supervised a subtoken as positive for
type *t* when the subtoken's original character span overlapped a gold entity of
that type (``s1_mention_smoke._labels_from_offsets``), so decoding is the inverse
rule: a maximal run of consecutive positive subtokens for one type is one mention,
and its character span runs from the first subtoken's ``original_start`` to the
last subtoken's ``original_end``.

Because the slow-tokenizer alignment backend gives every subtoken of a segmented
word that word's ORIGINAL character span, the decoded span always lands on real
word boundaries in ``original_text`` — never inside a BPE piece and never on a
normalized copy. Whitespace between two words inside a run is included (it lies
between the two words' spans), whitespace outside the run is not.

Nothing here imports Torch. The caller supplies plain Python lists, which is what
makes every rule below testable without a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...evaluation.exact_mention import (
    PROVENANCE_NEURAL,
    Mention,
    validate_mention,
)
from ...schemas.constants import ENTITY_TYPES

# Source-expert identifier, recorded on every proposal this expert produces.
EXPERT_ID = "E3_vihealthbert_span_type"

# Decoding policy, recorded in reports so a number can never be silently produced
# by a different rule than the one that produced the baseline.
DECODING_POLICY = "maximal_contiguous_positive_subtoken_runs_per_type_v1"


@dataclass(frozen=True, slots=True)
class NeuralSpan:
    """One decoded mention with exact ``original_text`` coordinates.

    ``score`` is the mean per-token probability over the run for this type. It is
    a raw sigmoid output, **not** a calibrated probability.
    """

    start: int
    end: int
    entity_type: str
    text: str
    score: float
    token_count: int

    @property
    def length(self) -> int:
        return self.end - self.start


def decode_type_runs(
    original_text: str,
    offsets: Sequence[tuple[int, int]],
    positives: Sequence[Sequence[int]],
    type_order: Sequence[str],
    *,
    probabilities: Sequence[Sequence[float]] | None = None,
) -> tuple[NeuralSpan, ...]:
    """Decode multi-label token predictions into character-exact mentions.

    ``offsets[i]`` is subtoken *i*'s half-open span in ``original_text``; special
    tokens and padding carry ``(0, 0)`` and are skipped. ``positives[i][t]`` is
    truthy when subtoken *i* fired for ``type_order[t]``.

    A maximal run of consecutive **decodable** subtokens positive for the same
    type becomes one mention. Two runs of the same type separated by a negative
    subtoken stay separate mentions, so a repeated mention at a different offset
    is never collapsed into its earlier occurrence.
    """
    if len(offsets) != len(positives):
        raise ValueError(
            f"offsets ({len(offsets)}) and positives ({len(positives)}) "
            "must describe the same token sequence")
    if probabilities is not None and len(probabilities) != len(offsets):
        raise ValueError("probabilities must describe the same token sequence")
    for entity_type in type_order:
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"unsupported entity type in type_order: {entity_type!r}")

    decoded: list[NeuralSpan] = []
    for type_index, entity_type in enumerate(type_order):
        # Each run is accumulated as (start, end, token count, probability sum) and
        # flushed the moment a decodable token votes negative for this type.
        run: tuple[int, int, int, float] | None = None
        for token_index, (start, end) in enumerate(offsets):
            if end <= start:
                # Special token or padding: it carries no characters, so it can
                # neither extend nor break a run.
                continue
            row = positives[token_index]
            if type_index >= len(row):
                raise ValueError(
                    f"token {token_index} has {len(row)} labels but type_order "
                    f"declares {len(type_order)}")
            if not row[type_index]:
                if run is not None:
                    decoded.append(_finish_run(run, entity_type, original_text))
                    run = None
                continue
            probability = (float(probabilities[token_index][type_index])
                           if probabilities is not None else 1.0)
            if run is None:
                run = (start, end, 1, probability)
            else:
                run = (run[0], max(run[1], end), run[2] + 1, run[3] + probability)
        if run is not None:
            decoded.append(_finish_run(run, entity_type, original_text))

    decoded.sort(key=lambda span: (span.start, span.end, span.entity_type))
    return tuple(decoded)


def _finish_run(
    run: tuple[int, int, int, float], entity_type: str, original_text: str,
) -> NeuralSpan:
    start, end, tokens, probability = run
    return NeuralSpan(
        start=start, end=end, entity_type=entity_type,
        text=original_text[start:end],
        score=round(probability / tokens, 6) if tokens else 0.0,
        token_count=tokens)


def validate_decoded_spans(spans: Sequence[NeuralSpan], original_text: str) -> None:
    """Fail loudly if any decoded span violates spec §4."""
    for span in spans:
        if span.start < 0 or span.end <= span.start:
            raise ValueError(
                f"decoded span [{span.start}, {span.end}) is not a non-empty "
                "end-exclusive interval")
        if span.end > len(original_text):
            raise ValueError(
                f"decoded span end {span.end} exceeds the text length "
                f"{len(original_text)}")
        if original_text[span.start:span.end] != span.text:
            raise ValueError(
                "decoded span violates original_text[start:end] == text at "
                f"[{span.start}, {span.end})")


def neural_spans_to_mentions(
    spans: Sequence[NeuralSpan],
    original_text: str,
    *,
    route: str = "",
    section: str = "",
) -> tuple[Mention, ...]:
    """Adapt decoded spans to the Audit 0033 exact evaluator's ``Mention``."""
    mentions = tuple(
        Mention(span.start, span.end, span.entity_type, span.text,
                route=route, section=section, provenance=PROVENANCE_NEURAL)
        for span in spans
    )
    for mention in mentions:
        validate_mention(mention, original_text, role="predicted")
    return mentions


__all__ = [
    "DECODING_POLICY",
    "EXPERT_ID",
    "NeuralSpan",
    "decode_type_runs",
    "neural_spans_to_mentions",
    "validate_decoded_spans",
]
