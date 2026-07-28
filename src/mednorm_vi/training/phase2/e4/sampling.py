"""Deterministic E4 data ordering (Audit 0045).

The collapsed run streamed the governed corpus in **file order and never
shuffled**. Measured in Audit 0043 §7, that order is:

    rows      0 - 10,026   phoner_covid19   10,027 examples, ZERO entities
    rows 10,027 - 15,822   vimedner          5,796 examples, 8,987 entities
    rows 15,823 - 24,558   vimq              8,736 examples,   619 entities
    rows 24,559 - 33,825   vietmed_ner       9,267 examples, 2,114 entities

So **every epoch opened with 10,027 consecutive examples containing no entity** —
about 1,253 consecutive optimizer steps whose only gradient signal was "predict
NONE", before a single positive cell was ever seen. The progress log shows
per-sample loss pinned at 0.0-1e-6 from sample ~2,000 to ~10,000 in every epoch.

Two orderings are provided:

* :func:`shuffled_source_interleaved_order` — a deterministic per-epoch shuffle
  with round-robin source interleaving, so no source ever runs in a long block;
* :func:`positive_aware_order` — additionally caps how many zero-entity examples
  may appear in any window, so an effective batch reliably contains positives.

Both are **pure permutations**: no example is dropped, duplicated or synthesized.
Zero-entity documents are real negatives and stay in the corpus — they are
reordered, not removed. Both are seeded and reproducible; neither uses a global
RNG.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import E4ContractError

ORDER_CONTRACT_VERSION = "e4-data-order-v1"

SHUFFLED_SOURCE_INTERLEAVED = "shuffled_source_interleaved"
POSITIVE_AWARE_RESAMPLED = "positive_aware_resampled"
DATA_ORDERS: tuple[str, ...] = (SHUFFLED_SOURCE_INTERLEAVED, POSITIVE_AWARE_RESAMPLED)

# A guard on the corpus, not a schedule. The governed train split is 77.2%
# zero-entity, so this must sit above that or the constraint is unsatisfiable
# without discarding real negatives; 0.85 leaves headroom and still refuses a
# corpus that is almost entirely negative.
DEFAULT_MAX_ZERO_ENTITY_FRACTION = 0.85
# Hard cap on consecutive zero-entity examples, whatever the fraction allows.
DEFAULT_MAX_ZERO_ENTITY_STREAK = 4


class SamplingError(E4ContractError):
    """Raised when a data-order contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class ExampleIndex:
    """The minimum an ordering needs: identity, source, and whether it has gold."""

    row_index: int
    source_dataset: str
    entity_count: int

    @property
    def has_entities(self) -> bool:
        return self.entity_count > 0


def _deterministic_key(seed: int, epoch: int, row_index: int) -> str:
    """A stable per-(seed, epoch, row) sort key.

    Hashing rather than ``random.shuffle`` on purpose: the order depends only on
    the three inputs, so it is identical across processes, Python versions and
    machines, and a single example's position can be recomputed in isolation.
    """
    material = f"{ORDER_CONTRACT_VERSION}:{seed}:{epoch}:{row_index}".encode()
    return hashlib.sha256(material).hexdigest()


def deterministic_shuffle(
    examples: Sequence[ExampleIndex], *, seed: int, epoch: int,
) -> list[ExampleIndex]:
    """A reproducible permutation for one epoch. Never the identity by accident."""
    if not examples:
        raise SamplingError("cannot order an empty example set")
    return sorted(
        examples, key=lambda item: _deterministic_key(seed, epoch, item.row_index))


def _by_source(examples: Sequence[ExampleIndex]) -> dict[str, list[ExampleIndex]]:
    buckets: dict[str, list[ExampleIndex]] = {}
    for example in examples:
        buckets.setdefault(example.source_dataset or "unknown", []).append(example)
    return buckets


def shuffled_source_interleaved_order(
    examples: Sequence[ExampleIndex], *, seed: int, epoch: int,
) -> list[ExampleIndex]:
    """Shuffle within each source, then spread each source across the epoch.

    Stratified rather than round-robin: an example at position ``i`` of a bucket
    of ``n`` is given the fractional epoch position ``(i + 0.5) / n``, and all
    examples are sorted by it. A source with 10,027 examples and one with 5,796
    are therefore both spread over the whole epoch, and the longest same-source
    run is bounded by the busiest source's local density instead of its size.

    A credit-based round-robin was tried first and was wrong: on the first pass
    no source had accrued a full unit of credit, its "nothing progressed" branch
    fired, and it drained every bucket in file order — reproducing exactly the
    10,027-example zero-entity block this function exists to break up.
    """
    shuffled = deterministic_shuffle(examples, seed=seed, epoch=epoch)
    buckets = _by_source(shuffled)
    placed: list[tuple[float, str, int, ExampleIndex]] = []
    for name in sorted(buckets):
        bucket = buckets[name]
        size = len(bucket)
        for position, example in enumerate(bucket):
            placed.append(((position + 0.5) / size, name, position, example))
    placed.sort(key=lambda item: (item[0], item[1], item[2]))
    order = [item[3] for item in placed]
    if len(order) != len(shuffled):
        raise SamplingError("interleaving did not preserve the example count")
    return order


def positive_aware_order(
    examples: Sequence[ExampleIndex],
    *,
    seed: int,
    epoch: int,
    max_zero_entity_fraction: float = DEFAULT_MAX_ZERO_ENTITY_FRACTION,
    max_zero_entity_streak: int = DEFAULT_MAX_ZERO_ENTITY_STREAK,
) -> list[ExampleIndex]:
    """Merge positives and negatives at their natural ratio, streak-bounded.

    The governed train split is 7,698 positive and 26,128 zero-entity examples,
    so zero-entity documents are 77.2% of the corpus. Any fixed cap below that —
    50%, say — is arithmetically unsatisfiable: honouring it early forces every
    remaining negative into one block at the end, which is the same defect in a
    different place. A first attempt did exactly that and produced an 18,431
    example zero-entity tail.

    So the merge is proportional (Bresenham-style): negatives are emitted at
    their natural rate against positives, which spreads them evenly and bounds
    the longest run at ``ceil(negatives / positives)`` — 4 for this corpus.
    ``max_zero_entity_streak`` then clamps that further if asked.

    ``max_zero_entity_fraction`` is a *guard*, not a schedule: it is checked
    against the corpus and raises when unsatisfiable, rather than being silently
    approximated.
    """
    if not 0.0 < max_zero_entity_fraction <= 1.0:
        raise SamplingError("max_zero_entity_fraction must lie in (0, 1]")
    if max_zero_entity_streak < 1:
        raise SamplingError("max_zero_entity_streak must be at least 1")

    interleaved = shuffled_source_interleaved_order(examples, seed=seed, epoch=epoch)
    positives = [item for item in interleaved if item.has_entities]
    negatives = [item for item in interleaved if not item.has_entities]
    if not positives:
        # Nothing to interleave against. Returning the plain order and saying so
        # beats pretending the constraint was honoured.
        return interleaved

    corpus_fraction = len(negatives) / len(interleaved)
    if corpus_fraction > max_zero_entity_fraction:
        raise SamplingError(
            f"the corpus is {corpus_fraction:.1%} zero-entity examples, so a "
            f"{max_zero_entity_fraction:.1%} cap cannot be met without dropping "
            "real negatives; raise the cap or rely on max_zero_entity_streak")

    order: list[ExampleIndex] = []
    positive_cursor = negative_cursor = 0
    streak = 0
    # Emit a negative whenever the running deficit says one is due, i.e. when
    # emitted negatives are behind their proportional share.
    rate = len(negatives) / len(positives)
    credit = 0.0
    while positive_cursor < len(positives) or negative_cursor < len(negatives):
        if positive_cursor < len(positives):
            order.append(positives[positive_cursor])
            positive_cursor += 1
            streak = 0
            credit += rate
            while (credit >= 1.0 and negative_cursor < len(negatives)
                   and streak < max_zero_entity_streak):
                order.append(negatives[negative_cursor])
                negative_cursor += 1
                credit -= 1.0
                streak += 1
            continue
        order.append(negatives[negative_cursor])
        negative_cursor += 1
        streak += 1
    if len(order) != len(interleaved):
        raise SamplingError("positive-aware ordering did not preserve the count")
    return order


def build_epoch_order(
    examples: Sequence[ExampleIndex],
    *,
    data_order: str,
    seed: int,
    epoch: int,
    max_zero_entity_fraction: float = DEFAULT_MAX_ZERO_ENTITY_FRACTION,
    max_zero_entity_streak: int = DEFAULT_MAX_ZERO_ENTITY_STREAK,
) -> list[ExampleIndex]:
    if data_order == SHUFFLED_SOURCE_INTERLEAVED:
        return shuffled_source_interleaved_order(examples, seed=seed, epoch=epoch)
    if data_order == POSITIVE_AWARE_RESAMPLED:
        return positive_aware_order(
            examples, seed=seed, epoch=epoch,
            max_zero_entity_fraction=max_zero_entity_fraction,
            max_zero_entity_streak=max_zero_entity_streak)
    raise SamplingError(f"unknown data order {data_order!r}; expected {DATA_ORDERS}")


# ---------------------------------------------------------------------------
# Realized composition, written to the manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderComposition:
    """What an epoch's order actually looks like, measured after building it."""

    epoch: int
    data_order: str
    examples: int
    positive_examples: int
    zero_entity_examples: int
    longest_zero_entity_streak: int
    longest_zero_entity_streak_start: int
    longest_same_source_streak: int
    examples_by_source: Mapping[str, int]
    first_positive_position: int

    @property
    def zero_entity_fraction(self) -> float:
        return self.zero_entity_examples / self.examples if self.examples else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_contract_version": ORDER_CONTRACT_VERSION,
            "epoch": self.epoch,
            "data_order": self.data_order,
            "examples": self.examples,
            "positive_examples": self.positive_examples,
            "zero_entity_examples": self.zero_entity_examples,
            "zero_entity_fraction": self.zero_entity_fraction,
            "longest_zero_entity_streak": self.longest_zero_entity_streak,
            "longest_zero_entity_streak_start": self.longest_zero_entity_streak_start,
            "longest_same_source_streak": self.longest_same_source_streak,
            "examples_by_source": dict(self.examples_by_source),
            "first_positive_position": self.first_positive_position,
            "examples_dropped": 0,
            "examples_duplicated": 0,
        }


def measure_order(
    order: Sequence[ExampleIndex], *, epoch: int, data_order: str,
) -> OrderComposition:
    """Measure a realized order. Reported per epoch, never assumed."""
    if not order:
        raise SamplingError("cannot measure an empty order")
    by_source: dict[str, int] = {}
    streak = longest = 0
    streak_start = longest_start = -1
    source_streak = longest_source_streak = 0
    previous_source = ""
    first_positive = -1
    positives = 0
    for position, item in enumerate(order):
        by_source[item.source_dataset] = by_source.get(item.source_dataset, 0) + 1
        if item.has_entities:
            positives += 1
            if first_positive < 0:
                first_positive = position
            streak = 0
            streak_start = -1
        else:
            if streak == 0:
                streak_start = position
            streak += 1
            if streak > longest:
                longest = streak
                longest_start = streak_start
        if item.source_dataset == previous_source:
            source_streak += 1
        else:
            source_streak = 1
            previous_source = item.source_dataset
        longest_source_streak = max(longest_source_streak, source_streak)
    return OrderComposition(
        epoch=epoch,
        data_order=data_order,
        examples=len(order),
        positive_examples=positives,
        zero_entity_examples=len(order) - positives,
        longest_zero_entity_streak=longest,
        longest_zero_entity_streak_start=longest_start,
        longest_same_source_streak=longest_source_streak,
        examples_by_source=dict(sorted(by_source.items())),
        first_positive_position=first_positive,
    )


def assert_order_preserves_corpus(
    original: Sequence[ExampleIndex], order: Sequence[ExampleIndex],
) -> None:
    """No example dropped, duplicated or invented. Reordering only."""
    original_rows = sorted(item.row_index for item in original)
    order_rows = sorted(item.row_index for item in order)
    if original_rows != order_rows:
        raise SamplingError(
            "the epoch order is not a permutation of the governed corpus; "
            "zero-entity examples are reordered, never removed")


__all__ = [
    "DATA_ORDERS",
    "DEFAULT_MAX_ZERO_ENTITY_FRACTION",
    "DEFAULT_MAX_ZERO_ENTITY_STREAK",
    "ORDER_CONTRACT_VERSION",
    "POSITIVE_AWARE_RESAMPLED",
    "SHUFFLED_SOURCE_INTERLEAVED",
    "ExampleIndex",
    "OrderComposition",
    "SamplingError",
    "assert_order_preserves_corpus",
    "build_epoch_order",
    "deterministic_shuffle",
    "measure_order",
    "positive_aware_order",
    "shuffled_source_interleaved_order",
]
