"""Reversible character alignment between original and normalized views.

``CharAlignment`` stores a single monotonic **boundary map** ``o2n`` (length
N+1): ``o2n[i]`` is the normalized boundary for original boundary ``i``. Covering
original spans are derived from it by binary search, so the structure supports
one-to-one, one-to-many, and many-to-one transformations, characters introduced
only in a normalized view, and original characters omitted from it — while
guaranteeing the **smallest valid covering** original span for any normalized
span.

Organizer output must always use ORIGINAL coordinates; a normalized span only
ever yields a covering original span for retrieval/matching.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..schemas.spans import OffsetAlignment


@dataclass(frozen=True, slots=True)
class CharAlignment:
    """Boundary map ``o2n`` between an original and a normalized string.

    ``o2n[i]`` (``0 <= i <= original_length``) is the normalized boundary for
    original boundary ``i``; it is monotonic non-decreasing with ``o2n[0] == 0``
    and ``o2n[original_length] == normalized_length``.
    """

    original_length: int
    normalized_length: int
    o2n: tuple[int, ...]

    # --- constructors ---------------------------------------------------------

    @staticmethod
    def identity(length: int) -> CharAlignment:
        return CharAlignment(length, length, tuple(range(length + 1)))

    @staticmethod
    def from_counts(out_counts: list[int]) -> CharAlignment:
        """Build an alignment from per-input-char output counts (O(n), exact).

        ``out_counts[i]`` is the number of normalized chars produced by original
        char ``i`` (0 = deleted, 1 = kept/substituted, >1 = expanded).
        """
        ilen = len(out_counts)
        o2n = [0] * (ilen + 1)
        for i, c in enumerate(out_counts):
            o2n[i + 1] = o2n[i] + c
        return CharAlignment(ilen, o2n[ilen], tuple(o2n))

    @staticmethod
    def from_transform(src: str, dst: str) -> CharAlignment:
        """Build the alignment implied by transforming ``src`` into ``dst`` (via diff).

        Prefer :meth:`from_counts` in hot paths; this diff-based builder is for
        tests/edge cases and is O(n·m) on dissimilar strings.
        """
        o2n = [0] * (len(src) + 1)
        sm = SequenceMatcher(a=src, b=dst, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1 + 1):
                    o2n[i1 + k] = j1 + k
            else:  # replace | delete | insert
                for i in range(i1, i2 + 1):
                    o2n[i] = j1 if i < i2 else j2
        o2n[len(src)] = len(dst)
        return CharAlignment(len(src), len(dst), tuple(o2n))

    # --- span mapping ---------------------------------------------------------

    def original_span_to_normalized(self, start: int, end: int) -> tuple[int, int]:
        self._check_original(start, end)
        return (self.o2n[start], self.o2n[end])

    def normalized_span_to_original(self, start: int, end: int) -> tuple[int, int]:
        """Smallest covering original span ``[os, oe)`` for a normalized span.

        ``os`` = largest original boundary mapping at/before ``start``;
        ``oe`` = smallest original boundary mapping at/after ``end``.
        """
        self._check_normalized(start, end)
        if start == end:
            # Empty normalized span → a zero-width original point mapping to it.
            b = bisect.bisect_left(self.o2n, start)
            return (b, b)
        os = bisect.bisect_right(self.o2n, start) - 1
        oe = bisect.bisect_left(self.o2n, end)
        return (os, oe)

    def recover_original(self, original_text: str, norm_start: int, norm_end: int) -> str:
        os, oe = self.normalized_span_to_original(norm_start, norm_end)
        return original_text[os:oe]

    @property
    def is_identity(self) -> bool:
        return (
            self.original_length == self.normalized_length
            and self.o2n == tuple(range(self.original_length + 1))
        )

    # --- composition ----------------------------------------------------------

    def then(self, other: CharAlignment) -> CharAlignment:
        """Compose ``self`` (orig→mid) with ``other`` (mid→final) → (orig→final)."""
        if self.normalized_length != other.original_length:
            raise ValueError(
                f"cannot compose: mid length mismatch "
                f"{self.normalized_length} != {other.original_length}"
            )
        o2n = tuple(other.o2n[m] for m in self.o2n)
        return CharAlignment(self.original_length, other.normalized_length, o2n)

    # --- validation -----------------------------------------------------------

    def consistency_errors(self) -> list[str]:
        errors: list[str] = []
        if len(self.o2n) != self.original_length + 1:
            errors.append("o2n length != original_length+1")
            return errors
        if self.o2n[0] != 0 or self.o2n[-1] != self.normalized_length:
            errors.append("o2n endpoints not anchored to [0, normalized_length]")
        if any(b < a for a, b in zip(self.o2n, self.o2n[1:], strict=False)):
            errors.append("o2n is not monotonic non-decreasing")
        return errors

    def to_offset_alignment(self) -> OffsetAlignment:
        """Bridge to the schema-level :class:`OffsetAlignment` (char-indexed)."""
        if self.is_identity:
            return OffsetAlignment(self.original_length, self.normalized_length)
        n2o_char = tuple(
            bisect.bisect_right(self.o2n, j) - 1 for j in range(self.normalized_length)
        )
        return OffsetAlignment(
            original_length=self.original_length,
            normalized_length=self.normalized_length,
            original_to_normalized=self.o2n[: self.original_length],
            normalized_to_original=n2o_char,
        )

    def _check_original(self, start: int, end: int) -> None:
        if not (0 <= start <= end <= self.original_length):
            raise ValueError(f"original span [{start}, {end}) out of bounds")

    def _check_normalized(self, start: int, end: int) -> None:
        if not (0 <= start <= end <= self.normalized_length):
            raise ValueError(f"normalized span [{start}, {end}) out of bounds")


__all__ = ["CharAlignment"]
