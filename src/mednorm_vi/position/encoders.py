"""Deterministic position encoders: raw code-point span -> submission coordinate.

Each encoder builds a monotonic *forward map* ``fmap`` of length ``len(text)+1``
where ``fmap[i]`` is the encoded offset at code-point boundary ``i``. Encoding a
raw span ``[start, end)`` is ``(fmap[start], fmap[end])``. Decoding reverses the
map; a target that does not land on a code-point boundary (e.g. a byte offset
inside a multi-byte character) fails clearly rather than silently rounding.

Encoders never edit entity text — they only re-express coordinates.
"""

from __future__ import annotations

from ..schemas.spans import OffsetAlignment
from .models import PositionEncodingResult, PositionPolicy, PositionProvenance


class PositionEncodingError(ValueError):
    """Raised when a raw span cannot be reversibly encoded/decoded under a policy."""


def _codepoint_map(text: str) -> list[int]:
    return list(range(len(text) + 1))


def _utf8_byte_map(text: str) -> list[int]:
    fmap = [0] * (len(text) + 1)
    total = 0
    for i, ch in enumerate(text):
        total += len(ch.encode("utf-8"))
        fmap[i + 1] = total
    return fmap


def _line_ending_map(text: str, *, mode: str, separator: str) -> list[int]:
    """Forward map for a line-ending reconstruction over code points.

    mode: 'lf' (CRLF/CR -> LF), 'crlf' (lone LF -> CRLF), 'sep' (any run -> separator).
    """
    fmap = [0] * (len(text) + 1)
    out = 0
    i = 0
    n = len(text)
    while i < n:
        fmap[i] = out
        ch = text[i]
        if ch == "\r" and i + 1 < n and text[i + 1] == "\n":
            # CRLF pair
            if mode == "lf":
                fmap[i + 1] = out  # '\r' dropped, no advance
                out += 1  # the '\n'
            elif mode == "crlf":
                out += 1
                fmap[i + 1] = out
                out += 1
            else:  # sep
                fmap[i + 1] = out
                out += len(separator)
            i += 2
            fmap[i] = out
            continue
        if ch in ("\r", "\n"):
            if mode == "lf":
                out += 1
            elif mode == "crlf":
                out += 2  # lone LF/CR -> CRLF
            else:
                out += len(separator)
        else:
            out += 1
        i += 1
        fmap[i] = out
    fmap[n] = out
    return fmap


def _normalized_map(text: str, alignment: OffsetAlignment | None) -> list[int]:
    if alignment is None:
        raise PositionEncodingError(
            "normalized-view policy requires a reversible OffsetAlignment; none provided")
    if alignment.original_length != len(text):
        raise PositionEncodingError(
            "OffsetAlignment original_length does not match text length")
    return [alignment.to_normalized(i) for i in range(len(text) + 1)]


def build_forward_map(
    policy: PositionPolicy, text: str, *, separator: str = "\n",
    alignment: OffsetAlignment | None = None,
) -> list[int]:
    """Build the monotonic forward map for ``policy`` over ``text``."""
    space, le = policy.coordinate_space, policy.line_ending
    if space == "normalized_codepoint":
        return _normalized_map(text, alignment)
    if space == "utf8_byte":
        if le not in ("raw",):
            raise PositionEncodingError(
                f"utf8_byte with line_ending {le!r} is not supported in Phase 1C-A")
        return _utf8_byte_map(text)
    # code-point space
    if le == "raw":
        return _codepoint_map(text)
    if le == "lf_canonical":
        return _line_ending_map(text, mode="lf", separator="\n")
    if le == "crlf_canonical":
        return _line_ending_map(text, mode="crlf", separator="\r\n")
    if le == "configurable":
        return _line_ending_map(text, mode="sep", separator=separator)
    raise PositionEncodingError(f"unsupported policy {policy.policy_id!r}")


def encode_span(
    policy: PositionPolicy, text: str, start: int, end: int, *,
    separator: str = "\n", alignment: OffsetAlignment | None = None,
) -> PositionEncodingResult:
    """Encode raw code-point span ``[start, end)`` under ``policy`` with provenance."""
    if not (0 <= start <= end <= len(text)):
        raise PositionEncodingError(f"raw span [{start},{end}) out of bounds for len {len(text)}")
    fmap = build_forward_map(policy, text, separator=separator, alignment=alignment)
    es, ee = fmap[start], fmap[end]
    if policy.interval == "closed":
        ee = ee - 1 if ee > es else es  # end-inclusive representation
    reversible = policy.reversible != "false"
    prov = PositionProvenance(
        policy_id=policy.policy_id, coordinate_space=policy.coordinate_space,
        raw_start=start, raw_end=end, reversible=reversible,
        note=f"line_ending={policy.line_ending}")
    return PositionEncodingResult(
        policy_id=policy.policy_id, coordinate_space=policy.coordinate_space,
        interval=policy.interval, start=es, end=ee, text=text[start:end], provenance=prov)


def decode_position(
    policy: PositionPolicy, text: str, enc_start: int, enc_end: int, *,
    separator: str = "\n", alignment: OffsetAlignment | None = None,
) -> tuple[int, int]:
    """Reverse an encoded position back to a raw code-point span, or fail clearly."""
    fmap = build_forward_map(policy, text, separator=separator, alignment=alignment)
    target_end = enc_end
    if policy.interval == "closed":
        target_end = enc_end + 1
    try:
        start = fmap.index(enc_start)
    except ValueError as exc:
        raise PositionEncodingError(
            f"encoded start {enc_start} does not land on a code-point boundary") from exc
    try:
        end = fmap.index(target_end, start)
    except ValueError as exc:
        raise PositionEncodingError(
            f"encoded end {enc_end} does not land on a code-point boundary") from exc
    return start, end


__all__ = [
    "PositionEncodingError",
    "build_forward_map",
    "encode_span",
    "decode_position",
]
