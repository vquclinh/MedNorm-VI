"""Configurable, reversible normalization pipeline (O(n), exact alignment).

Each stage is a pure transform on a COPY that also reports, for every input
character, how many output characters it produced. That exact per-character
count yields an O(n) reversible alignment (no diffing), and stage alignments
compose into one original→view alignment. No uncontrolled semantic rewriting;
the clinical abbreviation expander is intentionally NOT built here.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Callable

from .alignment import CharAlignment
from .models import NormalizationStageRecord, NormalizedDocument

# stage: input string -> (output string, per-input-char output counts)
StageFn = Callable[[str], tuple[str, list[int]]]

_WS = set(" \t\n\r\f\v ")

_PUNCT_ALIASES = {
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "–": "-", "—": "-", "―": "-", "‑": "-",
    "•": "*", "·": "*",
    " ": " ",
}


def _cluster_transform(text: str, fn: Callable[[str], str]) -> tuple[str, list[int]]:
    """Apply ``fn`` per normalization cluster (starter + following combining marks).

    The whole cluster's output count is assigned to its first char (rest 0), so
    a composed/decomposed form maps back to a covering original span.
    """
    out: list[str] = []
    counts = [0] * len(text)
    i = 0
    n = len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]) != 0:
            j += 1
        produced = fn(text[i:j])
        out.append(produced)
        counts[i] = len(produced)
        i = j
    return "".join(out), counts


def _nfc(text: str) -> tuple[str, list[int]]:
    return _cluster_transform(text, lambda c: unicodedata.normalize("NFC", c))


def _accent_strip(text: str) -> tuple[str, list[int]]:
    def strip(cluster: str) -> str:
        d = unicodedata.normalize("NFD", cluster)
        return "".join(ch for ch in d if not unicodedata.combining(ch))
    return _cluster_transform(text, strip)


def _casefold(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    counts = [0] * len(text)
    for i, ch in enumerate(text):
        folded = ch.casefold()
        out.append(folded)
        counts[i] = len(folded)
    return "".join(out), counts


def _whitespace_collapse(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    counts = [0] * len(text)
    prev_ws = False
    for i, ch in enumerate(text):
        if ch in _WS:
            if prev_ws:
                counts[i] = 0
            else:
                out.append(" ")
                counts[i] = 1
            prev_ws = True
        else:
            out.append(ch)
            counts[i] = 1
            prev_ws = False
    return "".join(out), counts


def _punctuation_alias(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    counts = [0] * len(text)
    for i, ch in enumerate(text):
        rep = _PUNCT_ALIASES.get(ch, ch)
        out.append(rep)
        counts[i] = len(rep)
    return "".join(out), counts


def _decimal_view(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    counts = [1] * len(text)
    n = len(text)
    for i, ch in enumerate(text):
        if ch == "," and 0 < i < n - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            out.append(".")
        else:
            out.append(ch)
    return "".join(out), counts


STAGES: dict[str, tuple[StageFn, str]] = {
    "nfc": (_nfc, "1"),
    "casefold": (_casefold, "1"),
    "whitespace_collapse": (_whitespace_collapse, "1"),
    "punctuation_alias": (_punctuation_alias, "1"),
    "decimal_view": (_decimal_view, "1"),
    "accent_strip": (_accent_strip, "1"),
}


def _stage_hash(name: str, version: str) -> str:
    return hashlib.sha256(f"{name}@{version}".encode()).hexdigest()[:16]


def _change_count(src: str, out_text: str, alignment: CharAlignment) -> int:
    changed = 0
    for i, ch in enumerate(src):
        piece = out_text[alignment.o2n[i] : alignment.o2n[i + 1]]
        if piece != ch:
            changed += 1
    return changed


def build_view(name: str, stage_names: tuple[str, ...], original_text: str) -> NormalizedDocument:
    """Apply an ordered list of stages to ``original_text`` → a normalized view."""
    src = original_text
    alignment = CharAlignment.identity(len(original_text))
    records: list[NormalizationStageRecord] = []
    for stage_name in stage_names:
        if stage_name not in STAGES:
            raise KeyError(f"unknown normalization stage {stage_name!r}")
        transform, version = STAGES[stage_name]
        dst, counts = transform(src)
        stage_alignment = CharAlignment.from_counts(counts)
        alignment = alignment.then(stage_alignment)
        records.append(NormalizationStageRecord(
            name=stage_name,
            version=version,
            input_length=len(src),
            output_length=len(dst),
            transformation_count=_change_count(src, dst, stage_alignment),
            config_hash=_stage_hash(stage_name, version),
        ))
        src = dst
    return NormalizedDocument(name=name, text=src, alignment=alignment, stages=tuple(records))


def build_views(
    original_text: str, views: dict[str, tuple[str, ...]]
) -> dict[str, NormalizedDocument]:
    """Build every configured named view (deterministic order by view name)."""
    return {name: build_view(name, views[name], original_text) for name in sorted(views)}


__all__ = ["STAGES", "build_view", "build_views"]
