"""Deterministic, high-recall section-header detection.

A section is EVIDENCE, not an absolute rule: L1 records the header span, proposed
category, confidence, matched rule, and prior — it never assigns final
assertions. Matching runs on a casefold + accent-stripped + whitespace-collapsed
view; a fuzzy hit additionally requires structural evidence (colon-terminated,
short/isolated, or upper-case heading) so ordinary sentences are not promoted to
headers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .lines import LinePiece, is_blank
from .models import L1Config
from .unicode_utils import strip_accents

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", strip_accents(unicodedata.normalize("NFC", text).casefold())).strip()


@dataclass(frozen=True, slots=True)
class SectionAlias:
    surface: str
    normalized: str
    language: str
    abbreviation: bool


@dataclass(frozen=True, slots=True)
class SectionCategory:
    category: str
    semantic_group: str
    prior_label: str | None
    prior_strength: float
    aliases: tuple[SectionAlias, ...]
    positive_examples: tuple[str, ...] = ()
    negative_examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SectionLexicon:
    version: int
    lexicon_version: int
    categories: tuple[SectionCategory, ...]

    def all_aliases(self) -> list[tuple[SectionAlias, SectionCategory]]:
        return [(a, c) for c in self.categories for a in c.aliases]


@dataclass(frozen=True, slots=True)
class SectionHit:
    line_index: int
    indent: int
    category: str
    confidence: float
    header_start: int
    header_end: int
    matched_rule: str
    prior_label: str | None = None
    prior_strength: float = 0.0
    warnings: tuple[str, ...] = field(default_factory=tuple)


def load_lexicon(path: str | Path) -> SectionLexicon:
    import yaml

    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    categories: list[SectionCategory] = []
    for cat in data.get("categories", []) or []:
        aliases = tuple(
            SectionAlias(
                surface=str(a["form"]),
                normalized=_norm(str(a["form"])),
                language=str(a.get("language", "vi")),
                abbreviation=bool(a.get("abbreviation", False)),
            )
            for a in cat.get("aliases", []) or []
        )
        categories.append(SectionCategory(
            category=str(cat["category"]),
            semantic_group=str(cat.get("semantic_group", "")),
            prior_label=(cat.get("prior_label") or None),
            prior_strength=float(cat.get("prior_strength", 0.0)),
            aliases=aliases,
            positive_examples=tuple(cat.get("positive_examples", []) or []),
            negative_examples=tuple(cat.get("negative_examples", []) or []),
        ))
    return SectionLexicon(
        version=int(data.get("version", 1)),
        lexicon_version=int(data.get("lexicon_version", 1)),
        categories=tuple(categories),
    )


def _leading_ws(text: str) -> int:
    i = 0
    while i < len(text) and text[i] in " \t":
        i += 1
    return i


def _header_key_region(content: str) -> tuple[int, int, bool]:
    """Return (key_start_in_content, key_end_in_content, colon_terminated).

    The key is the text before the first ``:`` (if any), stripped of surrounding
    whitespace; ``colon_terminated`` marks the strong header signal.
    """
    colon = content.find(":")
    if colon != -1:
        raw_end = colon
        colon_terminated = True
    else:
        raw_end = len(content)
        colon_terminated = False
    start = _leading_ws(content)
    end = raw_end
    while end > start and content[end - 1] in " \t":
        end -= 1
    return start, end, colon_terminated


def detect_sections(
    text: str, pieces: list[LinePiece], config: L1Config, lexicon: SectionLexicon
) -> list[SectionHit]:
    """Detect section-header lines deterministically (in document order)."""
    hits: list[SectionHit] = []
    aliases = lexicon.all_aliases()
    cat_by_name = {c.category: c for c in lexicon.categories}

    for i, piece in enumerate(pieces):
        if is_blank(text, piece):
            continue
        content = text[piece.content_start : piece.content_end]
        key_start, key_end, colon = _header_key_region(content)
        if key_end <= key_start:
            continue
        key = content[key_start:key_end]
        if len(key) > config.max_header_chars:
            continue
        # A key that ends with sentence punctuation is a sentence, not a header.
        if key.rstrip() and key.rstrip()[-1] in ".!?…":
            continue
        norm_key = _norm(key)
        if not norm_key:
            continue

        upper = key.strip() == key.strip().upper() and any(ch.isalpha() for ch in key)
        # Structural evidence for a FUZZY header: a colon-terminated heading or an
        # upper-case heading. A merely short/isolated line is NOT sufficient — that
        # would promote ordinary standalone sentences to headers.
        structural = colon or upper

        best_rule = ""
        best_conf = 0.0
        best_alias: SectionAlias | None = None
        best_category: SectionCategory | None = None
        for alias, category in aliases:
            if norm_key == alias.normalized:
                best_rule, best_conf, best_alias, best_category = (
                    "exact_alias", 0.99, alias, category)
                break
            ratio = SequenceMatcher(a=norm_key, b=alias.normalized, autojunk=False).ratio()
            if ratio > best_conf:
                best_conf, best_alias, best_category = ratio, alias, category
                best_rule = "fuzzy_alias"

        if best_alias is None or best_category is None:
            continue
        if best_rule == "exact_alias":
            accept = True
        else:
            accept = (
                best_conf >= config.fuzzy_threshold
                and (structural or not config.require_structural_evidence)
            )
        if not accept:
            continue

        abs_start = piece.content_start + key_start
        abs_end = piece.content_start + key_end
        cat = cat_by_name[best_category.category]
        warnings: tuple[str, ...] = ()
        if best_rule == "fuzzy_alias" and best_conf < 0.95:
            warnings = ("weak_section_header_confidence",)
        hits.append(SectionHit(
            line_index=i,
            indent=key_start,
            category=cat.category,
            confidence=best_conf,
            header_start=abs_start,
            header_end=abs_end,
            matched_rule=f"{best_rule}:{best_alias.surface}",
            prior_label=cat.prior_label,
            prior_strength=cat.prior_strength,
            warnings=warnings,
        ))
    return hits


__all__ = [
    "SectionAlias",
    "SectionCategory",
    "SectionLexicon",
    "SectionHit",
    "load_lexicon",
    "detect_sections",
]
