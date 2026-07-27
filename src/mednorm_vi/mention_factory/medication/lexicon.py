"""Medication lexicon + compiled grammar patterns (versioned, extensible).

Loads the medication grammar config plus a SMALL seed ingredient lexicon. No
external drug database is downloaded. A local, git-ignored production lexicon
path may be populated later; if present it is merged in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _alt(forms: list[str]) -> re.Pattern[str]:
    """Case-insensitive alternation with word boundaries, longest-first."""
    parts = sorted({f for f in forms if f}, key=len, reverse=True)
    if not parts:
        return re.compile(r"(?!x)x")  # never matches
    body = "|".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![\w]){body}(?![\w])", re.IGNORECASE | re.UNICODE)


@dataclass(frozen=True, slots=True)
class MedicationLexicon:
    grammar_version: str
    lexicon_version: str
    ingredients: frozenset[str]
    ingredient_re: re.Pattern[str]
    salt_re: re.Pattern[str]
    release_re: re.Pattern[str]
    dose_form_re: re.Pattern[str]
    route_re: re.Pattern[str]
    frequency_re: re.Pattern[str]
    prn_re: re.Pattern[str]
    strength_re: re.Pattern[str]
    concentration_re: re.Pattern[str]
    percent_re: re.Pattern[str]
    duration_res: tuple[re.Pattern[str], ...]
    hard_negative_res: tuple[re.Pattern[str], ...]
    scoring: dict[str, float]
    boundary_candidates: tuple[str, ...]
    unknown_requires_structure: bool
    config_hash: str
    features: dict[str, str] = field(default_factory=dict)


def _forms(entries: list[dict[str, Any]]) -> list[str]:
    return [str(e["form"]) for e in entries]


def load_medication_lexicon(config_path: str | Path) -> MedicationLexicon:
    import yaml

    path = Path(config_path)
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seed_path = path.parent / str(cfg.get("lexicon_seed_file", "lexicon_seed_v1.yaml"))
    seed: dict[str, Any] = yaml.safe_load(seed_path.read_text(encoding="utf-8")) or {}

    ingredients = [str(x).lower() for x in seed.get("ingredients", []) or []]
    ingredients += [str(x).lower() for x in seed.get("brand_hints", []) or []]
    # Optional local, git-ignored production lexicon (one name per line).
    prod = seed.get("production_lexicon_path")
    if prod:
        prod_path = Path(prod)
        if prod_path.is_file():
            ingredients += [
                ln.strip().lower()
                for ln in prod_path.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]

    strength = cfg.get("strength", {}) or {}
    number = str(strength.get("number", r"\d+(?:[.,]\d+)?"))
    units = [str(u) for u in strength.get("units", [])]
    unit_alt = "|".join(re.escape(u) for u in sorted(set(units), key=len, reverse=True))
    range_sep = str(strength.get("range_sep", r"[-–—]"))
    # STRENGTH: number [range number] optional-space unit
    strength_re = re.compile(
        rf"{number}(?:\s?{range_sep}\s?{number})?\s?(?:{unit_alt})(?![\w])",
        re.IGNORECASE | re.UNICODE,
    )
    concentration_re = re.compile(str(strength.get("concentration", r"(?!x)x")),
                                  re.IGNORECASE | re.UNICODE)
    percent_re = re.compile(str(strength.get("percent", r"\d+(?:[.,]\d+)?\s?%")))

    scoring = {str(k): float(v) for k, v in (cfg.get("scoring", {}) or {}).items()}
    return MedicationLexicon(
        grammar_version=str(cfg.get("grammar_version", "med-g1")),
        lexicon_version=str(seed.get("lexicon_version", "med-lex-seed-1")),
        ingredients=frozenset(ingredients),
        ingredient_re=_alt(ingredients),
        salt_re=_alt([str(x) for x in seed.get("salts", []) or []]),
        release_re=_alt(_forms(cfg.get("release_forms", []) or [])),
        dose_form_re=_alt(_forms(cfg.get("dose_forms", []) or [])),
        route_re=_alt(_forms(cfg.get("routes", []) or [])),
        frequency_re=_alt(_forms(cfg.get("frequencies", []) or [])),
        prn_re=_alt(_forms(cfg.get("prn_forms", []) or [])),
        strength_re=strength_re,
        concentration_re=concentration_re,
        percent_re=percent_re,
        duration_res=tuple(re.compile(str(p), re.IGNORECASE | re.UNICODE)
                           for p in cfg.get("duration", []) or []),
        hard_negative_res=tuple(re.compile(str(p), re.IGNORECASE | re.UNICODE)
                                for p in cfg.get("hard_negatives", []) or []),
        scoring=scoring,
        boundary_candidates=tuple(cfg.get("boundary_candidates", []) or []),
        unknown_requires_structure=bool(
            (cfg.get("min_context_evidence", {}) or {}).get(
                "unknown_name_requires_structure", True)),
        config_hash="",
    )


__all__ = ["MedicationLexicon", "load_medication_lexicon"]
