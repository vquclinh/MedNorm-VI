"""Laboratory lexicon + compiled patterns (versioned, extensible).

Loads the lab parser config, a SMALL seed test lexicon, and a unit inventory. No
lab ontology is downloaded; a local, git-ignored production lexicon may be merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _alt(forms: list[str]) -> re.Pattern[str]:
    parts = sorted({f for f in forms if f}, key=len, reverse=True)
    if not parts:
        return re.compile(r"(?!x)x")
    body = "|".join(re.escape(p) for p in parts)
    # allow a trailing % as part of the token (NEUT%), no leading word char
    return re.compile(rf"(?<![\w]){body}", re.IGNORECASE | re.UNICODE)


@dataclass(frozen=True, slots=True)
class LabLexicon:
    parser_version: str
    lexicon_version: str
    test_names: frozenset[str]  # normalized (lowercased) match forms
    test_re: re.Pattern[str]
    alias_to_canonical: dict[str, str]
    unit_re: re.Pattern[str]
    number_re: re.Pattern[str]
    inequality_re: re.Pattern[str]
    range_re: re.Pattern[str]
    percent_re: re.Pattern[str]
    qualitative_re: re.Pattern[str]
    reference_res: tuple[re.Pattern[str], ...]
    flag_forms: tuple[str, ...]
    flag_re: re.Pattern[str]
    hard_negative_res: tuple[re.Pattern[str], ...]
    copulas: tuple[str, ...]
    pairing_weights: dict[str, float]
    pairing_max_cost: float
    forbid_cross_section: bool
    keep_close_within: float
    scoring: dict[str, float]
    thresholds: dict[str, float]
    config_hash: str = ""
    features: dict[str, str] = field(default_factory=dict)


def load_lab_lexicon(config_path: str | Path) -> LabLexicon:
    import yaml  # type: ignore[import-untyped]

    path = Path(config_path)
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seed_path = path.parent / str(cfg.get("test_lexicon_seed_file", "test_lexicon_seed_v1.yaml"))
    units_path = path.parent / str(cfg.get("units_file", "units_v1.yaml"))
    seed: dict[str, Any] = yaml.safe_load(seed_path.read_text(encoding="utf-8")) or {}
    units_data: dict[str, Any] = yaml.safe_load(units_path.read_text(encoding="utf-8")) or {}

    names: list[str] = []
    alias_to_canonical: dict[str, str] = {}
    for entry in seed.get("tests", []) or []:
        canonical = str(entry["name"])
        forms = [canonical, *[str(a) for a in entry.get("aliases", []) or []]]
        for f in forms:
            names.append(f)
            alias_to_canonical[f.lower()] = canonical
    prod = seed.get("production_lexicon_path")
    if prod and Path(prod).is_file():
        for ln in Path(prod).read_text(encoding="utf-8").splitlines():
            t = ln.strip()
            if t and not t.startswith("#"):
                names.append(t)
                alias_to_canonical[t.lower()] = t

    units = [str(u) for u in units_data.get("units", [])]
    unit_alt = "|".join(re.escape(u) for u in sorted(set(units), key=len, reverse=True))

    result = cfg.get("result", {}) or {}
    qualitative = [str(q["term"]) for q in cfg.get("qualitative_terms", []) or []]
    flags = [str(f["form"]) for f in cfg.get("flags", []) or []]
    pairing = cfg.get("pairing", {}) or {}
    pairing_weights = {str(k): float(v) for k, v in (pairing.get("weights", {}) or {}).items()}

    return LabLexicon(
        parser_version=str(cfg.get("parser_version", "lab-p1")),
        lexicon_version=str(seed.get("lexicon_version", "lab-lex-seed-1")),
        test_names=frozenset(n.lower() for n in names),
        test_re=_alt(names),
        alias_to_canonical=alias_to_canonical,
        unit_re=re.compile(rf"(?<![\w])(?:{unit_alt})", re.IGNORECASE | re.UNICODE),
        number_re=re.compile(str(result.get("number", r"\d+(?:[.,]\d+)?"))),
        inequality_re=re.compile(str(result.get("inequality", r"(?:<=|>=|<|>|≤|≥)\s?\d+"))),
        range_re=re.compile(str(result.get("range", r"\d+\s?[-–]\s?\d+"))),
        percent_re=re.compile(str(result.get("percent", r"\d+(?:[.,]\d+)?\s?%"))),
        qualitative_re=_alt(qualitative),
        reference_res=tuple(re.compile(str(p), re.IGNORECASE | re.UNICODE)
                            for p in cfg.get("reference_range", []) or []),
        flag_forms=tuple(flags),
        flag_re=_alt(flags),
        hard_negative_res=tuple(re.compile(str(p), re.IGNORECASE | re.UNICODE)
                                for p in cfg.get("hard_negatives", []) or []),
        copulas=tuple((cfg.get("narrative", {}) or {}).get("copulas", []) or []),
        pairing_weights=pairing_weights,
        pairing_max_cost=float(pairing.get("max_cost", 0.9)),
        forbid_cross_section=bool(pairing.get("forbid_cross_section", True)),
        keep_close_within=float(pairing.get("keep_close_alternatives_within", 0.05)),
        scoring={str(k): float(v) for k, v in (cfg.get("scoring", {}) or {}).items()},
        thresholds={str(k): float(v)
                    for k, v in (cfg.get("confidence_thresholds", {}) or {}).items()},
    )


__all__ = ["LabLexicon", "load_lab_lexicon"]
