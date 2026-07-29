"""Router configuration, cue compilation, and deterministic signal detectors.

Cue groups are regexes matched case-insensitively against a line's original text
(accented and unaccented Vietnamese variants are listed in config). Structural
detectors read the L1 ``DocumentGraph`` context. Nothing here is a probability.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_UPPER_SHORT = re.compile(r"\b[A-ZĐÀ-Ỹ]{2,6}\b")
_UNACCENTED_CUE = re.compile(
    r"\b(khong|benh|tien su|chan doan|thuoc|xet nghiem|tang huyet ap|dtd|tha)\b", re.IGNORECASE
)
_MIXED_EN = re.compile(r"\b(po|iv|im|sc|bid|tid|qid|mg|ml|tablet|wbc|rbc|hgb|plt)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SignalSpec:
    name: str
    kind: str
    weight: float
    detector: str


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case: str
    name: str
    activated_specialists: tuple[str, ...]
    signals: tuple[SignalSpec, ...]
    # Signal names that constitute POSITIVE evidence for this case (Audit 0053).
    #
    # A weighted sum over soft structural cues is necessary but not sufficient.
    # Audit 0052 measured the consequence: Vietnamese narrative prose containing a
    # colon and any numeral scored C2 at 0.85 (key_value_row 0.30 + table_row 0.25
    # + numeric_present 0.30) with **no laboratory evidence at all**, so E2 —
    # correctly gated on C2 — fired on prose and produced only false positives.
    #
    # When this tuple is non-empty, the case activates only if at least one of
    # these signals fired. Empty means "score alone decides", which is the correct
    # behaviour for a case whose signals are already specific.
    required_evidence: tuple[str, ...] = ()

    def evidence_satisfied(self, fired_signal_names: frozenset[str]) -> bool:
        """Whether this case has the positive evidence it requires."""
        if not self.required_evidence:
            return True
        return bool(fired_signal_names & set(self.required_evidence))


@dataclass(frozen=True, slots=True)
class RouterConfig:
    router_version: str
    signals_version: str
    lexicon_version: int
    activate: float
    strong: float
    narrative_fallback: bool
    uncertainty_margin: float
    routable_kinds: tuple[str, ...]
    cue_groups: dict[str, tuple[re.Pattern[str], ...]]
    section_priors: dict[str, dict[str, float]]
    cases: tuple[CaseSpec, ...]
    config_hash: str

    def case_spec(self, case: str) -> CaseSpec | None:
        for c in self.cases:
            if c.case == case:
                return c
        return None


@dataclass(frozen=True, slots=True)
class LineContext:
    """Structural context for one canonical routable unit (from the L1 graph).

    A unit is a list-item content, a table/key-value row, a sentence-like span, or
    a line fallback — not necessarily a whole line.
    """

    document_id: str
    node_id: str
    start: int
    end: int
    text: str
    section_category: str | None
    is_list_item: bool
    row_kind: str | None  # None | "key_value_like" | "table_like"
    numeric_present: bool
    word_count: int
    tokens: tuple[str, ...]  # token texts
    node_kind: str = "line"  # list_item | table_row | sentence | line
    parent_line_id: str | None = None
    # A sentence nested inside a table/key-value row: routable for NARRATIVE cases
    # (C3/C4/C5) only — the row already represents its structured content (C1/C2),
    # so structured routing here would double-run a specialist.
    narrative_only: bool = False


def _hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_router_config(base_path: str | Path) -> RouterConfig:
    import yaml

    base = Path(base_path)
    base_data: dict[str, Any] = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    signals_path = base.parent / str(base_data.get("signals_file", "signals_v1.yaml"))
    sig_data: dict[str, Any] = yaml.safe_load(signals_path.read_text(encoding="utf-8")) or {}

    cue_groups: dict[str, tuple[re.Pattern[str], ...]] = {}
    for name, patterns in (sig_data.get("cue_groups", {}) or {}).items():
        cue_groups[str(name)] = tuple(
            re.compile(str(p), re.IGNORECASE | re.UNICODE) for p in patterns
        )
    section_priors = {
        str(cat): {str(k): float(v) for k, v in (m or {}).items()}
        for cat, m in (sig_data.get("section_priors", {}) or {}).items()
    }
    cases: list[CaseSpec] = []
    for case_id, spec in (sig_data.get("cases", {}) or {}).items():
        signals = tuple(
            SignalSpec(str(s["name"]), str(s["kind"]), float(s["weight"]), str(s["detector"]))
            for s in spec.get("signals", []) or []
        )
        cases.append(CaseSpec(
            case=str(case_id),
            name=str(spec.get("name", case_id)),
            activated_specialists=tuple(spec.get("activated_specialists", []) or []),
            signals=signals,
            required_evidence=tuple(
                str(name) for name in spec.get("required_evidence", []) or ()),
        ))
    cases.sort(key=lambda c: c.case)

    thresholds = base_data.get("thresholds", {}) or {}
    return RouterConfig(
        router_version=str(base_data.get("router_version", "r1b")),
        signals_version=str(sig_data.get("version", 1)),
        lexicon_version=int(sig_data.get("lexicon_version", 1)),
        activate=float(thresholds.get("activate", 0.5)),
        strong=float(thresholds.get("strong", 0.8)),
        narrative_fallback=bool(thresholds.get("narrative_fallback", True)),
        uncertainty_margin=float(base_data.get("uncertainty_margin", 0.1)),
        routable_kinds=tuple(base_data.get("routable_kinds", ["line"])),
        cue_groups=cue_groups,
        section_priors=section_priors,
        cases=tuple(cases),
        config_hash=_hash({"base": base_data, "signals": sig_data}),
    )


def cue_hit(config: RouterConfig, group: str, text: str) -> str | None:
    """Return the matched substring if any regex in ``group`` fires, else None."""
    for pat in config.cue_groups.get(group, ()):  # deterministic config order
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


# A key/value pair whose value side starts with a number: "PLT: 250",
# "WBC:14.43", "NEUT%: 76.4 %". The key is bounded to keep it short and row-like —
# a long clause before a colon is prose, not a test name. Decimal comma and point
# are both accepted (spec §14.2).
_NUMERIC_KEY_VALUE = re.compile(
    r"(?:^|[;\n])\s*([^:;\n]{1,40}?)\s*:\s*[<>~]?\s*\d+(?:[.,]\d+)?")


def evaluate_detector(
    detector: str, ctx: LineContext, config: RouterConfig
) -> tuple[bool, float | None, str | None]:
    """Evaluate a `cue.*` / `structural.*` detector → (fired, weight_override, detail).

    ``derived.*`` detectors are evaluated later (need specialist output) and here
    return not-fired.
    """
    kind, _, name = detector.partition(".")
    if kind == "cue":
        hit = cue_hit(config, name, ctx.text)
        return (hit is not None, None, hit)
    if kind == "derived":
        return (False, None, None)
    # structural.*
    if name == "is_list_item":
        return (ctx.is_list_item, None, None)
    if name == "is_key_value_row":
        return (ctx.row_kind == "key_value_like", None, None)
    if name == "is_table_row":
        # pure table rows only; key-value rows are covered by is_key_value_row
        return (ctx.row_kind == "table_like", None, None)
    if name == "semicolon_pairs":
        fired = ctx.text.count(";") >= 1 and bool(re.search(r":\s*\S", ctx.text))
        return (fired, None, None)
    if name == "numeric_present":
        return (ctx.numeric_present, None, None)
    if name == "numeric_key_value":
        # A key/value row whose VALUE side is numeric — "PLT: 250", "WBC:14.43",
        # "NEUT%: 76.4 %" (Audit 0053).
        #
        # This is the signal that actually separates a laboratory row from prose
        # that merely contains a colon and a number. `is_key_value_row` fires on
        # "Tiền sử gia đình: bố tăng huyết áp" and `numeric_present` fires on any
        # numeral anywhere, so their sum routed narrative text to C2 (Audit 0052
        # measured 11 false positives from exactly that). Requiring the value
        # *itself* to be numeric accepts a unit-less result — which spec §14.2
        # lists as a required laboratory variation — while rejecting a text value.
        match = _NUMERIC_KEY_VALUE.search(ctx.text)
        return (match is not None, None, match.group(1) if match else None)
    if name == "is_narrative":
        fired = (not ctx.is_list_item) and ctx.row_kind is None and ctx.word_count >= 6
        return (fired, None, None)
    if name == "multiple_clauses":
        fired = ctx.word_count >= 8 and (ctx.text.count(",") + ctx.text.count(";")) >= 1
        return (fired, None, None)
    if name == "uppercase_shortform":
        return (bool(_UPPER_SHORT.search(ctx.text)), None, None)
    if name == "unaccented":
        fired = ctx.text.isascii() and bool(_UNACCENTED_CUE.search(ctx.text))
        return (fired, None, None)
    if name == "mixed_language":
        fired = (not ctx.text.isascii()) and bool(_MIXED_EN.search(ctx.text))
        return (fired, None, None)
    if name == "section_prior":
        return (False, None, None)  # handled per-case via section_priors
    return (False, None, None)


__all__ = [
    "SignalSpec",
    "CaseSpec",
    "RouterConfig",
    "LineContext",
    "load_router_config",
    "cue_hit",
    "evaluate_detector",
]
