"""Deterministic safe-bridge filter over completed 0078 merges (experiment 0079).

0078 merged 57 of 70 eligible clusters (81.4%). Inspection showed the model merging genuinely
distinct findings whenever a connector sat between them: `hẹp tắc mạch vành` + `suy tim` became
`hẹp tắc mạch vành gây suy tim`, `Kawasaki` + `bệnh viêm mạch máu` became `Kawasaki là bệnh
viêm mạch máu`. Those are two entities joined by discourse, not one fragmented phrase.

The distinction is visible in the source text without any model. A genuine fragmentation
repair bridges its pieces with **at most one ordinary word** - `Thiếu` _men_ `G6PD`,
`tăng sản tuyến` _tiền_ `liệt`, `khó thở khi` _gắng_ `sức`. A spurious merge bridges them with
a connector, a clause, or punctuation.

So this filter reads only the exact source between consecutive fragments and rejects the whole
merge when that bridge crosses punctuation, runs to more than one token, carries a known
connector, or contains an unexpectedly capitalised word (which is how `ho` + `nhĩ` became
`ho Rung nhĩ`). It can only ACCEPT an existing 0078 merge or REVERT to the E3 originals; it
never invents one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

SAFE_BRIDGE_VERSION = "safe-bridge-filter-v1"

#: At most one ordinary word may sit between two fragments of the same phrase.
MAX_BRIDGE_TOKENS = 1

#: Connectors that join two distinct findings. Normalized (case-folded, whitespace-compacted)
#: before matching, so `Kèm Theo` and `kèm theo` behave alike. Multi-word forms are checked
#: against the whole bridge, single words against each token.
BRIDGE_DENYLIST: tuple[str, ...] = (
    "và",
    "hoặc",
    "hay",
    "kèm",
    "kèm theo",
    "gây",
    "do",
    "là",
    "với",
    "dẫn",
    "dẫn đến",
    "sau",
    "sau khi",
    "trước",
    "trước khi",
    "khi",
    "thì",
    "xuất hiện",
    "gồm",
    "bao gồm",
    "trở lại",
    "cùng",
    "đồng thời",
    "nhưng",
    "rồi",
    "tình trạng",
    "biểu hiện",
    "gây ra",
    "kết hợp",
    "đi kèm",
    "vì",
    "nên",
    "mà",
)

_TOKEN = re.compile(r"\S+")
_PUNCTUATION = re.compile(r"[.,;:!?()\[\]{}/\\|\"'“”‘’…–—\-+*=<>%&@#~`\n\r\t]")
_SPACE = re.compile(r"\s+")

REJECT_PUNCTUATION = "bridge_crosses_punctuation_or_newline"
REJECT_TOO_LONG = "bridge_longer_than_one_token"
REJECT_CONNECTOR = "bridge_contains_connector"
REJECT_CAPITALISED = "bridge_contains_unexpected_capitalised_word"
ACCEPT_ADJACENT = "fragments_overlap_or_touch"
ACCEPT_SINGLE_WORD = "bridge_is_a_single_ordinary_word"


def normalize(text: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFC", text or "")).strip().casefold()


@dataclass(frozen=True, slots=True)
class BridgeVerdict:
    accepted: bool
    reason: str
    gaps: tuple[str, ...] = field(default_factory=tuple)
    normalized_tokens: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": "ACCEPT" if self.accepted else "REJECT",
            "reason": self.reason,
            "inter_fragment_gaps": list(self.gaps),
            "normalized_gap_tokens": list(self.normalized_tokens),
        }


def _bridge_is_safe(bridge: str) -> tuple[bool, str, list[str]]:
    """Judge one inter-fragment gap. Returns (safe, reason, normalized tokens)."""
    if _PUNCTUATION.search(bridge):
        return False, REJECT_PUNCTUATION, []
    raw_tokens = _TOKEN.findall(bridge)
    tokens = [normalize(t) for t in raw_tokens]
    if not tokens:
        return True, ACCEPT_ADJACENT, []
    if len(tokens) > MAX_BRIDGE_TOKENS:
        return False, REJECT_TOO_LONG, tokens
    whole = normalize(bridge)
    if whole in BRIDGE_DENYLIST or any(t in BRIDGE_DENYLIST for t in tokens):
        return False, REJECT_CONNECTOR, tokens
    # An uppercase word inside the bridge is a different concept intruding, not connective
    # tissue. This looks ONLY at the bridge, so an acronym that is itself an E3 span - G6PD,
    # HIV - is untouched.
    for raw in raw_tokens:
        first = next((c for c in raw if c.isalpha()), "")
        if first and first.isupper():
            return False, REJECT_CAPITALISED, tokens
    return True, ACCEPT_SINGLE_WORD, tokens


def evaluate_merge(source: str, original_offsets: list[list[int]]) -> BridgeVerdict:
    """Accept a 0078 merge only if every inter-fragment gap is a safe bridge."""
    spans = sorted((int(a), int(b)) for a, b in original_offsets)
    gaps: list[str] = []
    tokens: list[str] = []
    covered = spans[0][1]
    for start, end in spans[1:]:
        if start > covered:
            bridge = source[covered:start]
            gaps.append(bridge)
            safe, reason, normalized = _bridge_is_safe(bridge)
            tokens.extend(normalized)
            if not safe:
                return BridgeVerdict(False, reason, tuple(gaps), tuple(tokens))
        covered = max(covered, end)
    reason = ACCEPT_SINGLE_WORD if tokens else ACCEPT_ADJACENT
    return BridgeVerdict(True, reason, tuple(gaps), tuple(tokens))


__all__ = [
    "ACCEPT_ADJACENT",
    "ACCEPT_SINGLE_WORD",
    "BRIDGE_DENYLIST",
    "MAX_BRIDGE_TOKENS",
    "REJECT_CAPITALISED",
    "REJECT_CONNECTOR",
    "REJECT_PUNCTUATION",
    "REJECT_TOO_LONG",
    "SAFE_BRIDGE_VERSION",
    "BridgeVerdict",
    "evaluate_merge",
    "normalize",
]
