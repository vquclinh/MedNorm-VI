"""Qwen generative query reformulation, BeLink-style (0082).

The verified 0081 mention is a retrieval *query* problem: a Vietnamese clinical phrase has to
find a governed concept whose label is often English, abbreviated, or written in a different
register. So before retrieval, Qwen writes two additional query views - a canonical
Vietnamese term and a canonical English/scientific term - using the document context around
the mention, which is the contextual-query idea from KRISSBERT without deploying KRISSBERT.

Three invariants make this safe:

* the reformulation is a **query only**. It never becomes an entity, never touches the span,
  never touches offsets, and never reaches the output;
* the model is forbidden to write a code, and any code-shaped output is discarded;
* an unparseable reply falls back to the raw mention, which is the view retrieval would have
  used anyway. Failure costs nothing.

Reformulations are cached per `(qwen revision, prompt version, document checksum, mention,
context)` so a mention repeated across a note costs one call, and a cache from a different
model or a different prompt can never be mistaken for this one's.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

PROMPT_VERSION = "ontofusion-genqr-v1"

_JSON_OBJECT = re.compile(r"\{.*?\}", re.S)

#: Anything shaped like a governed identifier. A query view is prose; a code here means the
#: model answered the wrong question, and the view is dropped rather than searched with.
_CODE_SHAPED = re.compile(r"\b([A-TV-Z]\d{2,4}(?:\.\d{1,2})?|\d{4,8})\b")

GENQR_RUBRIC = (
    "You rewrite a clinical mention into search terms. You are NOT diagnosing and you are "
    "NOT coding.\n"
    "\n"
    "Given a mention from a Vietnamese clinical note and its surrounding context, return "
    "two search queries:\n"
    "  canonical_vi - the standard Vietnamese medical term for what the mention denotes\n"
    "  canonical_en - the standard English or scientific term for the same thing\n"
    "\n"
    "RULES\n"
    "* These are SEARCH QUERIES ONLY. They are never shown to anyone and never become part "
    "of the answer.\n"
    "* NEVER return an ICD-10 code, an RxNorm identifier, or any other code or number that "
    "identifies a concept. Return words.\n"
    "* Do not add detail the mention and its context do not support. Do not guess a "
    "subtype, a site, a severity, a strength or a form.\n"
    "* Expand an abbreviation only when the context makes it certain.\n"
    "* If you are not confident, return the mention unchanged as canonical_vi and an empty "
    "canonical_en. That is a correct answer, not a failure.\n"
)

GENQR_SCHEMA = (
    'Return JSON only, exactly:\n'
    '{"canonical_vi": "...", "canonical_en": "..."}\n'
)


@dataclass(frozen=True, slots=True)
class MentionContext:
    """The document evidence a reformulation may use. Identifiers only, never free text out."""

    document: str
    mention: str
    entity_type: str
    section: str = ""
    previous_sentence: str = ""
    sentence: str = ""
    next_sentence: str = ""

    def cache_key(self, *, revision: str, document_checksum: str) -> str:
        payload = json.dumps(
            {
                "revision": revision,
                "prompt": PROMPT_VERSION,
                "document_checksum": document_checksum,
                "mention": self.mention,
                "type": self.entity_type,
                "section": self.section,
                "sentence": self.sentence,
            },
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": self.document, "mention": self.mention,
            "type": self.entity_type, "section": self.section,
            "previous_sentence": self.previous_sentence, "sentence": self.sentence,
            "next_sentence": self.next_sentence,
        }


def build_genqr_prompt(context: MentionContext) -> str:
    lines = [GENQR_RUBRIC, ""]
    lines.append(f"ENTITY TYPE: {context.entity_type}")
    if context.section:
        lines.append(f"SECTION: {context.section}")
    lines.append("CONTEXT:")
    for label, text in (
        ("previous", context.previous_sentence),
        ("current", context.sentence),
        ("next", context.next_sentence),
    ):
        if text.strip():
            lines.append(f"  ({label}) {' '.join(text.split())}")
    lines.append(f"MENTION: {context.mention}")
    lines.append("")
    lines.append(GENQR_SCHEMA)
    return "\n".join(lines)


PARSE_FAILED = "genqr_parse_failed"
DROPPED_CODE = "genqr_code_shaped_view_dropped"


@dataclass(frozen=True, slots=True)
class Reformulation:
    """Two extra query views. `raw` always participates in retrieval regardless."""

    raw: str
    canonical_vi: str = ""
    canonical_en: str = ""
    parse_failed: bool = False

    @property
    def views(self) -> tuple[str, ...]:
        """Distinct non-empty query views, raw first. Order is stable."""
        seen: list[str] = []
        for text in (self.raw, self.canonical_vi, self.canonical_en):
            cleaned = " ".join((text or "").split())
            if cleaned and cleaned.casefold() not in {s.casefold() for s in seen}:
                seen.append(cleaned)
        return tuple(seen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw, "canonical_vi": self.canonical_vi,
            "canonical_en": self.canonical_en, "parse_failed": self.parse_failed,
        }


def parse_reformulation(reply: str, mention: str) -> tuple[Reformulation, dict[str, int]]:
    """Strict parse. Anything unusable falls back to the raw mention."""
    counters: dict[str, int] = {}

    def count(reason: str) -> None:
        counters[reason] = counters.get(reason, 0) + 1

    match = _JSON_OBJECT.search(reply or "")
    if not match:
        count(PARSE_FAILED)
        return Reformulation(raw=mention, parse_failed=True), counters
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        count(PARSE_FAILED)
        return Reformulation(raw=mention, parse_failed=True), counters
    if not isinstance(payload, dict):
        count(PARSE_FAILED)
        return Reformulation(raw=mention, parse_failed=True), counters

    def view(key: str) -> str:
        text = " ".join(str(payload.get(key, "") or "").split())
        if not text:
            return ""
        if _CODE_SHAPED.search(text):
            count(DROPPED_CODE)
            return ""
        return text

    return Reformulation(
        raw=mention, canonical_vi=view("canonical_vi"), canonical_en=view("canonical_en")
    ), counters


class Asker:
    """Minimal protocol: anything that answers a prompt. The deployed Qwen, or a fake."""

    def ask(self, prompt: str) -> str:  # pragma: no cover - structural only
        raise TypeError("Asker is a protocol; supply the deployed model or a fake")


@dataclass
class ReformulationCache:
    """Provenance-keyed reformulations for one run."""

    revision: str
    document_checksums: dict[str, str] = field(default_factory=dict)
    entries: dict[str, Reformulation] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    calls: int = 0

    def count(self, reason: str, amount: int = 1) -> None:
        if amount:
            self.counters[reason] = self.counters.get(reason, 0) + amount

    def reformulate(self, context: MentionContext, asker: Any) -> Reformulation:
        checksum = self.document_checksums.get(context.document, "")
        key = context.cache_key(revision=self.revision, document_checksum=checksum)
        cached = self.entries.get(key)
        if cached is not None:
            self.count("reformulation_cache_hits")
            return cached
        if asker is None:
            result = Reformulation(raw=context.mention)
        else:
            self.calls += 1
            result, counters = parse_reformulation(
                asker.ask(build_genqr_prompt(context)), context.mention
            )
            for reason, value in counters.items():
                self.count(reason, value)
        self.entries[key] = result
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "prompt_version": PROMPT_VERSION,
            "unique_reformulations": len(self.entries),
            "qwen_calls": self.calls,
            "counters": dict(self.counters),
        }


__all__ = [
    "DROPPED_CODE",
    "GENQR_RUBRIC",
    "GENQR_SCHEMA",
    "PARSE_FAILED",
    "PROMPT_VERSION",
    "Asker",
    "MentionContext",
    "Reformulation",
    "ReformulationCache",
    "build_genqr_prompt",
    "parse_reformulation",
]
