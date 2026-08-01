"""Query and document representations for the pretrained hybrid linker (Audit 0071 §5-§6).

Everything here is deterministic string construction. No model is loaded, nothing is
downloaded, and no network call is made - which is what lets the representation be built,
reviewed and tested on a laptop while the encoders themselves run on Colab.

Two asymmetries are deliberate:

* **Context is evidence for linking, never for extraction.** The bounded context enters the
  *query* so a mention like `viêm` can be resolved by the sentence around it, but the entity
  span, its text and its type are read-only inputs. Nothing here can create, move or retype a
  mention.
* **Damaged titles never become positive semantic text.** Audit 0069 left 9.4% of ICD titles
  unrepaired. Embedding `Bao gồm: Bóng` as if it were a concept name would teach a retriever
  that an ICD instruction *is* the concept. Damaged titles are excluded from the semantic
  document and kept in provenance instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...kb.icd10.repair.title_recovery import is_damaged_title

#: Instruction prefix for Qwen3-Embedding, which is instruction-aware. Kept in English
#: because that is how the official model was trained, and deliberately negative about
#: inference: the linker must not invent specificity the clinical text does not state.
QUERY_INSTRUCTION = (
    "Instruct: Retrieve the ICD-10 or RxNorm concept that exactly normalizes the "
    "Vietnamese clinical mention. Do not infer unstated disease subtype, organism, "
    "drug strength, dose form, or brand."
)

#: Bounded context. Long enough to disambiguate a bare mention, short enough that the query
#: stays about the mention rather than the paragraph.
CONTEXT_CHARACTER_BUDGET = 160

REPRESENTATION_VERSION = "semantic-representation-v1"


def _clip(text: str, budget: int, *, keep_tail: bool) -> str:
    value = " ".join((text or "").split())
    if len(value) <= budget:
        return value
    return value[-budget:] if keep_tail else value[:budget]


@dataclass(frozen=True, slots=True)
class SemanticQuery:
    """One mention rendered for a dense retriever or a cross-encoder reranker."""

    mention_text: str
    entity_type: str
    context_before: str
    context_after: str
    ontology: str

    def render(self, *, with_instruction: bool = True) -> str:
        before = _clip(self.context_before, CONTEXT_CHARACTER_BUDGET, keep_tail=True)
        after = _clip(self.context_after, CONTEXT_CHARACTER_BUDGET, keep_tail=False)
        parts = [
            f"ontology: {self.ontology}",
            f"entity type: {self.entity_type}",
            f"mention: {self.mention_text}",
        ]
        if before:
            parts.append(f"context before: {before}")
        if after:
            parts.append(f"context after: {after}")
        body = " | ".join(parts)
        return f"{QUERY_INSTRUCTION}\nQuery: {body}" if with_instruction else body


@dataclass(frozen=True, slots=True)
class SemanticDocument:
    """One governed concept rendered for indexing. Carries provenance it never embeds."""

    concept_id: str
    ontology: str
    text: str
    excluded_damaged_title: str = ""
    fields_used: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "ontology": self.ontology,
            "text": self.text,
            "fields_used": list(self.fields_used),
            "excluded_damaged_title": self.excluded_damaged_title,
        }


def icd_document(
    record: dict[str, Any], *, parent_title: str = "", alias_limit: int = 8
) -> SemanticDocument:
    """Bounded ICD semantic document: code, repaired title, aliases, parent, specificity."""
    code = str(record.get("concept_id", ""))
    metadata = record.get("metadata") or {}
    title = str(record.get("canonical_name", ""))
    damaged = is_damaged_title(title)

    used: list[str] = ["code"]
    parts = [f"ICD-10 {metadata.get('dotted_code', code)}"]
    if not damaged and title:
        parts.append(title)
        used.append("canonical_name")

    aliases = [
        a
        for a in (record.get("aliases") or [])
        if a and a != title and not is_damaged_title(str(a))
    ][:alias_limit]
    if aliases:
        parts.append("cũng gọi là: " + "; ".join(str(a) for a in aliases))
        used.append("aliases")
    if parent_title and not is_damaged_title(parent_title):
        parts.append(f"thuộc nhóm: {parent_title}")
        used.append("parent_title")

    specificity = metadata.get("specificity")
    if specificity not in (None, ""):
        parts.append(f"specificity: {specificity}")
        used.append("specificity")
    parts.append(f"depth: {max(0, len(code) - 3)}")
    used.append("hierarchy_depth")

    return SemanticDocument(
        concept_id=code,
        ontology="ICD10",
        text=" | ".join(parts),
        excluded_damaged_title=title if damaged else "",
        fields_used=tuple(used),
    )


#: RxNorm attributes worth embedding, in a fixed order so the document is stable.
RXNORM_FIELDS: tuple[tuple[str, str], ...] = (
    ("tty", "TTY"),
    ("ingredient", "ingredient"),
    ("precise_ingredient", "precise ingredient"),
    ("strength", "strength"),
    ("dose_form", "dose form"),
    ("brand", "brand"),
)


def rxnorm_document(record: dict[str, Any], *, alias_limit: int = 8) -> SemanticDocument:
    """Bounded RxNorm semantic document. Graph neighbourhoods are deliberately excluded."""
    cui = str(record.get("concept_id", ""))
    metadata = record.get("metadata") or {}
    name = str(record.get("canonical_name", ""))

    used = ["rxcui"]
    parts = [f"RxCUI {cui}"]
    if name:
        parts.append(name)
        used.append("canonical_name")
    for key, label in RXNORM_FIELDS:
        value = metadata.get(key)
        if value not in (None, "", []):
            parts.append(f"{label}: {value}")
            used.append(key)
    aliases = [a for a in (record.get("aliases") or []) if a and a != name][:alias_limit]
    if aliases:
        parts.append("also known as: " + "; ".join(str(a) for a in aliases))
        used.append("aliases")

    return SemanticDocument(
        concept_id=cui, ontology="RXNORM", text=" | ".join(parts), fields_used=tuple(used)
    )


__all__ = [
    "CONTEXT_CHARACTER_BUDGET",
    "QUERY_INSTRUCTION",
    "REPRESENTATION_VERSION",
    "RXNORM_FIELDS",
    "SemanticDocument",
    "SemanticQuery",
    "icd_document",
    "rxnorm_document",
]
