"""Candidate-local ontology facts and KB document construction (GraphCENT 0080).

Lightweight by design: no GNN, no graph embedding, no message passing. For each candidate we
attach the few facts a clinician would check - what it is, what it sits under, how many
siblings compete with it, and for a drug its ingredient/strength/form.

One rule matters more than the rest. **A parent, sibling or child is never a substitute for
the right code.** Papers that report hierarchical metrics can afford to reward a near miss;
this competition scores exact candidate sets, so ontological proximity is evidence for
*disambiguation* and never a reason to emit a neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..kb.indexing.retrieval import LocalIndex
from ..kb.rxnorm.structured import StructuredDrug, normalize_unit, parse_strengths
from .icd10_titles import is_damaged_title
from .semantic_cache import KbDocument

ONTOLOGY_ICD = "ICD10"
ONTOLOGY_RXNORM = "RXNORM"

CONFLICT_INGREDIENT = "ingredient_mismatch"
CONFLICT_STRENGTH = "strength_conflict"
CONFLICT_FORM = "dose_form_conflict"


@dataclass(frozen=True, slots=True)
class IcdFacts:
    code: str
    label: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    parent_code: str = ""
    parent_label: str = ""
    sibling_count: int = 0
    child_count: int = 0

    def as_prompt_lines(self) -> list[str]:
        lines = [f"code {self.code}: {self.label}"]
        if self.aliases:
            lines.append("  also called: " + "; ".join(self.aliases[:4]))
        if self.parent_code:
            lines.append(f"  category: {self.parent_code} {self.parent_label}".rstrip())
        if self.sibling_count:
            lines.append(f"  competing sibling codes in this category: {self.sibling_count}")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "aliases": list(self.aliases),
            "parent_code": self.parent_code,
            "parent_label": self.parent_label,
            "sibling_count": self.sibling_count,
            "child_count": self.child_count,
        }


@dataclass(frozen=True, slots=True)
class RxNormFacts:
    rxcui: str
    label: str
    tty: str = ""
    ingredients: tuple[str, ...] = field(default_factory=tuple)
    strength: str = ""
    dose_form: str = ""
    conflicts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    def as_prompt_lines(self) -> list[str]:
        lines = [f"RxCUI {self.rxcui}: {self.label}"]
        detail = []
        if self.tty:
            detail.append(f"TTY {self.tty}")
        if self.ingredients:
            detail.append("ingredient " + "/".join(self.ingredients[:2]))
        if self.strength:
            detail.append(f"strength {self.strength}")
        if self.dose_form:
            detail.append(f"form {self.dose_form}")
        if detail:
            lines.append("  " + "; ".join(detail))
        if self.conflicts:
            lines.append("  CONFLICTS WITH THE MENTION: " + ", ".join(self.conflicts))
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "rxcui": self.rxcui,
            "label": self.label,
            "tty": self.tty,
            "ingredients": list(self.ingredients),
            "strength": self.strength,
            "dose_form": self.dose_form,
            "conflicts": list(self.conflicts),
        }


def icd_facts(index: LocalIndex, code: str) -> IcdFacts | None:
    record = index.records.get(code)
    if record is None:
        return None
    label = str(record.get("canonical_name", ""))
    parent = code[:3]
    parent_record = index.records.get(parent) if parent != code else None
    siblings = (
        sum(1 for c in index.records if c[:3] == parent and c != code and len(c) == len(code))
        if len(code) > 3
        else 0
    )
    children = sum(1 for c in index.records if c.startswith(code) and c != code)
    return IcdFacts(
        code=code,
        label="" if is_damaged_title(label) else label,
        aliases=tuple(str(a) for a in (record.get("aliases") or [])[:6]),
        parent_code=parent if parent_record is not None else "",
        parent_label=(
            str(parent_record.get("canonical_name", "")) if parent_record is not None else ""
        ),
        sibling_count=siblings,
        child_count=children,
    )


def rxnorm_facts(
    index: LocalIndex, rxcui: str, mention: str, structured: dict[str, StructuredDrug]
) -> RxNormFacts | None:
    record = index.records.get(rxcui)
    if record is None:
        return None
    drug = structured.get(rxcui)
    label = str(record.get("canonical_name", ""))
    metadata = record.get("metadata") or {}
    conflicts: list[str] = []

    mention_strengths = {s.key for s in parse_strengths(mention)}
    if drug is not None:
        if (
            mention_strengths
            and drug.strength_keys
            and not (mention_strengths & drug.strength_keys)
        ):
            conflicts.append(CONFLICT_STRENGTH)
        recorded_forms = {normalize_unit(f) for f in (*drug.dose_forms, drug.rxterm_form) if f}
        lowered = mention.casefold()
        if recorded_forms and drug.ingredients:
            stated = {f for f in recorded_forms if f and f in lowered.replace(" ", "")}
            if not stated and any(
                cue in lowered for cue in ("viên", "ống", "tiêm", "siro", "kem", "gói")
            ):
                conflicts.append(CONFLICT_FORM)
        if drug.ingredients and not any(
            i.casefold().split()[0] in lowered for i in drug.ingredients if i
        ):
            conflicts.append(CONFLICT_INGREDIENT)

    return RxNormFacts(
        rxcui=rxcui,
        label=label,
        tty=str(metadata.get("tty", "")),
        ingredients=tuple(drug.ingredients[:3]) if drug else (),
        strength=(
            (drug.strengths[0].raw or drug.strengths[0].key)
            if drug and drug.strengths
            else (drug.available_strength if drug else "")
        ),
        dose_form=(
            drug.dose_forms[0] if drug and drug.dose_forms else (drug.rxterm_form if drug else "")
        ),
        conflicts=tuple(dict.fromkeys(conflicts)),
    )


def build_kb_documents(
    icd_index: LocalIndex | None,
    rxnorm_index: LocalIndex | None,
    structured: dict[str, StructuredDrug] | None = None,
) -> list[KbDocument]:
    """Retrieval corpus over governed ids only. Deterministic order: ICD then RxNorm, sorted."""
    documents: list[KbDocument] = []
    if icd_index is not None:
        for code in sorted(icd_index.records):
            facts = icd_facts(icd_index, code)
            if facts is None:
                continue
            parts = [facts.label or code]
            if facts.aliases:
                parts.append("; ".join(facts.aliases[:4]))
            if facts.parent_label and not is_damaged_title(facts.parent_label):
                parts.append(f"thuộc nhóm {facts.parent_label}")
            documents.append(KbDocument(code, ONTOLOGY_ICD, " | ".join(p for p in parts if p)))
    if rxnorm_index is not None:
        structured = structured or {}
        for rxcui in sorted(rxnorm_index.records):
            record = rxnorm_index.records[rxcui]
            drug = structured.get(rxcui)
            parts = [str(record.get("canonical_name", ""))]
            if drug is not None:
                if drug.ingredients:
                    parts.append("ingredient " + "/".join(drug.ingredients[:2]))
                if drug.strengths:
                    parts.append("strength " + (drug.strengths[0].raw or ""))
                if drug.dose_forms:
                    parts.append("form " + drug.dose_forms[0])
            documents.append(KbDocument(rxcui, ONTOLOGY_RXNORM, " | ".join(p for p in parts if p)))
    return documents


__all__ = [
    "CONFLICT_FORM",
    "CONFLICT_INGREDIENT",
    "CONFLICT_STRENGTH",
    "ONTOLOGY_ICD",
    "ONTOLOGY_RXNORM",
    "IcdFacts",
    "RxNormFacts",
    "build_kb_documents",
    "icd_facts",
    "rxnorm_facts",
]
