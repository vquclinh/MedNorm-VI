"""Constrained prompts for the 8B reasoner (sprint 0075).

Three tasks, each returning strict JSON the validator can refuse. The prompts state the
constraints the validator enforces, so a compliant model and a strict validator agree instead
of fighting: exact substrings only, five types only, candidate ids only from the offered list,
`[]` when nothing fits.
"""

from __future__ import annotations

from typing import Any

from .pool import Candidate
from .validator import ASSERTION_KEYS, ORGANIZER_TYPES

SYSTEM = (
    "You are a precise Vietnamese clinical information-extraction engine. "
    "You return STRICT JSON and nothing else. You never invent text, offsets, or medical "
    "codes. Every span you output must be copied verbatim from the note."
)

TYPE_GUIDE = (
    "TRIỆU_CHỨNG = symptom/sign; "
    "TÊN_XÉT_NGHIỆM = name of a laboratory test; "
    "KẾT_QUẢ_XÉT_NGHIỆM = a laboratory result value; "
    "CHẨN_ĐOÁN = diagnosis/disease; "
    "THUỐC = medication/drug."
)


def entity_prompt(note: str, existing: list[dict[str, Any]] | None = None) -> str:
    """Entity discovery + typing. Existing E3 spans are shown so the model corrects, not
    duplicates."""
    seed = ""
    if existing:
        listed = "; ".join(f"{e['text']} [{e['type']}]" for e in existing[:60])
        seed = (
            "\nAlready detected by an earlier component (keep the correct ones, fix wrong "
            f"types, add what is missing):\n{listed}\n"
        )
    return (
        f"Extract every clinical entity from the Vietnamese note below.\n"
        f"Allowed types ONLY: {', '.join(ORGANIZER_TYPES)}.\n{TYPE_GUIDE}\n"
        f"{seed}\n"
        "RULES:\n"
        "1. `text` MUST be an exact substring of the note, copied character for character.\n"
        "2. Do NOT output character offsets; they are computed downstream.\n"
        "3. Do NOT translate, normalize, expand abbreviations, or fix spelling.\n"
        "4. If the same phrase occurs several times as separate entities, list it once per "
        "occurrence.\n\n"
        'Return JSON: {"entities":[{"text":"...","type":"..."}]}\n\n'
        f"NOTE:\n{note}\n"
    )


def assertion_prompt(note: str, entities: list[dict[str, Any]]) -> str:
    listed = "\n".join(f'{i}. "{e["text"]}" [{e["type"]}]' for i, e in enumerate(entities))
    flags = ",".join(f'"{key}":false' for key in ASSERTION_KEYS)
    schema = f'{{"assertions":[{{"index":0,{flags}}}]}}'
    return (
        "For each listed entity decide three independent booleans from the note:\n"
        "- isNegated: the note states it is absent/denied/ruled out.\n"
        "- isFamily: it belongs to a family member, not the patient.\n"
        "- isHistorical: it is past history, not the current episode.\n"
        "Default every flag to false unless the note supports it. Use surrounding context.\n\n"
        f"Return JSON: {schema}\n\n"
        f"NOTE:\n{note}\n\nENTITIES:\n{listed}\n"
    )


def candidate_prompt(note: str, mention: str, entity_type: str, candidates: list[Candidate]) -> str:
    listed = "\n".join(f"{i}. {c.as_prompt_line()}" for i, c in enumerate(candidates))
    ontology = "ICD-10" if entity_type == "CHẨN_ĐOÁN" else "RxNorm"
    return (
        f"Select the {ontology} concepts that correctly normalize the mention.\n\n"
        "RULES:\n"
        "1. Choose ONLY from the numbered list. Never write a code that is not listed.\n"
        "2. Return [] if none of them is correct. An empty answer is a valid, expected "
        "answer.\n"
        "3. Do NOT infer a disease subtype, organism, drug strength, dose form, or brand "
        "that the note does not state.\n"
        "4. You may select more than one when the mention is genuinely ambiguous.\n\n"
        'Return JSON: {"selected":[<list indices>]}\n\n'
        f"NOTE:\n{note}\n\nMENTION: {mention}\n\nCANDIDATES:\n{listed}\n"
    )


__all__ = ["SYSTEM", "TYPE_GUIDE", "assertion_prompt", "candidate_prompt", "entity_prompt"]
