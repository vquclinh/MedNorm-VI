"""Constrained structured-output parsing for the L7 LLM cascade (spec §12.1).

Spec §12.1 fixes the prompt contract: the model returns a *constrained* structured
decision, never free text and never the final document JSON. This module is the
model-independent half of that contract — recovering a JSON object from a
completion and refusing anything that is not one.

Extraction is not repair. An instruction-tuned model adds a fence or a sentence
however firmly it is told not to, so the first balanced object is located and
handed on; every schema, substring and ontology check the caller applies still
runs on whatever comes out, and a malformed payload still fails.

No model is imported here. Nothing in this module knows which model produced the
text it is given.
"""

from __future__ import annotations

import json

STRUCTURED_OUTPUT_CONTRACT_VERSION = "structured-output-v1"


class StructuredOutputError(ValueError):
    """Raised when a completion cannot be read as the required structure."""


def extract_json_object(completion: str) -> str:
    """Recover the first balanced JSON object from a completion.

    Returns the completion unchanged when no balanced object is present, so the
    caller's own parse fails with its own reason code rather than this function
    inventing one.
    """
    start = completion.find("{")
    if start < 0:
        return completion
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(completion)):
        character = completion[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return completion[start : index + 1]
    return completion


def normalize_json_completion(completion: str) -> str:
    """Normalize a completion to a JSON string the strict parsers can consume.

    Well-formed JSON is returned verbatim. Anything else is returned as the
    extracted candidate so the caller rejects it explicitly; this function never
    substitutes a default payload for a broken one.
    """
    candidate = extract_json_object(completion.strip())
    try:
        json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return candidate
    return candidate


def require_json_object(completion: str) -> dict[str, object]:
    """Parse a completion into a JSON object, or raise.

    Use this where a caller has no reason code of its own to record; the
    proposal and assertion paths use the reason-code parsers instead.
    """
    candidate = normalize_json_completion(completion)
    try:
        document = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as error:
        raise StructuredOutputError(
            f"completion is not well-formed JSON ({type(error).__name__})") from error
    if not isinstance(document, dict):
        raise StructuredOutputError(
            f"completion is JSON but not an object (got {type(document).__name__})")
    return document


__all__ = [
    "STRUCTURED_OUTPUT_CONTRACT_VERSION",
    "StructuredOutputError",
    "extract_json_object",
    "normalize_json_completion",
    "require_json_object",
]
