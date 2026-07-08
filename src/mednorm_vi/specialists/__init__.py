"""L5 — Specialists (spec sections 8-10).

Three specialist families produce SCORED evidence bundles (never final answers):

* ``assertion`` — Assertion Hydra (cue + scope + section + experiencer).
* ``icd``       — ICD-10 Super Linker (DIAGNOSIS -> ICD-10 codes).
* ``rxnorm``    — RxNorm Super Linker (MEDICATION -> RxCUIs).

Specialists may only emit scored boundary FEEDBACK; they must not edit spans,
invent codes, or link against the wrong ontology.

Status: NOT IMPLEMENTED (bootstrap). Interfaces only. No KB indices are built.
"""

from __future__ import annotations

__all__: list[str] = []
