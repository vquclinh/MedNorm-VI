"""L5 — Specialists (spec §8).

Specialists produce SCORED evidence bundles, never final answers: they may emit
scored boundary *feedback* but must not edit spans, invent codes, or link against
the wrong ontology (spec §19.1).

    ``assertion``  Assertion Hydra — cue, scope, section, experiencer (spec §8)

The two ontology linkers live under ``mednorm_vi.linking`` — the path spec §19
names (``linking/icd/``, ``linking/rxnorm/``). Audit 0051 removed the duplicate
``specialists.icd`` / ``specialists.rxnorm`` bootstrap packages: they wrapped the
real linkers in ``link()`` functions that returned an empty list, had no consumer,
and gave each linker two import paths with different behaviour.
"""

from __future__ import annotations

__all__: list[str] = []
