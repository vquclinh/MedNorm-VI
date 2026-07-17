"""Local Vietnamese ICD-10 snapshot contracts (Phase 1C-A).

Typed, offline model of a *locally acquired* Vietnamese ICD-10 table. Because the
organizer source/version/format is undisclosed, the loader is configurable
(column mapping) rather than tied to one file. A snapshot is never treated as
organizer-exact, and no dotted/undotted output policy is chosen here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import normalization as norm


@dataclass(frozen=True, slots=True)
class IcdConcept:
    """One ICD-10 concept with reversible dotted/undotted forms."""

    code_supplied: str  # exactly as in the source table
    dotted: str
    undotted: str
    label_vi: str
    label_en: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    parent: str | None = None  # undotted parent code
    children: tuple[str, ...] = field(default_factory=tuple)  # undotted child codes
    chapter: str = ""
    source: str = ""
    version: str = ""
    status: str = "active"  # active | inactive | unknown

    @property
    def specificity(self) -> int:
        return norm.specificity(self.undotted)

    @property
    def is_category(self) -> bool:
        return norm.is_category(self.undotted)


@dataclass(frozen=True, slots=True)
class IcdSnapshot:
    """A local ICD-10 snapshot indexed by undotted code."""

    snapshot_id: str
    source: str
    version: str
    concepts: tuple[IcdConcept, ...]
    _by_undotted: dict[str, IcdConcept] = field(default_factory=dict)

    def codes(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_undotted))

    def get(self, code: str) -> IcdConcept | None:
        return self._by_undotted.get(norm.to_undotted(code))

    def exists(self, code: str) -> bool:
        return norm.to_undotted(code) in self._by_undotted

    def children_of(self, code: str) -> tuple[str, ...]:
        c = self.get(code)
        return c.children if c else ()

    def parent_of(self, code: str) -> str | None:
        c = self.get(code)
        return c.parent if c else None


__all__ = ["IcdConcept", "IcdSnapshot"]
