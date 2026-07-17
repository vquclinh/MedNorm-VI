"""Phase 2A corpus-analysis contracts (descriptive statistics only).

These dataclasses describe the ORGANIZER'S PUBLIC INPUT documents structurally.
Nothing here predicts, extracts entities, links ontologies, or trains anything —
Phase 2A is measurement, not modelling. All collections are ordered
deterministically so reports are byte-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Distribution:
    """A deterministic numeric summary of one measured quantity."""

    n: int
    total: int
    minimum: int
    maximum: int
    mean: float
    median: float
    p25: float
    p75: float
    p90: float

    def as_dict(self) -> dict[str, float | int]:
        return {"n": self.n, "total": self.total, "min": self.minimum, "max": self.maximum,
                "mean": self.mean, "median": self.median, "p25": self.p25, "p75": self.p75,
                "p90": self.p90}


@dataclass(frozen=True, slots=True)
class LengthStats:
    characters: int
    bytes_utf8: int
    lines: int
    non_blank_lines: int
    blank_lines: int
    max_line_length: int
    tokens: int
    word_tokens: int
    number_tokens: int
    sentences: int


@dataclass(frozen=True, slots=True)
class SectionStats:
    """L1 section structure (headings are STRUCTURE labels, never entities)."""

    sections: int
    categorized: int
    uncategorized: int
    categories: tuple[str, ...]  # sorted distinct category names
    headings: tuple[str, ...]  # heading surface texts, in document order


@dataclass(frozen=True, slots=True)
class StructureStats:
    """List / table / key-value structure from the L1 graph."""

    list_items: int
    bullet_items: int
    numbered_items: int
    max_indent: int
    max_depth: int
    table_rows: int
    key_value_rows: int
    table_like_rows: int
    paragraphs: int


@dataclass(frozen=True, slots=True)
class RoutingStats:
    """Canonical routable-unit and processing-case counts (L2).

    A route case is a PROCESSING CASE, not an entity type and not a prediction.
    Only structural cases C1-C5 are counted: C6/C7 are derived from specialist
    proposals, which Phase 2A deliberately does not run.
    """

    units_by_kind: dict[str, int] = field(default_factory=dict)
    cases: dict[str, int] = field(default_factory=dict)
    multi_case_units: int = 0
    unrouted_units: int = 0


@dataclass(frozen=True, slots=True)
class LayoutStats:
    """Laboratory / medication / imaging LAYOUT counts (shape, not entities)."""

    lab_lines: int = 0
    lab_numeric_with_unit: int = 0
    lab_reference_range: int = 0
    lab_qualitative: int = 0
    lab_normality_phrase: int = 0
    lab_bare_numeric: int = 0
    lab_section_cue_lines: int = 0
    med_lines: int = 0
    med_strength: int = 0
    med_route: int = 0
    med_frequency: int = 0
    med_dose_form: int = 0
    med_section_cue_lines: int = 0
    med_full_pattern: int = 0  # strength + route + frequency on one line
    imaging_lines: int = 0
    imaging_modality: int = 0
    imaging_section_cue_lines: int = 0
    imaging_trailing_parenthetical: int = 0


@dataclass(frozen=True, slots=True)
class DocumentAnalysis:
    """Everything Phase 2A measured for one public input document."""

    document_id: str
    source_name: str  # file stem, e.g. "42"
    content_sha256: str
    length: LengthStats
    sections: SectionStats
    structure: StructureStats
    routing: RoutingStats
    layout: LayoutStats
    profile: str  # list_dominant | narrative_dominant | mixed
    l1_valid: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CorpusAnalysis:
    """Aggregated, deterministic view over the whole public corpus."""

    corpus_id: str
    n_documents: int
    corpus_sha256: str
    analysis_version: str
    config_hash: str
    l1_config_hash: str
    documents: tuple[DocumentAnalysis, ...]
    distributions: dict[str, Distribution] = field(default_factory=dict)
    heading_frequencies: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    category_frequencies: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    unit_kind_frequencies: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    case_frequencies: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    profile_frequencies: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    length_histogram: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    layout_totals: dict[str, int] = field(default_factory=dict)
    docs_with: dict[str, int] = field(default_factory=dict)  # doc-level presence counts
    warnings: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "Distribution",
    "LengthStats",
    "SectionStats",
    "StructureStats",
    "RoutingStats",
    "LayoutStats",
    "DocumentAnalysis",
    "CorpusAnalysis",
]
