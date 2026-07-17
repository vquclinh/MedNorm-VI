"""Descriptive laboratory / medication / imaging LAYOUT statistics (Phase 2A).

This module measures the *shape* of lines — does a line carry a numeric+unit, a
reference range, a strength+route+frequency pattern, a trailing modality
parenthetical — to describe how the corpus is written. It performs **no entity
extraction, no prediction, and no linking**: nothing here decides that a line
contains a medication or a lab result, only that it *looks like* a given layout.
"""

from __future__ import annotations

import re

from .config import AnalysisConfig
from .models import LayoutStats


def _hit(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _group(cfg_group: dict[str, tuple[re.Pattern[str], ...]], name: str,
           text: str) -> bool:
    return _hit(cfg_group.get(name, ()), text)


def analyze_layouts(lines: list[str], config: AnalysisConfig) -> LayoutStats:
    """Count layout signals over a document's non-blank lines."""
    lab_lines = lab_num = lab_ref = lab_qual = lab_norm = lab_bare = lab_cue = 0
    med_lines = med_str = med_route = med_freq = med_form = med_cue = med_full = 0
    img_lines = img_mod = img_cue = img_paren = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # --- laboratory layout ---
        n_unit = _group(config.laboratory, "numeric_with_unit", line)
        ref = _group(config.laboratory, "reference_range", line)
        qual = _group(config.laboratory, "qualitative_result", line)
        norm = _group(config.laboratory, "normality_phrase", line)
        bare = _group(config.laboratory, "bare_numeric_result", line)
        lab_cue_hit = _group(config.laboratory, "lab_section_cue", line)
        lab_num += n_unit
        lab_ref += ref
        lab_qual += qual
        lab_norm += norm
        lab_bare += bare
        lab_cue += lab_cue_hit
        if n_unit or ref or qual or bare:
            lab_lines += 1

        # --- medication-list pattern ---
        strength = _group(config.medication, "strength", line)
        route = _group(config.medication, "route", line)
        freq = _group(config.medication, "frequency", line)
        form = _group(config.medication, "dose_form", line)
        med_cue_hit = _group(config.medication, "med_section_cue", line)
        med_str += strength
        med_route += route
        med_freq += freq
        med_form += form
        med_cue += med_cue_hit
        if strength and route and freq:
            med_full += 1
        if strength or form or (route and freq):
            med_lines += 1

        # --- imaging / report style ---
        modality = _group(config.imaging, "modality", line)
        img_cue_hit = _group(config.imaging, "imaging_section_cue", line)
        img_mod += modality
        img_cue += img_cue_hit
        if modality:
            img_lines += 1
            # the observed "- <finding> (<modality>)" style
            if _group(config.imaging, "trailing_modality_parenthetical", line):
                img_paren += 1

    return LayoutStats(
        lab_lines=lab_lines, lab_numeric_with_unit=lab_num, lab_reference_range=lab_ref,
        lab_qualitative=lab_qual, lab_normality_phrase=lab_norm, lab_bare_numeric=lab_bare,
        lab_section_cue_lines=lab_cue,
        med_lines=med_lines, med_strength=med_str, med_route=med_route,
        med_frequency=med_freq, med_dose_form=med_form, med_section_cue_lines=med_cue,
        med_full_pattern=med_full,
        imaging_lines=img_lines, imaging_modality=img_mod,
        imaging_section_cue_lines=img_cue, imaging_trailing_parenthetical=img_paren)


__all__ = ["analyze_layouts"]
