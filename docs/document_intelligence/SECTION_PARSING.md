# L1 Section Parsing

A section is **evidence, not an absolute rule**. L1 records the header span,
proposed category, confidence, matched rule, and a discourse **prior** (e.g.
`isHistorical`, `isFamily`) with a strength — it **never assigns final
assertions**. Later layers weigh these priors, and local sentences may override
them.

## Versioned, extensible lexicon

Aliases live in `configs/document_intelligence/section_lexicon_v1.yaml`, keyed by
category, with per-alias metadata (surface form, language, accent status,
abbreviation status) and per-category metadata (semantic group, prior label,
prior strength, positive/negative examples, version). This is **representative,
not exhaustive** — the format is designed to grow. Do not hard-code a final
medical section lexicon in Python.

Supported categories (extensible): `medical_history`, `family_history`,
`pre_admission_medications`, `home_medications`, `current_medications`,
`laboratory`, `diagnosis`, `current_examination`, `symptoms`, `treatment`,
`admission_information`, `discharge_information`, plus `unknown` (preamble/other).

## Matching signals

A short, header-like line is matched against a casefold + accent-stripped +
whitespace-collapsed view of its **key** (text before the first `:`):

- **exact alias** → high confidence (accepted directly);
- **fuzzy alias** (`SequenceMatcher` ratio ≥ `fuzzy_threshold`) → accepted **only
  with structural evidence**.

Structural evidence includes: colon-terminated heading (`Chẩn đoán:`), a short
isolated line surrounded by blank lines, or an upper-case heading. Fuzzy hits
with lower confidence emit a `weak_section_header_confidence` warning.

## Why fuzzy matching cannot be an absolute rule

Ordinary clinical sentences often *mention* section words (“Cần **chẩn đoán**
phân biệt…”, “bệnh nhân có **tiền sử** hút thuốc…”). Requiring the whole short
**key** to match an alias **and** demanding structural evidence for fuzzy hits
keeps these long inline sentences from being promoted to headers. `max_header_chars`
bounds the key length so long lines can never be headers.

## Nesting

Subsections are detected by **indentation**: a more-indented header nests under
the nearest less-indented section (a `parent_id` link). Top-level sections tile
the document; subsections are contained within their parent.

## What L1 stores per section

`category`, `confidence`, `matched_rule` (`exact_alias:…` / `fuzzy_alias:…`),
`header_start`/`header_end`, `prior_label`, `prior_strength`, `parent_id`, and
absolute `start`/`end`. No entity or assertion is ever emitted here.
