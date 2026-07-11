# L1 Document Intelligence — Overview

L1 converts an exact UTF-8 clinical document into a deterministic
`DocumentGraph`: an immutable `original_text`, reversible normalized views, and a
tree of structural nodes (sections, paragraphs, lines, sentences, list items,
table-like rows, delimiters, tokens) — each carrying **absolute** coordinates
into `original_text`.

> **L1 detects STRUCTURE, not medical entities.** It may label a line
> "list-like", "table-like", or a "possible section header", but it never claims
> a `THUỐC`/`CHẨN_ĐOÁN`/… entity was found. NER, medication/lab extraction,
> assertions, and ICD/RxNorm linking belong to later layers.

## The core invariant

For every structural node,

```
original_text[node.start:node.end] == node.text(original_text)
```

with `[start, end)` end-exclusive over **Unicode code points** (Python slicing).
This is verified for every node by `validation.validate_graph`.

## Why `original_text` is immutable

`original_text` is the **sole** source of the organizer-facing `text`/`position`.
If L1 trimmed whitespace, re-cased, rewrote punctuation, or normalized newlines,
every downstream offset would silently shift and the mandatory invariant
`original_text[start:end] == entity.text` would break. So L1 never mutates it:
normalization happens only on **copies** (see `OFFSET_ALIGNMENT.md`).

## Pipeline

```
bytes ──io.load_document──▶ LoadedDocument (exact text + source metadata)
         │
         ├─ normalization.build_views ─▶ normalized views (match, accent) + alignment
         ├─ lines.split_lines / paragraphs
         ├─ sections.detect_sections (versioned lexicon + structural evidence)
         ├─ sentences / lists / table_rows / tokens
         └─ builder.build_document_graph ─▶ DocumentGraph
                                             │
                                             ├─ validation.validate_graph
                                             └─ serialization (debug JSON + determinism hash)
```

## Entry points

```python
from mednorm_vi.document_intelligence import analyze_text, analyze_document
graph = analyze_text("Chẩn đoán: viêm phổi.\n...")
graph = analyze_document("note.txt")
```

CLI:

```bash
python -m mednorm_vi.document_intelligence.cli \
  --input note.txt --config configs/document_intelligence/base.yaml \
  --output /tmp/graph.json --inspect
```

## Determinism

Node ids (`sec-0001`, `line-0007`, `tok-000012`, …) and node ordering are
deterministic; building the same document twice yields byte-identical debug JSON
(`serialization.determinism_hash`).

## Handoff to later layers

L1 provides the structural substrate the rest of the architecture reads:
`section` priors + provenance (evidence, not assertions), `line`/`sentence`
spans, `list_item`/`list_marker` separation, `table_like`/`key_value_like` rows,
and a canonical token view that model tokenizers must map back through. See
`SECTION_PARSING.md`, `SEGMENTATION.md`, `OFFSET_ALIGNMENT.md`, and
`KNOWN_LIMITATIONS.md`.
