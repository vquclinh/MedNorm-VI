"""L1 Document Intelligence CLI (structural inspection, not prediction output).

    python -m mednorm_vi.document_intelligence.cli \\
      --input path/to/document.txt \\
      --config configs/document_intelligence/base.yaml \\
      --output /tmp/document_graph.json [--inspect]

Prints a concise structural summary and writes the deterministic debug graph.
Never prints chain-of-thought or hidden reasoning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import build_document_graph, default_config, load_config
from .io import DocumentLoadError, load_document
from .models import DocumentGraph, NodeKind
from .serialization import determinism_hash, to_json
from .validation import validate_graph

DEFAULT_CONFIG = "configs/document_intelligence/base.yaml"


def _summary(graph: DocumentGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind in NodeKind:
        counts[kind.value] = len(graph.nodes_of_kind(kind))
    return counts


def _print_inspection(graph: DocumentGraph) -> None:
    text = graph.original_text
    print("\n-- section hierarchy --")
    for sec in graph.nodes_of_kind(NodeKind.SECTION):
        depth = 0
        pid = sec.parent_id
        index = graph.by_id()
        while pid is not None and pid in index and index[pid].kind is NodeKind.SECTION:
            depth += 1
            pid = index[pid].parent_id
        header = text[sec.header_start : sec.header_end] if sec.header_start is not None else ""
        label = header.replace("\n", " ") or "(preamble)"
        print(f"  {'  ' * depth}[{sec.category}] {label!r} conf={sec.confidence:.2f}")
    print("\n-- lines / sentences / lists / rows --")
    for line in graph.nodes_of_kind(NodeKind.LINE):
        content = text[line.start : line.end].replace("\n", "⏎")
        print(f"  line {line.node_id}: {content!r}")
    print("\n-- normalized-to-original examples (match view) --")
    view = graph.normalized.get("match")
    if view is not None and view.text:
        sample = view.text[:40]
        os, oe = view.alignment.normalized_span_to_original(0, min(len(view.text), 40))
        print(f"  norm[0:{len(sample)}]={sample!r} -> original[{os}:{oe}]="
              f"{text[os:oe]!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedNorm-VI L1 Document Intelligence")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=None)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args(argv)

    try:
        if Path(args.config).is_file():
            config, lexicon = load_config(args.config)
        else:
            config, lexicon = default_config()
        document = load_document(args.input, encoding=config.encoding,
                                 bom_policy=config.bom_policy)
    except DocumentLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    graph = build_document_graph(document, config, lexicon)
    result = validate_graph(graph)
    counts = _summary(graph)

    print("MedNorm-VI L1 Document Intelligence")
    print(f"input path        : {document.source.source_path}")
    print(f"characters        : {document.source.char_length}")
    print(f"bytes             : {document.source.byte_length}")
    print(f"newline style     : {document.source.newline_style}")
    print(f"has BOM           : {document.source.has_bom}")
    print(f"lines             : {counts['line']}")
    print(f"sections          : {counts['section']}")
    print(f"paragraphs        : {counts['paragraph']}")
    print(f"sentences         : {counts['sentence']}")
    print(f"list items        : {counts['list_item']}")
    print(f"table-like rows   : {counts['table_row']}")
    print(f"tokens            : {counts['token']}")
    print(f"delimiters        : {counts['delimiter']}")
    print(f"normalization     : {', '.join(sorted(graph.normalized))}")
    print(f"graph validation  : {'OK' if result.ok else 'FAILED'} "
          f"({len(result.errors)} errors, {len(result.warnings)} warnings)")
    print(f"structural warns  : {len(graph.warnings)}")
    print(f"determinism hash  : {determinism_hash(graph)}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_json(graph), encoding="utf-8")
        print(f"output path       : {out.resolve()}")

    if args.inspect:
        _print_inspection(graph)

    if not result.ok:
        for issue in result.errors:
            print(f"  ERROR {issue.code}: {issue.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
