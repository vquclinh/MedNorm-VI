"""Phase 2A corpus-analysis CLI (deterministic, offline, read-only).

    python -m mednorm_vi.corpus_analysis.cli \\
        --input-dir data/organizer_test/input \\
        --output-dir reports/corpus_analysis

Reads the organizer's PUBLIC INPUT documents read-only and writes a deterministic
JSON + Markdown report. Descriptive statistics only — no prediction, no entity
extraction, no linking, no LLM/ML, and no change to L1 preprocessing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loader import load_corpus
from .pipeline import Phase2AConfig, run_corpus_analysis
from .reporting import determinism_hash, to_json, to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MedNorm-VI Phase 2A — public corpus analysis (descriptive only)")
    parser.add_argument("--input-dir", default="data/organizer_test/input",
                        help="directory of public input .txt documents (read-only)")
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--output-dir", default=None,
                        help="write report.json + report.md here (git-ignored)")
    parser.add_argument("--corpus-id", default="public_input")
    parser.add_argument("--analysis-config", default="configs/corpus_analysis/analysis_v1.yaml")
    parser.add_argument("--router-config", default="configs/case_router/base.yaml")
    parser.add_argument("--l1-config", default="configs/document_intelligence/base.yaml")
    parser.add_argument("--top-headings", type=int, default=25)
    args = parser.parse_args(argv)

    documents = load_corpus(args.input_dir, pattern=args.pattern)
    if not documents:
        print(f"error: no documents matched {args.pattern!r} under {args.input_dir}",
              file=sys.stderr)
        return 2

    config = Phase2AConfig.load(args.analysis_config, args.router_config, args.l1_config)
    analysis = run_corpus_analysis(documents, config, corpus_id=args.corpus_id)

    dist = analysis.distributions
    print("MedNorm-VI Phase 2A — Public Corpus Analysis (descriptive only)")
    print(f"input dir           : {args.input_dir}")
    print(f"documents           : {analysis.n_documents}")
    print(f"corpus sha256       : {analysis.corpus_sha256}")
    print(f"L1 validation OK    : {sum(1 for d in analysis.documents if d.l1_valid)}"
          f"/{analysis.n_documents}")
    print(f"characters (total)  : {dist['characters'].total}")
    print(f"characters (median) : {dist['characters'].median}")
    print(f"lines (total)       : {dist['lines'].total}")
    print(f"sections (total)    : {dist['sections'].total}")
    print(f"list items (total)  : {dist['list_items'].total}")
    print(f"key-value rows      : {dist['key_value_rows'].total}")
    print(f"routable units      : {dist['routable_units'].total}")
    print("units by kind       : " + ", ".join(
        f"{k}={v}" for k, v in analysis.unit_kind_frequencies))
    print("cases (C1-C5)       : " + ", ".join(
        f"{k}={v}" for k, v in analysis.case_frequencies))
    print("profiles            : " + ", ".join(
        f"{k}={v}" for k, v in analysis.profile_frequencies))
    print(f"docs w/ lab layout  : {analysis.docs_with['lab_layout']}")
    print(f"docs w/ med layout  : {analysis.docs_with['medication_layout']}")
    print(f"docs w/ imaging     : {analysis.docs_with['imaging_layout']}")
    print(f"determinism hash    : {determinism_hash(analysis)}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(to_json(analysis), encoding="utf-8")
        (out / "report.md").write_text(
            to_markdown(analysis, top_headings=args.top_headings), encoding="utf-8")
        print(f"json report         : {(out / 'report.json').resolve()}")
        print(f"markdown report     : {(out / 'report.md').resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
