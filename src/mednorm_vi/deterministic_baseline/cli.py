"""Phase 1B inspection CLI (deterministic proposals; NOT organizer output).

    python -m mednorm_vi.deterministic_baseline.cli \\
      --input tests/fixtures/phase1b/synthetic_medical_document.txt \\
      --l1-config configs/document_intelligence/base.yaml \\
      --router-config configs/case_router/base.yaml \\
      --medication-config configs/medication/grammar_v1.yaml \\
      --laboratory-config configs/laboratory/parser_v1.yaml \\
      --output /tmp/mednorm-phase1b.json [--inspect]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..document_intelligence import analyze_document
from ..document_intelligence.builder import load_config as load_l1_config
from ..document_intelligence.io import DocumentLoadError, load_document
from .models import Phase1BConfig, Phase1BResult
from .pipeline import run_phase1b
from .serialization import count_summary, determinism_hash, to_json


def _print_inspection(result: Phase1BResult) -> None:
    print("\n-- routing --")
    for r in result.routings:
        tags = ", ".join(f"{c.case}:{c.score:.2f}" for c in r.cases) or "(none)"
        print(f"  {r.text[:44]!r:48} -> [{tags}]")
    print("\n-- proposals --")
    for p in result.proposals:
        print(f"  {p.proposal_id} [{p.start},{p.end}) {p.proposed_types[0]} "
              f"{p.text!r} score={p.local_score} rule={p.matched_rule}")
    print("\n-- has_result relations --")
    by_id = {p.proposal_id: p.text for p in result.proposals}
    for rel in result.relations:
        print(f"  {by_id.get(rel.source_proposal_id)!r} -> "
              f"{by_id.get(rel.target_proposal_id)!r} cost={rel.pairing_cost}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedNorm-VI Phase 1B deterministic baseline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--l1-config", default="configs/document_intelligence/base.yaml")
    parser.add_argument("--router-config", default="configs/case_router/base.yaml")
    parser.add_argument("--medication-config", default="configs/medication/grammar_v1.yaml")
    parser.add_argument("--laboratory-config", default="configs/laboratory/parser_v1.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args(argv)

    try:
        l1_config, l1_lexicon = load_l1_config(args.l1_config)
        # validate input is loadable before analysis
        load_document(args.input, encoding=l1_config.encoding, bom_policy=l1_config.bom_policy)
        graph = analyze_document(args.input, config=l1_config, lexicon=l1_lexicon)
        config = Phase1BConfig.load(args.router_config, args.medication_config,
                                    args.laboratory_config)
    except (DocumentLoadError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = run_phase1b(graph, config)
    counts = count_summary(result)

    print("MedNorm-VI Phase 1B — Deterministic Medical Specialists")
    print(f"input path                 : {args.input}")
    print(f"L1 graph validation        : {'OK' if result.l1_valid else 'FAILED'}")
    print(f"canonical routable units   : {counts['routable_nodes']}")
    for kind in ("list_item", "table_row", "sentence", "line"):
        print(f"  unit[{kind:<10}]      : {counts.get(f'unit_{kind}', 0)}")
    for i in range(1, 8):
        label = " (derived)" if i in (6, 7) else ""
        print(f"  C{i}{label:<10}            : {counts[f'C{i}']}")
    print(f"multi-route units          : {counts['multi_route']}")
    print(f"specialist exec[medication]: {counts['exec_medication']}")
    print(f"specialist exec[laboratory]: {counts['exec_laboratory']}")
    print(f"duplicate-exec prevented   : {counts['duplicate_specialist_prevention']}")
    print(f"medication parses          : {counts['medication_parses']}")
    print(f"medication boundary props  : {counts['medication_proposals']}")
    print(f"laboratory parses          : {counts['laboratory_parses']}")
    print(f"test-name proposals        : {counts['test_name_proposals']}")
    print(f"test-result proposals      : {counts['test_result_proposals']}")
    print(f"logical test-result pairs  : {counts['logical_test_result_pairs']}")
    print(f"relation alternatives      : {counts['relation_alternatives']}")
    print(f"duplicate-id groups        : {counts['duplicate_identity_groups']}")
    print(f"overlap pairs              : {counts['overlap_pairs']}")
    print(f"proposal validation        : {'OK' if result.proposals_valid else 'FAILED'}")
    print(f"warnings                   : {len(set(result.warnings))}")
    print(f"determinism hash           : {determinism_hash(result)}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_json(result), encoding="utf-8")
        print(f"output path           : {out.resolve()}")

    if args.inspect:
        _print_inspection(result)

    if not result.proposals_valid:
        for issue in result.issues:
            if issue.severity.value == "error":
                print(f"  ERROR {issue.code}: {issue.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
