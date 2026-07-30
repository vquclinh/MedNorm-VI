"""Phase 1C-A foundation CLI (deterministic, offline).

    python -m mednorm_vi.phase1c_foundation.cli doctor
    python -m mednorm_vi.phase1c_foundation.cli position-forensics --input IN.txt --output OUT.json
    python -m mednorm_vi.phase1c_foundation.cli validate-resource-manifest --manifest M.yaml [--ner]
    python -m mednorm_vi.phase1c_foundation.cli inspect-rxnorm-snapshot --dir DIR
    python -m mednorm_vi.phase1c_foundation.cli compare-rxnorm-snapshots --left A --right B
    python -m mednorm_vi.phase1c_foundation.cli inspect-icd-snapshot --table T.csv --columns C.yaml
    python -m mednorm_vi.phase1c_foundation.cli compare-icd-snapshots --left A.csv --right B.csv \
        --columns C.yaml
    python -m mednorm_vi.phase1c_foundation.cli resolve-phase1b-debug-output --input DOC.txt

All commands operate on explicit local paths. No network access; no downloads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .doctor import DoctorPaths, build_report, render_report


def _cmd_doctor(args: argparse.Namespace) -> int:
    paths = DoctorPaths(
        organizer_dir=Path(args.organizer_dir), position_config=Path(args.position_config),
        l4_config=Path(args.l4_config), manifests_dir=Path(args.manifests_dir),
        rxnorm_dir=Path(args.rxnorm_dir), icd_dir=Path(args.icd_dir))
    report = build_report(paths)
    if args.json:
        payload = {"report": report.data,
                   "missing_local_resources": list(report.missing_local_resources),
                   "determinism_hash": report.determinism_hash}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(render_report(report), end="")
    return 0


def _cmd_position_forensics(args: argparse.Namespace) -> int:
    from ..position import Observation, analyze, load_position_registry

    registry = load_position_registry(args.position_config)
    text = Path(args.input).read_text(encoding="utf-8")
    raw = json.loads(Path(args.output).read_text(encoding="utf-8"))
    entities = raw.get("entities", raw) if isinstance(raw, dict) else raw
    observations: list[Observation] = []
    for e in entities:
        pos = e.get("position") or [e.get("start"), e.get("end")]
        observations.append(Observation(str(e["text"]), int(pos[0]), int(pos[1])))
    report = analyze(registry, text, observations)
    out: dict[str, Any] = {
        "n_observations": report.n_observations, "n_locatable": report.n_locatable,
        "best_policy": report.best_policy_id, "byte_vs_codepoint": report.byte_vs_codepoint,
        "line_ending_evidence": report.line_ending_evidence,
        "interval_evidence": report.interval_evidence, "max_offset": report.max_offset,
        "policy_stats": [
            {"policy_id": s.policy_id, "matched": s.matched, "exact": s.exact,
             "near": s.near, "systematic_start_delta": s.systematic_start_delta,
             "max_abs_start_delta": s.max_abs_start_delta}
            for s in report.policy_stats],
        "cumulative_line_shift": [list(t) for t in report.cumulative_line_shift],
        "notes": list(report.notes),
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _cmd_validate_resource_manifest(args: argparse.Namespace) -> int:
    if args.ner:
        from ..resources.ner import load_ner_manifest, validate_ner_manifest

        m = load_ner_manifest(args.manifest)
        result = validate_ner_manifest(m)
        name = m.dataset_id
    else:
        from ..resources import load_manifest, validate_manifest

        rm = load_manifest(args.manifest)
        result = validate_manifest(rm)
        name = rm.resource_id
    print(f"resource manifest: {name} ({args.manifest})")
    for issue in result.issues:
        print(f"  {issue.severity.value}: {issue.code}: {issue.message}")
    if result.ok:
        print(f"OK ({len(result.warnings)} warning(s))")
        return 0
    print(f"FAILED: {len(result.errors)} error(s)", file=sys.stderr)
    return 1


def _cmd_inspect_rxnorm(args: argparse.Namespace) -> int:
    from ..kb.rxnorm import build_snapshot, snapshot_stats, validate_snapshot

    snap = build_snapshot(args.dir)
    stats = snapshot_stats(snap)
    vr = validate_snapshot(snap)
    print(f"RxNorm snapshot: {snap.snapshot_id} ({args.dir})")
    for k in sorted(stats):
        print(f"  {k:22}: {stats[k]}")
    print(f"  validation            : {'OK' if vr.ok else 'FAILED'} "
          f"({len(vr.warnings)} warning(s))")
    return 0 if vr.ok else 1


def _cmd_compare_rxnorm(args: argparse.Namespace) -> int:
    from ..kb.rxnorm import build_snapshot, diff_snapshots

    a, b = build_snapshot(args.left), build_snapshot(args.right)
    d = diff_snapshots(a, b)
    print(f"RxNorm compare: {a.snapshot_id} -> {b.snapshot_id}")
    print(f"  concepts               : {d.n_left} -> {d.n_right}")
    print(f"  only in left           : {len(d.only_left)} {list(d.only_left)}")
    print(f"  only in right          : {len(d.only_right)} {list(d.only_right)}")
    print(f"  label changed          : {len(d.label_changed)}")
    print(f"  tty changed            : {len(d.tty_changed)}")
    print(f"  suppression changed    : {len(d.suppression_changed)}")
    print(f"  remapped (left->right) : {[list(x) for x in d.remapped]}")
    return 0


def _cmd_inspect_icd(args: argparse.Namespace) -> int:
    from ..kb.icd10 import ColumnMap, build_snapshot, snapshot_stats, validate_snapshot

    cm = ColumnMap.from_mapping(yaml.safe_load(Path(args.columns).read_text(encoding="utf-8")))
    snap = build_snapshot(args.table, cm, source=args.source, version=args.version)
    stats = snapshot_stats(snap)
    vr = validate_snapshot(snap)
    print(f"ICD-10 snapshot: {snap.snapshot_id} ({args.table})")
    for k in sorted(stats):
        print(f"  {k:16}: {stats[k]}")
    print(f"  validation      : {'OK' if vr.ok else 'FAILED'} ({len(vr.warnings)} warning(s))")
    return 0 if vr.ok else 1


def _cmd_compare_icd(args: argparse.Namespace) -> int:
    from ..kb.icd10 import ColumnMap, build_snapshot, diff_snapshots

    cm = ColumnMap.from_mapping(yaml.safe_load(Path(args.columns).read_text(encoding="utf-8")))
    a = build_snapshot(args.left, cm, source="left", version="left")
    b = build_snapshot(args.right, cm, source="right", version="right")
    d = diff_snapshots(a, b)
    print(f"ICD-10 compare: {a.snapshot_id} -> {b.snapshot_id}")
    print(f"  concepts           : {d.n_left} -> {d.n_right}")
    print(f"  only in left       : {list(d.only_left)}")
    print(f"  only in right      : {list(d.only_right)}")
    print(f"  label changed      : {[x[0] for x in d.label_changed]}")
    print(f"  parent changed     : {[x[0] for x in d.parent_changed]}")
    print(f"  alias changed      : {[x[0] for x in d.alias_changed]}")
    print(f"  format changed     : {[x[0] for x in d.format_changed]}")
    print(f"  specificity changed: {[x[0] for x in d.specificity_changed]}")
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    """L1 -> L2/L3 -> unified lattice -> **canonical L4**, as a debug view.

    Audit 0055 migrated this command off the retired Phase-1C-A resolver. It now
    calls exactly the functions the canonical runner calls —
    ``lattice.builder.build_span_lattice`` then
    ``resolution.canonical.resolve_lattice_to_hypotheses`` — and stops at L4. It is
    deliberately **not** a second inference runner: there is no L5-L9 here, no
    linking, no packaging and no output.zip. For a full run use
    ``python -m mednorm_vi.inference.cli run``.
    """
    from ..deterministic_baseline import Phase1BConfig, run_phase1b
    from ..document_intelligence import analyze_document
    from ..document_intelligence.builder import load_config as load_l1_config
    from ..lattice.builder import build_span_lattice
    from ..resolution import (
        CANONICAL_L4_VERSION,
        determinism_hash,
        load_resolver_v1_config,
        resolve_lattice_to_hypotheses,
        to_json,
        validate_result,
    )

    l1_config, l1_lexicon = load_l1_config(args.l1_config)
    graph = analyze_document(args.input, config=l1_config, lexicon=l1_lexicon)
    cfg = Phase1BConfig.load(args.router_config, args.medication_config, args.laboratory_config)
    phase1b = run_phase1b(graph, cfg)
    l4_config = load_resolver_v1_config(args.l4_config)
    lattice = build_span_lattice(
        graph.document_id, graph.original_text, routings=phase1b.routings,
        specialist_proposals=phase1b.proposals, expert_spans=(),
        relations=phase1b.relations)
    result = resolve_lattice_to_hypotheses(
        lattice, l4_config, relations=phase1b.relations)
    valid = validate_result(result, graph.original_text)
    print("MedNorm-VI Phase 1C — canonical L4 debug view (deterministic; not finals)")
    print(f"input path      : {args.input}")
    print("l4 entry point  : resolution.canonical.resolve_lattice_to_hypotheses")
    print(f"l4 version      : {CANONICAL_L4_VERSION}")
    print(f"l4 config       : {args.l4_config}")
    print(f"l4 config sha256: {l4_config.config_sha256}")
    print(f"boundary policy : {dict(sorted(l4_config.boundary.group_preference.items()))}")
    print(f"abstain on tie  : {l4_config.overlap.abstain_on_conflict}")
    print(f"lattice nodes   : {len(lattice.proposals)}")
    print(f"hypotheses      : {len(result.hypotheses)}")
    print(f"  accepted      : {len(result.accepted())}")
    print(f"  rejected      : {len(result.rejected())}")
    print(f"  unresolved    : {len(result.unresolved())}")
    print(f"validation      : {'OK' if valid.ok else 'FAILED'}")
    print(f"determinism hash: {determinism_hash(result)}")
    if args.output:
        Path(args.output).write_text(to_json(result), encoding="utf-8")
        print(f"output path     : {Path(args.output).resolve()}")
    return 0 if valid.ok else 1


# The one canonical L4 configuration. Named once so no subcommand can drift.
CANONICAL_L4_CONFIG = "configs/resolution/boundary_type_resolver_v1.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedNorm-VI Phase 1C-A foundation")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="deterministic readiness report")
    d.add_argument("--organizer-dir", default="configs/organizer")
    d.add_argument("--position-config", default="configs/organizer/position_policies_v1.yaml")
    d.add_argument(
        "--l4-config", default=CANONICAL_L4_CONFIG,
        help="canonical L4 Boundary & Type Resolver config (spec section 7)")
    d.add_argument("--resolver-config", default=None, help=argparse.SUPPRESS)
    d.add_argument("--manifests-dir", default="data/manifests")
    d.add_argument("--rxnorm-dir", default="data/external/rxnorm")
    d.add_argument("--icd-dir", default="data/external/icd10_vi")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=_cmd_doctor)

    pf = sub.add_parser("position-forensics", help="compare position policies vs sample output")
    pf.add_argument("--input", required=True, help="sample input text file")
    pf.add_argument("--output", required=True, help="sample output JSON (entities with position)")
    pf.add_argument("--position-config", dest="position_config",
                    default="configs/organizer/position_policies_v1.yaml")
    pf.set_defaults(func=_cmd_position_forensics)

    vm = sub.add_parser("validate-resource-manifest", help="validate a resource/NER manifest")
    vm.add_argument("--manifest", required=True)
    vm.add_argument("--ner", action="store_true", help="treat as an NER dataset manifest")
    vm.set_defaults(func=_cmd_validate_resource_manifest)

    ir = sub.add_parser("inspect-rxnorm-snapshot",
                        help="ingest + summarize a local RxNorm snapshot")
    ir.add_argument("--dir", required=True)
    ir.set_defaults(func=_cmd_inspect_rxnorm)

    cr = sub.add_parser("compare-rxnorm-snapshots", help="diff two local RxNorm snapshots")
    cr.add_argument("--left", required=True)
    cr.add_argument("--right", required=True)
    cr.set_defaults(func=_cmd_compare_rxnorm)

    ii = sub.add_parser("inspect-icd-snapshot", help="ingest + summarize a local ICD-10 table")
    ii.add_argument("--table", required=True)
    ii.add_argument("--columns", required=True)
    ii.add_argument("--source", default="unknown")
    ii.add_argument("--version", default="unknown")
    ii.set_defaults(func=_cmd_inspect_icd)

    ci = sub.add_parser("compare-icd-snapshots", help="diff two local ICD-10 tables")
    ci.add_argument("--left", required=True)
    ci.add_argument("--right", required=True)
    ci.add_argument("--columns", required=True)
    ci.set_defaults(func=_cmd_compare_icd)

    rs = sub.add_parser(
        "resolve-phase1b-debug-output",
        help=("run L1->L2/L3->lattice->CANONICAL L4 on a local document "
              "(deterministic debug view; not a second inference runner)"))
    rs.add_argument("--input", required=True)
    rs.add_argument("--l1-config", default="configs/document_intelligence/base.yaml")
    rs.add_argument("--router-config", default="configs/case_router/base.yaml")
    rs.add_argument("--medication-config", default="configs/medication/grammar_v1.yaml")
    rs.add_argument("--laboratory-config", default="configs/laboratory/parser_v1.yaml")
    rs.add_argument(
        "--l4-config", default=CANONICAL_L4_CONFIG,
        help="canonical L4 Boundary & Type Resolver config (spec section 7)")
    rs.add_argument("--resolver-config", default=None, help=argparse.SUPPRESS)
    rs.add_argument("--output", default=None)
    rs.set_defaults(func=_cmd_resolve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # The retired Phase-1C-A resolver had its own config file. Audit 0055 deleted
    # both. A caller still passing the old flag gets an error naming the
    # replacement, never a silent fall-through to a different resolver.
    if getattr(args, "resolver_config", None):
        parser.error(
            "--resolver-config was removed with the Phase-1C-A resolver "
            f"(Audit 0055). Use --l4-config (default {CANONICAL_L4_CONFIG}); its "
            "boundary.group_preference and overlap.abstain_on_conflict keys carry "
            "the migrated medication_boundary / test_result_boundary / "
            "abstain_on_conflict policies.")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
