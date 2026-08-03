#!/usr/bin/env python3
"""OntoFusion linking (milestone 0082): pretrained-only, no new deployed model.

A second linker beside GraphCENT 0080. GraphCENT is not modified, so the ablation stays one
variable at a time:

    0080 = E3 entities        + GraphCENT
    0081 = contextual entities + GraphCENT
    0082 = contextual entities + OntoFusion

Consumes a 0081 entity directory (`entities_<variant>`) and emits organizer JSON with
candidates attached, in three confidence variants from ONE inference pass.

Subcommands:
    budget    re-certify the deployed stack from the canonical GraphCENT manifest
    smoke     a few documents end to end, with per-mention diagnostics
    run       every document; writes allnull / high / broad from one pass
    package   dotted derivation + strict validation + zip + SHA256 (reuses 0080's)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.ontofusion.pipeline import VARIANTS, ConfidencePolicy  # noqa: E402
from mednorm_vi.ontofusion.retrieval import RetrievalPolicy  # noqa: E402
from mednorm_vi.ontofusion.runtime import (  # noqa: E402
    OntoFusionConfig,
    diagnostics,
    log,
    run,
    write_variants,
)
from mednorm_vi.ontofusion.sparse import SparseSettings  # noqa: E402

DEFAULT_CONFIG = REPO / "configs/pipeline/ontofusion_0082.yaml"

#: 0082 loads Qwen and the retrievers strictly one at a time, so the floor is GraphCENT's.
MIN_VRAM_FOR_QWEN_GIB = 18.0


def graphcent_cli() -> Any:
    """The 0080 CLI as a module: host policy, KB loading and packaging are reused as-is."""
    spec = importlib.util.spec_from_file_location(
        "graphcent_cli", REPO / "scripts/graphcent_0080.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def ontofusion_config(config: dict[str, Any]) -> OntoFusionConfig:
    retrieval = config.get("retrieval") or {}
    sparse = config.get("sparse") or {}
    confidence = config.get("confidence") or {}
    return OntoFusionConfig(
        retrieval=RetrievalPolicy(
            sparse_top_k=int(retrieval.get("sparse_top_k", 4)),
            dense_top_k=int(retrieval.get("dense_top_k", 4)),
            candidate_cap=int(retrieval.get("candidate_cap", 10)),
            sparse_views=tuple(retrieval.get("sparse_views") or ("raw", "canonical_vi")),
            dense_views=tuple(
                retrieval.get("dense_views") or ("raw", "canonical_vi", "canonical_en")
            ),
        ),
        sparse=SparseSettings(
            ngram_n=int(sparse.get("ngram_n", 3)),
            strip_accents=bool(sparse.get("strip_accents", True)),
            shortlist=int(sparse.get("shortlist", 50)),
        ),
        confidence=ConfidencePolicy(
            min_retrievers=int(confidence.get("min_retrievers", 2)),
            min_views=int(confidence.get("min_views", 2)),
            exact_alias_is_sufficient=bool(
                confidence.get("exact_alias_is_sufficient", True)
            ),
        ),
        use_reformulation=bool(config.get("use_reformulation", True)),
        allow_multi_code=bool(config.get("allow_multi_code", False)),
    )


def read_documents(input_dir: Path) -> dict[str, str]:
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(input_dir.glob("*"))
        if path.is_file()
    }


def read_entities(entity_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(entity_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = payload if isinstance(payload, list) else []
    return out


def canonical_manifest(model_root: Path) -> dict[str, Any]:
    from mednorm_vi.graphcent.acquisition import MANIFEST_NAME

    path = Path(model_root) / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(
            f"canonical model manifest not found: {path}. Run "
            "scripts/graphcent_0080.py download-models first; 0082 never certifies its own."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def cmd_budget(args: argparse.Namespace) -> int:
    """0082 adds no deployed model. Prove it from the canonical manifest, not from here."""
    from mednorm_vi.graphcent.models import PARAMETER_CAP

    manifest = canonical_manifest(Path(args.model_root))
    total = int(manifest.get("total_deployed_parameters", 0))
    log("0082 is pretrained-only: nothing trained, nothing fine-tuned, no model added.")
    log("Sparse character retrieval contributes no parameters.")
    for model in manifest.get("deployed") or ():
        log(f"  {model['name']:48s} {model['parameter_count']:>15,}  {model['revision']}")
    log(f"  {'TOTAL':48s} {total:>15,}  (cap {PARAMETER_CAP:,})")
    if total <= 0 or total >= PARAMETER_CAP:
        print("BLOCKED: canonical manifest is not under the cap", file=sys.stderr)
        return 3
    if args.expect_total and total != args.expect_total:
        print(
            f"BLOCKED: canonical total {total:,} != expected {args.expect_total:,}. "
            "The deployed stack changed; re-certify before trusting this run.",
            file=sys.stderr,
        )
        return 3
    return 0


def _execute(args: argparse.Namespace, *, smoke: bool) -> int:
    graphcent = graphcent_cli()
    graphcent.require_host(
        "smoke" if smoke else "run",
        allow_download=args.allow_download,
        needs=MIN_VRAM_FOR_QWEN_GIB,
    )
    from mednorm_vi.graphcent.runtime import QwenDisambiguator

    profile = load_config(args.config)
    settings = ontofusion_config(profile)

    graph_profile = graphcent.load_config(Path(args.graphcent_config))
    specs = graphcent.pin_specs(
        graphcent.resolve_specs(graph_profile), Path(args.model_root)
    )
    graph_config = graphcent._runtime_config(graph_profile, args)
    icd_index, rxnorm_index, structured, documents = graphcent._load_kb(graph_profile)
    if not documents:
        print("BLOCKED: no governed KB documents; restore the indices first", file=sys.stderr)
        return 2

    notes = read_documents(Path(args.input_dir))
    entities = read_entities(Path(args.entities))
    if not notes or not entities:
        print("BLOCKED: need both organizer input and a 0081 entity directory",
              file=sys.stderr)
        return 2
    wanted = [d.strip() for d in (args.documents or "").split(",") if d.strip()]
    if wanted:
        notes = {k: v for k, v in notes.items() if k in wanted}
        entities = {k: v for k, v in entities.items() if k in wanted}

    manifest = canonical_manifest(Path(args.model_root))
    qwen_revision = next(
        (
            str(m.get("revision", ""))
            for m in manifest.get("deployed") or ()
            if "Qwen" in str(m.get("name", ""))
        ),
        "",
    )
    qwen = QwenDisambiguator(
        Path(args.qwen_root or graph_profile.get("qwen_model_root", "/content/qwen3-8b-local")),
        device=args.device,
    )

    outcome = run(
        notes, entities, specs, documents, graph_config,
        icd_index=icd_index, rxnorm_index=rxnorm_index, structured=structured,
        qwen=qwen, qwen_revision=qwen_revision, config=settings,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    counts = write_variants(run_dir, entities, outcome)
    summary = diagnostics(
        outcome, counts, settings,
        manifest_total=int(manifest.get("total_deployed_parameters", 0)),
    )
    (run_dir / "diagnostic-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    log(f"diagnostics -> {run_dir / 'diagnostic-summary.json'}")
    for variant in VARIANTS:
        log(f"  {variant:18s} -> {run_dir / f'output_{variant}_raw_dotless'}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    return _execute(args, smoke=True)


def cmd_run(args: argparse.Namespace) -> int:
    return _execute(args, smoke=False)


def cmd_package(args: argparse.Namespace) -> int:
    """Reuse GraphCENT's derivation, organizer validation, zipping and hashing.

    Only the variant names differ, so the loop is here and every check inside it is the
    0080 one - a 0082 submission passes exactly the same gate as a 0080 submission.
    """
    import subprocess

    from mednorm_vi.inference.packaging import package_output_zip

    graphcent = graphcent_cli()
    root = Path(args.run_dir)
    if not root.is_dir():
        print(f"BLOCKED: run directory not found: {root}", file=sys.stderr)
        return 2

    graph_profile = graphcent.load_config(Path(args.graphcent_config))
    icd, rxnorm, _, _ = graphcent._load_kb(graph_profile)
    governed = {
        "ICD10": frozenset(icd.records) if icd else frozenset(),
        "RXNORM": frozenset(rxnorm.records) if rxnorm else frozenset(),
    }
    notes: dict[str, str] = {}
    if args.input_dir:
        notes = read_documents(Path(args.input_dir))

    produced: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        source = root / f"output_{variant}_raw_dotless"
        if not source.is_dir():
            log(f"{variant}: {source} absent, skipped")
            continue
        final = root / f"output_{variant}"
        subprocess.run(
            [sys.executable, "-m", "mednorm_vi.inference.derive_submission",
             "--source-dir", str(source), "--output-dir", str(final),
             "--derivation", "icd_dotted",
             "--expected-documents", str(args.expected_documents),
             "--validation-report", str(root / f"derivation-{variant}.json")],
            cwd=REPO, check=True,
            env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
        )
        issues = graphcent.validate_output(
            final, notes, governed, args.expected_documents
        )
        if issues:
            print(f"BLOCKED: {variant} failed organizer validation:", file=sys.stderr)
            for issue in issues[:20]:
                print(f"  - {issue}", file=sys.stderr)
            return 3
        zip_path = package_output_zip(
            final, root / f"output_{variant}.zip", expected_count=args.expected_documents
        )
        digest = graphcent.sha256_file(Path(zip_path))
        produced[variant] = {"zip": str(zip_path), "sha256": digest}
        log(f"{variant}: {zip_path}  sha256 {digest}")

    (root / "package-summary.json").write_text(
        json.dumps({"variants": produced, "submitted": False}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0 if produced else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--graphcent-config", type=Path,
        default=REPO / "configs/pipeline/graphcent_0080.yaml",
        help="the canonical KB/retriever profile; 0082 does not define its own",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    budget = sub.add_parser("budget")
    budget.add_argument("--model-root", default="/content/graphcent_models")
    budget.add_argument("--expect-total", type=int, default=0)
    budget.set_defaults(func=cmd_budget)

    for name, handler in (("smoke", cmd_smoke), ("run", cmd_run)):
        runner = sub.add_parser(name)
        runner.add_argument("--input-dir", type=Path, required=True)
        runner.add_argument("--entities", type=Path, required=True,
                            help="a 0081 entities_<variant> directory")
        runner.add_argument("--run-dir", type=Path, required=True)
        runner.add_argument("--documents", default="")
        runner.add_argument("--model-root", default="/content/graphcent_models")
        runner.add_argument("--cache-root", default="/content/graphcent_cache")
        runner.add_argument("--qwen-root", default="")
        runner.add_argument("--device", default="cuda")
        runner.add_argument("--embed-batch-size", type=int, default=256)
        runner.add_argument("--allow-download", action="store_true")
        runner.set_defaults(func=handler)

    package = sub.add_parser("package")
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--expected-documents", type=int, default=100)
    package.add_argument("--input-dir", default="",
                         help="organizer input, so source offsets are verified exactly")
    package.set_defaults(func=cmd_package)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
