#!/usr/bin/env python3
"""Contextual entity proposal and verification (milestone 0081).

Produces a better-verified entity set for the unchanged GraphCENT 0080 linker to attach
candidates to. The deployed model is the SAME Qwen3-8B GraphCENT already loads, at the same
pinned revision from the canonical model manifest - nothing new enters the parameter budget.

Subcommands:
    smoke     a few documents end to end, with per-document diagnostics
    run       every document; writes all three entity variants from ONE pass
    budget    re-certify the deployed stack against the canonical manifest

Output of `run`/`smoke` is an entity directory per variant, which is exactly the shape
`scripts/graphcent_0080.py --seed-entities` consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.contextual.runtime import (  # noqa: E402
    VARIANTS,
    ContextualConfig,
    diagnostics,
    log,
    process_document,
    write_variants,
)

DEFAULT_CONFIG = REPO / "configs/pipeline/contextual_0081.yaml"

#: 0081 adds no deployed model, so its memory floor is GraphCENT's Qwen floor.
MIN_VRAM_FOR_QWEN_GIB = 18.0


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_documents(input_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(input_dir.glob("*")):
        if path.is_file():
            out[path.stem] = path.read_text(encoding="utf-8")
    return out


def read_seeds(seed_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(seed_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = payload if isinstance(payload, list) else []
    return out


def build_lexicon(config: dict[str, Any]) -> Any:
    """Governed surface forms from whatever indices the profile names. Proposals only."""
    from mednorm_vi.contextual.aliases import build_lexicon as build
    from mednorm_vi.kb.indexing.retrieval import load_index

    sources: list[tuple[Any, str]] = []
    for key, entity_type in (
        ("icd_index", "CHẨN_ĐOÁN"),
        ("rxnorm_index", "THUỐC"),
    ):
        path = REPO / str(config.get(key, ""))
        if path.is_file():
            sources.append((load_index(path), entity_type))
        else:
            log(f"{key}: {path} absent - no alias proposals from this ontology")
    if not sources:
        return None
    lexicon = build(sources, max_tokens=int(config.get("alias_max_tokens", 8)))
    log(f"governed alias lexicon: {json.dumps(lexicon.as_dict(), ensure_ascii=False)}")
    return lexicon


def qwen_from_manifest(config: dict[str, Any], args: argparse.Namespace) -> Any:
    """The canonical Qwen3-8B, at the revision the 0080 manifest already pinned."""
    from mednorm_vi.graphcent.runtime import QwenDisambiguator

    root = Path(getattr(args, "qwen_root", "") or config.get(
        "qwen_model_root", "/content/qwen3-8b-local"))
    return QwenDisambiguator(root, device=getattr(args, "device", "cuda"))


def _host_guard(stage: str, args: argparse.Namespace) -> None:
    """Reuse the 0080 host policy rather than inventing a second one."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "graphcent_cli", REPO / "scripts/graphcent_0080.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_host(stage, allow_download=args.allow_download,
                        needs=MIN_VRAM_FOR_QWEN_GIB)


def _config_from(config: dict[str, Any]) -> ContextualConfig:
    return ContextualConfig(
        use_qwen_proposer=bool(config.get("use_qwen_proposer", True)),
        use_alias_proposer=bool(config.get("use_alias_proposer", True)),
        use_verifier=bool(config.get("use_verifier", True)),
        context_radius=int(config.get("context_radius", 200)),
        max_options_per_group=int(config.get("max_options_per_group", 12)),
    )


def _execute(args: argparse.Namespace, *, smoke: bool) -> int:
    stage = "smoke" if smoke else "run"
    _host_guard(stage, args)

    config = load_config(args.config)
    settings = _config_from(config)

    documents = read_documents(Path(args.input_dir))
    if not documents:
        print(f"BLOCKED: no documents in {args.input_dir}", file=sys.stderr)
        return 2
    seeds = read_seeds(Path(args.seed_entities or config.get("seed_entities", "")))
    if not seeds:
        print("BLOCKED: no E3 seed entities; 0081 refines a seed, it does not replace it",
              file=sys.stderr)
        return 2

    wanted = [d.strip() for d in (args.documents or "").split(",") if d.strip()]
    if wanted:
        documents = {k: v for k, v in documents.items() if k in wanted}
    log(f"documents {len(documents)}")

    lexicon = build_lexicon(config) if settings.use_alias_proposer else None
    qwen = qwen_from_manifest(config, args)

    outcomes = []
    for name, source in documents.items():
        outcome = process_document(
            name, source, seeds.get(name, []),
            qwen=qwen, lexicon=lexicon, config=settings,
        )
        outcomes.append(outcome)
        log(f"{name}: entities {len(outcome.entities)}")
    qwen.unload()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    counts = write_variants(run_dir, outcomes)
    summary = diagnostics(outcomes, counts)
    (run_dir / "diagnostic-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    log(f"diagnostics -> {run_dir / 'diagnostic-summary.json'}")
    for variant in VARIANTS:
        log(f"  {variant:16s} -> {run_dir / f'entities_{variant}'}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    return _execute(args, smoke=True)


def cmd_run(args: argparse.Namespace) -> int:
    return _execute(args, smoke=False)


def cmd_budget(args: argparse.Namespace) -> int:
    """0081 deploys no new model. Prove it against the canonical 0080 manifest."""
    from mednorm_vi.graphcent.acquisition import MANIFEST_NAME
    from mednorm_vi.graphcent.models import PARAMETER_CAP

    path = Path(args.model_root) / MANIFEST_NAME
    if not path.is_file():
        print(f"BLOCKED: canonical manifest not found: {path}", file=sys.stderr)
        return 2
    manifest = json.loads(path.read_text(encoding="utf-8"))
    total = int(manifest.get("total_deployed_parameters", 0))
    log("0081 adds no deployed base model; the stack is the 0080 stack.")
    for model in manifest.get("deployed") or ():
        log(f"  {model['name']:48s} {model['parameter_count']:>15,}  {model['revision']}")
    log(f"  {'TOTAL':48s} {total:>15,}  (cap {PARAMETER_CAP:,})")
    if total <= 0 or total >= PARAMETER_CAP:
        print("BLOCKED: canonical manifest is not under the cap", file=sys.stderr)
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("smoke", cmd_smoke), ("run", cmd_run)):
        runner = sub.add_parser(name)
        runner.add_argument("--input-dir", type=Path, required=True)
        runner.add_argument("--run-dir", type=Path, required=True)
        runner.add_argument("--seed-entities", default="")
        runner.add_argument("--documents", default="")
        runner.add_argument("--qwen-root", default="")
        runner.add_argument("--device", default="cuda")
        runner.add_argument("--allow-download", action="store_true")
        runner.set_defaults(func=handler)

    budget = sub.add_parser("budget")
    budget.add_argument("--model-root", default="/content/graphcent_models")
    budget.set_defaults(func=cmd_budget)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
