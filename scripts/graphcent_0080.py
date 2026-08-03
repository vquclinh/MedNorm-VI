#!/usr/bin/env python3
"""GraphCENT 0080 CLI: download-models | build-index | smoke | run | package.

Local development never downloads a checkpoint. `download-models` and any stage that needs
weights refuse to run outside a GPU host, and every unit test uses injected fake retrievers.

Subcommands:
    download-models   fetch the frozen retrievers into --model-root, record resolved
                      revisions/licences/parameter counts, enforce the 9B cap
    build-index       embed the governed KB once per retriever into --cache-root
    smoke             a few documents end to end, with per-mention diagnostics
    run               all 100 documents; writes every output variant from ONE pass
    package           strict validation + dotted derivation + zip + SHA256
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.graphcent.disambiguation import TierPolicy  # noqa: E402
from mednorm_vi.graphcent.models import (  # noqa: E402
    DEFAULT_RETRIEVERS,
    RetrieverSpec,
)

DEFAULT_CONFIG = REPO / "configs/pipeline/graphcent_0080.yaml"

#: The only retriever fields a profile may set. Everything else - repo id, licence, licence
#: source, pooling, expected size - is provenance and belongs to the registry alone.
PROFILE_OVERRIDABLE: frozenset[str] = frozenset({"enabled", "roles"})


def log(message: str) -> None:
    print(f"[0080] {message}", flush=True)


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_specs(config: dict[str, Any]) -> list[RetrieverSpec]:
    """Apply profile overrides to the registry without editing code.

    `dataclasses.replace` and not a fresh `RetrieverSpec(...)`: the registry in
    `graphcent/models.py` is the single authority for provenance (repo id, licence, licence
    source URL, pooling, expected size). Rebuilding a spec field by field silently drops
    whatever the profile has no opinion about, which is exactly how `licence_source` went
    missing. A profile may only change what a profile is allowed to change.
    """
    overrides = config.get("retrievers") or {}
    out: list[RetrieverSpec] = []
    for spec in DEFAULT_RETRIEVERS:
        row = overrides.get(spec.key) or {}
        unknown = set(row) - PROFILE_OVERRIDABLE
        if unknown:
            raise SystemExit(
                f"{spec.key}: profile may not override {sorted(unknown)}; provenance and "
                "pooling come from the registry only."
            )
        roles = row.get("roles")
        out.append(
            dataclasses.replace(
                spec,
                role=tuple(roles) if roles else spec.role,
                enabled=bool(row.get("enabled", spec.enabled)),
            )
        )
    return out


def pin_specs(specs: list[RetrieverSpec], model_root: Path) -> list[RetrieverSpec]:
    """Attach the revision recorded by `download-models` to every enabled retriever.

    Every later stage runs against the exact snapshot acquisition pinned, and refuses to
    run at all if that record is missing - guessing here would mean embedding the KB with
    weights nobody can name.
    """
    from mednorm_vi.graphcent.acquisition import read_pinned_revisions
    from mednorm_vi.graphcent.models import RevisionNotPinned, is_hub_revision

    recorded = read_pinned_revisions(model_root)
    out: list[RetrieverSpec] = []
    missing: list[str] = []
    for spec in specs:
        if not spec.enabled:
            out.append(spec)
            continue
        revision = spec.revision if is_hub_revision(spec.revision) else recorded.get(
            spec.key, ""
        )
        if not is_hub_revision(revision):
            missing.append(spec.key)
        out.append(dataclasses.replace(spec, revision=revision))
    if missing:
        raise RevisionNotPinned(
            f"no pinned revision recorded for {missing} in {model_root / 'model-manifest.json'}. "
            "Run download-models first; it resolves and records the exact commit SHA."
        )
    return out


def certified_manifest_for_stage(
    specs: list[RetrieverSpec],
    model_root: Path,
    *,
    e3_checkpoint: Path | None = None,
) -> Any:
    """Canonical model manifest produced by `download-models`, validated for this stage."""
    from mednorm_vi.graphcent.acquisition import MANIFEST_NAME, file_digest
    from mednorm_vi.graphcent.models import (
        ModelManifest,
        RevisionNotPinned,
        is_local_checkpoint_revision,
    )
    from mednorm_vi.reasoner.budget import QWEN3_8B_REVISION

    path = model_root / MANIFEST_NAME
    if not path.is_file():
        raise RevisionNotPinned(
            f"no certified model manifest at {path}. Run download-models first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = ModelManifest.from_dict(payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RevisionNotPinned(f"{path}: malformed certified model manifest: {error}") from error

    manifest.assert_licences_cleared()
    manifest.assert_revisions_pinned()
    total = manifest.assert_within_cap()

    problems: list[str] = []
    try:
        recorded_total = int(payload.get("total_deployed_parameters", -1))
    except (TypeError, ValueError):
        recorded_total = -1
    if recorded_total != total:
        problems.append(
            "total_deployed_parameters does not match the deployed rows "
            f"({payload.get('total_deployed_parameters')} != {total})"
        )
    if payload.get("training_performed") is not False:
        problems.append("training_performed is not false")

    expected_enabled = {spec.key for spec in specs if spec.enabled}
    actual_enabled = {spec.key for spec in manifest.retrievers if spec.enabled}
    if actual_enabled != expected_enabled:
        problems.append(
            f"enabled retrievers differ from the current profile "
            f"({sorted(actual_enabled)} != {sorted(expected_enabled)})"
        )

    by_key = {spec.key: spec for spec in manifest.retrievers}
    expected_revisions = {spec.key: spec.revision for spec in specs if spec.enabled}
    if manifest.resolved_revisions != expected_revisions:
        problems.append(
            f"resolved_revisions differ from current pins "
            f"({manifest.resolved_revisions} != {expected_revisions})"
        )
    for spec in specs:
        if not spec.enabled:
            continue
        recorded = by_key.get(spec.key)
        if recorded is None:
            problems.append(f"{spec.key}: missing retriever row")
            continue
        if recorded.repo_id != spec.repo_id:
            problems.append(f"{spec.key}: repo differs from registry")
        if recorded.pooling != spec.pooling:
            problems.append(f"{spec.key}: pooling differs from registry")
        if recorded.role != spec.role:
            problems.append(f"{spec.key}: roles differ from current profile")
        if recorded.revision != spec.revision:
            problems.append(f"{spec.key}: revision differs from current pin")
        if recorded.parameter_count is None or recorded.parameter_count <= 0:
            problems.append(f"{spec.key}: parameter count was not certified")

    deployed = {row.name: row for row in manifest.deployed}
    qwen = deployed.get("Qwen/Qwen3-8B")
    if qwen is None or qwen.revision != QWEN3_8B_REVISION:
        problems.append("Qwen/Qwen3-8B is missing or not at the repository-pinned revision")
    e3 = deployed.get("ViHealthBERT E3")
    if e3 is None or not is_local_checkpoint_revision(e3.revision):
        problems.append("ViHealthBERT E3 is missing or not recorded as a sha256 local checkpoint")
    elif e3_checkpoint is not None and e3.revision != file_digest(e3_checkpoint):
        problems.append("ViHealthBERT E3 digest differs from --e3-checkpoint")

    if problems:
        raise RevisionNotPinned(
            f"{path}: certified model manifest is inconsistent:\n  - "
            + "\n  - ".join(problems)
        )
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


#: Sequential loading means peak VRAM tracks the LARGEST single model, not their sum.
#: Qwen3-8B in bf16 is ~16.4 GiB of weights plus activations; the retrievers are far smaller.
MIN_VRAM_FOR_QWEN_GIB = 18.0
MIN_VRAM_FOR_RETRIEVER_GIB = 6.0
RECOMMENDED_VRAM_GIB = 40.0


def host_vram_gib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.get_device_properties(0).total_memory) / 2**30
    except ImportError:  # pragma: no cover
        pass
    return 0.0


def require_host(stage: str, *, allow_download: bool, needs: float) -> float:
    """Refuse only when the requested operation genuinely cannot run here.

    `--allow-download` is an explicit statement of intent a script cannot infer, and keeps a
    development machine from starting a 16 GB fetch by habit. The memory floor is then per
    operation rather than a blanket threshold: GraphCENT loads one model at a time, so a
    22.5 GiB L4 runs the whole pipeline even though it is under the recommended 40 GiB.
    """
    if not allow_download:
        raise SystemExit(
            f"{stage} is refused without --allow-download. GraphCENT weights are never "
            "fetched or executed on a development machine by accident; see "
            "docs/colab/0080-graphcent-commands.md."
        )
    available = host_vram_gib()
    if available <= 0:
        raise SystemExit(f"{stage} needs a CUDA device; none is visible.")
    if available < needs:
        raise SystemExit(
            f"{stage} refused: this host reports {available:.1f} GiB VRAM but the largest "
            f"model in this stage needs about {needs:.0f} GiB even with sequential loading."
        )
    if available < RECOMMENDED_VRAM_GIB:
        log(f"WARNING: {available:.1f} GiB VRAM is below the recommended "
            f"{RECOMMENDED_VRAM_GIB:.0f} GiB. Sequential loading should still fit; reduce "
            "--embed-batch-size if you hit OOM.")
    return available


def _runtime_config(config: dict[str, Any], args: argparse.Namespace) -> Any:
    from mednorm_vi.graphcent.disambiguation import TierPolicy
    from mednorm_vi.graphcent.runtime import RuntimeConfig

    tiers = config.get("tiers") or {}
    mode = str(config.get("mode", "graphcent_candidate_only"))
    joint = mode == "graphcent_joint_safe_span"
    return RuntimeConfig(
        model_root=Path(getattr(args, "model_root", "") or config.get(
            "model_root", "/content/graphcent_models")),
        cache_root=Path(getattr(args, "cache_root", "") or config.get(
            "cache_root", "/content/graphcent_cache")),
        qwen_model_root=Path(
            getattr(args, "qwen_root", "") or config.get(
                "qwen_model_root", "/content/qwen3-8b-local"
            )
        ),
        device=getattr(args, "device", "cuda"),
        top_k_per_retriever=int(config.get("top_k_per_retriever", 5)),
        max_candidate_context=int(config.get("max_candidate_context", 10)),
        embed_batch_size=int(getattr(args, "embed_batch_size", 256)),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        joint_safe_span=joint,
        apply_spans=joint and bool(config.get("allow_boundary_changes", False)),
        tier_policy=TierPolicy(
            allow_tier_a=bool(tiers.get("allow_tier_a", True)),
            allow_tier_b=bool(tiers.get("allow_tier_b", True)),
            allow_tier_c=bool(tiers.get("allow_tier_c", True)),
            min_retrievers_for_tier_b=int(tiers.get("min_retrievers_for_tier_b", 2)),
        ),
    )


def _load_kb(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], list[Any]]:
    from mednorm_vi.graphcent.runtime import governed_kb
    from mednorm_vi.kb.indexing.retrieval import load_index as load_local
    from mednorm_vi.kb.rxnorm.structured import StructuredDrug

    icd_path = REPO / str(config.get("icd_index", ""))
    rx_path = REPO / str(config.get("rxnorm_index", ""))
    icd = load_local(icd_path) if icd_path.is_file() else None
    rxnorm = load_local(rx_path) if rx_path.is_file() else None
    structured: dict[str, Any] = {}
    structured_path = REPO / str(config.get("rxnorm_structured", ""))
    if structured_path.is_file():
        with structured_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    drug = StructuredDrug.from_dict(json.loads(line))
                    structured[drug.rxcui] = drug
    return icd, rxnorm, structured, governed_kb(icd, rxnorm, structured)


def _read_note(path: Path) -> str:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("text", "content", "note"):
                if key in payload:
                    return str(payload[key])
        return str(payload)
    return path.read_text(encoding="utf-8")


def cmd_download_models(args: argparse.Namespace) -> int:
    """Fetch enabled retrievers at pinned revisions and certify the parameter budget."""
    require_host("download-models", allow_download=args.allow_download,
                 needs=MIN_VRAM_FOR_RETRIEVER_GIB)
    from mednorm_vi.graphcent.acquisition import acquire
    from mednorm_vi.graphcent.runtime import QwenDisambiguator, certify_budget

    config = load_config(args.config)
    specs = resolve_specs(config)
    runtime = _runtime_config(config, args)
    runtime.model_root.mkdir(parents=True, exist_ok=True)

    # Resolve the exact commit SHA first, then download AT that SHA. Pinning after the
    # fact would only describe what was fetched; pinning first decides it.
    specs = acquire(specs, runtime.model_root)
    qwen = QwenDisambiguator(runtime.qwen_model_root, device=runtime.device)
    manifest = certify_budget(
        specs, runtime,
        e3_checkpoint=Path(args.e3_checkpoint) if args.e3_checkpoint else None,
        qwen=qwen if args.qwen_root or runtime.qwen_model_root.exists() else None,
        licence_overrides=config.get("licence_overrides_accepted") or (),
    )
    qwen.unload()
    resolved_revisions = {s.key: s.revision for s in manifest.retrievers if s.enabled}
    manifest.resolved_revisions = resolved_revisions
    manifest.assert_revisions_pinned()  # nothing is written that cannot be reproduced
    payload = manifest.as_dict()
    (runtime.model_root / "model-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    log(f"manifest -> {runtime.model_root / 'model-manifest.json'}")
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    """Embed the governed KB once per enabled retriever, one model resident at a time."""
    require_host("build-index", allow_download=args.allow_download,
                 needs=MIN_VRAM_FOR_RETRIEVER_GIB)
    from mednorm_vi.graphcent.runtime import build_index_for

    config = load_config(args.config)
    runtime = _runtime_config(config, args)
    runtime.cache_root.mkdir(parents=True, exist_ok=True)
    _, _, _, documents = _load_kb(config)
    if not documents:
        print("BLOCKED: no governed KB documents; restore the indices first", file=sys.stderr)
        return 2
    log(f"governed documents {len(documents):,}")
    specs = pin_specs(resolve_specs(config), runtime.model_root)
    certified_manifest_for_stage(specs, runtime.model_root)
    for spec in specs:
        if not spec.enabled:
            continue
        build_index_for(spec, documents, runtime)
    return 0


def _execute(args: argparse.Namespace, *, smoke: bool) -> int:
    from mednorm_vi.graphcent.runtime import (
        QwenDisambiguator,
        run_documents,
        write_variants,
    )

    stage = "smoke" if smoke else "run"
    require_host(stage, allow_download=args.allow_download, needs=MIN_VRAM_FOR_QWEN_GIB)
    config = load_config(args.config)
    runtime = _runtime_config(config, args)
    specs = pin_specs(resolve_specs(config), runtime.model_root)

    seed_dir = Path(args.seed_entities or config.get("seed_entities", ""))
    if not seed_dir.is_dir():
        print(f"BLOCKED: E3 seed directory not found: {seed_dir}", file=sys.stderr)
        return 2
    seed_config = REPO / str(config.get("seed_run_config", ""))
    if seed_config.is_file() and "semantic_s1_0072" in seed_config.read_text(encoding="utf-8"):
        print("BLOCKED: the seed profile deploys Embedding-4B/Reranker-4B; that stack plus "
              "GraphCENT exceeds the 9B cap. Seed from an E3-only profile.", file=sys.stderr)
        return 2

    notes_dir = Path(args.input_dir)
    paths = sorted(p for p in notes_dir.glob("*") if p.is_file())
    if args.documents:
        wanted = {s.strip() for s in args.documents.split(",") if s.strip()}
        paths = [p for p in paths if p.stem in wanted]
    if smoke and not args.documents:
        paths = paths[:3]
    if not paths:
        print("BLOCKED: no documents selected", file=sys.stderr)
        return 2

    notes = {p.stem: _read_note(p) for p in paths}
    seeds: dict[str, list[dict[str, Any]]] = {}
    for stem in notes:
        seed_path = seed_dir / f"{stem}.json"
        seeds[stem] = (
            json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.is_file() else []
        )

    icd, rxnorm, structured, documents = _load_kb(config)
    manifest = certified_manifest_for_stage(
        specs,
        runtime.model_root,
        e3_checkpoint=Path(args.e3_checkpoint) if args.e3_checkpoint else None,
    )
    qwen = QwenDisambiguator(
        runtime.qwen_model_root, device=runtime.device,
        max_new_tokens=runtime.max_new_tokens,
    )

    started = time.time()
    outcome = run_documents(
        notes, seeds, specs, documents, runtime,
        icd_index=icd, rxnorm_index=rxnorm, structured=structured,
        disambiguator=qwen,
    )
    qwen.unload()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    emitted = write_variants(run_dir, seeds, outcome, runtime)

    peak = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except ImportError:  # pragma: no cover
        pass
    diagnostics = {
        **outcome.diagnostics,
        "mode": config.get("mode"),
        "variants_emitted_codes": emitted,
        "model_manifest": manifest.as_dict(),
        "runtime_seconds": round(time.time() - started, 1),
        "peak_vram_gib": peak,
        "smoke": smoke,
        "submitted": False,
        "training_performed": False,
    }
    (run_dir / "diagnostic-summary.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    log(json.dumps({k: v for k, v in diagnostics.items() if k != "model_manifest"},
                   ensure_ascii=False)[:600])
    if not smoke and len(seeds) != args.expected_documents:
        print(f"BLOCKED: processed {len(seeds)} documents, expected "
              f"{args.expected_documents}", file=sys.stderr)
        return 3
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    return _execute(args, smoke=True)


def cmd_run(args: argparse.Namespace) -> int:
    return _execute(args, smoke=False)


ALLOWED_TYPES = frozenset({
    "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC",
})
ASSERTION_ALLOWED = frozenset({"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"})
CANDIDATE_ALLOWED = frozenset({"CHẨN_ĐOÁN", "THUỐC"})
DOTTED_ICD = re.compile(r"^[A-Z]\d{2}(\.\d+)?$")


def validate_output(
    directory: Path, notes: dict[str, str], governed: dict[str, frozenset[str]], expected: int
) -> list[str]:
    """Full organizer-contract validation. Every failure is named, none is suppressed."""
    issues: list[str] = []
    files = sorted(directory.glob("*.json"))
    if len(files) != expected:
        issues.append(f"expected {expected} documents, found {len(files)}")
    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            issues.append(f"{path.name}: top level is not a list")
            continue
        source = notes.get(path.stem, "")
        for row in rows:
            entity_type = row.get("type")
            if entity_type not in ALLOWED_TYPES:
                issues.append(f"{path.name}: bad type {entity_type!r}")
                continue
            position = row.get("position")
            if (not isinstance(position, list) or len(position) != 2
                    or not all(isinstance(v, int) for v in position)):
                issues.append(f"{path.name}: bad position {position!r}")
                continue
            if source and source[position[0]:position[1]] != row.get("text"):
                issues.append(f"{path.name}: offsets do not match the source text")
            if "assertions" in row and entity_type not in ASSERTION_ALLOWED:
                issues.append(f"{path.name}: assertions on {entity_type}")
            if "candidates" in row and entity_type not in CANDIDATE_ALLOWED:
                issues.append(f"{path.name}: candidates on {entity_type}")
            for extra in set(row) - {"text", "type", "position", "assertions", "candidates"}:
                issues.append(f"{path.name}: unsupported field {extra!r}")
            for code in row.get("candidates") or []:
                if entity_type == "CHẨN_ĐOÁN":
                    if not DOTTED_ICD.match(str(code)):
                        issues.append(f"{path.name}: ICD code not dotted-serializable: {code}")
                    if str(code).replace(".", "") not in governed.get("ICD10", frozenset()):
                        issues.append(f"{path.name}: ungoverned ICD code {code}")
                elif str(code) not in governed.get("RXNORM", frozenset()):
                    issues.append(f"{path.name}: ungoverned RxCUI {code}")
    return issues


def cmd_package(args: argparse.Namespace) -> int:
    """Derive dotted ICD, validate strictly, package and hash every produced variant."""
    import subprocess

    from mednorm_vi.graphcent.disambiguation import VARIANTS
    from mednorm_vi.inference.packaging import package_output_zip

    root = Path(args.run_dir)
    if not root.is_dir():
        print(f"BLOCKED: run directory not found: {root}", file=sys.stderr)
        return 2
    config = load_config(args.config)
    icd, rxnorm, _, _ = _load_kb(config)
    governed = {
        "ICD10": frozenset(icd.records) if icd else frozenset(),
        "RXNORM": frozenset(rxnorm.records) if rxnorm else frozenset(),
    }
    notes: dict[str, str] = {}
    if args.input_dir:
        notes = {
            p.stem: _read_note(p)
            for p in sorted(Path(args.input_dir).glob("*")) if p.is_file()
        }

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
        issues = validate_output(final, notes, governed, args.expected_documents)
        if issues:
            print(f"BLOCKED: {variant} failed organizer validation:", file=sys.stderr)
            for issue in issues[:20]:
                print(f"  - {issue}", file=sys.stderr)
            return 3
        zip_path = package_output_zip(
            final, root / f"output_{variant}.zip", expected_count=args.expected_documents
        )
        digest = sha256_file(Path(zip_path))
        produced[variant] = {"zip": str(zip_path), "sha256": digest}
        log(f"{variant}: {zip_path}  sha256 {digest}")

    (root / "package-summary.json").write_text(
        json.dumps({"variants": produced, "submitted": False}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0 if produced else 2


def build_parser() -> argparse.ArgumentParser:
    """Built separately from `main` so tests can preflight a real command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download-models")
    download.add_argument("--model-root", default="")
    download.add_argument("--e3-checkpoint", default="")
    download.add_argument("--qwen-root", default="")
    download.add_argument(
        "--allow-download",
        action="store_true",
        help="explicit confirmation that this is a GPU host",
    )
    download.set_defaults(func=cmd_download_models)

    index = sub.add_parser("build-index")
    index.add_argument("--cache-root", default="")
    index.add_argument("--model-root", default="")
    index.add_argument("--device", default="cuda")
    index.add_argument("--embed-batch-size", type=int, default=256)
    index.add_argument("--allow-download", action="store_true")
    index.set_defaults(func=cmd_build_index)

    for name, handler in (("smoke", cmd_smoke), ("run", cmd_run)):
        runner = sub.add_parser(name)
        runner.add_argument("--input-dir", type=Path, required=True)
        runner.add_argument("--run-dir", type=Path, required=True)
        runner.add_argument("--seed-entities", default="")
        runner.add_argument("--e3-checkpoint", default="")
        runner.add_argument("--documents", default="")
        runner.add_argument("--device", default="cuda")
        runner.add_argument("--embed-batch-size", type=int, default=256)
        runner.add_argument("--model-root", default="")
        runner.add_argument("--cache-root", default="")
        runner.add_argument("--expected-documents", type=int, default=100)
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
    _ = TierPolicy()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
