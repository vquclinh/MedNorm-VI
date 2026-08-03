#!/usr/bin/env python3
"""MedNorm-VI production CLI. One system, one inference path.

    python scripts/run_mednorm.py budget   --model-root ...
    python scripts/run_mednorm.py acquire  --model-root ... --allow-download
    python scripts/run_mednorm.py index    --model-root ... --cache-root ...
    python scripts/run_mednorm.py run      --input-dir ... --output-dir ...
    python scripts/run_mednorm.py package  --output-dir ...

`budget` certifies the deployed stack against the canonical manifest, `acquire` resolves and
downloads the frozen retrievers at pinned commits, `index` embeds the governed knowledge base
once per retriever, `run` performs the whole pipeline, and `package` validates and archives a
submission. Everything reads one profile: configs/pipeline/production.yaml.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.config import RuntimeConfig  # noqa: E402
from mednorm_vi.extraction.runtime import ContextualConfig  # noqa: E402
from mednorm_vi.linking.pipeline import ConfidencePolicy  # noqa: E402
from mednorm_vi.linking.retrieval import RetrievalPolicy  # noqa: E402
from mednorm_vi.linking.runtime import OntoFusionConfig  # noqa: E402
from mednorm_vi.linking.sparse import SparseSettings  # noqa: E402
from mednorm_vi.models.qwen import log  # noqa: E402
from mednorm_vi.models.registry import (  # noqa: E402
    DEFAULT_RETRIEVERS,
    PARAMETER_CAP,
    RetrieverSpec,
)
from mednorm_vi.production import pipeline as production  # noqa: E402
from mednorm_vi.validation.organizer import (  # noqa: E402
    ASSERTION_TYPES,
    CANDIDATE_TYPES,
    ORGANIZER_TYPES,
)

DEFAULT_CONFIG = REPO / "configs/pipeline/production.yaml"
MANIFEST_NAME = "model-manifest.json"

#: The certified deployed total. `budget` refuses to continue if the stack has moved.
CANONICAL_PARAMETERS = 8_729_759_237

#: Sequential loading means peak memory tracks the largest single model, not their sum.
MIN_VRAM_FOR_QWEN_GIB = 18.0
MIN_VRAM_FOR_RETRIEVER_GIB = 6.0
RECOMMENDED_VRAM_GIB = 40.0

#: Retriever fields a profile may set. Everything else is provenance and belongs to the
#: registry, so a config file can never forge a licence or a repository id.
PROFILE_OVERRIDABLE: frozenset[str] = frozenset({"enabled", "roles"})


# ----------------------------------------------------------------------- configuration


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def resolve_specs(config: dict[str, Any]) -> list[RetrieverSpec]:
    """Apply profile overrides to the registry without editing code."""
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
    """Attach the commit each retriever was acquired at. Refuses to guess."""
    from mednorm_vi.models.acquisition import read_pinned_revisions
    from mednorm_vi.models.registry import RevisionNotPinned, is_pinned_revision

    recorded = read_pinned_revisions(model_root)
    out, missing = [], []
    for spec in specs:
        if not spec.enabled:
            out.append(spec)
            continue
        revision = spec.revision or recorded.get(spec.key, "")
        if not is_pinned_revision(revision):
            missing.append(spec.key)
        out.append(dataclasses.replace(spec, revision=revision))
    if missing:
        raise RevisionNotPinned(
            f"no pinned revision recorded for {missing} in {model_root / MANIFEST_NAME}. "
            "Run `acquire` first; it resolves and records the exact commit."
        )
    return out


def runtime_config(config: dict[str, Any], args: argparse.Namespace) -> RuntimeConfig:
    retrieval = config.get("retrieval") or {}
    return RuntimeConfig(
        model_root=Path(getattr(args, "model_root", "") or config["model_root"]),
        cache_root=Path(getattr(args, "cache_root", "") or config["cache_root"]),
        qwen_model_root=Path(getattr(args, "qwen_root", "") or config["qwen_model_root"]),
        device=getattr(args, "device", "cuda"),
        top_k_per_retriever=int(retrieval.get("dense_top_k", 4)),
        max_candidate_context=int(retrieval.get("candidate_cap", 10)),
        embed_batch_size=int(getattr(args, "embed_batch_size", 0) or 256),
        max_new_tokens=int(config.get("max_new_tokens", 1536)),
    )


def extraction_config(config: dict[str, Any]) -> ContextualConfig:
    section = config.get("extraction") or {}
    return ContextualConfig(
        use_qwen_proposer=bool(section.get("use_qwen_proposer", True)),
        use_alias_proposer=bool(section.get("use_alias_proposer", True)),
        use_verifier=bool(section.get("use_verifier", True)),
        context_radius=int(section.get("context_radius", 200)),
        max_options_per_group=int(section.get("max_options_per_group", 12)),
        alias_support_only=bool(section.get("alias_support_only", True)),
    )


def linking_config(config: dict[str, Any]) -> OntoFusionConfig:
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
    )


# ----------------------------------------------------------------------- assets


def read_documents(input_dir: Path) -> dict[str, str]:
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(Path(input_dir).glob("*"))
        if path.is_file()
    }


def read_entities(entity_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(entity_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = payload if isinstance(payload, list) else []
    return out


def load_kb(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], list[Any]]:
    """The governed knowledge base: both indices, structured drugs, semantic documents."""
    from mednorm_vi.kb.index_cache import governed_kb
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
                    row = json.loads(line)
                    structured[str(row.get("rxcui", ""))] = StructuredDrug.from_dict(row)
    return icd, rxnorm, structured, governed_kb(icd, rxnorm, structured)


def build_lexicon(config: dict[str, Any], icd: Any, rxnorm: Any) -> Any:
    """Governed surface forms, filtered by measured quality. A proposal source only."""
    from mednorm_vi.extraction.aliases import build_lexicon as build

    alias = config.get("alias") or {}
    sources = [(index, kind) for index, kind in ((icd, "CHẨN_ĐOÁN"), (rxnorm, "THUỐC"))
               if index is not None]
    if not sources:
        return None
    lexicon = build(
        sources,
        max_tokens=int(alias.get("max_tokens", 8)),
        max_concepts_per_form=int(alias.get("max_concepts_per_form", 3)),
        max_single_token_label_share=float(
            alias.get("max_single_token_label_share", 0.01)
        ),
    )
    log(f"governed alias lexicon: {json.dumps(lexicon.as_dict(), ensure_ascii=False)}")
    return lexicon


def canonical_manifest(model_root: Path) -> dict[str, Any]:
    path = Path(model_root) / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(
            f"canonical model manifest not found: {path}. Run `acquire` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def host_vram_gib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.get_device_properties(0).total_memory) / 2**30
    except ImportError:  # pragma: no cover - torch present on any run host
        pass
    return 0.0


def require_host(stage: str, *, allow_download: bool, needs: float) -> float:
    """Refuse only when the requested operation genuinely cannot run here."""
    if not allow_download:
        raise SystemExit(
            f"{stage} is refused without --allow-download. Model weights are never fetched "
            "or executed on a development machine by accident; see "
            "docs/colab/final-production-commands.md."
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
            f"{RECOMMENDED_VRAM_GIB:.0f} GiB. Sequential loading should still fit.")
    return available


# ----------------------------------------------------------------------- commands


def cmd_budget(args: argparse.Namespace) -> int:
    """Certify the deployed stack from the canonical manifest. Nothing is recomputed."""
    manifest = canonical_manifest(Path(args.model_root))
    total = int(manifest.get("total_deployed_parameters", 0))
    log("Deployed model stack (every model deployed, not only those resident at once):")
    for model in manifest.get("deployed") or ():
        log(f"  {model['name']:48s} {model['parameter_count']:>15,}  {model['revision']}")
    log(f"  {'TOTAL':48s} {total:>15,}  (cap {PARAMETER_CAP:,})")
    if total <= 0 or total >= PARAMETER_CAP:
        print("BLOCKED: the deployed total is not under the cap", file=sys.stderr)
        return 3
    expected = args.expect_total or CANONICAL_PARAMETERS
    if total != expected:
        print(
            f"BLOCKED: deployed total {total:,} != certified {expected:,}. The stack "
            "changed; re-certify before trusting this run.",
            file=sys.stderr,
        )
        return 3
    log("certified: the deployed stack is the canonical one and is under the cap")
    return 0


def cmd_acquire(args: argparse.Namespace) -> int:
    """Resolve the exact commit for every enabled retriever, then download at that commit."""
    require_host("acquire", allow_download=args.allow_download,
                 needs=MIN_VRAM_FOR_RETRIEVER_GIB)
    from mednorm_vi.models.acquisition import acquire
    from mednorm_vi.models.certification import certify_budget
    from mednorm_vi.models.qwen import QwenDisambiguator

    config = load_config(args.config)
    runtime = runtime_config(config, args)
    runtime.model_root.mkdir(parents=True, exist_ok=True)

    specs = acquire(resolve_specs(config), runtime.model_root)
    qwen = QwenDisambiguator(runtime.qwen_model_root, device=runtime.device)
    manifest = certify_budget(
        specs, runtime,
        e3_checkpoint=Path(args.e3_checkpoint) if args.e3_checkpoint else None,
        qwen=qwen if runtime.qwen_model_root.exists() else None,
        licence_overrides=config.get("licence_overrides_accepted") or (),
    )
    qwen.unload()
    manifest.assert_revisions_pinned()

    payload = manifest.as_dict()
    payload["resolved_revisions"] = {s.key: s.revision for s in specs if s.enabled}
    (runtime.model_root / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    log(f"manifest -> {runtime.model_root / MANIFEST_NAME}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Embed the governed knowledge base once per retriever, one model resident at a time."""
    require_host("index", allow_download=args.allow_download,
                 needs=MIN_VRAM_FOR_RETRIEVER_GIB)
    from mednorm_vi.kb.index_cache import build_index_for

    config = load_config(args.config)
    runtime = runtime_config(config, args)
    runtime.cache_root.mkdir(parents=True, exist_ok=True)
    _, _, _, documents = load_kb(config)
    if not documents:
        print("BLOCKED: no governed KB documents; restore the indices first", file=sys.stderr)
        return 2
    log(f"governed documents {len(documents):,}")
    for spec in pin_specs(resolve_specs(config), runtime.model_root):
        if spec.enabled:
            build_index_for(spec, documents, runtime)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """The whole pipeline: extraction, verification, assertions, retrieval, reranking."""
    require_host("run", allow_download=args.allow_download, needs=MIN_VRAM_FOR_QWEN_GIB)
    from mednorm_vi.models.qwen import QwenDisambiguator

    config = load_config(args.config)
    runtime = runtime_config(config, args)
    specs = pin_specs(resolve_specs(config), runtime.model_root)

    notes = read_documents(Path(args.input_dir))
    if not notes:
        print(f"BLOCKED: no documents in {args.input_dir}", file=sys.stderr)
        return 2
    seed_dir = Path(args.seed_entities or config.get("seed_entities", ""))
    if not seed_dir.is_dir():
        print(f"BLOCKED: seed entity directory not found: {seed_dir}", file=sys.stderr)
        return 2
    seeds = read_entities(seed_dir)

    wanted = [d.strip() for d in (args.documents or "").split(",") if d.strip()]
    if wanted:
        notes = {k: v for k, v in notes.items() if k in wanted}

    icd, rxnorm, structured, documents = load_kb(config)
    if not documents:
        print("BLOCKED: no governed KB documents; restore the indices first", file=sys.stderr)
        return 2

    manifest = canonical_manifest(runtime.model_root)
    qwen_revision = next(
        (str(m.get("revision", "")) for m in manifest.get("deployed") or ()
         if "Qwen" in str(m.get("name", ""))),
        "",
    )
    qwen = QwenDisambiguator(
        runtime.qwen_model_root, device=runtime.device,
        max_new_tokens=runtime.max_new_tokens,
    )

    outcome = production.run(
        notes, seeds,
        specs=specs, kb_documents=documents, runtime=runtime,
        icd_index=icd, rxnorm_index=rxnorm, structured=structured,
        qwen=qwen, qwen_revision=qwen_revision,
        lexicon=build_lexicon(config, icd, rxnorm),
        extraction_config=extraction_config(config),
        linking_config=linking_config(config),
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    production.write_output(output, outcome)
    summary = production.diagnostics(
        outcome, manifest_total=int(manifest.get("total_deployed_parameters", 0))
    )
    (output / "diagnostic-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    log(f"diagnostics -> {output / 'diagnostic-summary.json'}")
    return 0


_DOTTED = re.compile(r"^[A-TV-Z]\d{2}(?:\.\d{1,2})?$")


def validate_output(
    directory: Path, notes: dict[str, str], governed: dict[str, frozenset[str]],
    expected: int,
) -> list[str]:
    """Every organizer rule, checked against the source text and the governed KB."""
    issues: list[str] = []
    files = sorted(Path(directory).glob("*.json"))
    if len(files) != expected:
        issues.append(f"expected {expected} documents, found {len(files)}")

    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        source = notes.get(path.stem, "")
        for row in rows:
            entity_type = str(row.get("type", ""))
            if entity_type not in ORGANIZER_TYPES:
                issues.append(f"{path.name}: unknown type {entity_type!r}")
                continue
            allowed = {"position", "text", "type"}
            if entity_type in ASSERTION_TYPES:
                allowed.add("assertions")
            if entity_type in CANDIDATE_TYPES:
                allowed.add("candidates")
            for field_name in set(row) - allowed:
                issues.append(f"{path.name}: unsupported field {field_name!r} on {entity_type}")
            if "assertions" in row and entity_type not in ASSERTION_TYPES:
                issues.append(f"{path.name}: assertions on {entity_type}")
            if "candidates" in row and entity_type not in CANDIDATE_TYPES:
                issues.append(f"{path.name}: candidates on {entity_type}")

            position = row.get("position") or []
            if source and len(position) == 2:
                start, end = int(position[0]), int(position[1])
                if source[start:end] != row.get("text"):
                    issues.append(f"{path.name}: offsets do not match the source text")

            for code in row.get("candidates") or []:
                text = str(code)
                if entity_type == "CHẨN_ĐOÁN":
                    if not _DOTTED.match(text):
                        issues.append(f"{path.name}: {text!r} is not dotted-serializable")
                    elif text.replace(".", "") not in governed["ICD10"]:
                        issues.append(f"{path.name}: ungoverned ICD code {text!r}")
                elif text not in governed["RXNORM"]:
                    issues.append(f"{path.name}: ungoverned RxCUI {text!r}")
    return issues


def cmd_package(args: argparse.Namespace) -> int:
    """Derive dotted ICD codes, validate strictly, archive and hash."""
    import subprocess

    from mednorm_vi.packaging import package_output_zip

    config = load_config(args.config)
    root = Path(args.output_dir)
    source = root / "output_raw_dotless"
    if not source.is_dir():
        print(f"BLOCKED: {source} not found; run `run` first", file=sys.stderr)
        return 2

    icd, rxnorm, _, _ = load_kb(config)
    governed = {
        "ICD10": frozenset(icd.records) if icd else frozenset(),
        "RXNORM": frozenset(rxnorm.records) if rxnorm else frozenset(),
    }
    notes = read_documents(Path(args.input_dir)) if args.input_dir else {}

    final = root / "output"
    subprocess.run(
        [sys.executable, "-m", "mednorm_vi.packaging.derive_submission",
         "--source-dir", str(source), "--output-dir", str(final),
         "--derivation", "icd_dotted",
         "--expected-documents", str(args.expected_documents),
         "--validation-report", str(root / "derivation.json")],
        cwd=REPO, check=True,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
    )
    issues = validate_output(final, notes, governed, args.expected_documents)
    if issues:
        print("BLOCKED: submission failed organizer validation:", file=sys.stderr)
        for issue in issues[:20]:
            print(f"  - {issue}", file=sys.stderr)
        return 3

    zip_path = package_output_zip(
        final, root / "output.zip", expected_count=args.expected_documents
    )
    digest = sha256_file(Path(zip_path))
    (root / "package-summary.json").write_text(
        json.dumps({"zip": str(zip_path), "sha256": digest, "submitted": False},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log(f"{zip_path}  sha256 {digest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    budget = sub.add_parser("budget", help="certify the deployed parameter total")
    budget.add_argument("--model-root", default="")
    budget.add_argument("--expect-total", type=int, default=0)
    budget.set_defaults(func=cmd_budget)

    acquire = sub.add_parser("acquire", help="download the frozen retrievers at pinned commits")
    acquire.add_argument("--model-root", default="")
    acquire.add_argument("--qwen-root", default="")
    acquire.add_argument("--e3-checkpoint", default="")
    acquire.add_argument("--cache-root", default="")
    acquire.add_argument("--device", default="cuda")
    acquire.add_argument("--allow-download", action="store_true")
    acquire.set_defaults(func=cmd_acquire)

    index = sub.add_parser("index", help="embed the governed KB once per retriever")
    index.add_argument("--model-root", default="")
    index.add_argument("--cache-root", default="")
    index.add_argument("--qwen-root", default="")
    index.add_argument("--device", default="cuda")
    index.add_argument("--embed-batch-size", type=int, default=0)
    index.add_argument("--allow-download", action="store_true")
    index.set_defaults(func=cmd_index)

    runner = sub.add_parser("run", help="the whole pipeline, end to end")
    runner.add_argument("--input-dir", type=Path, required=True)
    runner.add_argument("--output-dir", type=Path, required=True)
    runner.add_argument("--seed-entities", default="")
    runner.add_argument("--documents", default="")
    runner.add_argument("--model-root", default="")
    runner.add_argument("--cache-root", default="")
    runner.add_argument("--qwen-root", default="")
    runner.add_argument("--device", default="cuda")
    runner.add_argument("--embed-batch-size", type=int, default=0)
    runner.add_argument("--allow-download", action="store_true")
    runner.set_defaults(func=cmd_run)

    package = sub.add_parser("package", help="validate, archive and hash a submission")
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--input-dir", default="",
                         help="organizer input, so source offsets are verified exactly")
    package.add_argument("--expected-documents", type=int, default=100)
    package.set_defaults(func=cmd_package)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
