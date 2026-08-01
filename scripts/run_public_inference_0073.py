#!/usr/bin/env python3
"""Public inference with the frozen S1 semantic linker (Audit 0073 §9).

Uses the canonical production CLI path - the same `PipelineConfig` / `run_pipeline` the scored
11.9188 baseline used, in the same `specialist` mode - and changes exactly one thing: the
linker backend. Frozen S1 covers BOTH ontologies (ICD and RxNorm), because that is the system
Audit 0072 selected. It never submits anywhere.

All three heavy artifacts are supplied explicitly and reused in place; nothing is copied or
re-encoded:

    python scripts/run_public_inference_0073.py \\
      --input-dir data/organizer_test/input \\
      --run-dir runs/semantic_s1_0073_final_dotted \\
      --semantic-model-root  <0072>/models \\
      --semantic-doc-vectors <0072>/cache/S1_doc_vectors.pt \\
      --semantic-documents   <0072>/artifacts/documents.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from mednorm_vi.linking.semantic.runtime import (  # noqa: E402
    EXPECTED_DOCUMENT_COUNT,
    MODEL_ROOT_ENV,
    S1_EMBEDDING_DIRNAME,
    S1_RERANKER_DIRNAME,
    model_manifest,
)

EXPECTED_DOCUMENTS = 100
DOTTED = re.compile(r"^[A-Z]\d{2}(\.\d+)?$")
REQUIRED_KEYS = {"text", "type", "position"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(config: Any) -> list[str]:
    """Validate every heavy artifact BEFORE a single model byte is loaded.

    Checks identity, not just presence: a dense index built over a different corpus would map
    every retrieved vector to the wrong concept, and that failure is silent at inference time.
    """
    from mednorm_vi.linking.semantic.runtime import (
        BACKEND_SEMANTIC_S1,
        SemanticLinkerSettings,
        missing_requirements,
    )

    if config.linker_backend != BACKEND_SEMANTIC_S1:
        return []
    settings = SemanticLinkerSettings.from_mapping(config.semantic_linker)
    problems = list(missing_requirements(settings))
    if problems:
        return problems

    documents = Path(settings.documents)
    count = 0
    ontologies: dict[str, int] = {}
    with documents.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
                ontology = json.loads(line).get("ontology", "")
                ontologies[ontology] = ontologies.get(ontology, 0) + 1
    if count != EXPECTED_DOCUMENT_COUNT:
        problems.append(f"documents.jsonl has {count:,} rows, expected {EXPECTED_DOCUMENT_COUNT:,}")
    for ontology in ("ICD10", "RXNORM"):
        if not ontologies.get(ontology):
            problems.append(f"documents.jsonl contains no {ontology} documents")

    try:
        import torch

        vectors = torch.load(settings.dense_index, map_location="cpu")
        rows = int(vectors.shape[0])
        if rows != EXPECTED_DOCUMENT_COUNT:
            problems.append(
                f"dense index first dimension is {rows:,}, expected "
                f"{EXPECTED_DOCUMENT_COUNT:,}; this cache belongs to a different corpus"
            )
        if rows != count:
            problems.append(f"dense index rows ({rows:,}) != documents.jsonl rows ({count:,})")
    except ImportError:  # pragma: no cover - torch is present on any run host
        problems.append("torch is not installed; cannot validate the dense index")

    manifest = documents.parent / "manifest.json"
    if manifest.is_file():
        recorded = json.loads(manifest.read_text(encoding="utf-8")).get("documents")
        if recorded is not None and int(recorded) != count:
            problems.append(
                f"0071 manifest records {int(recorded):,} documents but documents.jsonl has "
                f"{count:,}; the corpus and its manifest disagree"
            )
    else:
        problems.append(f"0071 manifest not found next to documents.jsonl: {manifest}")
    return problems


def validate(output_dir: Path, expected: int) -> dict[str, Any]:
    """Exactly N files, schema intact, offsets sane, ICD dotted. Fails closed."""
    files = sorted(output_dir.glob("*.json"))
    issues: list[str] = []
    if len(files) != expected:
        issues.append(f"expected {expected} output files, found {len(files)}")
    entities = icd_codes = dotted = 0
    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            issues.append(f"{path.name}: top level is not a list")
            continue
        for row in rows:
            entities += 1
            missing = REQUIRED_KEYS - set(row)
            if missing:
                issues.append(f"{path.name}: entity missing {sorted(missing)}")
            position = row.get("position")
            if (
                not isinstance(position, list)
                or len(position) != 2
                or not all(isinstance(v, int) for v in position)
                or position[0] < 0
                or position[1] < position[0]
            ):
                issues.append(f"{path.name}: bad position {position!r}")
            if row.get("type") == "CHẨN_ĐOÁN":
                for code in row.get("candidates", []) or []:
                    icd_codes += 1
                    if not DOTTED.match(str(code)):
                        issues.append(f"{path.name}: ICD code not well formed: {code!r}")
                    elif "." in str(code):
                        dotted += 1
    return {
        "documents": len(files),
        "entities": entities,
        "icd_codes": icd_codes,
        "icd_dotted_form": dotted,
        "issues": issues[:50],
        "issue_count": len(issues),
        "ok": not issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=REPO / "runs/semantic_s1_0073_final_dotted")
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/pipeline/semantic_s1_0073.yaml"
    )
    parser.add_argument(
        "--mode",
        default="specialist",
        help="the scored baseline's mode; do not change without evidence",
    )
    parser.add_argument(
        "--semantic-model-root",
        default="",
        help=f"directory holding {S1_EMBEDDING_DIRNAME}/ and {S1_RERANKER_DIRNAME}/; "
        f"overrides the profile and ${MODEL_ROOT_ENV}",
    )
    parser.add_argument(
        "--semantic-doc-vectors",
        default="",
        help="frozen S1 dense index (S1_doc_vectors.pt). Reused in place, never copied "
        "or re-encoded.",
    )
    parser.add_argument(
        "--semantic-documents",
        default="",
        help="documents.jsonl the dense index was built over",
    )
    parser.add_argument("--expected-documents", type=int, default=EXPECTED_DOCUMENTS)
    args = parser.parse_args(argv)

    import os

    if args.semantic_model_root:
        os.environ[MODEL_ROOT_ENV] = args.semantic_model_root

    from mednorm_vi.inference.config import PipelineConfig, evaluate_readiness

    config = PipelineConfig.load(args.config)
    # Explicit CLI overrides win over the profile, so no user-specific Drive path is ever a
    # repository default and the artifacts can live wherever the owner actually put them.
    overrides = {
        "dense_index": args.semantic_doc_vectors,
        "documents": args.semantic_documents,
    }
    merged = {**config.semantic_linker, **{k: v for k, v in overrides.items() if v}}
    if args.semantic_model_root:
        merged["model_root"] = args.semantic_model_root
    config = dataclasses.replace(config, semantic_linker=merged)

    blockers = preflight(config)
    if blockers:
        print("BLOCKED - artifact preflight failed:", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 2
    readiness = evaluate_readiness(config, mode=args.mode)
    if readiness.errors:
        print("BLOCKED - readiness failed closed:", file=sys.stderr)
        for error in readiness.errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    args.run_dir.mkdir(parents=True, exist_ok=True)
    raw = args.run_dir / "output_raw_dotless"
    final = args.run_dir / "output"
    started = time.time()

    print(
        f"[0073] backend={config.linker_backend} mode={args.mode} input={args.input_dir}",
        flush=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mednorm_vi.inference.cli",
            "run",
            "--input-dir",
            str(args.input_dir),
            "--output-zip",
            str(args.run_dir / "raw.zip"),
            "--config",
            str(args.config),
            "--mode",
            args.mode,
            "--run-manifest",
            str(args.run_dir / "run-manifest.json"),
        ],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    if result.returncode != 0:
        print("BLOCKED - pipeline run failed", file=sys.stderr)
        return result.returncode
    produced = args.run_dir / "output"
    if produced.is_dir() and not raw.exists():
        produced.rename(raw)

    # Governed dotted serialization - the same step the scored baseline used.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mednorm_vi.inference.derive_submission",
            "--source-dir",
            str(raw),
            "--output-dir",
            str(final),
            "--derivation",
            "icd_dotted",
            "--expected-documents",
            str(args.expected_documents),
            "--validation-report",
            str(args.run_dir / "validation-report.json"),
        ],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=True,
    )

    report = validate(final, args.expected_documents)
    if not report["ok"]:
        print("BLOCKED - output validation failed:", file=sys.stderr)
        for issue in report["issues"]:
            print(f"  - {issue}", file=sys.stderr)
        return 3

    from mednorm_vi.inference.packaging import package_output_zip

    zip_path = package_output_zip(
        final, args.run_dir / "output.zip", expected_count=args.expected_documents
    )
    digest = sha256_file(Path(zip_path))
    (args.run_dir / "model-manifest.json").write_text(
        json.dumps(model_manifest(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.run_dir / "diagnostic-summary.json").write_text(
        json.dumps(
            {
                "linker_backend": config.linker_backend,
                "mode": args.mode,
                **report,
                "runtime_seconds": round(time.time() - started, 1),
                "output_zip_sha256": digest,
                "submitted": False,
                "contains_clinical_text": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[0073] documents {report['documents']} | entities {report['entities']} "
        f"| ICD codes {report['icd_codes']} ({report['icd_dotted_form']} dotted)"
    )
    print(f"[0073] output.zip {zip_path}")
    print(f"[0073] sha256     {digest}")
    print("[0073] NOT submitted - submit manually after reviewing the comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
