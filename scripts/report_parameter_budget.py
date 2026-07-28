#!/usr/bin/env python3
"""Report the candidate inventory and gate a deployment manifest at 9B.

Usage::

    env PYTHONPATH=src python scripts/report_parameter_budget.py \
        --registry configs/models/candidate_model_registry.yaml \
        --deployment configs/models/deployment_budget_template.yaml

The candidate total is reported but never gated. The deployment total is gated
and fails closed on any unverified selected component (spec §17).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mednorm_vi.governance.parameter_budget import (  # noqa: E402
    ParameterBudgetError,
    compute_deployment_budget,
    iter_unverified,
    load_candidate_registry,
    load_deployment_selection,
    render_budget_report,
)

DEFAULT_REGISTRY = "configs/models/candidate_model_registry.yaml"
DEFAULT_DEPLOYMENT = "configs/models/deployment_budget_template.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedNorm-VI parameter budget report")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)

    registry = load_candidate_registry(args.registry)
    summary = registry.summary()
    manifest_name, selected = load_deployment_selection(args.deployment)

    try:
        report = compute_deployment_budget(
            registry, selected, manifest_name=manifest_name, enforce=True)
    except ParameterBudgetError as error:
        print(json.dumps({"candidate_inventory": summary,
                          "deployment_gate": "FAILED", "detail": str(error)},
                         indent=2, sort_keys=True))
        return 1

    if args.json:
        print(json.dumps({"candidate_inventory": summary,
                          "deployment_budget": report.as_dict()},
                         indent=2, sort_keys=True))
        return 0

    print("=== candidate inventory (never gated) ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    unverified = [c.component_id for c in iter_unverified(registry)]
    print(f"\nunverified / planned components ({len(unverified)}): "
          f"{', '.join(unverified) or 'none'}")
    print("\n=== deployment budget (gated at 9B) ===")
    print(render_budget_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
