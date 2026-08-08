#!/usr/bin/env python3

"""Emit the non-launching Grounding DINO campaign preparation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .contract import build_preparation, read_json, validate_preparation
except ImportError:  # pragma: no cover - direct script execution
    from contract import build_preparation, read_json, validate_preparation


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "campaign.inputs.v1.json"
DEFAULT_OUTPUT = HERE / "campaign.preparation.v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate inputs and print readiness without writing an artifact",
    )
    arguments = parser.parse_args()

    preparation = build_preparation(
        experiment_dir=HERE,
        inputs=read_json(arguments.inputs),
    )
    validate_preparation(preparation)
    if not arguments.check_only:
        arguments.output.write_text(
            json.dumps(preparation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "campaign_id": preparation["campaign_id"],
                "jobs_submitted": 0,
                "launch_authorized": preparation["automatic_gate"][
                    "launch_authorized"
                ],
                "blocker_codes": [
                    item["code"]
                    for item in preparation["automatic_gate"]["blockers"]
                ],
                "official_ptm_count": preparation["official_ptm_inventory"][
                    "count"
                ],
                "preparation_sha256": preparation["preparation_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
