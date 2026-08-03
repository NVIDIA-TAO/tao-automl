# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the direct-virtualenv AutoML smoke example."""

import argparse
import json
from pathlib import Path

from tao_automl.runner import AutoMLRunner
from tao_sdk.platforms.virtualenv import VirtualEnvSDK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--recommendations", type=int, default=2)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    sdk = VirtualEnvSDK(
        venv_path=args.venv_path,
        work_dir=work_dir / "jobs",
        state_file=work_dir / "jobs.db",
    )
    runner = AutoMLRunner(
        sdk=sdk,
        skill_dir=Path(__file__).parent / "skill",
        action="train",
        poll_interval=0.1,
    )
    result = runner.run(
        automl_settings={
            "algorithm": "bayesian",
            "metric": "accuracy",
            "direction": "maximize",
            "automl_max_recommendations": args.recommendations,
            "run_baseline": False,
            "run_final_evaluation": False,
            "session_id": "virtualenv-random-forest-smoke",
        },
        automl_hyperparameters=["model.n_estimators", "model.max_depth"],
        workspace_path=str(work_dir / "automl"),
        gpu_count=0,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
