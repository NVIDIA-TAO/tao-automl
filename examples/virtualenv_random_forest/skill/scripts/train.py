# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train and evaluate a small public Random Forest model on Iris."""

import argparse
import json
import sys
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    features, labels = load_iris(return_X_y=True)
    train_x, eval_x, train_y, eval_y = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=11,
        stratify=labels,
    )
    model = RandomForestClassifier(**config["model"])
    model.fit(train_x, train_y)
    accuracy = accuracy_score(eval_y, model.predict(eval_x))

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "python_prefix": sys.prefix,
                "model": config["model"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"accuracy: {accuracy:.8f}", flush=True)


if __name__ == "__main__":
    main()
