# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny CPU demonstration adapted from the existing virtualenv random-forest example."""

import argparse
import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
x, y = load_iris(return_X_y=True)
train_x, val_x, train_y, val_y = train_test_split(
    x, y, test_size=0.25, stratify=y, random_state=11
)
model = RandomForestClassifier(**config["model"]).fit(train_x, train_y)
(args.output / "metrics.json").write_text(
    json.dumps({"accuracy": accuracy_score(val_y, model.predict(val_x))})
)
