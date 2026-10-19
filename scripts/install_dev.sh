#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use the active Python environment and the core from this exact checkout.
set -euo pipefail
automl_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Supplying both projects in one resolution keeps the app's version constraint
# enforced without requiring an independently published core distribution.
python -m pip install "${automl_repo_root}/packages/automl-core" -e "${automl_repo_root}[dev]"
python -m pip check
