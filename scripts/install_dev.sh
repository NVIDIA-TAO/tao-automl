#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use the active Python environment and the core from this exact checkout.
set -euo pipefail
automl_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# The application's own metadata points at the in-tree core, so a single
# editable install resolves both projects without an independently published
# core distribution.
python -m pip install -e "${automl_repo_root}[dev]"
python -m pip check
