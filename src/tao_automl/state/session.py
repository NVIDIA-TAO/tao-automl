# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve experiment identity without creating or overwriting resume state."""
import json
from pathlib import Path
import re
import uuid


_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def resolve_session_id(workspace, requested=None, *, resume=False):
    """Select one durable controller on resume; never generate a new identity."""
    if requested is not None and (
        not isinstance(requested, str) or not _SESSION_ID.fullmatch(requested)
    ):
        raise ValueError("session_id must be a non-empty filename-safe string (max 128 characters)")
    if not resume:
        return requested if requested is not None else uuid.uuid4().hex[:12]

    directory = Path(workspace) / ".automl" / "controller"
    if requested is None:
        candidates = sorted(directory.glob("*.json"))
        if len(candidates) != 1:
            identities = ", ".join(path.stem for path in candidates) or "none"
            raise ValueError(
                f"Resume requires exactly one controller in {directory}; found {identities}. "
                "Use the full run workspace and set session_id to the intended existing controller."
            )
        requested = candidates[0].stem
        if not _SESSION_ID.fullmatch(requested):
            raise ValueError(f"Resume controller has an invalid session_id: {requested!r}")
    selected = directory / f"{requested}.json"
    try:
        state = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Resume controller is missing or unreadable: {selected}") from exc
    if not isinstance(state, list) or any(not isinstance(rec, dict) or "id" not in rec for rec in state):
        raise ValueError(f"Resume controller must contain a list of recommendation records: {selected}")
    return requested
