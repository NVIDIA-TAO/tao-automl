# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seeded ask/tell search; failed observations never become synthetic scores."""

import numpy as np

from automl_core.bayesian import (
    make_gp,
    negative_expected_improvement,
    optimize_acquisition,
)
from automl_core.contracts import canonical, override


def features(spec, point):
    """One-hot nominal choices; numeric observations use actual quantized values."""
    result = []
    for name, unit in zip(sorted(spec.parameters), point):
        parameter = spec.parameters[name]
        value = parameter.decode(float(unit))
        if parameter.kind == "categorical":
            result.extend(
                float(canonical(value) == canonical(choice))
                for choice in parameter.choices
            )
        elif parameter.log:
            result.append(
                (np.log(value) - np.log(parameter.low))
                / (np.log(parameter.high) - np.log(parameter.low))
            )
        else:
            result.append((value - parameter.low) / (parameter.high - parameter.low))
    return result


def ask(spec, history):
    # Per-trial seed makes replay independent of process hash seeds and GP restarts.
    rng = np.random.default_rng(np.random.SeedSequence([spec.seed, len(history)]))
    successful = [trial for trial in history if trial["state"] == "COMPLETE"]
    dimensions = len(spec.parameters)
    if spec.algorithm == "bfbo" and successful:
        from automl_core.samplers import bfbo

        point = bfbo(spec, successful, rng)
    elif spec.algorithm == "bayesian" and len(successful) >= 3:
        samples = np.asarray([features(spec, trial["point"]) for trial in successful])
        sign = 1 if spec.objective.direction == "maximize" else -1
        scores = sign * np.asarray([trial["result"]["metric"] for trial in successful])
        gp = make_gp(samples.shape[1], spec.seed)
        gp.fit(samples, scores)
        point = optimize_acquisition(
            lambda x: negative_expected_improvement(gp, samples, features(spec, x)),
            dimensions,
            rng.random,
        )
    else:
        point = rng.random(dimensions)
    values = {
        name: spec.parameters[name].decode(float(value))
        for name, value in zip(sorted(spec.parameters), point)
    }
    return point.tolist(), override(spec.base_config, values)


def tell(trial, state, result=None):
    if state not in {"COMPLETE", "ERROR", "CANCELED"}:
        raise ValueError("Only terminal observations can be reported")
    if (state == "COMPLETE") != (result is not None):
        raise ValueError(
            "Successful observations require a validated result; failures must not supply one"
        )
    payload = result.model_dump() if result else None
    if trial["state"] in {"COMPLETE", "ERROR", "CANCELED"}:
        if trial["state"] != state or trial.get("result") != payload:
            raise ValueError("Conflicting duplicate result")
        return
    trial.update(state=state, result=payload)


def best(spec, history):
    successful = [trial for trial in history if trial["state"] == "COMPLETE"]
    if spec.fidelity:
        successful = [
            t for t in successful if t.get("resource") == spec.fidelity.maximum
        ]
    if not successful:
        return None
    return sorted(
        successful,
        key=lambda trial: trial["result"]["metric"],
        reverse=spec.objective.direction == "maximize",
    )[0]
