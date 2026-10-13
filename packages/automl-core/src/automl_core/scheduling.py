# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replayable budget allocation and checkpoint lineage, independent of workloads.

Adapted from the original Hyperband/ASHA/BOHB/DEHB/PBT/HyperBandES brains.
Every evaluated stage has a unique trial ID; previous results are immutable.
The controller currently executes one stage at a time, including ASHA.
"""

import math
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from automl_core.contracts import lookup, override
from automl_core.samplers import bohb, decode, dehb, encode, score
from automl_core.search import ask


def proposal(spec, point, **metadata):
    return {"point": list(point), "config": decode(spec, point), **metadata}


def fresh(spec, history, rng, resource, **metadata):
    successful = [t for t in history if t["state"] == "COMPLETE"]
    sampler = "random"
    if spec.algorithm == "bohb":
        point, sampler = bohb(spec, successful, rng)
    elif spec.algorithm == "dehb":
        population = [t for t in successful if t["resource"] == resource]
        point, sampler = dehb(spec, population, rng)
    else:
        point = rng.random(len(spec.parameters))
    return proposal(spec, point, resource=resource, sampler=sampler, **metadata)


def promote(parent, resource, **metadata):
    return {
        "point": parent["point"],
        "config": parent["config"],
        "resource": resource,
        "source_trial": parent["id"],
        "resume_mode": "promotion",
        **metadata,
    }


def curve_allows(spec, trial, history):
    """Source's power-law/R² guard, using only the candidate's evaluated rungs.

    Compare with full-budget peers only; failed or incomparable partial metrics
    cannot prune a trajectory. No speculative mid-step cancellation is implied.
    """
    chain, current = [], trial
    indexed = {t["id"]: t for t in history}
    while current:
        if current["state"] == "COMPLETE":
            chain.append((current["resource"], score(spec, current)))
        current = indexed.get(current.get("source_trial"))
    peers = [
        score(spec, t)
        for t in history
        if t["state"] == "COMPLETE" and t["resource"] == spec.fidelity.maximum
    ]
    if len(chain) < spec.options.min_curve_points or not peers:
        return True
    x, y = np.asarray(sorted(chain), dtype=float).T

    def power(steps, a, b, c):
        return a * steps**b + c

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            fit, _ = curve_fit(power, x, y, p0=[y[0] - y[-1], -0.5, y[-1]], maxfev=1000)
        total = np.sum((y - y.mean()) ** 2)
        confidence = 1 - np.sum((y - power(x, *fit)) ** 2) / total if total else 0
        predicted = float(power(spec.fidelity.maximum, *fit))
    except (ValueError, RuntimeError, FloatingPointError, OptimizeWarning):
        return True
    if not np.isfinite(predicted) or confidence < spec.options.curve_confidence:
        return True
    peer = max(peers)
    return predicted >= peer - abs(peer) * spec.options.early_stop_margin


def brackets(spec):
    rungs = spec.fidelity.rungs()
    eta = spec.fidelity.reduction_factor
    return [
        (rungs[-(s + 1) :], math.ceil(len(rungs) / (s + 1) * eta**s))
        for s in reversed(range(len(rungs)))
    ]


def hyperband(spec, history, rng):
    eta = spec.fidelity.reduction_factor
    for bracket, (rungs, count) in enumerate(brackets(spec)):
        group = [t for t in history if t.get("bracket") == bracket]
        first = [t for t in group if t.get("rung") == 0]
        if len(first) < count:
            return fresh(spec, history, rng, rungs[0], bracket=bracket, rung=0)
        for rung in range(1, len(rungs)):
            parents = [
                t for t in group if t["rung"] == rung - 1 and t["state"] == "COMPLETE"
            ]
            parents.sort(key=lambda t: score(spec, t), reverse=True)
            if spec.algorithm == "hyperband_es":
                parents = [t for t in parents if curve_allows(spec, t, history)]
            selected = parents[: max(1, count // eta**rung)]
            launched = {t.get("source_trial") for t in group if t["rung"] == rung}
            for parent in selected:
                if parent["id"] not in launched:
                    return promote(parent, rungs[rung], bracket=bracket, rung=rung)
    return None


def asha(spec, history, rng):
    rungs, eta = spec.fidelity.rungs(), spec.fidelity.reduction_factor
    # No synchronous bracket barrier: a completed rung can promote immediately.
    for rung in reversed(range(len(rungs) - 1)):
        observed = [
            t for t in history if t.get("rung") == rung and t["state"] == "COMPLETE"
        ]
        quota = len(observed) // eta
        observed.sort(key=lambda t: score(spec, t), reverse=True)
        launched = {t.get("source_trial") for t in history if t.get("rung") == rung + 1}
        if len(launched) < quota:
            for parent in observed[:quota]:
                if parent["id"] not in launched:
                    return promote(parent, rungs[rung + 1], rung=rung + 1)
    return fresh(spec, history, rng, rungs[0], rung=0)


def pbt(spec, history, rng):
    size = spec.options.population_size
    generation, member = divmod(len(history), size)
    resource = min(spec.fidelity.maximum, (generation + 1) * spec.fidelity.minimum)
    if generation * spec.fidelity.minimum >= spec.fidelity.maximum:
        return None
    if not generation:
        return fresh(spec, history, rng, resource, generation=0, member=member)
    previous = [
        t
        for t in history
        if t.get("generation") == generation - 1 and t["state"] == "COMPLETE"
    ]
    if not previous:
        return None
    previous.sort(key=lambda t: score(spec, t), reverse=True)
    own = next((t for t in previous if t["member"] == member), None)
    tail = max(1, math.ceil(size * spec.options.exploit_fraction))
    exploit = own is None or own in previous[-tail:]
    parent = previous[int(rng.integers(min(tail, len(previous))))] if exploit else own
    config = parent["config"]
    if exploit:
        values = {}
        for name, domain in spec.parameters.items():
            old = lookup(config, name)
            if domain.kind == "categorical":
                value = domain.choices[int(rng.integers(len(domain.choices)))]
            else:
                factor = spec.options.perturbation_factor ** int(rng.choice([-1, 1]))
                value = float(np.clip(old * factor, domain.low, domain.high))
                value = round(value) if domain.kind == "int" else value
            values[name] = value
        config = override(config, values)
    return {
        **promote(parent, resource, generation=generation, member=member),
        "point": encode(spec, config),
        "config": config,
        "resume_mode": "pbt",
        "exploited": exploit,
    }


def next_trial(spec, history):
    rng = np.random.default_rng(np.random.SeedSequence([spec.seed, len(history)]))
    if spec.algorithm in {"hyperband", "hyperband_es", "bohb", "dehb"}:
        return hyperband(spec, history, rng)
    if spec.algorithm == "asha":
        return asha(spec, history, rng)
    if spec.algorithm == "pbt":
        return pbt(spec, history, rng)
    if spec.algorithm in {"random", "bayesian", "bfbo"}:
        point, config = ask(spec, history)
        return {"point": point, "config": config}
    raise ValueError("LLM proposals require the journaled controller path")


def review(spec):
    result = {
        "algorithm": spec.algorithm,
        "execution_concurrency": 1,
        "trial_budget_counts": "each evaluated stage, including promotions and population members",
    }
    if spec.fidelity:
        result["fidelity"] = spec.fidelity.model_dump()
        result["best_requires_resource"] = spec.fidelity.maximum
    if spec.algorithm in {"hyperband", "hyperband_es", "bohb", "dehb"}:
        result["brackets"] = [
            {"resources": rungs, "initial_candidates": count}
            for rungs, count in brackets(spec)
        ]
        result["full_schedule_stage_cap"] = sum(
            sum(
                max(1, count // spec.fidelity.reduction_factor**i)
                for i in range(len(rungs))
            )
            for rungs, count in brackets(spec)
        )
    if spec.algorithm == "pbt":
        result["full_schedule_stage_cap"] = spec.options.population_size * math.ceil(
            spec.fidelity.maximum / spec.fidelity.minimum
        )
    if spec.llm:
        result["llm"] = spec.llm.model_dump()
        result["data_sent"] = (
            "approved parameter domains, parameter values, scalar validation metrics, statuses and research instructions; no datasets or logs"
        )
    return result
