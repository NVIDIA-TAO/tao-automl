# Multi-Objective AutoML Literature Review

## Summary

Multi-objective AutoML is usually framed as finding a set of non-dominated
solutions rather than a single universally best model. For TAO AutoML, the
immediate deployment-relevant objective is to trade task quality
(`loss`, `accuracy`, `mAP`, etc.) against inference latency. The implemented
feature follows a pragmatic hybrid pattern:

- Store raw objective values for every recommendation.
- Maintain and expose the Pareto front.
- Use a weighted scalar score so the existing single-scalar search algorithms
  can still rank, promote, and resume configurations.
- Treat latency as a minimization objective, with optional `scale` and `weight`
  controls to avoid unit mismatch with accuracy/loss.

This gives users a Pareto archive immediately while preserving the current
Bayesian, Hyperband, BOHB, ASHA, DEHB, PBT, and runner flows.

## Core Concepts

Pareto dominance is the central abstraction: a configuration dominates another
when it is no worse on every objective and strictly better on at least one.
The returned Pareto front approximates the trade-off curve, for example:

- higher accuracy with higher latency,
- lower latency with slightly lower accuracy,
- lower loss with higher runtime cost.

Scalarization remains useful when existing algorithms need a total ordering.
The common approaches are weighted sums, constrained optimization, and
preference-aware scalarizations. Weighted sums are simple and compatible with
legacy code, but they require sensible weights/scales. Pareto methods preserve
the trade-off set without forcing one preference too early.

## Representative Literature

### Evolutionary Pareto Search

Deb et al.'s NSGA-II is a foundation for evolutionary multi-objective
optimization: it uses fast non-dominated sorting and elitism to preserve good
fronts across generations.

LEMONADE applies this idea to neural architecture search. Elsken, Metzen, and
Hutter propose a Lamarckian evolutionary NAS algorithm that approximates an
entire Pareto front under objectives such as predictive performance and model
size. The important AutoML lesson is that resource objectives should be modeled
directly instead of bolted on after a single-objective search.

References:

- NSGA-II: https://research.birmingham.ac.uk/en/publications/a-fast-and-elitist-multi-objective-genetic-algorithm-nsga-ii
- LEMONADE: https://arxiv.org/abs/1804.09081

### Multi-Objective Bayesian Optimization

ParEGO showed that expensive multi-objective black-box optimization can be
handled by repeatedly scalarizing objectives and fitting a surrogate. Later
MOBO work focuses on Pareto-set information gain and hypervolume improvement.
PESMO targets the Pareto set directly with predictive entropy search. qEHVI and
qNEHVI improve expected hypervolume improvement for parallel and noisy
settings, which are common in AutoML because training metrics are stochastic and
jobs run in batches.

References:

- ParEGO: https://staff.cs.manchester.ac.uk/~jknowles/parego/
- PESMO: https://proceedings.mlr.press/v48/hernandez-lobatoa16.html
- qEHVI: https://arxiv.org/abs/2006.05078
- qNEHVI: https://arxiv.org/abs/2105.08195

### Multi-Fidelity AutoML and HPO

Hyperband and BOHB are not multi-objective by themselves, but they are central
to efficient AutoML because they spend small budgets on many configurations and
promote promising ones. Multi-objective extensions replace a scalar promotion
criterion with non-dominated sorting or multi-objective surrogate logic.

Chen et al.'s multiobjective multi-fidelity BOHB variant argues that
multi-objective HPO is harder because each objective has its own surrogate, and
low-fidelity observations can be useful when integrated carefully.

Salinas et al. extend Hyperband-style search to jointly tune hardware and
hyperparameters, using non-dominated sorting for early stopping and transfer
learning to estimate Pareto fronts. This is especially relevant to TAO because
latency depends on deployment hardware, not just model structure.

References:

- Hyperband: https://www.jmlr.org/beta/papers/v18/16-558.html
- BOHB: https://proceedings.mlr.press/v80/falkner18a.html
- Multiobjective multi-fidelity BOHB: https://link.springer.com/chapter/10.1007/978-3-031-14714-2_12
- Joint hardware and hyperparameter tuning: https://arxiv.org/abs/2106.05680

### Hardware-Aware NAS and Latency

MnasNet is one of the clearest examples of adding latency directly to AutoML.
It optimizes accuracy-latency trade-offs and measures real device latency
instead of relying only on proxies such as FLOPs.

HURRICANE and related hardware-aware NAS systems further show that architectures
with the same FLOPs can have different latencies on different accelerators.
The practical implication is that TAO AutoML should accept measured latency
from the target evaluation path or `eval_fn`, not assume a universal model-side
proxy.

HW-PR-NAS emphasizes Pareto-rank-preserving surrogate models for
hardware-aware NAS, optimizing accuracy, power, and performance budgets on edge
platforms. This supports keeping a Pareto front in TAO AutoML rather than
returning only one scalar winner.

References:

- MnasNet: https://arxiv.org/abs/1807.11626
- HURRICANE: https://arxiv.org/abs/1910.11609
- HW-PR-NAS: https://research.ibm.com/publications/multi-objective-hardware-aware-neural-architecture-search-with-pareto-rank-preserving-surrogate-models

## Implementation Guidance for TAO AutoML

Latency should be a measured objective. The runner now extracts latency-like
metrics from logs and status artifacts using aliases such as `latency`,
`latency_ms`, `inference_latency_ms`, `avg_latency`, and `runtime_ms`. For
more reliable deployment latency, pass an `eval_fn` that runs an inference or
benchmark job on the intended target hardware and returns:

```python
{"val_mAP": 0.72, "latency": 18.4}
```

Recommended settings:

```python
automl_settings = {
    "algorithm": "hyperband",
    "metric": "val_mAP",
    "direction": "maximize",
    "multi_objective": True,
    "latency_metric": "latency",
    "latency_direction": "minimize",
    "latency_scale": 100.0,
    "latency_weight": 1.0,
}
```

For explicit objective lists:

```python
automl_settings = {
    "algorithm": "bayesian",
    "objectives": [
        {"metric": "accuracy", "direction": "maximize", "weight": 1.0},
        {"metric": "latency", "direction": "minimize", "weight": 1.0, "scale": 100.0},
    ],
}
```

The scalar score used internally is:

```text
sum(weight * value / scale) for maximize objectives
sum(-weight * value / scale) for minimize objectives
```

The Pareto front is exposed in `get_status()` and runner results, so downstream
selection can choose a model based on the desired latency-quality operating
point even when the scalar score selects a single default best recommendation.
