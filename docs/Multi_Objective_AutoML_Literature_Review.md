# Multi-Objective AutoML Literature Review

## Summary

Multi-objective AutoML is usually framed as finding a set of non-dominated
solutions rather than a single universally best model. For TAO AutoML, the
immediate deployment-relevant objective is to trade task quality
(`loss`, `accuracy`, `mAP`, etc.) against inference latency. The implementation
uses a shared measured archive and separates search acquisition from final
deployment selection:

- Store raw objective values for every recommendation.
- Reject failed, missing, boolean, NaN, and infinite measurements.
- Derive latency feasibility from the measured accuracy winner.
- Maintain non-dominated ranks and explicit dominated-by relationships.
- Normalize accuracy and latency regret on the feasible rank-zero front.
- Select a final compromise with augmented Chebyshev regret.
- Keep normalized acquisition utility separate from the final selector.

This preserves the current Bayesian, Hyperband, BOHB, ASHA, DEHB, PBT, and
runner flows while guaranteeing that a dominated candidate cannot be returned
as the final two-objective compromise.

## Core Concepts

Pareto dominance is the central abstraction: a configuration dominates another
when it is no worse on every objective and strictly better on at least one.
The returned Pareto front approximates the trade-off curve, for example:

- higher accuracy with higher latency,
- lower latency with slightly lower accuracy,
- lower loss with higher runtime cost.

Scalarization remains useful when existing algorithms need a total ordering,
but raw accuracy and latency are never added. The acquisition utility is
archive-normalized and is recomputed as the archive changes. Final selection
is always performed independently from acquisition and only after feasibility
and Pareto filtering.

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

## Methods Evaluated for the Current Selector

The current problem is a small, already measured, two-objective archive rather
than a large evolutionary population. The following methods were evaluated
against that structure:

| Method | Strength | Limitation here | Decision |
|---|---|---|---|
| Raw weighted sum | Simple total ordering | Unit- and scale-dependent; can miss non-convex trade-offs; the old implementation mixed mAP and milliseconds | Rejected |
| Pareto nondominance | Removes objectively inferior choices without preferences | Produces a set rather than one deployment choice | Required first-stage filter |
| Epsilon constraint | Directly expresses “fastest subject to retained accuracy” | Requires a declared, configurable accuracy rule | Selected for latency feasibility |
| Normalized ideal-point distance | Scale-independent and intuitive | Fully compensatory: a large regret on one objective can be offset by the other | Retained as a deterministic tie-break |
| Knee point | Can identify a high marginal-trade-off point | Sparse or nearly linear fronts may have no stable geometric knee; sensitive to endpoints and noise | Rejected as the default |
| Augmented Chebyshev / achievement scalarization | Minimizes the worst normalized regret and supports non-convex fronts; augmentation makes the ordering strict | Requires explicit weights and normalization bounds | Selected for final compromise |
| Hypervolume contribution | Valuable for Pareto-set acquisition and archive diversity | Final-point rankings depend on a reference point and often favor endpoints; disproportionate machinery for this archive | Rejected for final selection |
| NSGA-II rank and crowding | Strong population-based search and diversity preservation | An evolutionary population is unnecessary for three parameters and 30 expensive measured trials | Non-dominated ranking adopted; evolutionary search rejected |
| qEHVI / qNEHVI | Principled parallel or noisy multi-objective Bayesian acquisition | Adds surrogate and reference-point complexity and does not itself define the final deployment point | Deferred to a larger-budget acquisition upgrade |

The epsilon-constraint choice follows the established formulation of optimizing
one objective subject to bounds on the others
(Mavrotas, https://doi.org/10.1016/j.amc.2009.03.037). The selected compromise
is an achievement-style scalarizing function in the sense of Wierzbicki
(https://pure.iiasa.ac.at/id/eprint/12466/). Knee behavior and its ambiguity are
discussed by Deb and Gupta
(https://doi.org/10.1080/0305215X.2010.548863). For larger future searches,
BoTorch documents qEHVI/qNEHVI and Chebyshev-based qNParEGO as multi-objective
acquisition options
(https://botorch.org/docs/v0.16.0/multi_objective).

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
    "algorithm": "bayesian",
    "selection_mode": "multi_objective",
    "objectives": [
        {"metric": "val_mAP50", "direction": "maximize", "weight": 1.0},
        {"metric": "latency_ms", "direction": "minimize", "weight": 1.0},
    ],
    "latency_accuracy_retention": {
        "type": "relative",
        "retained_fraction": 0.98,
        "reference": "accuracy_winner",
    },
    # Optional and independent of latency_accuracy_retention. None includes
    # every valid candidate in multi-objective Pareto analysis.
    "multi_objective_min_accuracy": None,
    "objective_normalization": "pareto_front",
    "augmentation_rho": 1.0e-6,
    "accuracy_tolerance": 1.0e-12,
    "latency_tolerance": 0.05,
    "random_seed": 314159,
    "require_eval_fn_success": True,
}
```

An absolute accuracy rule is also supported:

```python
latency_accuracy_retention = {
    "type": "absolute",
    "max_absolute_degradation": 0.02,
    "reference": "accuracy_winner",
}
```

For accuracy \(A\) and latency \(L\), the default latency-mode relative rule is:

```text
A(x) >= 0.98 * A*
```

where \(A^*\) is the accuracy-mode winner. Latency mode minimizes stabilized
latency subject to this constraint. It returns an explicit
`no_accuracy_feasible_candidates` status instead of applying a penalty or
silently falling back.

Multi-objective eligibility is separate. By default,
`multi_objective_min_accuracy` is unset and every candidate with finite valid
measurements is eligible for Pareto analysis. An optional absolute metric floor
can be written as a number or explicitly:

```python
multi_objective_min_accuracy = {
    "type": "absolute",
    "value": 0.80,
}
```

A reference-relative sensitivity policy must be explicit, avoiding ambiguity
between an absolute `0.90` metric floor and 90% retention:

```python
multi_objective_min_accuracy = {
    "type": "relative",
    "value": 0.90,
    "reference": "accuracy_winner",
}
```

This resolves to \(A(x) \mathrel{\ge} 0.90 A^*\), independently of the
latency-mode retention fraction. The resolved reference candidate, reference
accuracy, and threshold are persisted in the selection audit.

The compromise selector orients both objectives as regrets:

```text
r_accuracy(x) = (A_max - A(x)) / (A_max - A_min)
r_latency(x)  = (L(x) - L_min) / (L_max - L_min)
```

The bounds are persisted from the rank-zero front under the configured
multi-objective eligibility policy. A zero-range objective is inactive and
contributes zero regret. With normalized positive weights \(w_A,w_L\), the
selected candidate minimizes:

```text
max(w_A * r_accuracy, w_L * r_latency)
  + rho * (w_A * r_accuracy + w_L * r_latency)
```

Finite dominated outliers cannot affect these bounds because they are removed
before front normalization. A finite nondominated endpoint is treated as a
real measured trade-off and therefore does define an ideal or nadir bound; it
is not silently clipped. Measurement-level latency outliers are controlled
separately by the median/MAD protocol and quality gates. Product deployments
that require externally stable normalization across changing archives may
configure fixed domain bounds in a future extension, but must record those
bounds rather than deriving them after seeing a desired winner.

The selector first removes dominated points. Score ties use normalized ideal
distance, balance gap, accuracy-safe regret, canonical SHA-256 configuration
fingerprint, and candidate ID. Exact duplicate objective points are represented
by the candidate with the smallest canonical fingerprint, with all aliases
retained in the audit record.

Accuracy mode selects maximum valid accuracy. Only candidates equivalent within
`accuracy_tolerance` may use latency as a tie-break. Latency medians whose
configured tolerance or confidence intervals overlap are treated as
statistically tied; higher accuracy and then the canonical fingerprint decide.
For Pareto dominance, "no worse" always follows the observed directions
exactly: accuracy cannot decrease and median latency cannot increase.
Tolerance controls whether an accuracy gain is strict, while non-overlapping
latency confidence intervals are required for a latency-only strict
improvement. Thus statistical equivalence can withhold a dominance claim but
can never allow a numerically worse point to dominate.

If no distinct non-dominated point exists between the accuracy and latency
extremes under the configured multi-objective eligibility policy, the result
contains:

```text
No distinct Pareto compromise exists under the configured multi-objective eligibility policy.
```

The augmented-Chebyshev ordering then supplies the deterministic extreme-point
fallback. The result does not claim that this fallback is a middle ground.

`get_status()` and runner results expose the constraint reference and threshold,
normalization bounds, per-candidate validity, latency feasibility,
multi-objective eligibility, both global and multi-objective Pareto ranks,
dominated-by IDs, normalized regrets, compromise and acquisition scores,
confidence bounds, canonical fingerprint, tie-break values, and all three
mode-winner flags.

For two-objective benchmarking, `eval_fn` is required by default. An exception
or missing required metric invalidates the trial instead of falling back to a
training progress-bar rate. Boolean, NaN, infinite, non-positive latency, and
malformed or incomplete confidence-interval values are also rejected before
Pareto ranking.
