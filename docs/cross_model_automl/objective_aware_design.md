# Objective-aware recommendation design

Status: implemented recommendation design; DINO and cross-model validation
remain pending. This document was written before the new independent DINO jobs
and before any cross-model campaign.

## Separation of responsibilities

Search acquisition and terminal deployment selection are deliberately
different:

* acquisition decides which candidate to evaluate next;
* the terminal selector chooses a winner from measured, valid candidates;
* matched post-selection measurements validate the frozen winner but never
  replace selection-time objectives.

The existing accuracy, constrained-latency, and Pareto-safe terminal selectors
remain authoritative. This change makes recommendation behavior mode-specific
instead of relying only on terminal archive selection.

## Accuracy acquisition

Accuracy mode models the configured predictive-quality metric directly. For a
maximized metric \(A(x)\), Bayesian search maximizes expected improvement:

\[
\operatorname{EI}_A(x)
=
\mathbb{E}\left[\max(A(x)-A^*-\xi, 0)\right].
\]

The reported accuracy remains on its original scale. Latency does not
influence recommendations or terminal selection except as the already
documented deterministic terminal tie-break within the numerical accuracy
tolerance.

Accuracy mode uses the same deterministic low-discrepancy calibration design
as the other native modes before fitting its response surface. Sharing the
initial-design rule keeps equal-dimensional independent jobs comparable; the
response supplied after calibration remains accuracy alone.

## Self-calibrating constrained-latency acquisition

Latency mode does not require a user-provided baseline.

### Calibration

The job first evaluates a deterministic low-discrepancy initial design. The
default design size is:

\[
n_\text{cal} = \min(12,\max(4,2d)),
\]

where \(d\) is the inner PTM-arm search dimension. The value is configurable
and is frozen in the campaign manifest.

No quality constraint steers recommendations during this initial design.
After calibration, the reference is the best valid accuracy observed within
the same job:

\[
A^*_t = \max_{i \leq t} A(x_i).
\]

Because it is a running maximum, the reference is monotonic and cannot
oscillate downward. A relative retained-quality policy resolves:

\[
T_t = r A^*_t,\qquad 0 < r \leq 1.
\]

Relative retention is not considered calibrated until a positive, task-valid
reference exists. If calibration produces only invalid, zero, or negative
quality, the job remains in quality-discovery mode and the metric-sanity gate
must classify the run; a zero reference is not silently treated as proof that
degenerate candidates are viable.

### Constrained expected improvement

Accuracy and latency are fitted as separate response surfaces. Let
\(\mu_A,\sigma_A\) and \(\mu_L,\sigma_L\) be their predictions. The probability
of satisfying the current internally derived quality constraint is:

\[
p_f(x) = \Pr(A(x) \geq T_t).
\]

Once at least one feasible latency observation exists, acquisition maximizes:

\[
\operatorname{CEI}_L(x)
=
\operatorname{EI}_{\min L}(x)\,p_f(x).
\]

Before a feasible latency incumbent exists, it maximizes \(p_f(x)\) rather
than exploiting a structurally fast but predicted-infeasible region.

Terminal latency selection is still resolved once against the complete final
archive: derive the terminal accuracy winner, apply the configured retention
policy, anchor the equivalent-fastest cohort at the raw minimum stabilized
latency, and use accuracy only inside that cohort.

## Multi-objective acquisition

### Approaches considered

| Approach | Strength | Limitation in this repository | Decision |
| --- | --- | --- | --- |
| EHVI / noisy EHVI | Direct expected Pareto-volume gain | Robust implementations require multi-output posterior integration and normally BoTorch/GPyTorch, which are not current dependencies | Revisit if those dependencies become acceptable |
| Fixed weighted sum | Simple | Misses non-convex fronts, depends on one fixed preference, and can repeatedly target one endpoint | Rejected |
| Fixed augmented Chebyshev | Scale independent and handles non-convex fronts | One fixed weight can repeatedly coincide with an endpoint | Retained for terminal compromise, not sufficient alone for acquisition |
| Pareto-rank scalar proxy | Uses both objectives and existing selector utilities | Discontinuous and archive-relative; one surrogate does not model objective surfaces | Backward-compatible fallback only |
| ParEGO | Established MO Bayesian method; low dependency cost; covers non-convex fronts through changing scalarizations | Sequential and approximate rather than direct hypervolume maximization | Selected for Bayesian acquisition |
| Pareto-aware successive halving | Natural for multi-fidelity algorithms | New configurations may remain random and PTM conditionals still need an outer design | Supported fallback where explicitly reported |

### Deterministic ParEGO

For every iteration, objective values are normalized to regret using extrema
of the observed rank-zero front:

\[
r_A(x) = \frac{A_\max-A(x)}{A_\max-A_\min},\qquad
r_L(x) = \frac{L(x)-L_\min}{L_\max-L_\min}.
\]

If the observed front has zero span in one dimension, the ideal remains
front-derived but the nadir for only that collapsed dimension falls back to
the complete valid archive. This prevents a one-point front from assigning
zero regret to an objectively worse dominated observation. If the complete
archive is also identical in that dimension, it contributes zero regret and
never divides by zero. The chosen bound source is recorded in every
acquisition audit. Invalid or failed observations never enter normalization.

The two-objective weights follow a deterministic van-der-Corput sequence:

```text
(0.5, 0.5), (0.25, 0.75), (0.75, 0.25), (0.125, 0.875), ...
```

For weight vector \(w_t\), the maximize-oriented response is:

\[
u_t(x) =
-\left[
\max\{w_{A,t}r_A(x),w_{L,t}r_L(x)\}
+\rho(w_{A,t}r_A(x)+w_{L,t}r_L(x))
\right].
\]

A scalar GP is refit to all raw observations transformed using the current
algorithmic weight, and the next point maximizes EI on \(u_t\). Varying the
weight is part of the frozen algorithm; it is never changed after inspecting
outcomes.

The terminal multi-objective selector remains independently eligible,
Pareto-rank-zero, normalized augmented Chebyshev with deterministic
tie-breaking. It may select an endpoint when the measured geometry warrants
one, and reports that fact.

## PTM identity

PTM identity is not encoded as an ordinal scalar in the GP. The initial
production design is hierarchical:

1. registry preflight resolves all task- and TAO-compatible PTMs;
2. every PTM receives a deterministic equal-fidelity initial quota;
3. its conditional hyperparameter space is optimized by an independent inner
   objective-aware search;
4. subsequent PTM-arm budget is allocated by an algorithmic outer policy;
5. terminal selection compares only task-correct, fidelity-comparable
   objectives.

This avoids false distances between categorical checkpoints and avoids
feeding inactive checkpoint-specific parameters into a fixed-dimensional
kernel. A joint mixed-variable GP remains a future option after one-hot
categorical kernels and explicit inactive-parameter masks are implemented.

## Edge behavior

* Missing, failed, boolean, NaN, or infinite objectives are excluded.
* A failed placeholder `0.0` cannot be promoted over a valid negative
  multi-objective utility.
* Identical objective ranges normalize to zero without division.
* With no feasible latency observation, acquisition seeks feasibility.
* With no positive task-valid quality reference, latency remains in
  quality-discovery mode and the campaign cannot pass its sanity gate.
* With a one-point multi-objective archive, calibration continues rather than
  claiming learned Pareto geometry.
* Candidate enumeration order does not choose terminal winners.
* Search recommendation order is intentionally observation-dependent and is
  reproduced from the frozen initial design, seed, ordered observation ledger,
  and persisted acquisition/RNG state.

## Product-claim boundary

This design supports:

> AutoML uses mode-specific objective-aware search. Accuracy mode targets
> predictive quality, latency mode targets deployment latency under a
> self-calibrated quality constraint, and multi-objective mode explores and
> selects from the accuracy–latency Pareto frontier. When a meaningful
> interior Pareto solution exists and is discovered within the search budget,
> multi-objective mode selects a balanced compromise. When no such point
> exists, AutoML reports the observed geometry rather than manufacturing a
> tradeoff.

It does not guarantee that every search space, dataset, seed, budget, or model
contains or discovers a candidate distinct from both endpoints.
