# DINO latency-mode tied-cohort review

Review baseline: `b7160e0a9ef38a48ee95d44eb6510b818fdb00c4`.

Scope: `src/tao_automl/selection.py` and the frozen expanded DINO
selection evidence. This review did not invoke selection on matched-validation
measurements and did not change a winner, objective, threshold, tolerance,
multi-objective weight, or selection algorithm.

## Result

No latency tied-cohort correctness defect was found.

The production path has the required order:

1. `analyze_archive` chooses the highest-accuracy valid candidate.
2. The configured accuracy-winner-relative retention policy resolves its
   reference and threshold.
3. Only candidates satisfying that threshold enter `latency_feasible`.
4. `_choose_latency` anchors the cohort at the raw minimum stabilized latency.
5. A feasible candidate enters the tied cohort when either
   \(L_i-L_{\min}\leq\epsilon_L\) or its reported latency interval overlaps the
   raw-minimum anchor's interval.
6. The winner is the highest-accuracy member of that cohort.
7. Equal-accuracy ties use the canonical configuration fingerprint and then
   candidate ID.

`_build_audits` canonicalizes candidates by fingerprint and candidate ID before
mode selection. The final keys are complete, so candidate enumeration order
does not affect the winner. `_choose_latency` does not consume normalized
multi-objective regrets, compromise scores, or multi-objective weights.

## Frozen DINO evidence

The relative-retention reference is `seed_271828_rec_18` at
`mAP50=0.6554138278683255`. With retention `0.98`, the frozen threshold is
`0.6423055513109589`.

| Candidate | mAP50 | Selection-time median (ms) | Delta from raw minimum (ms) |
| --- | ---: | ---: | ---: |
| `seed_271828_rec_18` | 0.6554138278683255 | 66.23099475 | 0 |
| `seed_161803_rec_14` | 0.6503731411565659 | 66.58940950 | 0.35841475 |
| `seed_314159_rec_12` | 0.6517250365478822 | 66.68512100 | 0.45412625 |
| `seed_271828_rec_16` | 0.6544218576499151 | 66.82186425 | 0.59086950 |

All four deltas are within the frozen `0.73553775 ms` practical tolerance.
The three non-anchor selection-time confidence intervals do not overlap the
anchor interval, so the historical cohort was formed by the absolute tolerance
branch, not by confidence-interval overlap. The higher-accuracy tie-break then
selected `seed_271828_rec_18`.

## Semantics clarification

The historical serialized reason says “statistically equivalent latencies.”
That phrase is too narrow because the configured practical-tolerance branch can
form the cohort without statistical interval overlap. The accurate product
semantics are:

> Select the fastest accuracy-feasible candidate when latency differences are
> meaningful. When candidates are equivalent under the configured latency
> tie policy, select the highest-accuracy member of the equivalent fastest
> cohort, followed by fingerprint and candidate-ID tie-breaking.

This is a reporting clarification, not a change to the frozen algorithm or
winner.

## Validation-only isolation

The feasible-cohort matched experiment is required to preserve:

```text
selector_invoked_on_matched_measurements = false
selection_time_objectives_replaced = false
measurements_feed_selection = false
measurements_feed_reselection = false
algorithm_selected_candidate_overridden = false
```

Its result can validate or challenge the historical latency claim, but cannot
replace the original archive values or rerun `analyze_archive` on matched
latencies.
