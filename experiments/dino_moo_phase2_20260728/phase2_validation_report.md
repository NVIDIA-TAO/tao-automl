# DINO multi-objective AutoML phase-2 validation

Expanded-experiment committed-evidence cutoff: AutoML commit
`b1a0ae235be53ba3ced7e4c880cb0be1f6b8157d` on 2026-07-28 UTC. The
corrected runtime implementation is commit
`e4b6a412545614668affd371a82231e090998ec0`; the corrected v2 manifest is
frozen by the cutoff commit. A separate live-evidence snapshot through
`2026-07-28T05:56:16Z` records completed rec0 and rec1 measurements from
mutable runtime state. Those six records are partial search evidence, not a
sealed archive or selection result. The post-front implementation described
below is tracked and committed at
`2453079abbe93e2fd854dcf2a910256dfd164669`, with the separately hashed source
snapshot verified at `2026-07-28T06:28:26Z`. A future launch must still pass
the manifest-bound clean-source gate.

Scope: DINO ResNet50 only, using
`s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/`.
No other model family, PTM compatibility repair, or dataset is included.

This is an in-progress report for MR !22. It records immutable sensitivity
evidence, the failed-and-excluded v1 expanded execution, the corrected
preregistered v2 contract, and v2 measurements through recommendation one for
all three search seeds. Recommendations 2–19, the complete expanded archive,
the post-front measurements, and final selection remain
`PENDING LIVE EVIDENCE`; no partial archive or manually selected value is used
to fill those final sections.

## Current status

| Work item | Status | Result at evidence cutoff |
| --- | --- | --- |
| Independent latency and multi-objective accuracy policies | Complete | `latency_accuracy_retention` applies only to latency mode; optional `multi_objective_min_accuracy` is independent and defaults to no floor. |
| Frozen 30-candidate archive replay | Complete | With no multi-objective floor, the production selector returns `seed_271828_rec_1`; with the old 98% multi-objective floor, only `seed_271828_rec_5` is eligible. |
| Historical six-front matched latency study | Complete | Six candidates × six allocations = 36 valid measurements; all 15 pairs are practically equivalent under the preregistered `0.75 ms` tolerance. |
| One-factor DINO training and accuracy study | Complete | 33 training jobs and 42 accuracy evaluations completed for 14 profiles across three seeds. |
| Sensitivity-latency v1 attempt and audit | Complete, invalid for timing | All nine jobs failed runtime preflight before any benchmark measurement. |
| Sensitivity-latency v2 | Complete | All 126 matched profile/allocation measurements passed validation. Encoder and decoder depth qualified; query and selection count did not. The effective tolerance is `0.73553775 ms`. |
| Expanded-search v1 execution | Complete, invalid and excluded | Rec0 training/evaluation completed for all seeds, but the v1 reader rejected finite mAP50 JSON strings. No accepted accuracy, latency measurement, usable Bayesian response, seed archive, or combined selection exists; rec1 work was canceled during controller shutdown. |
| Corrected expanded v2 manifest and remote preflight | Complete | Manifest whole/internal hashes are `9ac29e1a…` / `910744ae…`; its new runtime contract excludes all v1 state. Every SQSH, PTM, dataset, source, and directory check passed. |
| Corrected expanded shared 60-candidate archive | In progress; rec2–rec19 **PENDING LIVE EVIDENCE** | Rec0 and rec1 completed successfully for all three seeds: six complete training/evaluation/stabilized-latency objective pairs. No seed archive is sealed, so no production selector result is reported. |
| Post-front hardening implementation | Committed, independently re-audited with no remaining blocker, and tested; not launched | The final blocker—a self-rehashed manifest could preserve internal consistency while drifting launch-affecting runtime or latency-protocol semantics—is closed by exact deterministic reconstruction of the full manifest from pinned sources before every launcher mode and aggregation. The harness also freezes schedule derivation, stages job-private inputs, inspects status with retries disabled, applies exact pairwise direction gates, reconciles crash windows, and contains secrets/SQSH state. Its immutable manifest cannot be generated before the 60-record archive, combined selection, candidate table, and integrity audit exist. |
| Matched remeasurement of final Pareto front | **PENDING LIVE EVIDENCE** | No post-front manifest, SLURM allocation, or measurement exists yet. Future measurements must not replace selection-time objectives or alter the selected winner. |
| Final combined selection and hypothesis verdict | **PENDING LIVE EVIDENCE** | No final classification is made in this draft. |

The required correction to the previous conclusion is:

> The global archive contains six rank-zero Pareto candidates. No distinct
> Pareto compromise exists under the configured 98% multi-objective
> accuracy-feasibility constraint.

The first sentence must not be shortened to “the archive has no intermediate
candidate.” The global archive does contain intermediate trade-off points; the
old shared 98% floor excluded them from multi-objective scoring.

## 1. Root cause

### 1.1 Original dominated result

The original dominated result came from comparing winners produced by different
candidate populations. Accuracy, latency, and multi-objective mode each ran an
independent ten-candidate Bayesian search. The old latency winner was not
present in the multi-objective archive, so the multi-objective selector could
not compare against it.

The old implementation compounded that population mismatch:

1. It combined raw accuracy and scaled latency with a weighted sum instead of
   filtering a shared archive by Pareto nondominance.
2. Final selection was a plain maximum of that scalar result.
3. The latency feasibility floor used the raw PTM baseline mAP50
   `0.007808934173321529`; every candidate passed, so “latency mode” was
   effectively unconstrained.
4. Latency was inferred from a single evaluation progress-rate observation,
   without explicit device synchronization, repeated rounds, or an uncertainty
   estimate.

Across the old, separate archives, the latency winner had both higher mAP50 and
lower repeated proxy latency than the multi-objective winner. That dominance
finding was real, but it was a cross-archive comparison the selector had no
opportunity to make.

MR !22 first corrected this class of failure by selecting every mode from one
complete measured archive, rejecting invalid observations before ranking, and
requiring the multi-objective winner to come from the eligible rank-zero
Pareto front.

### 1.2 Why the first corrected DINO run still did not validate a compromise

The first Pareto-safe implementation reused the latency-mode 98% retained
accuracy rule as the multi-objective feasibility rule. For the frozen archive:

\[
A^*=0.6359897329231639
\]

\[
0.98A^*=0.6232699382647006
\]

Only `seed_271828_rec_5` met that threshold. Accuracy, latency, and
multi-objective mode therefore all returned the same point. This was a safe,
algorithmic result, but it did not test the intended joint trade-off behavior.

The global rank-zero front was not a singleton. It contained:

- `seed_161803_rec_0`
- `seed_314159_rec_7`
- `seed_271828_rec_2`
- `seed_161803_rec_2`
- `seed_271828_rec_1`
- `seed_271828_rec_5`

| Candidate | Queries | Learning rate | Weight decay | mAP50 | Historical median ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `seed_161803_rec_0` | 636 | 0.0000881875950 | 0.000413834280 | 0.527910926 | 79.330832 |
| `seed_314159_rec_7` | 523 | 0.000195490444 | 0.000678848752 | 0.559732921 | 79.334508 |
| `seed_271828_rec_2` | 878 | 0.000354940199 | 0.000389278783 | 0.576942047 | 79.406049 |
| `seed_161803_rec_2` | 311 | 0.000406006144 | 0.000752116849 | 0.578060811 | 79.5114385 |
| `seed_271828_rec_1` | 662 | 0.000221764451 | 0.000968322677 | 0.606457815 | 79.5460065 |
| `seed_271828_rec_5` | 594 | 0.00045 | 0.000269677238 | 0.635989733 | 79.89227525 |

The semantic defect was therefore constraint coupling: a latency deployment
policy silently narrowed the multi-objective decision population. It was not a
lack of global Pareto candidates.

## 2. Algorithm changes

The mode-semantics correction is commit
`83d9d7ecc783724f674cb954f9fbb6c91ea8b0eb`:
“Separate latency and multi-objective accuracy policies.”

The principal implementation is `src/tao_automl/selection.py`; configuration
parsing is in `src/tao_automl/objectives.py`.

### 2.1 Independent mode semantics

Accuracy mode evaluates every valid candidate:

\[
x_A=\arg\max_{x\in V} A(x).
\]

Only candidates equivalent within `accuracy_tolerance` may use lower latency as
a tie-break. The remaining order is canonical specification SHA-256 and
candidate ID.

Latency mode uses its own configurable retained-accuracy rule:

\[
F_L=\{x\in V:A(x)\ge rA^*\},\qquad
x_L=\arg\min_{x\in F_L}L(x).
\]

For this validation, \(r=0.98\). Absolute maximum degradation is also
supported. If \(F_L\) is empty, the selector returns
`no_accuracy_feasible_candidates`; it does not silently fall back.

Multi-objective mode now has a separate optional policy:

```text
latency_accuracy_retention
multi_objective_min_accuracy
```

`multi_objective_min_accuracy=None` admits every valid candidate to
multi-objective nondominated sorting. An explicit absolute or
accuracy-winner-relative multi-objective floor is supported for product policy
use, but it never inherits `latency_accuracy_retention`.

Legacy accuracy-constraint settings remain accepted only as aliases for
latency-mode retention. Conflicting new and legacy latency settings fail
closed.

### 2.2 Validity, dominance, and front construction

Candidates are rejected before ranking for non-success status, missing or
non-finite objectives, boolean values, non-positive latency, malformed
configuration, incomplete or invalid latency confidence intervals, or an
interval that excludes its median.

For valid candidates \(x,y\), \(x\) dominates \(y\) only when:

\[
A(x)\ge A(y),\qquad L(x)\le L(y),
\]

and at least one objective is strictly better. Confidence intervals or the
configured tolerance determine whether a latency improvement is strict; they
never permit a numerically slower point to dominate a faster point.
Deterministic non-dominated sorting records both zero-based rank and exact
`dominated_by` relationships.

Exact duplicate objective points use the candidate with the smallest canonical
configuration fingerprint as the representative, while preserving every alias
in the audit.

### 2.3 Normalization and compromise score

Only the eligible, deduplicated rank-zero front defines normalization bounds.
For accuracy \(A\) and latency \(L\):

\[
r_A(x)=\frac{A_{\max}-A(x)}{A_{\max}-A_{\min}},
\qquad
r_L(x)=\frac{L(x)-L_{\min}}{L_{\max}-L_{\min}}.
\]

This orients accuracy as maximize and latency as minimize without combining raw
mAP and milliseconds. An objective whose front range is within tolerance is
inactive and contributes zero regret, avoiding division by zero.

With normalized weights \(w_A=w_L=0.5\) and
\(\rho=10^{-6}\), the selected point minimizes:

\[
C(x)=\max(w_A r_A(x),w_L r_L(x))
     +\rho(w_A r_A(x)+w_L r_L(x)).
\]

Ties within `1e-12` use, in order:

1. lower weighted Euclidean distance to the ideal;
2. lower absolute weighted-regret balance gap;
3. lower normalized accuracy regret;
4. canonical specification SHA-256;
5. candidate ID.

Selection cannot depend on enumeration order. A dominated point is filtered
before scoring and cannot be returned.

### 2.4 Extreme and fallback reporting

The selector does not force a middle point. It records whether the selected
rank-zero point is distinct from the policy-specific accuracy and latency
extremes under the configured tolerances.

When no distinct eligible point exists, it emits:

> No distinct Pareto compromise exists under the configured multi-objective
> eligibility policy.

The deterministic Chebyshev ordering may still return an extreme, but the
audit sets `distinct_compromise=false` and distinguishes a fallback from a
successful compromise.

### 2.5 Alternatives considered

| Method | Decision |
| --- | --- |
| Pareto nondominance | Required safety filter. |
| Epsilon constraint | Used for latency mode because retained accuracy is naturally a constraint. |
| Normalized ideal-point distance | Retained as a tie-break; not primary because it is fully compensatory. |
| Knee point | Not the default because sparse or nearly linear fronts can have unstable knees. |
| Augmented Chebyshev | Selected for the final compromise because it minimizes worst normalized regret, handles non-convex fronts, and is deterministic. |
| Hypervolume contribution | Not used for one final point because it requires a reference point and can favor endpoints; more suitable for archive acquisition. |
| NSGA-II ranking and crowding | Nondominated ranking is adopted; a full evolutionary population is unnecessary for this small, expensive archive. |
| qEHVI/qNEHVI | Deferred as a possible larger-budget acquisition method; it would not by itself define the final deployment point. |

## 3. Test coverage

The original Pareto-hardening suite covers dominated-point exclusion, valid and
invalid middle points, duplicate objectives, enumeration-order invariance,
positive objective-scale invariance, zero-range objectives, missing/non-finite
metrics, deterministic fallback, directionality, confidence-aware tie-breaking,
and a property test asserting that a selected compromise is never dominated.

Commit `83d9d7e` adds the independent-policy cases:

- latency retention does not filter the multi-objective front;
- an optional absolute multi-objective floor affects only compromise mode;
- a relative multi-objective floor resolves from the accuracy winner;
- predefined 90%, 95%, and 98% sensitivity policies resolve without a fitted
  threshold;
- an optional floor can explicitly report no eligible candidate;
- multi-objective dominance is independent of latency retention;
- separate policies are candidate-order invariant;
- public objective configuration parses both policies independently;
- legacy accuracy settings map to latency mode only;
- conflicting new and legacy latency settings are rejected;
- an external latency reference cannot block unconstrained multi-objective
  selection.

The added parameterization contributes 13 test cases. The MR test result after
this change is:

```bash
python -m pytest -q tests
```

```text
387 passed, 1 skipped
```

The committed sensitivity-analysis, expanded-search, and post-front hardening
harnesses were rerun together:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  -m pytest -q \
  experiments/dino_moo_phase2_20260728/test_sensitivity_latency_analysis_erratum.py \
  experiments/dino_moo_phase2_20260728/test_sensitivity_latency_runtime_contract.py \
  experiments/dino_moo_phase2_20260728/test_expanded_search_manifest_generator.py \
  experiments/dino_moo_phase2_20260728/test_expanded_search_runner.py \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_tools.py \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_launcher_recovery.py \
  experiments/dino_moo_phase2_20260728/test_post_front_complete_invalid_recovery.py
```

```text
206 passed in 1.63s
```

The post-front subset alone passed:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  -m pytest -q \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_tools.py \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_launcher_recovery.py \
  experiments/dino_moo_phase2_20260728/test_post_front_complete_invalid_recovery.py
```

```text
81 passed in 0.43s
```

The combined focused run covers the provenance-safe analysis erratum,
immutable runtime contract, strict native-number/JSON-number-string metric
parsing, rejection of booleans, NaN, infinity, whitespace, and junk, clean
v1-to-v2 runtime supersession, manifest derivation, exact integer-domain
encoding, remote preflight, resume/reconciliation behavior,
complete-archive gating, candidate-order invariance, and algorithm-only union
selection. The post-front cases additionally cover full rank-zero derivation,
candidate-order independence, Williams projection for front sizes one through
ten, immutable SQSH/runtime/no-reselection bindings, full-block
supersession, complete matched-matrix enforcement, source and node
provenance, exact practical-tolerance boundary behavior, all-pair coverage,
selector isolation, owner-only secret handling, nonblocking launch locking,
zero/one/multiple durable-job-delta reconciliation, uncertain-launch
rejection, remote entrypoint binding, pre-scheduler orphan terminalization,
and exact-versus-tampered replacement-intent replay. The Complete-invalid
cases prove that a valid Complete job cannot be replaced without aggregator
evidence, an exactly attributed semantic failure discards and reruns the
whole allocation, stale/tampered/cross-allocation evidence is rejected, and
unattributed or multi-allocation failures cannot authorize replacement.
A pre-existing immutable successful analysis is an additional veto: it proves
the allocation set was valid and prevents replacement even if stale
invalidation evidence is supplied.

The final manifest-authority cases exercise 16 launch-affecting runtime and
latency-protocol mutations. Each test deliberately recomputes the manifest's
internal digest, demonstrating that a valid self-hash is only an integrity
check; exact reconstruction from pinned source artifacts still rejects every
semantic drift. Static call-path checks also prove that this reconstruction
precedes dry-run rendering and every fresh-launch, resume, replacement, and
aggregation path.

The deterministic archive replay additionally verifies all four policy results
under source order, reverse order, and a SHA-256-keyed permutation. Every
policy produced one identical analysis hash across the three enumerations.

The matched-latency artifact independently verifies hardware, runtime,
benchmark-input, protocol, rank-file, and candidate-count contracts for all
36 historical-front measurements.

## 4. Phase 1: frozen 30-candidate archive replay

The replay was generated only by
`tao_automl.selection.analyze_archive`; no candidate ID is promoted or
overridden by the replay driver.

### 4.1 Integrity anchors

| Artifact | SHA-256 |
| --- | --- |
| `phase1_offline_replay.json` | `5b58dab75fcc6a05e658a5de6455cff767326e0bb299c815c0dc1255918a88ca` |
| Production selector source used by replay | `7e787a18bca05464e0043367aee4f2c8cff3d93aef7f9e92aaf88c47d255a532` |
| Replay script | `09956cf9001b55d482dd54d65f14ef115187e3ed476ba73e296a36ecf6df6a26` |
| Frozen combined selection | `794c038c9506b805b57c7812355ec173e5ea0275b21587befaf8c91a78cbe2f7` |
| Seed 161803 candidate archive | `9f8cc57d0939e6744f78057da5b284d639402d6a91d9fdd1d66a6fb65818e258` |
| Seed 271828 candidate archive | `5e10721a3ee0016222c6c9a23054e4a2837d15a8182cd94a1e4941fe84b316f3` |
| Seed 314159 candidate archive | `d7701574825b7d8df5acbe84fc9b87df27f1afce784e1c9322a8deec8f7507ff` |

The archive contains ten candidates from each of search seeds `161803`,
`271828`, and `314159`. All 30 candidate IDs are unique, and every per-seed
record exactly matches the corresponding union record.

### 4.2 Policy replay summary

Latency mode remains fixed at 98% retention for every row. The policy column
changes only `multi_objective_min_accuracy`.

| Multi-objective policy | Resolved floor | Eligible count | Eligible rank-zero count | Selected candidate | Selector classification | Fallback |
| --- | ---: | ---: | ---: | --- | --- | :---: |
| Relative 98% | 0.6232699382647006 | 1 | 1 | `seed_271828_rec_5` | Shared accuracy/latency extreme | Yes |
| Relative 95% | 0.6041902462770057 | 5 | 2 | `seed_271828_rec_5` | Shared accuracy/latency extreme | Yes |
| Relative 90% | 0.5723907596308475 | 12 | 4 | `seed_271828_rec_1` | Policy-front extreme | No |
| None | None | 30 | 6 | `seed_271828_rec_1` | Distinct compromise | No |

Eligible candidate IDs are:

- **98%:** `seed_271828_rec_5`.
- **95%:** `seed_161803_rec_4`, `seed_161803_rec_8`,
  `seed_271828_rec_1`, `seed_271828_rec_5`, `seed_271828_rec_9`.
- **90%:** `seed_161803_rec_1`, `seed_161803_rec_2`,
  `seed_161803_rec_4`, `seed_161803_rec_5`, `seed_161803_rec_6`,
  `seed_161803_rec_8`, `seed_271828_rec_0`, `seed_271828_rec_1`,
  `seed_271828_rec_2`, `seed_271828_rec_5`, `seed_271828_rec_7`,
  `seed_271828_rec_9`.
- **No floor:** all 30 valid candidates.

Eligible rank-zero IDs are:

- **98%:** `seed_271828_rec_5`.
- **95%:** `seed_271828_rec_1`, `seed_271828_rec_5`.
- **90%:** `seed_161803_rec_2`, `seed_271828_rec_1`,
  `seed_271828_rec_2`, `seed_271828_rec_5`.
- **No floor:** `seed_161803_rec_0`, `seed_161803_rec_2`,
  `seed_271828_rec_1`, `seed_271828_rec_2`,
  `seed_271828_rec_5`, `seed_314159_rec_7`.

Accuracy and latency mode are unchanged in all four replays:
`seed_271828_rec_5` wins both. The latency winner is unchanged because the
latency-mode 98% policy remains independent and admits only that candidate.

### 4.3 Normalization bounds

| Policy | Accuracy ideal | Accuracy nadir | Accuracy range | Latency ideal ms | Latency nadir ms | Latency range ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 98% | 0.6359897329231639 | 0.6359897329231639 | 0 | 79.89227525 | 79.89227525 | 0 |
| 95% | 0.6359897329231639 | 0.6064578150355908 | 0.0295319178875730 | 79.54600650 | 79.89227525 | 0.34626875 |
| 90% | 0.6359897329231639 | 0.5769420472162294 | 0.0590476857069344 | 79.40604900 | 79.89227525 | 0.48622625 |
| None | 0.6359897329231639 | 0.5279109255336222 | 0.108078807389542 | 79.33083200 | 79.89227525 | 0.56144325 |

The two 98% axes are inactive because the eligible front has one point.

### 4.4 Rank-zero scores and tie values

`rA` and `rL` are unweighted normalized regrets. `C` is the augmented
Chebyshev score. `D` is weighted ideal distance and `G` is weighted balance
gap.

| Policy | Candidate | Global rank | Eligible rank | rA | rL | C | D | G | Winner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 98% | `seed_271828_rec_5` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | Yes |
| 95% | `seed_271828_rec_5` | 0 | 0 | 0 | 1 | 0.5000005 | 0.5 | 0.5 | Yes |
| 95% | `seed_271828_rec_1` | 0 | 0 | 1 | 0 | 0.5000005 | 0.5 | 0.5 | No |
| 90% | `seed_161803_rec_2` | 0 | 0 | 0.981053219 | 0.216749918 | 0.490527208 | 0.502355936 | 0.382151650 | No |
| 90% | `seed_271828_rec_5` | 0 | 0 | 0 | 1 | 0.5000005 | 0.5 | 0.5 | No |
| 90% | `seed_271828_rec_2` | 0 | 0 | 1 | 0 | 0.5000005 | 0.5 | 0.5 | No |
| 90% | `seed_271828_rec_1` | 0 | 0 | 0.500136754 | 0.287844393 | 0.250068771 | 0.288526935 | 0.106146181 | Yes |
| None | `seed_161803_rec_0` | 0 | 0 | 1 | 0 | 0.5000005 | 0.5 | 0.5 | No |
| None | `seed_314159_rec_7` | 0 | 0 | 0.705566744 | 0.006547412 | 0.352783728 | 0.352798561 | 0.349509666 | No |
| None | `seed_271828_rec_2` | 0 | 0 | 0.546339168 | 0.133970798 | 0.273169924 | 0.281262627 | 0.206184185 | No |
| None | `seed_161803_rec_2` | 0 | 0 | 0.535987799 | 0.321682557 | 0.267994328 | 0.312555030 | 0.107152621 | No |
| None | `seed_271828_rec_1` | 0 | 0 | 0.273244298 | 0.383252448 | 0.191626552 | 0.235342774 | 0.055004075 | Yes |
| None | `seed_271828_rec_5` | 0 | 0 | 0 | 1 | 0.5000005 | 0.5 | 0.5 | No |

The selected multi-objective tie tuples
`(C, D, G, rA, fingerprint, candidate_id)` are:

- **98%:** `(0, 0, 0, 0,
  e608fa8d2b79795abfe909e7126c2762a66871387756ab13cc35a6ba7be48f05,
  seed_271828_rec_5)`.
- **95%:** `(0.5000005, 0.5, 0.5, 0,
  e608fa8d2b79795abfe909e7126c2762a66871387756ab13cc35a6ba7be48f05,
  seed_271828_rec_5)`. The two scores, distances, and gaps tie; lower
  accuracy regret deterministically chooses `rec_5`.
- **90%:** `(0.2500687712183107, 0.2885269346114728,
  0.1061461805137502, 0.5001367544554736,
  f4e09477dbbe4d4091e2f93ffd305666c04221992256fb4d3f5c281f2bcd2a5c,
  seed_271828_rec_1)`.
- **No floor:** `(0.1916265522701285, 0.2353427742100136,
  0.0550040748656632, 0.2732442983121843,
  f4e09477dbbe4d4091e2f93ffd305666c04221992256fb4d3f5c281f2bcd2a5c,
  seed_271828_rec_1)`.

### 4.5 All 30 candidates: global and policy-front ranks

`—` means the candidate is ineligible under that optional multi-objective
floor. Ranks are zero-based. The no-floor eligible rank equals the global rank.
The complete machine-readable record also contains exact parameters,
confidence intervals, dominators, fingerprints, objective regrets, scores, and
winner flags for every row.

| Candidate | mAP50 | Historical median ms | Global rank | 98% rank | 95% rank | 90% rank | No-floor rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed_161803_rec_0` | 0.527911 | 79.3308 | 0 | — | — | — | 0 |
| `seed_161803_rec_1` | 0.575453 | 79.6028 | 1 | — | — | 1 | 1 |
| `seed_161803_rec_2` | 0.578061 | 79.5114 | 0 | — | — | 0 | 0 |
| `seed_161803_rec_3` | 0.568511 | 80.1014 | 4 | — | — | — | 4 |
| `seed_161803_rec_4` | 0.613110 | 79.9475 | 1 | — | 1 | 1 | 1 |
| `seed_161803_rec_5` | 0.601234 | 80.0611 | 3 | — | — | 3 | 3 |
| `seed_161803_rec_6` | 0.574713 | 79.5433 | 1 | — | — | 1 | 1 |
| `seed_161803_rec_7` | 0.550185 | 79.6100 | 2 | — | — | — | 2 |
| `seed_161803_rec_8` | 0.604800 | 79.7275 | 2 | — | 2 | 2 | 2 |
| `seed_161803_rec_9` | 0.417475 | 79.6353 | 4 | — | — | — | 4 |
| `seed_271828_rec_0` | 0.594847 | 79.7459 | 3 | — | — | 3 | 3 |
| `seed_271828_rec_1` | 0.606458 | 79.5460 | 0 | — | 0 | 0 | 0 |
| `seed_271828_rec_2` | 0.576942 | 79.4060 | 0 | — | — | 0 | 0 |
| `seed_271828_rec_3` | 0.527213 | 79.5900 | 3 | — | — | — | 3 |
| `seed_271828_rec_4` | 0.553107 | 79.5114 | 1 | — | — | — | 1 |
| `seed_271828_rec_5` | 0.635990 | 79.8923 | 0 | 0 | 0 | 0 | 0 |
| `seed_271828_rec_6` | 0.534644 | 79.5586 | 2 | — | — | — | 2 |
| `seed_271828_rec_7` | 0.601110 | 79.6654 | 2 | — | — | 2 | 2 |
| `seed_271828_rec_8` | 0.550797 | 80.1411 | 5 | — | — | — | 5 |
| `seed_271828_rec_9` | 0.605146 | 79.6290 | 1 | — | 1 | 1 | 1 |
| `seed_314159_rec_0` | 0.553572 | 79.8155 | 4 | — | — | — | 4 |
| `seed_314159_rec_1` | 0.516658 | 79.6716 | 4 | — | — | — | 4 |
| `seed_314159_rec_2` | 0.550627 | 80.0261 | 5 | — | — | — | 5 |
| `seed_314159_rec_3` | 0.550203 | 79.7711 | 4 | — | — | — | 4 |
| `seed_314159_rec_4` | 0.399534 | 79.7834 | 5 | — | — | — | 5 |
| `seed_314159_rec_5` | 0.549730 | 79.8438 | 5 | — | — | — | 5 |
| `seed_314159_rec_6` | 0.531360 | 80.0884 | 6 | — | — | — | 6 |
| `seed_314159_rec_7` | 0.559733 | 79.3345 | 0 | — | — | — | 0 |
| `seed_314159_rec_8` | 0.561366 | 79.5194 | 1 | — | — | — | 1 |
| `seed_314159_rec_9` | 0.547490 | 80.4125 | 6 | — | — | — | 6 |

The offline replay proves the semantic distinction: the production selector
can select `seed_271828_rec_1` as a distinct no-floor compromise without any
manual intervention. It does not prove that the historical latency differences
are allocation-stable.

## 5. Phase 2: historical six-front matched latency validation

This experiment never feeds selection. It benchmarks every global rank-zero
candidate in every allocation, so candidate comparisons are matched by node and
allocation.

### 5.1 Protocol and execution

- Six independent SLURM jobs, each one node and eight A100 GPUs.
- A six-row Williams/Latin-square schedule balances candidate position and
  ordered adjacency.
- Batch size one, FP32, TF32 disabled.
- Fixed preprocessed model input `[1,4,800,1333]`.
- 16 preloaded validation batches.
- 50 untimed warm-ups.
- Five rounds × 100 timed requests × eight ranks = 4,000 samples per candidate
  and allocation.
- CUDA synchronization around every request and NCCL barriers around candidate
  and round boundaries.
- Timed scope: DINO model forward plus GPU postprocessing.
- Excluded: checkpoint load, disk I/O, decode/resize/normalization, H2D,
  COCO accumulation, and distributed gather.
- Primary estimator: median of 40 device-round medians.
- Tail: pooled p95.
- Within-allocation uncertainty: deterministic 5,000-resample device-round
  cluster bootstrap.
- Pairwise uncertainty: deterministic 5,000-resample paired bootstrap at the
  allocation level.
- Practical-equivalence tolerance: `0.75 ms`, rounded up from the prior
  `0.73553775 ms` independent-allocation range.

The fixed input identity was
`1b43c34913bff097054d6a76cdd7dd0a02546dd07db8adce50d40a8986774d08`
on every rank.

Phase-1 retention had deleted four of the six historical checkpoints. Their
exact configurations were retrained with the frozen training seed, ten-epoch
budget, PTM, dataset, and eight-GPU topology. These are configuration-exact
reconstructions, not byte-identical replicas. Comparable evaluation confirmed
nonzero reconstruction variation:

| Candidate | Historical mAP50 | Reconstructed mAP50 | Delta |
| --- | ---: | ---: | ---: |
| `seed_161803_rec_0` | 0.527910926 | 0.558982277 | +0.031071352 |
| `seed_161803_rec_2` | 0.578060811 | 0.583272765 | +0.005211955 |
| `seed_271828_rec_1` | 0.606457815 | 0.594148762 | -0.012309053 |
| `seed_271828_rec_2` | 0.576942047 | 0.599031769 | +0.022089722 |

The reconstructions are used only to execute the matched latency graph for the
historical configurations. Their re-evaluated accuracy never replaces the
frozen archive, changes a Pareto rank, or feeds selection. The revalidation
artifact SHA-256 is
`5546c8e38a9c667ebfce235c1d92be873fae5237108e91b668c07aeb96ca0dc2`.

| Block | TAO job | SLURM job | Node | Terminal evidence |
| --- | --- | ---: | --- | --- |
| `block_00` | `d92a5548-a0f9-4df6-ab60-18d3d54347b3` | 30939506 | `batch-block7-01825` | `Complete / COMPLETED / 0:0` |
| `block_01` | `ed7154d1-d2cd-45c4-b10d-825ad01decec` | 30939515 | `batch-block7-01958` | `Complete / COMPLETED / 0:0` |
| `block_02` | `5046d900-76cd-4ecc-90c0-48006ad5d16d` | 30939522 | `batch-block7-03049` | `Complete / COMPLETED / 0:0` |
| `block_03` | `d4d81b87-a522-4270-9643-058481a40431` | 30939533 | `batch-block7-01304` | `Complete / COMPLETED / 0:0` |
| `block_04` | `84d02bb9-7d36-43b5-876a-362fae256117` | 30939557 | `batch-block7-01976` | `Complete / COMPLETED / 0:0` |
| `block_05` | `c0991e1e-959d-4f46-9e49-a72ec080123d` | 30939572 | `batch-block7-02907` | `Complete / COMPLETED / 0:0` |

### 5.2 Per-allocation measurements

The interval is the within-allocation cluster-bootstrap interval for the
median. Values are milliseconds.

| Block | Candidate | Position | Median | p95 | Median 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| 00 | `seed_161803_rec_0` | 0 | 78.8630 | 79.1662 | [78.8019, 78.9125] |
| 00 | `seed_161803_rec_2` | 1 | 78.9385 | 79.2020 | [78.8925, 78.9646] |
| 00 | `seed_314159_rec_7` | 2 | 78.8543 | 79.1388 | [78.8336, 78.8943] |
| 00 | `seed_271828_rec_1` | 3 | 78.7978 | 79.1424 | [78.7639, 78.8300] |
| 00 | `seed_271828_rec_5` | 4 | 79.0033 | 79.4672 | [78.9193, 79.0473] |
| 00 | `seed_271828_rec_2` | 5 | 79.0387 | 79.4470 | [79.0048, 79.0558] |
| 01 | `seed_161803_rec_2` | 0 | 79.4705 | 79.8003 | [79.4044, 79.6115] |
| 01 | `seed_271828_rec_1` | 1 | 79.4653 | 79.9113 | [79.4003, 79.5631] |
| 01 | `seed_161803_rec_0` | 2 | 79.4493 | 79.6198 | [79.2996, 79.4679] |
| 01 | `seed_271828_rec_2` | 3 | 79.5653 | 79.9523 | [79.5238, 79.6345] |
| 01 | `seed_314159_rec_7` | 4 | 79.4543 | 79.8039 | [79.4151, 79.5001] |
| 01 | `seed_271828_rec_5` | 5 | 79.4138 | 79.7125 | [79.3355, 79.4803] |
| 02 | `seed_271828_rec_1` | 0 | 79.4672 | 79.9048 | [79.4351, 79.4898] |
| 02 | `seed_271828_rec_2` | 1 | 79.5118 | 80.1119 | [79.4569, 79.5545] |
| 02 | `seed_161803_rec_2` | 2 | 79.6283 | 79.9314 | [79.5515, 79.6881] |
| 02 | `seed_271828_rec_5` | 3 | 79.5953 | 79.8963 | [79.5338, 79.6288] |
| 02 | `seed_161803_rec_0` | 4 | 79.3830 | 80.0768 | [79.3194, 79.4665] |
| 02 | `seed_314159_rec_7` | 5 | 79.5264 | 79.7357 | [79.4938, 79.5714] |
| 03 | `seed_271828_rec_2` | 0 | 79.3477 | 79.6369 | [79.2944, 79.3986] |
| 03 | `seed_271828_rec_5` | 1 | 79.2708 | 79.6335 | [79.2449, 79.3413] |
| 03 | `seed_271828_rec_1` | 2 | 79.3909 | 79.7468 | [79.2928, 79.4533] |
| 03 | `seed_314159_rec_7` | 3 | 79.4023 | 79.7178 | [79.3700, 79.4350] |
| 03 | `seed_161803_rec_2` | 4 | 79.4035 | 79.7720 | [79.3118, 79.4920] |
| 03 | `seed_161803_rec_0` | 5 | 79.3245 | 79.9729 | [79.2042, 79.4087] |
| 04 | `seed_271828_rec_5` | 0 | 79.3729 | 79.7007 | [79.3441, 79.4633] |
| 04 | `seed_314159_rec_7` | 1 | 79.4524 | 79.7789 | [79.3785, 79.4999] |
| 04 | `seed_271828_rec_2` | 2 | 79.4672 | 80.0199 | [79.4043, 79.5234] |
| 04 | `seed_161803_rec_0` | 3 | 79.3922 | 79.6800 | [79.3625, 79.4747] |
| 04 | `seed_271828_rec_1` | 4 | 79.3560 | 79.7003 | [79.2685, 79.4568] |
| 04 | `seed_161803_rec_2` | 5 | 79.3061 | 79.7798 | [79.2636, 79.3754] |
| 05 | `seed_314159_rec_7` | 0 | 79.1877 | 79.4492 | [79.1378, 79.2349] |
| 05 | `seed_161803_rec_0` | 1 | 79.1624 | 79.4161 | [79.1310, 79.2248] |
| 05 | `seed_271828_rec_5` | 2 | 79.1622 | 79.4229 | [79.1316, 79.2127] |
| 05 | `seed_161803_rec_2` | 3 | 79.2093 | 79.4926 | [79.1720, 79.2325] |
| 05 | `seed_271828_rec_2` | 4 | 79.3022 | 79.5264 | [79.2039, 79.3677] |
| 05 | `seed_271828_rec_1` | 5 | 79.1941 | 79.4671 | [79.1505, 79.2408] |

### 5.3 Between-allocation summaries

| Candidate | Median of allocation medians | Median CI | Allocation range | Median of allocation p95 | p95 range |
| --- | ---: | --- | ---: | ---: | ---: |
| `seed_271828_rec_5` | 79.3218 | [79.0828, 79.5046] | 0.5920 | 79.6671 | 0.4734 |
| `seed_161803_rec_0` | 79.3538 | [79.0127, 79.4208] | 0.5863 | 79.6499 | 0.9106 |
| `seed_161803_rec_2` | 79.3548 | [79.0739, 79.5494] | 0.6898 | 79.7759 | 0.7294 |
| `seed_271828_rec_1` | 79.3735 | [78.9960, 79.4663] | 0.6694 | 79.7235 | 0.7689 |
| `seed_271828_rec_2` | 79.4075 | [79.1704, 79.5386] | 0.5267 | 79.7946 | 0.6649 |
| `seed_314159_rec_7` | 79.4273 | [79.0210, 79.4903] | 0.6721 | 79.7267 | 0.6651 |

### 5.4 Paired differences

The delta is first candidate minus second candidate; negative means the first
candidate was faster. A directional claim requires the entire paired 95% CI to
lie below `-0.75 ms` or above `+0.75 ms`.

| First | Second | Median paired delta ms | Paired 95% CI ms | Classification |
| --- | --- | ---: | --- | --- |
| `161803/0` | `161803/2` | -0.0612 | [-0.1621, 0.0324] | Stable practical equivalence |
| `161803/0` | `271828/1` | -0.0238 | [-0.0753, 0.0507] | Stable practical equivalence |
| `161803/0` | `271828/2` | -0.1224 | [-0.1577, -0.0491] | Stable practical equivalence |
| `161803/0` | `271828/5` | 0.0098 | [-0.1763, 0.0446] | Stable practical equivalence |
| `161803/0` | `314159/7` | -0.0427 | [-0.1106, 0.0019] | Stable practical equivalence |
| `161803/2` | `271828/1` | 0.0139 | [-0.0223, 0.1509] | Stable practical equivalence |
| `161803/2` | `271828/2` | -0.0938 | [-0.1306, 0.0861] | Stable practical equivalence |
| `161803/2` | `271828/5` | 0.0400 | [-0.0658, 0.0947] | Stable practical equivalence |
| `161803/2` | `314159/7` | 0.0189 | [-0.0725, 0.0930] | Stable practical equivalence |
| `271828/1` | `271828/2` | -0.1040 | [-0.1760, -0.0007] | Stable practical equivalence |
| `271828/1` | `271828/5` | 0.0075 | [-0.1668, 0.0858] | Stable practical equivalence |
| `271828/1` | `314159/7` | -0.0339 | [-0.0777, 0.0087] | Stable practical equivalence |
| `271828/2` | `271828/5` | 0.0856 | [-0.0241, 0.1457] | Stable practical equivalence |
| `271828/2` | `314159/7` | 0.0630 | [-0.0346, 0.1494] | Stable practical equivalence |
| `271828/5` | `314159/7` | -0.0330 | [-0.1055, 0.1090] | Stable practical equivalence |

Some intervals exclude zero, but none approaches the preregistered practical
threshold. The artifact therefore supports no stable total latency order and
no meaningful latency advantage among the six historical front candidates.
These repeat values are validation-only and do not replace frozen selection
measurements.

Integrity anchors:

| Artifact | SHA-256 |
| --- | --- |
| Historical matched manifest v2 | `ccf88ad1a8c95a808bb9e217de50dc296b700e5af6e1dca474d56b967186e0d2` |
| Six-block submission ledger | `863a1bd11687222b53b9e1690673a4a1ab8bd2b2878be2d77b3cbc293db0f480` |
| `matched_pareto_latency_comparison.json` | `533aea85ee9bb5c7cd7d88e0c0d5efe5003cfd0af968dcf01c79b182a3c0ea2f` |

## 6. Phase 3: DINO latency-sensitivity study and search-space rationale

### 6.1 Supported one-factor axes

The frozen study uses the phase-1 algorithm-selected accuracy winner as its
reference profile, not a manually chosen result:

```text
candidate: seed_271828_rec_5
model.num_queries: 594
model.enc_layers: 6
model.dec_layers: 6
model.num_select: 300
train.optim.lr: 0.00045
train.optim.weight_decay: 0.00026967723799334445
```

The supported one-factor axes are:

| Axis | Frozen levels | Why it can affect inference cost | Compatibility evidence |
| --- | --- | --- | --- |
| `model.num_queries` | 300, 450, 594, 750, 900 | Changes decoder query workload. | All levels trained and evaluated; `num_queries >= num_select` enforced. |
| `model.enc_layers` | 3, 4, 5, 6 | Changes encoder graph depth. | All levels trained and evaluated with the pinned PTM/runtime. |
| `model.dec_layers` | 3, 4, 5, 6 | Changes decoder graph depth. | All levels trained and evaluated with the pinned PTM/runtime. |
| `model.num_select` | 50, 100, 200, 300 | Changes the postprocess selection workload in the timed scope. | All levels evaluated from the same-seed reference checkpoint; `num_select <= num_queries` enforced. |

Support does not imply a material latency effect. An axis enters the expanded
search only if the matched v2 latency analysis qualifies at least one
non-reference level.

The following categories remain excluded:

- backbone choice: only ResNet50 is compatible with the frozen TAO 7.0.1
  runtime; GCViT and FAN compatibility work remains out of scope;
- hidden dimension, head count, feed-forward dimension, and encoder/decoder
  sample-point counts: these alter pinned PTM tensor shapes;
- feature-level count and return indices: coupled, and the return-index list is
  not AutoML-searchable in the frozen schema;
- major architecture semantics such as two-stage type, decoder self-attention
  type, pre-norm, and dilation;
- input/evaluation resolution: held fixed as a deployment profile;
- precision, batch size, workers, activation checkpointing, and other
  measurement/training protocol controls;
- learning rate and weight decay as latency axes. They remain accuracy-search
  parameters but are not claimed to alter inference graph cost.

### 6.2 Completed training and accuracy evidence

The study completed 33 independent training jobs and 42 controlled mAP50
evaluations. `model.num_select` profiles reused each seed's exact reference
checkpoint, so they required evaluation but no separate training.

| Profile | Axis/value | Seed 1234 | Seed 271828 | Seed 314159 | Median mAP50 | Seeds passing same-seed 98% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Reference | — | 0.616349 | 0.594082 | 0.637878 | 0.616349 | 3/3 |
| Queries 300 | `num_queries=300` | 0.603047 | 0.589169 | 0.592356 | 0.592356 | 1/3 |
| Queries 450 | `num_queries=450` | 0.563076 | 0.624094 | 0.602008 | 0.602008 | 1/3 |
| Queries 750 | `num_queries=750` | 0.557525 | 0.649796 | 0.594984 | 0.594984 | 1/3 |
| Queries 900 | `num_queries=900` | 0.621423 | 0.616138 | 0.655693 | 0.621423 | 3/3 |
| Encoder 3 | `enc_layers=3` | 0.519272 | 0.562078 | 0.518955 | 0.519272 | 0/3 |
| Encoder 4 | `enc_layers=4` | 0.512927 | 0.583577 | 0.551196 | 0.551196 | 1/3 |
| Encoder 5 | `enc_layers=5` | 0.529423 | 0.587381 | 0.591078 | 0.587381 | 1/3 |
| Decoder 3 | `dec_layers=3` | 0.637137 | 0.651348 | 0.631784 | 0.637137 | 3/3 |
| Decoder 4 | `dec_layers=4` | 0.643576 | 0.682005 | 0.646883 | 0.646883 | 3/3 |
| Decoder 5 | `dec_layers=5` | 0.625369 | 0.576455 | 0.598474 | 0.598474 | 1/3 |
| Select 50 | `num_select=50` | 0.615510 | 0.592934 | 0.636935 | 0.615510 | 3/3 |
| Select 100 | `num_select=100` | 0.616029 | 0.593654 | 0.637465 | 0.616029 | 3/3 |
| Select 200 | `num_select=200` | 0.616087 | 0.594012 | 0.637857 | 0.616087 | 3/3 |

The same-seed 98% result is only a constrained-latency annotation. It does not
qualify or disqualify an architecture axis for the shared multi-objective
search.

Integrity anchors:

| Artifact | SHA-256 |
| --- | --- |
| One-factor preregistration manifest | `ee65fd9a09d7cacc40f88a0b95b07af3fd0560d8496407447344b310bb5eaa44` |
| Checkpoint artifact | `20188a8858a9329ce4b861730ad3b0b2f6185389c8af1b02ad29284e5ed1b012` |
| Accuracy artifact | `459da2ebe557ec26947dc723b2864f2bc31880ae3181ad1216c3a47825ec466b` |

### 6.3 Frozen latency decision rule

The v2 sensitivity design contains 14 profiles, three training seeds, and three
matched allocations per seed: 126 profile measurements and 1,008 rank files.
The nine partial-Williams rows balance profile position.

For each non-reference level:

1. Require all nine same-seed, same-allocation matched measurements to be valid.
2. Compute candidate-minus-reference median latency within each allocation.
3. Use a deterministic seed-stratified hierarchical bootstrap: resample the
   three seeds, resample three allocations within each sampled seed, take a
   median within seed, then a median across seeds.
4. Set the effective noise floor to
   `max(0.73553775 ms, largest three-allocation reference range among seeds)`.
5. Set `latency_effect_qualified=true` only when the entire hierarchical 95% CI
   lies below the negative floor or above the positive floor.

Qualification is direction-agnostic: reliably faster and reliably slower
levels both establish that an axis changes inference cost. A separate
`latency_reduction_qualified` annotation identifies the faster branch.
`latency_mode_98pct_suitable` remains independent and never gates expanded
multi-objective eligibility.

If any level qualifies, the future shared search admits the axis's complete
preregistered domain, not only the qualified level or a result-fitted hull. If
no axis qualifies, generation fails closed.

The frozen expanded-search policy always includes:

- `train.optim.lr` in `[1e-5, 5e-4]`;
- `train.optim.weight_decay` in `[1e-5, 1e-3]`.

It uses search seeds `314159`, `271828`, and `161803`, 20 recommendations per
seed, 60 candidates total, training seed `1234`, one shared archive, 98%
latency-mode retention, and no multi-objective floor. The effective latency
noise floor is imported exactly as the selection tolerance; weights and
thresholds cannot be changed after results are observed.

The pre-result revision of the derivation policy has SHA-256
`571818582644fbb60c9474327dcb445b845a791b19e312a887887d534389e7e4`.
Revision 2 binds the approved analysis erratum and preregisters post-front
validation; it explicitly records that the original derivation rule, search
domains, and selection rule did not change. Its final manifest-consumed
SHA-256 is
`03453dc6e04bcc5ca2e0a3eb3043df6059f252bd7b0fcf0fa388f7fff162e324`.

### 6.4 Sensitivity-latency v1 failure audit

The first nine-allocation latency batch produced no timings. The pinned SQSH
reported:

```text
2.11.0a0+a6c236b9fd.nv26.03.46836102
```

while v1 required literal equality with `2.11.0`. Every block stopped at
runtime preflight before the first benchmark subprocess.

| Allocation | TAO job | SLURM job | Node | Root row | `srun` step | Profiles | Rank JSON |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| `seed_001234_allocation_0_row_00` | `ee899aa0-1b57-4545-9f78-9030e6ad872a` | 30943464 | `batch-block7-00339` | `COMPLETED/0:0` | `FAILED/1:0` | 0 | 0 |
| `seed_001234_allocation_1_row_03` | `69d3bd6c-a8b7-46b3-9d05-3c414537ec50` | 30943481 | `batch-block7-03401` | `COMPLETED/0:0` | `FAILED/1:0` | 0 | 0 |
| `seed_001234_allocation_2_row_06` | `335fc12b-4291-4a24-99db-eaa6fb75a649` | 30943494 | `batch-block7-02119` | `COMPLETED/0:0` | `FAILED/1:0` | 0 | 0 |
| `seed_271828_allocation_0_row_01` | `36a9f7f8-2291-4a52-bdbb-1988c6f76a61` | 30943509 | `batch-block7-03295` | `COMPLETED/0:0` | `FAILED/1:0` | 0 | 0 |
| `seed_271828_allocation_1_row_04` | `ccf234ae-c6de-4bde-ba5a-ab8929ff3c6d` | 30943521 | `batch-block7-00339` | `COMPLETED/0:0` | `FAILED/1:0` | 0 | 0 |
| `seed_271828_allocation_2_row_07` | `838246c2-a232-44a0-a90b-6f16760021dd` | 30943525 | `batch-block7-03401` | `COMPLETED/0:0` | `FAILED/1:0` | 0 | 0 |
| `seed_314159_allocation_0_row_02` | `37d1ca0e-1163-4b98-9bac-ee756cee016b` | 30943542 | `batch-block7-02119` | `COMPLETED/0:0` | `FAILED/1:0` | 0 | 0 |
| `seed_314159_allocation_1_row_05` | `6fc686ae-b722-4d9d-b2ba-dd83e5b45a78` | 30943558 | `batch-block7-03295` | `COMPLETED/0:0` | `FAILED/1:0` | 0 | 0 |
| `seed_314159_allocation_2_row_08` | `44343f8b-dbc6-4d5b-a300-700cb66ef0ce` | 30943574 | `batch-block7-00670` | `COMPLETED/0:0` | `FAILED/1:0` | 0 | 0 |

The root rows appeared successful because the SDK's default requeue wrapper
masked the non-timeout `srun` failure. Job-scoped `allocation_result.json`
records and `.0` step states are authoritative. There was no `torchrun`,
benchmark invocation, profile directory, or rank JSON, so no v1 value can enter
analysis.

V2 makes only runtime-contract corrections:

- retain and report the complete PyTorch string, but validate its
  major.minor.patch prefix;
- set `SLURM_USE_REQUEUE=false` so a non-timeout `srun` failure propagates.

The profile schedule, statistical decision rule, benchmark protocol, and
expanded-space derivation remain frozen.

| Evidence | SHA-256 |
| --- | --- |
| Sensitivity latency manifest v1 | `c569f858f4513139292d7189ab5e57f897b8794fdbe5b2dcafc45b0efcd663aa` |
| Failed v1 submission ledger | `f227b6123f762091a81b341bd9b824599e399e801dd2d2fb85ce26a948ba2214` |
| Failed v1 SDK-state snapshot | `9f78e14247a6cbf68221ee2a923432661a27b230c73dba8ace5d14ab9a522fc6` |
| Failure audit JSON | `65d096fed00a4857b6b5f7170fa59a289d2ba8e323b08a79991155fdbac0de3f` |
| Sensitivity latency manifest v2 | `aedc117414b2691c1a70b73fa4e9e0ac123cb4d20dfd9d25dfe2d4aa490d7655` |

### 6.5 V2 sensitivity-latency result

The committed v2 result is complete. It contains 14 profiles × three training
seeds × three matched allocations = 126 valid profile/allocation
measurements. Every measurement contains eight rank records and 4,000 timed
samples, for 1,008 rank files in total. All 126 passed the frozen hardware,
runtime, input, checkpoint, benchmark-protocol, scheduler-identity, and
evidence-integrity gates; there were no retries or result mutations.

The effective practical tolerance remained the historical value:

```text
max(
  historical floor = 0.73553775 ms,
  maximum same-seed reference allocation range = 0.46283150 ms
) = 0.73553775 ms
```

The same-seed reference ranges were `0.21178750 ms` for seed `1234`,
`0.46283150 ms` for seed `271828`, and `0.28123175 ms` for seed `314159`.

Every latency row below summarizes the same nine matched allocations.
Bracketed values are the minimum and maximum allocation statistics. Robust
CV, round range, device range, and bootstrap-CI width are medians of the nine
within-allocation values.

| Profile | Allocation median ms, median [min, max]; range | Allocation p95 ms, median [min, max]; range | Robust CV | Round range ms | Device range ms | Bootstrap-CI width ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reference` | 77.850480 [77.675432, 78.138263]; 0.462831 | 78.221278 [78.008426, 78.408494]; 0.400069 | 0.002465 | 0.071756 | 0.490268 | 0.180408 |
| `num_queries_300` | 77.839945 [77.745868, 78.078378]; 0.332510 | 78.229924 [78.023935, 78.404316]; 0.380381 | 0.001845 | 0.027163 | 0.438119 | 0.123847 |
| `num_queries_450` | 77.846599 [77.654939, 78.178153]; 0.523214 | 78.028248 [77.901645, 78.419266]; 0.517621 | 0.001831 | 0.054036 | 0.344987 | 0.093317 |
| `num_queries_750` | 77.856575 [77.778955, 78.172055]; 0.393100 | 78.103423 [77.962866, 78.491139]; 0.528273 | 0.001950 | 0.051418 | 0.456944 | 0.119180 |
| `num_queries_900` | 77.975581 [77.856863, 78.177543]; 0.320680 | 78.229157 [78.124557, 78.396042]; 0.271485 | 0.001908 | 0.043493 | 0.425614 | 0.103891 |
| `num_select_50` | 77.829609 [77.678319, 78.092528]; 0.414209 | 78.122627 [77.977861, 78.272152]; 0.294291 | 0.002346 | 0.060281 | 0.459609 | 0.199186 |
| `num_select_100` | 77.791613 [77.645191, 78.167030]; 0.521840 | 78.129567 [77.974935, 78.430027]; 0.455092 | 0.002070 | 0.053481 | 0.441444 | 0.130648 |
| `num_select_200` | 77.855110 [77.736421, 78.182713]; 0.446292 | 78.192842 [78.015246, 78.611795]; 0.596549 | 0.002127 | 0.049018 | 0.505804 | 0.138930 |
| `enc_layers_3` | 63.616600 [63.424290, 63.834087]; 0.409797 | 63.865575 [63.579640, 64.206095]; 0.626455 | 0.002264 | 0.024283 | 0.368871 | 0.069892 |
| `enc_layers_4` | 68.451955 [68.251379, 68.605489]; 0.354110 | 68.745372 [68.429252, 69.024749]; 0.595497 | 0.001880 | 0.034951 | 0.365479 | 0.096634 |
| `enc_layers_5` | 73.101013 [72.874398, 73.371647]; 0.497249 | 73.422016 [73.153734, 73.729958]; 0.576225 | 0.002480 | 0.040096 | 0.570166 | 0.155023 |
| `dec_layers_3` | 65.662855 [65.548677, 65.801854]; 0.253177 | 65.925439 [65.714845, 66.139269]; 0.424424 | 0.001970 | 0.023326 | 0.385259 | 0.074462 |
| `dec_layers_4` | 69.715108 [69.678782, 69.846359]; 0.167576 | 69.962947 [69.831032, 70.175989]; 0.344957 | 0.001802 | 0.037998 | 0.439388 | 0.085859 |
| `dec_layers_5` | 73.748757 [73.677204, 74.036833]; 0.359629 | 74.074877 [73.879684, 74.333874]; 0.454190 | 0.001843 | 0.026028 | 0.344270 | 0.093417 |

The paired effect is candidate median latency minus the same-seed,
same-allocation reference latency. Qualification requires the complete
hierarchical 95% interval to lie outside `±0.73553775 ms`; the 98% column is a
separate latency-mode annotation and is not an architecture-axis gate.

| Profile | Seed effects ms: 1234 / 271828 / 314159 | Median effect ms | Hierarchical 95% CI ms | Effect qualified | 98% suitable | Deterministic action |
| --- | ---: | ---: | ---: | :---: | :---: | --- |
| `num_queries_300` | -0.058378 / +0.013447 / -0.009623 | -0.009623 | [-0.198996, +0.111202] | No | No | Exclude query axis |
| `num_queries_450` | -0.076109 / -0.020493 / -0.097944 | -0.076109 | [-0.151213, +0.039890] | No | No | Exclude query axis |
| `num_queries_750` | -0.020133 / +0.033792 / +0.052647 | +0.033792 | [-0.025487, +0.124537] | No | No | Exclude query axis |
| `num_queries_900` | +0.132980 / +0.143963 / +0.097262 | +0.132980 | [+0.025521, +0.181431] | No | Yes | Exclude query axis |
| `enc_layers_3` | -14.277178 / -14.248272 / -14.283632 | -14.277178 | [-14.304176, -14.214302] | Yes | No | Retain value 3 |
| `enc_layers_4` | -9.446314 / -9.457763 / -9.535503 | -9.457763 | [-9.535503, -9.386882] | Yes | No | Retain value 4 |
| `enc_layers_5` | -4.783854 / -4.766617 / -4.770424 | -4.770424 | [-4.863874, -4.712530] | Yes | No | Retain value 5 |
| `dec_layers_3` | -12.195899 / -12.196516 / -12.169440 | -12.195899 | [-12.336410, -12.119992] | Yes | Yes | Retain value 3 |
| `dec_layers_4` | -8.144672 / -8.151055 / -8.108099 | -8.144672 | [-8.262732, -8.023165] | Yes | Yes | Retain value 4 |
| `dec_layers_5` | -4.122297 / -4.101430 / -4.093055 | -4.101430 | [-4.137750, -3.998228] | Yes | No | Retain value 5 |
| `num_select_50` | -0.059954 / -0.045735 / +0.108790 | -0.045735 | [-0.097175, +0.108790] | No | Yes | Exclude select axis |
| `num_select_100` | -0.036462 / -0.030241 / -0.055941 | -0.036462 | [-0.072900, +0.004732] | No | Yes | Exclude select axis |
| `num_select_200` | -0.050978 / +0.044449 / +0.020197 | +0.020197 | [-0.113545, +0.060989] | No | Yes | Exclude select axis |

All encoder- and decoder-depth reductions qualified as reliably faster. The
smallest qualified effect, decoder depth 5 at about `-4.10 ms`, is more than
five times the effective tolerance. Query and selection effects remained near
zero; even query count 900's wholly positive interval lies well inside the
practical-equivalence band. The deterministic outcome is therefore:

- retain the complete supported `model.enc_layers` domain `3–6`;
- retain the complete supported `model.dec_layers` domain `3–6`;
- exclude the complete `model.num_queries` axis;
- exclude the complete `model.num_select` axis.

The screen did not select an AutoML winner and cannot feed a winner override.
Its artifact records `winner_selected=false`,
`feeds_final_selection=false`, and `manual_promotion_permitted=false`.

The immutable ledger was produced by nine one-node/eight-GPU A100 jobs.
Because that launch path did not run the SDK monitor, its durable SDK rows
remained `Pending`; exact read-only `sacct` reconciliation established
`COMPLETED/0:0` for every allocation. The provenance erratum permits that
reconciliation only behind exact job, runtime, artifact, and source identity.

| Evidence | SHA-256 |
| --- | --- |
| `sensitivity_latency_analysis.v2.json` whole file | `33aea1c13ece0ce632587abd16ed6020ecc88c63220f89891a5f30183322eaea` |
| v2 internal `report_sha256` | `40a8bccb6e43b8238c2cf6b47eaf3253e735d82fd160212d12915b3137a3fa79` |
| `latency_sensitivity_report.md` whole file | `352404e3e5f727ac1fe3593f85ff48b3aa93519935865d9c5b4f6761f709ab82` |
| Analysis erratum JSON | `8e19287bf2ffd674f62b21cdaf11e000b0eae1ed8af9d0ada1238491588993f2` |
| Corrected erratum aggregator | `9209e748093e0555fe5cba339327a8216744ec9ca6b9dae276c7041703a409c6` |
| Original manifest-pinned aggregator | `5f5aebd4274c746ec9674f28f978af5d228d98c6ba0af8d76cff8b1742dab967` |
| Immutable nine-job ledger | `b1c170c0d4697463d171cbeca3e4adcbd34cc1cb7429c236f48b58c46c3b6d54` |
| Verified 1,017-file evidence inventory | `a0527c5f687b7660e208a009972cc4c2de5a0f684b1e62316cd7671e9de15021` |

Relevant commits are measurement launch
`cb62ef447704b95980b17aa82604992564b4e71f`, provenance erratum
`e0d41edfd1efcbb374128b03995234ff8f8e623e`, provenance correction
`5578feba03bf78ad8d40de62a0d0b943ca22b740`, SDK evidence inspection
correction `6472954ec6996f3d7872c6dcb6217f7c3b228a61`, and committed measurement
evidence `211d8fd6a5d4e718fdb28a5f57f0483f8bbf4c40`.

## 7. Phase 4: expanded DINO validation

### 7.1 Derived search space

The manifest generator validated the complete sensitivity result against the
preregistered derivation policy and froze this search space before any
expanded-search result existed:

| Parameter | Frozen domain | Role |
| --- | --- | --- |
| `model.enc_layers` | ordered integer levels `{3, 4, 5, 6}` | Qualified inference-cost architecture axis |
| `model.dec_layers` | ordered integer levels `{3, 4, 5, 6}` | Qualified inference-cost architecture axis |
| `train.optim.lr` | continuous `[1e-5, 5e-4]` | Accuracy-influencing training parameter |
| `train.optim.weight_decay` | continuous `[1e-5, 1e-3]` | Accuracy-influencing training parameter |

`model.num_queries=594` and `model.num_select=300` remain fixed reference
values because their complete axes failed the preregistered latency-effect
screen. No qualified-value hull was fitted: once an axis qualified, its
complete preregistered supported domain `3–6` was retained.

The search uses three deterministic Bayesian subarchives with seeds `314159`,
`271828`, and `161803`, exactly 20 sequential recommendations per seed, and
training seed `1234`: 60 candidates total. All modes receive the same
successful union. Latency mode uses 98% accuracy-winner-relative retention;
multi-objective mode has no minimum-accuracy floor. The selection-time
practical tolerance is imported without rounding or tuning as
`0.73553775 ms`. Manual candidate injection, result-driven range changes, and
winner override are all forbidden.

#### 7.1.1 Expanded-search v1 failure and exclusion

The first expanded launch used manifest v1 at commit
`fae47d3406ea29bfc03893f9808b50958eef70c6`. Its whole-file manifest hash was
`57e331686b8896989263a39f72edb69543fc58833f20a1e6e698c31f34d2e8be`;
the launch-pinned runner hash was
`211c926065ee63a9d7476e312d2e89ee48b9f0189bb6330ce632c3604d4af668`.
The execution is invalid and excluded from every selection and hypothesis
result.

All three algorithm-generated rec0 training and evaluation allocations
completed with scheduler exit `0:0`. TAO wrote a finite mAP50 value, but wrote
it as a JSON string:

| Candidate | Training TAO / SLURM | Evaluation TAO / SLURM | Exact serialized `test_mAP50` | Evaluation status SHA-256 |
| --- | --- | --- | --- | --- |
| `seed_161803_rec_0` | `959f0fe6-d0c9-48d9-a0d8-aac26ebd485a` / 30950521 | `04792472-d79e-4eaf-917c-f332c3c9d487` / 30951491 | `"0.499933605841208"` | `8fa3a749c4999423e9bee15f3fbd1cd837515e93a9f0cf73c085a93ebb400faf` |
| `seed_271828_rec_0` | `b3933f47-fa9b-405b-aa0b-2555dec8ced7` / 30950520 | `f50e5737-3a93-4937-8c6a-e39fcf56ec66` / 30951528 | `"0.5175292656942001"` | `070237fe94f85c715f5495d4a938ce71778fd299b4711b7c8f6091929595c66f` |
| `seed_314159_rec_0` | `7be04920-5c71-43b1-bdf4-d29c00d56405` / 30950522 | `aec669db-9c32-4e5a-b3b0-4db023d3b62c` / 30951502 | `"0.5728509799562066"` | `1adcf675a0513b7d13eec87447c0deb66225ffc65cfd088b2614d07e66f41c05` |

The v1 call path was `read_status_map50()` → `finite_number()`.
`finite_number()` accepted only native `int`/`float` values and raised on a
string. The regex fallback ran only when the status reader returned `None`,
not when it raised. Each candidate therefore stopped before
`launch_latency_benchmark()` with:

```text
ContractError: evaluation mAP50 must be a finite number
required_eval_fn_failed:evaluation mAP50 must be a finite number
```

Each rec0 candidate was recorded as `training_or_measurement_failure` with
`metric=null`; no selection-time latency job exists. Each Bayesian brain has
one proposed `X` and zero observed `y` values. Thus v1 has no usable Bayesian
response, no successful candidate with a complete objective pair, no seed
archive, no combined selection, no candidate table, and no integrity audit.
The exact remote strings above diagnose the parser failure; they are not
manually imported objective values.

Controller shutdown interrupted the resulting rec1 work:

| Search seed | Rec1 TAO job | SLURM job | Terminal state |
| ---: | --- | ---: | --- |
| 161803 | `21acba1e-4cbb-43e4-97dc-fd1d4193decc` | 30952017 | SDK `Canceled`; SLURM `CANCELLED/0:15` |
| 271828 | `8676986e-4620-4cf4-a9bc-75dc5b77957e` | — | SDK `Canceled`; no SLURM assignment |
| 314159 | `85c92e03-0e45-41b7-813e-56bd5f7a5686` | 30952020 | SDK `Canceled`; SLURM `CANCELLED/0:15` |

The controller group finalized at
`2026-07-28T04:07:55.739442003Z`, with exit code 1 for every seed. The
post-stop audit found empty active-job ledgers, no assigned v1 job in
`squeue`, and no v1 runner process. All six SDK-routed v1 training roots are
marked deleted and are remotely absent; only the three evaluation roots remain
as failure evidence. The complete audit is
`expanded_search_v1_failure_audit.md`, whole-file SHA-256
`189f785921a551a729c5abe17b8de44c548bda6e7aac7b95447a8a1896ab74a4`.

The separate credential-containment change at commit
`ce0797c4896dd0e79e9e70c2222bcc63217891b3` changed staging-file access modes
only. It did not change a recommendation, metric, latency value, selector
input, or winner, and it did not cause or repair this parser failure.

#### 7.1.2 Corrected v2 manifest, runtime contract, and launch

Commit `e4b6a412545614668affd371a82231e090998ec0` introduced a strict finite
numeric metric parser that accepts a native finite number or a strict JSON
number string, while rejecting booleans, empty values, junk, NaN, and
infinity. It also requires a clean v2 runtime: no v1 controller, brain,
candidate, SDK, or workspace state is resumable or reusable.

Commit `b1a0ae235be53ba3ced7e4c880cb0be1f6b8157d` froze the corrected manifest
before v2 launch:

| Corrected expanded-search artifact | Identity |
| --- | --- |
| Runtime-fix commit | `e4b6a412545614668affd371a82231e090998ec0` |
| V2 manifest-freeze / evidence-cutoff commit | `b1a0ae235be53ba3ced7e4c880cb0be1f6b8157d` |
| Runtime erratum whole-file SHA-256 | `a89b5816b45e1df9c6286c25ccbe8314daee53843decc4400882ec33f10ffa17` |
| V2 manifest internal self-hash | `910744ae2fead7e4e2e9a53fc672baef1ac43307e3979671b2b876fff422de96` |
| V2 manifest whole-file SHA-256 | `9ac29e1aa07167a040d217fdab2d3cfdea0baad690dc95a70f2fe6715908793a` |
| Corrected runner SHA-256 | `0eb1948d4fb887b9c3fe938d60865ebb4ef86ae00d9ca80aa0d42b465a073073` |
| Manifest generator SHA-256 | `54f418003c96df4fea4e1b6e1cba2747d15a5592584d1a1fa997ecd675514c9c` |
| Derived schema SHA-256 | `9eb137c5f7aec7c28468ca9b642a29d1e96b522b362bf437938bdce2882c8681` |
| V2 runtime marker internal contract hash | `5acb00ddd33ddbba69491c4a9c2bef87148a2e43455e59aa1032242d5d979f5f` |

The runtime marker binds the exact v2 manifest to
`runtime/expanded_search_v2`, records `v1_runtime_reused=false` and
`valid_objective_observations_reused=0`, and supersedes
`runtime/expanded_search`. This is a fresh deterministic restart from rec0,
not `--resume` against v1.

The read-only remote preflight reported `all_verified=true`. It verified the
committed, clean corrected runner; exact SQSH
`88ba75e3a8eb9524fc0dbf026f2ea5da2c68696ae8d918b0afde5e0384ca641e`;
PTM `7a391fb84a18714b60258becdb512594ec54faff5dccbf17ca53c5d902137512`;
train and validation annotation hashes
`7401a1245dc0b691c40f9f53cf4f46f9b96a3e0bc3dcfd357de038074acc1994`
and
`9b715b689e9a17588805faad26ed94597886d28ac687438dcb778de433f997af`;
and both staged image directories. Secret names were audited without recording
secret values.

The three fresh seed controllers generated rec0 algorithmically and launched
concurrently through the TAO SDK. Each scheduler job requests one node and
eight GPUs on `polar3` and runs the pinned TAO 7.0.1 PyTorch SQSH:

| Search seed | Candidate | Encoder / decoder | Learning rate | Weight decay | TAO SDK job | SLURM job | Node at launch snapshot |
| ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| 314159 | `seed_314159_rec_0` | 5 / 6 | 0.0002156899238307862 | 0.00010770493619675102 | `fea7a910-fe49-48ef-98cd-2cfb86edcf7e` | 30955312 | `batch-block7-01741` |
| 271828 | `seed_271828_rec_0` | 6 / 6 | 0.0000459777499171801 | 0.0006077207436969115 | `cefd45b8-cb08-4ba3-a4f0-ca39b2d4f85d` | 30955313 | `batch-block7-03161` |
| 161803 | `seed_161803_rec_0` | 3 / 5 | 0.0002098775727573059 | 0.0006123641159628601 | `82bd8395-4d66-40e5-836b-f6a7ec75bc61` | 30955314 | `batch-block7-03312` |

The rec0 parameter mappings intentionally repeat under the same deterministic
seeds; the TAO jobs, SLURM allocations, SDK databases, workspaces, and every
future measurement are new. Parameter reproducibility is not v1 state reuse.

At the launch evidence snapshot, `sacct` reported all three as `RUNNING` with
one node, eight allocated GPUs, partition `polar3`, and exit field `0:0`.
That statement is retained as launch-time provenance; the same three jobs later
completed as the rec0 training jobs reported below.

#### 7.1.3 Live partial evidence through rec1

As of `2026-07-28T05:56:16Z`, rec0 and rec1 have completed successfully for
all three deterministic search seeds. Every one of the 18 associated training,
evaluation, and selection-time latency SLURM allocations is
`COMPLETED/0:0`; every accepted latency record passes the frozen validity
checks, has retry count zero, and has `launch_uncertain=false`.

| Candidate | Encoder / decoder | Learning rate | Weight decay | mAP50 | Median ms | p95 ms | Bootstrap median CI95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `seed_314159_rec_0` | 5 / 6 | 0.0002156899238307862 | 0.00010770493619675102 | 0.5719041815412683 | 74.61342325 | 74.8451050 | [74.56798700, 74.66867200] |
| `seed_314159_rec_1` | 3 / 3 | 0.00043836528814622386 | 0.0006459840646532157 | 0.5518847844262931 | 52.39449225 | 52.6443065 | [52.32853825, 52.46170325] |
| `seed_271828_rec_0` | 6 / 6 | 0.0000459777499171801 | 0.0006077207436969115 | 0.5140622451913158 | 79.26377375 | 79.5854720 | [79.15900350, 79.36380150] |
| `seed_271828_rec_1` | 5 / 3 | 0.0004417551531468059 | 0.0004796687699134978 | 0.6044653164228678 | 61.82193000 | 62.0668055 | [61.79774925, 61.88626850] |
| `seed_161803_rec_0` | 3 / 5 | 0.0002098775727573059 | 0.0006123641159628601 | 0.5305309558679956 | 60.73003050 | 61.1440695 | [60.71404825, 60.75802350] |
| `seed_161803_rec_1` | 5 / 5 | 0.00030565624727243724 | 0.0005057619353603205 | 0.5527822169224204 | 70.18545575 | 71.2943900 | [70.12402475, 70.38869425] |

The exact job lineage is:

| Candidate | Training TAO / SLURM | Evaluation TAO / SLURM | Latency TAO / SLURM | Latency node |
| --- | --- | --- | --- | --- |
| `seed_314159_rec_0` | `fea7a910-fe49-48ef-98cd-2cfb86edcf7e` / 30955312 | `ddd196dc-a5d3-4f4e-b2d5-6c94efc62813` / 30955720 | `0e40b43d-4c9c-4fa8-ad07-550c181d60f2` / 30955882 | `batch-block7-03471` |
| `seed_314159_rec_1` | `18b851c1-9360-46aa-a46b-53874368a999` / 30956226 | `559ac3bd-3408-4b6d-91cb-45b8efa78a2a` / 30956537 | `332346df-f363-4c07-ab76-ec809a36e05a` / 30957504 | `batch-block7-03289` |
| `seed_271828_rec_0` | `cefd45b8-cb08-4ba3-a4f0-ca39b2d4f85d` / 30955313 | `6289c304-1a6b-40d7-a14c-28db0e9a5e08` / 30955721 | `dc0442a1-6c24-447a-adcd-e425b8f07bed` / 30955883 | `batch-block7-01955` |
| `seed_271828_rec_1` | `ebfc2a5f-5e1a-42f9-889c-c399b7075b1f` / 30956154 | `46fbda9b-19ba-4a19-bd19-be60d1125608` / 30956572 | `0d893507-c3d8-4b69-b6f0-47bc1004c359` / 30957506 | `batch-block7-01817` |
| `seed_161803_rec_0` | `82bd8395-4d66-40e5-836b-f6a7ec75bc61` / 30955314 | `b24a02b4-6312-457b-a83f-ef8a0f2a9af8` / 30955702 | `390fabdd-520f-4013-8094-3bfb4c71dbd5` / 30955846 | `batch-block7-00124` |
| `seed_161803_rec_1` | `8633a6d8-4e2d-453c-be09-9c51fa31be4b` / 30956062 | `0b21e426-e94d-4f52-bcd3-8be69cfb3236` / 30956528 | `b1d21f55-5b60-4e20-adbc-3d884174079c` / 30957416 | `batch-block7-01817` |

These are selection-time observations on independent, unmatched allocations.
Their latency values must not be interpreted as allocation-stable pairwise
advantages. Each objective pair is legitimately returned to its seed's
sequential Bayesian brain to generate later recommendations, and every record
states `winner_selected_during_measurement=false`. The production
`tao_automl.selection.analyze_archive` result is unavailable until all three
20-record archives seal and reconcile.

Recommendations 2–19 remain **PENDING LIVE EVIDENCE**. No partial six-point
front, interim accuracy winner, interim latency winner, interim
multi-objective winner, Pareto rank, or hypothesis conclusion is reported.

### 7.2 Complete expanded candidate table

**PENDING LIVE EVIDENCE**

The final table must include every successfully evaluated candidate:

- candidate ID and full relevant parameter mapping;
- search seed and training seed;
- mAP50;
- original selection-time median and p95 latency;
- within-allocation and between-allocation variability fields;
- latency-mode feasibility;
- multi-objective eligibility;
- global and eligible Pareto ranks;
- exact `dominated_by` relationships;
- normalized accuracy and latency regrets;
- augmented-Chebyshev score;
- ideal distance and balance gap;
- all tie-breaking values;
- accuracy, latency, and multi-objective winner flags.

Failed, missing, NaN, or infinite candidates must be listed separately and
must not enter ranking.

### 7.3 Final Pareto-front matched remeasurement

**PENDING LIVE EVIDENCE**

No final-front manifest, launch ledger, allocation, or measurement exists yet.
The following is the completed, tested implementation contract, not an
experimental result.

The implementation snapshot audited for this report is:

| Committed source or test at `2453079abbe93e2fd854dcf2a910256dfd164669` | SHA-256 |
| --- | --- |
| `post_front_matched_manifest_generator.py` | `43f6928e5dcb7677a186e5e902ee2049ba18471ff0c37c2ee55b2934121effe8` |
| `post_front_matched_launcher.py` | `13b15540e7992091e57f35ebebaf20093d32be580879ff24e06b8c5202708fd9` |
| `post_front_matched_block_runner.py` | `8a82ea7d9e0a06c617c94ae83c3cf5b333ed887d051a9fe816c3cc138c37aae6` |
| `post_front_matched_aggregator.py` | `00187790203f75dde6ff91b7498bc8cb3ece79683b6884668f1045726cb70a27` |
| `test_post_front_matched_tools.py` | `ac1870da2b6f66dd7e162caf56b5215ab013341a7e1d08e56562af121a56ddb9` |
| `test_post_front_matched_launcher_recovery.py` | `8f020f7a36222ac240830f9b3bb61984e80e4ea65c8c344de498062b4daecae4` |
| `test_post_front_complete_invalid_recovery.py` | `6d1082d30d23db286953c310e7ba7dacdb135b3221be88c077f26cbb5e24aebe` |

These are tracked, committed identities, but they do not by themselves
authorize a launch. The manifest generator intentionally refuses to create
the future immutable post-front manifest until the expanded archive and all
source inputs are complete.

#### 7.3.1 Manifest authority, candidate derivation, and selector isolation

The final prelaunch review found one blocker in the earlier hardening
snapshot. A manifest could be edited in a runtime or latency-protocol field
and then have `manifest_sha256` recomputed. Its whole-file and internal hashes
would be self-consistent, and fragmentary schema checks did not make that
self-hash an authority for all launch-affecting semantics.

The launcher now reconstructs the complete canonical manifest through
`post_front_matched_manifest_generator.build_manifest` from the exact pinned
expanded manifest, combined selection, candidate table, integrity audit,
runtime contract, selector stack, and post-front tool sources.
`require_exact_reconstructed_manifest` then requires whole-object equality,
not merely equality of selected fields or digests. This source validation
runs before config generation and every dry-run, fresh launch,
incomplete-submission resume, and allocation replacement. The read-only
aggregator calls the same validation before job inspection or result
aggregation. Thus a self-rehashed drift in SQSH, SDK, SLURM, hardware, or
latency-protocol settings fails before it can affect execution or analysis.

Manifest generation fails closed until the expanded run has exactly 60
terminal candidate-table records plus the final combined selection and
integrity audit. It then:

1. loads every successful selection-time objective record;
2. imports the manifest-pinned production objective parser and
   `tao_automl.selection.analyze_archive` implementation;
3. independently replays that frozen archive under candidate-table, reverse,
   and candidate-ID order;
4. requires the replayed analysis, every candidate audit, and the global
   rank-zero front to exactly match the combined-selection artifacts; and
5. includes every and only global-rank-zero candidate, in ascending UTF-8
   candidate-ID order, with its exact checkpoint and complete resolved model
   mapping.

The production selector is therefore invoked during frozen-archive source
validation. It is not invoked on post-front measurements. The replay result is
used only to prove candidate-set integrity before those measurements are
loaded. The original accuracy, latency, and multi-objective selection snapshot
is copied unchanged into the post-front manifest; remeasurement cannot select,
reselect, replace a selection-time objective, or override a winner.

An independent read-only re-audit after the exact-reconstruction change found
no remaining prelaunch or aggregation blocker in the implemented contract.
This is a code-path conclusion, not live experimental evidence: no post-front
manifest was generated and no TAO or SLURM post-front job was launched.

#### 7.3.2 Frozen Williams-row projection

For \(n\) canonical final-front candidates, the implementation constructs the
Williams base row:

\[
[0,1,n-1,2,n-2,\ldots].
\]

It produces all \(n\) modulo-addition rotations and, when \(n\) is odd, also
their reversals. The complete design therefore contains \(R=n\) rows for even
\(n\) or \(R=2n\) rows for odd \(n\). Exactly six allocations are frozen by:

\[
\operatorname{row}(k)=\left\lfloor\frac{kR}{6}\right\rfloor,\qquad
k\in\{0,\ldots,5\}.
\]

This is not an adaptive choice and is not generally “the first six rows.”
Every selected row must be a complete candidate permutation. The manifest
records the exact row indices, every order, per-candidate position counts,
maximum position-count imbalance, and ordered immediate-adjacency counts in a
canonical schedule hash. Perfect one-per-position balance is asserted only
for the special six-candidate/six-row case; for another front size the audit
reports the actual projection rather than making a stronger balance claim.

Each of the six independent allocations uses one node and eight A100 GPUs and
benchmarks the complete front sequentially. An incomplete allocation is
discarded in full and may be replaced only by rerunning the same complete
candidate order under a new TAO job identity. Partial candidate measurements
from a failed block are never combined.

#### 7.3.3 Job-private staging and runtime containment

Each allocation command embeds a deterministic gzip/base64 bundle containing
the exact block runner, benchmark, statistics module, plan, and per-candidate
configs. At runtime it creates a `TAO_JOB_ID`-scoped `mktemp` directory under
`TMPDIR`, enforces an owner-controlled non-symlink `0700` root and parent
directories, writes owner-only `0600` regular files with `O_EXCL` and
`O_NOFOLLOW` where supported, and validates the compressed bundle, decoded
bundle, file set, and every post-write SHA-256 before execution.

The block runner revalidates the staging directory, source hashes, complete
model mappings, checkpoint paths and hashes, eight-GPU topology, runtime, and
exact `rank_0.json` through `rank_7.json` result set for each candidate. Output
must be exactly rooted at `$TAO_RESULTS_ROOT/$TAO_JOB_ID`; allocation and
candidate run labels keep every raw record job-private and non-overlapping.

The runtime is pinned to the already-built TAO 7.0.1 PyTorch SQSH and its
SHA-256. `SLURM_USE_SQSH=false` prevents SDK conversion of that prebuilt image,
and `SLURM_USE_REQUEUE=false` prevents a status observation or failed block
from silently moving to another allocation. Before every submission, the
launcher validates the remote SDK base plus `sbatch`, `env`, `meta`,
`entrypoints`, `specs`, `slurm-logs`, and `results` as owner-controlled,
non-symlink `0700` directories.

The secrets file must be an owner-owned, non-symlink regular file with mode
`0600` or stricter. Conflicting ambient values fail closed. Evidence records
only loaded key names and containment hashes, always with
`secret_values_recorded=false`.

#### 7.3.4 Side-effect-free inspection and crash reconciliation

Read-only status inspection calls the SLURM handler with
`allow_retry=false`, or consumes an already durable terminal status. It hashes
and compares the scheduler job ID, failed-ID lineage, retry count, submission
attempt flag, launch-uncertainty state, launch token, and prelaunch scheduler
ID before and after inspection. Aggregation additionally requires one exact
`sacct` row per effective job, `COMPLETED/0:0`, and node/hostname agreement.
Inspection therefore cannot submit, retry, requeue, or change scheduler
identity.

All mutating launch/recovery paths share a nonblocking owner-only file lock.
Before each `create_job`, the launcher atomically writes an intent containing
the allocation, exact command hash, and complete durable SDK-job-ID snapshot.
After a crash, resume requires the exact ledger hash and launch contract and
reconciles the durable set:

- a zero-job delta proves absence before one exact submission;
- a one-job delta is adopted only after image, job-scoped results URI, backend
  identity, command/entrypoint hash, scheduler identity, and launch certainty
  all match;
- a delta larger than one or any unresolved launch uncertainty blocks;
- a proven pre-scheduler orphan is terminalized and audited before one
  replacement; and
- a durably `Error` or `Canceled` block is superseded only as a complete block
  with a new identity and `partial_measurements_reused=false`.

Durable scheduler success is not treated as proof that the measurement bundle
is semantically usable. If all six jobs are `Complete` /
`COMPLETED/0:0` but result fetching or semantic aggregation fails, the
read-only aggregator writes immutable invalidation evidence bound to the exact
manifest, schedule, complete-ledger whole-file/internal hashes and revision,
prior TAO/SLURM identity, command and block-plan hashes, deterministic failure
digest, and the hashes of any available artifacts. Replacement is authorized
only when the deterministic failure is attributable to exactly one
allocation. Zero-attribution or multi-allocation failures are recorded with
`replacement_blocked=true`; an operator cannot supply an allocation ID to
override that result.

A `Complete` job can then be superseded only by explicitly providing the exact
whole-file and internal invalidation-evidence hashes together with the exact
current ledger hash. The launcher revalidates every binding, discards the
entire implicated block, preserves the old job and evidence in the
supersession chain, writes a create-once replacement intent before
`create_job`, and never reuses available partial measurements. A Complete job
without that aggregator proof cannot be replaced.

If an immutable successful `post_front_matched_analysis.json` already exists,
its validated manifest and complete-ledger bindings veto Complete-job
replacement. Stale invalidation evidence cannot reopen an allocation set that
has already produced a valid immutable analysis.

Every recovery event, superseded identity, parent-ledger hash, and replacement
intent is retained. Duplicate TAO or SLURM identities and concurrent launch
operations fail closed.

#### 7.3.5 Pairwise inference contract

The aggregator requires the complete six-allocation-by-front-candidate
measurement matrix. For each unordered pair it forms six allocation-matched
median differences and six matched p95 differences. The preregistered
10,000-resample paired percentile bootstrap remains descriptive only.

A directional claim requires both:

1. a one-sided exact paired sign-flip permutation test after shifting by the
   relevant `±0.73553775 ms` practical-tolerance boundary, with
   \(p\le0.05\); and
2. all six paired differences strictly beyond that same boundary in the
   claimed direction.

With six allocations the exact randomization enumerates all \(2^6=64\) sign
assignments. Claims are pairwise only. There is no multiplicity adjustment,
so unadjusted pairwise evidence never establishes a simultaneous or stable
total order. Descriptive sorting and bootstrap intervals are reported as such,
not promoted into direction claims.

When the expanded archive is complete, remeasure every final rank-zero
candidate under this frozen contract. Preserve original selection-time
measurements and the algorithm-selected winner. Report remeasurement only as
stability evidence for the hypothesis verdict.

### 7.4 Final combined selection and integrity audit

**PENDING LIVE EVIDENCE**

Required artifacts:

- final combined selection JSON;
- exact shared-archive digest;
- algorithm configuration and source hash;
- candidate-order invariance proof;
- immutable launch/submission ledgers;
- complete TAO/SLURM identity mapping;
- source, data, PTM, SQSH, checkpoint, and raw-rank hashes;
- proof that no manual candidate injection, promotion, or winner override
  occurred.

## 8. Final mode comparison

**PENDING LIVE EVIDENCE — do not infer from Phase 1**

| Mode | Candidate | Accuracy | Stable median latency | Eligibility rule | Pareto status | Selection reason |
| --- | --- | ---: | ---: | --- | --- | --- |
| Accuracy | PENDING | PENDING | PENDING | All valid candidates | PENDING | Highest valid mAP50, algorithm-selected |
| Latency | PENDING | PENDING | PENDING | mAP50 ≥ 98% of accuracy winner | PENDING | Lowest statistically stable latency among feasible candidates |
| Multi-objective | PENDING | PENDING | PENDING | All valid candidates unless the frozen expanded manifest explicitly sets an independent floor | PENDING | Minimum normalized augmented-Chebyshev score on eligible rank zero |

The completed table must additionally report:

- accuracy delta from the accuracy winner;
- latency delta from the accuracy winner;
- latency delta from the constrained-latency winner;
- whether each latency delta exceeds the matched uncertainty tolerance;
- whether the multi-objective point is strictly between the extremes;
- whether it remains distinct after applying accuracy and latency tolerances.

## 9. Hypothesis verdict

### Evidence available now

| Question | Interim result | Evidence |
| --- | --- | --- |
| Does accuracy mode select the highest-accuracy valid candidate? | Supported on the frozen archive | `seed_271828_rec_5`, mAP50 `0.6359897329231639`. |
| Does latency mode select the fastest candidate satisfying 98% retention? | Supported on the frozen archive, but degenerate | Only `seed_271828_rec_5` satisfies the threshold. |
| Can no-floor multi-objective mode select a nondominated archive compromise algorithmically? | Yes, for frozen objective values | The selector returns global-rank-zero `seed_271828_rec_1` with no manual override. |
| Is that historical compromise latency-distinct and allocation-stable? | No evidence of distinction | All six historical global-front candidates are practically equivalent under matched six-allocation analysis. |
| Did the sensitivity screen establish supported DINO axes with stable latency effects? | Yes | Encoder and decoder depth effects are well beyond the frozen `0.73553775 ms` tolerance across matched allocations. |
| Does the expanded supported DINO space contain a stable intermediate point? | **PENDING LIVE EVIDENCE** | Requires the launched expanded search and final-front matched remeasurement. |

The offline replay corrects mode semantics, but it does not satisfy the full
hypothesis criteria because the relative latency ordering is not stable.

### Final classification

**PENDING LIVE EVIDENCE**

Do not classify the overall hypothesis until the expanded search and matched
post-front validation finish. The final classification must be one of:

- fully supported;
- partially supported;
- not supported;
- inconclusive because the supported DINO search space did not produce a
  stable intermediate Pareto candidate.

If the algorithm chooses an extreme despite stable intermediate points, report
the frozen normalization bounds and scores. Do not change weights or floors
after seeing the result.

## 10. Reproducibility

### 10.1 Source, data, model, and runtime

| Repository | Branch | Required commit |
| --- | --- | --- |
| `~/tao-automl` selection core | `rarunachalam/pre-platform-sdk-removal-20260714` | `83d9d7ecc783724f674cb954f9fbb6c91ea8b0eb` |
| `~/tao-automl` sensitivity evidence | same | `211d8fd6a5d4e718fdb28a5f57f0483f8bbf4c40` |
| `~/tao-automl` expanded preregistration | same | `67ad2cfd6662c84871d28cd0106ec8cc143aa4b7` |
| `~/tao-automl` excluded v1 manifest freeze | same | `fae47d3406ea29bfc03893f9808b50958eef70c6` |
| `~/tao-automl` corrected metric/runtime implementation | same | `e4b6a412545614668affd371a82231e090998ec0` |
| `~/tao-automl` v2 manifest freeze / evidence cutoff | same | `b1a0ae235be53ba3ced7e4c880cb0be1f6b8157d` |
| `~/tao-sdk` | same | `3d3e1adc1849493d29dc926cb99492417e3a9250` |
| `~/tao-skills-external` | same | `18f831c7c83b424861a60353fb735dd80efcfded` |
| TAO PyTorch source | tag `7.0.1` | `1ac00f8e9c511591e6e1cfb048c1bad9101b3d32` |

```text
dataset source:
  s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/
staged dataset:
  /lustre/fsw/portfolios/edgeai/users/rarunachalam/data/tao_od_synthetic_full_dino_coco
train annotation SHA-256:
  7401a1245dc0b691c40f9f53cf4f46f9b96a3e0bc3dcfd357de038074acc1994
validation annotation SHA-256:
  9b715b689e9a17588805faad26ed94597886d28ac687438dcb778de433f997af
dataset classes:
  5 output classes; evaluated category IDs 1, 2, 3, 4

PTM:
  /lustre/fsw/portfolios/edgeai/users/rarunachalam/ptm/pretrained_dino_coco/
  dino_resnet_50_trainable_v1.0/dino_resnet50_ep12.pth
PTM SHA-256:
  7a391fb84a18714b60258becdb512594ec54faff5dccbf17ca53c5d902137512

SQSH:
  /lustre/fsw/portfolios/edgeai/users/rarunachalam/
  nvcr.io_nvidia_tao_tao-toolkit_7.0.1-pyt.sqsh
SQSH SHA-256:
  88ba75e3a8eb9524fc0dbf026f2ea5da2c68696ae8d918b0afde5e0384ca641e

SLURM:
  partition polar3
  account edgeai_tao-ptm_image-foundation-model-clip
  one node, eight NVIDIA A100-SXM4-80GB GPUs per job
precision:
  FP32, TF32 disabled
```

The local Python environment is:

```text
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314
```

Credentials are loaded from `~/.tao/config.env` without printing values.

### 10.2 Deterministic offline replay

```bash
cd ~/tao-automl
source /localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/activate

python experiments/dino_moo_phase2_20260728/replay_phase1_archive.py --check
sha256sum experiments/dino_moo_phase2_20260728/phase1_offline_replay.json
```

### 10.3 Historical matched-latency aggregation

This command reads the already finalized six-block ledger and SDK state. It
does not feed selection:

```bash
cd ~/tao-automl
set -a
source ~/.tao/config.env >/dev/null 2>&1
set +a

python experiments/dino_moo_phase2_20260728/aggregate_matched_latency.py \
  --report /tmp/matched_pareto_latency_comparison.reproduced.json

sha256sum \
  experiments/dino_moo_phase2_20260728/matched_pareto_latency_comparison.json \
  /tmp/matched_pareto_latency_comparison.reproduced.json
```

The regenerated file has a new `checked_at_utc`; compare the substantive
analysis and integrity fields rather than expecting a byte-identical hash.

### 10.4 One-factor training/accuracy artifact verification

```bash
cd ~/tao-automl
source /localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/activate

python experiments/dino_moo_phase2_20260728/sensitivity_training_workflow.py \
  validate

sha256sum \
  experiments/dino_moo_phase2_20260728/one_factor_sensitivity_manifest.v1.json \
  experiments/dino_moo_phase2_20260728/sensitivity_training_checkpoints.v1.json \
  experiments/dino_moo_phase2_20260728/sensitivity_training_accuracy.v1.json
```

### 10.5 Read-only v1 failure audit

```bash
set -a
source ~/.tao/config.env >/dev/null 2>&1
set +a
audit_host="${SLURM_HOSTNAME%%,*}"

job_ids='30943464,30943481,30943494,30943509,30943521,30943525,30943542,30943558,30943574'
ssh -o BatchMode=yes "${SLURM_USER}@${audit_host}" \
  "sacct -j ${job_ids} --noheader --parsable2 \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NodeList,Elapsed,Start,End"

python -m json.tool \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_v1_failure_audit.json \
  >/dev/null
sha256sum \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_v1_failure_audit.json
```

### 10.6 V2 sensitivity-latency commands

The preregistered launch command is:

```bash
cd ~/tao-automl
set -a
source ~/.tao/config.env >/dev/null 2>&1
set +a

python experiments/dino_moo_phase2_20260728/sensitivity_latency_launcher.py \
  --submit-blocks \
  --verify-remote \
  --manifest \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_manifest.v2.json \
  --runtime-dir \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency_v2 \
  --checkpoint-artifact \
  experiments/dino_moo_phase2_20260728/sensitivity_training_checkpoints.v1.json \
  --checkpoint-artifact-sha256 \
  20188a8858a9329ce4b861730ad3b0b2f6185389c8af1b02ad29284e5ed1b012 \
  --acknowledgement \
  USER_AUTHORIZED_CONCURRENT_9X8GPU_SLURM_DINO_SENSITIVITY_LATENCY_V2_20260728
```

The completed analysis is reproduced into a new output path with the
provenance-safe erratum aggregator:

```bash
python experiments/dino_moo_phase2_20260728/sensitivity_latency_aggregate_erratum.py \
  --analysis-erratum \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_analysis_erratum.v1.json \
  --analysis-erratum-sha256 \
  8e19287bf2ffd674f62b21cdaf11e000b0eae1ed8af9d0ada1238491588993f2 \
  --manifest \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_manifest.v2.json \
  --checkpoint-artifact \
  experiments/dino_moo_phase2_20260728/sensitivity_training_checkpoints.v1.json \
  --checkpoint-artifact-sha256 \
  20188a8858a9329ce4b861730ad3b0b2f6185389c8af1b02ad29284e5ed1b012 \
  --accuracy-artifact \
  experiments/dino_moo_phase2_20260728/sensitivity_training_accuracy.v1.json \
  --accuracy-artifact-sha256 \
  459da2ebe557ec26947dc723b2864f2bc31880ae3181ad1216c3a47825ec466b \
  --submission-ledger \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency_v2/block_submissions.json \
  --submission-ledger-sha256 \
  b1c170c0d4697463d171cbeca3e4adcbd34cc1cb7429c236f48b58c46c3b6d54 \
  --sdk-state \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency_v2/slurm_state.json \
  --evidence-snapshot \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency_v2/evidence_snapshot \
  --output /tmp/sensitivity_latency_analysis.reproduced.json
```

This revalidates the exact 1,017-file evidence inventory. Wall-clock metadata
and the output path may differ; samples, statistics, six qualified profiles,
and the deterministic axis decision must not.

### 10.7 Corrected expanded-search v2 derivation and launch

```bash
result='experiments/dino_moo_phase2_20260728/sensitivity_latency_analysis.v2.json'
result_sha='33aea1c13ece0ce632587abd16ed6020ecc88c63220f89891a5f30183322eaea'

python experiments/dino_moo_phase2_20260728/expanded_search_manifest_generator.py \
  --policy \
  experiments/dino_moo_phase2_20260728/expanded_search_derivation_policy.v1.json \
  --sensitivity-result "${result}" \
  --sensitivity-result-sha256 "${result_sha}" \
  --output \
  experiments/dino_moo_phase2_20260728/expanded_search_manifest.v2.json

manifest='experiments/dino_moo_phase2_20260728/expanded_search_manifest.v2.json'
manifest_sha='9ac29e1aa07167a040d217fdab2d3cfdea0baad690dc95a70f2fe6715908793a'

python experiments/dino_moo_phase2_20260728/expanded_search_runner.py \
  --dry-run \
  --manifest "${manifest}" \
  --manifest-file-sha256 "${manifest_sha}" \
  --runtime-dir \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2 \
  --report \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2/dry_run.json \
  --verify-remote
```

Verify the frozen and runtime-bound identities without reading mutable
candidate results:

```bash
sha256sum "${manifest}" \
  experiments/dino_moo_phase2_20260728/expanded_search_runner.py \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2/runtime_contract.v2.json

jq '{
  manifest_file_sha256,
  manifest_internal_sha256,
  contract_sha256,
  target_runtime_path,
  v1_runtime_reused,
  valid_objective_observations_reused
}' \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2/runtime_contract.v2.json

jq '.remote_checks.all_verified' \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2/dry_run.json
```

The exact fresh-launch command corresponding to v2 SLURM jobs
`30955312`, `30955313`, and `30955314` was:

```bash
python experiments/dino_moo_phase2_20260728/expanded_search_runner.py \
  --launch \
  --manifest "${manifest}" \
  --manifest-file-sha256 "${manifest_sha}" \
  --runtime-dir \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2 \
  --report \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2/dry_run.json \
  --verify-remote \
  --acknowledgement \
  USER_AUTHORIZED_3X8GPU_SLURM_DINO_EXPANDED_SEARCH_20260728
```

Do not repeat this fresh-launch form while those v2 controllers are live.
Recovery, if required after a controller interruption, must use the same
immutable v2 manifest and v2 runtime directory with `--launch --resume`; the
runner must reconcile the manifest-bound marker and persisted job identity
before submitting anything. The excluded v1 runtime must never be passed to
the v2 runner.

The v1 failure disposition can be checked without importing its metric strings:

```bash
sha256sum \
  experiments/dino_moo_phase2_20260728/expanded_search_manifest.v1.json \
  experiments/dino_moo_phase2_20260728/expanded_search_runtime_erratum.v1.json \
  experiments/dino_moo_phase2_20260728/expanded_search_v1_failure_audit.md

for seed in 161803 271828 314159; do
  jq '[.records[] | select(.rec_id == 0) |
      {candidate_id,status,automl_result_status,metric,failure_reason}]' \
    "experiments/dino_moo_phase2_20260728/runtime/expanded_search/seed_${seed}/candidate_evaluations.json"
done
```

### 10.8 Post-front implementation verification

The committed pre-manifest snapshot is reproduced without launching or reading
future measurements:

```bash
cd ~/tao-automl

/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  -m pytest -q \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_tools.py \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_launcher_recovery.py \
  experiments/dino_moo_phase2_20260728/test_post_front_complete_invalid_recovery.py

sha256sum \
  experiments/dino_moo_phase2_20260728/post_front_matched_manifest_generator.py \
  experiments/dino_moo_phase2_20260728/post_front_matched_launcher.py \
  experiments/dino_moo_phase2_20260728/post_front_matched_block_runner.py \
  experiments/dino_moo_phase2_20260728/post_front_matched_aggregator.py \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_tools.py \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_launcher_recovery.py \
  experiments/dino_moo_phase2_20260728/test_post_front_complete_invalid_recovery.py
```

The expected focused result at the implementation snapshot is `81 passed`.
No post-front manifest, ledger, dry-run, invalidation evidence, analysis, TAO
job, or SLURM job exists yet. Exact completed expanded-search,
manifest-generation, final-front launch/aggregation, combined-selection, and
integrity-audit result commands remain **PENDING LIVE EVIDENCE**. They must be
copied from the tracked, committed final harness and immutable artifacts after
the 60-record archive exists; no placeholder hash or job identity is supplied
here.
