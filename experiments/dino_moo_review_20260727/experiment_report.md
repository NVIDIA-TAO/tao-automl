# DINO multi-objective AutoML technical review and validation

> Historical 30-candidate/98%-floor report. Its conclusions and provenance are
> preserved as executed. The completed phase-2 explicit-90% authority is
> [`../dino_moo_phase2_20260728/latency_90_policy/latency_90_policy_report.md`](../dino_moo_phase2_20260728/latency_90_policy/latency_90_policy_report.md).

Date: 2026-07-28 UTC

Scope: DINO ResNet50 only, using
`s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/`.
No other model family, PTM compatibility repair, or dataset is included.

The implementation defect is fixed and the core safety invariant now holds: the
multi-objective selector can return only an eligible, Pareto-rank-zero
candidate. The global 30-candidate archive contains six intermediate
accuracy/latency Pareto points, but the originally configured 98% accuracy
floor excluded five of them from multi-objective scoring. All three modes
therefore selected the same candidate through the AutoML selector, and the
overall hypothesis is **inconclusive because no distinct Pareto compromise
exists under the configured 98% accuracy-feasibility constraint**.

Terminal evidence comprises 30/30 successful candidates, three zero-exit seed
processes, a reverified 374-pass test suite, and three independent eight-GPU
validation-only repeats of the frozen algorithm-selected winner.

## 1. Root cause

### Why the old result was dominated

The direct cause was a population mismatch, not a bad comparison inside one
shared archive.

The old experiment launched three independent ten-candidate Bayesian searches,
one for each mode (`experiments/dino_multi_objective_20260724_021917/run_full_experiments.py:259-306`
and `:609-617`). Consequently, the old latency winner was not present in the
multi-objective archive. The multi-objective selector could not compare against
it. In the old evidence, the latency winner is row 17 of
`experiments/dino_multi_objective_20260724_021917/all_selection_trials.csv`,
whereas the multi-objective winner is row 28. Across those two separate
archives:

- latency winner: mAP50 `0.593234252953`, one-pass proxy latency
  `4.251700680272 ms`, repeated proxy median `4.222972972973 ms`;
- multi-objective winner: mAP50 `0.574220770650`, one-pass proxy latency
  `4.274965800274 ms`, repeated proxy median `4.257493188011 ms`.

The latency winner is therefore more accurate and faster. The cross-archive
dominance finding was real, but the old multi-objective run had no opportunity
to observe it.

The old implementation also made that population error easy to miss:

1. Multi-objective values were reduced to the raw weighted sum

   \[
   S(x)=\sum_j w_j\,d_j\,\frac{f_j(x)}{s_j},
   \]

   where \(d_j=+1\) for maximize and \(-1\) for minimize. The old code is
   `758efb8:src/tao_automl/objectives.py:181-199`. For this experiment, the
   driver configured

   \[
   S(x)=\operatorname{mAP50}(x)
        - \frac{\operatorname{latency}_{ms}(x)}{3.531073446328}
   \]

   (`run_full_experiments.py:54-55` and `:286-305`). The arbitrary baseline
   scale determined the trade-off; no Pareto filtering, front normalization,
   or compromise rationale was part of final selection.

   Applying that old formula across archives gives approximately
   `-0.61084738` for the latency winner and `-0.63644954` for the old
   multi-objective winner; higher was better. Thus the latency winner would
   also have won the old scalar rule had it been present in the same archive.
   This confirms that separate candidate populations—not a scalar comparison
   that preferred the dominated point—were the direct cause of the observed
   result.

2. Final selection was plain `max(rec.result)` for maximize-oriented scores
   (`758efb8:src/tao_automl/controller/controller.py:230-248`). There was no
   final Pareto-safety gate, no shared-archive assertion, and no audit explaining
   which other candidates dominated a returned point. With positive weights,
   this particular weighted sum normally preserves dominance *within its own
   archive*; it did nothing about the superior point hidden in another mode's
   archive.

3. The latency constraint used the raw PTM baseline mAP50
   `0.007808934173321529` (`run_full_experiments.py:54`, `:353-365`,
   `:420-426`). All 30 old candidates passed. Latency mode was therefore
   effectively unconstrained and did not mean “retain accuracy relative to the
   accuracy winner.”

4. Selection latency was inferred from the last Lightning test progress rate:

   \[
   L=1000/(\text{last it/s}\times 32)
   \]

   (`run_full_experiments.py:226-230`). It was one pass, mixed evaluation-loop
   behavior with throughput, assumed the hard-coded global batch, did not
   explicitly synchronize CUDA, and supplied no dispersion or confidence
   estimate. The five-pass check happened only after selection.

5. The maximize-only Bayesian surrogate previously received minimize values
   without a single, explicit orientation boundary; failed/non-finite
   observations could contaminate X/Y state; and fallback seeding relied on
   Python's process-randomized `hash`. These did not by themselves cause the
   reported cross-archive domination, but they weakened candidate generation
   and reproducibility.

The old 4–5 ms numbers and the new approximately 79–80 ms numbers must not be
compared as a speed regression: the former was an evaluation-throughput proxy,
whereas the latter is synchronized batch-one request latency for model forward
plus DINO GPU postprocessing.

## 2. Algorithm changes

The principal implementation is
`src/tao_automl/selection.py`. It deliberately separates acquisition utility
from final deployment selection (`src/tao_automl/objectives.py:207-253` and
`src/tao_automl/controller/controller.py:232-265`).

### Candidate generation and archive ownership

- **Old:** each mode generated and selected from a different ten-point archive.
- **New:** three seeded, sequential Bayesian sub-archives (seeds `314159`,
  `271828`, and `161803`; ten recommendations each) are united after every
  successful candidate has the same mAP50 evaluation and latency benchmark.
  Accuracy, latency, and multi-objective selection all consume this identical
  30-candidate union. The final union is constructed only from successful
  records at `experiments/dino_moo_review_20260727/run_experiment.py:966-1043`.
- Training budget, search ranges, model, dataset split, runtime, and latency
  protocol are common to the whole archive. Search acquisition and final
  deployment choice are distinct operations; no mode-specific post-run
  candidate promotion is possible.
- Bayesian/BFBO observations are now explicitly oriented for maximize-only
  acquisitions, non-success or non-finite observations are discarded, prior
  utilities are rebuilt after archive normalization changes, resume state is
  filtered, and the random seed is explicit or SHA-256-derived
  (`src/tao_automl/brain/base.py:29-54` and `:110-217`).
- For two-objective selection, `eval_fn` is required and fails closed instead of
  silently reverting to a training-log proxy (`src/tao_automl/runner.py:3895-3934`).

### Validation and objectives

Candidates with a non-success status, missing metric, boolean value, NaN,
infinity, non-positive median latency, malformed specification, incomplete or
non-positive-low latency confidence interval, or a confidence interval that
excludes its median are invalidated before ranking
(`src/tao_automl/selection.py:344-445`). Failed or physically invalid
candidates cannot enter a Pareto front.

The objective directions are explicit:

- maximize \(A(x)=\operatorname{mAP50}(x)\);
- minimize \(L(x)=\) stabilized median latency in milliseconds.

The provisional two-objective scalar is never a sum of raw mAP and
milliseconds. Archive-normalized acquisition utility is refreshed as the
archive grows (`src/tao_automl/controller/controller.py:816-844`), while the
final result is always recomputed from the complete archive
(`src/tao_automl/controller/controller.py:248-251`, `:399-450`).

### Accuracy mode

Accuracy mode solves

\[
\arg\max_{x\in V} A(x),
\]

where \(V\) is the set of valid candidates. Only candidates within
`accuracy_tolerance = 1e-12` of the maximum may use latency as a tie-break.
Remaining exact latency ties use canonical SHA-256 parameter fingerprint and
candidate ID. See `src/tao_automl/selection.py:538-556`.

### Latency mode and accuracy retention

The old untrained-baseline floor is replaced by a configurable
accuracy-winner-relative epsilon constraint. The default used here is:

\[
A(x)\ge 0.98 A^*,\qquad A^*=\max_{x\in V} A(x).
\]

An alternative general absolute rule,
\(A(x)\ge A^*-\Delta_A\), is also supported
(`src/tao_automl/selection.py:59-113`). The relative 98% rule is a declared
retention policy, not a threshold fitted to these DINO observations.

Latency mode then solves

\[
\arg\min_{x\in V:\,A(x)\ge 0.98A^*} L(x).
\]

If an externally configured reference makes the feasible set empty, selection
returns `no_accuracy_feasible_candidates`; it does not silently choose an
unrelated point (`src/tao_automl/selection.py:803-852`).

Median latencies are tied when they fall within configured
`latency_tolerance` or their 95% cluster-bootstrap intervals overlap. A latency
tie is resolved by higher accuracy, then canonical fingerprint, then candidate
ID (`src/tao_automl/selection.py:573-597`). This run configured an explicit
point tolerance of `0 ms`; the confidence-interval rule still guards against
claiming significance for overlapping measurements.

### Pareto dominance and front construction

For candidates \(x,y\), \(x\) dominates \(y\) only if:

\[
A(x)\ge A(y),\quad L(x)\le L(y),
\]

and at least one improvement is strict. Accuracy strictness uses
`accuracy_tolerance`; latency-only strictness requires non-overlapping
confidence intervals in the favorable direction (or the configured point
tolerance when intervals are unavailable). Statistical equivalence may
withhold a latency-only dominance claim, but a numerically slower point can
never be considered “no worse” (`src/tao_automl/selection.py:449-483`).

Deterministic non-dominated sorting produces zero-based Pareto ranks and exact
`dominated_by` IDs (`src/tao_automl/selection.py:486-535`). The compromise
selector sees only accuracy-feasible rank-zero candidates. Exact duplicate
objective points use the smallest canonical specification fingerprint as their
representative while retaining all aliases (`:600-622`).

### Normalization and compromise selection

For the accuracy-feasible, deduplicated Pareto front \(P\), objective regrets
are:

\[
r_A(x)=\frac{A_{\max}-A(x)}{A_{\max}-A_{\min}},\qquad
r_L(x)=\frac{L(x)-L_{\min}}{L_{\max}-L_{\min}}.
\]

Bounds come only from \(P\), are persisted in the selection audit, and orient
opposite objective directions correctly. Finite dominated outliers are removed
before bounds are computed and therefore cannot distort the compromise score.
A valid nondominated endpoint is a real measured trade-off, so it defines an
ideal or nadir bound without hidden clipping. An axis whose range is within its
tolerance is inactive and contributes zero, preventing division by zero
(`src/tao_automl/selection.py:625-690`).

Raw weights `1.0, 1.0` normalize to \(w_A=w_L=0.5\). The final compromise
minimizes the augmented Chebyshev achievement score

\[
C(x)=\max(w_A r_A(x),w_L r_L(x))
     +10^{-6}\left(w_A r_A(x)+w_L r_L(x)\right).
\]

The worst normalized regret therefore controls the decision, while the small
augmentation makes otherwise equivalent orderings strict without mixing raw
units. Score ties within `1e-12` use, in order: weighted Euclidean distance to
the ideal, absolute balance gap, lower normalized accuracy regret, canonical
fingerprint, and candidate ID (`src/tao_automl/selection.py:700-733`).

A candidate is excluded from the compromise calculation before scoring if it
is invalid, accuracy-infeasible, a duplicate alias, or dominated on the
feasible front. Thus a dominated multi-objective winner is structurally
impossible.

If no eligible nondominated candidate exists apart from the accuracy and
latency extremes, the audit reports:

> No distinct Pareto compromise exists under the configured multi-objective
> eligibility policy.

The same deterministic Chebyshev ordering supplies an extreme-point fallback,
and `distinct_compromise=false` plus `fallback_used=true` prevents that fallback
from being mislabeled as a middle ground
(`src/tao_automl/selection.py:854-922`).

### Why this method was selected

The method comparison is documented at
`docs/Multi_Objective_AutoML_Literature_Review.md:122-149`.

| Method | Decision for this 30-point, two-objective archive |
| --- | --- |
| Pareto nondominance | Required safety filter; it removes objectively inferior points but does not alone pick one deployment. |
| Epsilon-constrained optimization | Selected for latency mode because “fastest while retaining accuracy” is naturally a constraint. |
| Normalized ideal/utopia distance | Useful and scale-independent, but fully compensatory; retained as a tie-break rather than the primary rule. |
| Knee point | Not used as default because a sparse, nearly linear, or two-point front has no stable geometric knee. |
| Augmented Chebyshev | Selected: deterministic, scale-independent, minimizes worst regret, and works on non-convex fronts. |
| Hypervolume contribution | Rejected for final-point selection because it needs a reference point and can favor endpoints; better suited to archive acquisition/diversity. |
| NSGA-II rank/crowding | Non-dominated sorting is adopted, but an evolutionary population is excessive for three variables and 30 expensive evaluations. |
| qEHVI/qNEHVI/EHVI | Deferred for a larger, noisy, parallel acquisition upgrade; these methods still do not define the final deployment point. |

### Stabilized latency procedure

The benchmark implementation is
`experiments/dino_moo_review_20260727/dino_latency_benchmark.py:121-347`;
aggregation is `src/tao_automl/latency_stats.py:207-462`.

- 8 independent DINO replicas, one per A100 GPU;
- batch size 1 per GPU, FP32, TF32 disabled;
- cuDNN benchmark disabled and deterministic mode enabled;
- the same 16 already decoded, resized, normalized, and device-resident
  validation batches are cycled for every candidate; the persisted workload is
  model input `[1,4,800,1333]`, comprising RGB `[1,3,800,1333]` and padding
  mask `[1,1,800,1333]`;
- 50 untimed warm-ups per replica;
- 5 repeated rounds × 100 timed requests × 8 replicas = 4,000 raw request
  samples per candidate, 500 samples per device, and 40 equal-weight
  device-round median clusters; the unambiguous serialized fields are
  `raw_sample_count_total=4000` and `samples_per_device=500`;
- NCCL barrier and `torch.cuda.synchronize` before each request, CUDA
  synchronization before stopping the timer;
- timed scope: model forward plus DINO GPU postprocessing;
- excluded: checkpoint load, disk I/O, decode/resize/normalization, H2D transfer,
  COCO accumulation, and distributed gather;
- primary statistic: median of 40 device-round medians;
- tail: pooled p95; dispersion: MAD, IQR, robust
  \(CV=1.4826\,MAD/\text{median}\);
- secondary diagnostic: slowest-device synchronized median and p95;
- uncertainty: deterministic 5,000-resample, 95% device-round cluster bootstrap;
- validity gates: robust CV ≤10%, round median range ≤5%, absolute round drift
  ≤5%, device median range ≤5%, and bootstrap-CI width ≤3%.

All 30 candidate measurements passed every quality gate. Across the archive,
the maximum robust CV was `0.00355448` (0.355%), maximum CI width was
`0.32658 ms`, maximum absolute round drift fraction was `0.00276904`, and
maximum device-median range fraction was `0.0116763`, all well inside policy.

Commit `5839c945e1d08c9638f2c8a0ddac377b6058f66a` then hardened the
benchmark evidence without changing selection. It records all 16 image names,
tensor shapes, per-batch hashes, and a canonical complete-workload digest:

```text
1b43c34913bff097054d6a76cdd7dd0a02546dd07db8adce50d40a8986774d08
```

The already algorithm-selected winner was benchmarked in three new,
validation-only eight-GPU allocations. The repeat driver requires a single
winner across all three frozen mode selections, reads that winner and
checkpoint from `combined_selection.json`, sets `feeds_selection=false`, and
never invokes or rewrites archive selection
(`repeat_selected_winner_latency.py:55-108`, `:128-201`, `:204-289`).

| Repeat | TAO job | SLURM job | Node | Median ms | p95 ms | 95% within-job CI ms |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 0 | `c5e5ca6d-fe96-4e99-8f70-9210b60543e5` | 30931806 | `batch-block7-00843` | 79.1567375 | 79.5096065 | [79.0355938, 79.2041465] |
| 1 | `10450946-9dc2-4dcb-9a6c-c0addd7261bf` | 30932032 | `batch-block7-00843` | 79.4215653 | 79.7488390 | [79.3965773, 79.4842370] |
| 2 | `ccdcfc05-2893-48bb-9640-610c859098ea` | 30932183 | `batch-block7-00843` | 79.6241778 | 80.1542135 | [79.5880240, 79.6813815] |

All repeats used the same A100/SQSH contract and the exact same input digest.
Their median-of-job-medians is `79.42156525 ms`; median range is
`0.46744025 ms` with cross-job CV `0.00241048`. Their p95 range is
`0.644607 ms` with CV `0.00333388`.

The original selection measurement was `79.89227525 ms` on
`batch-block7-03393`; the three repeats ran on `batch-block7-00843`. Across all
four protocol- and input-identical allocations, median range is
`0.73553775 ms` and CV is `0.00339193`. The within-job cluster-bootstrap
intervals do not include this allocation shift. They quantify device/round
sampling within one allocation and must not be used to claim that candidate
latency differences below `0.73553775 ms` are portable across allocations.
This does not affect the final mode selections because the accuracy constraint
left exactly one feasible candidate.

## 3. Test coverage

The complete suite was reverified after the final hardening commit:

```text
374 passed, 1 skipped, 1 warning in 4.89s
```

The warning was a benign scikit-learn convergence warning. The command was
`python -m pytest -q`; `git diff --check` was also clean. The targeted selector
suite completed with `33 passed`.

Targeted Pareto correctness tests in
`tests/test_multi_objective_selection.py` cover:

1. dominated candidate exclusion (`:64-74`);
2. accuracy extreme, latency extreme, and valid middle point (`:77-98`);
3. rejection of a fake middle dominated by the latency winner (`:101-119`);
4. deterministic exact-duplicate representation (`:122-143`);
5. candidate-order permutation invariance (`:146-166`);
6. positive affine/raw-scale invariance (`:169-191`);
7. finite dominated-outlier isolation from front normalization (`:194-223`);
8. identical-axis/zero-range safety (`:226-238`);
9. missing, boolean, NaN, infinite, zero/negative latency, and failed
   measurements (`:241-269`);
10. documented no-intermediate fallback (`:282-290`);
11. the compromise winner is rank zero and has no dominator (`:327-340`);
12. accuracy-maximize/latency-minimize directionality (`:343-355`);
13. deterministic, confidence-interval-aware tie-breaking and dominance
    (`:358-474`).

Additional selection tests cover incomplete, malformed, and non-positive-low
latency intervals, relative and absolute accuracy constraints, an explicit
no-feasible status, and inclusive constraint boundaries (`:477-564`). A
deterministic Hypothesis test runs 100 generated archives and jointly asserts
nondominance, permutation invariance, positive-scale invariance, and finite
compromise scores (`:568-606`).

`tests/test_latency_stats.py:43-414` covers protocol validation, equal weighting
of device-round clusters, deterministic/bootstrap-repeatable aggregation,
single-axis zero spread, mapping-order independence, quality-gate boundary and
failure reasons, incomplete/non-finite sample rejection, expected device
identity, point/relative latency tolerance, overlapping confidence intervals,
invalid-statistic comparison, and the unambiguous
`raw_sample_count_total`/`samples_per_device` serialization contract.

`tests/test_bayesian_objective_direction.py:80-464` covers minimize orientation,
failure/non-finite exclusion, incomplete two-objective exclusion, archive
utility rebuilding, pending-point removal, explicit and SHA-derived seeds,
process independence, deterministic initial recommendations, legacy resume
migration, and stale-utility rebuilding. Controller integration at
`tests/test_wheel.py:205-273` verifies final archive selection, Pareto exposure,
and audit reporting.

The SLURM integration validation also completed:

- smoke accuracy: TAO job `66f96a27-d4b1-4273-9049-b459cbd0badc`,
  SLURM `30906807`;
- smoke latency: TAO job `462b4a60-b6f1-4a30-8880-276ad7c9311e`,
  SLURM `30907153`;
- 30/30 full candidates successful, zero invalid measurements, and seed process
  exit codes `0,0,0`;
- three independent validation-only winner repeats completed on SLURM
  `30931806`, `30932032`, and `30932183`, with identical input digest and no
  path back into selection.

## 4. Full candidate table

Candidate IDs below abbreviate `seed_<seed>_rec_<rec>` as `<seed>/<rec>`.
`P` is the zero-based global Pareto rank over all 30 valid candidates; the
`dominated by` column lists exact abbreviated IDs. Variability is
`MAD / IQR / robust-CV`. `A/L/M` marks the three mode winners.

The feasible threshold is `0.6232699382647006`; only `271828/5` is feasible and
its feasible Pareto rank is zero. Normalized values are shown as
`accuracy regret / latency regret / compromise score`. They are `N/A` for
infeasible candidates because those points never enter feasible-front
normalization or compromise selection. The raw JSON/CSV currently serializes
mechanical zeros for those candidates when the one-point feasible front makes
both axes inactive; those zeros have no selection meaning and are intentionally
not presented as scores here. Exact confidence intervals, TAO/SLURM job IDs,
checkpoint paths, fingerprints, and tie values remain in
`experiments/dino_moo_review_20260727/combined_selection.json` and
`full_candidate_table.csv`.

| ID | Queries | LR | Weight decay | mAP50 | Median ms | p95 ms | MAD / IQR / rCV | Feasible | P | Dominated by | Norm A/L/score | Winner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- | --- | :---: |
| 161803/0 | 636 | 8.8187595e-05 | 0.00041383428 | 0.527911 | 79.3308 | 79.7036 | 0.0871 / 0.2063 / 0.163% | N | 0 | — | N/A | — |
| 161803/1 | 672 | 0.00037299168 | 0.0006073463 | 0.575453 | 79.6028 | 80.4018 | 0.1084 / 0.2837 / 0.202% | N | 1 | 161803/2, 271828/1, 271828/2 | N/A | — |
| 161803/2 | 311 | 0.00040600614 | 0.00075211685 | 0.578061 | 79.5114 | 79.7560 | 0.1500 / 0.3075 / 0.280% | N | 0 | — | N/A | — |
| 161803/3 | 427 | 0.00045 | 0.0009 | 0.568511 | 80.1014 | 80.4308 | 0.1008 / 0.2093 / 0.187% | N | 4 | 161803/1, 161803/2, 161803/4, 161803/5, 161803/6, 161803/8, 271828/0, 271828/1, 271828/2, 271828/5, 271828/7, 271828/9 | N/A | — |
| 161803/4 | 469 | 0.00045 | 1.1e-05 | 0.613110 | 79.9475 | 80.3042 | 0.0987 / 0.1990 / 0.183% | N | 1 | 271828/5 | N/A | — |
| 161803/5 | 390 | 0.00042067257 | 0.00029484691 | 0.601234 | 80.0611 | 80.4220 | 0.0932 / 0.2385 / 0.173% | N | 3 | 161803/4, 161803/8, 271828/1, 271828/5, 271828/9 | N/A | — |
| 161803/6 | 300 | 0.00035563125 | 1.1e-05 | 0.574713 | 79.5433 | 79.8115 | 0.0928 / 0.1878 / 0.173% | N | 1 | 161803/2, 271828/2 | N/A | — |
| 161803/7 | 510 | 0.00012479148 | 0.00018314874 | 0.550185 | 79.6100 | 79.8676 | 0.1140 / 0.2322 / 0.212% | N | 2 | 161803/1, 161803/2, 161803/6, 271828/1, 271828/2, 271828/4, 314159/7, 314159/8 | N/A | — |
| 161803/8 | 900 | 0.00048979367 | 0.00021758636 | 0.604800 | 79.7275 | 79.9467 | 0.0991 / 0.1990 / 0.184% | N | 2 | 271828/1, 271828/9 | N/A | — |
| 161803/9 | 365 | 1.1e-05 | 0.00019510358 | 0.417475 | 79.6353 | 80.1514 | 0.1345 / 0.2725 / 0.250% | N | 4 | 161803/0, 161803/1, 161803/2, 161803/6, 161803/7, 271828/1, 271828/2, 271828/3, 271828/4, 271828/6, 271828/9, 314159/7, 314159/8 | N/A | — |
| 271828/0 | 880 | 0.0004209469 | 8.268974e-05 | 0.594847 | 79.7459 | 80.0789 | 0.1414 / 0.2838 / 0.263% | N | 3 | 161803/8, 271828/1, 271828/7, 271828/9 | N/A | — |
| 271828/1 | 662 | 0.00022176445 | 0.00096832268 | 0.606458 | 79.5460 | 80.0288 | 0.1387 / 0.3711 / 0.258% | N | 0 | — | N/A | — |
| 271828/2 | 878 | 0.0003549402 | 0.00038927878 | 0.576942 | 79.4060 | 79.7141 | 0.1360 / 0.2719 / 0.254% | N | 0 | — | N/A | — |
| 271828/3 | 543 | 6.6296672e-05 | 0.00040694368 | 0.527213 | 79.5900 | 79.8336 | 0.1182 / 0.2356 / 0.220% | N | 3 | 161803/0, 161803/2, 161803/6, 271828/1, 271828/2, 271828/4, 271828/6, 314159/7, 314159/8 | N/A | — |
| 271828/4 | 528 | 0.0001550183 | 0.00016477914 | 0.553107 | 79.5114 | 79.7540 | 0.1082 / 0.2157 / 0.202% | N | 1 | 271828/2, 314159/7 | N/A | — |
| 271828/5 | 594 | 0.00045 | 0.00026967724 | 0.635990 | 79.8923 | 80.2170 | 0.1311 / 0.2600 / 0.243% | Y | 0 | — | 0 / 0 / 0 | A/L/M |
| 271828/6 | 300 | 5.6030845e-05 | 7.3857082e-05 | 0.534644 | 79.5586 | 79.7951 | 0.0913 / 0.1808 / 0.170% | N | 2 | 161803/2, 161803/6, 271828/1, 271828/2, 271828/4, 314159/7, 314159/8 | N/A | — |
| 271828/7 | 516 | 0.00026384214 | 0.00054245721 | 0.601110 | 79.6654 | 79.9784 | 0.1326 / 0.2661 / 0.247% | N | 2 | 271828/1, 271828/9 | N/A | — |
| 271828/8 | 629 | 0.00013689022 | 0.00089603314 | 0.550797 | 80.1411 | 80.5896 | 0.1921 / 0.3834 / 0.355% | N | 5 | 161803/1, 161803/2, 161803/3, 161803/4, 161803/5, 161803/6, 161803/8, 271828/0, 271828/1, 271828/2, 271828/4, 271828/5, 271828/7, 271828/9, 314159/0, 314159/7, 314159/8 | N/A | — |
| 271828/9 | 687 | 0.00031334147 | 0.00071167775 | 0.605146 | 79.6290 | 79.8235 | 0.0783 / 0.1570 / 0.146% | N | 1 | 271828/1 | N/A | — |
| 314159/0 | 791 | 0.00028001269 | 0.0004255776 | 0.553572 | 79.8155 | 80.1168 | 0.1207 / 0.2417 / 0.224% | N | 4 | 161803/1, 161803/2, 161803/6, 161803/8, 271828/0, 271828/1, 271828/2, 271828/7, 271828/9, 314159/7, 314159/8 | N/A | — |
| 314159/1 | 359 | 0.00040740017 | 0.00096768284 | 0.516658 | 79.6716 | 80.0332 | 0.1613 / 0.3243 / 0.300% | N | 4 | 161803/0, 161803/1, 161803/2, 161803/6, 161803/7, 271828/1, 271828/2, 271828/3, 271828/4, 271828/6, 271828/7, 271828/9, 314159/7, 314159/8 | N/A | — |
| 314159/2 | 825 | 0.00032477999 | 0.00085429691 | 0.550627 | 80.0261 | 80.2857 | 0.1035 / 0.2051 / 0.192% | N | 5 | 161803/1, 161803/2, 161803/4, 161803/6, 161803/8, 271828/0, 271828/1, 271828/2, 271828/4, 271828/5, 271828/7, 271828/9, 314159/0, 314159/7, 314159/8 | N/A | — |
| 314159/3 | 900 | 5.7034691e-05 | 0.00069411064 | 0.550203 | 79.7711 | 80.0056 | 0.1002 / 0.1984 / 0.186% | N | 4 | 161803/1, 161803/2, 161803/6, 161803/8, 271828/0, 271828/1, 271828/2, 271828/4, 271828/7, 271828/9, 314159/7, 314159/8 | N/A | — |
| 314159/4 | 603 | 1.1e-05 | 1.1e-05 | 0.399534 | 79.7834 | 80.0398 | 0.1263 / 0.2503 / 0.235% | N | 5 | 161803/0, 161803/1, 161803/2, 161803/6, 161803/7, 161803/8, 161803/9, 271828/0, 271828/1, 271828/2, 271828/3, 271828/4, 271828/6, 271828/7, 271828/9, 314159/1, 314159/3, 314159/7, 314159/8 | N/A | — |
| 314159/5 | 530 | 0.00030161926 | 0.00056729438 | 0.549730 | 79.8438 | 80.1268 | 0.1806 / 0.4104 / 0.335% | N | 5 | 161803/1, 161803/2, 161803/6, 161803/7, 161803/8, 271828/0, 271828/1, 271828/2, 271828/4, 271828/7, 271828/9, 314159/0, 314159/3, 314159/7, 314159/8 | N/A | — |
| 314159/6 | 844 | 0.00013768524 | 0.0009 | 0.531360 | 80.0884 | 80.7232 | 0.1488 / 0.3073 / 0.276% | N | 6 | 161803/1, 161803/2, 161803/4, 161803/5, 161803/6, 161803/7, 161803/8, 271828/0, 271828/1, 271828/2, 271828/4, 271828/5, 271828/6, 271828/7, 271828/9, 314159/0, 314159/2, 314159/3, 314159/5, 314159/7, 314159/8 | N/A | — |
| 314159/7 | 523 | 0.00019549044 | 0.00067884875 | 0.559733 | 79.3345 | 79.7184 | 0.1143 / 0.3149 / 0.214% | N | 0 | — | N/A | — |
| 314159/8 | 300 | 0.00034136987 | 0.0007523813 | 0.561366 | 79.5194 | 79.8719 | 0.1061 / 0.2532 / 0.198% | N | 1 | 161803/2, 271828/2 | N/A | — |
| 314159/9 | 900 | 5.4252688e-05 | 0.00027645081 | 0.547490 | 80.4125 | 80.6941 | 0.1277 / 0.2567 / 0.235% | N | 6 | 161803/1, 161803/2, 161803/3, 161803/4, 161803/5, 161803/6, 161803/7, 161803/8, 271828/0, 271828/1, 271828/2, 271828/4, 271828/5, 271828/7, 271828/8, 271828/9, 314159/0, 314159/2, 314159/3, 314159/5, 314159/7, 314159/8 | N/A | — |

The unconstrained global rank-zero front contains six points:
`161803/0`, `314159/7`, `271828/2`, `161803/2`, `271828/1`, and `271828/5`.
There are real accuracy/latency trade-offs in the measured archive, but the
first five fail the 98%-retention constraint. The feasible Pareto front is
therefore the one-point set `{271828/5}`.

Those global ranks and dominator relationships are correct for the recorded
medians and within-job confidence intervals, but they are not evidence of
deployment-significant latency separation. Every alternative global-front
latency differs from the winner by at most `0.56144325 ms`, below the observed
four-allocation winner range of `0.73553775 ms`. This noise qualification does
not affect any final mode winner: all five alternative front points fail the
accuracy-retention floor before latency or compromise scoring.

## 5. Final mode comparison

The algorithm-selected candidate is:

```text
candidate_id: seed_271828_rec_5
model.num_queries: 594
train.optim.lr: 0.00045
train.optim.weight_decay: 0.00026967723799334445
canonical fingerprint:
  e608fa8d2b79795abfe909e7126c2762a66871387756ab13cc35a6ba7be48f05
```

| Mode | Selected candidate | mAP50 | Median latency | Accuracy delta vs accuracy mode | Latency delta vs accuracy mode | Pareto status | Selection reason |
| ---- | ------------------ | ----: | -------------: | ------------------------------: | -----------------------------: | ------------- | ---------------- |
| Accuracy | seed_271828_rec_5 | 0.635990 | 79.8923 ms | 0.000000 | 0.0000 ms | Global rank 0; feasible rank 0 | Highest valid mAP50 in the shared 30-candidate archive. |
| Latency | seed_271828_rec_5 | 0.635990 | 79.8923 ms | 0.000000 | 0.0000 ms | Global rank 0; feasible rank 0 | Lowest stabilized median among candidates meeting mAP50 ≥ 0.6232699383; it was the only feasible point. |
| Multi-objective | seed_271828_rec_5 | 0.635990 | 79.8923 ms | 0.000000 | 0.0000 ms | Global rank 0; feasible rank 0; nondominated | No distinct feasible Pareto compromise exists; deterministic augmented-Chebyshev extreme fallback. |

For this candidate:

- mAP50 reference: `0.6359897329231639`;
- 98% threshold: `0.6232699382647006`;
- median: `79.89227525 ms`;
- p95: `80.21695199999999 ms`;
- MAD: `0.13106925 ms`;
- IQR: `0.26000925 ms`;
- robust CV: `0.002432316` (0.243%);
- 95% cluster-bootstrap interval:
  `[79.8205115, 79.98804975] ms`;
- device-median range: `0.326456 ms`;
- round drift: `0.1310485 ms` (0.164%);
- both normalization axes inactive because the feasible front has one point;
- normalized regrets `(0,0)`, augmented-Chebyshev score `0`, ideal distance
  `0`, balance gap `0`;
- accuracy, latency, and multi-objective winner flags are all true.

The table intentionally reports `79.89227525 ms`, the stabilized measurement
available to the algorithm when the archive was frozen. The three later
validation-only measurements were `79.1567375`, `79.42156525`, and
`79.62417775 ms` (median `79.42156525 ms`). They confirm the same approximately
79–80 ms request-latency regime but do not replace, rescore, or override the
selection-time objective.

The multi-objective result is no longer invalid: it is nondominated. It is also
not a distinct compromise, and the result audit says so rather than forcing a
synthetic middle point.

## 6. Hypothesis verdict

| Question | Verdict | Evidence |
| --- | --- | --- |
| Did accuracy mode select the highest-accuracy valid candidate? | **Yes.** | `0.6359897329` is the maximum mAP50 among all 30 valid candidates. |
| Did latency mode select the fastest candidate satisfying the new retention constraint? | **Yes.** | The 98% floor was `0.6232699383`; only `seed_271828_rec_5` passed, so it is necessarily the constrained minimum. The unconstrained fastest point (`161803/0`, `79.330832 ms`) was correctly rejected at mAP50 `0.527911`. |
| Is the multi-objective winner Pareto-nondominated? | **Yes.** | It has global and feasible Pareto rank zero and an empty `dominated_by` set. |
| Is it a distinct, defensible middle ground? | **No.** | The feasible Pareto front has one point. The selector emitted the documented no-distinct-compromise fallback. |
| Is its accuracy between the two extremes? | **Equal to both, not strictly between.** | All modes selected the same point. |
| Is its latency between the two extremes? | **Equal to both, not strictly between.** | All modes selected the same point. |
| Was selection entirely algorithmic? | **Yes.** | The three winners, feasibility, ranks, scores, ties, fingerprint, and flags were emitted by `analyze_archive` over the persisted 30-point union. No candidate ID or DINO-specific result rule exists in the selector. |
| Are results stable across repeats or seeds? | **Partly.** | Selector invariants hold in every seed replay. Three independent validation allocations put the winner's medians in a 0.46744025 ms range (CV 0.00241048), while all four allocations including the selection job span 0.73553775 ms (CV 0.00339193). Search quality still varies materially by seed, and sub-0.73553775 ms candidate differences are not established as allocation-stable. |

The persisted read-only replay in `per_seed_selection.json`
(`feeds_union_selection=false`) applies the same selector to each ten-point seed
sub-archive and produces:

| Search seed | Accuracy winner (mAP50 / ms) | Latency winner (mAP50 / ms) | Multi winner | Threshold | Outcome |
| ---: | --- | --- | --- | ---: | --- |
| 161803 | rec 4: `0.613110 / 79.9475` | rec 8: `0.604800 / 79.7275` | rec 4 | `0.600848172` | Three feasible points; no distinct selected compromise, deterministic accuracy-extreme fallback. |
| 271828 | rec 5: `0.635990 / 79.8923` | rec 5: `0.635990 / 79.8923` | rec 5 | `0.623269938` | One feasible point; no distinct compromise. |
| 314159 | rec 8: `0.561366 / 79.5194` | rec 7: `0.559733 / 79.3345` | rec 8 | `0.550138907` | Five feasible points; no distinct selected compromise, deterministic accuracy-extreme fallback. |

This shows stable *algorithmic behavior* across seeds: accuracy is maximized,
latency is minimized subject to the declared floor, and the multi-objective
winner is never dominated or mislabeled as a middle point. It does not show
stable search quality: per-seed best mAP50 ranges from `0.561366` to `0.635990`.
The 30-point union reduces seed sensitivity for this run but does not establish
convergence.

Latency is repeatable at the approximately 79–80 ms level: the three independent
repeat medians have only 0.241% CV, and all four allocations have 0.339% CV.
However, the independent jobs also expose an allocation offset larger than the
within-job bootstrap intervals. Those intervals do not model allocation
variance, so this experiment does not support fine-grained claims among
candidate medians separated by less than `0.73553775 ms`. The selected result
is unchanged because no second candidate satisfies the accuracy floor.

**Overall classification: inconclusive because no distinct Pareto compromise
exists under the configured 98% accuracy-feasibility constraint.** The global
archive does contain six nondominated points. The accuracy and
constrained-latency endpoint rules are supported, and Pareto safety is
supported. The proposed distinct multi-objective middle-ground behavior cannot
be confirmed or rejected from the one-point front that remained after applying
the shared 98% floor.

## 7. Reproducibility

### Source identities

All three repositories used branch
`rarunachalam/pre-platform-sdk-removal-20260714`:

| Repository | Commit |
| --- | --- |
| `~/tao-automl` | `09f9fe53a6050422a928046072397b41c4fdd857` |
| `~/tao-sdk` | `3d3e1adc1849493d29dc926cb99492417e3a9250` |
| `~/tao-skills-external` | `18f831c7c83b424861a60353fb735dd80efcfded` |

The algorithm implementation is AutoML commit
`171b47c` (“make multi-objective selection Pareto-safe”); the two following
commits repair smoke-only DINO PTM-head and evaluation-schema handling:
`7a45466` and `c0393ea`. The 30-candidate archive was launched and frozen at
`c0393ead46605f133cef8121c7f1335777b49b90`. Commit
`5839c945e1d08c9638f2c8a0ddac377b6058f66a` adds explicit input identity,
unambiguous sample counts, and validation-only independent winner repeats;
`82d473f` persists read-only per-seed replay; and
`09f9fe53a6050422a928046072397b41c4fdd857` rejects non-positive latency
measurements and proves that finite dominated outliers cannot alter
front-relative bounds. These evidence and selector-hardening commits do not
change the frozen measured archive or its mode selections.

### Data and model

```text
source:
  s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/
staged:
  /lustre/fsw/portfolios/edgeai/users/rarunachalam/data/tao_od_synthetic_full_dino_coco
train:
  1,414 images; 8,395 annotations
  annotations SHA-256:
  7401a1245dc0b691c40f9f53cf4f46f9b96a3e0bc3dcfd357de038074acc1994
validation:
  353 images; 2,186 annotations
  annotations SHA-256:
  9b715b689e9a17588805faad26ed94597886d28ac687438dcb778de433f997af
category IDs:
  1, 2, 3, 4
PTM:
  /lustre/fsw/portfolios/edgeai/users/rarunachalam/ptm/pretrained_dino_coco/
  dino_resnet_50_trainable_v1.0/dino_resnet50_ep12.pth
  SHA-256:
  7a391fb84a18714b60258becdb512594ec54faff5dccbf17ca53c5d902137512
selected checkpoint SHA-256:
  f09741401515e972b9b82bae4b80d7d800aca4db7a1a87f66bebb10b204c3d2e
```

The raw COCO PTM has a 91-class classifier and is used that way only for the
compatibility smoke. Trained candidates and evaluation use five output classes
with evaluation IDs 1–4. No GCViT or FAN repair was attempted.

### Search and training configuration

```text
algorithm: Bayesian
search seeds: 314159, 271828, 161803
recommendations: 10 per seed, 30 total
training seed: 1234
epochs: 10
parameters:
  model.num_queries: [300, 900]
  train.optim.lr: [1e-5, 5e-4]
  train.optim.weight_decay: [1e-5, 1e-3]
training: 8 GPUs, DDP, FP32, batch 4/GPU (global 32)
cuDNN benchmark: false
cuDNN deterministic: true
accuracy evaluation: 8 GPUs, FP32, batch 4/GPU,
  resize 800/max 1333, fixed padding
selection:
  accuracy tolerance: 1e-12
  retained accuracy: relative 0.98 of measured accuracy winner
  latency tolerance: 0 ms plus 95%-CI overlap equivalence
  normalization: feasible Pareto front
  weights: 1.0 accuracy, 1.0 latency -> 0.5, 0.5
  augmentation rho: 1e-6
  score tolerance: 1e-12
```

The authoritative machine-readable configuration is
`experiments/dino_moo_review_20260727/launch_manifest.json`; the full driver is
`run_experiment.py`, and the container benchmark is
`dino_latency_benchmark.py`.

### Runtime and hardware

```text
SLURM partition: polar3
account: edgeai_tao-ptm_image-foundation-model-clip
nodes/job: 1
GPUs/job: 8
GPU: NVIDIA A100-SXM4-80GB, compute capability 8.0
driver: 535.129.03
container Python: 3.12.3
PyTorch: 2.11.0
CUDA: 13.2
cuDNN: 92000
local aggregation Python: 3.14.6
local NumPy: 2.5.1
image tag: nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt
SQSH:
  /lustre/fsw/portfolios/edgeai/users/rarunachalam/
  nvcr.io_nvidia_tao_tao-toolkit_7.0.1-pyt.sqsh
SQSH SHA-256:
  88ba75e3a8eb9524fc0dbf026f2ea5da2c68696ae8d918b0afde5e0384ca641e
```

`SLURM_USE_SQSH=false` in the driver is intentional: the image argument is
already the exact absolute `.sqsh` path, so this disables registry conversion
or fallback while Pyxis still launches the SQSH directly
(`run_experiment.py:705-712`). The smoke freezes and subsequent benchmarks
enforce the hardware contract in `hardware_contract.json`.

### Commands

From the recorded commits:

```bash
cd ~/tao-automl

# Exact editable environment used by the controller and tests.
source /localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/activate

set -a
source ~/.tao/config.env
set +a

# Unit, integration, and property tests.
python -m pytest -q

# Required launch/hardware smoke.
python experiments/dino_moo_review_20260727/run_experiment.py --smoke

# Three seeded sub-archives; each candidate uses 8-GPU SLURM jobs.
python experiments/dino_moo_review_20260727/run_experiment.py

# Pure deterministic recomputation from persisted measurements; launches no job.
python experiments/dino_moo_review_20260727/run_experiment.py --combine-only

# Three independent 8-GPU validation-only repeats of the frozen unique winner.
# Safe to rerun: a complete matching artifact returns without launching jobs.
python experiments/dino_moo_review_20260727/repeat_selected_winner_latency.py
```

The full command first verifies the successful smoke and frozen hardware
contract (`run_experiment.py:1255-1289`), launches one process per search seed,
waits for all three, requires zero exit codes, and invokes the union selector
only afterward (`:1291-1337`). Re-running `--combine-only` is the direct proof
that the persisted objective values, rather than human interpretation, select
all three winners. It also writes `per_seed_selection.json` as a read-only
stability replay (`run_experiment.py:995-1043`). The repeat command requires
that frozen selection to have one unique winner across the three modes and
records `validation_only=true` and `feeds_selection=false`.

### Evidence and logs

- `launch_manifest.json`: commits, exact model/data/runtime/search/selection
  configuration;
- `hardware_contract.json`: frozen eight-A100 container contract;
- `smoke/result.json` and `smoke/events.jsonl`: smoke metrics and job IDs;
- `seed_<seed>/candidate_evaluations.json`: per-candidate mAP50, latency
  statistics, hardware, checkpoint, TAO job ID, and SLURM job ID;
- `seed_<seed>/events.jsonl`: recommendation/submission/completion event stream;
- `seed_<seed>/result.json`: each Bayesian sub-archive result;
- `process_status.json`: all seed processes exited zero;
- `combined_selection.json`: complete selection configuration, objective values,
  normalization bounds, ranks, dominators, fingerprints, ties, reasons, and
  mode flags;
- `per_seed_selection.json`: read-only selector evidence for each ten-candidate
  seed archive; it explicitly does not feed union selection;
- `full_candidate_table.csv`: all 30 candidate parameters and measurements;
- `winner_latency_repeats.json`: terminal three-allocation repeat evidence,
  input shapes/hashes, aggregate median and p95 variation, frozen mode-selection
  snapshot, and `feeds_selection=false`;
- `integrity_audit.json`: 104/104 independent consistency checks, hashes for
  source/result artifacts, and live accounting showing all 90 primary plus
  three repeat allocations completed with exit `0:0` and eight GPUs;
- `winner_latency_repeats/events.jsonl`: repeat submission/completion events;
- each candidate record's `latency.raw_samples_dir`: the eight rank JSON files
  containing all 4,000 raw request samples and per-rank runtime/hardware.

Winner identities:

```text
training TAO job:
  768b852a-aa1d-42c0-b0ec-586f5ceca222
checkpoint:
  /lustre/fs11/portfolios/edgeai/projects/
  edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/results/
  768b852a-aa1d-42c0-b0ec-586f5ceca222/results_dir/train/
  model_epoch_009_step_00440.pth
accuracy TAO / SLURM:
  56ad4580-ce0a-4ba0-862e-2ae063f0a0bd / 30917205
latency TAO / SLURM:
  7eb78cc8-be0b-48e0-84e4-ceaa2736802a / 30917544
raw latency samples:
  /lustre/fs11/portfolios/edgeai/projects/
  edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/results/
  7eb78cc8-be0b-48e0-84e4-ceaa2736802a/latency
```

Two infrastructure-stalled allocations were requeued only after exceeding four
times the normal observed phase duration: SLURM `30913697` (latency container
prolog) and `30913760` (candidate training). Requeue retained the same TAO job,
candidate, parameters, seed, and ordering; each completed with one restart. It
did not change or override selection.

### Remaining reproducibility limits

1. Independent repeats show that within-job bootstrap intervals do not include
   allocation-level shifts. Treat differences below the observed four-job
   `0.73553775 ms` envelope as unresolved unless every candidate being compared
   receives matched independent allocations. The final winner is unaffected
   here because it is the only accuracy-feasible candidate.
2. The three new repeats happened on one node (`batch-block7-00843`) and the
   original selection measurement on another (`batch-block7-03393`). This is
   independent-allocation and two-node evidence on A100, not a fleet-wide or
   H100 portability result.
3. The hardened repeat records persist exact input metadata and digest for the
   selected winner. The earlier 30 candidate raw records predate that hardening,
   so the same deterministic configuration is recorded for them but their rank
   files do not retroactively gain the complete workload digest.
4. `per_device_sample_count=4000` remains as a backward-compatible historical
   alias. New serialized fields remove ambiguity:
   `raw_sample_count_total=4000` and `samples_per_device=500`.
5. The smoke-frozen hardware contract is enforced for latency, the
   hardware-sensitive deployment objective. Training/evaluation use the same
   requested SLURM runtime, but their result records do not provide the same
   rank-level hardware proof.

These limits do not affect the algorithm-only selection or the conclusion that
a dominated compromise can no longer be returned. They limit fine-grained
latency ordering and generalization beyond the measured archive.
