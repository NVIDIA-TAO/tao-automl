# DINO multi-objective AutoML phase-2 validation

Current sealed-evidence cutoff: AutoML commit
`2307d86a9a2cf6c0883a977a6dfcd8e1f885ea77`. The matched analysis records
`created_at_utc=2026-07-28T10:44:10Z`; that is the artifact creation time,
not the commit timestamp. The corrected runtime implementation is commit
`e4b6a412545614668affd371a82231e090998ec0`; the corrected v2 manifest was
frozen at commit `b1a0ae235be53ba3ced7e4c880cb0be1f6b8157d`; the
60-candidate archive and algorithmic selections were sealed at
`a6fc0bbd7947cf58b95f1c037b0513092b31a2f9`; the immutable post-front
manifest was frozen at
`a3099803f4a4fa7494c9564c0b6806576a203d7b`; all six post-front
allocations were durably submitted at
`8289d5988f55149857c8e04340c0580d470de11e`; and their complete matched
analysis was sealed at the current evidence cutoff.

The phase-two protocol erratum was issued at `2026-07-28T06:36:41Z` and
committed at
`ba2cf95211ecddc9cb38dfe51d189357b05dc8e2`. Its whole-file SHA-256 is
`95bba65099027459a50b5e74e43a4ab32c56057e534e70aa7f85bdc9246a7d13`.
At issuance, exactly 15 of 60 expanded candidates were successful—five for
each search seed—and the complete union selection, candidate-table JSON and
CSV, integrity audit, completion record, final global Pareto front, and every
post-front manifest, job, and measurement were absent. The erratum therefore
aligned stale tie-break prose to the already-pinned production selector and
froze the two-branch post-front inference policy before a final winner or
post-front result could be known; it did not change an objective value,
selection setting, search range, budget, Pareto rank, or winner.

The earlier six-record rec0/rec1 snapshot remains valid provenance but has
been superseded as the report's current result by three exact 20-record seed
archives. All 60 candidates completed successfully, no candidate was injected
manually, and the combined selection was reproduced under archive, reverse,
and candidate-ID order before sealing. The post-front protocol-binding
implementation remains commit
`6850d71c2f3dea5f37505dd6831d41cb07a4d255`; the generated manifest and
submitted jobs passed its exact-source, immutable-archive, dataset, PTM,
SQSH, checkpoint, and runtime gates.

Scope: DINO ResNet50 only, using
`s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/`.
No other model family, PTM compatibility repair, or dataset is included.

This report for MR !22 records immutable sensitivity
evidence, the failed-and-excluded v1 execution, the complete corrected v2
archive, the algorithm-only mode selections, the frozen four-candidate
post-front manifest, six completed matched allocations, and the immutable
post-front analysis. Matched results are validation-only: no matched value
replaced a selection-time objective or altered a winner.

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
| Corrected expanded shared 60-candidate archive | Complete and sealed | 60/60 successful, 20 per deterministic search seed, zero failures, zero manual injections. Accuracy and constrained-latency select `seed_271828_rec_18`; no-floor multi-objective selects `seed_271828_rec_19`. |
| Pre-post-front protocol erratum | Complete and immutable before final selection or post-front data | Issued at `2026-07-28T06:36:41Z`, committed at `ba2cf95…`, whole-file SHA-256 `95bba650…`. It preserves the original bootstrap classification and separately makes the stricter exact-sign-flip/all-six rule authoritative for effective directional claims. |
| Post-front hardening and protocol binding | Complete and frozen | Exact reconstruction binds the three seed archives, canonical 60-record union, semantic JSON and byte-exact CSV projections, combined selection, integrity audit, four retained checkpoints, and the protocol erratum. Manifest whole/internal hashes are `d468d5d2…` / `c49eb5eb…`. |
| Matched remeasurement of final Pareto front | Complete | Six one-node/eight-A100 jobs completed `COMPLETED/0:0`; all 24 candidate/allocation cells are valid with 4,000 samples each. Five of six median pairs have an effective stable direction; `rec_15` versus `rec_3` has no stable direction. |
| Final combined selection | Complete and sealed | Algorithm-only winners are `seed_271828_rec_18` (accuracy), `seed_271828_rec_18` (98%-retained latency), and `seed_271828_rec_19` (multi-objective). |
| Final hypothesis verdict | **Partially supported** | The selector found a nondominated, stable geometric compromise, but accuracy and 98%-constrained latency both select `rec_18`; therefore the multi-objective point cannot lie strictly between two distinct actual mode extremes. |

The required historical correction is:

> The frozen historical 30-candidate archive contains six rank-zero Pareto
> candidates. No distinct Pareto compromise exists under the configured 98%
> multi-objective accuracy-feasibility constraint.

The first sentence must not be shortened to “the archive has no intermediate
candidate.” The historical global archive does contain intermediate trade-off
points; the old shared 98% floor excluded them from multi-objective scoring.

That statement refers only to the historical 30-candidate archive used in
Phase 1. The corrected expanded archive is a different, 60-candidate shared
archive. Its global rank-zero front contains exactly four candidates:
`seed_271828_rec_15`, `seed_271828_rec_18`,
`seed_271828_rec_19`, and `seed_271828_rec_3`. The historical six-point
front and expanded four-point front must not be conflated.

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

The selector does not force a middle point. Its
`distinct_compromise` field is a geometric property of the
multi-objective-eligible front: it records whether the selected rank-zero
point is distinct, under the configured tolerances, from that population's
accuracy extreme and its unconstrained latency extreme. The latter is not
necessarily the actual latency-mode winner, because latency mode separately
applies the 98% accuracy-retention constraint.

When no distinct eligible point exists, it emits:

> No distinct Pareto compromise exists under the configured multi-objective
> eligibility policy.

The deterministic Chebyshev ordering may still return an extreme, but the
audit sets `distinct_compromise=false` and distinguishes a fallback from a
successful compromise.

The phase-two hypothesis uses a second, non-selector definition. It compares
the frozen multi-objective winner with the actual accuracy winner and the
actual 98%-constrained latency-mode winner, then evaluates the selection-time
accuracy/latency geometry and the matched post-front latency evidence. This
verdict-only comparison never changes the selector's geometric flag, candidate
identity, or winner.

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

### 2.6 Pre-post-front tie-break documentation erratum

The expanded manifest's frozen prose summarized the multi-objective tie-break
as lower maximum regret, then lower regret sum, higher accuracy, lower
latency, and a deterministic key. That prose was not the implementation
executed by the already-pinned selector. The protocol erratum changes no code;
it makes the following pre-existing production behavior authoritative:

- canonical candidate-audit order is specification-fingerprint ascending,
  then candidate ID;
- accuracy mode takes every point within `1e-12` of the best accuracy, anchors
  at that cohort's raw minimum latency, retains points within
  `0.73553775 ms` of that anchor, then uses fingerprint and candidate ID;
- latency mode first applies its accuracy-retention constraint, anchors at raw
  minimum latency (with canonical fingerprint ordering), regards a candidate
  as latency-tied when it is within `0.73553775 ms` of the anchor or its
  latency confidence interval overlaps the anchor's interval, then orders the
  tied cohort by higher accuracy, fingerprint, and candidate ID; and
- multi-objective mode filters to its deduplicated eligible rank-zero front,
  minimizes augmented-Chebyshev score, then uses ideal distance, balance gap,
  normalized accuracy regret, fingerprint, and candidate ID, with
  `1e-12` score-stage tolerances.

The erratum records
`selection_execution_changed=false`,
`candidate_objectives_changed=false`,
`pareto_ranks_changed=false`, and
`algorithm_winner_changed_or_overridden=false`.

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

The added parameterization contributes 13 test cases.

At the pre-post-front protocol cutoff
`6850d71c2f3dea5f37505dd6831d41cb07a4d255`, the focused protocol and
post-front suite was:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  -m pytest -q \
  experiments/dino_moo_phase2_20260728/test_phase2_protocol_erratum.py \
  experiments/dino_moo_phase2_20260728/test_post_front_protocol_binding.py \
  experiments/dino_moo_phase2_20260728/test_post_front_protocol_analysis.py \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_tools.py \
  experiments/dino_moo_phase2_20260728/test_post_front_matched_launcher_recovery.py \
  experiments/dino_moo_phase2_20260728/test_post_front_complete_invalid_recovery.py
```

```text
134 passed
```

The complete phase-two experiment suite was also rerun:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  -m pytest -q experiments/dino_moo_phase2_20260728
```

```text
259 passed
```

The unchanged production core suite remained:

```bash
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  -m pytest -q tests
```

```text
387 passed, 1 skipped
```

The final pre-analysis source validation at HEAD
`8289d5988f55149857c8e04340c0580d470de11e` repeated both suites:

```text
phase-two suite: 259 passed in 2.25s
production core: 387 passed, 1 skipped in 4.82s
post-front tools: 4/4 compiled in memory
git diff --check: clean
```

The core run emitted one non-failing scikit-learn warning; there were no test
failures. Commit `2307d86a9a2cf6c0883a977a6dfcd8e1f885ea77` adds only the
immutable analysis artifact and does not change selector or experiment code.

The combined phase-two run covers the provenance-safe analysis erratum,
immutable runtime contract, strict native-number/JSON-number-string metric
parsing, rejection of booleans, NaN, infinity, whitespace, and junk, clean
v1-to-v2 runtime supersession, manifest derivation, exact integer-domain
encoding, remote preflight, resume/reconciliation behavior,
complete-archive gating, candidate-order invariance, and algorithm-only union
selection. The new protocol cases prove the exact phase-two erratum is
required by generation, launch, and aggregation; the original and effective
inference branches cannot be silently merged; the three 20-record seed
archives form exactly one canonical 60-record union; JSON row semantics and
the byte-exact CSV projection are derived from that union; archive order does
not affect the canonical digests; and drift in any seed archive, combined
selection, candidate-table JSON/CSV, or integrity-audit binding fails closed.
The post-front cases additionally cover full rank-zero derivation,
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

The protocol-analysis tests separately verify the selector-geometric and
actual-mode-winner distinctness definitions, preserve selection-time
objective values, report matched aggregate medians and p95 values as
validation-only evidence, and assert that no selector is invoked on matched
measurements.

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

If any level qualifies, the expanded shared search admits the axis's complete
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
subsequent measurement are new. Parameter reproducibility is not v1 state
reuse.

At the launch evidence snapshot, `sacct` reported all three as `RUNNING` with
one node, eight allocated GPUs, partition `polar3`, and exit field `0:0`.
That statement is retained as launch-time provenance; the same three jobs later
completed as the rec0 training jobs reported below.

#### 7.1.3 Complete corrected v2 archive and algorithm-only selection

The corrected execution completed all 20 sequential Bayesian recommendations
for each of search seeds `314159`, `271828`, and `161803`. The three seed
archives contain exactly 60 terminal records: 60 successful, zero failed, and
zero manually injected. Every successful row has finite mAP50 and a valid
selection-time latency record produced by the same frozen 50-warm-up,
five-round-by-100-iteration, eight-rank protocol. Training seed `1234`,
dataset, PTM, SQSH, batch size, precision, input digest, and timed scope are
identical across the archive.

The archive identities are:

| Search seed | Records | Successful | Whole-file SHA-256 | Internal archive SHA-256 |
| ---: | ---: | ---: | --- | --- |
| 314159 | 20 | 20 | `c8a1f937a6208ba2e9bd305a272b7a94b2269a46ef5e683d67d01596d6d5c044` | `45418f66fa435d35427725c64f7e2c58b48dbdd56397c43e3e07575185a5c424` |
| 271828 | 20 | 20 | `a42a989ea27940ea9ae481212a75216c7f23f01602b0c260b6750c9fdb709c9e` | `eedaa0a37e49cfa86e54be15a56352e4891044856ad5db782f6a4eed464dfb36` |
| 161803 | 20 | 20 | `0057080db477db3acd544ec360ce08b1c1c902a3d6ae820f52180f8d492109c9` | `7fe334b80f03cee36a6c5d283574cc36d8ad7ab5d098c9d2f5f1389a94240f34` |

The production selector consumed the canonical union of those archives.
Archive order, reverse order, and candidate-ID order produced the same
selection signature
`75fa7afe25d0fda3dd50b96c405434ebf52a57727cd3f04ac3bba5b006a5f11a`.
The authority records `manual_override_used=false` and
`candidate_reordering_used=false`.

The accuracy reference is
\(A^*=0.6554138278683255\), so the frozen 98% latency threshold is
`0.6423055513109589`. Four candidates satisfy it:
`seed_161803_rec_14`, `seed_271828_rec_16`,
`seed_271828_rec_18`, and `seed_314159_rec_12`. Their selection-time
latencies all fall in the configured `0.73553775 ms` tied cohort anchored at
the raw minimum. The deterministic higher-accuracy tie-break selects
`seed_271828_rec_18`, which is also the accuracy winner.

Multi-objective mode has no accuracy floor, so all 60 valid candidates are
eligible. Its front-relative bounds are:

| Objective | Direction | Ideal | Nadir | Range |
| --- | --- | ---: | ---: | ---: |
| mAP50 | maximize | 0.6554138278683255 | 0.5398520557657904 | 0.11556177210253504 |
| Latency ms | minimize | 52.04909275 | 66.23099475000001 | 14.181902000000008 |

The expanded global rank-zero front contains exactly four points:

| Candidate | Enc/dec | mAP50 | Selection-time median ms | p95 ms | Accuracy regret | Latency regret | Chebyshev score | Ideal distance | Balance gap | Winner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `seed_271828_rec_18` | 6/3 | 0.6554138278683255 | 66.23099475000001 | 66.59742 | 0 | 1 | 0.5000005 | 0.5 | 0.5 | Accuracy and latency |
| `seed_271828_rec_3` | 3/3 | 0.5398520557657904 | 52.04909275 | 52.27028595 | 1 | 0 | 0.5000005 | 0.5 | 0.5 | — |
| `seed_271828_rec_15` | 3/3 | 0.5606606395568864 | 52.0782885 | 52.351142 | 0.8199354041349156 | 0.002058662512263835 | 0.40996811306449116 | 0.4099689942682333 | 0.4089383708113259 | — |
| `seed_271828_rec_19` | 4/3 | 0.6175134981289873 | 57.146624 | 57.362506 | 0.32796597914499104 | 0.3594391817120158 | 0.1797199345585883 | 0.24328903018135456 | 0.01573660128351237 | Multi-objective |

The normalized augmented-Chebyshev rule therefore selects
`seed_271828_rec_19` without an override. Its selector-geometric
`distinct_compromise` flag is `true`, and the audit confirms the selected
point is nondominated. It retains `94.21734358846141%` of the accuracy
winner and is `9.084370750000005 ms` faster by the original, unmatched
selection-time medians. Those latency differences are not yet
allocation-stability claims.

### 7.2 Complete expanded candidate table

All 60 records are successful and valid; there is no failed-candidate table.
Training seed is `1234` for every row. `E/D` is encoder/decoder depth.
`MAD/IQR` and all latency values are milliseconds from the original
selection-time allocation. `L/M eligible` means latency-mode 98% feasibility
and multi-objective feasibility. `Dby` is the exact global `dominated_by`
list using compact `search-seed/recommendation` notation; `—` denotes global
rank zero. Because multi-objective has no floor, its eligible Pareto rank and
eligible `dominated_by` relationship equal the global values for every row.
`rA`, `rL`, and `C` are normalized accuracy regret, normalized
latency regret, and augmented-Chebyshev score. Winner flags are `A`, `L`, and
`M`. Exact full-precision records, confidence intervals, job lineage,
fingerprints, ideal distances, balance gaps, and mode-specific tie tuples are
retained in `expanded_candidate_table.json` and
`expanded_combined_selection.json`.

| Candidate | E/D | Learning rate | Weight decay | mAP50 | Median | p95 | MAD / IQR | L/M eligible | Rank | Dby | rA | rL | C | Winner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `seed_161803_rec_0` | 3/5 | 0.0002098775727573059 | 0.0006123641159628601 | 0.5305309558679956 | 60.7300305 | 61.1440695 | 0.0832215000000005 / 0.24712874999999457 | N/Y | 5 | 161803/12, 161803/15, 161803/18, 161803/3, 161803/5, 271828/10, 271828/15, 271828/19, 271828/3, 271828/5, 271828/6, 271828/9, 314159/1, 314159/10, 314159/9 | 1.0806590252832438 | 0.6121137876992799 | 0.5403303590280284 | — |
| `seed_161803_rec_1` | 5/5 | 0.00030565624727243724 | 0.0005057619353603205 | 0.5527822169224204 | 70.18545575 | 71.29438999999999 | 0.13810249999999513 / 0.34892924999999764 | N/Y | 7 | 161803/11, 161803/12, 161803/14, 161803/15, 161803/3, 161803/6, 161803/8, 161803/9, 271828/1, 271828/10, 271828/12, 271828/13, 271828/15, 271828/16, 271828/17, 271828/18, 271828/19, 271828/6, 271828/8, 314159/12, 314159/17, 314159/18, 314159/19, 314159/4, 314159/7, 314159/9 | 0.8881103939358309 | 1.2788385507106164 | 0.6394203588297805 | — |
| `seed_161803_rec_10` | 6/3 | 1.1000000000000001e-05 | 1.1000000000000001e-05 | 0.3857981269205332 | 66.59312150000001 | 66.73719344999999 | 0.07733699999999999 / 0.1569314999999989 | N/Y | 10 | 161803/0, 161803/11, 161803/12, 161803/14, 161803/15, 161803/17, 161803/18, 161803/2, 161803/3, 161803/4, 161803/5, 161803/7, 161803/8, 161803/9, 271828/1, 271828/10, 271828/11, 271828/12, 271828/15, 271828/17, 271828/18, 271828/19, 271828/3, 271828/5, 271828/6, 271828/7, 271828/8, 271828/9, 314159/1, 314159/10, 314159/13, 314159/15, 314159/16, 314159/17, 314159/18, 314159/19, 314159/2, 314159/6, 314159/7, 314159/8, 314159/9 | 2.333087283470948 | 1.0255344276106266 | 1.1665453210463295 | — |
| `seed_161803_rec_11` | 5/3 | 0.0002684985944234011 | 0.0009485964192103246 | 0.5984845568353709 | 61.89238125 | 62.149561049999996 | 0.09910775000000172 / 0.207902500000003 | N/Y | 2 | 271828/1, 271828/19, 271828/6 | 0.4926306510983808 | 0.6940739331014976 | 0.34703755990304086 | — |
| `seed_161803_rec_12` | 3/5 | 0.00024962684684468685 | 0.0006066305410362497 | 0.5607754726645933 | 60.51118675 | 60.7231585 | 0.0989015000000002 / 0.20061400000000162 | N/Y | 4 | 161803/3, 271828/19, 271828/6, 314159/9 | 0.818941709545281 | 0.5966825888375196 | 0.4094715625847897 | — |
| `seed_161803_rec_13` | 3/6 | 1.1000000000000001e-05 | 0.00012679359566591052 | 0.3050856249448289 | 65.24652825 | 65.519609 | 0.11789475000000493 / 0.23935849999999448 | N/Y | 9 | 161803/0, 161803/11, 161803/12, 161803/15, 161803/17, 161803/18, 161803/3, 161803/4, 161803/5, 161803/7, 161803/8, 161803/9, 271828/1, 271828/10, 271828/12, 271828/15, 271828/17, 271828/19, 271828/3, 271828/5, 271828/6, 271828/7, 271828/9, 314159/1, 314159/10, 314159/13, 314159/15, 314159/16, 314159/6, 314159/8, 314159/9 | 3.0315232844703974 | 0.9305829006574711 | 1.5157636232882912 | — |
| `seed_161803_rec_14` | 6/3 | 0.00045 | 0.0007784663668118407 | 0.6503731411565659 | 66.58940949999999 | 66.9217075 | 0.09126249999999914 / 0.18958050000000526 | Y/Y | 1 | 271828/18 | 0.043618980741201664 | 1.0252726855678442 | 0.5126368772297553 | — |
| `seed_161803_rec_15` | 3/3 | 0.0004662258975438763 | 0.0004027148908248362 | 0.5585657123145856 | 52.325866250000004 | 52.529766550000005 | 0.08674200000000098 / 0.16722749999999564 | N/Y | 1 | 271828/15 | 0.8380636069496145 | 0.019515964783849465 | 0.4190322322645931 | — |
| `seed_161803_rec_16` | 3/4 | 1.1000000000000001e-05 | 0.0005887444947583944 | 0.24516113032364692 | 56.46002675 | 56.80340455 | 0.09855674999999664 / 0.19451249999999476 | N/Y | 6 | 161803/15, 161803/17, 161803/18, 161803/5, 161803/7, 271828/10, 271828/15, 271828/3, 271828/5, 271828/9, 314159/1, 314159/10, 314159/16 | 3.550072745342393 | 0.3110255591950921 | 1.7750383032203487 | — |
| `seed_161803_rec_17` | 3/3 | 0.0004137239694774212 | 0.0004155351609850492 | 0.5189410586254114 | 52.06582875 | 52.34656345 | 0.06829600000000013 / 0.1454854999999995 | N/Y | 1 | 271828/3 | 1.1809508175577754 | 0.0011800955894351569 | 0.5904759998443443 | — |
| `seed_161803_rec_18` | 3/3 | 0.0003230696641703003 | 0.0009401130670142016 | 0.5481507932869331 | 52.13929525 | 52.3762605 | 0.1054797500000042 / 0.2195315000000022 | N/Y | 1 | 271828/15 | 0.9281878655012371 | 0.006360395100742037 | 0.46409440002474883 | — |
| `seed_161803_rec_19` | 6/6 | 0.00017589779877069415 | 0.0006023857870527973 | 0.5880638351234523 | 79.37313775 | 79.663423 | 0.07717225000000383 / 0.15934024999999963 | N/Y | 4 | 161803/11, 161803/14, 271828/1, 271828/12, 271828/13, 271828/16, 271828/18, 271828/19, 271828/6, 271828/8, 314159/11, 314159/12, 314159/17, 314159/18, 314159/4, 314159/5, 314159/7 | 0.5828051224856196 | 1.926684093572215 | 0.9633433015307155 | — |
| `seed_161803_rec_2` | 4/5 | 0.00011443965835046119 | 0.0008389293930662631 | 0.5146082114188394 | 65.5497615 | 65.96043750000001 | 0.1298180000000002 / 0.35155474999999115 | N/Y | 7 | 161803/0, 161803/11, 161803/12, 161803/15, 161803/17, 161803/18, 161803/3, 161803/4, 161803/5, 161803/7, 161803/8, 161803/9, 271828/1, 271828/10, 271828/12, 271828/15, 271828/17, 271828/19, 271828/3, 271828/5, 271828/6, 271828/9, 314159/1, 314159/10, 314159/13, 314159/16, 314159/9 | 1.2184445936373562 | 0.9519646060168795 | 0.609223382023278 | — |
| `seed_161803_rec_3` | 4/3 | 0.00024240840108533996 | 1.1000000000000001e-05 | 0.585087593426202 | 57.24132175 | 57.445274950000005 | 0.09651700000000218 / 0.19750125000000196 | N/Y | 2 | 271828/19, 271828/6 | 0.6085596747315782 | 0.3661165476957882 | 0.3042803247039003 | — |
| `seed_161803_rec_4` | 3/5 | 0.0003556288785750765 | 1.1000000000000001e-05 | 0.530571016563574 | 61.2095875 | 61.69216195 | 0.11362349999999921 / 0.23820849999999893 | N/Y | 5 | 161803/12, 161803/15, 161803/18, 161803/3, 161803/5, 271828/10, 271828/15, 271828/19, 271828/3, 271828/5, 271828/6, 271828/9, 314159/1, 314159/10, 314159/9 | 1.0803123648361987 | 0.6459285045122998 | 0.540157045538534 | — |
| `seed_161803_rec_5` | 3/3 | 0.00045 | 0.0007152414221498921 | 0.5335163241008113 | 52.287032499999995 | 53.0490875 | 0.14563950000000503 / 0.28957600000000383 | N/Y | 4 | 161803/18, 271828/10, 271828/15, 271828/3, 271828/5, 271828/9 | 1.0548254976512268 | 0.01677770372408406 | 0.527413284627214 | — |
| `seed_161803_rec_6` | 6/3 | 0.00012479149198942115 | 0.00018314874110030832 | 0.5712836532478843 | 66.61979275 | 67.00445495000001 | 0.09688649999999654 / 0.1914665000000042 | N/Y | 6 | 161803/11, 161803/14, 161803/3, 161803/8, 271828/1, 271828/12, 271828/17, 271828/18, 271828/19, 271828/6, 271828/8, 314159/17, 314159/18, 314159/19, 314159/7, 314159/9 | 0.7280104232547989 | 1.0274150815595817 | 0.5137084184925432 | — |
| `seed_161803_rec_7` | 3/3 | 0.00041193916379915263 | 0.0002822306155890354 | 0.5298053616731744 | 52.33808325 | 52.501121049999995 | 0.09205275000000412 / 0.24800000000000466 | N/Y | 5 | 161803/15, 161803/18, 161803/5, 271828/10, 271828/15, 271828/3, 271828/5, 271828/9, 314159/10 | 1.0869378680321886 | 0.020377414820663468 | 0.5434694876737357 | — |
| `seed_161803_rec_8` | 4/4 | 0.00045 | 0.0007280636459500239 | 0.580870432307355 | 61.2660045 | 61.54148905 | 0.16006399999999843 / 0.3210045000000008 | N/Y | 3 | 161803/3, 271828/19, 271828/6 | 0.6450523750607595 | 0.6499066027955909 | 0.32495394887728435 | — |
| `seed_161803_rec_9` | 3/6 | 0.00045 | 6.930370367983633e-05 | 0.5542465901400749 | 64.86551850000001 | 65.102609 | 0.07949649999999764 / 0.1579940000000022 | N/Y | 5 | 161803/11, 161803/12, 161803/15, 161803/3, 161803/8, 271828/1, 271828/10, 271828/12, 271828/15, 271828/17, 271828/19, 271828/6, 314159/9 | 0.8754386151026431 | 0.9037169873265236 | 0.451859383241063 | — |
| `seed_271828_rec_0` | 6/6 | 4.59777499171801e-05 | 0.0006077207436969115 | 0.5140622451913158 | 79.26377375000001 | 79.58547200000001 | 0.15291124999998118 / 0.3221730000000065 | N/Y | 10 | 161803/0, 161803/1, 161803/11, 161803/12, 161803/14, 161803/15, 161803/17, 161803/18, 161803/2, 161803/3, 161803/4, 161803/5, 161803/6, 161803/7, 161803/8, 161803/9, 271828/1, 271828/10, 271828/12, 271828/13, 271828/15, 271828/16, 271828/17, 271828/18, 271828/19, 271828/2, 271828/3, 271828/4, 271828/5, 271828/6, 271828/8, 271828/9, 314159/0, 314159/1, 314159/10, 314159/11, 314159/12, 314159/13, 314159/14, 314159/16, 314159/17, 314159/18, 314159/19, 314159/4, 314159/5, 314159/7, 314159/9 | 1.2231690472139174 | 1.9189725750467037 | 0.959487858594163 | — |
| `seed_271828_rec_1` | 5/3 | 0.0004417551531468059 | 0.0004796687699134978 | 0.6044653164228678 | 61.82193 | 62.0668055 | 0.08563599999999738 / 0.16950549999999964 | N/Y | 1 | 271828/19 | 0.440876861945768 | 0.6891062461156477 | 0.3445536880493779 | — |
| `seed_271828_rec_10` | 3/3 | 0.00045 | 1.1000000000000001e-05 | 0.5557698152865863 | 52.26119725 | 52.491550049999994 | 0.12237349999999836 / 0.24458775000000088 | N/Y | 1 | 271828/15 | 0.8622575681283906 | 0.01495599814467779 | 0.4311292226709784 | — |
| `seed_271828_rec_11` | 3/6 | 0.00045 | 1.1000000000000001e-05 | 0.5131015716735464 | 65.43014475000001 | 65.700018 | 0.1268024999999966 / 0.25446175000000437 | N/Y | 7 | 161803/0, 161803/11, 161803/12, 161803/15, 161803/17, 161803/18, 161803/3, 161803/4, 161803/5, 161803/7, 161803/8, 161803/9, 271828/1, 271828/10, 271828/12, 271828/15, 271828/17, 271828/19, 271828/3, 271828/5, 271828/6, 271828/9, 314159/1, 314159/10, 314159/13, 314159/16, 314159/9 | 1.2314821208220048 | 0.9435301414436514 | 0.6157421479171336 | — |
| `seed_271828_rec_12` | 5/3 | 0.00045 | 0.0009 | 0.5924694360231197 | 61.770416749999995 | 61.930805 | 0.06519600000000025 / 0.13096399999999875 | N/Y | 2 | 271828/19, 271828/6 | 0.5446817810076221 | 0.685473923032326 | 0.34273757659401505 | — |
| `seed_271828_rec_13` | 6/3 | 0.0003409813457951682 | 0.00037590573325465534 | 0.6388208734864198 | 66.6654645 | 66.9667695 | 0.10486700000000582 / 0.20982474999999567 | N/Y | 2 | 161803/14, 271828/18 | 0.14358515000257316 | 1.030635506436301 | 0.5153183403284788 | — |
| `seed_271828_rec_14` | 6/6 | 0.00045 | 0.0009 | 0.5790024762594287 | 79.4215175 | 79.70750505 | 0.08426100000000503 / 0.17125799999999458 | N/Y | 5 | 161803/11, 161803/14, 161803/19, 161803/3, 161803/8, 271828/1, 271828/12, 271828/13, 271828/16, 271828/17, 271828/18, 271828/19, 271828/6, 271828/8, 314159/11, 314159/12, 314159/14, 314159/17, 314159/18, 314159/4, 314159/5, 314159/7, 314159/9 | 0.6612165097390418 | 1.9300954660383338 | 0.9650490286751549 | — |
| `seed_271828_rec_15` | 3/3 | 0.0004560144015085677 | 0.0008350193861096457 | 0.5606606395568864 | 52.0782885 | 52.351142 | 0.07303750000000164 / 0.15044174999999882 | N/Y | 0 | — | 0.8199354041349156 | 0.002058662512263835 | 0.40996811306449116 | — |
| `seed_271828_rec_16` | 6/3 | 0.0003007572504594793 | 1.1000000000000001e-05 | 0.6544218576499151 | 66.82186425 | 67.0504435 | 0.10689974999999663 / 0.20638999999999896 | Y/Y | 1 | 271828/18 | 0.008583895871120805 | 1.0416636287572707 | 0.5208323395023977 | — |
| `seed_271828_rec_17` | 5/3 | 0.00045 | 1.1000000000000001e-05 | 0.5803046876147155 | 61.61185975 | 61.8523905 | 0.10302649999999858 / 0.20704899999999782 | N/Y | 4 | 161803/3, 161803/8, 271828/19, 271828/6, 314159/9 | 0.6499479792241978 | 0.6742936878283319 | 0.3371475060349995 | — |
| `seed_271828_rec_18` | 6/3 | 0.00045 | 0.0001962863874708991 | 0.6554138278683255 | 66.23099475000001 | 66.59742 | 0.11910974999999269 / 0.24108624999999506 | Y/Y | 0 | — | 0 | 1 | 0.5000005 | AL |
| `seed_271828_rec_19` | 4/3 | 0.00045 | 0.0006630648780334237 | 0.6175134981289873 | 57.146624 | 57.362505999999996 | 0.08395600000000059 / 0.16272475000000242 | N/Y | 0 | — | 0.32796597914499104 | 0.3594391817120158 | 0.1797199345585883 | M |
| `seed_271828_rec_2` | 6/4 | 4.507647103898838e-05 | 0.0005017054111366341 | 0.5173866525337586 | 70.62054175 | 71.07720145 | 0.08689425000000028 / 0.2565385000000049 | N/Y | 9 | 161803/0, 161803/1, 161803/11, 161803/12, 161803/14, 161803/15, 161803/17, 161803/18, 161803/3, 161803/4, 161803/5, 161803/6, 161803/7, 161803/8, 161803/9, 271828/1, 271828/10, 271828/12, 271828/13, 271828/15, 271828/16, 271828/17, 271828/18, 271828/19, 271828/3, 271828/4, 271828/5, 271828/6, 271828/8, 271828/9, 314159/1, 314159/10, 314159/12, 314159/13, 314159/14, 314159/16, 314159/17, 314159/18, 314159/19, 314159/4, 314159/7, 314159/9 | 1.194401685118664 | 1.309517510415739 | 0.6547600071674673 | — |
| `seed_271828_rec_3` | 3/3 | 0.00026554662395385974 | 1.0000000000000028e-05 | 0.5398520557657904 | 52.04909275 | 52.27028595 | 0.09033925000000309 / 0.17970150000000018 | N/Y | 0 | — | 1 | 0 | 0.5000005 | — |
| `seed_271828_rec_4` | 4/6 | 0.00034936912534309716 | 0.0009371413334568777 | 0.5404276745153964 | 70.20810449999999 | 70.7079505 | 0.19523500000000382 / 0.37427824999998904 | N/Y | 8 | 161803/1, 161803/11, 161803/12, 161803/14, 161803/15, 161803/18, 161803/3, 161803/6, 161803/8, 161803/9, 271828/1, 271828/10, 271828/12, 271828/13, 271828/15, 271828/16, 271828/17, 271828/18, 271828/19, 271828/5, 271828/6, 271828/8, 314159/1, 314159/12, 314159/17, 314159/18, 314159/19, 314159/4, 314159/7, 314159/9 | 0.9950189518632921 | 1.2804355685154205 | 0.6402189219849704 | — |
| `seed_271828_rec_5` | 3/3 | 0.00045 | 0.0007030726224609912 | 0.5468671109871244 | 52.2364525 | 52.41068155 | 0.06547649999999905 / 0.1277900000000045 | N/Y | 2 | 161803/18, 271828/15 | 0.9392960570463591 | 0.013211186341578071 | 0.46964850477680126 | — |
| `seed_271828_rec_6` | 4/3 | 0.000487310659095131 | 0.0009 | 0.6000121414379619 | 57.17349525 | 57.37278405 | 0.0828967500000033 / 0.1657094999999984 | N/Y | 1 | 271828/19 | 0.47941188009134256 | 0.36133393814172454 | 0.2397063604185804 | — |
| `seed_271828_rec_7` | 3/5 | 0.0002185219497797617 | 0.0009 | 0.5079375458862916 | 60.769151750000006 | 60.9531745 | 0.07911599999999908 / 0.15768299999999869 | N/Y | 6 | 161803/0, 161803/12, 161803/15, 161803/17, 161803/18, 161803/3, 161803/5, 161803/7, 271828/10, 271828/15, 271828/19, 271828/3, 271828/5, 271828/6, 271828/9, 314159/1, 314159/10, 314159/16, 314159/9 | 1.2761684015297197 | 0.6148723210751281 | 0.6380851462852212 | — |
| `seed_271828_rec_8` | 6/3 | 0.00045 | 0.0006547724796069428 | 0.6278092014414639 | 66.41633275000001 | 66.71847199999999 | 0.11313299999999771 / 0.23333000000000936 | N/Y | 1 | 271828/18 | 0.23887333955357382 | 1.0130686278892636 | 0.5065349399156155 | — |
| `seed_271828_rec_9` | 3/3 | 0.0003547109372725832 | 0.0009 | 0.5369294997159904 | 52.24879825 | 52.5572445 | 0.07337174999999974 / 0.1550745000000049 | N/Y | 3 | 161803/18, 271828/15, 271828/3, 271828/5 | 1.0252899899043342 | 0.01408171485037763 | 0.5126455146380194 | — |
| `seed_314159_rec_0` | 5/6 | 0.0002156899238307862 | 0.00010770493619675102 | 0.5719041815412683 | 74.61342325 | 74.845105 | 0.10082200000000086 / 0.20654225000001247 | N/Y | 6 | 161803/11, 161803/14, 161803/3, 161803/8, 271828/1, 271828/12, 271828/13, 271828/16, 271828/17, 271828/18, 271828/19, 271828/6, 271828/8, 314159/12, 314159/14, 314159/17, 314159/18, 314159/19, 314159/4, 314159/5, 314159/7, 314159/9 | 0.7226407557419693 | 1.5910651829352638 | 0.7955337483206012 | — |
| `seed_314159_rec_1` | 3/3 | 0.00043836528814622386 | 0.0006459840646532157 | 0.5518847844262931 | 52.39449225 | 52.644306500000006 | 0.10309200000000018 / 0.21644750000000101 | N/Y | 2 | 161803/15, 271828/10, 271828/15 | 0.8958762189123726 | 0.024354949004724402 | 0.44793856957177025 | — |
| `seed_314159_rec_10` | 3/3 | 0.0003730086950927352 | 0.0009 | 0.5401583561434012 | 52.33108725 | 52.54840105 | 0.07142250000000061 / 0.1573940000000036 | N/Y | 3 | 161803/15, 161803/18, 271828/10, 271828/15, 271828/5 | 0.9973494662461645 | 0.01988411004391393 | 0.4986752417398704 | — |
| `seed_314159_rec_11` | 6/6 | 0.00045 | 0.0001417474630742632 | 0.6107135016797408 | 79.24220374999999 | 79.7769715 | 0.10983024999998747 / 0.29966800000001115 | N/Y | 3 | 161803/14, 271828/13, 271828/16, 271828/18, 271828/19, 271828/8, 314159/12, 314159/17, 314159/7 | 0.38680893668646055 | 1.91745162249746 | 0.9587269633790096 | — |
| `seed_314159_rec_12` | 6/3 | 0.00043650890201375357 | 0.0007144522744519978 | 0.6517250365478822 | 66.68512100000001 | 66.85309749999999 | 0.09715300000000582 / 0.20530825000000164 | Y/Y | 1 | 271828/18 | 0.0319205153514809 | 1.0320215335009366 | 0.5160112987214928 | — |
| `seed_314159_rec_13` | 5/3 | 9.951633448884923e-05 | 0.00021720818950386837 | 0.5279527217020306 | 61.709461000000005 | 62.01156745 | 0.08581099999999964 / 0.18800625000000082 | N/Y | 6 | 161803/0, 161803/12, 161803/15, 161803/18, 161803/3, 161803/4, 161803/5, 161803/7, 161803/8, 271828/10, 271828/15, 271828/17, 271828/19, 271828/3, 271828/5, 271828/6, 271828/9, 314159/1, 314159/10, 314159/9 | 1.1029694668683507 | 0.6811757865764408 | 0.5514856255068021 | — |
| `seed_314159_rec_14` | 5/5 | 0.0002541354639251281 | 9.510630067968789e-05 | 0.5814514222600689 | 70.5209285 | 70.87765645 | 0.14011400000000407 / 0.3224689999999839 | N/Y | 4 | 161803/11, 161803/14, 161803/3, 271828/1, 271828/12, 271828/13, 271828/16, 271828/18, 271828/19, 271828/6, 271828/8, 314159/12, 314159/17, 314159/18, 314159/4, 314159/7 | 0.6400248478591315 | 1.3024935407112521 | 0.6512477416148204 | — |
| `seed_314159_rec_15` | 3/6 | 0.00010198882389846422 | 0.0003555956199915227 | 0.45750983920686134 | 65.063861 | 65.5891855 | 0.10266200000000225 / 0.2381835000000052 | N/Y | 8 | 161803/0, 161803/11, 161803/12, 161803/15, 161803/17, 161803/18, 161803/3, 161803/4, 161803/5, 161803/7, 161803/8, 161803/9, 271828/1, 271828/10, 271828/12, 271828/15, 271828/17, 271828/19, 271828/3, 271828/5, 271828/6, 271828/7, 271828/9, 314159/1, 314159/10, 314159/13, 314159/16, 314159/6, 314159/9 | 1.7125385416023988 | 0.9177025937705673 | 0.8562705859217671 | — |
| `seed_314159_rec_16` | 3/3 | 0.00016848785957376196 | 0.0009 | 0.5241677080073948 | 52.203284 | 52.4420115 | 0.0841214999999984 / 0.16747474999999667 | N/Y | 2 | 161803/18, 271828/15, 271828/3 | 1.135722631048608 | 0.010872395677250948 | 0.5678618888218173 | — |
| `seed_314159_rec_17` | 6/3 | 0.0003213304064567958 | 0.0007597966799758081 | 0.6316439030814589 | 66.5648545 | 66.84640005 | 0.11709400000000159 / 0.23245674999999721 | N/Y | 1 | 271828/18 | 0.20569020666952142 | 1.02354125349336 | 0.51177124136241 | — |
| `seed_314159_rec_18` | 6/3 | 0.00045 | 1.1000000000000001e-05 | 0.600109443334249 | 66.53805125 | 66.71788005 | 0.08662749999999875 / 0.18285975000000576 | N/Y | 2 | 271828/1, 271828/18, 271828/19, 271828/8 | 0.4785698897469858 | 1.0216512919071072 | 0.5108263960641445 | — |
| `seed_314159_rec_19` | 5/4 | 0.00045 | 0.0009 | 0.5775982731731624 | 65.83171675 | 66.11465899999999 | 0.09055099999999783 / 0.20336799999998334 | N/Y | 5 | 161803/11, 161803/3, 161803/8, 271828/1, 271828/12, 271828/17, 271828/19, 271828/6, 314159/9 | 0.673367613522916 | 0.971845948448945 | 0.4859237968312535 | — |
| `seed_314159_rec_2` | 5/4 | 2.444984531245279e-05 | 0.0001416271774474263 | 0.4556483434840066 | 65.91049724999999 | 66.1223085 | 0.09315699999999794 / 0.18581974999999318 | N/Y | 9 | 161803/0, 161803/11, 161803/12, 161803/15, 161803/17, 161803/18, 161803/2, 161803/3, 161803/4, 161803/5, 161803/7, 161803/8, 161803/9, 271828/1, 271828/10, 271828/11, 271828/12, 271828/15, 271828/17, 271828/19, 271828/3, 271828/5, 271828/6, 271828/7, 271828/9, 314159/1, 314159/10, 314159/13, 314159/15, 314159/16, 314159/19, 314159/6, 314159/8, 314159/9 | 1.7286467726288584 | 0.9774009508738661 | 0.8643247393382909 | — |
| `seed_314159_rec_3` | 6/6 | 0.00042331502060985004 | 0.0005537484468632733 | 0.5970032852425997 | 79.80455425 | 80.1799975 | 0.16350200000000115 / 0.3344942500000059 | N/Y | 4 | 161803/11, 161803/14, 271828/1, 271828/13, 271828/16, 271828/18, 271828/19, 271828/6, 271828/8, 314159/11, 314159/12, 314159/17, 314159/18, 314159/5, 314159/7 | 0.5054486580034404 | 1.9571043080117165 | 0.9785533852823413 | — |
| `seed_314159_rec_4` | 4/6 | 0.00028470234945640816 | 0.00015117546987152063 | 0.5884661300517811 | 70.0168475 | 70.2715625 | 0.090722999999997 / 0.18618574999999282 | N/Y | 3 | 161803/11, 161803/14, 271828/1, 271828/12, 271828/13, 271828/16, 271828/18, 271828/19, 271828/6, 271828/8, 314159/12, 314159/17, 314159/18, 314159/7 | 0.5793239113462487 | 1.2669495777082642 | 0.6334757119908766 | — |
| `seed_314159_rec_5` | 6/4 | 0.0003255265920347484 | 0.000175988605139262 | 0.5987692227100141 | 70.88016575 | 71.137551 | 0.14601900000000256 / 0.2804220000000015 | N/Y | 3 | 161803/14, 271828/1, 271828/13, 271828/16, 271828/18, 271828/19, 271828/6, 271828/8, 314159/12, 314159/17, 314159/18, 314159/7 | 0.49016732893341275 | 1.3278242227311958 | 0.6639130203613738 | — |
| `seed_314159_rec_6` | 4/4 | 6.964454566985233e-05 | 0.0009 | 0.4967856675409136 | 61.562906749999996 | 61.762082 | 0.07900049999999936 / 0.15562924999999694 | N/Y | 7 | 161803/0, 161803/12, 161803/15, 161803/17, 161803/18, 161803/3, 161803/4, 161803/5, 161803/7, 161803/8, 271828/10, 271828/15, 271828/19, 271828/3, 271828/5, 271828/6, 271828/7, 271828/9, 314159/1, 314159/10, 314159/16, 314159/9 | 1.3726698495646565 | 0.6708418941267533 | 0.6863359465382001 | — |
| `seed_314159_rec_7` | 6/3 | 0.00045 | 0.00043930511913497345 | 0.6300270529757901 | 66.58953225 | 66.800954 | 0.1075025000000025 / 0.21441724999999678 | N/Y | 2 | 161803/14, 271828/18, 314159/17 | 0.21968142605160412 | 1.0252813409654076 | 0.5126412929640873 | — |
| `seed_314159_rec_8` | 3/6 | 0.0001987779698535635 | 7.772603220386812e-05 | 0.4565118130625365 | 64.66067225 | 64.82202844999999 | 0.05224650000000253 / 0.10625149999999906 | N/Y | 8 | 161803/0, 161803/11, 161803/12, 161803/15, 161803/17, 161803/18, 161803/3, 161803/4, 161803/5, 161803/7, 161803/8, 271828/1, 271828/10, 271828/12, 271828/15, 271828/17, 271828/19, 271828/3, 271828/5, 271828/6, 271828/7, 271828/9, 314159/1, 314159/10, 314159/13, 314159/16, 314159/6, 314159/9 | 1.7211748417055097 | 0.8892727858364835 | 0.8605887260765687 | — |
| `seed_314159_rec_9` | 4/3 | 0.00045 | 0.0009262898467818477 | 0.5805845050638105 | 57.31344675 | 57.4978615 | 0.07946699999999751 / 0.15983650000000438 | N/Y | 3 | 161803/3, 271828/19, 271828/6 | 0.6475266123309424 | 0.3712022548174423 | 0.3237638155299048 | — |

### 7.3 Final Pareto-front matched remeasurement

The immutable manifest contains every and only expanded global-rank-zero
candidate: `seed_271828_rec_15`, `seed_271828_rec_18`,
`seed_271828_rec_19`, and `seed_271828_rec_3`. Its whole-file SHA-256 is
`d468d5d26f607b115c7c1732966f0ac98664fd232ce83abfa6becc0ce062b7b6`;
its internal manifest hash is
`c49eb5ebc694c991164ab74d9d90b9f700cf56e63efc852b6a6d9e6ff8b5701c`;
candidate-set SHA-256 is
`b19e3be3e99d3bee0c0180e567dfd608dcfdf94a50975df73fc89b7c55075a44`;
and schedule SHA-256 is
`ec53c4e5088201d10eb65270511180a274d1aa4c9746cf6c51097b15d17c2cb9`.

The six-allocation ledger was committed at
`8289d5988f55149857c8e04340c0580d470de11e`. Its whole-file/internal
SHA-256 values are
`356b6dbcdd21fd036b8a8034a82b57e7aaa9a1521f963c13c807e67331e19dcc`
and
`b042435b7f236e8bed9a644e23cdee75d5b0682e5563fb64ab3a031ad988ba4a`.
Every submission has retry count zero, `launch_uncertain=false`, and
`feeds_reselection=false`.

Two earlier launch attempts failed contract checks before SDK submission—one
for a report-path mismatch and one for missing
`SLURM_BASE_RESULTS_DIR`—and created no TAO or SLURM job. The successful
submission used zero-delta resume against the exact zero-submission ledger
SHA-256
`a8fa28d9fe0b5dc6959f8e5e9984fa193c60d27caee29a725d5bd395d9c3acf0`;
this reconciled absence before creating exactly the six jobs below.

| Allocation | Williams row | TAO job | SLURM job | Node | Terminal state |
| --- | ---: | --- | ---: | --- | --- |
| `post_front_allocation_00` | 0 | `59e9a8f6-2c19-4b4e-aec5-0773994e6b09` | 30977076 | `batch-block7-03289` | `Complete / COMPLETED / 0:0` |
| `post_front_allocation_01` | 0 | `364a88a4-40ae-47bb-8d27-6824f66c0359` | 30977080 | `batch-block7-02986` | `Complete / COMPLETED / 0:0` |
| `post_front_allocation_02` | 1 | `1a30085b-3296-46b7-b978-955cf7ce5ceb` | 30977182 | `batch-block7-03411` | `Complete / COMPLETED / 0:0` |
| `post_front_allocation_03` | 2 | `46a40c08-595e-4c71-a031-9f07120adf15` | 30977183 | `batch-block7-01850` | `Complete / COMPLETED / 0:0` |
| `post_front_allocation_04` | 2 | `e78d2027-b184-4c4d-bc30-a362bdfc0f72` | 30977185 | `batch-block7-00255` | `Complete / COMPLETED / 0:0` |
| `post_front_allocation_05` | 3 | `cfd65414-1705-436d-9ada-c3e4892b63ea` | 30977187 | `batch-block7-01833` | `Complete / COMPLETED / 0:0` |

All 24 candidate/allocation cells passed validation. Each cell contains eight
rank files and 4,000 timed samples; the complete analysis binds 198 raw input
artifacts under inventory SHA-256
`65c16462bb1f235150ee6be2c9ead6a145ad2eb95131d00d7c27293e9414aa67`.

The implementation snapshot audited for this report is:

| Committed source or test at `6850d71c2f3dea5f37505dd6831d41cb07a4d255` | SHA-256 |
| --- | --- |
| `phase2_protocol_erratum.v1.json` | `95bba65099027459a50b5e74e43a4ab32c56057e534e70aa7f85bdc9246a7d13` |
| `post_front_matched_manifest_generator.py` | `e535ad59beb11b02c49527f6490719f29b1bd680ae990a7b8beb739f8c7899da` |
| `post_front_matched_launcher.py` | `e64108dff03da40a91642217dcac453e29bdad7046cb58ad5c05e7e65fa9a887` |
| `post_front_matched_block_runner.py` | `8a82ea7d9e0a06c617c94ae83c3cf5b333ed887d051a9fe816c3cc138c37aae6` |
| `post_front_matched_aggregator.py` | `ff652784fdd63b73020eb32d0b22b6fe8bca3385413b6060aa8ba1d5bdc604db` |
| `test_phase2_protocol_erratum.py` | `8bbd4d6e64807239c479fd30fb9067dceaac83e7b35f05b83bbf589aa4ee9acc` |
| `test_post_front_protocol_binding.py` | `ff1654ed72c745cbb59c738f145e36e4787f2ac2f313d5bfba7cb2edc71a9ef5` |
| `test_post_front_protocol_analysis.py` | `49703256613f781f9be4f8cfe0c26a5cb49874e947e4692180a411ab6315f25e` |
| `test_post_front_matched_tools.py` | `0c31573754d7b2c2c70feea5f75141b73e52f40b6c20bc7be36e51721cc90ff9` |
| `test_post_front_matched_launcher_recovery.py` | `8f020f7a36222ac240830f9b3bb61984e80e4ea65c8c344de498062b4daecae4` |
| `test_post_front_complete_invalid_recovery.py` | `6d1082d30d23db286953c310e7ba7dacdb135b3221be88c077f26cbb5e24aebe` |

These tracked identities were reconstructed exactly during generation,
launch, and aggregation.

The protocol erratum's issuance state is itself part of the authority. It
records 15 successful candidates (five per seed), no completed union
selection, no known or used final Pareto front, and zero post-front manifests,
allocations, measurements, or pairwise comparisons at issuance. That temporal
record proves the rule preceded the later outcomes and was not fitted to a
winner. Both launcher and aggregator required the exact erratum path and
whole-file hash.

#### 7.3.1 Manifest authority, candidate derivation, and selector isolation

The final prelaunch review found one blocker in the earlier hardening
snapshot. A manifest could be edited in a runtime or latency-protocol field
and then have `manifest_sha256` recomputed. Its whole-file and internal hashes
would be self-consistent, and fragmentary schema checks did not make that
self-hash an authority for all launch-affecting semantics.

The launcher now reconstructs the complete canonical manifest through
`post_front_matched_manifest_generator.build_manifest` from the exact pinned
protocol erratum, expanded manifest, three seed archives, combined selection,
candidate-table JSON and CSV, integrity audit, runtime contract, selector
stack, and post-front tool sources.
`require_exact_reconstructed_manifest` then requires whole-object equality,
not merely equality of selected fields or digests. This source validation
runs before config generation and every dry-run, fresh launch,
incomplete-submission resume, and allocation replacement. The read-only
aggregator calls the same validation before job inspection or result
aggregation. Thus a self-rehashed drift in SQSH, SDK, SLURM, hardware, or
latency-protocol settings fails before it can affect execution or analysis.

Manifest generation fails closed until the expanded run has exactly 60
terminal records plus the final combined selection, candidate-table JSON and
CSV, and integrity audit. It then:

1. validates exactly one complete `seed_archive.v1.json` for each frozen
   search seed, in manifest order `314159`, `271828`, `161803`;
2. requires exactly 20 terminal records per seed, exact candidate IDs
   `seed_<seed>_rec_0` through `seed_<seed>_rec_19`, no manual injection,
   exact expanded-manifest bindings, and valid whole-file and internal archive
   hashes;
3. canonicalizes all 60 full records by ascending UTF-8 candidate ID and
   records per-seed full-record hashes, the union candidate-ID hash, and the
   canonical full-record-union hash;
4. reconstructs every candidate-table JSON row from its authoritative seed
   record plus the combined-selection audit and requires semantic equality;
5. reconstructs the complete candidate-table CSV byte for byte and requires
   exact equality, while separately pinning the combined-selection JSON,
   candidate-table JSON, candidate-table CSV, and integrity-audit hashes;
6. imports the manifest-pinned production objective parser and
   `tao_automl.selection.analyze_archive` implementation;
7. independently replays that frozen archive under candidate-table, reverse,
   and candidate-ID order;
8. requires the replayed analysis, every candidate audit, and the global
   rank-zero front to exactly match the combined-selection artifacts; and
9. includes every and only global-rank-zero candidate, in ascending UTF-8
   candidate-ID order, with its exact checkpoint and complete resolved model
   mapping.

The manifest's `expanded_archive_snapshot` therefore binds the exact 3 × 20
record authority, not merely a table row count or the winning candidates. The
production selector is invoked during frozen-archive source validation. It is
not invoked on post-front measurements. The replay result is used only to
prove candidate-set integrity before those measurements are loaded. The
original accuracy, latency, and multi-objective selection snapshot and its
selection-time metrics are copied unchanged into the post-front manifest;
remeasurement cannot select, reselect, replace a selection-time objective, or
override a winner.

An independent read-only re-audit after the exact-reconstruction change found
no remaining blocker. The later live execution exercised those gates:
generation, dry-run, six submissions, scheduler reconciliation, result
loading, semantic validation, and immutable aggregation all passed.

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
median differences and six matched p95 differences. It reports two explicitly
separate policy branches for both endpoints.

The original preregistered branch uses the deterministic 10,000-resample
paired percentile bootstrap. Under that original rule, it classifies the
first candidate as stably faster when the complete paired 95% interval is
below `-0.73553775 ms`, the second as stably faster when the complete interval
is above `+0.73553775 ms`, and the pair as practically equivalent when the
complete interval is within the tolerance band; otherwise it reports
uncertainty. That original classification is preserved and emitted as
`original_preregistered_bootstrap_classification`. It was not originally
defined as descriptive-only.

The effective pre-post-front erratum branch makes the bootstrap descriptive
only for effective claims. An effective directional claim requires both:

1. a one-sided exact paired sign-flip permutation test after shifting by the
   relevant `±0.73553775 ms` practical-tolerance boundary, with
   \(p\le0.05\); and
2. all six paired differences strictly beyond that same boundary in the
   claimed direction.

With six allocations the exact randomization enumerates all \(2^6=64\) sign
assignments. Failure to establish an effective direction is not evidence of
equivalence. Median and p95 are classified independently. Claims are pairwise
only. There is no multiplicity adjustment, so unadjusted pairwise evidence
never establishes a simultaneous or stable total order. Descriptive sorting
and bootstrap intervals are not promoted into effective directional claims.

The preregistered requirement was to remeasure every final rank-zero candidate
under this frozen contract, preserve the original selection-time measurements
and algorithm-selected winner, and use remeasurement only as verdict
stability evidence. The six completed allocations fulfilled that requirement.

#### 7.3.6 Distinctness and measurement roles

Two different questions must remain separate:

1. **Selector-geometric distinctness.** The frozen production selector's
   `distinct_compromise` flag compares the multi-objective winner with the
   accuracy and unconstrained-latency extremes of the
   multi-objective-eligible Pareto population. This is a selector output and
   remains unchanged.
2. **Actual-mode-winner distinctness.** The hypothesis comparison asks whether
   the same frozen multi-objective winner differs from both the actual
   accuracy-mode winner and the actual latency-mode winner produced under 98%
   retained accuracy. It separately reports identity, accuracy position,
   selection-time latency position, and matched-latency evidence against
   those two actual winners.

Selection-time mAP50 and latency are the only objective values used to build
the archive, construct Pareto ranks, normalize regrets, score the compromise,
and select all three winners. The matched six-allocation median and p95 values
are validation-only measurements. They are reported by mode with deltas from
the accuracy and constrained-latency winners and with separate practical- and
statistical-tolerance evidence, but they never replace selection-time metrics.
The post-front aggregator records
`selector_invoked_on_matched_measurements=false`,
`selection_time_objectives_replaced=false`, and
`feeds_reselection=false`.

#### 7.3.7 Matched measurements and relative-latency result

The interval below is the within-allocation device-round cluster-bootstrap
95% interval for the median. Every row is valid and contains 4,000 samples.

| Allocation | Candidate | Position | Median ms | p95 ms | Median 95% CI ms | MAD / IQR ms | Robust CV |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 00 | `seed_271828_rec_15` | 0 | 52.398748 | 52.57318095 | [52.34688875, 52.44410375] | 0.07642150 / 0.15579575 | 0.00216231 |
| 00 | `seed_271828_rec_18` | 1 | 66.53260975 | 66.80764350 | [66.49996125, 66.57365406] | 0.10847800 / 0.23815850 | 0.00241730 |
| 00 | `seed_271828_rec_3` | 2 | 52.374702 | 52.63235250 | [52.27414325, 52.39863050] | 0.11397050 / 0.23439000 | 0.00322623 |
| 00 | `seed_271828_rec_19` | 3 | 57.13886025 | 57.39506450 | [57.10449150, 57.19241900] | 0.07527750 / 0.15839400 | 0.00195325 |
| 01 | `seed_271828_rec_15` | 0 | 52.46612850 | 52.73343695 | [52.43804075, 52.53951100] | 0.09551650 / 0.18515150 | 0.00269913 |
| 01 | `seed_271828_rec_18` | 1 | 66.63606700 | 67.14680955 | [66.59261575, 66.65021725] | 0.09074950 / 0.18458050 | 0.00201910 |
| 01 | `seed_271828_rec_3` | 2 | 52.44349400 | 52.63215605 | [52.40000600, 52.47539675] | 0.06406700 / 0.12683700 | 0.00181120 |
| 01 | `seed_271828_rec_19` | 3 | 57.22198775 | 58.65574450 | [57.16477113, 57.28843350] | 0.10334175 / 0.20693875 | 0.00267755 |
| 02 | `seed_271828_rec_18` | 0 | 66.46072575 | 66.71065150 | [66.40532725, 66.59162100] | 0.13723475 / 0.26056925 | 0.00306142 |
| 02 | `seed_271828_rec_19` | 1 | 57.07905325 | 57.34709350 | [56.92982050, 57.10264100] | 0.16078575 / 0.31527025 | 0.00417633 |
| 02 | `seed_271828_rec_15` | 2 | 52.34634800 | 52.56795050 | [52.22288100, 52.38583875] | 0.11597950 / 0.24038475 | 0.00328487 |
| 02 | `seed_271828_rec_3` | 3 | 52.32037475 | 52.53347750 | [52.29842500, 52.35535850] | 0.07177600 / 0.14097725 | 0.00203391 |
| 03 | `seed_271828_rec_19` | 0 | 57.10053750 | 57.28967400 | [57.06101675, 57.14033850] | 0.07437850 / 0.15045875 | 0.00193122 |
| 03 | `seed_271828_rec_3` | 1 | 52.25068725 | 52.50351545 | [52.23566000, 52.30875825] | 0.07910375 / 0.18165575 | 0.00224455 |
| 03 | `seed_271828_rec_18` | 2 | 66.42088725 | 66.73120705 | [66.39505450, 66.53277950] | 0.08893600 / 0.20882325 | 0.00198517 |
| 03 | `seed_271828_rec_15` | 3 | 52.23401025 | 52.74906645 | [52.20014225, 52.26674050] | 0.07672650 / 0.16012200 | 0.00217779 |
| 04 | `seed_271828_rec_19` | 0 | 56.95001200 | 57.22027455 | [56.92316175, 57.01612625] | 0.08618200 / 0.18131975 | 0.00224361 |
| 04 | `seed_271828_rec_3` | 1 | 52.24231750 | 52.50463155 | [52.22493450, 52.25404000] | 0.07185200 / 0.14762400 | 0.00203911 |
| 04 | `seed_271828_rec_18` | 2 | 66.59766475 | 66.78319650 | [66.49723475, 66.63769025] | 0.10998300 / 0.24491850 | 0.00244845 |
| 04 | `seed_271828_rec_15` | 3 | 52.24965800 | 52.46097795 | [52.21901700, 52.32178400] | 0.07157200 / 0.14848150 | 0.00203088 |
| 05 | `seed_271828_rec_3` | 0 | 52.25261175 | 52.38091905 | [52.14426925, 52.27824450] | 0.07334625 / 0.19079550 | 0.00208110 |
| 05 | `seed_271828_rec_15` | 1 | 52.15146450 | 52.35178095 | [52.12938925, 52.18956325] | 0.07556150 / 0.15377875 | 0.00214812 |
| 05 | `seed_271828_rec_19` | 2 | 56.94584425 | 57.16935250 | [56.91919125, 57.01321350] | 0.07713650 / 0.15592125 | 0.00200827 |
| 05 | `seed_271828_rec_18` | 3 | 66.42781300 | 66.74751105 | [66.33758550, 66.53362050] | 0.14532350 / 0.29096150 | 0.00324347 |

Between-allocation summaries use the median of six allocation statistics:

| Candidate | Stable median ms | Median range ms | Median sample SD ms | Stable p95 ms | p95 range ms | p95 sample SD ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `seed_271828_rec_15` | 52.29800300 | 0.31466400 | 0.11669754 | 52.570565725 | 0.39728550 | 0.15378642 |
| `seed_271828_rec_18` | 66.49666775 | 0.21517975 | 0.09074304 | 66.765353775 | 0.43615805 | 0.16333976 |
| `seed_271828_rec_19` | 57.089795375 | 0.27614350 | 0.10828114 | 57.318383750 | 1.48639200 | 0.56585421 |
| `seed_271828_rec_3` | 52.28649325 | 0.20117650 | 0.08174284 | 52.519054525 | 0.25143345 | 0.09432478 |

For every pair, delta is first minus second; a negative delta means the first
is faster. Bootstrap intervals are preserved descriptive evidence. Effective
direction additionally requires the one-sided exact shifted sign-flip
\(p\le0.05\) and all six allocation differences beyond the
`0.73553775 ms` boundary.

| First | Second | Median paired delta ms | Descriptive 95% CI ms | Effective median result | p95 paired delta ms | Descriptive 95% CI ms | Effective p95 result |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| `seed_271828_rec_15` | `seed_271828_rec_18` | -14.17840775 | [-14.312177625, -14.124119750] | First stably faster; p=0.015625; all 6 | -14.27834055 | [-14.404551350, -14.062420800] | First stably faster |
| `seed_271828_rec_15` | `seed_271828_rec_19` | -4.74798575 | [-4.830453500, -4.716529625] | First stably faster; p=0.015625; all 6 | -4.798357275 | [-5.372095550, -4.649952075] | First stably faster |
| `seed_271828_rec_15` | `seed_271828_rec_3` | +0.01498750 | [-0.058912125, +0.025009625] | No stable direction; descriptive practical equivalence | +0.00266745 | [-0.051412575, +0.173415950] | No stable direction; descriptive practical equivalence |
| `seed_271828_rec_18` | `seed_271828_rec_19` | +9.403914375 | [+9.351011125, +9.564810750] | Second stably faster; p=0.015625; all 6 | +9.427056025 | [+8.927311525, +9.570540250] | Second stably faster |
| `seed_271828_rec_18` | `seed_271828_rec_3` | +14.172700625 | [+14.149129375, +14.273960125] | Second stably faster; p=0.015625; all 6 | +14.253128275 | [+14.176232500, +14.440622750] | Second stably faster |
| `seed_271828_rec_19` | `seed_271828_rec_3` | +4.761418375 | [+4.700463500, +4.814172000] | Second stably faster; p=0.015625; all 6 | +4.787296000 | [+4.739177500, +5.418602225] | Second stably faster |

The effective evidence establishes the pairwise geometric ordering
`rec_3/rec_15` (no direction between them) faster than `rec_19`, faster than
`rec_18`. No simultaneous total order is claimed because the protocol has no
multiplicity adjustment. This validates `rec_19` as a stable latency
intermediate on the global front, but not as a point between the actual
accuracy and 98%-constrained latency winners: those two actual winners are the
same `rec_18` candidate.

### 7.4 Final combined selection and integrity audit

| Artifact or binding | SHA-256 |
| --- | --- |
| Expanded combined selection | `78ab9d2fa83cc3abe9057d137c0b88f120158b6ad77268482d2c18f5a1533af1` |
| Expanded candidate table JSON | `5ba323d05d9ec8e3703e636f8b5e2975cc620eeec10df75ec6e792318dc2df03` |
| Expanded candidate table CSV | `0b313942968805879ac0f3bfb386dd45156d4be160c070aedf83bac579df6e5a` |
| Expanded integrity audit | `a11eeeaf77bd2f289c6363133882bb78c6889205d4cb9be5f0dacf79a1bea159` |
| Expanded completion record | `74fcf1392d8bad4d9e0681544d2c06b4d9a792fd3b3cbb0dc1b55572b2adac7f` |
| Canonical 60-record union | `a55964fa0c5762c1a8df45dd6b8a55047a68691cdefc5a8764c4426e68c1d365` |
| Candidate-table semantic projection | `9e00262945d521468b973745a67d8f9c7e3c85c4c7b66712454d2b4f36922551` |
| Post-front manifest whole/internal | `d468d5d26f607b115c7c1732966f0ac98664fd232ce83abfa6becc0ce062b7b6` / `c49eb5ebc694c991164ab74d9d90b9f700cf56e63efc852b6a6d9e6ff8b5701c` |
| Post-front dry-run | `75bdc60d015e1359afaf74603d0235cd8b245ae6a380de09082bf24d198b4d8d` |
| Post-front launch contract | `6cdf20d056ea361055df3d8a38047389f736aa9911fecdfe32361e0deb978465` |
| Six-job ledger whole/internal | `356b6dbcdd21fd036b8a8034a82b57e7aaa9a1521f963c13c807e67331e19dcc` / `b042435b7f236e8bed9a644e23cdee75d5b0682e5563fb64ab3a031ad988ba4a` |
| Post-front analysis whole/internal | `150d66fd1648c458807bdce9871313b5b17a7a33c63564f34b86156e392094b9` / `d82ea45e6622690e7208d54911c8b57213486069691039443c73f1951dec8299` |
| Raw matched-input inventory | `65c16462bb1f235150ee6be2c9ead6a145ad2eb95131d00d7c27293e9414aa67` |

The selection source is `src/tao_automl/selection.py`, SHA-256
`7e787a18bca05464e0043367aee4f2c8cff3d93aef7f9e92aaf88c47d255a532`.
The order-invariance audit used archive, reverse, and candidate-ID order and
produced signature
`75fa7afe25d0fda3dd50b96c405434ebf52a57727cd3f04ac3bba5b006a5f11a`.
The final evidence records:

```text
manual_candidate_injection_used = false
manual_override_used = false
candidate_reordering_used = false
algorithm_selected_candidate_overridden = false
selector_invoked_on_postfront_measurements = false
selection_time_objectives_replaced = false
measurements_feed_selection = false
measurements_feed_reselection = false
```

All source, dataset, PTM, SQSH, checkpoint, input-digest, runtime, hardware,
scheduler-hostname, complete-block, and rank-file contracts passed. The
zero-submission resume reconciled that no prior job existed; consequently the
final ledger contains no adopted-job recovery event or supersession. There
were no complete-invalid allocation replacements.

## 8. Final mode comparison

| Mode | Candidate | Accuracy | Stable median latency | Eligibility rule | Pareto status | Selection reason |
| --- | --- | ---: | ---: | --- | --- | --- |
| Accuracy | `seed_271828_rec_18` | 0.6554138278683255 | 66.49666775 ms | All 60 valid candidates | Global rank zero, nondominated | Highest valid mAP50; no accuracy tie required |
| Latency | `seed_271828_rec_18` | 0.6554138278683255 | 66.49666775 ms | mAP50 ≥ 0.6423055513109589 (98% of accuracy winner); 4 candidates | Global rank zero, nondominated | Raw minimum selection-time latency in the feasible cohort; all four are within the frozen latency tie rule, then highest accuracy chooses `rec_18` |
| Multi-objective | `seed_271828_rec_19` | 0.6175134981289873 | 57.089795375 ms | All 60 valid candidates; independent minimum-accuracy floor unset | Global rank zero, nondominated | Minimum front-normalized augmented-Chebyshev score, 0.1797199345585883; selector-geometric distinct compromise |

Candidate identity, accuracy, Pareto status, and selection reason come only
from the immutable selection-time snapshot. Stable median and p95 come only
from matched validation and were never input to a second selection.

| Mode | Accuracy delta from A | Selection-time median ms | Selection-time delta from A / L | Matched p95 ms | Stable median delta from A / L | Exceeds uncertainty? |
| --- | ---: | ---: | --- | ---: | --- | --- |
| Accuracy | 0 | 66.23099475000001 | 0 / 0 ms | 66.765353775 | 0 / 0 ms | Reference |
| Latency | 0 | 66.23099475000001 | 0 / 0 ms | 66.765353775 | 0 / 0 ms | Same candidate as accuracy |
| Multi-objective | -0.03790032973933821 | 57.146624 | -9.084370750000005 / -9.084370750000005 ms | 57.318383750 | -9.406872374999999 / -9.406872374999999 ms | Yes versus `rec_18`: paired median CI for `rec_18 - rec_19` is [+9.351011125, +9.564810750] ms, exact p=0.015625, all six beyond tolerance |

The multi-objective stable aggregate p95 delta from either actual winner is
`-9.446970024999999 ms`; the allocation-paired p95 delta for
`rec_18 - rec_19` is `+9.427056025 ms`, with descriptive CI
`[+8.927311525, +9.570540250]` and the same effective directional result.

`rec_19` is distinct from `rec_18` by candidate identity, accuracy tolerance,
and matched latency tolerance. It is strictly between the global front's
unconstrained latency extreme (`rec_3`) and accuracy extreme (`rec_18`):
`rec_3` is stably faster than `rec_19`, and `rec_19` is stably faster than
`rec_18`. It is not strictly between the two actual mode extremes, because
the actual accuracy and 98%-constrained latency modes both selected
`rec_18`.

## 9. Hypothesis verdict

| Criterion | Result | Evidence |
| --- | --- | --- |
| Accuracy mode selects the highest-accuracy valid candidate | Supported | `seed_271828_rec_18`, mAP50 `0.6554138278683255`, is the maximum across all 60 valid rows. |
| Latency mode selects the fastest candidate satisfying 98% retention | Supported under the frozen tolerance rule | Threshold `0.6423055513109589` admits four candidates. `rec_18` has the raw minimum selection-time median (`66.23099475 ms`); the four-point latency-tied cohort is resolved by higher accuracy, which also selects `rec_18`. |
| Multi-objective winner is Pareto-nondominated | Supported | `seed_271828_rec_19` has global and eligible Pareto rank zero and empty `dominated_by`. |
| Winner is produced entirely by the documented algorithm | Supported | No manual injection, reorder, promotion, override, reselection, or matched-objective replacement occurred; three archive orderings reproduce the same result. |
| A distinct intermediate point exists in the eligible global front | Supported geometrically | `rec_19` has regret pair `(0.3279659791, 0.3594391817)`, score `0.1797199346`, and differs from both global-front extremes. |
| Multi-objective selection differs from both actual mode winners | Supported by identity, but actual extremes are degenerate | Accuracy and constrained latency both select `rec_18`; multi-objective selects `rec_19`. There are not two distinct actual extreme candidates. |
| Multi-objective accuracy lies between the actual extreme winners | Not supported | Both actual winners have mAP50 `0.6554138278683255`; `rec_19` has `0.6175134981289873`. |
| Multi-objective latency lies between the actual extreme winners | Not supported | Both actual winners have stable median `66.49666775 ms`; `rec_19` has `57.089795375 ms`. |
| Relative latency position is stable across matched allocations | Supported for global Pareto geometry | `rec_3` is stably faster than `rec_19`, and `rec_19` is stably faster than `rec_18`; each effective test has p=`0.015625` and all six differences beyond tolerance. |
| Stable across repeated runs or seeds | Partially established | Relative latency is stable across six independent matched allocations. Candidate generation used three deterministic search seeds. The final winner identity was not independently retrained under multiple training seeds, so full training-seed winner stability is not claimed. |

### Final classification: partially supported

The phase-two implementation supports the algorithmic claims: accuracy is
maximized, latency is constrained independently, multi-objective selection is
normalized and Pareto-safe, `rec_19` is a genuine global-front geometric
compromise, and its matched latency position is stable. The full three-mode
hypothesis nevertheless fails its strict “between two actual extremes”
criteria because accuracy and 98%-constrained latency collapse to the same
`rec_18` candidate. The result is not inconclusive: the supported DINO search
space did produce a statistically stable intermediate global-front point.
The limitation is the degenerate actual mode endpoints, not absence of a
stable Pareto intermediate.

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
| `~/tao-automl` phase-two protocol erratum | same | `ba2cf95211ecddc9cb38dfe51d189357b05dc8e2` |
| `~/tao-automl` post-front protocol binding cutoff | same | `6850d71c2f3dea5f37505dd6831d41cb07a4d255` |
| `~/tao-automl` complete expanded archive and selection | same | `a6fc0bbd7947cf58b95f1c037b0513092b31a2f9` |
| `~/tao-automl` immutable post-front manifest | same | `a3099803f4a4fa7494c9564c0b6806576a203d7b` |
| `~/tao-automl` six-allocation submission ledger | same | `8289d5988f55149857c8e04340c0580d470de11e` |
| `~/tao-automl` matched post-front analysis | same | `2307d86a9a2cf6c0883a977a6dfcd8e1f885ea77` |
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

That launch is complete. Do not repeat the fresh-launch form: the exact
60-record archive is already sealed. Verify it read-only with:

```bash
archive_root=experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2

sha256sum \
  "${archive_root}"/seed_*/seed_archive.v1.json \
  "${archive_root}/expanded_candidate_table.json" \
  "${archive_root}/expanded_candidate_table.csv" \
  "${archive_root}/expanded_combined_selection.json" \
  "${archive_root}/expanded_integrity_audit.json" \
  "${archive_root}/expanded_completion.json"

jq '{
  candidate_count,
  successful_count,
  manual_candidate_injection_used,
  failed_count: ([.rows[] | select(.status != "success")] | length)
}' "${archive_root}/expanded_candidate_table.json"

jq '{
  selections,
  selection_authority,
  normalization_bounds: .algorithm.normalization_bounds,
  accuracy_threshold: .algorithm.accuracy_threshold
}' "${archive_root}/expanded_combined_selection.json"
```

Expected counts are 60/60/false/0; expected winners are `rec_18`,
`rec_18`, and `rec_19`. The excluded v1 runtime must never be passed to any
v2 reproduction or validation command.

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

### 10.8 Post-front manifest, launch, and aggregation

The immutable manifest was originally generated from the sealed archive with
the following inputs. The tracked output is create-once and the generator
correctly refuses to overwrite it.

```bash
cd ~/tao-automl
py=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python
d=experiments/dino_moo_phase2_20260728
expanded_runtime="${d}/runtime/expanded_search_v2"
post_runtime="${d}/runtime/post_front_matched"

"${py}" "${d}/post_front_matched_manifest_generator.py" \
  --expanded-manifest "${d}/expanded_search_manifest.v2.json" \
  --expanded-manifest-sha256 \
  9ac29e1aa07167a040d217fdab2d3cfdea0baad690dc95a70f2fe6715908793a \
  --combined-selection \
  "${expanded_runtime}/expanded_combined_selection.json" \
  --combined-selection-sha256 \
  78ab9d2fa83cc3abe9057d137c0b88f120158b6ad77268482d2c18f5a1533af1 \
  --candidate-table \
  "${expanded_runtime}/expanded_candidate_table.json" \
  --candidate-table-sha256 \
  5ba323d05d9ec8e3703e636f8b5e2975cc620eeec10df75ec6e792318dc2df03 \
  --integrity-audit \
  "${expanded_runtime}/expanded_integrity_audit.json" \
  --integrity-audit-sha256 \
  a11eeeaf77bd2f289c6363133882bb78c6889205d4cb9be5f0dacf79a1bea159 \
  --protocol-erratum "${d}/phase2_protocol_erratum.v1.json" \
  --protocol-erratum-sha256 \
  95bba65099027459a50b5e74e43a4ab32c56057e534e70aa7f85bdc9246a7d13 \
  --output "${d}/post_front_matched_manifest.v1.json"
```

For read-only regeneration, use the same command with
`--output /tmp/post_front_matched_manifest.reproduced.json`, then compare the
reconstructed object and internal hash to the tracked manifest. A new output
path is mandatory.

The exact dry-run form was:

```bash
manifest="${d}/post_front_matched_manifest.v1.json"
manifest_sha=d468d5d26f607b115c7c1732966f0ac98664fd232ce83abfa6becc0ce062b7b6
erratum="${d}/phase2_protocol_erratum.v1.json"
erratum_sha=95bba65099027459a50b5e74e43a4ab32c56057e534e70aa7f85bdc9246a7d13

PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
"${py}" "${d}/post_front_matched_launcher.py" \
  --dry-run \
  --manifest "${manifest}" \
  --manifest-file-sha256 "${manifest_sha}" \
  --protocol-erratum "${erratum}" \
  --protocol-erratum-sha256 "${erratum_sha}" \
  --runtime-dir "${post_runtime}" \
  --report "${post_runtime}/dry_run.json" \
  --verify-remote
```

Two initial `--launch` attempts failed before submission: the first used a
non-authoritative report path, and the second lacked the required
`SLURM_BASE_RESULTS_DIR`. Neither attempt created a TAO or SLURM job. The
second attempt left a durable zero-submission incomplete ledger whose
whole-file SHA-256 was
`a8fa28d9fe0b5dc6959f8e5e9984fa193c60d27caee29a725d5bd395d9c3acf0`.
The actual successful six-job submission used the launcher's exact
zero-delta reconciliation path:

```bash
export SLURM_BASE_RESULTS_DIR=/lustre/fsw/portfolios/edgeai/users/rarunachalam

PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
"${py}" "${d}/post_front_matched_launcher.py" \
  --resume-incomplete-submission \
  --manifest "${manifest}" \
  --manifest-file-sha256 "${manifest_sha}" \
  --protocol-erratum "${erratum}" \
  --protocol-erratum-sha256 "${erratum_sha}" \
  --runtime-dir "${post_runtime}" \
  --report "${post_runtime}/dry_run.json" \
  --submission-ledger-sha256 \
  a8fa28d9fe0b5dc6959f8e5e9984fa193c60d27caee29a725d5bd395d9c3acf0 \
  --verify-remote \
  --acknowledgement \
  USER_AUTHORIZED_DINO_POST_FRONT_6X8GPU_VALIDATION_20260728
```

Do not repeat the launch against the existing runtime. The committed
six-allocation ledger is the durable submission authority.

After all six jobs reached `Complete / COMPLETED / 0:0`, the read-only
aggregation form was:

```bash
export SLURM_BASE_RESULTS_DIR=/lustre/fsw/portfolios/edgeai/users/rarunachalam

"${py}" "${d}/post_front_matched_aggregator.py" \
  --manifest "${manifest}" \
  --manifest-file-sha256 "${manifest_sha}" \
  --submission-ledger "${post_runtime}/block_submissions.json" \
  --submission-ledger-sha256 \
  356b6dbcdd21fd036b8a8034a82b57e7aaa9a1521f963c13c807e67331e19dcc \
  --sdk-state "${post_runtime}/slurm_state.json" \
  --output "${post_runtime}/post_front_matched_analysis.json" \
  --secrets-env ~/.tao/config.env
```

Read-only reaggregation requires the same preserved local SDK database (the
untracked `slurm_state.db` reached through the `.json` SDK-state argument)
and the 198 remote Lustre result artifacts. It is not reproducible from a
clean Git clone alone, and the private SDK database must not be added to Git.
In that preserved environment, change `--output` to a new path such as
`/tmp/post_front_matched_analysis.reproduced.json`; never overwrite the
tracked analysis. A reproduced analysis has a fresh `created_at_utc`, so
compare contract, source, candidate, measurement, and analysis fields after
normalizing that timestamp rather than requiring whole-file equality.

The tracked analysis is the portable evidence artifact: it embeds the six
terminal TAO/SLURM identities and states plus the complete 198-artifact
path/hash inventory. Its whole-file/internal hashes are
`150d66fd1648c458807bdce9871313b5b17a7a33c63564f34b86156e392094b9`
and
`d82ea45e6622690e7208d54911c8b57213486069691039443c73f1951dec8299`.

### 10.9 Final tests and integrity checks

```bash
cd ~/tao-automl
py=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python

"${py}" -m pytest -q experiments/dino_moo_phase2_20260728
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  "${py}" -m pytest -q tests

"${py}" - <<'PY'
import pathlib
for name in (
    "post_front_matched_manifest_generator.py",
    "post_front_matched_launcher.py",
    "post_front_matched_block_runner.py",
    "post_front_matched_aggregator.py",
):
    path = pathlib.Path("experiments/dino_moo_phase2_20260728") / name
    compile(path.read_text(), str(path), "exec")
print("4/4 post-front tools compiled")
PY

git diff --check

sha256sum \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2/expanded_combined_selection.json \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2/expanded_candidate_table.json \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2/expanded_candidate_table.csv \
  experiments/dino_moo_phase2_20260728/runtime/expanded_search_v2/expanded_integrity_audit.json \
  experiments/dino_moo_phase2_20260728/post_front_matched_manifest.v1.json \
  experiments/dino_moo_phase2_20260728/runtime/post_front_matched/block_submissions.json \
  experiments/dino_moo_phase2_20260728/runtime/post_front_matched/post_front_matched_analysis.json
```

The sealed validation result is `259 passed` for the phase-two suite,
`387 passed, 1 skipped` for the production core, four of four post-front tools
compiled, and `git diff --check` clean.
