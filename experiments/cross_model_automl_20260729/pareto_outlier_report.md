# Cross-model Pareto outlier investigation

Scope: accuracy, latency, and multi-objective AutoML selection only. This
report does not include hardware-aware AutoML.

## Evidence and method

The read-only source manifest is
`pareto_validation_sources.v1.json`. The machine-derived replay is produced by
`pareto_outlier_audit.py`; it verifies every source digest before loading an
artifact, invokes the production selector on the unchanged per-mode archive,
repeats the replay under ten shuffled candidate orders, and records the full
rank-zero MOO front and selector geometry.

The terminal replay artifact, including Mask Grounding DINO v11, is:

```text
/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/pareto_outlier_validation/mgdino_v11/matrix.json
sha256 7625986066f6edd08d53e3a8bcd6bce01b8b95571d448b9b56655da549e598f8
```

The audit also constructs a read-only union of the three independently
acquired mode archives. That union is diagnostic acquisition-coverage
evidence, not an input that was available to any production selector. It does
not replace or reselect a frozen winner.

## Candidate-universe product contract

The implemented and documented product contract is **Design B: three
independent objective-aware searches**. `campaign_manifest_contract.md`, the
campaign manifests, and the production runner require a unique job ID and
observation namespace for each mode, disable cross-mode observation sharing,
and start every mode from an empty observation history. Accuracy uses expected
improvement, constrained latency uses constrained expected improvement, and
MOO uses deterministic ParEGO acquisition before each mode applies its own
terminal selector to its own terminal archive.

Consequently, accuracy, latency, and Pareto correctness are mode-local
invariants. Ordering winners from three different finite archives is useful
coverage evidence, but it is not a selector invariant. A candidate discovered
by the latency acquisition cannot retroactively make the accuracy selector
incorrect. The diagnostic union remains read-only and cannot become a
production selection archive.

## Machine-derived matrix

| Model | Dataset | Accuracy winner | Latency winner | MOO winner | Accuracy invariant | Latency invariant | Pareto invariant | Middle-ground invariant | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DINO | TAO OD synthetic full DINO COCO | `rec_14` | `rec_13` | `rec_5` | true | true | true | true | `PASS_EXPECTED_COMPROMISE` |
| Deformable DETR | TAO OD synthetic full DINO COCO | `rec_19` | `rec_19` | `rec_5` | true | true | true | false | `PASS_EXPECTED_COMPROMISE` |
| RT-DETR | TAO OD synthetic full DINO COCO | `rec_6` | `rec_17` | `rec_6` | true | true | true | false | `PASS_ENDPOINT_COLLAPSE` |
| Grounding DINO | TAO OD synthetic full DINO COCO (ODVG conversion) | `rec_13` | `rec_18` | `rec_3` | true | true | true | false | `PASS_EXPECTED_COMPROMISE` |
| SegFormer | VOC2012 segmentation | `rec_23` | `rec_12` | `rec_25` | true | true | true | false | `PASS_EXPECTED_COMPROMISE` |
| Mask2Former | COCO2017 instance segmentation | `rec_5` | `rec_2` | `rec_4` | true | true | true | false | `PASS_EXPECTED_COMPROMISE` |
| OneFormer | COCO2017 panoptic segmentation | `rec_6` | `rec_17` | `rec_16` | true | true | true | true | `PASS_EXPECTED_COMPROMISE` |
| Mask Grounding DINO | COCO2017 category-prompted instance segmentation | `rec_16` | `rec_14` | `rec_19` | true | true | true | true | `PASS_EXPECTED_COMPROMISE` |

Every persisted selector result matches the current production replay and is
invariant to candidate order. Every listed MOO winner is rank zero and
nondominated in the independently acquired archive that the selector actually
received. Multi-objective eligibility is independent: all seven MOO jobs have
`multi_objective_min_accuracy: null` and do not inherit latency retention.
The classification column is the machine-derived **mode-local selection-policy
classification**. The middle-ground column separately records the observed
ordering of active winners across independent archives; under Design B that
column is not a selector invariant. Measurement-stability conclusions are
reported separately below and can remain inconclusive without changing a
mode-local selection-policy pass.

## Differential diagnosis

### Deformable DETR

The MOO winner is a genuine normalized augmented-Chebyshev compromise on its
six-point rank-zero front. It is faster and less accurate than the winner of
the separate constrained-latency job, but Design B does not require ordering
across those two finite archives. Offline replay reproduces the frozen winner
and no candidate in the MOO archive dominates it. The mode-local selection
policy is `PASS_EXPECTED_COMPROMISE`; the cross-mode endpoint ordering remains
observational and is not grounds for a selector change or result-driven rerun.

### RT-DETR

The MOO archive has no distinct selectable interior point under the frozen
policy. Deterministic endpoint fallback selects the accuracy-side endpoint.
This is valid `PASS_ENDPOINT_COLLAPSE` geometry; three distinct outputs are not
required.

### Grounding DINO

The accuracy selector returns the maximum valid mAP50 in its own archive
(`0.7699382696378168`). The independent constrained-latency acquisition later
discovers a different specification with mAP50 `0.7774643271`. The latency
recommendation audit proves that this point was generated by constrained
expected improvement from the frozen search space and seed, with raw accuracy
and latency observations visible to the algorithm and all intervention flags
false. The accuracy job never acquired that fingerprint.

Thus the first divergence is candidate acquisition/archive coverage, not
metric extraction or selection. The signed recommendation audit verifies all
60 recommendations, the maximize/minimize directions, complete visible
history, frozen bounds, canonical fingerprints, unique recommendations,
mode-aware acquisition routing, and the deterministic 16-candidate common
calibration prefix. Only four model-based recommendations per mode remained;
the higher point was the latency job's algorithmic recommendation 18 and was
reachable but not proposed by the accuracy job. No acquisition implementation
defect was reproduced.

The difference (`0.007526`) is smaller than the median (`0.0310464`) and
maximum (`0.0617162`) accuracy ranges observed when the same 16 calibration
fingerprints were independently trained across jobs. A preregistered
six-repeat matched validation therefore compared the two frozen fingerprints
without changing any selection-time value. All 12 training/evaluation cells
completed successfully.

| Repeat | Accuracy-archive winner | Higher external fingerprint | Paired difference |
| ---: | ---: | ---: | ---: |
| 0 | 0.7535306551 | 0.7574336039 | +0.0039029487 |
| 1 | 0.7378166318 | 0.7466457950 | +0.0088291633 |
| 2 | 0.7764560807 | 0.6749889587 | -0.1014671220 |
| 3 | 0.7684152840 | 0.7578483290 | -0.0105669549 |
| 4 | 0.7683333918 | 0.7565038404 | -0.0118295514 |
| 5 | 0.7537537553 | 0.7516969656 | -0.0020567897 |

The median paired difference is `-0.0063118723`, paired-difference MAD is
`0.0078662500`, and the descriptive paired-bootstrap 95% interval is
`[-0.0566483367, 0.0063660560]`. Direction changes across repeats and the
interval crosses zero. The preregistered classification is therefore
`DIRECTION_UNRESOLVED_OR_TRAINING_NOISE`. The apparent cross-archive advantage
is not reproducibly established; no acquisition implementation defect and no
selector defect were demonstrated. The original archive winners and objective
values remain unchanged.

```text
matched validation contract:
/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/pareto_outlier_validation/grounding_dino/matched_accuracy_v1/contract.json
file sha256 58210a8e483bf88720fd2a48ff4f2347f12ed79991452658b39d4e025f9402c9
canonical contract sha256 6abfa3181c649b2e43647296b212bae704030106c7afe37f82849317e48fa902
implementation commit 3b23e2dc76e15315e2cb48c57db83bcaca8cd3f2
matched result sha256 ac14324309a447f1d4c03e0a38b4e3776e27007aa685937f7815197fef7dd361
execution state sha256 c9d9006c6021dff11d9a19fed626bbb6462cf23c1c790aa26a5da0d20dc8c48c
```

The acquisition audit is:

```text
/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/pareto_outlier_validation/grounding_dino/acquisition_audit.json
sha256 01d51850c34b8e63fb07e8bc771dee1f741f90fbf97d5fd880a63cde9ee01577
```

### SegFormer

The selected MOO point is a distinct rank-zero compromise in its own archive.
In the read-only union, latency `rec_12` has higher mIoU and a median only
`0.034172 ms` lower. That difference is far inside the configured
`0.73553775 ms` practical tolerance and was not measured in a matched-allocation
design. Mode-local selection is correct; the cross-job latency direction is
practically unresolved and no selector or search setting is changed.

### Mask2Former

All three per-mode selectors replay correctly and the selected MOO point is a
distinct nondominated compromise in its own archive. The complete 24 signed
rank traces for the three active-mode winners reproduce the production
device-round-cluster median and raw-sample p95 exactly. They also localize the
large tails to deterministic positions in the frozen 16-input schedule on all
eight replicas: latency `rec_2` has one 128.924-ms position and MOO `rec_4` has
seven roughly 128.4--129.0-ms positions. Their within-position dispersion is
small; the tail is input-shape-sensitive model-forward behavior, not an
isolated cold start, missing synchronization, units error, CPU path, or
selector defect.

The frozen product policy uses the robust median objective, whose round,
device, confidence, drift, and robust-CV gates passed. p95 remains a reported
workload diagnostic and is not substituted after observing the result. No
matched-allocation evidence exists for the cross-job endpoints, so cross-job
direction remains measurement-limited even though mode-local selection is
`PASS_EXPECTED_COMPROMISE`.

The trace audit is:

```text
/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/pareto_outlier_validation/mask2former_latency_tail_audit.json
sha256 876bf4c121ce3b7cd863e613f165e5b2b3bc92be44679bd77a1463e1bee43929
```

The apparent cross-archive accuracy violation is the same fingerprint trained
independently: `0.3092069205` versus the accuracy winner's `0.3088468709`.
That is explicitly recorded as retraining variation rather than a distinct
search configuration defeating accuracy mode.

### Mask Grounding DINO

The prior v8 campaign is not valid Pareto evidence. Training and evaluation
could complete, but latency packaging then attempted to read
`latency_stats.py` and `latency_benchmark.py` from the retired path
`/localhome/local-rarunachalam/tao-automl`. The first incorrect state was the
sealed `runtime.repository`; `_payload_command` consequently raised
`FileNotFoundError` before a complete objective vector could enter the
archive. This is a model-integration/objective-measurement failure, not a
selector defect.

The source fix binds new manifests and successor helpers to the active
GitHub checkout and validates that the exact packaging modules exist. The v8
artifacts and failures remain immutable. A fresh v11 successor uses the same
candidate budget, search spaces, PTMs, training budget, seeds, objectives,
latency protocol, SQSH identity, and eight-GPU resource contract. Its source
is the immutable GitHub-backed worktree at commit `8cdcd36`.

MGDINO v11 completed all 24 algorithm-generated recommendations per mode.
Accuracy produced 23 valid objective vectors and one preserved terminal
failure; latency and MOO each produced 22 valid vectors and two preserved
terminal failures. Failed records were excluded from selection but retained
in the audit.

| Mode | Candidate | Accuracy | Median latency | Pareto rank | Compromise score | Selection reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Accuracy | `accuracy_rec_16` | 0.3062185856 | 207.434521 ms | 0 | 0.5000005 | Maximum valid accuracy |
| Latency | `latency_rec_14` | 0.3019983359 | 204.706699 ms | 0 | 0.1029697379 | Highest-accuracy member of the raw-minimum-anchored equivalent-fastest feasible cohort |
| Multi-objective | `multi_objective_rec_19` | 0.3035605194 | 204.449796 ms | 0 | 0.1399983584 | Minimum normalized augmented-Chebyshev regret on the eligible rank-zero front |

The complete MOO rank-zero front is `rec_13`, `rec_11`, `rec_19`, `rec_10`,
and `rec_9`. Its accuracy bounds are `[0.2893246404, 0.3090965905]`; latency
bounds are `[203.60024475, 207.35517325]` ms. `rec_19` is a distinct
nondominated compromise. Its latency and the independently acquired latency
winner differ by only `0.2569035 ms`, inside the frozen `0.73553775 ms`
tolerance; Design B does not impose a raw cross-archive total order.

Persisted winners match production replay and ten candidate-order
permutations. All intervention and post-hoc selection flags are false. The
terminal audit is:

```text
/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/pareto_outlier_validation/mgdino_v11/mgdino_v11_final_audit.json
file sha256 71accf1d5a58c54746b434e94af0569cc98eccc93880bcc375fe024f8d37b5eb
canonical audit sha256 e0033201c0ba18eef212afe4c56c93b41ed1afea53cb41e85d1bbb8b0161b5de
```

## Production selector review

No selector correctness defect was found.

* Accuracy maximizes the finite task metric and uses latency only after an
  actual accuracy tie.
* Latency resolves its own accuracy reference, filters for feasibility,
  anchors the equivalent cohort at the raw minimum stabilized latency, then
  applies the documented accuracy/fingerprint/ID tie policy.
* MOO independently filters valid points, performs maximize-accuracy /
  minimize-latency nondominated sorting, normalizes regrets over the eligible
  rank-zero front, and minimizes the configured augmented-Chebyshev score.
* Zero-range objectives contribute zero regret. Failed, missing, NaN,
  infinite, non-positive-latency, and incomplete-confidence-interval records
  cannot enter the valid front.
* Persistent fingerprints and ties use canonical SHA-256 identities, not
  process-dependent Python hashes or archive enumeration order.

## Intervention and selection isolation

The audit and fresh rerun freeze all of the following to false:

```text
agent_selected_candidate
agent_overrode_winner
agent_injected_candidate
agent_removed_candidate_to_change_winner
agent_changed_objective_weights_after_results
agent_changed_accuracy_retention_after_results
agent_changed_multi_objective_policy_after_results
agent_changed_search_space_after_results
agent_changed_seed_after_results
agent_replaced_measurement
agent_modified_metric_to_favor_candidate
agent_increased_budget_for_preferred_candidate
agent_reordered_candidates_to_affect_ties
```

Matched or post-hoc measurements are not used for selection or reselection.

## Verification

```text
production: 985 passed, 1 skipped
cross-model experiment suite: 567 passed
focused affected-model/replay suite: 235 passed
objective-aware acquisition/resume suite: 66 passed
outlier audit/finalization regression tests: 31 passed
```

The only warnings are the existing sklearn Gaussian-process convergence
warnings. Historical experiment checks now use separate immutable
GitHub-backed worktrees at each originally frozen SDK/source revision, avoiding
the former shared-path commit collision without regenerating evidence.

## Final status

The eight completed model campaigns satisfy their independently defined
mode-local selection policies. Seven are `PASS_EXPECTED_COMPROMISE`; RT-DETR
is a valid `PASS_ENDPOINT_COLLAPSE`. Cross-archive ordering remains
observational under Design B. Grounding DINO is training-noise-limited under
matched validation, SegFormer is latency-tolerance-limited, and Mask2Former is
latency-tail/matched-allocation-limited; none demonstrates a selector defect.

```text
Passing before: 7 completed model campaigns with valid mode-local selectors
True outliers investigated: 5
Generic AutoML bugs fixed: 0 selector bugs; 1 finalization automation bug
Model-specific bugs fixed: 1 MGDINO runtime/objective-packaging integration bug
Valid no-compromise cases: 1 endpoint collapse (RT-DETR)
Inconclusive/noise-limited cases: 3 validation qualifiers (Grounding DINO, SegFormer, Mask2Former)
Affected campaigns rerun: 1 full three-mode campaign plus 1 matched validation
Passing after: 8 model campaigns
Remaining failures: 0 selector or model-integration failures
Algorithm-selected winners overridden by agent: 0
```
