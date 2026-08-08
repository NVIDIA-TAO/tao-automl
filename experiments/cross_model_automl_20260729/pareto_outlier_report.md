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

The current replay artifact is:

```text
/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/pareto_outlier_audit_v6_design_b/matrix.json
sha256 78b7daed210cbb01c3b3e3c9a261abbea97c86e6a7f37bc274f229d39d932963
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
fingerprints were independently trained across jobs. The coverage observation
therefore remains `INCONCLUSIVE_PENDING_MATCHED_ACCURACY_VALIDATION`, not a
selector failure. A prospective six-repeat validation contract compares the
two fingerprints without changing any frozen selection-time value. It is
gated on successful MGDINO v11 completion and all validation measurements are
isolated from selection and reselection.

```text
matched validation contract:
/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/pareto_outlier_validation/grounding_dino/matched_accuracy_v1/contract.json
file sha256 857e829751924456f842ab915f60da17866abeb72fe33efead72c0b13db52e4c
canonical contract sha256 b4e44109cdace0603b5bf46ab4dd3ab51d664762d2996157093133376d51406e
implementation commit 5cb07807c747e2d4cb5e02541d958cae949393e3
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
cross-model experiment suite: 565 passed
focused affected-model/replay suite: 235 passed
objective-aware acquisition/resume suite: 66 passed
outlier audit/finalization regression tests: 27 passed
```

The only warnings are the existing sklearn Gaussian-process convergence
warnings. Historical experiment checks now use separate immutable
GitHub-backed worktrees at each originally frozen SDK/source revision, avoiding
the former shared-path commit collision without regenerating evidence.

## Rerun status

Mask Grounding DINO v11 is the only campaign being rerun. Final winners,
complete candidate evidence, the final rank-zero front, and the after
classification must be appended from that algorithm-generated completion;
they must not be inferred or selected manually while the campaign is active.
