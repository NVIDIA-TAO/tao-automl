# DINO latency-mode 90% retained-accuracy validation

Status: **complete**

Evidence cutoff: 2026-07-29 UTC

Source branch: `rarunachalam/pre-platform-sdk-removal-20260714`

Frozen replay-v1, launch, and recovery evidence source commit:
`2bd7b9087648dfba57e6ce915738f0cc350b3808`

Routing-explicit replay-v2 source commit:
`330dbc61efbebf6158e7dcecc59aea704039737b`

Complete matched-evidence commit:
`c5e45c46c61ab459eb300b84abbdbbc678069239`

Complete low-latency follow-up evidence commit:
`f3203c3ab4699dc5f629a9f6cb8d100e7498ceff`

Review target: existing MR !22

## 1. Scope and evidence boundary

This report covers only DINO ResNet50 and:

`s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/`

It records the production policy implementation, a deterministic replay of
the sealed 60-candidate archive, the complete 90%-feasible population, the
equivalent-fastest-cohort audit, checkpoint-recovery provenance, completed
matched validation, and the completed preregistered lower-latency follow-up
search.

All requested conclusions are complete:

- the 90% relative threshold and feasible population;
- the production-selector replay and archive-order invariance;
- explicit active `mode="latency"` final-winner routing;
- the algorithm-selected latency winner;
- the separation of latency and multi-objective eligibility;
- the matched-campaign candidate scope and schedule;
- all 12 quality-gated matched-latency cells and the allocation-paired
  classification;
- identity-preserving recovery of the unavailable `rec_6` checkpoint
  configuration;
- the follow-up search space, seeds, budget, prohibitions, all 60 terminal
  recommendation records, and the 120-record union selection;
- the negative result that no new accuracy-feasible candidate improves on
  the existing approximately-57-ms latency result;
- the final mode-specific hypothesis and DINO reference-readiness verdict.

Matched measurements remain validation-only. This report does not use them
to replace selection-time objectives or re-run selection.

## 2. Production latency policy

The experiment profile is:

```yaml
automl_settings:
  selection_mode: latency
  latency_accuracy_retention:
    type: relative
    retained_fraction: 0.90
    reference: accuracy_winner
  latency_tolerance: 0.73553775
  multi_objective_min_accuracy: null
```

The repository-wide implicit relative-retention default remains `0.98`.
The explicit DINO profile selects `0.90`; this work does not silently change
the global default.

For valid candidates \(x\), production latency mode computes:

\[
A^* = \max_x A(x)
\]

\[
A_{\min} = 0.90 A^*
\]

\[
F = \{x : A(x) \geq A_{\min}\}
\]

It then:

1. finds the raw minimum stabilized latency in \(F\);
2. anchors the equivalent-fastest cohort at that raw minimum;
3. admits only candidates whose observed median-latency disadvantage is no
   greater than the hard `0.73553775 ms` practical tolerance;
4. chooses the highest-accuracy member of that cohort;
5. uses canonical specification fingerprint and candidate ID for any
   remaining deterministic tie.

Accuracy cannot promote a candidate whose observed latency disadvantage
exceeds the practical tolerance. Candidate latency confidence intervals are
required to be finite, ordered, and to contain the median. They remain part
of uncertainty reporting and Pareto analysis, but confidence-interval overlap
does not widen the hard raw-minimum-anchored cohort beyond the practical
tolerance.

The direct-winner reason is:

> Lowest stabilized latency candidate satisfying the accuracy-winner-relative
> constraint; no equivalent-fastest tie-break was required.

The tied-cohort reason is:

> Highest-accuracy member of the equivalent-fastest cohort satisfying the
> accuracy-winner-relative constraint; deterministic specification
> fingerprint and candidate ID resolve remaining ties.

Retention values must be finite and satisfy \(0 < r \leq 1\). Boolean,
zero, negative, greater-than-one, NaN, and infinite values are rejected.
Conflicting preferred and legacy representations are also rejected rather
than resolved silently.

Accuracy mode remains highest-valid-accuracy selection. Multi-objective mode
continues to use its independently eligible Pareto front and does not inherit
`latency_accuracy_retention`; `multi_objective_min_accuracy` is unset here.
No weighted accuracy-latency score is used by latency mode.

Relevant implementation paths:

- `src/tao_automl/selection.py`
- `src/tao_automl/objectives.py`
- `experiments/dino_moo_phase2_20260728/dino_latency_90_policy_profile.v1.json`

## 3. Sealed 60-candidate archive replay

### 3.1 Input integrity and threshold

The replay used all 60 successful, unique candidates from the unchanged
expanded DINO archive. No training or benchmarking occurred during replay.

| Item | Value |
| --- | --- |
| Sealed candidate table SHA-256 | `5ba323d05d9ec8e3703e636f8b5e2975cc620eeec10df75ec6e792318dc2df03` |
| Sealed combined selection SHA-256 | `78ab9d2fa83cc3abe9057d137c0b88f120158b6ad77268482d2c18f5a1533af1` |
| Valid/successful candidates | 60/60 |
| Accuracy winner | `seed_271828_rec_18` |
| Accuracy-winner mAP50 | `0.6554138278683255` |
| Retained fraction | `0.90` |
| Exact decimal product | `0.589872445081492950` |
| Production IEEE-754 threshold | `0.589872445081493` |
| Feasible candidates | 17 |

The shorter value `0.5898724450814929` is a display truncation of the same
policy calculation; no candidate lies close enough to the boundary for the
representation difference to affect membership.

### 3.2 Selector result

| Field | Production replay result |
| --- | --- |
| Raw minimum-latency feasible candidate | `seed_271828_rec_19` |
| Raw minimum selection-time median | `57.146624 ms` |
| Equivalent-fastest cohort | `seed_271828_rec_19`, `seed_271828_rec_6` |
| Selected latency candidate | `seed_271828_rec_19` |
| Accuracy tie-break invoked | yes |
| Fingerprint/candidate-ID tie-break required | no |
| Manual candidate injection | no |
| Manual winner override | no |

The exact interpretation is:

> Latency mode selected the highest-accuracy member of the
> equivalent-fastest cohort satisfying 90% retained accuracy.

`seed_271828_rec_19` is also the raw-fastest member, so the accuracy tie-break
did not change the raw-minimum candidate identity. It did resolve the
two-member cohort according to the configured product semantics.

This replay alone does **not** establish that `rec_19` is measurably faster
than `rec_6`. The completed matched validation in Section 7 establishes that
there is no stable direction and that the two-member cohort is descriptively
practically equivalent.

### 3.3 Archive-order invariance

The production selector was replayed over the archive order, reverse order,
candidate-ID order, and six deterministic permutations. All nine orderings
produced byte-identical complete selector output.

| Check | Result |
| --- | --- |
| Ordering count | 9 |
| Complete selector output identical | yes |
| Common selector-output SHA-256 | `53adbab0620865dd9f386a73e6aa985fd50f5f5f16776c3311ceda37ccfc0f00` |
| Candidate-ID set SHA-256 | `7b4de18434ce46fa7f9238a7be2fda4185fbbe68d1046ca9dbb923957bfb6549` |

Accuracy and multi-objective output hashes also remain identical to the
sealed selection:

| Mode | Selection SHA-256 |
| --- | --- |
| Accuracy | `f2e471894857753bd369422645134240febbfc688b3b7f2c9cc48b24ae81fa92` |
| Multi-objective | `7cd6a729e43250a8fc182d0b61958f0b5e274fa42b546bb331fda645b7e46f43` |

### 3.4 Replay-version boundary

Two replay artifacts have different audit purposes and must not be
conflated:

- `archive_replay.v1.json` is the immutable replay bound by whole-file digest
  into the already submitted matched execution projection. It derives the
  17-member feasible population, the two-member cohort, and the latency
  selection from the complete `analyze_archive` output. Its original
  configuration inherited the sealed archive analysis mode, so v1 does not
  by itself prove that the top-level active-mode winner-routing API was
  invoked with `mode="latency"`.
- `archive_replay.v2.json` is the finalized routing-explicit successor. Its
  generator sets `SelectionConfig(mode="latency")`, asserts the active mode,
  calls the final `analysis.winner()` route, and verifies that the returned
  candidate object is the constrained latency-selection winner. It also
  continues to compare accuracy and multi-objective selections with the
  sealed archive.

V2 is a routing-proof addition, not a new archive, threshold, feasible set,
cohort, candidate promotion, or matched-scope change. The matched campaign
remains bound to v1 for immutable launch provenance. V2 is committed
separately at `330dbc61efbebf6158e7dcecc59aea704039737b` and records:

```text
selection_mode = latency
analysis_winner_candidate_id = seed_271828_rec_19
latency_selection_candidate_id = seed_271828_rec_19
winner_route_matches_latency_selection = true
```

### 3.5 Complete 90%-feasible candidate table

All selection-time values below remain frozen. Stable medians for the
two-member matched scope are validation-only values from the completed
90%-policy campaign; they did not replace the selection-time objectives or
feed reselection.

| Candidate | Search specs `(enc, dec, lr, wd)` | mAP50 | Accuracy retained | Selection-time median ms | Selection-time p95 ms | Selection-time median 95% CI ms | Delta from raw min ms | 90%-feasible | Equivalent-fastest cohort | Stable median | Selected |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | :---: | :---: | ---: | :---: |
| `seed_271828_rec_19` | `(4, 3, 0.00045, 0.0006630648780334237)` | 0.6175134981289873 | 94.2173% | 57.146624 | 57.362506 | [57.11720625, 57.19984475] | 0 | yes | yes | 56.921768375 ms (90%-policy matched) | yes |
| `seed_271828_rec_6` | `(4, 3, 0.000487310659095131, 0.0009)` | 0.6000121414379619 | 91.5471% | 57.17349525 | 57.37278405 | [57.10523675, 57.219922] | 0.02687125 | yes | yes | 56.9539065 ms (90%-policy matched) | no |
| `seed_271828_rec_12` | `(5, 3, 0.00045, 0.0009)` | 0.5924694360231197 | 90.3962% | 61.77041675 | 61.930805 | [61.74224375, 61.79941525] | 4.62379275 | yes | no | not in derived scope | no |
| `seed_271828_rec_1` | `(5, 3, 0.0004417551531468059, 0.0004796687699134978)` | 0.6044653164228678 | 92.2265% | 61.82193 | 62.0668055 | [61.79774925, 61.8862685] | 4.675306 | yes | no | not in derived scope | no |
| `seed_161803_rec_11` | `(5, 3, 0.0002684985944234011, 0.0009485964192103246)` | 0.5984845568353709 | 91.3140% | 61.89238125 | 62.14956105 | [61.8307895, 61.98849875] | 4.74575725 | yes | no | not in derived scope | no |
| `seed_271828_rec_18` | `(6, 3, 0.00045, 0.0001962863874708991)` | 0.6554138278683255 | 100% | 66.23099475 | 66.59742 | [66.1995845, 66.26569825] | 9.08437075 | yes | no | 66.49666775 ms (prior global-front campaign) | no |
| `seed_271828_rec_8` | `(6, 3, 0.00045, 0.0006547724796069428)` | 0.6278092014414639 | 95.7882% | 66.41633275 | 66.718472 | [66.38007425, 66.526863] | 9.26970875 | yes | no | not in derived scope | no |
| `seed_314159_rec_18` | `(6, 3, 0.00045, 0.000011)` | 0.600109443334249 | 91.5619% | 66.53805125 | 66.71788005 | [66.50235025, 66.5806925] | 9.39142725 | yes | no | not in derived scope | no |
| `seed_314159_rec_17` | `(6, 3, 0.0003213304064567958, 0.0007597966799758081)` | 0.6316439030814589 | 96.3733% | 66.5648545 | 66.84640005 | [66.4994655, 66.63726025] | 9.4182305 | yes | no | not in derived scope | no |
| `seed_161803_rec_14` | `(6, 3, 0.00045, 0.0007784663668118407)` | 0.6503731411565659 | 99.2309% | 66.5894095 | 66.9217075 | [66.539746, 66.62343725] | 9.4427855 | yes | no | not in derived scope | no |
| `seed_314159_rec_7` | `(6, 3, 0.00045, 0.00043930511913497345)` | 0.6300270529757901 | 96.1266% | 66.58953225 | 66.800954 | [66.5359365, 66.662066] | 9.44290825 | yes | no | not in derived scope | no |
| `seed_271828_rec_13` | `(6, 3, 0.0003409813457951682, 0.00037590573325465534)` | 0.6388208734864198 | 97.4683% | 66.6654645 | 66.9667695 | [66.595203, 66.71631525] | 9.5188405 | yes | no | not in derived scope | no |
| `seed_314159_rec_12` | `(6, 3, 0.00043650890201375357, 0.0007144522744519978)` | 0.6517250365478822 | 99.4372% | 66.685121 | 66.8530975 | [66.5915665, 66.73869475] | 9.538497 | yes | no | not in derived scope | no |
| `seed_271828_rec_16` | `(6, 3, 0.0003007572504594793, 0.000011)` | 0.6544218576499151 | 99.8486% | 66.82186425 | 67.0504435 | [66.785156, 66.87998275] | 9.67524025 | yes | no | not in derived scope | no |
| `seed_314159_rec_5` | `(6, 4, 0.0003255265920347484, 0.000175988605139262)` | 0.5987692227100141 | 91.3574% | 70.88016575 | 71.137551 | [70.707171, 70.89530125] | 13.73354175 | yes | no | not in derived scope | no |
| `seed_314159_rec_11` | `(6, 6, 0.00045, 0.0001417474630742632)` | 0.6107135016797408 | 93.1798% | 79.24220375 | 79.7769715 | [79.198435, 79.28856475] | 22.09557975 | yes | no | not in derived scope | no |
| `seed_314159_rec_3` | `(6, 6, 0.00042331502060985004, 0.0005537484468632733)` | 0.5970032852425997 | 91.0880% | 79.80455425 | 80.1799975 | [79.6680815, 79.9265] | 22.65793025 | yes | no | not in derived scope | no |

## 4. Equivalent-fastest-cohort audit

The production cohort is anchored at `rec_19`, the raw minimum
`57.146624 ms`.

| Candidate | mAP50 | Median ms | Delta from anchor ms | Practical-tolerance test | Selection-time CI relation | Result |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `seed_271828_rec_19` | 0.6175134981289873 | 57.146624 | 0 | within | anchor | selected |
| `seed_271828_rec_6` | 0.6000121414379619 | 57.17349525 | +0.02687125 | within | overlaps anchor CI | tied, then loses accuracy tie-break |

The nearest excluded feasible candidate is `seed_271828_rec_12`:

- median delta from raw minimum: `4.62379275 ms`;
- CI gap from the raw anchor: `4.542399 ms`;
- it is outside the practical tolerance and its selection-time uncertainty
  cannot plausibly move it into the cohort.

Therefore, the selector-derived matched scope contains exactly `rec_19` and
`rec_6`. No candidate was manually added or removed.

The matched evidence confirms that both median and p95 have no stable
directional claim and are descriptively practically equivalent within
`±0.73553775 ms`. The production higher-accuracy tie-break is therefore
consistent with the independent validation.

## 5. Final three-mode comparison

The accuracy-mode stable median comes from the completed global Pareto-front
campaign. The shared `rec_19` value shown for latency and multi-objective mode
comes from the newer completed 90%-cohort campaign. These values are
validation context only and did not feed the replay or change either mode's
winner.

| Mode | Candidate | Accuracy | Accuracy retained | Stable median latency | Latency reduction from accuracy mode | Selection reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Accuracy | `seed_271828_rec_18` | 0.6554138278683255 | 100% | 66.49666775 ms (global-front campaign) | 0 ms | Highest valid mAP50 |
| Latency, 90% retention | `seed_271828_rec_19` | 0.6175134981289873 | 94.21734359% | 56.921768375 ms (90%-cohort campaign) | 9.574899375 ms (14.3991%) | Highest-accuracy member of the validated equivalent-fastest feasible cohort |
| Multi-objective | `seed_271828_rec_19` | 0.6175134981289873 | 94.21734359% | 56.921768375 ms (same candidate, 90%-cohort campaign) | 9.574899375 ms (14.3991%) | Independent global-front minimum normalized augmented-Chebyshev regret; distinct Pareto compromise |

Latency mode and multi-objective mode independently select the same archive
candidate under different policies. That is permitted and is not evidence of
threshold inheritance. The completed follow-up added 59 valid objective
vectors to the selection population but did not change any mode winner. The
final union contains 34 latency-feasible candidates: 17 sealed candidates and
17 new candidates.

## 6. `rec_6` checkpoint recovery provenance

The exact historical `seed_271828_rec_6` checkpoint had been deleted from its
sealed result path. The validation harness did not substitute another
candidate or architecture. It reconstructed and retrained the exact frozen
candidate configuration under a preregistered, fail-closed recovery contract.

| Item | Value |
| --- | --- |
| Candidate | `seed_271828_rec_6` |
| Search seed / recommendation | `271828` / `6` |
| Training seed | `1234` |
| Frozen spec SHA-256 | `1366f23682c5c495b65ee6132cd883f2891a5c8b0e278605ec372301b11319df` |
| Resolved model SHA-256 | `2891ae9dbb6097c1da53ce68201359f7da7992d30515ab815c89f906cdce21b1` |
| Reconstructed train-spec SHA-256 | `0ce980ef6e6f793ab3a3aaac27957bc6daee4ab7b8269958036cd900d3dd9092` |
| Historical checkpoint SHA-256 | `0338c35be50bbad6189d38e8f9007856a60e87a0861c8a6ff5d0bf85cd6df6c5` |
| Recovered checkpoint SHA-256 | `5f259679d61db1a85bbbec0625cb71367f2be2660ab4d09fe416e97d2717f6f0` |
| Recovered checkpoint size | 475,869,698 bytes |
| TAO job ID | `1c8a5f67-ff12-4f5b-a4ab-fcae3714c92f` |
| SLURM job ID | `31021375` |
| Node | `batch-block7-01780` |
| SLURM state / exit | `COMPLETED` / `0:0` |
| Start / end UTC | `2026-07-29T00:14:48Z` / `2026-07-29T00:20:53Z` |

The recovery is configuration-exact, not byte-identical to the deleted
historical checkpoint. The evidence explicitly records
`configuration_exact_not_byte_identity = true` and
`historical_sha256_match = false`. It is retained only for matched latency
validation. It does not mutate the sealed archive, replace selection-time
metrics, invoke the selector, or override the winner.
The frozen historical selector, its candidate objectives, and its selected
candidate remain untouched.

## 7. Matched 90%-cohort validation

### 7.1 Frozen design

Campaign: `dino_latency_90pct_matched_20260728_v1`

Status at this report cutoff: **complete; six allocations and all 12
candidate/allocation cells passed the frozen quality and provenance gates**.

The immutable execution projection derives its two candidates directly from
`archive_replay.v1.json` and joins their records from the sealed 60-row table.

| Protocol field | Frozen value |
| --- | --- |
| Candidates | `seed_271828_rec_19`, `seed_271828_rec_6` |
| Independent allocations | 6 |
| Nodes / GPUs per allocation | 1 / 8 A100-SXM4-80GB |
| Container | prebuilt TAO 7.0.1 PyTorch SQSH |
| Precision / TF32 | fp32 / disabled |
| Batch size | 1 per GPU |
| Fixed model input | `[1, 4, 800, 1333]` |
| Warm-ups | 50 |
| Timed rounds | 5 |
| Timed iterations per round | 100 |
| Raw samples per candidate/allocation | 4,000 |
| Complete cells / raw samples | 12 / 48,000 |
| Timed scope | model forward plus DINO GPU postprocess |
| Synchronization | CUDA sync per sample plus NCCL barrier |
| Per-cell bootstrap | 5,000 resamples, seed `424242`, 95% |
| Practical tolerance | `0.73553775 ms` |

Every complete cell must pass the frozen median/p95, MAD, IQR, robust-CV,
round-drift, device-spread, bootstrap-width, and provenance gates. A failed
cell invalidates its complete allocation block.

The balanced two-row Williams schedule gives each candidate each execution
position exactly three times and balances both ordered adjacencies:

| Allocations | Execution order |
| --- | --- |
| 00, 01, 02 | `rec_19`, then `rec_6` |
| 03, 04, 05 | `rec_6`, then `rec_19` |

Schedule SHA-256:
`bef77aaafa7d6ad7693ed2d4b554bb799e08286392db33dd1054e67668e810e5`

### 7.2 Frozen statistical rule

Median and p95 are analyzed independently using the six allocation-matched
differences, with delta defined as first candidate minus second candidate.
For each endpoint the report must contain:

- all six paired differences;
- their median;
- a descriptive allocation-paired percentile-bootstrap 95% interval
  (`10,000` resamples, seed `20260728`);
- an exact one-sided tolerance-shifted sign-flip result;
- whether all six differences cross the same practical-tolerance boundary;
- a separate descriptive-equivalence classification.

A stable direction requires both:

1. exact one-sided shifted sign-flip \(p \leq 0.05\); and
2. all six differences strictly beyond `±0.73553775 ms` in the same
   direction.

Failure to prove direction does not imply equivalence. Descriptive practical
equivalence is reported only if the entire descriptive interval lies inside
the tolerance band. No simultaneous total ordering is claimed.

### 7.3 Submission provenance

The immutable projection passed local source, recovery-evidence, remote
dataset/PTM/SQSH, and credential-containment checks before submission.
The submission ledger records all six jobs:

| Allocation | Candidate order | TAO job ID | SLURM job ID |
| --- | --- | --- | --- |
| 00 | `rec_19`, `rec_6` | `88825e1c-5ca0-462f-ad79-7cf14251650f` | `31021945` |
| 01 | `rec_19`, `rec_6` | `98688222-6e20-4910-a9fc-5b29e38b461d` | `31021984` |
| 02 | `rec_19`, `rec_6` | `6915e2cf-b9dc-4486-b736-ab1e7eb5a303` | `31021991` |
| 03 | `rec_6`, `rec_19` | `67a941eb-bee8-4310-9e70-62ffef7740c5` | `31022014` |
| 04 | `rec_6`, `rec_19` | `f7351cd3-365d-4f9d-824e-af49124cf234` | `31022016` |
| 05 | `rec_6`, `rec_19` | `19b45940-5e88-4e5c-ae48-8ed3e1e9424f` | `31022018` |

All six submitted jobs subsequently reached SLURM `COMPLETED`, SDK
`Complete`, and exit code `0:0`. The final aggregation artifact independently
verifies scheduler-hostname binding, eight GPU UUIDs per allocation,
benchmark-input identity, complete-block, hardware, protocol, and runtime
contracts.

The following isolation flags are frozen false:

```text
selector_invoked_on_matched_measurements = false
selection_time_objectives_replaced = false
measurements_feed_selection = false
measurements_feed_reselection = false
algorithm_selected_candidate_overridden = false
```

### 7.4 Complete 12-cell evidence

Every cell contains 4,000 raw measurements. `Drift` and `device spread` are
fractions of the cell median. Every row passed every frozen gate.

| Allocation | Candidate | Position | Median ms | p95 ms | Median bootstrap 95% CI ms | MAD / IQR ms | Robust CV | Abs. drift | Device spread | Gates |
| ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | :---: |
| 00 | `rec_19` | 0 | 56.9399595 | 57.17573095 | [56.90437375, 56.975143] | 0.072246 / 0.15156975 | 0.00188114 | 0.00115747 | 0.00591688 | pass |
| 00 | `rec_6` | 1 | 56.99692325 | 57.16131705 | [56.91848225, 57.012594] | 0.07964125 / 0.1603755 | 0.00207162 | 0.00032657 | 0.00451544 | pass |
| 01 | `rec_19` | 0 | 56.866871 | 57.10177795 | [56.8343805, 56.926504] | 0.072611 / 0.153172 | 0.00189307 | 0.00077660 | 0.00394118 | pass |
| 01 | `rec_6` | 1 | 56.81540775 | 57.05208255 | [56.773135, 56.8385405] | 0.07167775 / 0.15969925 | 0.00187043 | 0.00034973 | 0.00460665 | pass |
| 02 | `rec_19` | 0 | 57.32534325 | 57.52484895 | [57.29392225, 57.358616] | 0.09033725 / 0.17619975 | 0.00233638 | 0.00094604 | 0.00674867 | pass |
| 02 | `rec_6` | 1 | 57.32423075 | 57.51382945 | [57.25523425, 57.3783295] | 0.093507 / 0.190555 | 0.00241841 | 0.00021955 | 0.00557248 | pass |
| 03 | `rec_19` | 1 | 56.90247625 | 57.144475 | [56.8785285, 56.923182] | 0.068811 / 0.17178625 | 0.00179288 | 0.00042601 | 0.00480111 | pass |
| 03 | `rec_6` | 0 | 56.910804 | 57.14359155 | [56.89520575, 56.959942] | 0.094756 / 0.17546275 | 0.00246852 | 0.00053129 | 0.00737281 | pass |
| 04 | `rec_19` | 1 | 56.90357725 | 57.110867 | [56.88355675, 56.92163525] | 0.08191725 / 0.16016325 | 0.00213432 | 0.00049312 | 0.00579217 | pass |
| 04 | `rec_6` | 0 | 56.9322125 | 57.09979495 | [56.90457725, 56.96428825] | 0.059036 / 0.11640125 | 0.00153739 | 0.00098628 | 0.00364202 | pass |
| 05 | `rec_19` | 1 | 56.98435025 | 57.2017485 | [56.94514775, 57.01424575] | 0.07338075 / 0.15859525 | 0.00190920 | 0.00002596 | 0.00481910 | pass |
| 05 | `rec_6` | 0 | 56.9756005 | 57.199902 | [56.8996525, 57.058294] | 0.095346 / 0.193668 | 0.00248106 | 0.00045437 | 0.00522169 | pass |

### 7.5 Between-allocation stability

| Candidate | Stable median ms | Median range / SD ms | Median MAD / IQR ms | Stable p95 ms | p95 range / SD ms | p95 MAD / IQR ms |
| --- | ---: | --- | --- | ---: | --- | --- |
| `seed_271828_rec_19` | 56.921768375 | 0.45847225 / 0.17042918 | 0.03709475 / 0.07050106 | 57.160102975 | 0.423071 / 0.15887426 | 0.04544075 / 0.07597511 |
| `seed_271828_rec_6` | 56.9539065 | 0.508823 / 0.17436796 | 0.04305963 / 0.07543644 | 57.1524543 | 0.4617469 / 0.16428206 | 0.05005353 / 0.07951166 |

The stable median is the median of the six allocation-level medians; the
stable p95 is the median of the six allocation-level p95 values.

### 7.6 Allocation-paired comparison

Delta is `rec_19 - rec_6`; negative means `rec_19` is faster.

| Endpoint | Six paired differences ms | Median paired difference ms | Descriptive bootstrap 95% CI ms | Exact shifted sign-flip | All six beyond tolerance | Effective classification |
| --- | --- | ---: | --- | --- | --- | --- |
| Median | `[-0.05696375, +0.05146325, +0.00111250, -0.00832775, -0.02863525, +0.00874975]` | -0.003607625 | [-0.0427995, +0.0301065] | no direction; both one-sided p=1.0 | no, either direction | `no_stable_direction_descriptive_practical_equivalence` |
| p95 | `[+0.01441390, +0.04969540, +0.01101950, +0.00088345, +0.01107205, +0.00184650]` | +0.011045775 | [+0.001364975, +0.03205465] | no direction; both one-sided p=1.0 | no, either direction | `no_stable_direction_descriptive_practical_equivalence` |

Both descriptive intervals lie completely inside
`[-0.73553775, +0.73553775] ms`. The exact directional gate fails, and no
individual allocation difference crosses the practical boundary. Thus:

> Latency mode selected the highest-accuracy member of the
> equivalent-fastest cohort satisfying 90% retained accuracy.

`rec_19` must not be described as measurably fastest. It is the
higher-accuracy member of a validated practically equivalent fastest cohort.
The matched results do not invoke the selector, replace objectives, or
change the frozen winner. No tied-cohort algorithm change is required by
this result.

## 8. Lower-latency opportunity follow-up

### 8.1 Evidence motivating the search

The sealed archive contains 3-encoder/3-decoder candidates near 52 ms, but
they fail the frozen existing-archive opportunity floor:

| Candidate | mAP50 | Accuracy retained | Selection-time median | Prior stable median | 90%-feasible |
| --- | ---: | ---: | ---: | ---: | :---: |
| `seed_271828_rec_15` | 0.5606606395568864 | 85.5430% | 52.0782885 ms | 52.298003 ms | no |
| `seed_271828_rec_3` | 0.5398520557657904 | 82.3681% | 52.04909275 ms | 52.28649325 ms | no |

Therefore, approximately 52 ms has **not** been demonstrated at 90% retained
accuracy. The current archive satisfies the new policy through `rec_19`
(`57.146624 ms` selection-time median, `57.089795375 ms` in the earlier
global-front campaign, and `56.921768375 ms` in the completed 90%-cohort
campaign). The preregistered opportunity question remained anchored to the
approximately-57.09-ms evidence available when the search was frozen. The
completed follow-up below tested whether the accuracy gap of a lower-latency
architecture could be closed without changing that floor.

### 8.2 Frozen search design

Manifest: `dino_low_latency_followup_90pct_20260728_v1`

Result status: **complete**.

| Field | Frozen value |
| --- | --- |
| Algorithm | Bayesian |
| New search seeds | `409976740`, `1455024938`, `1415367367` |
| Recommendations per seed | 20 |
| Total new-candidate budget | 60 |
| Training seed | 1234 |
| Training epochs | 10 |
| Execution | three concurrent controllers; each trial is one SQSH-backed one-node/eight-GPU job |
| New-candidate generation population | new candidates only |
| Final selection population | sealed 60 plus every valid new candidate |
| Latency retention | relative 0.90 to the final union accuracy winner |
| Multi-objective minimum accuracy | unset |
| Latency tolerance | `0.73553775 ms` |

The supported search domain was frozen before results:

```yaml
model.enc_layers: [3, 4, 5, 6]
model.dec_layers: [3, 4, 5, 6]
train.optim.lr: [0.00001, 0.0005]        # continuous
train.optim.weight_decay: [0.00001, 0.001] # continuous
```

No result-dependent categorical weighting or narrowing is used. Encoder and
decoder depths remain algorithm-generated; no 3/3 candidate is manually
injected.

Seeds are deterministically derived by:

```text
int.from_bytes(SHA256(UTF8(material)).digest()[:4], "big") & 0x7fffffff
```

for `dino-low-latency-followup-v1:0`, `:1`, and `:2`.

Two distinct thresholds are recorded without conflating them:

- fixed opportunity question:
  `mAP50 >= 0.589872445081493`, relative to the sealed archive winner;
- final production policy:
  recompute `0.90 * highest valid accuracy` over the sealed-old-plus-valid-new
  union.

The manifest prohibits manual candidate injection, post-result range/seed/
budget/threshold changes, winner override, matched-measurement feedback, and
multi-objective inheritance of latency retention. Every produced checkpoint
and trial artifact is retained.

### 8.3 Completed follow-up result

The three controllers completed automatically. The running process loaded
AutoML commit `e2b7b91a1cad2772f91f85f9b9e829aedda0d1a1` and production
selector SHA-256
`c06c690f5600ead366f27bb3d4688b9e0b0e9ab463514ee6ea245962b06c919a`.
The later routing hardening at
`330dbc61efbebf6158e7dcecc59aea704039737b` is behavior-equivalent for
this manifest because external accuracy-reference overrides are null. It
was not retroactively substituted into the live controller provenance.

#### Budget and validity accounting

| Population | Expected records | Observed records | Successful finite objectives | Terminal failures |
| --- | ---: | ---: | ---: | ---: |
| Sealed archive | 60 | 60 | 60 | 0 |
| New follow-up | 60 | 60 | 59 | 1 |
| Final union | 120 | 120 | 119 | 1 |

The budget is 60 algorithm-generated recommendations—20 for each frozen
seed—not 60 successful objective vectors.

| Seed | Records | Successful | Failed | Seed-archive whole-file SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| `409976740` | 20 | 20 | 0 | `1e6ecb1c306961c277b1ee0b5062fda5092423331fc6379511eb2c1986fdfe77` |
| `1455024938` | 20 | 19 | 1 | `6a92dc3b5189a5edcc9a92e89e1596f2982a9d6f3efd282ca2ad5b316a069606` |
| `1415367367` | 20 | 20 | 0 | `b7a4b6a3742b173346910e6474993267a1857eafecee83599056d83826ff7351` |

`seed_1455024938_rec_15` was algorithm-generated with encoder/decoder
depth `5/3`, learning rate `0.00037640328330991744`, and weight decay
`0.00036075042297128654`. It failed before training-job creation because
SLURM did not provide a stable recoverable cluster identity. It remains an
immutable `training_or_measurement_failure` record with no train job ID,
attempt, checkpoint, or objective vector. The optimizer skipped the unusable
observation and advanced to recommendation 16. No retry, `rec_20`,
replacement recommendation, manual candidate, or synthetic failure metric
was introduced.

#### Complete new 90%-feasible population

The fixed opportunity floor and the final-union production floor are both
`0.589872445081493` because the union accuracy winner remains
`seed_271828_rec_18`. Exactly 17 of the 59 successful new candidates meet
that floor:

| Candidate | Encoder/decoder | mAP50 | Accuracy retained | Selection-time median ms |
| --- | --- | ---: | ---: | ---: |
| `seed_1455024938_rec_16` | 4/4 | 0.590593075575994 | 90.109950% | 61.272711500 |
| `seed_409976740_rec_3` | 5/3 | 0.612625369511345 | 93.471536% | 61.555876750 |
| `seed_1415367367_rec_2` | 4/4 | 0.592276754021086 | 90.366838% | 61.645349750 |
| `seed_1455024938_rec_8` | 5/3 | 0.611601724153127 | 93.315353% | 61.736783000 |
| `seed_1455024938_rec_7` | 5/3 | 0.621505697997251 | 94.826455% | 61.781110000 |
| `seed_409976740_rec_12` | 5/3 | 0.604333286947928 | 92.206368% | 61.878712000 |
| `seed_1455024938_rec_4` | 5/3 | 0.598599955437887 | 91.331603% | 61.952549750 |
| `seed_1415367367_rec_19` | 4/5 | 0.593946672299549 | 90.621627% | 65.646623500 |
| `seed_409976740_rec_6` | 5/4 | 0.617311049464411 | 94.186455% | 66.009539750 |
| `seed_409976740_rec_4` | 6/3 | 0.632757635564474 | 96.543223% | 66.246603000 |
| `seed_409976740_rec_7` | 6/3 | 0.622604921875076 | 94.994169% | 66.443895000 |
| `seed_409976740_rec_11` | 6/3 | 0.629088710275341 | 95.983436% | 66.560352000 |
| `seed_1415367367_rec_8` | 5/5 | 0.596650561606045 | 91.034174% | 70.264918250 |
| `seed_1415367367_rec_14` | 6/4 | 0.643816403937537 | 98.230519% | 70.638092500 |
| `seed_1455024938_rec_14` | 6/4 | 0.603433964804425 | 92.069154% | 70.941328250 |
| `seed_409976740_rec_8` | 6/4 | 0.643879562555165 | 98.240155% | 70.953235500 |
| `seed_409976740_rec_10` | 6/5 | 0.622543667050247 | 94.984823% | 75.038002250 |

The complete 120-row audit—including all successful metrics, full
hyperparameters, runtime provenance, Pareto ranks, dominance relations, and
the immutable failed row—is retained in
`runtime/low_latency_followup_v1/expanded_candidate_table.json` and its CSV
projection.

#### Lower-latency opportunity answer

The fastest new accuracy-feasible candidate is
`seed_1455024938_rec_16`:

| Field | Value |
| --- | --- |
| Encoder/decoder | 4/4 |
| Learning rate / weight decay | `0.0003166398901931195` / `0.00048458989731432127` |
| mAP50 | `0.5905930755759942` |
| Accuracy retained | `90.1099504563%` |
| Margin above floor | `0.0007206304945012` |
| Median / p95 | `61.2727115` / `61.47020945 ms` |
| Median 95% CI | `[61.242067500000005, 61.288489068749996] ms` |
| Delta from `rec_19` selection-time median | `+4.1260875 ms` slower |
| Delta from preregistered `57.089795375 ms` reference | `+4.182916125 ms` slower |
| Pareto rank / dominated by | 2 / `seed_271828_rec_19`, `seed_271828_rec_6` |

It lies `3.39054975 ms` beyond the raw-minimum-plus-tolerance cohort
boundary. Even its lower confidence bound is `3.35990575 ms` beyond that
boundary, so its selection-time uncertainty cannot plausibly move it into
the equivalent-fastest cohort.

The strongest new 3/3 candidate near 52 ms is
`seed_1455024938_rec_17`: mAP50 `0.5780320693410467`, retained accuracy
`88.1934504222%`, and median `52.23281375 ms`. It improves mAP50 over the
old best 3/3 point by `0.0173714297841603`, but misses the frozen floor by
`0.0118403757404463`.

Therefore:

> No qualifying approximately-52-ms candidate was found among 59 successful
> new evaluations; one of the 60 frozen algorithmic recommendations failed
> before submission.

The finite search does not prove that approximately 52 ms at 90% retained
accuracy is impossible. It establishes only that this preregistered,
algorithm-generated search did not demonstrate it.

#### Final union selection and Pareto geometry

| Mode | Candidate | mAP50 | Selection-time median | Eligibility / Pareto status | Exact selection basis |
| --- | --- | ---: | ---: | --- | --- |
| Accuracy | `seed_271828_rec_18` | 0.6554138278683255 | 66.23099475 ms | all valid; global rank zero | highest valid accuracy |
| Latency | `seed_271828_rec_19` | 0.6175134981289873 | 57.146624 ms | 34 candidates meet 90%; equivalent-fastest cohort is `rec_19`, `rec_6` | highest-accuracy member of the equivalent-fastest feasible cohort |
| Multi-objective | `seed_271828_rec_19` | 0.6175134981289873 | 57.146624 ms | independently eligible global rank zero | minimum normalized augmented-Chebyshev regret |

The final global rank-zero front has six points:

| Candidate | Encoder/decoder | mAP50 | Median ms | 90%-feasible | Normalized accuracy regret | Normalized latency regret | Compromise score |
| --- | --- | ---: | ---: | :---: | ---: | ---: | ---: |
| `seed_271828_rec_3` | 3/3 | 0.5398520557657904 | 52.04909275 | no | 1.0000000000 | 0.0000000000 | 0.5000005000 |
| `seed_271828_rec_15` | 3/3 | 0.5606606395568864 | 52.07828850 | no | 0.8199354041 | 0.0020586625 | 0.4099681131 |
| `seed_1455024938_rec_17` | 3/3 | 0.5780320693410467 | 52.23281375 | no | 0.6696138102 | 0.0129546093 | 0.3348072464 |
| `seed_271828_rec_19` | 4/3 | 0.6175134981289873 | 57.14662400 | yes | 0.3279659791 | 0.3594391817 | **0.1797199346** |
| `seed_1455024938_rec_7` | 5/3 | 0.6215056979972510 | 61.78111000 | yes | 0.2934199541 | 0.6862279298 | 0.3431144547 |
| `seed_271828_rec_18` | 6/3 | 0.6554138278683255 | 66.23099475 | yes | 0.0000000000 | 1.0000000000 | 0.5000005000 |

The two new global-front points are `seed_1455024938_rec_17` and
`seed_1455024938_rec_7`. Front-relative bounds are:

- accuracy ideal/nadir/range:
  `0.6554138278683255` / `0.5398520557657904` /
  `0.11556177210253504`;
- latency ideal/nadir/range:
  `52.04909275` / `66.23099475000001` /
  `14.181902000000008 ms`.

Archive, reverse, and candidate-ID order all select the same winners;
the order-invariance signature is
`88eb3d89b00d1c5668d766ed80e620f5f5a0f3ffc263fab7aa8885fa27574ef9`.
Every manual-injection, reordering, reselection, matched-feedback, and winner-
override flag remains false.

No new matched campaign is required for the latency-policy or opportunity
conclusion. The new fastest feasible candidate is more than 4 ms slower and
dominated; the new 3/3 front point is accuracy-infeasible; and the other new
front point is `4.634486 ms` slower than `rec_19`. If a separate blanket
audit requires matched remeasurement of every point on the post-follow-up
global front, only the two new front points need an additional
selection-isolated completeness campaign; such measurements must not feed
selection or change the frozen winners.

## 9. Test evidence

The complete production suite passed:

```text
414 passed, 1 skipped, 1 warning in 4.76s
```

The complete DINO phase-two experiment suite also passed:

```text
355 passed in 5.40s
```

The suite covers:

- 90% relative-threshold calculation from the accuracy winner;
- feasible-population derivation;
- raw-minimum latency selection;
- hard equivalent-fastest-cohort behavior;
- accuracy tie-breaking only within that cohort;
- direct and tied exact output reasons;
- archive-order invariance;
- empty feasible populations;
- finite `(0, 1]` configuration validation and legacy-conflict rejection;
- independence from multi-objective weights and constraints;
- replay integrity;
- matched projection, recovery binding, schedule, launch, aggregation, and
  selection-isolation contracts;
- checkpoint-recovery fail-closed behavior;
- follow-up manifest, source, union-selection, and no-manual-injection
  contracts;
- immutable historical-selector pins tested from their recorded git commit
  rather than incorrectly requiring the current hardened selector to remain
  byte-identical.

The focused selector, replay, and wheel suite also passed:

```text
121 passed, 1 warning
```

The production and focused runs each emit the same sklearn Gaussian-process
convergence warning; it is not a test failure. The experiment-only suite
emits no warnings.

Exact production-suite command:

```bash
cd /localhome/local-rarunachalam/tao-automl
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  pytest -q
```

The venv path must lead `PATH` because `tests/test_wheel.py` intentionally
spawns `pip` as a subprocess.

Exact complete experiment-suite command:

```bash
cd /localhome/local-rarunachalam/tao-automl
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider -q \
  experiments/dino_moo_phase2_20260728
```

Exact focused selector/replay/wheel command:

```bash
cd /localhome/local-rarunachalam/tao-automl
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  -m pytest -q \
  tests/test_multi_objective_selection.py \
  experiments/dino_moo_phase2_20260728/test_replay_latency_90_policy.py \
  tests/test_wheel.py
```

## 10. Artifact integrity

| Artifact | Whole-file SHA-256 | Canonical/internal SHA-256 |
| --- | --- | --- |
| 90% policy profile | `f6e56ff8d61c91654a13c9759d7cc63f371ed66a9e958bb90ae56cba5112739e` | `8074b4db7b35f0247d7ebb395b15690a232cbb2376891bfd38e334b7af1f4353` |
| Archive replay v1 (matched binding) | `7441053780c8b400a239a71f5e50b0b37813f8968a1284567fccd3dc857d33a9` | `3d37f17d30d1bbcaac013d49ee80b4c8afdd85964fa08ae91ba23e69720bf869` |
| Archive replay v2 (active-mode routing) | `4c7d34bc485789dae9ab5c0435ef9a0a655e7f2f9ab293067a74fff0028964f0` | `01ac3f09dafa37cae35e96c125b43c44453cf2d2203f24b14b2cf83cf7a8171b` |
| Matched execution projection | `c2ab8e650da99dc243b943f0ba2a89089aaa1e72331a254d6f4ee11793cd8090` | `402ad679b8e4705a56a2918b23f5ef102d40f1eeff2114e40cab129db4d9deea` |
| `rec_6` recovery preregistration | `05802e9d0d6e49c2885e2e2e73f01cb053b50d294d8ca284507fb888e2c95f4d` | recorded in preregistration |
| `rec_6` recovery evidence | `acc0c46f9a58121a7e77e266ab0fe6611082809fe62ef641eed6299a287c39d0` | `de528df5b0d93df7fdd22575f244bb3dd767762ea71303951f0a403d14b0400b` |
| Matched submission ledger at cutoff | `2c50632ae9eb5099cb6068a94d101180c998f21ab8d74fdba549956d0d103172` | `e8882169ed0e5e2afffb2aa7361fb991231dd76425fb3c2c68a03ce15f4cd254` |
| Complete matched-latency analysis | `04f71d96cabd3ad717482fd243d74e4b4f9ebea8d0da4d7737e1f270483d77c4` | `aef1e6f3b506c5c9a05f3dde643f9406df5f4574c23588221760c476374b337d` |
| Follow-up search manifest | `1f0e25f0230fa84f702f3ff60dac91bce0fa284e90b4faf950b27e9cb2cbed7a` | `c1b6b5c5d704ec2a4eb5fc792b7666470e20d5245f26301a67afeae1ba684280` |
| Follow-up runner | `003704d292d011d58081d8309af0ddc534aeefccdf2c5a653079ae5d97f077fe` | n/a |
| Production selector at v1 matched binding | `c06c690f5600ead366f27bb3d4688b9e0b0e9ab463514ee6ea245962b06c919a` | n/a |
| Production selector at routing-explicit v2 | `5fbdfe0d754ed3a3c4662fb6640afaf0cead9d99664d2a3061ed3c426d85037c` | n/a |
| TAO 7.0.1 PyTorch SQSH | `88ba75e3a8eb9524fc0dbf026f2ea5da2c68696ae8d918b0afde5e0384ca641e` | n/a |

Completed follow-up artifacts:

| Artifact | Whole-file SHA-256 |
| --- | --- |
| Dry run | `c0226b8450fc54074b3ada1452486e9e9a7f6f24bfc564c4e739d071a8bc3f11` |
| Runtime contract v2 | `668ec18026bd241038df7a088d8e8923bfc1caf0b0064cc8c2570da4909d685d` |
| Hardware contract | `75888145e590203cdaec1197db1ee434c47fe88056260538ec015cefa543173b` |
| Input contract | `a4816b84ec3b7d53b073c41986c5cdae067a1b7c4ff5bb58158d403c2a463e53` |
| Seed-process status | `3917efe29d98c16bca8eebb54c5af56808c24901fa8ae368073e4ea0c1d8af44` |
| Complete candidate table JSON | `fe423a61b3efa9222b4b25b6da0f303c3e84eb4e5cc1864aad09c5b6a311479c` |
| Complete candidate table CSV | `a52003ce292c63f5904f1a632d986f40e22403dda7a667748b0f5a0a82167e03` |
| Final combined selection | `d94a95a8a5cdf5116465caa7944912a5a2fb1de37983aa5032516411ae2c773c` |
| Final integrity audit | `4cba337281f131a8859a7f5da7d7ca67a888fd7dda6b1ba94b208879f5a36b8e` |
| Completion manifest | `2a6d3ac8f7b75eed8ef01b4efd3949e3f94c30254d7021706614dd731145661d` |
| Controller log | `1d43897a4552bb55fecb05cfdcbf01b574f7572931f141d15aca66320d21a476` |
| Zero-submission first-attempt log | `4a2bef1bcdbf436c2d2a7280be5959e2f6bfbd915d2ac61f62204db668cf97e5` |

The runtime contract internal SHA-256 is
`a354b888e19e03fa27d231130349f03725c89b770ae556ae2ab861157aa9f3b0`.
The two human-readable logs are intentionally stored under
`latency_90_policy/logs/`, outside the frozen runtime root, so future
read-only resume validation sees only contracted entries.

| Seed | Events whole-file SHA-256 | Result whole-file SHA-256 | Archive whole-file SHA-256 | Archive internal SHA-256 |
| ---: | --- | --- | --- | --- |
| `409976740` | `3c309d5683e51751270fbb0f2ad40c04270d766b0e1eea23796f536139a0d8fa` | `62bd2b7b819a2739e4f9038cb6e8bcb11b89e4e4bba74c173f9552ba5a054848` | `1e6ecb1c306961c277b1ee0b5062fda5092423331fc6379511eb2c1986fdfe77` | `3ff4b410e7987f659ef8abcd915792a9c0c60a2b4c85248f4069e5150519ac82` |
| `1455024938` | `525a21338488fbac1f67004da5a968cd3fbf6b7daf264b4a3513583cdac2bafa` | `4dee932af8c300629437edc8414d14adf58e2e44c9cf4cef1c0a7be7957d3ea1` | `6a92dc3b5189a5edcc9a92e89e1596f2982a9d6f3efd282ca2ad5b316a069606` | `9c88e795bfc31dbffa97e89e89a91530a129b135d276d374c17fd29736abfcee` |
| `1415367367` | `9f7e376d67af6fa950d5e59d4fa0b96bb93cc97bcbe6f651c5663dcf5f150b55` | `c87de5f6026b98a3d7d1e0c6cb9f38f192721aa7e08f01ea4dbace19e9d23b09` | `b7a4b6a3742b173346910e6474993267a1857eafecee83599056d83826ff7351` | `1f33c0c3a43162f21d24c05d6c5278c39cd7591556020077827e6dcb272d6174` |

Runtime:

- TAO SDK branch:
  `rarunachalam/pre-platform-sdk-removal-20260714`
- TAO SDK commit:
  `3d3e1adc1849493d29dc926cb99492417e3a9250`
- partition/account: `polar3` /
  `edgeai_tao-ptm_image-foundation-model-clip`
- secrets source: `/localhome/local-rarunachalam/.tao/config.env`;
  secret values are never recorded.

## 11. Exact reproduction and continuation commands

### 11.1 Replay v1: immutable matched-manifest binding

Reproduce the v1 replay from its exact implementation commit in a detached
temporary worktree:

```bash
REPLAY_V1_TREE=$(mktemp -d)
git -C /localhome/local-rarunachalam/tao-automl worktree add \
  --detach "$REPLAY_V1_TREE" \
  b1b25700ca478fa847cdaa402520be376d55a00b
cd "$REPLAY_V1_TREE"
PYTHONPATH="$REPLAY_V1_TREE/src" \
  /localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/replay_latency_90_policy.py \
  --check
```

Expected replay summary:

```text
status=verified
selected_latency_candidate_id=seed_271828_rec_19
feasible_candidate_count=17
whole_file_sha256=7441053780c8b400a239a71f5e50b0b37813f8968a1284567fccd3dc857d33a9
```

The detached worktree may be removed after verification with:

```bash
git -C /localhome/local-rarunachalam/tao-automl worktree remove \
  "$REPLAY_V1_TREE"
```

### 11.2 Replay v2: active latency winner routing

Verify the finalized routing-explicit v2 artifact with:

```bash
cd /localhome/local-rarunachalam/tao-automl
PYTHONPATH=src \
  /localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/replay_latency_90_policy.py \
  --check
```

Expected v2 evidence:

```text
whole_file_sha256=4c7d34bc485789dae9ab5c0435ef9a0a655e7f2f9ab293067a74fff0028964f0
canonical_payload_sha256=01ac3f09dafa37cae35e96c125b43c44453cf2d2203f24b14b2cf83cf7a8171b
selection_mode=latency
winner_route_matches_latency_selection=true
selected_latency_candidate_id=seed_271828_rec_19
```

### 11.3 Completed `rec_6` recovery workflow

```bash
cd /localhome/local-rarunachalam/tao-automl
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/rec6_checkpoint_recovery.py \
  --dry-run

/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/rec6_checkpoint_recovery.py \
  --launch \
  --acknowledgement USER_AUTHORIZED_VALIDATION_ONLY_REC6_CHECKPOINT_RECOVERY_20260728

/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/rec6_checkpoint_recovery.py \
  --status

/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/rec6_checkpoint_recovery.py \
  --finalize
```

### 11.4 Matched validation

Projection verification/dry run:

```bash
cd /localhome/local-rarunachalam/tao-automl
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_90_policy_matched_launcher.py \
  --projection-sha256 c2ab8e650da99dc243b943f0ba2a89089aaa1e72331a254d6f4ee11793cd8090 \
  --checkpoint-recovery-evidence-sha256 acc0c46f9a58121a7e77e266ab0fe6611082809fe62ef641eed6299a287c39d0 \
  --verify-remote
```

Submitted launch:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_90_policy_matched_launcher.py \
  --launch \
  --projection-sha256 c2ab8e650da99dc243b943f0ba2a89089aaa1e72331a254d6f4ee11793cd8090 \
  --checkpoint-recovery-evidence-sha256 acc0c46f9a58121a7e77e266ab0fe6611082809fe62ef641eed6299a287c39d0 \
  --verify-remote \
  --acknowledgement USER_AUTHORIZED_DINO_LATENCY_90_MATCHED_6X8GPU_VALIDATION_20260728
```

Reproduce the completed terminal aggregation:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_90_policy_matched_aggregator.py \
  --projection-sha256 c2ab8e650da99dc243b943f0ba2a89089aaa1e72331a254d6f4ee11793cd8090 \
  --checkpoint-recovery-evidence-sha256 acc0c46f9a58121a7e77e266ab0fe6611082809fe62ef641eed6299a287c39d0 \
  --submission-ledger-sha256 2c50632ae9eb5099cb6068a94d101180c998f21ab8d74fdba549956d0d103172
```

Expected analysis artifact:

```text
whole_file_sha256=04f71d96cabd3ad717482fd243d74e4b4f9ebea8d0da4d7737e1f270483d77c4
canonical_payload_sha256=aef1e6f3b506c5c9a05f3dde643f9406df5f4574c23588221760c476374b337d
status=complete
complete_cells=12
median_effective_classification=no_stable_direction_descriptive_practical_equivalence
p95_effective_classification=no_stable_direction_descriptive_practical_equivalence
```

### 11.5 Lower-latency follow-up

Dry run and remote verification:

```bash
cd /localhome/local-rarunachalam/tao-automl
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_90_policy/low_latency_followup_runner.py \
  --dry-run \
  --verify-remote \
  --manifest-file-sha256 1f0e25f0230fa84f702f3ff60dac91bce0fa284e90b4faf950b27e9cb2cbed7a
```

Launch/resume command:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_90_policy/low_latency_followup_runner.py \
  --launch \
  --verify-remote \
  --resume \
  --manifest-file-sha256 1f0e25f0230fa84f702f3ff60dac91bce0fa284e90b4faf950b27e9cb2cbed7a \
  --acknowledgement USER_AUTHORIZED_3X8GPU_SLURM_DINO_LOW_LATENCY_FOLLOWUP_20260728
```

The successful launch automatically wrote all three seed archives and then
performed the final union analysis. No `--combine-only` command was run after
`expanded_completion.json` existed, because doing so would rewrite
timestamped evidence. The following is a recovery command only when all
three seed archives exist but the automatic completion artifact is absent:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_90_policy/low_latency_followup_runner.py \
  --combine-only \
  --manifest-file-sha256 1f0e25f0230fa84f702f3ff60dac91bce0fa284e90b4faf950b27e9cb2cbed7a
```

Expected automatic completion identities:

```text
candidate_table_json_sha256=fe423a61b3efa9222b4b25b6da0f303c3e84eb4e5cc1864aad09c5b6a311479c
candidate_table_csv_sha256=a52003ce292c63f5904f1a632d986f40e22403dda7a667748b0f5a0a82167e03
combined_selection_sha256=d94a95a8a5cdf5116465caa7944912a5a2fb1de37983aa5032516411ae2c773c
integrity_audit_sha256=4cba337281f131a8859a7f5da7d7ca67a888fd7dda6b1ba94b208879f5a36b8e
completion_sha256=2a6d3ac8f7b75eed8ef01b4efd3949e3f94c30254d7021706614dd731145661d
union_records=120
union_successful_candidates=119
```

## 12. Final conclusion

The revised mode-specific hypothesis is **fully supported**:

1. Accuracy mode selects the highest-valid-accuracy candidate,
   `seed_271828_rec_18` at mAP50 `0.6554138278683255`.
2. Latency mode derives `0.589872445081493` from 90% of that winner,
   forms the raw-minimum-anchored equivalent-fastest cohort
   `seed_271828_rec_19`/`seed_271828_rec_6`, and selects the cohort's
   higher-accuracy member. Matched evidence establishes no stable median or
   p95 direction and descriptive practical equivalence; `rec_19` is not
   described as measurably fastest.
3. Multi-objective mode independently evaluates all valid candidates,
   selects global-rank-zero `seed_271828_rec_19`, and uses the minimum
   front-normalized augmented-Chebyshev score. It does not inherit the
   latency retention floor.
4. Latency and multi-objective legitimately share a winner through different
   policy paths. No threshold, tolerance, range, candidate, objective,
   selection-time value, or winner was manually changed or overridden.

The existing sealed archive already satisfies the desired 90%-retention
product profile. The additional search did not find a better
accuracy-feasible latency point: its fastest qualifying result is
`61.2727115 ms`, more than 4 ms slower than `rec_19`. The strongest new
approximately-52-ms point retains only 88.1935% accuracy. Thus,
approximately 52 ms at 90% retained accuracy remains **not demonstrated**;
the finite negative result must not be misreported as proof of
impossibility.

No further blind expansion of the same DINO depth/optimizer domain is needed
to validate the latency policy. If sub-57-ms DINO remains a separate product
goal, the next search should be independently preregistered around a
materially new accuracy-recovery lever, training fidelity, or supported
architecture dimension—not a post-result range or threshold adjustment.

DINO is ready to serve as the completed reference validation for AutoML mode
semantics and the selection/measurement isolation framework before extending
that framework to another model and dataset. A separate matched campaign for
the two newly added global-front points is necessary only if a blanket audit
requires every point on the post-follow-up global front to be remeasured; it
is not required for the completed latency-policy conclusion and must not feed
selection.
