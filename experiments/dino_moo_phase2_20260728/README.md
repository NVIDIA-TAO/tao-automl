# DINO multi-objective phase-2 latency validation

This directory contains the completed DINO-only phase-2 investigation and its
historical phases. The current merge authority is
[`latency_90_policy/latency_90_policy_report.md`](latency_90_policy/latency_90_policy_report.md).
It records the explicit 90%-retention replay, matched `rec_19`/`rec_6`
validation, preregistered follow-up search, and final 120-row union. The
98%-retention material below remains historical evidence for the earlier
phase. No matched-validation measurement changes an archive objective or
feeds AutoML reselection.

## Current merge result

| Mode | Candidate | Policy basis |
| --- | --- | --- |
| Accuracy | `seed_271828_rec_18` | highest valid accuracy |
| Latency, explicit 90% profile | `seed_271828_rec_19` | highest-accuracy member of the raw-minimum-anchored equivalent-fastest feasible cohort |
| Multi-objective | `seed_271828_rec_19` | independently eligible Pareto-rank-zero augmented-Chebyshev compromise |

Latency and multi-objective independently select `rec_19` for different
algorithmic reasons. The repository-wide implicit latency-retention default
remains 98%; only the DINO validation profile explicitly selects 90%.

Final committed artifacts:

- [`runtime/low_latency_followup_v1/expanded_candidate_table.json`](runtime/low_latency_followup_v1/expanded_candidate_table.json)
- [`runtime/low_latency_followup_v1/expanded_combined_selection.json`](runtime/low_latency_followup_v1/expanded_combined_selection.json)
- [`runtime/low_latency_followup_v1/expanded_integrity_audit.json`](runtime/low_latency_followup_v1/expanded_integrity_audit.json)
- [`runtime/low_latency_followup_v1/expanded_completion.json`](runtime/low_latency_followup_v1/expanded_completion.json)

## Historical 98% phase status

The expanded search completed all 60 candidates (20 recommendations for each
of seeds `314159`, `271828`, and `161803`) with no failed or manually injected
record. The production selectors independently chose
`seed_271828_rec_18` for both accuracy and 98%-retained latency mode and
`seed_271828_rec_19` for unconstrained multi-objective mode. Its global
rank-zero front contains exactly four candidates:
`seed_271828_rec_15`, `seed_271828_rec_18`,
`seed_271828_rec_19`, and `seed_271828_rec_3`. This expanded front is distinct
from both the historical six-candidate front and the four-candidate
98%-accuracy-feasible latency cohort discussed below.

### Expanded global-front study

All six post-front jobs finished as `Complete / COMPLETED / 0:0` on six
distinct A100 nodes. Every job measured every front candidate, yielding 24/24
valid candidate-allocation cells. The stable aggregates across the six
matched allocations are:

| Expanded-front candidate | Median latency (ms) | p95 latency (ms) |
| --- | ---: | ---: |
| `seed_271828_rec_3` | 52.286493 | 52.519055 |
| `seed_271828_rec_15` | 52.298003 | 52.570566 |
| `seed_271828_rec_19` | 57.089795 | 57.318384 |
| `seed_271828_rec_18` | 66.496668 | 66.765354 |

Under the effective preregistered pairwise rule and `0.73553775 ms` practical
tolerance, both `rec_3` and `rec_15` are stably faster than `rec_19` and
`rec_18`, and `rec_19` is stably faster than `rec_18`, for both median and
p95. `rec_3` versus `rec_15` has no effective directional claim; its preserved
bootstrap interval lies within the tolerance band. These are pairwise claims,
not a simultaneous total order.

These global-front measurements validate `rec_19` as the
algorithm-selected, Pareto-nondominated, allocation-stable geometric
compromise. They do not require the compromise to lie between the actual
accuracy and constrained-latency winners when those modes legitimately share
`rec_18`.

### Separate 98%-accuracy-feasible latency-cohort study

The latency-cohort manifest independently replayed the pinned production
selector and froze exactly the four candidates satisfying
`mAP50 >= 0.98 * 0.6554138278683255 = 0.6423055513109589`:

- `seed_161803_rec_14`
- `seed_271828_rec_16`
- `seed_271828_rec_18`
- `seed_314159_rec_12`

This is a retained-accuracy cohort, not another global Pareto front. Only
`seed_271828_rec_18` is common to the expanded global rank-zero front.
Candidate membership, execution order, and the six-allocation Williams
schedule were frozen before matched measurement and did not use measured
latency values.

All six one-node/eight-A100 jobs completed as
`Complete / COMPLETED / 0:0` on six distinct nodes. Every candidate was
measured in every allocation: 24/24 cells passed the frozen provenance and
quality gates. Each cell contains five rounds of 100 timed requests on each of
eight replicas after 50 warm-ups, or 4,000 raw samples per
candidate/allocation and 96,000 samples in total.

| Allocation | TAO job | SLURM job | Node | Terminal status |
| --- | --- | ---: | --- | --- |
| `00` | `32862ad8-e015-46a1-892d-e399ae71a4a5` | 31004302 | `batch-block7-00886` | `Complete / COMPLETED / 0:0` |
| `01` | `ee1f5f87-ef4a-4e4a-b13c-fdbf3a90d792` | 31004303 | `batch-block7-01632` | `Complete / COMPLETED / 0:0` |
| `02` | `eecf47a7-aed4-4d5b-8921-7e50f644a5a6` | 31004306 | `batch-block7-00064` | `Complete / COMPLETED / 0:0` |
| `03` | `a789904e-a852-406b-b392-5b2c733dda3c` | 31004308 | `batch-block7-00288` | `Complete / COMPLETED / 0:0` |
| `04` | `2e7d214e-b5a3-44d5-b402-88a25db016d5` | 31004316 | `batch-block7-01760` | `Complete / COMPLETED / 0:0` |
| `05` | `2d88ee5f-1fd0-4644-b446-fa75c205e8a2` | 31004322 | `batch-block7-03295` | `Complete / COMPLETED / 0:0` |

For all six unordered candidate pairs, the allocation-matched descriptive
bootstrap 95% intervals for both median and p95 lie wholly inside
`[-0.73553775, +0.73553775] ms`. No pair passes the effective directional
rule, and there are no median or p95 stable-direction edges. The result is
therefore **Outcome B**: the complete cohort is descriptively practically
equivalent, and latency mode selected its highest-accuracy member.

| Candidate | mAP50 | Stable median | Stable p95 | Feasible at 98% | Pairwise latency conclusion | Latency-mode interpretation |
| --------- | ----: | ------------: | ---------: | :-------------: | --------------------------- | --------------------------- |
| `seed_161803_rec_14` | 0.6503731412 | 66.567490250 ms | 66.884389500 ms | Yes | Descriptively practically equivalent to the other three; no stable median or p95 direction | Equivalent-fastest cohort member; lower accuracy than `rec_18` |
| `seed_271828_rec_16` | 0.6544218576 | 66.610895375 ms | 67.016615025 ms | Yes | Descriptively practically equivalent to the other three; no stable median or p95 direction | Equivalent-fastest cohort member; lower accuracy than `rec_18` |
| `seed_271828_rec_18` | 0.6554138279 | 66.615607500 ms | 66.904386000 ms | Yes | Descriptively practically equivalent to the other three; no stable median or p95 direction | Frozen winner and highest-accuracy member of the equivalent-fastest cohort |
| `seed_314159_rec_12` | 0.6517250365 | 66.639216500 ms | 66.878630975 ms | Yes | Descriptively practically equivalent to the other three; no stable median or p95 direction | Equivalent-fastest cohort member; lower accuracy than `rec_18` |

The product semantics validated by the two distinct matched studies are:

- **Accuracy mode:** select the highest-accuracy valid candidate; use latency
  only as a deterministic tie-break within the configured accuracy tolerance.
- **Latency mode:** select the highest-accuracy member of the
  raw-minimum-anchored equivalent-fastest cohort satisfying the configured
  retained-accuracy constraint. The configured hard practical tolerance
  defines that cohort; confidence-interval overlap does not widen it.
- **Multi-objective mode:** select a deterministic, normalized,
  Pareto-nondominated balance over its independently eligible front. It does
  not silently inherit latency mode's retained-accuracy constraint.

| Mode | Candidate | Selection-time reason | Matched-validation conclusion | Algorithm change required |
| ---- | --------- | --------------------- | ----------------------------- | ------------------------- |
| Accuracy | `seed_271828_rec_18` | Highest valid mAP50, `0.6554138279` | Preserved; matched latency is not an accuracy reselection input | No |
| Latency, 98% retention | `seed_271828_rec_18` | All four feasible candidates were within `0.73553775 ms` of the raw-minimum anchor; the higher-accuracy tie-break selected `rec_18` | Outcome B: all four are descriptively practically equivalent, so `rec_18` remains the highest-accuracy member of the equivalent-fastest cohort | No |
| Multi-objective | `seed_271828_rec_19` | Minimum deterministic normalized augmented-Chebyshev compromise score on the independently eligible global rank-zero front | Stable, Pareto-nondominated global-front compromise; it is distinct from the global accuracy and unconstrained-latency extremes | No |

The feasible-cohort measurements are validation-only. The following audit
values are all exactly `false`:

```text
selector_invoked_on_matched_measurements = false
selection_time_objectives_replaced = false
measurements_feed_selection = false
measurements_feed_reselection = false
algorithm_selected_candidate_overridden = false
```

The revised mode-specific hypothesis is **fully supported**: accuracy mode
returns the highest-accuracy valid candidate; latency mode returns the fastest
accuracy-feasible candidate or the highest-accuracy member of its equivalent
fastest cohort; multi-objective mode returns a deterministic, normalized,
Pareto-nondominated compromise over an independently eligible front; shared
winners are allowed; and no candidate was manually promoted or overridden.
The earlier requirement for three distinct mode winners is not a valid product
semantic.

DINO is ready to serve as the completed reference validation before applying
the same frozen-selection and matched-measurement framework to another model
and dataset. No multi-objective or latency-selector algorithm change is
recommended from this validation.

The feasible-cohort launch ledger is sealed at AutoML commit
`0ef3880bad0f3bdfa6a20a64bb76fd56f5b3899f`; its complete analysis,
read-only integrity verifier, and integrity audit are sealed at
`bb9bfad86ca18091532d196ad9a96ccb3ad116d0`. The runtime uses TAO SDK
commit `3d3e1adc1849493d29dc926cb99492417e3a9250`.

Committed authority artifacts and whole-file SHA256 values:

- `runtime/expanded_search_v2/expanded_combined_selection.json`:
  `78ab9d2fa83cc3abe9057d137c0b88f120158b6ad77268482d2c18f5a1533af1`
- `runtime/expanded_search_v2/expanded_candidate_table.json`:
  `5ba323d05d9ec8e3703e636f8b5e2975cc620eeec10df75ec6e792318dc2df03`
- `runtime/expanded_search_v2/expanded_integrity_audit.json`:
  `a11eeeaf77bd2f289c6363133882bb78c6889205d4cb9be5f0dacf79a1bea159`
- `post_front_matched_manifest.v1.json`:
  `d468d5d26f607b115c7c1732966f0ac98664fd232ce83abfa6becc0ce062b7b6`
- `runtime/post_front_matched/post_front_matched_analysis.json`:
  `150d66fd1648c458807bdce9871313b5b17a7a33c63564f34b86156e392094b9`
- `latency_feasible_matched_manifest.v1.json`: whole-file
  `f83be40f9e00bcd5fc62959bf5a732327353a91ca12ee2efc118325ded2d0db4`;
  internal
  `32c268f11742c26bdc47c61272f73a6d9f651317606e9102e75585bc7d2100e9`
- `latency_feasible_rec16_checkpoint_overlay.v2.json`: whole-file
  `d054df9923a717759786628177814d5d531d7af3bf136da7361a4882b83410aa`;
  internal
  `e2e6446e3dca7f52e7b2bac682c26bc5b711067b0a18dae71cfbb24f5a44402c`
- `rec16_checkpoint_recovery_evidence.v1.json`: whole-file
  `923284e51f9e41f6954a1615f683ea6221b62fce0f490c3d5252b7c0d67f3e56`;
  internal
  `01715fecc26c3f2b5085650f3bd17f62a502dfc6a85560f11cfa968d08ac9456`
- `runtime/latency_feasible_matched/launch_contract.v1.json`: whole-file
  `a9fdc853d5847b9a945a012c5c281877a0e8d70c5bec09b5caea31382639b84d`;
  internal
  `9a29cad7a9863b7a5b9c961d62a0761a2462422468499948ff840667a608d94f`
- `runtime/latency_feasible_matched/block_submissions.json`: whole-file
  `a5768627797d91f8c9ab88d5893bcf607705b83ce30a1ceef9d5bc20ce39fd81`;
  internal
  `0c0a69ca6203a88fac9f04aaf62bc7a7ed2644b18307c1664243af2d389ad168`
- `runtime/latency_feasible_matched/latency_feasible_matched_analysis.json`:
  whole-file
  `30505ee6fd1635ac4edba01f95136f527a2df9e54257675a1fb505bd0b422efd`;
  internal
  `8ff21a404357e59949dfde7c9889710e854b0e5721ef53f39f439e71478c7351`;
  raw-input inventory
  `0f8ed28335d68f7d6359f11dd8e617dc71b75f6a0b73460b31a9b9745391a0d5`
- `runtime/latency_feasible_matched/overlay_campaign_integrity_audit.v1.json`:
  whole-file
  `03784ce116783c12d6373ad4e5842974deae75898b37af08cf672354c1611e7b`;
  internal
  `03083c5c1f604c3e47eee2644e1ef932266610bec41cf19a2dbf78b5ae483ac4`

## 98%-feasible cohort reproducibility

The launch used the exact TAO 7.0.1 SQSH at
`/lustre/fsw/portfolios/edgeai/users/rarunachalam/nvcr.io_nvidia_tao_tao-toolkit_7.0.1-pyt.sqsh`,
one node and eight A100 GPUs per allocation, the pinned dataset input digest,
batch size one, FP32 with TF32 disabled, synchronized forward plus DINO
postprocessing, 50 warm-ups, and five rounds of 100 timed samples on every
replica. The Williams schedule and all candidate/config/checkpoint identities
are embedded in the manifest and launch contract.

The exact dry-run/preflight and historical six-job launch command was:

```bash
cd /localhome/local-rarunachalam/tao-automl
set -a
source /localhome/local-rarunachalam/.tao/config.env
set +a
export SLURM_BASE_RESULTS_DIR=/lustre/fsw/portfolios/edgeai/users/rarunachalam
export SLURM_TIME_HOURS=4
export SLURM_TIMEOUT_HOURS=3.8

/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_feasible_matched_launcher_v2.py \
  --dry-run --verify-remote \
  --manifest experiments/dino_moo_phase2_20260728/latency_feasible_matched_manifest.v1.json \
  --manifest-file-sha256 f83be40f9e00bcd5fc62959bf5a732327353a91ca12ee2efc118325ded2d0db4 \
  --checkpoint-overlay experiments/dino_moo_phase2_20260728/latency_feasible_rec16_checkpoint_overlay.v2.json \
  --checkpoint-overlay-sha256 d054df9923a717759786628177814d5d531d7af3bf136da7361a4882b83410aa \
  --protocol-erratum experiments/dino_moo_phase2_20260728/phase2_protocol_erratum.v1.json \
  --protocol-erratum-sha256 95bba65099027459a50b5e74e43a4ab32c56057e534e70aa7f85bdc9246a7d13

/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_feasible_matched_launcher_v2.py \
  --launch --verify-remote \
  --manifest experiments/dino_moo_phase2_20260728/latency_feasible_matched_manifest.v1.json \
  --manifest-file-sha256 f83be40f9e00bcd5fc62959bf5a732327353a91ca12ee2efc118325ded2d0db4 \
  --checkpoint-overlay experiments/dino_moo_phase2_20260728/latency_feasible_rec16_checkpoint_overlay.v2.json \
  --checkpoint-overlay-sha256 d054df9923a717759786628177814d5d531d7af3bf136da7361a4882b83410aa \
  --protocol-erratum experiments/dino_moo_phase2_20260728/phase2_protocol_erratum.v1.json \
  --protocol-erratum-sha256 95bba65099027459a50b5e74e43a4ab32c56057e534e70aa7f85bdc9246a7d13 \
  --acknowledgement USER_AUTHORIZED_DINO_LATENCY_FEASIBLE_6X8GPU_VALIDATION_20260728
```

The exact terminal aggregation command was:

```bash
cd /localhome/local-rarunachalam/tao-automl
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_feasible_matched_aggregator_v2.py \
  --manifest experiments/dino_moo_phase2_20260728/latency_feasible_matched_manifest.v1.json \
  --manifest-file-sha256 f83be40f9e00bcd5fc62959bf5a732327353a91ca12ee2efc118325ded2d0db4 \
  --checkpoint-overlay experiments/dino_moo_phase2_20260728/latency_feasible_rec16_checkpoint_overlay.v2.json \
  --checkpoint-overlay-sha256 d054df9923a717759786628177814d5d531d7af3bf136da7361a4882b83410aa \
  --submission-ledger experiments/dino_moo_phase2_20260728/runtime/latency_feasible_matched/block_submissions.json \
  --submission-ledger-sha256 a5768627797d91f8c9ab88d5893bcf607705b83ce30a1ceef9d5bc20ce39fd81 \
  --sdk-state experiments/dino_moo_phase2_20260728/runtime/latency_feasible_matched/slurm_state.json \
  --output experiments/dino_moo_phase2_20260728/runtime/latency_feasible_matched/latency_feasible_matched_analysis.json \
  --secrets-env /localhome/local-rarunachalam/.tao/config.env
```

The exact read-only campaign-integrity audit command was:

```bash
cd /localhome/local-rarunachalam/tao-automl
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/latency_feasible_overlay_campaign_integrity.py \
  --manifest-whole-file-sha256 f83be40f9e00bcd5fc62959bf5a732327353a91ca12ee2efc118325ded2d0db4 \
  --manifest-internal-sha256 32c268f11742c26bdc47c61272f73a6d9f651317606e9102e75585bc7d2100e9 \
  --overlay-whole-file-sha256 d054df9923a717759786628177814d5d531d7af3bf136da7361a4882b83410aa \
  --overlay-internal-sha256 e2e6446e3dca7f52e7b2bac682c26bc5b711067b0a18dae71cfbb24f5a44402c \
  --recovery-evidence-whole-file-sha256 923284e51f9e41f6954a1615f683ea6221b62fce0f490c3d5252b7c0d67f3e56 \
  --recovery-evidence-internal-sha256 01715fecc26c3f2b5085650f3bd17f62a502dfc6a85560f11cfa968d08ac9456 \
  --launch-contract-whole-file-sha256 a9fdc853d5847b9a945a012c5c281877a0e8d70c5bec09b5caea31382639b84d \
  --launch-contract-internal-sha256 9a29cad7a9863b7a5b9c961d62a0761a2462422468499948ff840667a608d94f \
  --ledger-whole-file-sha256 a5768627797d91f8c9ab88d5893bcf607705b83ce30a1ceef9d5bc20ce39fd81 \
  --ledger-internal-sha256 0c0a69ca6203a88fac9f04aaf62bc7a7ed2644b18307c1664243af2d389ad168 \
  --analysis experiments/dino_moo_phase2_20260728/runtime/latency_feasible_matched/latency_feasible_matched_analysis.json \
  --analysis-whole-file-sha256 30505ee6fd1635ac4edba01f95136f527a2df9e54257675a1fb505bd0b422efd \
  --analysis-internal-sha256 8ff21a404357e59949dfde7c9889710e854b0e5721ef53f39f439e71478c7351
```

The focused feasible-cohort and latency tie-policy regression command is:

```bash
cd /localhome/local-rarunachalam/tao-automl
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  -m pytest -q \
  tests/test_multi_objective_selection.py \
  experiments/dino_moo_phase2_20260728/test_freeze_rec16_checkpoint_recovery.py \
  experiments/dino_moo_phase2_20260728/test_latency_feasible_checkpoint_overlay.py \
  experiments/dino_moo_phase2_20260728/test_latency_feasible_matched_manifest_launcher.py \
  experiments/dino_moo_phase2_20260728/test_latency_feasible_matched_runner_aggregator.py \
  experiments/dino_moo_phase2_20260728/test_latency_feasible_overlay_campaign_integrity.py
```

The historical 98%-phase reruns reported `91 passed` for the focused
selector/feasible-cohort/recovery/overlay/integrity suite, `299 passed` for
that experiment package, and `392 passed, 1 skipped` for the repository suite.
Those are sealed phase-specific counts, not the current MR merge gate. Exact
timings and compile checks remain recorded in `phase2_validation_report.md`.

## Frozen archive replay

`replay_phase1_archive.py` invokes the production selector over the unchanged
30-candidate Phase-1 archive. It evaluates the predefined multi-objective
policies `none`, 90%, 95%, and 98% while holding latency-mode retention at 98%.
All winner IDs, Pareto ranks, normalization values, scores, and tie-breaks in
`phase1_offline_replay.json` come directly from
`tao_automl.selection.analyze_archive`.

Regenerate or verify the immutable replay:

```bash
python experiments/dino_moo_phase2_20260728/replay_phase1_archive.py
python experiments/dino_moo_phase2_20260728/replay_phase1_archive.py --check
```

With no multi-objective floor, the selector independently returns
`seed_271828_rec_1` as a distinct compromise on the six-point global front.
This remains a frozen-measurement replay only. The completed matched experiment
shows that its small historical latency differences are not
allocation-stable.

## Completed historical-front latency comparison

`matched_pareto_latency_comparison.json` contains 36 measurements: all six
historical global-front candidates on each of six independent eight-GPU
allocations and six distinct A100 nodes. Every allocation completed through
the TAO SDK and SLURM with `Complete / COMPLETED / 0:0` evidence.

The descriptive median ordering is not a stable total order. All 15 paired
candidate comparisons are practically equivalent under the preregistered
`0.75 ms` tolerance, and no paired-bootstrap confidence interval lies wholly
beyond that equivalence band. These measurements never replace the frozen
historical objective values and do not rerun selection.

## Completed sensitivity accuracy evidence

The one-factor study completed 33 exact SQSH training jobs and 42 controlled
accuracy evaluations across three training seeds. The immutable artifacts are:

- `sensitivity_training_checkpoints.v1.json`, SHA256
  `20188a8858a9329ce4b861730ad3b0b2f6185389c8af1b02ad29284e5ed1b012`;
- `sensitivity_training_accuracy.v1.json`, SHA256
  `459da2ebe557ec26947dc723b2864f2bc31880ae3181ad1216c3a47825ec466b`.

The accuracy artifact is descriptive input to the independently matched
latency-sensitivity analysis. It selects no winner and cannot alter the frozen
30-candidate archive.

## Current latency-sensitivity execution contract

The first nine-allocation attempt is preserved under
`sensitivity_latency_manifest.v1.json`, SHA256
`c569f858f4513139292d7189ab5e57f897b8794fdbe5b2dcafc45b0efcd663aa`.
Every v1 allocation failed runtime preflight before its first benchmark
invocation because the exact SQSH reports an NVIDIA prerelease/local PyTorch
build string while v1 required literal equality with the public `2.11.0`
version. Consequently, v1 produced no latency measurements and none of its
allocations can enter the sensitivity analysis. The SDK's default requeue
wrapper then masked those non-timeout `srun` failures at the root-job level;
the per-allocation preflight records, rather than the apparent root success,
are the authoritative v1 evidence.

`sensitivity_latency_manifest.v2.json`, SHA256
`aedc117414b2691c1a70b73fa4e9e0ac123cb4d20dfd9d25dfe2d4aa490d7655`,
explicitly supersedes v1. V2 retains and reports the complete runtime string
but validates its major.minor.patch prefix. It also freezes
`SLURM_USE_REQUEUE=false`, ensuring non-timeout `srun` failures propagate
instead of being hidden by the SDK wrapper. The frozen benchmark design,
profile order, statistical decision rule, and expanded-search derivation rule
are unchanged. `expanded_search_derivation_policy.v1.json` is pinned to this
exact v2 manifest and to the approved `sensitivity_latency_analysis.v2.json`
whole-file/report digests. It also pins the analysis-only erratum, corrected
aggregator, validation test, immutable submission ledger, and every erratum
contract fingerprint. It cannot consume v1, a pre-erratum result, or an
unpinned replacement.

## Expanded shared-archive harness

The first expanded-search launch is preserved as failed pre-selection
evidence: TAO emitted finite mAP50 values as JSON-number strings, the v1
runner rejected them, no latency measurement or Bayesian response was
recorded, and no selector ran. `expanded_search_runtime_erratum.v1.json` and
`expanded_search_v1_failure_audit.md` prohibit reuse of that runtime.

`expanded_search_runner.py` now consumes only the immutable
`expanded_search_manifest.v2.json` produced by
`expanded_search_manifest_generator.py`. V2 changes only strict finite metric
parsing, manifest/supersession identity, and the fresh runtime path; the
search space, seeds, budget, training, latency, and selection contracts remain
byte-identical to v1. The runner does not choose axes or ranges.
Before spawning a seed controller, a fresh launch atomically creates
`runtime_contract.v2.json`, bound to the exact manifest and v2 runtime path.
Fresh launches reject any pre-existing seed or SDK state; resumes require the
same marker, a manifest-bound candidate ledger, and an exact state allowlist.
The manifest pins the absolute launcher path and exact launcher SHA256.
Dry-run reports whether that source is tracked, committed, and clean; launch
refuses unless all three conditions hold and the self-hash still matches.
Preregistered finite integer levels are represented through the existing JSON
Schema integer-`enum` mechanism, which the production search-space loader
exposes as `ordered_int`. The launcher rejects empty, duplicate, non-integer,
or off-grid options instead of widening or quantizing them.

The execution shape is skill-native AutoMLRunner orchestration:

- three local seed-controller processes run concurrently, with collision-free
  per-seed SDK state, AutoML workspace, event log, candidate ledger, and final
  archive roots;
- each controller generates exactly 20 sequential Bayesian recommendations;
- every recommendation submits its own one-node/eight-GPU SQSH training child,
  followed by its own one-node/eight-GPU accuracy-evaluation and stabilized
  selection-time-latency children;
- the aggregate time for 20 candidates is controller wall time, not one SLURM
  allocation. Each child retains the verified `polar3` four-hour allocation
  and 3.8-hour SDK timeout. The observed 422–451 second ten-epoch training time
  plus evaluation and latency is safely below that per-child limit;
- `SLURM_USE_REQUEUE=false` is frozen so non-timeout child failures propagate;
- a partial seed resumes only its sole persisted `run_*` workspace, while a
  seed that never started begins fresh under the same global `--resume`.
  Resume never reopens an immutable completed seed archive, starts a fresh
  controller over partial state, or remeasures a completed candidate; child
  evaluation and latency job IDs are persisted at submission so controller
  restart reconciles the same allocation instead of submitting a duplicate;
- terminal scheduler identity is refreshed after SDK polling. The active
  SLURM ID, retry count, failed-ID lineage, launch-uncertainty flag, and
  durable runtime revision are retained for training, evaluation, and latency
  jobs. Legitimate infrastructure retries are accepted, identity regression
  is rejected, and any unresolved launch uncertainty blocks evidence;
- union selection is blocked until three immutable 20-record seed archives
  reconcile to all 60 terminal candidate IDs. Failed candidates remain in the
  full table but only successful finite measurements enter the selector.

The entire resolved DINO `model` mapping is carried from each recommendation
into training evidence, mAP50 evaluation, and latency configuration. The final
accuracy, 98%-retained latency, and unconstrained multi-objective selections
all come from `tao_automl.selection.analyze_archive` over the same successful
union. The launcher repeats selection under three candidate orderings and
fails if ranks, normalized scores, or winners differ. Selection-time latency
records are preserved; later matched Pareto-front remeasurement is a separate
analysis and cannot replace them.

`phase2_protocol_erratum.v1.json` was issued at
`2026-07-28T06:36:41Z`, when exactly 15 candidates had succeeded (five per
search seed), before the complete union, final Pareto front, combined
selection, candidate table, integrity audit, completion record, post-front
manifest, post-front jobs, or post-front measurements existed. Its exact
whole-file SHA256 is
`95bba65099027459a50b5e74e43a4ab32c56057e534e70aa7f85bdc9246a7d13`.
It leaves the frozen expanded manifest and production selection behavior
unchanged. It records that the manifest's abbreviated tie-break prose is not
the executable authority: the selector pinned at
`83d9d7ecc783724f674cb954f9fbb6c91ea8b0eb` uses the complete recorded
score, ideal-distance, balance-gap, normalized-accuracy-regret, canonical
specification-fingerprint, and candidate-ID chain.

Before any expanded-search result exists, the policy also preregisters the
post-front validation. Every algorithmic global rank-zero candidate is included
in ascending candidate-ID order, with no manual additions or removals. Six
independent one-node/eight-GPU allocations each benchmark the complete front
using a deterministic balanced Williams-row schedule, 50 warm-ups, and five
rounds of 100 timed samples. All candidate pairs use allocation-matched
percentile-bootstrap differences (`10,000` resamples, seed `20260728`) and the
exact imported `0.73553775 ms` practical tolerance. These measurements are
stability evidence only: they never feed reselection, replace selection-time
objectives, or override the algorithm-selected winner.

The erratum preserves two explicitly named analysis branches for both median
and p95. The original preregistered bootstrap-CI classification is retained
and reported. The effective directional classification treats that bootstrap
interval as descriptive and permits a stable direction only when the exact
one-sided, tolerance-shifted sign-flip test passes and all six paired
differences lie strictly beyond the same practical-tolerance boundary. A lack
of effective directional evidence is not an equivalence claim. Effective
claims remain pairwise only and do not establish a simultaneous total order.

Dry-run after the expanded manifest is generated:

```bash
MANIFEST_SHA256="$(sha256sum \
  experiments/dino_moo_phase2_20260728/expanded_search_manifest.v2.json \
  | awk '{print $1}')"
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/expanded_search_runner.py \
  --dry-run \
  --manifest-file-sha256 "$MANIFEST_SHA256"
```

The launch path additionally requires `--verify-remote` and the exact
acknowledgement
`USER_AUTHORIZED_3X8GPU_SLURM_DINO_EXPANDED_SEARCH_20260728`. It writes
runtime-only mutable ledgers below `runtime/expanded_search_v2/`, then seals one
`seed_archive.v1.json` per seed and emits
`expanded_combined_selection.json`, complete JSON/CSV candidate tables, and
`expanded_integrity_audit.json`.

## Completed historical six-front matched design

This design and its `0.75 ms` tolerance apply only to the completed historical
six-candidate-front comparison. The expanded global-Pareto-front
four-candidate validation described in **Final status** uses the exact
`0.73553775 ms` protocol-erratum tolerance recorded in
`post_front_matched_manifest.v1.json`. The separate four-candidate
98%-accuracy-feasible latency-cohort study uses that same exact tolerance
through `latency_feasible_matched_manifest.v1.json`; it is not a Pareto-front
study.

- Six independent SLURM jobs requested one node and eight GPUs each.
- Every job benchmarked every historical-front candidate sequentially on its
  allocated node.
- For `n` canonical candidates, the complete deterministic Williams design has
  `R` rows. Allocation `k` in `[0, 5]` uses design row
  `floor(k * R / 6)`. The manifest records the actual selected row indices,
  per-candidate position counts, ordered immediate-adjacency counts, and their
  measured imbalances. Exact once-per-position and once-per-adjacency balance
  is claimed only when those recorded counts establish it; it is not assumed
  for an arbitrary final-front size.
- Because every candidate runs in every allocation, node assignment is exactly
  matched across candidates. The six jobs are submitted together to encourage
  node diversity, but distinct physical nodes are not assumed.
- Every candidate invocation uses a unique block/position/candidate run label
  and output root, so its eight rank JSON files cannot overwrite another run.
- The latency procedure is byte-for-byte pinned to the phase-1 benchmark:
  50 warm-ups, five rounds of 100 timed samples on each of eight replicas,
  batch size one, FP32, TF32 disabled, synchronized model-forward plus DINO
  postprocessing, median-of-device-round-medians, pooled p95, and the original
  quality gates.
- The historical six-front practical-equivalence tolerance was `0.75 ms`,
  rounded up from the prior `0.73553775 ms` independent-allocation range.

## Completed historical six-front checkpoint recovery

The read-only artifact audit found that phase-1 retention removed four
historical checkpoint paths:

- `seed_161803_rec_0`
- `seed_161803_rec_2`
- `seed_271828_rec_1`
- `seed_271828_rec_2`

The exact checkpoints for `seed_271828_rec_5` and `seed_314159_rec_7` remain
available and are SHA256-pinned. `manifest.v1.json` therefore cannot submit a
partial matched experiment.

The launcher rendered four exact-config recovery jobs using the original DINO
training template, candidate hyperparameters, seed `1234`, ten-epoch budget,
eight-GPU DDP topology, dataset, PTM, SQSH, and terminal-checkpoint policy.
These retrains were validation-only. Their completed checkpoints were hashed
and recorded in the immutable `manifest.v2.json`; v1 was never edited to point
at runtime outputs.

## Historical six-front workflow commands

These commands are retained to reproduce the completed historical
six-candidate-front workflow; they are not the expanded four-front protocol.

Structural dry-run (no credentials and no job submission):

```bash
cd /localhome/local-rarunachalam/tao-automl
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/phase2_launcher.py \
  --dry-run \
  --report experiments/dino_moo_phase2_20260728/runtime/dry_run.json
```

Read-only remote artifact verification:

```bash
cd /localhome/local-rarunachalam/tao-automl
set -a
source /localhome/local-rarunachalam/.tao/config.env
set +a
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/phase2_launcher.py \
  --dry-run --verify-remote \
  --report experiments/dino_moo_phase2_20260728/runtime/remote_preflight.json
```

Historical recovery submission command, used after reviewing the dry-run
report:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/phase2_launcher.py \
  --submit-recovery --verify-remote --acknowledge-validation-only
```

For a historical-workflow reproduction, monitor all four TAO and SLURM
identities from the durable SDK state after submission. This is read-only with
respect to SLURM and writes only the ignored runtime status report:

```bash
cd /localhome/local-rarunachalam/tao-automl
set -a
source /localhome/local-rarunachalam/.tao/config.env
set +a
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/finalize_checkpoint_recovery.py \
  --status
```

For that reproduction, create v2 when the status report says
`ready_for_manifest_v2`:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/finalize_checkpoint_recovery.py \
  --finalize
```

The finalizer requires all four SDK statuses to be `Complete`, all four SLURM
allocation states to be `COMPLETED` with exit code `0:0`, and exactly one
`model_epoch_009_step_*.pth` or `.ckpt` below each job-scoped Lustre result
root. It hashes each checkpoint over read-only SSH, regenerates all six latency
evaluation-config digests, and creates `manifest.v2.json` with recovery
provenance. It never edits v1. Repeating `--finalize` accepts byte-equivalent v2
content and refuses to overwrite different content.

Historical matched-block submission command; it requires the immutable v2
manifest with all six recovered checkpoint digests and fails against v1:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/phase2_launcher.py \
  --manifest experiments/dino_moo_phase2_20260728/manifest.v2.json \
  --submit-blocks --verify-remote --acknowledge-validation-only
```
