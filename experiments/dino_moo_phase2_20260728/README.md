# DINO multi-objective phase-2 latency validation

This directory preregisters a validation-only matched latency experiment for
the six candidates on the historical global Pareto front. It does not change
the frozen AutoML archive or feed any phase-2 value back into selection.

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

## Design

- Six independent SLURM jobs request one node and eight GPUs each.
- Every job benchmarks all six candidates sequentially on its allocated node.
- The six Williams/Latin-square rows place every candidate in every execution
  position once and contain every ordered immediate adjacency once.
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
- The preregistered practical-equivalence tolerance is `0.75 ms`, rounded up
  from the historical `0.73553775 ms` independent-allocation range.

## Fail-closed checkpoint recovery

The read-only artifact audit found that phase-1 retention removed four
historical checkpoint paths:

- `seed_161803_rec_0`
- `seed_161803_rec_2`
- `seed_271828_rec_1`
- `seed_271828_rec_2`

The exact checkpoints for `seed_271828_rec_5` and `seed_314159_rec_7` remain
available and are SHA256-pinned. `manifest.v1.json` therefore cannot submit a
partial matched experiment.

The launcher renders four exact-config recovery jobs using the original DINO
training template, candidate hyperparameters, seed `1234`, ten-epoch budget,
eight-GPU DDP topology, dataset, PTM, SQSH, and terminal-checkpoint policy.
These retrains are validation-only. After they complete, their checkpoints
must be hashed and recorded in a new immutable `manifest.v2.json`; v1 is never
edited to point at runtime outputs.

## Commands

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

Future recovery submission, only after reviewing the dry-run report:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/phase2_launcher.py \
  --submit-recovery --verify-remote --acknowledge-validation-only
```

After submission, monitor all four TAO and SLURM identities from the durable
SDK state. This is read-only with respect to SLURM and writes only the ignored
runtime status report:

```bash
cd /localhome/local-rarunachalam/tao-automl
set -a
source /localhome/local-rarunachalam/.tao/config.env
set +a
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/finalize_checkpoint_recovery.py \
  --status
```

When the status report says `ready_for_manifest_v2`, create v2:

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

Future matched-block submission requires a new manifest with all six recovered
checkpoint digests and will fail against v1:

```bash
/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python \
  experiments/dino_moo_phase2_20260728/phase2_launcher.py \
  --manifest experiments/dino_moo_phase2_20260728/manifest.v2.json \
  --submit-blocks --verify-remote --acknowledge-validation-only
```
