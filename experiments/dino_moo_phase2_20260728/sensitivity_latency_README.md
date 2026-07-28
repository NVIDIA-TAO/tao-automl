# DINO sensitivity matched-latency harness

This package is the downstream measurement stage for
`one_factor_sensitivity_manifest.v1.json`. It never trains, promotes manually,
or feeds final AutoML selection. The launcher consumes the immutable 33-entry
training checkpoint artifact. Aggregation additionally requires the immutable
42-entry accuracy artifact, a finalized nine-entry submission ledger, and its
matching TAO SDK/SLURM state.

## Files

- `sensitivity_latency_manifest.v1.json`, SHA256
  `c569f858f4513139292d7189ab5e57f897b8794fdbe5b2dcafc45b0efcd663aa`,
  is retained as immutable failed-attempt evidence. All nine v1 allocations
  stopped in runtime preflight before the first benchmark invocation, so v1
  produced no latency measurements.
- `sensitivity_latency_manifest.v2.json`, SHA256
  `aedc117414b2691c1a70b73fa4e9e0ac123cb4d20dfd9d25dfe2d4aa490d7655`,
  supersedes v1. It preserves the nine-block design, benchmark, aggregation,
  and effect-qualification contracts while matching the public
  major.minor.patch prefix of the SQSH's full NVIDIA PyTorch build string. It
  also disables SDK SLURM requeue handling so a non-timeout `srun` failure
  cannot be masked by the wrapper's requeue tail.
- `sensitivity_latency_checkpoint_artifact.schema.json` documents the immutable
  33-entry training checkpoint artifact.
- `sensitivity_latency_accuracy_artifact.schema.json` documents the completed
  42-entry mAP50 evidence artifact.
- `sensitivity_latency_common.py` validates frozen profiles, artifact
  completeness and checkpoint reuse, and the schedule.
- `sensitivity_latency_launcher.py` stages nine independent eight-GPU jobs.
- `sensitivity_latency_block_runner.py` verifies one allocation and executes
  all 14 profiles sequentially.
- `sensitivity_latency_aggregate.py`, SHA256
  `5f5aebd4274c746ec9674f28f978af5d228d98c6ba0af8d76cff8b1742dab967`,
  is retained byte-for-byte as the measurement-manifest-pinned analysis
  source. Its allocation-level runtime check incorrectly used full-string
  PyTorch equality even though v2 declares a major.minor.patch comparison.
- `sensitivity_latency_analysis_erratum.v1.json`, SHA256
  `a86a66822137433882af079b8384698cbf9b06124cc173a8b36c666aefa60a80`,
  pins the immutable v2 manifest, submission ledger, measurement-generation
  sources, unchanged measurement and qualification policies, and corrected
  analysis source. It also pins the exact remote-evidence acquisition policy.
- `sensitivity_latency_aggregate_erratum.py` is the analysis-only entrypoint.
  It regenerates the original plans and command hashes, validates the original
  source and ledger chain, preserves the full runtime string, and corrects only
  allocation-level PyTorch validation to use the v2-declared
  major.minor.patch rule. Because `/lustre/fs11` is not mounted on the analysis
  host, it derives all 1,017 expected remote paths from the trusted ledger and
  regenerated plans, hashes them through read-only SSH, fetches only the exact
  missing paths through `rsync --files-from`, and aggregates from a verified
  local mirror. It does not rewrite measurements or objective values.
  The immutable ledger's launch commit
  `cb62ef447704b95980b17aa82604992564b4e71f` is retained as the
  measurement-generation identity. Analysis from a later commit is accepted
  only on the exact same branch when Git proves that launch commit is its
  ancestor and every manifest-pinned measurement source hash remains exact.

## Frozen execution design

There are three allocations per training seed and nine allocations total. Each
allocation runs all 14 profiles on the same node in one deterministic
Williams-style order. Rows 0 through 8 are used exactly once and assigned three
rows per seed. Every profile therefore has three matched allocation
measurements per seed and occupies a distinct position in each repeat.

Every profile uses:

- one A100 node with all eight GPUs;
- the pinned TAO 7.0.1 SQSH;
- the exact full resolved model mapping frozen by the one-factor manifest;
- `train.activation_checkpoint=false`;
- FP32 with TF32 disabled;
- 16 fixed preprocessed inputs;
- 50 warmups followed by five rounds of 100 timed iterations;
- CUDA synchronization and NCCL barriers;
- model forward plus DINO GPU postprocessing as the timed scope.

Each runner accepts only the exact SDK-owned output root
`$TAO_RESULTS_ROOT/$TAO_JOB_ID`. The allocation result is written under that
job-scoped directory at:

```text
dino_moo_phase2_20260728/sensitivity_latency/
  <manifest-id>/seed_<seed>/<allocation-id>/allocation_result.json
```

The runner never appends a second TAO job ID to the allocation root. The
pinned benchmark retains its own job-ID subdirectory beneath each profile
directory for raw `rank_0.json` through `rank_7.json` files.

`model.num_select` profiles must reuse the exact path and SHA256 of the
same-seed reference checkpoint. The launcher rejects any other mapping.

## Immutable checkpoint artifact

The completed artifact is
`sensitivity_training_checkpoints.v1.json`, SHA256
`20188a8858a9329ce4b861730ad3b0b2f6185389c8af1b02ad29284e5ed1b012`.
It contains the 33 independently trained checkpoints. The launcher derives the
nine `num_select` records exclusively by exact same-seed reference reuse.

Record the whole-file SHA256 independently:

```bash
sha256sum experiments/dino_moo_phase2_20260728/sensitivity_training_checkpoints.v1.json
```

The launcher and aggregator both require that digest on the command line.
Editing the artifact after recording it makes both tools fail closed.

## Dry run

Dry-run validates the immutable artifact, all frozen sources, exact model
digests, nine order-balanced blocks, 126 planned measurements, staged configs,
and commands without reading remote state or creating jobs:

```bash
cd /localhome/local-rarunachalam/tao-automl
python experiments/dino_moo_phase2_20260728/sensitivity_latency_launcher.py \
  --dry-run \
  --manifest \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_manifest.v2.json \
  --runtime-dir \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency_v2 \
  --checkpoint-artifact \
  experiments/dino_moo_phase2_20260728/sensitivity_training_checkpoints.v1.json \
  --checkpoint-artifact-sha256 \
  20188a8858a9329ce4b861730ad3b0b2f6185389c8af1b02ad29284e5ed1b012 \
  --report /tmp/sensitivity_latency_v2_dry_run.json
```

Add `--verify-remote` to perform read-only SSH verification of the SQSH,
validation data, and every unique checkpoint.

## Guarded concurrent submission

Submission is impossible unless the complete remote verification passes and the
user-authorized acknowledgement matches exactly:

```bash
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

This queues all nine independent eight-GPU blocks through
`tao_sdk.platforms.slurm.SlurmSDK.create_job`. A dedicated submission ledger
prevents an accidental duplicate launch. Submission additionally requires the
launcher, runner, common utilities, aggregator, and latency statistics source
to be tracked, committed, clean, and equal to the SHA256 values pinned in the
manifest. After all nine submissions are recorded, preserve the ledger and
record its whole-file SHA256. Aggregation refuses incomplete ledgers.

If one complete block must be retried, use `--retry-allocation` with the prior
ledger and its SHA256, a new retry-ledger path, and immutable retry evidence.
The evidence must identify the prior TAO and SLURM IDs, explicitly forbid
partial-measurement reuse, and use one of the frozen terminal
failure/invalid-artifact reason codes. The launcher reruns all 14 profiles in
the same frozen order under a fresh TAO/SLURM allocation and writes a new
nine-entry ledger linked to the prior ledger. It never edits the prior ledger
or reuses a partial block.

## Aggregation

The completed `sensitivity_training_accuracy.v1.json` has SHA256
`459da2ebe557ec26947dc723b2864f2bc31880ae3181ad1216c3a47825ec466b`.
Aggregation has no arbitrary `--results-root` option. It derives every exact
result path from the finalized ledger and `sdk.get_job_results_dir()` after
verifying SDK and scheduler identity:

```bash
python experiments/dino_moo_phase2_20260728/sensitivity_latency_aggregate_erratum.py \
  --analysis-erratum \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_analysis_erratum.v1.json \
  --analysis-erratum-sha256 \
  a86a66822137433882af079b8384698cbf9b06124cc173a8b36c666aefa60a80 \
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
  --output /path/to/sensitivity_latency_aggregation.json
```

The erratum aggregator requires nine distinct TAO IDs and nine distinct SLURM
allocation IDs. For every entry it checks the submitted command and block-plan
digests, durable SDK status `Complete`, SDK/ledger SLURM identity, exact
job-scoped result URI, and a single `sacct` root row in `COMPLETED` state with
exit code `0:0`, one node, eight GPUs, and the pinned partition/account. It
then validates the allocation seed, repeat, Williams row, profile positions,
config/model/checkpoint digests, hostname/node assignment, eight stable and
distinct GPU UUIDs, runtime, protocol, benchmark-input identity, and all 1008
raw rank-file hashes.

Before original ledger reconciliation, the erratum strict-reads and hashes the
already pinned ledger, validates its complete launch-time `source_checks`
against the immutable manifest, and proves the current analysis HEAD is a
descendant of the recorded launch commit on the same configured branch. It
then passes the original launch-time source-check mapping to the original
ledger validator. Regenerated plans and command hashes are still compared
exactly. The report records launch commit, analysis commit, merge base, commit
distance, both source-check mappings, and the ancestry result. Unrelated
history, branch drift, or any measurement-source drift fails closed.

The evidence snapshot path is explicit and must be a dedicated child beneath
`runtime/sensitivity_latency_v2`. Remote paths are never discovered with
`rglob`, `find`, or wildcard expansion: they are derived exactly from the nine
ledger roots, frozen relative layout, regenerated run labels, and ranks zero
through seven. SSH rejects missing, non-regular, symlinked, duplicate, or extra
inventory entries before transfer. Existing local files are accepted only
when their SHA256 equals the fresh remote SHA256; non-identical files are never
overwritten. Transfer uses an isolated sibling staging directory, verifies all
bytes, then installs files without replacement. A repeated command is
resume-safe and performs no transfer for byte-identical evidence. The report
records every remote and local path, size, SHA256, digest-equality result, and
the aggregate inventory SHA256 while continuing to validate the original
embedded remote result-root strings.

For each allocation/profile the report contains median, p95, bootstrap median
CI, robust CV, round range, drift, and device range. Per-profile summaries make
both within-allocation dispersion and between-allocation/seed variability
explicit. Paired effects are candidate minus the same allocation and same-seed
reference.

The new baseline noise estimate is the largest within-seed range across each
seed's three reference allocations. The effective floor is:

```text
max(0.73553775 ms, new reference range)
```

For each non-reference level, the deterministic hierarchical paired bootstrap
independently resamples the three training seeds, resamples the three matched
allocations within each sampled seed, computes a within-seed median, and then
the median across sampled seeds. A level is
`latency_effect_qualified=true` only when its 95% CI lies wholly below
`-effective_noise_floor_ms` (effect direction `faster`) or wholly above
`+effective_noise_floor_ms` (effect direction `slower`). This rule is
direction-agnostic because either side can define a future accuracy/latency
trade-off. CIs overlapping the practical-equivalence band do not qualify.

The same-seed 98% accuracy-retention result is reported separately as
`latency_mode_98pct_suitable`. It never gates
`latency_effect_qualified` or inclusion in the future shared multi-objective
search. `latency_reduction_qualified` identifies only the reliably faster
branch for constrained-latency analysis.

Missing, duplicate, invalid, unscoped, or inconsistent evidence fails closed.
The report always records `winner_selected=false`,
`feeds_final_selection=false`, and `manual_promotion_permitted=false`.
