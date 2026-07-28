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
This remains a frozen-measurement replay only; its small historical latency
differences are not treated as allocation-stable until the matched experiment
finishes.

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
