# SegFormer / full VOC2012 campaign

This directory freezes a direct production campaign for TAO network identifier
`segformer` and the complete official VOC2012 semantic-segmentation train/val
splits. It does not contain a CPU model run, model smoke test, mini-step, or
synthetic-input benchmark.

The campaign launches three independent Bayesian AutoML controllers:

| Mode | Recommendation target | Final policy |
| --- | --- | --- |
| Accuracy | Expected improvement on `val_miou` | Highest valid accuracy |
| Latency | Constrained expected improvement | Raw-minimum-anchored equivalent-fastest cohort at 90% retained accuracy |
| Multi-objective | ParEGO expected improvement on mIoU and latency | Independent rank-zero augmented-Chebyshev compromise |

Every train, standalone validation, and latency child requests one node and
eight A100 GPUs and runs directly from the pinned TAO 7.1 SQSH. Latency uses
real validation images, 50 warm-ups, five rounds of 100 model-forward samples
on each of eight replicas, and a 4,000-sample quality gate.

A finite `val_miou` below 0.10 halts the first-candidate gate for data,
optimization, fidelity, and metric root-cause analysis. This preregistered
VOC experiment sanity gate is not an AutoML selection constraint.

## Fail-closed PTM state

The repository currently records 13 official SegFormer PTMs, all as
`unverified`. Each arm must first complete a real full-dataset, 50-epoch,
one-node/eight-GPU train and standalone validation workflow. The sealed
successor projects only an exact successful arm with complete existing license
metadata to `supported` in a campaign-local in-memory registry. Terminal
failures and metadata-incomplete records remain exclusions. The repository
registry file and ordinary runtime behavior are never mutated, and no manual
promotion or candidate choice occurs.

`qualification_campaign.py` implements that missing qualification step. Its
data-only stage resolves all 13 exact NGC members, verifies their immutable
identities and registered sizes, generates the checkpoint-target-specific train
and evaluation YAMLs from the packaged SegFormer templates, checksums them, and
publishes every checkpoint and spec read-only on Lustre. It does not import a
model framework, load a checkpoint, or construct a scheduler job.

Qualification v5 re-hashes the complete stage and executes a sealed selective
recovery plan: four Cityscapes arms reuse their exact successful v4 50-epoch
train phases, while the nine backbone-prefix load-failure arms run new full
50-epoch trains. All 13 arms run a new standalone evaluation. The controller
therefore submits exactly 9 new train jobs and 13 new evaluation jobs, and its
atomic launch claim forbids re-entry that could repeat a successful train.
Fresh training uses AdamW learning rate `1e-4`, weight decay `5e-4`,
random-color and random-blur augmentation disabled, and the distributed
sampler enabled. Validation runs every epoch. Each new train and evaluation
job uses one node, eight
`NVIDIA A100-SXM4-80GB` GPUs, and the pinned SQSH. `polar3` is capped at four
hours, so the controller freezes the skill-compliant `4.0`-hour scheduler limit
and `3.8`-hour SDK timeout.

Every new v5 train and evaluation verifies and installs the TAO PyTorch
SegFormer product-fix overlay from commit
`2681dea4c876b759f8a0446491b3619e6120b531`, archive SHA-256
`a7d5316816710b258c52001f979a22723c88fca5101a05ca3a48838ce81d1ee4`.
The four reused trains retain their truthful v4 runtime provenance: commit
`3b1e073571f3bbf3702b0ae837e9279ad12f4286`, archive SHA-256
`b055100d0d3e9e8c5daf94dfd4caf3cccacfb54fbebb423129fb5832066e420b`.

The terminal v1 evidence remains immutable at
`/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_ptm_qualification_v1`.
Its completion SHA-256 is
`e7d604d63b2e79a54e21f7cac708ad6b1ff12ea8ad24556f911d662876412661`;
the terminal v2 evidence is likewise retained at
`/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_ptm_qualification_v2`,
with completion SHA-256
`0e74b0016711bf614b40382c7a28b512c87971951dff6295cf107f44a901cf1a`.
V2 exposed a controller mismatch: twelve completed 50-epoch trains produced
the exact `model_epoch_049_step_09150.pth`, but the qualification controller
called the ten-epoch search resolver and looked for epoch 9. The thirteenth
train failed before optimization when rank zero could not bind a transient
Lightning rendezvous port. V3 has a qualification-specific epoch-49 resolver
and attempted to export a deterministic, allocation-derived single-node
rendezvous port, but its braced shell variable crossed the SDK runner's Python
format boundary and all 13 jobs failed before TAO started. Its completion is
retained at
`/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_ptm_qualification_v3/completion.json`,
with SHA-256
`b8279dd87df2389c56a02db69dc8038f8bd84dcebe93cd1d3d66d04cb3fdfabc`.
V4 kept the epoch-49 and rendezvous fixes and completed all 13 train phases.
Its terminal audit proves positive exact-model loads for four Cityscapes arms;
the other nine checkpoints exposed a uniform loadable `backbone.` prefix
mismatch. V4 evaluation evidence was terminal but not workflow-qualifying.
V5 preserves v1-v4 byte-for-byte, binds the v4 completion and load audit by
whole-file and internal hashes, reuses only those four proven train phases,
and retrains only the nine failed-load arms with the corrected product loader.

All arms are attempted. A failed arm is retained as a terminal structured
failure; it is never replaced with a fallback checkpoint. Completion writes
the exact `qualification_gate.py` input and a non-promoting handoff. The gate
binds the v5 contract and controller, clean source, wheel, PTM stage, source
checkpoint, architecture, backbone, task, target field, positive-load receipt,
terminal checkpoint, and metrics. The successor contract also binds both the
whole-file and internal completion hashes before constructing the in-memory
projection.

## Frozen data

The prepared root is:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/voc2012_segmentation_v1/prepared
```

It contains 1,464 train and 1,449 validation image/mask pairs. Masks retain
VOC IDs 0–20 and ignore ID 255. TAO uses `label_transform: "None"`,
`num_classes: 21`, and an explicit grayscale palette. The byte-identical local
and Lustre stage record is frozen at
`437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d`;
all 5,827 manifest entries passed and the remote dataset has zero writable
files.

## Reproduction

First seal the clean integrated source into an external campaign contract:

```bash
cd /localhome/local-rarunachalam/.tao/worktrees/tao-automl-segformer-v5-successor
PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.manifest_generator \
  --output /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode/campaign.v5.json
```

Stage and independently verify all PTM/spec inputs without reserving GPUs:

```bash
PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.qualification_campaign \
  --stage

PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.qualification_campaign \
  --check-stage
```

The following explicit command is the only qualification path that submits
jobs. It starts 13 independent workers implementing the exact 4-reuse,
9-train, 13-evaluate plan; no smoke or mini-step precedes them:

```bash
PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.qualification_campaign \
  --launch
```

## Durable automatic successor

The following single watcher command may be started while v5 is still running.
It waits only while the exact completion is absent. A present invalid or
zero-success terminal completion is rejected immediately. Once v5 passes the
gate, the watcher atomically publishes a new read-only `campaign.v6.json`
without replacing any existing file, constructs the campaign-local eligibility
projection, and invokes `run_campaign --automatic-trigger --launch`. The fresh
v6 runtime root and contract path do not overlap the v1-v5 evidence:

```bash
cd /localhome/local-rarunachalam/.tao/worktrees/tao-automl-segformer-v5-auto-successor
nohup env PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.manifest_generator \
  --repository "$PWD" \
  --qualification /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_ptm_qualification_v5/completion.json \
  --qualification-contract /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode/campaign.v5.json \
  --ptm-stage-manifest /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_ptm_qualification_v5/ptm_stage_manifest.json \
  --output /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode/campaign.v6.json \
  --runtime-root /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode_v6 \
  --automatic-trigger \
  --launch \
  > /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_v5_to_v6_automatic_successor.log 2>&1 &
```

There is no post-gate confirmation. The immutable launch claim prevents a
second submission. If and only if this claimed v6 runtime was interrupted,
rerun the same command with `--resume`; a fresh `--resume`, a changed contract,
or a concurrent watcher fails closed. A completed three-mode runtime is
restart-idempotent and is not submitted again.

The lower-level invocation performed automatically by the watcher is:

```bash
PYTHONPATH="$PWD/src" \
python -m experiments.cross_model_automl_20260729.segformer_voc2012_campaign.run_campaign \
  --contract /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode/campaign.v6.json \
  --runtime-root /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/segformer_voc2012_three_mode_v6 \
  --automatic-trigger \
  --launch
```

No confirmation is required after prerequisites pass. The first real
candidate in each mode runs in parallel; the remaining 29 candidates per mode
are released automatically only when all three first candidates pass training,
standalone validation, stabilized latency, audit, and provenance gates.
