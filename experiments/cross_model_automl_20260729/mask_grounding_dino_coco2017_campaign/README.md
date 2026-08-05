# Mask Grounding DINO / full COCO 2017 AutoML campaign

This directory prepares three independent, objective-aware AutoML jobs for the
exact TAO model identifier `mask_grounding_dino`. It has no CPU model-test,
smoke-test, mini-step, or local-GPU path. Every model job is bound to the pinned
TAO 7.1 SQSH and one node with eight A100 GPUs.

The immutable four-PTM stage is complete. Qualification v1 is preserved at
SHA-256 `a48d8d8d2a5c65e35c9d39bd5ed1362be54e2be0b89dcda5471812da331a6996`:
all four jobs loaded their exact PTM and full COCO data, then failed on the
first distributed training batch because plain DDP did not detect unused
parameters. V2 changes only the effective DDP strategy and writes to a new
runtime root. After v2 completes, a schema-v2 campaign-local eligibility
decision can authorize only exact successful PTM identities through a
validated in-memory registry projection. The repository registry remains
unchanged, failed arms remain exclusions, and the automatic trigger stays
fail-closed until that evidence-bound decision is valid.

## Scientific contract

This is category-prompted COCO instance segmentation through Mask Grounding
DINO's `data_type: OD` path. The prompts are the 80 COCO category names. This
does not claim phrase-grounding coverage.

The primary metric is `segm_val_mAP50_95`, read from TAO's exact
`[segm] val_mAP@50-95` status key. The inspected TAO 7.1 implementation creates
an OD evaluator with both `bbox` and `segm` IoU types. VG `overall_IoU` is a
different referring-expression metric and is never accepted as COCO mask AP.
The repository metric policy records the mask metric as an unscaled fraction
in `[0, 1]`.

## Frozen data and text assets

The complete official COCO 2017 stage is read-only at:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/coco2017_instance_panoptic_v1
```

Training uses the official TAO COCO-to-ODVG projection:

```text
tao/mask_grounding_dino/train/instances_train2017_odvg.jsonl
tao/mask_grounding_dino/train/instances_train2017_odvg_labelmap.json
```

The projection contains 117,266 annotated training images and preserves all
860,001 instance annotations and all 860,001 masks. The underlying full COCO
stage still contains all 118,287 train images. The ODVG JSONL SHA-256 is
`d5deb4f5cfe027786fb1ceb52632ad6d3ef027e95e434525ba715d6841fb2921`;
the label-map SHA-256 is
`02075d96f6bf06d061f9329b4775dc7c3bb5ac140c77bc5c0e465d305c46d6c1`.

TAO's OD evaluator requires contiguous category IDs. Validation and standalone
evaluation therefore use the deterministic, lossless annotation derivative:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/mask_grounding_dino_coco2017_od_v1/instances_val2017_remapped.json
```

Its SHA-256 is
`9c9af9918e29292adfaa78a694d471e2be6d226e150300d9f4b22c2d77723ebc`.
It contains the same 5,000 images, 36,781 annotations, and mask segmentations as
the official validation JSON, with category IDs remapped to `0..79`. The
conversion manifest is
`coco2017_contiguous_validation.v1.json`, SHA-256
`3c2d09d20211017575a2c51a6797ef91f1939340d978a5d11d1edab1a30b2d`.
Both staged derivative files are read-only, and a repeated conversion was
byte-identical.

The original full-stage manifest SHA-256 is
`437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d`;
the 246,593-entry file-manifest SHA-256 is
`10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e`.

The frozen offline BERT asset is:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/ptms/huggingface/bert-base-uncased/86b5e0934494bd15c9632b12f734a8a67f723594
```

Its five-file tree SHA-256 is
`04cd5cc67804f4752df93e7c05dd51d904e82fc05d28794ddb03504cca689fb5`.
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are mandatory.

## Official checkpoint inventory and qualification

The repository-owned inventory contains these four official Swin-T arms:

```text
mask_grounding_dino.commercial.swin_tiny.trainable.v2.1
mask_grounding_dino.commercial.swin_tiny.trainable.v2.0
mask_grounding_dino.commercial.swin_tiny.trainable.v1.0
mask_grounding_dino.research.swin_tiny.trainable.v2.0
```

Each record has an exact immutable NGC identity, expected member size,
TAO-7.1 compatibility declaration, license/access metadata, and a
repository-owned path-free YAML sidecar. Known checkpoint hashes are retained;
the older commercial v1.0 member must be hashed while staging. All four remain
`unverified` and are not globally runtime eligible; only a later sealed v3
campaign-local projection can authorize exact successful qualification arms.

`ptm_stage.py` is the repository-owned data-only staging path. It supports
either direct execution where `/lustre` is mounted or local execution through
an already-active SSHFS mount of the remote `/lustre` root. In mapped mode,
`--lustre-root` remains the canonical remote identity and
`--physical-lustre-mount` names only the local mount root; the physical stage
path is derived and cannot be supplied independently. The stager verifies the
mount point and canonical-to-physical correspondence before any network
access.

The stager resolves each exact NGC member with the production authenticated
HTTPS client, verifies the remote and downloaded size, verifies registered
checksums (or records the observed checksum for the immutable v1.0 member),
uses `AtomicArtifactCache`, and atomically publishes an exact read-only
four-file stage. Reuse is byte-verified; unexpected files, symlinks, writable
completed artifacts, partial completed manifests, or identity drift fail
closed. The local and Lustre manifest copies are byte-identical, contain
canonical `/lustre/...` checkpoint identities even in mapped mode, and use
exactly the schema consumed by `ptm_stage_record()` and `load_ptm_stage()`.

Qualification is deliberately stronger than a smoke test. Each staged arm must
complete one real three-epoch full-dataset train and standalone full-validation
workflow on one node/eight A100s. The four arms launch concurrently with an
independent durable SDK state store per workflow. In-epoch and standalone mask
AP must be finite and pass the preregistered experiment sanity gate of `0.05`.
Failures are terminal preserved exclusions. Successful evidence does not
mutate the repository registry. Ordinary product runtime continues to require
repository `supported` status. This sealed campaign additionally permits a
versioned in-memory projection for an exact successful identity, bound to the
completion-file hash, internal evidence hash, base registry and record hashes,
TAO/SQSH identity, source commit, wheel, SDK, and skills. The projection is
never written back to the repository registry and cannot promote a failed or
explicitly unsupported arm.

### Qualification v2 DDP correction

Static inspection of the pinned SQSH is recorded in
`ddp_strategy_audit.v2.json`. TAO's public configuration field accepts only
`ddp` and `fsdp`; the literal Lightning alias
`ddp_find_unused_parameters_true` is not a valid direct TAO config value. The
pinned Mask Grounding DINO launcher resolves the supported combination below
to that Lightning strategy:

```yaml
train:
  distributed_strategy: ddp
  activation_checkpoint: false
```

V1 used `activation_checkpoint: true`, which resolved to plain DDP and caused
the preserved first-batch failures in SLURM jobs `31243535`–`31243538`. V2
keeps every PTM, dataset identity, batch size, epoch, objective, search range,
seed, hardware requirement, and SQSH identity unchanged. It changes only the
effective distributed strategy to unused-parameter-aware DDP. The direct-full
GPU jobs—not a CPU or mini-step probe—validate whether that supported TAO
resolution works for this model.

## Frozen objective-aware search

PTM identity is a hierarchical, non-ordinal outer arm. Within each arm, the
search space is:

```yaml
model.num_select: [50, 100, 200, 300]
train.optim.lr: [1.0e-5, 5.0e-4]          # log scale
train.optim.lr_backbone: [1.0e-6, 5.0e-5] # log scale
train.optim.weight_decay: [1.0e-5, 1.0e-3] # log scale
```

`model.enc_layers` and `model.dec_layers` are fixed at six by the packaged TAO
schema and are not searched. Each mode gets 24 recommendations, three complete
training epochs per candidate, search seed `271828`, and training seed `1234`.
The modes have separate empty observation namespaces and share no observations.

| Mode | Recommendation acquisition | Terminal selection |
| --- | --- | --- |
| Accuracy | Expected improvement on mask AP | Highest valid mask AP |
| Latency | Constrained expected improvement using a monotonic best-observed in-job reference | Highest-accuracy member of the raw-minimum-anchored equivalent-fastest cohort satisfying 90% retained accuracy |
| Multi-objective | ParEGO expected improvement on mask AP and latency | Pareto-rank-zero, front-normalized augmented-Chebyshev compromise |

Latency retention applies only to latency mode. Multi-objective mode does not
inherit it. All agent-intervention and validation-to-selection feedback flags
are frozen `false`.

## Selection-time latency

Every candidate uses the same 16 immutable real COCO validation images, 80
frozen prompts, FP32, and batch size one. The timed scope includes Mask
Grounding DINO model forward and GPU mask postprocessing. Checkpoint loading,
disk I/O, image decode, resize/normalization, host-to-device transfer, text
tokenization, mask serialization, metric accumulation, and distributed gather
are excluded.

Each one-node allocation runs 50 warm-ups and five rounds of 100 requests on
eight synchronized replicas: 4,000 raw samples per candidate. Median, p95, MAD,
IQR, robust CV, bootstrap interval, round drift, and device-spread gates must
pass.

## Automatic launch gate

The final campaign is sealed only after source, wheel, SDK, skills, SQSH,
dataset, BERT, PTM stage, and terminal qualification evidence all match their
frozen identities. Its runtime-local eligibility record deterministically
projects only exact successful, previously unverified records in memory;
`--automatic-trigger --launch` then starts the three independent mode
controllers without requesting confirmation.
Candidate zero in all three modes must pass full train, standalone evaluation,
latency, provenance, and audit gates before the remaining 23 recommendations
per mode are released automatically.

The v1 failure is immutable and cannot be overwritten or treated as v2
evidence. The v3 contract cannot be sealed before v2 terminal evidence exists.
Once sealed, its durable `--automatic-trigger --launch` watcher submits no
three-mode job unless the exact evidence, projected-registry digest, and every
other frozen prerequisite pass.

## Reproduction sequence

After an operator has independently established an SSHFS mount of the remote
`/lustre` root, run the data-only PTM stage locally (not as a SLURM job):

```bash
cd /localhome/local-rarunachalam/tao-automl
export PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH
export PYTHONPATH=$PWD:$PWD/src

python -m \
  experiments.cross_model_automl_20260729.mask_grounding_dino_coco2017_campaign.ptm_stage \
  --env-file /localhome/local-rarunachalam/.tao/config.env \
  --lustre-root \
  /lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/ptms/cross_model_automl_20260729/mask_grounding_dino_v1 \
  --physical-lustre-mount \
  /localhome/local-rarunachalam/.tao/mounts/slurm-lustre
```

The command fails closed unless the physical root is an active mount point.
When the repository and Python environment are available on a host with direct
`/lustre` access, omit `--physical-lustre-mount`; direct behavior and the
canonical publication root are unchanged. This preparation did not establish
a mount or execute the command.

The CLI records zero model, smoke, mini-step, GPU, and SLURM executions in its
secret-free summary. After this stage and a wheel from a clean reviewed commit:

```bash
cd /localhome/local-rarunachalam/tao-automl
export PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH

# Direct full qualification used the already sealed v2 contract: four
# concurrent one-node/eight-A100 jobs.
python -m \
  experiments.cross_model_automl_20260729.mask_grounding_dino_coco2017_campaign.qualification_campaign \
  --contract \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask_grounding_dino_coco2017_three_mode_v2/campaign.v2.json \
  --runtime-root \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask_grounding_dino_coco2017_ptm_qualification_v2 \
  --launch

# After completion.json exists, seal the evidence-bound v3 campaign from the
# clean source commit and matching wheel.
python -m \
  experiments.cross_model_automl_20260729.mask_grounding_dino_coco2017_campaign.manifest_generator \
  --repository /localhome/local-rarunachalam/.tao/worktrees/tao-automl-mgdino-eligibility-v3 \
  --output \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask_grounding_dino_coco2017_three_mode_v3/campaign.v3.json

# The trigger validates the exact local eligibility projection and launches
# all three objective-aware controllers automatically.
python -m \
  experiments.cross_model_automl_20260729.mask_grounding_dino_coco2017_campaign.run_campaign \
  --contract \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask_grounding_dino_coco2017_three_mode_v3/campaign.v3.json \
  --runtime-root \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/mask_grounding_dino_coco2017_three_mode_v3 \
  --automatic-trigger \
  --launch
```
