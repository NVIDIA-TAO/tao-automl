# Mask Grounding DINO / COCO 2017 campaign preparation report

## Verdict

The immutable four-PTM stage is complete. Qualification v1 submitted four
concurrent direct-full one-node/eight-A100 jobs (`31243535`–`31243538`). All
loaded the sealed PTM and dataset and failed on the first distributed training
batch because unused-parameter detection was disabled. No CPU/model smoke,
mini-step, or local model execution occurred. V1 remains immutable at SHA-256
`a48d8d8d2a5c65e35c9d39bd5ed1362be54e2be0b89dcda5471812da331a6996`.

Qualification v2 is a preregistered narrow correction. The pinned TAO config
does not accept the Lightning alias directly; it resolves
`distributed_strategy: ddp` plus `activation_checkpoint: false` to
`ddp_find_unused_parameters_true`. PTMs, data, fidelity, objectives, search
space, seeds, hardware, and SQSH are unchanged. The automatic three-mode
watcher stays fail-closed until v2 succeeds and exact PTMs are independently
promoted to repository status `supported`.

## Implementation audit

The exact TAO identifier is `mask_grounding_dino`. Training uses ODVG records
with masks; validation and test use COCO OD annotations. The TAO 7.1
`MaskGDINOPlModel` OD path constructs `OD_Evaluator` with `bbox` and `segm` IoU
types and emits `[segm] val_mAP@50-95` and `[segm] test_mAP@50-95`.
`segm_val_mAP50_95` is therefore registered as a supported fraction-scale
metric. VG `overall_IoU` remains a separate percent-scale
referring-expression metric.

The packaged schema fixes encoder and decoder depth at six, so neither is
searched. The allowed inner variables are `model.num_select`,
`train.optim.lr`, `train.optim.lr_backbone`, and
`train.optim.weight_decay`. PTM identity is represented as a hierarchical
non-ordinal arm.

## Frozen dataset and assets

| Evidence | Frozen value |
| --- | --- |
| Full COCO stage | `/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/coco2017_instance_panoptic_v1` |
| Full stage SHA-256 | `437ff12490637950707b9b951d820ea34d38b926080a478a5d182c2d284a0c5d` |
| 246,593-file manifest SHA-256 | `10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e` |
| Training ODVG JSONL SHA-256 | `d5deb4f5cfe027786fb1ceb52632ad6d3ef027e95e434525ba715d6841fb2921` |
| Training ODVG label map SHA-256 | `02075d96f6bf06d061f9329b4775dc7c3bb5ac140c77bc5c0e465d305c46d6c1` |
| Contiguous validation JSON SHA-256 | `9c9af9918e29292adfaa78a694d471e2be6d226e150300d9f4b22c2d77723ebc` |
| Contiguous conversion manifest SHA-256 | `3c2d09d20211017575a2c51a6797ef91f1939340d978a5d11d1d1edab1a30b2d` |
| Offline BERT tree SHA-256 | `04cd5cc67804f4752df93e7c05dd51d904e82fc05d28794ddb03504cca689fb5` |

The official TAO conversion retains 117,266 annotated training image records
and all 860,001 annotations/masks. The validation derivative retains all 5,000
images and 36,781 annotations/masks and changes only category IDs to contiguous
`0..79`. Repeated conversion was byte-identical. Both the source stage and
derivative are read-only.

## PTM inventory

Four official Swin-T records are now bound to TAO-7.1-compatible,
repository-owned path-free sidecars:

1. `mask_grounding_dino.commercial.swin_tiny.trainable.v2.1`
2. `mask_grounding_dino.commercial.swin_tiny.trainable.v2.0`
3. `mask_grounding_dino.commercial.swin_tiny.trainable.v1.0`
4. `mask_grounding_dino.research.swin_tiny.trainable.v2.0`

Their status intentionally remains `unverified`. The qualification controller
requires one real three-epoch full-COCO train plus standalone validation on one
node/eight A100s per arm. All four workflows run concurrently with independent
durable SDK state stores. Every unsuccessful arm is preserved as a terminal
exclusion. The repository records remain unchanged. After terminal v2 evidence
exists, the v3 campaign may project only exact successful unverified identities
to `supported` in a validated, evidence-bound in-memory registry. Ordinary
runtime still requires repository support; the campaign projection is never
persisted globally and cannot promote a failed or explicitly unsupported arm.

The repository includes a minimal data-only `ptm_stage.py` path for all
four arms. It uses `NGCHTTPSClient` and `AtomicArtifactCache`, verifies exact
immutable member identity, size, and checksum, and atomically publishes
read-only bytes plus the existing stage-manifest schema on Lustre. It is
create-or-verify idempotent and rejects unexpected content, symlinks, writable
completed artifacts, checksum drift, and manifest replacement. It contains no
model or scheduler import and records zero model and SLURM executions.

The publication root can now use direct `/lustre` access or an active SSHFS
mount of remote `/lustre`. In mapped mode the operator provides the canonical
`/lustre/...` stage root and the mount root only; the physical destination is
derived, verified to correspond, and never accepted as an independent
identity. Both manifest copies retain canonical `/lustre/...` checkpoint
paths. Unsafe canonical roots, inactive or symlinked mounts, path escapes,
non-corresponding roots, and non-inventory stage content fail closed. No mount
was used only for data staging; no model ran in that step.

## Three independent mode jobs

Each mode has 24 recommendations, three full epochs per candidate, its own
empty observation namespace, search seed `271828`, and training seed `1234`.

| Mode | Acquisition | Terminal policy |
| --- | --- | --- |
| Accuracy | Expected improvement | Highest valid mask AP |
| Latency | Constrained expected improvement with monotonic in-job quality reference | Raw-minimum-anchored equivalent-fastest cohort under 90% retained accuracy, then accuracy tie-break |
| Multi-objective | ParEGO expected improvement | Independent Pareto-rank-zero normalized augmented-Chebyshev compromise |

Latency and multi-objective constraints are independent. All eight
agent-intervention flags and all five selection-isolation flags are frozen
`false`.

## Latency and automatic release

The selection-time worker uses 16 immutable real validation images, the frozen
80-category prompt, batch size one, FP32, and eight synchronized replicas. The
timed scope is model forward plus GPU mask postprocessing. It runs 50 warm-ups
and five rounds of 100 requests for 4,000 samples per candidate and enforces the
standard dispersion, drift, bootstrap, and device-spread gates.

Once immutable prerequisites pass, the automatic trigger starts all three mode
controllers. Candidate zero in every mode must pass full training, standalone
evaluation, stabilized latency, provenance, and recommendation audit before
the remaining 23 recommendations per mode are automatically released.

## Current launch blockers

| Blocker | State |
| --- | --- |
| Four-checkpoint immutable PTM stage manifest | Complete and read-only |
| Direct-full qualification v1 | Preserved terminal first-batch DDP failure for 4/4 arms |
| Direct-full qualification v2 | Terminal completion evidence required before v3 sealing |
| Repository Mask Grounding DINO records | Preserved unchanged; 4/4 unverified before qualification |
| Runtime-local eligibility | Exact successful identities only; schema-v2 in-memory projection, failed/unsupported arms excluded |
| Clean v3 source commit and matching wheel | Required before automatic three-mode launch |
| Final `campaign.v3.json` | Generated only after binding exact v2 completion evidence |

The v1 failures are diagnostic qualification evidence, not benchmark results;
there is no valid Mask Grounding DINO accuracy or latency metric yet.

## Verification

Campaign contract tests cover the task-correct metric, dataset identities,
contiguous conversion, frozen BERT, four-arm registry inventory, search
parameters, independent acquisition semantics, PTM staging, direct-full
qualification, automatic release, real-input latency descriptor, evaluation
specification, and integrity failure paths.

The final static verification completed with:

```text
campaign-specific suite: 44 passed
full production suite: 970 passed, 1 skipped
complete cross-model experiment suite: 429 passed
python compilation: passed
git diff --check: passed
```

The three production-suite and three cross-model-suite warnings are the
established sklearn Gaussian-process convergence warnings; no additional
warning was observed. The source commit is recorded in the sealed v2 campaign
after this report is committed.
