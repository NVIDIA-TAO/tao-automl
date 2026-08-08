# Grounding DINO shared synthetic detection preparation

This directory prepares, but does not launch, the exact TAO model identifier
`grounding_dino` on the same immutable synthetic COCO detection dataset used
for DINO, Deformable DETR, and RT-DETR.

The source is a valid category-detection corpus. NVIDIA TAO Data Services can
derive ODVG detection records and a label map from the four COCO category
names, and can remap validation IDs from `1..4` to the contiguous `0..3`
contract required by Grounding DINO. The prompt list is derived verbatim and
in category-ID order:

`cone`, `forklift`, `cart`, `fire_extinguisher`

No synonym, phrase, candidate, PTM, or preferred label is injected by an
agent. The campaign is explicitly category-prompted open-vocabulary detection
with `val_mAP50`; it is not a referring-expression grounding campaign.

The source contains zero image `caption` fields and zero annotation
`tokens_positive` fields. It therefore cannot support a phrase-grounding or
`Pr@0.5` product claim. Production now has a separate supported
`grounding_dino`/`val_mAP50` object-detection metric policy; the
referring-expression `val_Pr@0.5` policy remains blocked. The two policies are
not aliases.

## Sealed shared-dataset view

The official `tao-dataservices` converters at revision
`dcea3a39bd3e4709e2325e4b61a4f179efebde4c` were run twice against the exact
DINO/DDETR/RT-DETR synthetic COCO annotations. Both runs were byte-identical.
The read-only published view is inside the existing dataset tree:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/
  tao_od_synthetic_full_dino_coco/grounding_dino_odvg_v1/
```

| Artifact | SHA-256 |
| --- | --- |
| `train/annotations_odvg.jsonl` | `abf109f0fcbabcbc185942399857582ab1db0b8664c3f166c220b9def936d3df` |
| `train/annotations_odvg_labelmap.json` | `50e203a52565c8c00f41454ad3656662c98cd2902bd6a02d35a2ab3059578d81` |
| `validation/annotations_remapped.json` | `621f89401b4ab219274486b6c392540772707bcc0e5983a40c5c359f7277737b` |

All 8,395 training annotations and all 2,186 validation annotations are
preserved. The official ODVG converter intentionally omits only the 49 source
training images with zero annotations. Their exact IDs are preserved in
`dataset_conversion.v1.json`; the report therefore claims annotation
losslessness, not image-count losslessness. Validation preserves all 353
images and remaps only category IDs from `1..4` to contiguous `0..3`.

Prepared execution is one direct full 10-epoch, one-node/eight-GPU
qualification per official PTM, followed only after evidence-backed PTM
promotion by three independent objective-aware AutoML jobs:

- accuracy: expected improvement on validation accuracy;
- latency: constrained expected improvement with a monotonically
  self-calibrated 90% retained-accuracy reference;
- multi-objective: ParEGO expected improvement over accuracy and latency.

Every model job uses the pinned TAO 7.1.0 RC245 `.sqsh` directly. There are no
CPU model runs, smoke runs, mini-steps, shared archives, manually injected
candidates, or scheduler submissions in this preparation.

`successor.contract.v1.json` prepares two full PTM qualification workflows,
one for each official Grounding DINO PTM, and three independent
algorithm-generated first-candidate mode pilots. Each train and standalone
evaluate action uses one node and all eight GPUs. Training selection reads
`val_mAP50`; standalone qualification reads `test_mAP50` from exact status
evidence and is prohibited from feeding AutoML selection.

The corrected RT-DETR release is now bound and all three of its first
candidates passed. The automatic trigger remains deliberately closed and
requires:

- a fresh corrected DDETR automatic release and all three passing
  first-candidate records (the preserved failed DDETR v2 runtime cannot count);
- both official PTMs and the `bert-base-uncased` cache staged and hashed before
  any GPU allocation;
- full ten-epoch train/validation plus standalone evaluation evidence for at
  least one official PTM.

Only then may the two PTM qualification workflows be submitted in parallel,
followed by one candidate per objective mode. The remaining 19 candidates per
mode are released automatically only when all three first-candidate gates
pass. No fallback, manual candidate, or manually selected PTM is permitted.

Generate and verify the immutable preparation record with:

```bash
PYTHONPATH=src \
  python -m experiments.cross_model_automl_20260729.grounding_dino_shared_detection.prepare_campaign \
  --check-only

PYTHONPATH=src \
  python -m experiments.cross_model_automl_20260729.grounding_dino_shared_detection.dataset_conversion \
  --check-only

PYTHONPATH=src \
  python -m experiments.cross_model_automl_20260729.grounding_dino_shared_detection.successor_contract \
  --check-only
```

The historical `campaign.preparation.v1.json` is retained unchanged. Its
preparation-only authorization fields describe that historical stage; later
qualification execution is recorded separately by the v2 runtime contract
and its immutable completion evidence.

## Future structured-config successor

The historical v1/v2 inputs and `successor.contract.v1.json` remain immutable.
New execution is bound separately by `campaign.inputs.v3.json` and
`successor.runtime.contract.v2.json`. The only accepted DDETR predecessor is:

```text
/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/
  deformable_detr_automl_synthetic_structured_config_fix_v1/
  first_candidate_gate/automatic_release.json
```

Its required manifest identity is
`d70063f3fc6c4ed7c44d8c7d979e2dc3ffc27f576ddd13cf000648a2c2a26e83`
from source head `8386f52`. The release depends only on the three
algorithm-generated candidate-zero gates. It explicitly does not wait for
DDETR candidates 1–19.

All auxiliary model bytes are already staged read-only on Lustre. No model was
loaded and no scheduler job was submitted while staging:

| Input | Immutable identity |
| --- | --- |
| Grounding DINO commercial Swin-T v1.0 | `20c3ea116d1b841063aa5efffdd386b3d85a1c35f2d702d3c95150ef1efead73` |
| Grounding DINO commercial Swin-T v1.1 | `8ea7e089e174e72a7fe57ff63cdba5e1e4994b159e41cf72122a7e0d841beaa6` |
| `google-bert/bert-base-uncased` revision | `86b5e0934494bd15c9632b12f734a8a67f723594` |
| BERT staged tree | `04cd5cc67804f4752df93e7c05dd51d904e82fc05d28794ddb03504cca689fb5` |
| Runtime-input stage record | `3b52818de9bd438330a8530c36c5e60c62fdb367b9f7ae93688eebafaa38db8f` |

The automatic watcher is fail-closed. It requires the exact fresh DDETR
candidate-zero release, the still-valid RT-DETR release, unchanged dataset and
metric contracts, and the sealed PTM/BERT record. It then runs two official
PTM qualifications in parallel. The first model operation is a real
one-node/eight-A100-or-H100, ten-epoch train/validation job using the pinned
RC245 SQSH, followed by standalone evaluation of one exact terminal
checkpoint. No CPU, smoke, mini-step, fallback PTM, or replacement workflow is
available.

The qualification completion artifact automatically records the qualified PTM
population and opens the algorithm-generated three-mode pilot handoff without
manual confirmation. Candidate values remain the responsibility of the
production search algorithm.

Verify the new immutable records without launching:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python -m \
  experiments.cross_model_automl_20260729.grounding_dino_shared_detection.runtime_input_stage \
  --inputs experiments/cross_model_automl_20260729/grounding_dino_shared_detection/campaign.inputs.v3.json \
  --output experiments/cross_model_automl_20260729/grounding_dino_shared_detection/runtime_inputs.stage.v1.json \
  --check-only

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python -m \
  experiments.cross_model_automl_20260729.grounding_dino_shared_detection.future_contract \
  --inputs experiments/cross_model_automl_20260729/grounding_dino_shared_detection/campaign.inputs.v3.json \
  --stage experiments/cross_model_automl_20260729/grounding_dino_shared_detection/runtime_inputs.stage.v1.json \
  --output experiments/cross_model_automl_20260729/grounding_dino_shared_detection/successor.runtime.contract.v2.json \
  --check-only
```

## Qualification-driven three-mode pilot

The integrated successor qualification completed both official PTM arms with
ten training epochs, ten validation records, standalone evaluation of each
exact terminal checkpoint, one node, and all eight A100 GPUs. Both arms are
eligible regardless of their observed metric:

| PTM | final `val_mAP50` | standalone `test_mAP50` |
| --- | ---: | ---: |
| `grounding_dino.commercial.swin_tiny.trainable.v1.0` | `0.15590265377426205` | `0.15600353298044012` |
| `grounding_dino.commercial.swin_tiny.trainable.v1.1` | `0.7452574943938746` | `0.7466380476209992` |

The two metrics are independently required to be finite and valid. They are
not required to be bit-equal; the qualification adapter records their signed
difference without applying a result-fitted tolerance. Terminal checkpoint
identity must match exactly between training and standalone evaluation.

The repository registry promotes exactly these two successful qualifications
to `supported` for TAO 7.1.0. The validation records are bound to:

- qualification completion canonical SHA-256
  `172688d1af2479886c46f55fa148bf43a7487b517b2ea145c3359136100de698`;
- qualification completion file SHA-256
  `d09b9940aaa98c4f1f5b24dd802546e6dad98b9cf34ff5ce1d8504c0652edb13`;
- automatic handoff canonical SHA-256
  `318d93fc04260650a67452eb00a710658744bf83fe3462115a9c320b12315ec8`;
- automatic handoff file SHA-256
  `82ee846b27b42bffeb559a88dcf18d353d72eea8cc1e9b0eae241725124807cf`.

`pilot.inputs.v1.json` freezes one portable, input-driven campaign:

- three independent Bayesian jobs and observation namespaces;
- 20 algorithm-generated candidates per mode;
- accuracy EI, constrained-latency EI with a self-calibrated 90% retention
  guard, and multi-objective ParEGO EI;
- hierarchical nonordinal PTM arms derived from all successful
  qualifications;
- the same shared synthetic dataset and ten-epoch training fidelity;
- one node and eight GPUs for training, standalone evaluation, and latency;
- the pinned RC245 SQSH, SDK, skills, offline BERT tree, and A100 hardware
  contract;
- 50 warm-ups and five rounds of 100 timed requests on each of eight
  replicas, yielding 4,000 latency samples per candidate;
- an automatic cross-mode candidate-zero gate that releases candidates 1–19
  only after all three candidate-zero records pass.

There is no CPU/model smoke path, manual candidate injection, manual PTM
exclusion, shared observation archive, or confirmation pause. Failed
recommendations are preserved and are not replaced.

After the controller commit is integrated into the clean source checkout,
seal the launch manifest outside the repository and start the automatic
handoff consumer:

```bash
RUNTIME_ROOT=/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/grounding_dino_three_mode_pilot_v1
mkdir -p "$RUNTIME_ROOT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python -m \
  experiments.cross_model_automl_20260729.grounding_dino_shared_detection.pilot_manifest \
  --inputs experiments/cross_model_automl_20260729/grounding_dino_shared_detection/pilot.inputs.v1.json \
  --output "$RUNTIME_ROOT/pilot.campaign.v1.json"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python -m \
  experiments.cross_model_automl_20260729.grounding_dino_shared_detection.pilot_campaign \
  --automatic-trigger \
  --inputs experiments/cross_model_automl_20260729/grounding_dino_shared_detection/pilot.inputs.v1.json \
  --manifest "$RUNTIME_ROOT/pilot.campaign.v1.json" \
  --runtime-root "$RUNTIME_ROOT" \
  --env-file /localhome/local-rarunachalam/.tao/config.env
```

The controller re-audits the completion, handoff, registry, source, SDK,
skills, SQSH, dataset, and latency-input hashes before constructing the SDK.
The automatic consumer then submits the three candidate-zero jobs in parallel
and continues with the frozen remaining budget without user confirmation.
This controller branch does not itself submit pilot jobs.
