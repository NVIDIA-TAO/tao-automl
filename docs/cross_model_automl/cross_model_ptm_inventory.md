# Cross-model PTM discovery inventory

The schema-v1 registry contains an official, fail-closed discovery inventory
for the seven target models beyond DINO. Metadata was resolved from exact
`nvidia/tao` NGC resource versions and members on 2026-07-29. These entries
support qualification planning only:

- every model has `default_ptm: null`;
- every checkpoint has `status: unverified`;
- none can enter production runtime resolution or AutoML categorical search;
- compatibility metadata records the intended TAO release only; it does not
  qualify a checkpoint for runtime use;
- an NGC `sha256_base64` value is decoded to lowercase 64-hex SHA-256 when the
  API exposes it, and the checksum is omitted when NGC does not publish one.

No checkpoint was downloaded and no training, inference, latency, local-GPU,
or SLURM job was launched while constructing this inventory.

## Coverage

| Model key | Official NGC resources | Trainable records | Checkpoint SHA-256 published | Repository sidecars | Principal gate still open |
| --- | --- | ---: | ---: | ---: | --- |
| `deformable_detr` | `pretrained_deformable_detr_coco` | 2 | 0 | 2 | TAO load/mini-step; NGC checkpoint checksums absent |
| `rtdetr` | `trafficcamnet_transformer_lite`, `rtdetr_2d_warehouse` | 4 | 4 | 4 | TAO load/mini-step |
| `grounding_dino` | `grounding_dino` | 2 | 1 | 2 | TAO load/mini-step; v1.0 checkpoint checksum absent |
| `segformer` | `pretrained_segformer_cityscapes`, `pretrained_segformer_imagenet` | 13 | 0 | 0 | Exact checkpoint/YAML merge; checksum absence; ImageNet card license inconsistency |
| `oneformer` | `oneformer`, `oneformer_its_pretrained_commercial` | 4 | 4 | 4 | Direct full-COCO train/eval qualification; all records remain unverified |
| `mask2former` | `mask2former` | 1 | 0 | 1 | TAO load/mini-step; checkpoint checksum absent |
| `mask_grounding_dino` | `mask_grounding_dino`, `pretrained_mask_grounding_dino_v2` | 4 | 3 | 0 | Exact checkpoint YAML and TAO load/mini-step |

The registry is the exact machine-readable record of every version, member,
byte size, checksum, architecture, input contract, checkpoint target, task,
and license that was safe to assert. Missing fields are deliberate rather
than implied defaults.

## Exact official versions

### Detection and grounding

| Model | Version | Member | Bytes |
| --- | --- | --- | ---: |
| Deformable DETR | `ddetr_resnet_50_trainable_v1.0` | `dd_resnet50_ep50.pth` | 492,568,963 |
| Deformable DETR | `ddetr_gc_vit_tiny_trainable_v1.0` | `dd_gcvit_tiny_ep50.pth` | 497,618,156 |
| RT-DETR | `trainable_resnet50_v2.0` | `resnet50_trafficcamnet_rtdetr.pth` | 511,956,488 |
| RT-DETR | `trainable_resnet18_v2.0` | `resnet18_trafficcamnet_rtdetr.pth` | 357,560,178 |
| RT-DETR | `trainable_rn50_v1.0.2` | `rtdetr_warehouse_v1.0.2.pth` | 514,392,577 |
| RT-DETR | `trainable_efficientvit_l2_v1.0` | `rtdetr_warehouse_v1.0.pth` | 813,085,917 |
| Grounding DINO | `grounding_dino_swin_tiny_commercial_trainable_v1.1` | `grounding_dino_swin_tiny_commercial_trainable.pth` | 2,070,860,191 |
| Grounding DINO | `grounding_dino_swin_tiny_commercial_trainable_v1.0` | `grounding_dino_swin_tiny_commercial_trainable.pth` | 2,070,704,394 |

### Segmentation

The SegFormer inventory has four complete Cityscapes checkpoints (FAN
Tiny/Small/Base/Large) and nine ImageNet-pretrained FAN backbone checkpoints.
The latter are recorded with
`checkpoint_target: model.backbone.pretrained_backbone_path`, not as complete
SegFormer model checkpoints. Their model card contains conflicting license
sections, so the registry deliberately omits a normalized license until that
source ambiguity is resolved.

| Model | Version | Member | Bytes |
| --- | --- | --- | ---: |
| OneFormer | `oneformer_ade_pretrained_research_trainable_v1.0` | `model_epoch_003_step_02528.pth` | 2,840,732,008 |
| OneFormer | `oneformer_coco_pretrained_research` | `oneformer_pretrained_research.pth` | 2,840,372,264 |
| OneFormer | `oneformer_pretrained_commercial_dinat_its` | `oneformer_pretrained_commercial_dinat_its.pth` | 2,887,215,443 |
| OneFormer | `oneformer_its_swinl_commercial_trainable_v1.0` | `its_miou=82.pth` | 2,834,018,088 |
| Mask2Former | `mask2former_swint_trainable_v1.0` | `mask2former_swint.pth` | 569,716,712 |
| Mask Grounding DINO | `mask_grounding_dino_swin_tiny_commercial_trainable_v2.1` | `model_epoch_029_step_20642.pth` | 2,216,696,690 |
| Mask Grounding DINO | `mask_grounding_dino_swin_tiny_commercial_trainable_v2.0` | `model_epoch_005_step_09942.pth` | 2,216,696,690 |
| Mask Grounding DINO | `mask_grounding_dino_swin_tiny_commercial_trainable_v1.0` | `model_epoch_049.pth` | 718,739,024 |
| Mask Grounding DINO | `mask_grounding_dino_swin_tiny_research_trainable_v2.0` | `model_epoch_021_step_35970.pth` | 2,216,677,507 |

## Sidecar policy

Thirteen deterministic repository sidecars were added only where an official NGC
`experiment.yaml` or a model-card transfer-learning template supplied
checkpoint-specific architecture values:

- Deformable DETR ResNet50 and GCViT-Tiny;
- all four RT-DETR records;
- both Grounding DINO records;
- all four OneFormer records, projected from the corresponding official TAO
  `spec_ade.yaml`, `spec_coco.yaml`, `spec_its_dinat.yaml`, and
  `spec_its_swin.yaml` architecture sections;
- Mask2Former Swin-Tiny.

Official YAMLs containing private dataset, checkpoint, results, or Lustre
paths were not copied. Sidecars are curated architecture-only projections,
bind their provenance to the official YAML SHA-256, and are tested to contain
none of those path prefixes. Their parsed YAML must equal
`default_spec_overrides` exactly.

SegFormer and Mask Grounding DINO remain partial when the official
resource exposes no checkpoint-specific YAML or the repository cannot yet
represent the checkpoint's exact conditional configuration. Their structured
reasons preserve that distinction and prevent accidental qualification.

The OneFormer projections resolve architecture representation only. They do
not establish checkpoint load compatibility or metric correctness. In the
current no-smoke execution profile, promotion requires a stronger direct
full-dataset one-node/eight-GPU training plus standalone-evaluation workflow;
no CPU model smoke or mini-step can qualify an arm.

Static inspection of the pinned TAO 7.1 SQSH found a direct runtime blocker:
the OneFormer training entrypoint calls
`OneformerPlModule.load_pretrained_weights` for a full checkpoint, while that
method is absent from the packaged `OneformerPlModule`. The same packaged
evaluation path reports semantic `mIoU`, not Panoptic Quality, and writes its
status KPI from per-process aggregates. OneFormer records therefore remain
`unverified`; no campaign launch is authorized until the full-checkpoint load
path and task-correct distributed metric path are resolved and directly
qualified.

## Qualification boundary

The records with complete path-free sidecars can be enumerated only through
the explicit qualification API. That API remains non-runtime and does not
change `status`. Partial entries remain structured qualification exclusions
until their missing exact configuration or license evidence is supplied.

Promotion to `supported` requires, at minimum:

1. credential and member-access verification;
2. size and SHA-256 verification, computing and recording a digest when NGC
   does not publish one;
3. exact TAO-version and container identity;
4. deterministic sidecar/user/candidate merge;
5. checkpoint load;
6. one training, validation, and inference mini-step;
7. checkpoint save/reload;
8. finite task-correct metric and latency instrumentation.

Checksums alone never promote a checkpoint.
