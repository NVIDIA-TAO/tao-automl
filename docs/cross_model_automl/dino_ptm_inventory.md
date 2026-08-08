# DINO pretrained-model inventory

This documents the DINO portion of the schema-v1 repository-owned PTM
registry. Official NGC resource, version, member, and byte-size metadata were
resolved on 2026-07-29. It contains 31 trainable identities: five complete
DINO detector checkpoints and all 26 official DINO backbone-only artifacts.
Checkpoint SHA-256 values for the five detector artifacts come from previously
staged DINO validation evidence. NGC does not publish checkpoint SHA-256 for
the 26 backbone members, so none was invented during source discovery.
Subsequent checksum-gated qualification recorded the observed SHA-256 for the
NVImageNet ResNet50 member. No checkpoint was downloaded to create the
original source inventory.

Official sources:

- [DINO COCO PTMs](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_dino_coco)
- [DINO with foundation-model backbone](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dino_with_fm_backbone)
- [DINO noncommercial ImageNet backbones](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_dino_imagenet)
- [DINO commercial NVImageNet backbones](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_dino_nvimagenet)

## Complete-detector inventory

| Registry ID | Exact NGC version and member | Bytes | SHA-256 | Status for production resolution |
| --- | --- | ---: | --- | --- |
| `dino.coco.resnet50.trainable.v1.0` | `pretrained_dino_coco:dino_resnet_50_trainable_v1.0#dino_resnet50_ep12.pth` | 568,767,395 | `7a391fb84a18714b60258becdb512594ec54faff5dccbf17ca53c5d902137512` | `supported`; validated and default only for TAO 7.0.1 |
| `dino.coco.gcvit_tiny.trainable.v1.0` | `pretrained_dino_coco:dino_gc_vit_tiny_trainable_v1.0#dino_gcvit_tiny0_ep12.pth` | 577,043,347 | `6322af1a26eb025139bc7bfe32591e38fe2998c998e376bab7ca644261d2bfbe` | `unsupported`; TAO 7.0.1 rejected GCViT and pinned 7.1 source has no GCViT construction path |
| `dino.coco.fan_small.trainable.v1.0` | `pretrained_dino_coco:dino_fan_small_trainable_v1.0#dino_fan_small_ep12.pth` | 580,862,106 | `df3e4e07d411f3d61c882ee9e61d5cb7cef613d1e4748d75b9b86e3dd1c83185` | `unverified`; the 7.1 adapter permits explicit target-release qualification, not runtime use |
| `dino.coco.fan_large.trainable.v1.0` | `pretrained_dino_coco:dino_fan_large_trainable_v1.0#dino_fan_large_imagenet22k_36ep.pth` | 1,197,490,926 | `8e9f8d865a315a40d4854cc2846317a03ca1f87a40a9ddde6fa7631828f73cb8` | `unverified`; the 7.1 adapter permits explicit target-release qualification, not runtime use |
| `dino.coco.nvdinov2_large.trainable.v1.0` | `dino_with_fm_backbone:trainable_v1.0#dino_nvdinov2_518_1536_coco_e36.pth` | 4,232,490,433 | `013e8e6e6a0a913ac56cf1a581f9d1dd7abe6cb47a57664a21290d0f44866c78` | `unverified`; immutable NGC metadata and checksum are recorded, but load, mini-step, and TAO-version preflight remain pending |

The compatibility status is intentionally fail-closed. Unsupported and
unverified entries remain visible as structured exclusions, but are never
eligible categorical search values. No compatibility with a TAO release other
than the tested 7.0.1 ResNet50 path is implied.

## Backbone-only inventory and completeness

The production registry mirrors every trainable member from the official
source inventory:

| Resource | Official versions | Registry records | Status policy | Checkpoint target |
| --- | ---: | ---: | --- | --- |
| `pretrained_dino_imagenet` | 16 | 16 | 9 FAN `unverified`; 7 GCViT `unsupported` | `model.pretrained_backbone_path` |
| `pretrained_dino_nvimagenet` | 10 | 10 | 1 ResNet + 4 FAN `unverified`; 5 GCViT `unsupported` | `model.pretrained_backbone_path` |

Each record preserves the exact version, member, byte size, immutable NGC
identity, DINO runtime backbone key, and official pretraining resolution. Each
has its own checksum-bound repository sidecar. The complete 35-member source
inventory, including exact tables and normalization evidence, is
`experiments/cross_model_automl_20260729/dino_ptm_inventory/source_inventory.v1.json`.

Four official ONNX deployables are not training PTMs and are deliberately not
duplicated into this registry. Their explicit non-training classification in
the source inventory proves that the difference between 35 official members
and 31 production registry records is not a silent omission.

Across the complete source inventory there are 14 GCViT artifacts. Thirteen
are trainable and are represented as `unsupported` registry records; the
fourteenth is the GCViT deployable and is recorded as structurally excluded in
the source inventory. The frozen static evidence is bound to TAO
7.1.0-rc-245 SQSH SHA-256
`0dfaf824025b69e66c0c239ec6e11e64f6b2eecdcd42b41f9226e14f1348568e`.

The TAO 7.0.1 runtime used for the supported record has SQSH SHA-256
`88ba75e3a8eb9524fc0dbf026f2ea5da2c68696ae8d918b0afde5e0384ca641e`.
The committed validation evidence is
`experiments/dino_moo_phase2_20260728/phase2_validation_report.md`.

## Version-scoped TAO 7.1 artifact adaptation

The registry records a declarative TAO 7.1 adaptation recipe for ResNet50,
FAN-small, FAN-large, and NVDINOv2-large. It retains only the checkpoint's
`state_dict` top-level entry and adds `tao_model: dino`. The tensor key set and
tensor values must remain exact. The registry contains no shell command,
Python module, or other executable adapter reference: production preflight
verifies the official NGC input first and then invokes a model-owned callback
supplied by the TAO integration.

| Registry ID | Pinned TAO 7.1 output | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `dino.coco.resnet50.trainable.v1.0` | `tao71_dino_resnet50_ep12.pth` | 195,109,331 | `678064a0706ec778edb17583be78e9a138afac1c48832ba419b8c774ac7d5756` |
| `dino.coco.fan_small.trainable.v1.0` | `tao71_dino_fan_small_ep12.pth` | 193,716,107 | `0a9e5ebfba383bbba8084db72a595bac2be512742998a2b4c0168b4300f3b580` |
| `dino.coco.fan_large.trainable.v1.0` | `tao71_dino_fan_large_imagenet22k_36ep.pth` | 399,422,743 | `149b670a4ca0cb701bdd32c69244593f2d6c699fd0d8b1851a9ad385434c7303` |
| `dino.coco.nvdinov2_large.trainable.v1.0` | `tao71_dino_nvdinov2_518_1536_coco_e36.pth` | 1,410,846,731 | `d7bacddff9393d5f37ecca67686467bce2cd77d95b26c6a01908a90dbc6b6333` |

These output identities were reproduced byte-for-byte in two independent
runs of the exact pinned TAO 7.1 image. They replace older wrapper identities
whose zip archive roots encoded a different serializer staging filename;
tensor key and value digests remained exact.
The machine-readable run identities, source and worker hashes, tensor
digests, and isolation flags are frozen in
`experiments/cross_model_automl_20260729/dino_ptm_qualification/serializer_qualification.v1.json`.
The first failed qualification and correction boundary are documented in
`docs/cross_model_automl/dino_ptm_qualification_correction.md`.

Adapted outputs enter the cache only by atomic replacement after registered
size and SHA-256 verification. Provenance binds the recipe, official input
size and SHA-256, adapted output size and SHA-256, and callback-supplied
tensor-key and tensor-value digests. The load-smoke callback receives the
adapted artifact, not the official source artifact.

These records are qualification infrastructure, not compatibility claims.
They do not change any checkpoint status or TAO compatibility range, and
successful qualification remains `runtime_eligible: false`. A missing
callback, ambiguous version match, output mismatch, or tensor-preservation
mismatch is a structured exclusion. GCViT has no wrapper because the preserved
TAO 7.1 preflight failed earlier with an unsupported-backbone
`NotImplementedError`.

## Repository sidecars

NGC exposes the DINO checkpoint members but not checkpoint-specific YAML
members for these versions. Therefore, the registry packages deterministic
sidecars derived from the corresponding official model-card transfer-learning
templates and the TAO DINO backbone contract. The five detector records retain
their existing sidecars:

| Registry ID | Sidecar SHA-256 |
| --- | --- |
| `dino.coco.resnet50.trainable.v1.0` | `35334ee26ad716deed83bcae737f2dee832569b4deacb33fbff9883e9f8270c9` |
| `dino.coco.gcvit_tiny.trainable.v1.0` | `e90725346e0ad21da63b531ab0ee6cd7f12175fab4a7f7981aeadc4ec3f84498` |
| `dino.coco.fan_small.trainable.v1.0` | `0ee49339b12f477cf1b79ae0a38553e0b2aa550b14351329938583940397a1d5` |
| `dino.coco.fan_large.trainable.v1.0` | `731c8b0a140580199cfa043d1ea674bbec9cbb7a269335b847bad05244e41ae4` |
| `dino.coco.nvdinov2_large.trainable.v1.0` | `cde09655633b228f8e89f79efd04f0f2182d573e18397c47fbd4e40d51907439` |

The published NVDINOv2 transfer snippet names `vit_large_dinov2`, but that
constructor uses standard 4096-wide MLP blocks and is shape-incompatible with
the checkpoint's 5472-wide SwiGLU blocks. The model card, resource identity,
checkpoint tensors, and pinned TAO 7.1 implementation agree on
`vit_large_nvdinov2`; qualification therefore corrects only that backbone key.

Sidecar values are PTM defaults, not user policy. The deterministic merge
order remains:

```text
TAO model defaults
< PTM sidecar overrides
< AutoML profile overrides
< user experiment overrides
< algorithm-generated candidate values
```

The registry loader verifies the supported sidecar before advertising the
ResNet50 checkpoint. The 26 backbone-only records each add a record-specific
sidecar under `data/ptm_specs/dino/`. Tests verify every one of the 31 DINO
sidecar digests and require parsed YAML to equal registered defaults exactly.

The COCO and ImageNet model cards identify CC-BY-NC-SA-4.0. The NVImageNet
model card instead identifies the NVIDIA AI Enterprise Model EULA; the
registry preserves that distinction.
Runtime preflight must still enforce credentials, access, member size,
artifact checksum, YAML merge, and load/mini-step gates before an unverified
record can become supported.
