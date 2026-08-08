# Official DINO NGC source inventory

## Result

The four official `nvidia/tao` resources expose 35 exact versions, each with
one member, totaling 15,621,218,789 bytes:

| Official resource | Source classification | Versions | Full detector trainable | Deployable | Backbone only |
| --- | --- | ---: | ---: | ---: | ---: |
| `pretrained_dino_coco` | full detector | 7 | 4 | 3 | 0 |
| `dino_with_fm_backbone` | full detector | 2 | 1 | 1 | 0 |
| `pretrained_dino_imagenet` | backbone only | 16 | 0 | 0 | 16 |
| `pretrained_dino_nvimagenet` | backbone only | 10 | 0 | 0 | 10 |
| **Total** | | **35** | **5** | **4** | **26** |

This is a source inventory, not a runtime-qualification result. It records
official immutable `resource:version#member` identities and byte sizes without
downloading checkpoint payloads. Non-GCViT trainable records remain
`catalogued_requires_preflight`; they are not promoted to runtime-supported by
metadata alone.

The machine-readable source of truth is
[`source_inventory.v1.json`](source_inventory.v1.json). Its canonical
35-record projection has SHA-256:

```text
715942c92b7ee5523cc421621913e7715cdd1abe9713240ccfb70d117d133ea7
```

## Exact official inventory

### `pretrained_dino_coco`

All members are complete DINO detector artifacts. Trainable `.pth` members map
to `train.pretrained_model_path`; deployable `.onnx` members are not AutoML
training PTMs. The detector input contract is `B x 3 x 544 x 960`.

| Exact version | Exact member | Bytes | Role | Runtime backbone |
| --- | --- | ---: | --- | --- |
| `dino_fan_large_trainable_v1.0` | `dino_fan_large_imagenet22k_36ep.pth` | 1,197,490,926 | trainable detector | `fan_large` |
| `dino_fan_small_deployable_v1.0` | `dino_fan_small_ep12.onnx` | 215,567,293 | deployable detector | `fan_small` |
| `dino_fan_small_trainable_v1.0` | `dino_fan_small_ep12.pth` | 580,862,106 | trainable detector | `fan_small` |
| `dino_gc_vit_tiny_deployable_v1.0` | `dino_gcvit_tiny0_ep12.onnx` | 234,840,909 | deployable detector | `gc_vit_tiny` |
| `dino_gc_vit_tiny_trainable_v1.0` | `dino_gcvit_tiny0_ep12.pth` | 577,043,347 | trainable detector | `gc_vit_tiny` |
| `dino_resnet_50_deployable_v1.0` | `dino_resnet50_ep12.onnx` | 209,270,924 | deployable detector | `resnet_50` |
| `dino_resnet_50_trainable_v1.0` | `dino_resnet50_ep12.pth` | 568,767,395 | trainable detector | `resnet_50` |

The model card names `dino_fan_large_deployable_v1.0`, but the official
versions API does not return that version. It is therefore not invented or
included.

### `dino_with_fm_backbone`

The frozen metadata-only inventory records runtime backbone
`vit_large_dinov2`, exactly as prescribed by the official transfer-learning
YAML available during discovery. Subsequent checksum-gated TAO 7.1
qualification found that this published key is inconsistent with the
checkpoint's 5472-wide SwiGLU tensors. The production registry therefore uses
TAO's supported `vit_large_nvdinov2` constructor, which matches 930/930 target
tensors. The source-inventory JSON remains unchanged so it continues to
preserve the original published metadata. Their detector input contract is
`B x 3 x 1536 x 1536`.

| Exact version | Exact member | Bytes | Role | Checkpoint target |
| --- | --- | ---: | --- | --- |
| `deployable_v1.0` | `dino_nvdinov2_518_1536_coco_e36_op17.onnx` | 1,473,738,476 | deployable detector | n/a |
| `trainable_v1.0` | `dino_nvdinov2_518_1536_coco_e36.pth` | 4,232,490,433 | trainable detector | `train.pretrained_model_path` |

The model card uses the labels `dino_dinov2_trainable_v1.0` and
`dino_dinov2_deployable_v1.0`; the exact official API version IDs are
`trainable_v1.0` and `deployable_v1.0`, which this inventory preserves.

### `pretrained_dino_imagenet`

These members initialize only the DINO feature extractor. They map to
`model.pretrained_backbone_path`, not `train.pretrained_model_path`.
Resolution below is checkpoint pretraining resolution, not the downstream
DINO detector input.

| Exact version | Exact member | Bytes | Runtime backbone | Resolution |
| --- | --- | ---: | --- | ---: |
| `gcvit_large_imagenet22k_384` | `gcvit_large_imagenet22k_384.pth` | 864,054,197 | `gc_vit_large_384` | 384 |
| `gcvit_large_imagenet1k` | `gcvit_large_imagenet1k.pth` | 814,560,029 | `gc_vit_large` | 224 |
| `gcvit_base_imagenet1k` | `gcvit_base_imagenet1k.pth` | 367,540,965 | `gc_vit_base` | 224 |
| `gcvit_small_imagenet1k` | `gcvit_small_imagenet1k.pth` | 210,626,653 | `gc_vit_small` | 224 |
| `gcvit_tiny_imagenet1k` | `gcvit_tiny_imagenet1k.pth` | 119,124,861 | `gc_vit_tiny` | 224 |
| `gcvit_xtiny_imagenet1k` | `gcvit_xtiny_imagenet1k.pth` | 82,106,937 | `gc_vit_xtiny` | 224 |
| `gcvit_xxtiny_imagenet1k` | `gcvit_xxtiny_imagenet1k.pth` | 50,020,581 | `gc_vit_xxtiny` | 224 |
| `fan_hybrid_tiny` | `fan_hybrid_tiny.pth` | 30,032,227 | `fan_tiny` | 224 |
| `fan_hybrid_small` | `fan_hybrid_small.pth` | 104,729,466 | `fan_small` | 224 |
| `fan_hybrid_large_in22k` | `fan_hybrid_large_in22k.pth` | 342,977,114 | `fan_large` | 224 |
| `fan_hybrid_large_in22k_384` | `fan_hybrid_large_in22k_384.pth` | 308,027,006 | `fan_large` | 384 |
| `fan_hybrid_large_in22k_1k` | `fan_hybrid_large_in22k_1k.pth` | 308,026,197 | `fan_large` | 224 |
| `fan_hybrid_large_in22k_1k_384` | `fan_hybrid_large_in22k_1k_384.pth` | 308,029,433 | `fan_large` | 384 |
| `fan_hybrid_base_in22k` | `fan_hybrid_base_in22k.pth` | 234,955,105 | `fan_base` | 224 |
| `fan_hybrid_base_in22k_1k` | `fan_hybrid_base_in22k_1k.pth` | 202,329,016 | `fan_base` | 224 |
| `fan_hybrid_base_in22k_1k_384` | `fan_hybrid_base_in22k_1k_384.pth` | 202,331,436 | `fan_base` | 384 |

The card describes `fan_hybrid_base_in22k` as FAN-Hybrid-Small despite its
exact Base version/member identity. The runtime mapping follows the immutable
identity and records the card inconsistency.

### `pretrained_dino_nvimagenet`

These are also backbone-only artifacts targeting
`model.pretrained_backbone_path`. All listed members use a 224-pixel
checkpoint pretraining resolution.

| Exact version | Exact member | Bytes | Runtime backbone |
| --- | --- | ---: | --- |
| `resnet50` | `resnet50_nvimagenetv2.pth.tar` | 307,117,121 | `resnet_50` |
| `fan_large_hybrid_nvimagenet` | `fan_large_hybrid_nvimagenet.pth` | 308,027,815 | `fan_large` |
| `fan_small_hybrid_nvimagenet` | `fan_small_hybrid_nvimagenet.pth` | 104,734,139 | `fan_small` |
| `fan_base_hybrid_nvimagenet` | `fan_base_hybrid_nvimagenet.pth` | 202,330,226 | `fan_base` |
| `gcvit_base_nvimagenet` | `gcvit_base_nvimagenet.pth` | 367,540,965 | `gc_vit_base` |
| `gcvit_small_nvimagenet` | `gcvit_small_nvimagenet.pth` | 210,626,653 | `gc_vit_small` |
| `gcvit_tiny_nvimagenet` | `gcvit_tiny_nvimagenet.pth` | 119,124,861 | `gc_vit_tiny` |
| `gcvit_xtiny_nvimagenet` | `gcvit_xtiny_nvimagenet.pth` | 82,106,937 | `gc_vit_xtiny` |
| `gcvit_xxtiny_nvimagenet` | `gcvit_xxtiny_nvimagenet.pth` | 50,020,581 | `gc_vit_xxtiny` |
| `fan_hybrid_tiny_nvimagenet` | `fan_hybrid_tiny_nvimagenetv2.pth.tar` | 30,046,459 | `fan_tiny` |

The card describes `fan_large_hybrid_nvimagenet` as FAN-Hybrid-Base despite
its exact Large version/member identity. The runtime mapping follows the
immutable identity and preserves the discrepancy.

## Pinned TAO 7.1 GCViT exclusion

This exclusion was established by static source inspection only; no checkpoint
was downloaded and no model was run.

| Evidence | Immutable value |
| --- | --- |
| Container | `nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.1.0-rc-245-multiarch` |
| SQSH SHA-256 | `0dfaf824025b69e66c0c239ec6e11e64f6b2eecdcd42b41f9226e14f1348568e` |
| SQSH bytes | 28,860,358,656 |
| Source member | `/usr/local/lib/python3.12/dist-packages/nvidia_tao_pytorch/cv/dino/model/backbone.py` |
| Source SHA-256 | `4be339eb5791d168cf295e8ebb4bb2c32439399577848e9d4c4731f799e012c7` |

In that exact source:

- lines 21–40 import and define ResNet, FAN, ViT, Swin, ViTDet, and
  EfficientViT construction paths, but no GCViT path;
- lines 143–230 construct the supported architecture set and dispatch
  branches, again with no GCViT branch;
- line 230 raises `NotImplementedError` for an unmatched backbone.

Consequently, all 14 catalog records mapped to `gc_vit_*` are structurally
excluded from this pinned 7.1 runtime before checkpoint loading. This is a
container-specific conclusion, not a claim that the official artifacts are
invalid or that a different TAO runtime cannot support them.

## Collection and reproducibility

Metadata came from the authenticated official endpoints:

```text
GET https://api.ngc.nvidia.com/v2/org/nvidia/team/tao/models/{resource}/versions
GET https://api.ngc.nvidia.com/v2/org/nvidia/team/tao/models/{resource}/versions/{version}/files
```

Only resource, exact version, member, and `sizeInBytes` were projected.
Credentials, request IDs, dates, signed redirects, and other volatile response
fields were not persisted.

Run the frozen inventory contracts with:

```bash
cd /localhome/local-rarunachalam/tao-automl
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  pytest -q \
  experiments/cross_model_automl_20260729/dino_ptm_inventory/test_source_inventory.py
```

The contracts assert exact completeness, source order, identity
deduplication, runtime mappings, static exclusion provenance, secret-free
metadata URLs, and canonical hash reproducibility.
