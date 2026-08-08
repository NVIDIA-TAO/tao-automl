# DINO PTM CPU qualification correction

## Immutable v1 evidence

The first CPU qualification is an immutable diagnostic record at:

```text
~/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/cpu/
```

Its identities are:

| Artifact | Semantic SHA-256 | Raw file SHA-256 |
| --- | --- | --- |
| `qualification_manifest.v1.json` | `4d24e37aad64a0916bc7d34e295b4eb33fd71ccdd3df7f873e41047070cc4933` | `41d80990ecc8ea816758522924815e529bc3af286ec80c8efb01bd0771d8e934` |
| `qualification_completion.v1.json` | `354ade6f10b0aecc0478ed1e48eea5d1127566ca0505728bf82b17997889838a` | `2647999063aebc5e2a687bb7895f956a3f5ed7088dc081e9e56e95a29199f80c` |

The completion embeds production-report semantic SHA-256
`077f41dedfb4670899983bb84dfbd97999d024aa3bbbacb1f872b370982e0b3c`.
It records complete accounting: 18 registry-derived TAO 7.1 qualification
candidates were evaluated, 11 passed, and seven failed. Thirteen separately
registered GCViT identities were structurally excluded before evaluation
because their registry status was `unsupported`. Thus all 31 official
trainable DINO identities are accounted for.

The v1 completion is evidence of what the pinned implementation observed, not
evidence to overwrite after diagnosis. Its recorded audit flags remain:

```text
qualification_only = true
selection_invoked = false
runtime_eligibility_mutated = false
agent_selected_checkpoint = false
```

No AutoML candidate was generated or selected, no checkpoint was manually
promoted, and no winner was chosen. The 18-member population came from the
registered `supported` and `unverified` statuses; the 13 structural exclusions
came from the registered `unsupported` status. The campaign intervention
flags therefore remain false:

```text
agent_selected_candidate = false
agent_injected_candidate = false
agent_modified_search_space_after_results = false
agent_changed_seed_after_results = false
agent_changed_budget_after_results = false
agent_changed_threshold_after_results = false
agent_changed_ptm_after_results = false
agent_overrode_winner = false
```

## Root-cause corrections

### Scalar tensor hashing

The projection worker hashed tensor values by applying a dtype-changing
`view(torch.uint8)` directly to each tensor. PyTorch 2.6 rejects that operation
for a zero-dimensional tensor, such as BatchNorm `num_batches_tracked`. This
caused the opaque `docker_projection_failed` results for the full FAN-small,
FAN-large, and NVDINO detector checkpoints. The corrected path first flattens
the logical tensor, then views its bytes:

```text
tensor.detach().cpu().contiguous().reshape(-1)
      .view(torch.uint8).reshape(-1)
```

This preserves the byte-exact framing and digest algorithm for non-scalars
while making scalar hashing well-defined. A regression test exercises a
zero-dimensional `torch.int64` value whose direct dtype-changing view fails.

### Pinned serializer identities

The old four wrapper identities were produced with a serializer staging name
different from the one pinned by the production projection callback. PyTorch's
zip serialization includes that archive root, so the file SHA-256 and byte
count changed even though the logical tensor-key and tensor-value digests were
identical. Two independent runs in the exact pinned TAO 7.1 image produced
byte-identical outputs with these identities:

| Checkpoint | Output member | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `dino.coco.resnet50.trainable.v1.0` | `tao71_dino_resnet50_ep12.pth` | 195,109,331 | `678064a0706ec778edb17583be78e9a138afac1c48832ba419b8c774ac7d5756` |
| `dino.coco.fan_small.trainable.v1.0` | `tao71_dino_fan_small_ep12.pth` | 193,716,107 | `0a9e5ebfba383bbba8084db72a595bac2be512742998a2b4c0168b4300f3b580` |
| `dino.coco.fan_large.trainable.v1.0` | `tao71_dino_fan_large_imagenet22k_36ep.pth` | 399,422,743 | `149b670a4ca0cb701bdd32c69244593f2d6c699fd0d8b1851a9ad385434c7303` |
| `dino.coco.nvdinov2_large.trainable.v1.0` | `tao71_dino_nvdinov2_518_1536_coco_e36.pth` | 1,410,846,731 | `d7bacddff9393d5f37ecca67686467bce2cd77d95b26c6a01908a90dbc6b6333` |

Those diagnostic temporary outputs are not qualification artifacts. The next
create-only qualification supplies authoritative wrapper evidence.

### The two ResNet50 cases

The NVImageNet backbone checkpoint
`dino.backbone.nvimagenet.resnet50` contains an `argparse.Namespace` of
training arguments beside its tensor state. The pinned `weights_only=True`
loader correctly rejected that global in v1. The correction does not disable
restricted loading: it permits only `argparse.Namespace`, only for that stable
checkpoint ID, and only when the downloaded artifact SHA-256 is exactly
`49b0df2b517a28760e17158c9ad78371c1f833d6ad257f117ff81356743060b7`.
The corrected diagnostic load matches 318/318 target tensors and
23,561,205/23,561,205 target parameters.

The complete detector checkpoint
`dino.coco.resnet50.trainable.v1.0` had a different failure: v1 generated
195,109,331 bytes while the registry expected the old 195,112,691-byte
wrapper. With the pinned serializer identity above, its diagnostic load
matches 626/679 target tensors and 48,715,411/48,715,464 target parameters
(tensor fraction `0.9219440353`, parameter fraction `0.9999989120`). This is
above the unchanged full-detector coverage gate; no load threshold was
relaxed.

### FAN coverage

After the scalar-hash correction, the full FAN-small detector matches 775/780
target tensors (tensor fraction `0.9935897436`) and 99.3206% of target
parameters. The full FAN-large detector matches 1,165/1,170 target tensors
(tensor fraction `0.9957264957`) and 99.6462% of target parameters. These
diagnostics explain the v1 projection failures without changing the coverage
policy.

The two FAN-tiny backbone failures are legitimate exclusions, not projection
defects. Both load and match 351/361 target tensors (tensor fraction
`0.9722991689750693`) but only `0.8455837911258958` of target parameters,
below the frozen `0.90` backbone parameter-coverage requirement. They remain
excluded; neither the threshold nor their status is changed to improve the
result.

### NVDINO backbone key

The metadata-only source inventory preserved the published transfer-snippet
key `vit_large_dinov2`. That constructor has standard 4,096-wide MLP blocks,
whereas the official checkpoint contains 5,472-wide SwiGLU blocks. The
supported TAO 7.1 key `vit_large_nvdinov2` matches the checkpoint: the
corrected diagnostic load covers 930/930 target tensors and 100% of target
parameters.

This is a qualification-derived registry/sidecar correction, not a rewrite of
source discovery. The immutable source record remains
`experiments/cross_model_automl_20260729/dino_ptm_inventory/source_inventory.v1.json`
with raw SHA-256
`045c9b7bf40af2e62cf32dbcd483e97440a7096bcda82d3d659240d2c1e45087`;
it continues to record the originally published `vit_large_dinov2` metadata.

## Rerun boundary

The v1 manifest and completion must remain byte-for-byte unchanged. The
corrected implementation, registry, serializer pins, and sidecar change the
qualification identity, so resume in the v1 directory is prohibited. The
authoritative rerun must write create-only v2 evidence to a new output
directory, for example:

```text
~/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/cpu_v2/
```

V2 must evaluate the complete registry-derived population under the same
coverage gates. Its pass/fail accounting must come from the qualification
algorithm; diagnostic results in this note neither preselect checkpoints nor
promote registry entries.
