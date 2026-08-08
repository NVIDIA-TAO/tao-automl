# Cross-model AutoML dataset and execution preregistration

Status: Phase-4 dataset acquisition update. The segmentation datasets below
were frozen before model execution, passed data-only validation, and were
staged on Lustre without running models or reserving GPUs.

Audit date: 2026-07-29; dataset acquisition update: 2026-07-30

This document remains a planning and correctness gate for model execution.
The linked machine-readable reports are evidence for dataset acquisition and
data-only integrity only. They are not evidence that a model has passed
preflight or that a campaign is ready to launch.

During the original 2026-07-29 audit:

- no dataset was downloaded;
- no checkpoint was downloaded;
- no model training, evaluation, inference, or latency benchmark ran;
- no local or SLURM GPU job was submitted;
- no experiment artifact or winner was changed;
- no dataset choice was made after observing cross-model results.

On 2026-07-30, the complete public labeled PASCAL VOC2012 semantic
segmentation train/validation release and complete COCO2017
train/validation image, instance, and panoptic releases were downloaded from
their authoritative public endpoints. They were validated with data-only
code and copied to the frozen Lustre roots recorded below. No checkpoint was
downloaded, no model was imported or executed, and no local or SLURM training,
evaluation, inference, latency, or smoke job ran as part of dataset staging.

The dataset, split, task metric, and conversion contracts must be frozen and
hashed before model execution. Dataset readiness alone does not lift any
model, metric, PTM, container, or campaign gate.

## Exact TAO model contracts

The implemented model identifiers, not conversational aliases, are:

| Model | Exact identifier | Task intended by this preregistration | Packaged data format |
| --- | --- | --- | --- |
| DINO | `dino` | Object detection | COCO |
| Deformable DETR | `deformable_detr` | Object detection | COCO |
| RT-DETR | `rtdetr` | Object detection | COCO |
| Grounding DINO | `grounding_dino` | Referring-expression box grounding | ODVG for training; current validation path is COCO-only |
| SegFormer | `segformer` | Semantic segmentation | TAO/UNet-style image and mask folders |
| OneFormer | `oneformer` | Panoptic segmentation | COCO panoptic |
| Mask2Former | `mask2former` | Instance segmentation | COCO instance plus panoptic assets |
| Mask Grounding DINO | `mask_grounding_dino` | Category-prompted grounded instance segmentation for the staged path; referring-expression segmentation remains a separate blocked claim | ODVG/VG with masks |

All eight current model-skill records resolve the nominal default image to
`nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt`. That tag must not be assumed to
contain changes from a local release/7.1.0 branch. Preflight must pin and hash
an SQSH whose installed wheel or mounted source identity matches the campaign
source commit.

## Authoritative dataset matrix and staging state

The estimated epoch times below are capacity-planning ranges, not
measurements. They assume one A100, a model-supported batch size and crop, and
one complete pass over the stated training split. The first successful local
epoch must replace them with measured timing before campaign resource
manifests are frozen.

| Model | Task | Proposed complete dataset | Dataset size and official splits | Annotation contract | Primary metric | License and access | Estimated one-A100 epoch | Why selected |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| `dino` | Object detection | PASCAL VOC2007 | 9,963 images, 24,640 objects, 20 classes; train 2,501, validation 2,510, test 4,952; approximately 450 MB train/validation plus 430 MB annotated test | VOC XML converted losslessly to canonical COCO bbox JSON | `val_mAP50` (COCO AP50), with `val_mAP` secondary | Public challenge data; constituent Flickr/MSRC image terms apply rather than one blanket open-source license | 15–35 min | Complete, moderate, multi-object scenes; public test annotations |
| `deformable_detr` | Object detection | PASCAL VOC2007 | Same complete corpus and splits | Same canonical COCO archive | COCO bbox AP@[0.50:0.95], with AP50 secondary | Same VOC image-specific terms | 10–30 min | Enables a fair shared detection corpus without forcing a subset |
| `rtdetr` | Object detection | PASCAL VOC2007 | Same complete corpus and splits | Same canonical COCO archive | COCO bbox AP@[0.50:0.95], with AP50 secondary | Same VOC image-specific terms | 5–20 min | Moderate enough for repeated AutoML while retaining realistic multi-object scenes |
| `grounding_dino` | Referring-expression box grounding | RefCOCOg, UMD split | 25,799 images, 49,822 referred objects, 95,010 expressions; train 21,899 images/42,226 objects/80,512 expressions, validation 1,300/2,573/4,896, test 2,600/5,023/9,602; image-disjoint UMD splits | RefCOCOg expression and COCO annotation identity converted to VG-style ODVG while preserving expression, bbox, source IDs, and image identity | Percentage of expressions whose predicted box has IoU >= 0.5 (`Pr@0.5`); mean box IoU secondary | RefCOCOg annotations originate from Google RefExp, whose official author release states CC BY 4.0; `refer` API code is Apache-2.0; underlying COCO images retain their source-image licenses | 2–6 h | A complete, task-correct grounding corpus rather than category-prompted detection |
| `segformer` | Semantic segmentation | PASCAL VOC2012 segmentation | Complete public labeled train/validation release: 1,464 train and 1,449 validation images, 20 foreground classes plus background, ignore ID 255; 1,999,639,040-byte source archive | Byte-identical JPEG and indexed-PNG hard-link projection as `images/<split>` and `masks/<split>`; no pixel or label conversion | mIoU over 21 IDs including background, with 255 ignored | Public without login; PASCAL database-rights notice and constituent Flickr image terms apply rather than one blanket permissive license | 10–30 min | Complete public labeled release of moderate size; avoids registration-gated Cityscapes while exercising multiclass semantic segmentation |
| `oneformer` | Panoptic segmentation | COCO2017 instance and panoptic train/validation | Complete official train/validation corpus: 118,287 train and 5,000 validation images; 133 panoptic categories, including 80 things | Native official COCO panoptic JSON and RGB segment-ID PNGs plus the TAO 133-category panoptic label map | Panoptic Quality (PQ); PQ-things and PQ-stuff secondary | Public without login; annotations are CC BY 4.0 and each image retains its source license | 4–12 h | Native, task-correct panoptic data; no custom semantic conversion and shared source imagery with Mask2Former |
| `mask2former` | Instance segmentation | COCO2017 instance and panoptic train/validation | Same complete official 118,287/5,000 image corpus; 860,001 train and 36,781 validation instance annotations over 80 thing classes | Native official COCO instance JSON with exact polygon/RLE masks and an official TAO 80-category instance label map; panoptic assets are co-staged for the supported alternate mode | COCO mask AP@[0.50:0.95] | Same COCO annotation and image-specific terms | 4–12 h | Native complete instance masks without a conversion; shared native panoptic data remains available without changing this campaign's instance task |
| `mask_grounding_dino` | Category-prompted grounded instance segmentation | COCO2017 instance train/validation | Same complete official corpus; all 860,001 train masks projected into ODVG and native COCO validation retained | Pinned official TAO Data Services COCO-to-ODVG conversion, contiguous category labels, and byte-for-byte preserved polygon/RLE mask JSON; `data_type: OD` | COCO mask/detection metric supported by the pinned product path; this staging does not establish referring-expression IoU | Same COCO annotation and image-specific terms | 4–12 h | Exercises the supported category-prompted mask path with complete masks; explicitly not a replacement for RefCOCOg phrase-grounding validation |

Every detection metric uses the TAO COCO evaluator after a lossless
VOC-to-COCO conversion; none is the historical VOC2007 11-point AP. DINO uses
the release/7.1 skill's explicit AutoML recommendation, `val_mAP50`, so the
new pilot remains comparable with the completed DINO evidence and the
implemented metric extractor. `val_mAP` remains a required secondary metric.
The Deformable DETR and RT-DETR campaigns retain preregistered COCO
AP@[0.50:0.95] as their primary metric unless their local product contract
fails that extraction before any campaign is launched.

## Authoritative source and license records

### PASCAL VOC2007

- Overview, statistics, and database rights:
  <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/>
- Project usage guidance:
  <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/>
- Official train/validation archive:
  <https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar>
- Official annotated test archive:
  <https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar>

The official page reports 9,963 images and 24,640 annotated objects. It lists
the train/validation archive as approximately 450 MB and annotated test archive
as approximately 430 MB. The dataset contains images from multiple sources,
including Flickr and MSRC. The dataset card must preserve those source-specific
rights and must not describe VOC as a single permissively licensed corpus.

### RefCOCOg and COCO 2014

- Official Google RefExp author toolbox and license statement:
  <https://github.com/mjhucla/Google_Refexp_toolbox>
- Original Google RefExp paper:
  <https://arxiv.org/abs/1511.02283>
- Official Google release identity:
  <https://storage.googleapis.com/refexp/google_refexp_dataset_release.zip>
- Referring-expression API, which recommends the image-disjoint UMD split:
  <https://github.com/lichengunc/refer>
- COCO source:
  <https://cocodataset.org/>

The Google author toolbox states that its RefExp data is CC BY 4.0 and that
COCO 2014 images and annotations are required. It describes the COCO image
download as approximately 13 GB and the annotations as approximately 158 MB.
COCO images retain their individual source-image licensing.

The UMD split is preregistered because it supplies image-disjoint train,
validation, and test populations. The current `refer` README also warns that
its historical download server is broken. The Google storage URL returned an
access error during this read-only audit. Before adoption, dataset preflight
must resolve an authorized immutable source, prove that it is the intended
release, and record its SHA256. An unpinned community mirror is not an
acceptable silent replacement.

### PASCAL VOC2012 segmentation

- Official challenge page, task details, and release statistics:
  <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/>
- Official database-rights notice:
  <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/dbstats.html>
- Exact official train/validation archive:
  <https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar>

The downloaded archive is 1,999,639,040 bytes with SHA256
`e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb`.
Its tar integrity passed before extraction. The official semantic split lists
contain 1,464 training and 1,449 validation identities and are disjoint. The
TAO projection uses hard links to the untouched source JPEGs and indexed PNG
masks; every image/mask dimension and every mask label was checked.

PASCAL VOC does not grant one blanket permissive license over its images. Its
database-rights notice and the terms of the originating image sources,
including Flickr, must be retained. This is the complete public labeled
VOC2012 semantic train/validation release, not a claim that private challenge
test labels are available.

### COCO2017 instance and panoptic

- Official dataset and terms:
  <https://cocodataset.org/>
- Exact official image archives:
  <https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip> and
  <https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip>
- Exact official annotation archives:
  <https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip>
  and
  <https://s3.amazonaws.com/images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip>

The frozen archive identities are:

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| `train2017.zip` | 19,336,861,798 | `69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929` |
| `val2017.zip` | 815,585,330 | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| `annotations_trainval2017.zip` | 252,907,541 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| `panoptic_annotations_trainval2017.zip` | 860,725,834 | `c05f76d2129b6b561eb70efe16e7006df62f73fb92889132d373b9d90e31a370` |

All four zip integrity tests and all frozen archive checksums passed. The
native instance JSON, panoptic JSON, image sets, and panoptic PNG sets were
checked for exact reference equality. Every image and panoptic-mask dimension
was checked, and the deep validation additionally checked each RGB panoptic
segment ID and pixel area against JSON.

COCO annotations are distributed under CC BY 4.0. COCO does not relicense the
images: each source image retains its own license, recorded in the annotation
metadata.

### Rejected staged alternative: Cityscapes

Cityscapes remained scientifically appropriate, but its authenticated,
registration-gated access and non-redistribution terms prevented unattended
public acquisition. No Cityscapes archive or derivative was downloaded.
VOC2012 and COCO2017 were selected before any model result: VOC2012 supplies a
complete public semantic corpus, and native COCO supplies the complete
instance and panoptic contracts.

## Frozen segmentation staging records

Repository-owned evidence:

- `experiments/cross_model_automl_20260729/segmentation_datasets/dataset_stage_manifest.v1.json`;
- `experiments/cross_model_automl_20260729/segmentation_datasets/segmentation_dataset_validation.v1.json`;
- `experiments/cross_model_automl_20260729/segmentation_datasets/VOC2012_DATASET_CARD.md`;
- `experiments/cross_model_automl_20260729/segmentation_datasets/COCO2017_SEGMENTATION_DATASET_CARD.md`;
- `experiments/cross_model_automl_20260729/segmentation_datasets/voc2012_segformer_dataset_profile.yaml`;
- `experiments/cross_model_automl_20260729/segmentation_datasets/coco2017_tao_dataset_bindings.yaml`.

The full per-file manifests are intentionally stored beside the datasets
rather than adding approximately 20 MB of file hashes to Git:

| Dataset | Immutable Lustre root | File-manifest entries | File-manifest SHA256 |
| --- | --- | ---: | --- |
| VOC2012 segmentation | `/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/voc2012_segmentation_v1` | 5,827 | `051ab20215b8e6976763ac82a3db20a68264759edef3d62fd0c8553c501123ff` |
| COCO2017 instance/panoptic | `/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/coco2017_instance_panoptic_v1` | 246,593 | `10566a60498de9998154f44a34445a488c9f030e09f2a7346d20a4a1c55f804e` |

The committed stage manifest records the corresponding local paths,
validation report hashes, converter and collateral commits, transfer
provenance, remote verification result, and read-only audit.

## Split and final-evaluation policy

- VOC2007 keeps the official train, validation, and annotated test lists
  unchanged. AutoML observes validation metrics; the test split remains
  untouched until final frozen-candidate evaluation.
- RefCOCOg uses the complete, image-disjoint UMD train, validation, and test
  lists unchanged.
- VOC2012 segmentation keeps its official train and validation identity lists
  unchanged.
- COCO2017 keeps the complete official train and validation image and
  annotation populations unchanged.
- Neither public dataset supplies labeled public challenge-test data in the
  staged package. If a campaign needs a terminal holdout independent of
  AutoML validation, its split must be preregistered and hashed before the
  first recommendation. No post-result split construction is permitted.

No test or terminal-evaluation metric may feed recommendation, feasibility,
selection, or reselection.

## Existing TAO Data Services capability

The local `tao-dataservices` source at audit time was:

```text
dcea3a39bd3e4709e2325e4b61a4f179efebde4c
```

Reusable implementation exists under
`nvidia_tao_ds/annotations/conversion/`:

- COCO to ODVG;
- ODVG to COCO;
- COCO category remapping to contiguous IDs;
- COCO to KITTI;
- KITTI to COCO.

`coco_to_odvg.py`:

- emits category-detection or caption/token-span grounding records;
- converts COCO `xywh` boxes to ODVG `xyxy`;
- preserves polygon or RLE `segmentation` values as ODVG masks;
- writes a label map for detection data;
- remaps present category IDs to a contiguous ODVG label domain.

The staged Mask Grounding DINO training projection invokes the exact
`convert_coco_to_odvg` function from that pinned commit with
`use_all_categories=false`. The installed command-line entry point imported an
unrelated optional `h5py` dependency through its global command map, so the
conversion function itself was invoked directly rather than reimplemented.
Two independent conversions produced byte-identical JSONL and label-map
outputs. The data-only verifier then compared every projected field against
every one of the 860,001 source annotations, including exact polygon/RLE mask
JSON. No custom converter or candidate-specific edit was used.

The official converter omits source images with no instance annotations. The
verifier records that policy explicitly and proves that no annotated image or
annotation was lost. Official COCO source annotations with zero bbox height or
area are also retained and reported, not silently corrected or deleted.

### Existing capability that is not yet sufficient

The repository-owned data-only verifier added by this staging change proves
source-to-output image and annotation accounting, category-map bijection,
converter-equivalent bbox equality, exact polygon/RLE JSON identity, source
image/panoptic-mask dimension identity, and repeat-conversion determinism for
the category-prompted path. It does not prove expression or token-span
preservation because COCO category detection contains no referring
expressions, and it does not claim decoded-mask equivalence beyond the exact
source JSON identity plus native panoptic pixel validation.

The `use_all_categories=true` branch in the current COCO-to-ODVG converter
indexes a category-count array using raw category IDs. That is unsafe for
non-contiguous category IDs. Until fixed and tested, the preregistered path
must use the default `use_all_categories=false` and separately audit categories
with zero instances.

Optional correction in the COCO validator is inappropriate for immutable
campaign preparation. Validation must fail closed and report an error rather
than silently modifying source-derived annotations.

## Required conversion and validation work

The following repository-owned preparation paths remain absent:

1. VOC XML to canonical COCO detection JSON.
2. Native RefCOCOg plus COCO annotation IDs to expression-preserving VG ODVG.
3. A task-correct RefCOCOg acquisition and expression-preserving preparation
   path for the requested phrase-grounding claims.

The following segmentation preparation paths are now present:

1. byte-identical VOC2012 indexed masks in the SegFormer folder contract;
2. native complete COCO2017 instance and panoptic assets plus exact TAO
   panoptic and instance label maps for OneFormer and Mask2Former;
3. pinned official COCO-to-ODVG conversion with exact source-mask preservation
   for category-prompted Mask Grounding DINO;
4. acquisition records, archive hashes, file manifests, dataset cards, and a
   machine-readable data-only integrity report.

### Conversion-specific requirements

#### VOC2007

- Preserve the official split lists.
- Preserve every image and non-difficult annotation according to one declared
  difficult-object policy.
- Use stable image, annotation, and category IDs.
- Preserve the canonical category names and their order.
- Report source and output object counts by split and class.
- Configure each model for the actual category IDs. DINO's class capacity must
  exceed the maximum retained ID; Deformable DETR must receive every
  `eval_class_id`; RT-DETR remapping must be explicit and audited.

#### RefCOCOg

- Preserve `ref_id`, `sent_id`, COCO `ann_id`, source image ID, expression,
  bbox, and segmentation.
- Prove UMD train, validation, and test image sets are disjoint.
- Emit one unambiguous evaluation record per expression and target.
- Do not reconstruct masks from boxes. Decode the original COCO polygon/RLE.
- Record empty or invalid source records as structured failures; do not drop
  them silently.

#### VOC2012 and COCO2017 segmentation

- SegFormer must use 21 output IDs including background, preserve ignore ID
  255, and set `label_transform: "None"` for the already-indexed masks.
- Every VOC image/mask pair remains byte-identical to the official extraction.
- OneFormer and Mask2Former consume the native COCO2017 instance/panoptic
  structures; no Cityscapes or semantic-label remap is implied.
- The official 133-category COCO panoptic identity and thing/stuff mapping are
  preserved in the pinned TAO label map.
- Mask Grounding DINO's staged ODVG path is `data_type: OD`, uses contiguous
  category IDs, and retains the original polygon/RLE masks.

## Per-model metric and readiness blockers

### DINO

This segmentation staging change did not stage the proposed VOC2007 detection
corpus or a DINO PTM. The VOC category-ID contract and COCO mAP parsing must
pass before execution against that proposed public corpus.

### Deformable DETR

The complete `eval_class_ids` set must be supplied. A finite aggregate metric
with silently omitted classes is not a passing preflight.

### RT-DETR

Category remapping and foreground/background indexing must be proved against
the canonical VOC COCO file. A numerically finite result with shifted labels is
invalid.

### Grounding DINO

Current source constructs `CocoDetection` for validation in
`nvidia_tao_pytorch/cv/grounding_dino/dataloader/pl_odvg_data_module.py`.
Training accepts ODVG, including phrase records, but validation reports COCO
bbox mAP from category captions.

Consequently, the current path can validate category-prompted open-vocabulary
detection but cannot establish the requested referring-expression grounding
claim. Grounding DINO is not SLURM-ready until a phrase-grounding validation
loader and Pr@0.5 evaluator are implemented and locally verified. Substituting
VOC category prompts must be labeled as detection and is not an acceptable
grounding result.

The separately scoped synthetic-COCO campaign under
`grounding_dino_shared_detection` intentionally exercises only the supported
category-prompted detection path. It uses official COCO-to-ODVG conversion,
contiguous-ID COCO validation, and `val_mAP50`; it does not change the blocked
referring-expression row below or make a `Pr@0.5` claim.

### SegFormer

Preflight must verify:

- 21 IDs including background rather than the binary template default;
- ignore ID 255;
- exact grayscale mask IDs;
- a palette that represents IDs 0 through 20 and ignore ID 255;
- `label_transform: "None"` across all stages;
- task-correct mIoU after checkpoint reload.

The full official VOC2012 train/validation data and lossless TAO folder
projection pass the data-only gate. No model-side condition above has been
executed.

### OneFormer

Panoptic inference exists, but inspected validation code reports semantic
mIoU and pixel accuracy. No training-time PQ path was found. A panoptic
campaign cannot optimize semantic mIoU and then claim panoptic quality.
OneFormer remains blocked until PQ is emitted, parsed, and independently
checked against the official COCO evaluator. The complete native COCO
panoptic dataset is staged; that does not remove the metric blocker.

### Mask2Former

The inspected validation path also reports semantic mIoU and pixel accuracy,
although instance and panoptic inference modes exist. No mask AP validation
path was found. Mask2Former remains blocked until COCO mask AP is emitted,
parsed, and checked against the official COCO instance evaluator. The complete
native COCO instance and panoptic data are staged.

### Mask Grounding DINO

Current source code accepts `data_type: VG` for validation and its VG evaluator
returns:

- `mIoU`;
- `overall_IoU`;
- `mAP50`;
- `mAP`;
- target/no-target accuracy;
- Pr@IoU thresholds.

The packaged skill, however, instructs train-stage AutoML to use `val_loss` and
describes validation as COCO-format. The exact pinned container may therefore
not match the inspected source. The discrepancy remains a hard preflight
blocker for a referring-expression campaign.

The staged COCO2017 path is deliberately narrower: category-prompted grounded
instance segmentation with `data_type: OD`. Its 860,001 masks pass exact
source-to-ODVG structural comparison. This data readiness must not be reported
as phrase-grounding or RefCOCOg validation.

The model-specific smoke contract must also keep:

- `model.enc_layers = 6`;
- `model.dec_layers = 6`;
- a non-degenerate query count, at least 100 for the generic smoke case.

The existing generic AutoML smoke ranges that reduce depths to one or two are
invalid for this model.

Mask Grounding DINO's VG evaluator reports percentage metrics on a 0-to-100
scale, while several TAO detection metrics are reported on a 0-to-1 scale.
The objective registry and sanity policy must declare metric scale and
direction explicitly.

## Dataset integrity gate

Every prepared dataset must produce a machine-readable report that proves:

1. Every downloaded archive matches its frozen SHA256.
2. Every extracted file belongs to a canonical manifest with a root hash.
3. Every referenced image exists, decodes, and matches declared dimensions.
4. Train, validation, test, and final-evaluation image identities obey the
   preregistered overlap policy.
5. Duplicate image, annotation, expression, and candidate IDs are rejected.
6. Boxes are finite, positive-area, in bounds, and in the declared coordinate
   convention.
7. Categories have a bijective, documented mapping and no class silently loses
   annotations.
8. Polygon and RLE masks decode successfully and match source area and pixel
   identity.
9. Semantic masks contain only declared labels and ignore IDs.
10. Panoptic segment IDs, JSON records, masks, areas, and thing/stuff flags
    agree.
11. Every grounding expression is nonempty and maps to the exact source object
    and mask.
12. Conversion output is deterministic and invariant to source enumeration
    order.
13. No source record is silently corrected, excluded, or relabeled.
14. Class, annotation, mask-area, image-size, and empty-sample statistics are
    recorded by split.
15. Dataset card, conversion config, converter commit, output manifest, and
    integrity report are all hash-linked.

When an upstream source publishes no checksum, preflight must calculate SHA256
over the untouched archive immediately after download and freeze it before
extraction. A mutable URL is not an immutable identity.

## Required local model preflight

The existing
`tao-automl/scripts/validate_skill_automl_model.py` is an internal small-dataset
smoke harness. It runs minimal AutoML recommendations and post-check
evaluation/inference, but it does not satisfy this campaign's local gate.

For each model, local single-GPU preflight must prove:

1. Complete prepared dataset passes the integrity gate.
2. Default registered PTM checksum and load succeed.
3. Every additional registered PTM loads and completes train, validation, and
   inference mini-steps.
4. Merged PTM YAML, AutoML profile, user values, and candidate overrides have
   the declared precedence.
5. One training batch succeeds.
6. One validation batch succeeds with a task-correct finite metric.
7. One inference batch succeeds.
8. One complete epoch over the full training split succeeds.
9. Validation runs within that epoch.
10. Standalone evaluation succeeds.
11. A checkpoint is saved, reloaded exactly, and produces a valid evaluation.
12. Resume behavior is demonstrated from a non-`latest` checkpoint.
13. Stabilized latency instrumentation succeeds under a frozen input and
    runtime contract.
14. Outputs, logs, configuration, PTM, dataset, image, and package identities
    are hash-linked.

No model is SLURM-ready merely because its configuration parses or a single
mini-batch is finite.

## Local readiness matrix after data staging

`Pending` means the design is identified but no model execution has occurred.
`Blocked` means a concrete implementation or access defect must be resolved
before execution.

| Model | Dataset contract | Task metric path | Conversion path | Access | Full local epoch | SLURM ready |
| --- | --- | --- | --- | --- | --- | --- |
| `dino` | Proposed | Pending | Missing VOC-to-COCO | Pending | Not run | No |
| `deformable_detr` | Proposed | Pending | Missing VOC-to-COCO | Pending | Not run | No |
| `rtdetr` | Proposed | Pending | Missing VOC-to-COCO | Pending | Not run | No |
| `grounding_dino` | Proposed | Blocked: no phrase-grounding validation metric | Missing RefCOCOg-to-VG ODVG | Blocked pending immutable RefCOCOg access | Not run | No |
| `segformer` | Staged and data-only validated: complete VOC2012 train/val | Pending model execution | Complete byte-identical TAO folder projection | Public source staged locally and on Lustre | Not run by instruction | No |
| `oneformer` | Staged and data-only validated: complete COCO2017 train/val instance/panoptic | Blocked: no verified PQ | Native COCO panoptic plus pinned TAO label map | Public source staged locally and on Lustre | Not run by instruction | No |
| `mask2former` | Staged and data-only validated: complete COCO2017 train/val instance/panoptic | Blocked: no verified mask AP | Native COCO instance and panoptic assets | Public source staged locally and on Lustre | Not run by instruction | No |
| `mask_grounding_dino` | Staged and data-only validated for category-prompted COCO; phrase-grounding dataset not staged | Blocked for phrase grounding: skill/container/source metric mismatch | Official TAO Data Services COCO-to-ODVG conversion verified; RefCOCOg-to-VG path still absent | COCO public source staged; immutable RefCOCOg access remains blocked | Not run by instruction | No |

## Later SLURM and strict SQSH contract

This section records the launch contract for use only after local preflight and
the DINO three-job pilot pass. It does not authorize or describe a job launched
during this audit.

Follow the packaged `tao-run-on-slurm` workflow and use `SlurmSDK.create_job`
and `build_entrypoint`. Do not hand-roll campaign `sbatch` scripts.

### Data and storage

- Prestage complete datasets, PTMs, specs, and wheels on Lustre.
- Verify every required path with `test -e` from the login host.
- Use `lustre:///absolute/path` or its resolved absolute path in job inputs.
- Do not download S3, HTTP, Hugging Face, or NGC training data after a GPU
  allocation starts.
- Mount `/lustre` read-only for inputs and use an explicit writable results
  path.

### One-node, eight-GPU shape

Every full candidate job must request and use the entire node:

```text
num_nodes = 1
gpu_count = 8
train.num_gpus = 8
train.gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
cpus_per_task = 16
```

Use the packaged four-hour/3.8-hour timeout defaults unless the selected
partition explicitly supports a longer frozen request. The SDK emits one
Pyxis task with `#SBATCH --gres=gpu:8`; the TAO/Lightning entrypoint creates
the local distributed workers.

### SQSH must fail closed

Set and verify:

```text
SLURM_USE_SQSH = true
```

Preconvert the pinned container on the CPU conversion partition, record the
absolute `.sqsh` path and SHA256, and use:

```text
srun --container-image=/absolute/pinned-image.sqsh --container-mounts=/lustre
```

At audit time,
`tao_sdk/platforms/slurm/handler.py::_prepare_container_image` caught SQSH
conversion failures and returned the registry image. That silent fallback
violates the campaign requirement. Before submission, production must either:

1. provide a strict-SQSH option that raises on conversion failure; or
2. accept only a pre-staged `.sqsh` and assert that the resolved job image is
   the exact hashed file.

A job that falls back to a registry pull must fail preflight and must not enter
the campaign.

### Retries and staged release

The SDK's audited constant was `MAX_JOB_RETRIES = 10`. The campaign must
preregister a smaller finite infrastructure retry budget and preserve every
failed training recommendation as a failed record. A failed candidate may not
be silently replaced to reach a requested successful count.

Submission remains staged:

1. one candidate per mode;
2. a small pilot batch after artifact validation;
3. the frozen full budget only after the pilot passes;
4. matched validation only after winners and relevant Pareto candidates are
   frozen.

## Preregistration conclusion

The dataset choices are scientifically appropriate and moderate relative to
the supported tasks. The complete VOC2012 semantic and COCO2017
instance/panoptic roots are staged and data-only validated, but this matrix
does not mark any model ready for SLURM. The phrase-grounding evaluator,
panoptic PQ, instance mask AP, Mask Grounding DINO metric contract, remaining
detection/grounding conversion and access paths, and strict SQSH behavior are
concrete model or campaign gates.

Dataset acquisition and validation invoked no model, latency benchmark, or
SLURM submission.
