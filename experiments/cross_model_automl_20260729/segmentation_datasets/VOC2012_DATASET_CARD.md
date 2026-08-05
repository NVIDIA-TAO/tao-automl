# PASCAL VOC2012 semantic-segmentation dataset card

## Identity and scope

- Dataset ID: `pascal_voc2012_segmentation_trainval`
- TAO consumer: `segformer`
- Task: multiclass semantic segmentation
- Complete staged labeled release: official VOC2012 segmentation train/val
- Official train images/masks: 1,464
- Official validation images/masks: 1,449
- Classes: background plus 20 foreground classes
- Valid indexed labels: 0 through 20
- Ignore label: 255

The private challenge-test annotations are not represented as public labeled
data. “Complete” here means the complete official public labeled
train/validation segmentation release, not a private-test-label claim.

## Authoritative source and rights

- Challenge page:
  <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/>
- Database-rights notice:
  <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/dbstats.html>
- Exact archive:
  <https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar>
- Archive size: 1,999,639,040 bytes
- Archive SHA256:
  `e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb`

PASCAL VOC is not one uniformly permissively licensed image corpus. The
database-rights notice and the terms attached to constituent source images,
including Flickr images, continue to apply.

## Frozen layout

Local source and byte-identical TAO projection:

```text
/localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/voc2012_segmentation_full/
  source/VOCdevkit/VOC2012/
    ImageSets/Segmentation/{train,val,trainval}.txt
    JPEGImages/
    SegmentationClass/
  prepared/
    images/{train,val}/*.jpg
    masks/{train,val}/*.png
```

Lustre:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/voc2012_segmentation_v1/
  source_archives/VOCtrainval_11-May-2012.tar
  prepared/images/{train,val}
  prepared/masks/{train,val}
```

The prepared files are hard links to the official extraction. There is no
image, palette, pixel, label, resize, or compression conversion.

## Required SegFormer contract

The model-side profile must preserve the following semantics:

```yaml
dataset:
  segment:
    num_classes: 21
    label_transform: "None"
```

The palette must include exact mappings for grayscale IDs 0 through 20 and
ignore ID 255. Training loss and mIoU evaluation must ignore 255 consistently
across training, evaluate, inference, export, and checkpoint reload.

These are required configuration constraints, not evidence that a model was
run. No model-side preflight was executed during staging.

## Data-only integrity result

The verifier enforces:

- exact official split identities and counts;
- disjoint train and validation identities;
- `trainval` equals the exact train/validation union;
- exact image and mask filename sets;
- source and prepared paths resolve to the same inode;
- every JPEG and PNG decodes;
- every image/mask dimension agrees;
- every pixel is in 0 through 20 or 255;
- source archive size and SHA256.

The canonical result and full pixel-count statistics are recorded in
`dataset_stage_manifest.v1.json` and the referenced validation JSON. No model
or scheduler code is imported by the verifier.
