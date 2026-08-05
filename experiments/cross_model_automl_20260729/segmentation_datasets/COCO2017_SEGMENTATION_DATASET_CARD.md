# COCO2017 instance/panoptic dataset card

## Identity and scope

- Dataset ID: `coco2017_instance_panoptic`
- TAO consumers: `oneformer`, `mask2former`, and the category-prompted
  `mask_grounding_dino` path
- Complete official train images: 118,287
- Complete official validation images: 5,000
- Instance annotations: 860,001 train and 36,781 validation
- Instance/thing classes: 80
- Panoptic categories: 133
- Panoptic PNGs: one for every train and validation image

OneFormer consumes native panoptic COCO. The preregistered Mask2Former
instance task consumes native instance COCO and its 80-category instance label
map; the same frozen root also contains panoptic assets for supported
panoptic-mode use. Mask Grounding DINO training uses a pinned official TAO
Data Services COCO-to-ODVG projection with the original instance masks.

The Mask Grounding DINO scope is category-prompted grounded instance
segmentation (`data_type: OD`). COCO category labels are not referring
expressions, so this dataset must not be used to claim RefCOCOg-style
phrase-grounding performance.

## Authoritative source and rights

- Official dataset and terms: <https://cocodataset.org/>
- Train images:
  <https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip>
- Validation images:
  <https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip>
- Instance annotations:
  <https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip>
- Panoptic annotations:
  <https://s3.amazonaws.com/images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip>

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| `train2017.zip` | 19,336,861,798 | `69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929` |
| `val2017.zip` | 815,585,330 | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| `annotations_trainval2017.zip` | 252,907,541 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| `panoptic_annotations_trainval2017.zip` | 860,725,834 | `c05f76d2129b6b561eb70efe16e7006df62f73fb92889132d373b9d90e31a370` |

All four frozen archive SHA256 checks and zip integrity tests passed.

COCO annotations are distributed under CC BY 4.0. Each image retains its
source-image license; COCO does not apply one blanket image license.

## Frozen layout

Local:

```text
/localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/coco2017_full/
  source_archives/{train2017,val2017,annotations_trainval2017,panoptic_annotations_trainval2017}.zip
  images/{train2017,val2017}/*.jpg
  annotations/
    instances_{train,val}2017.json
    panoptic_{train,val}2017.json
    panoptic_{train,val}2017/*.png
  tao/
    label_map_panoptic.json
    label_map_instance.json
    mask_grounding_dino/train/
      instances_train2017_odvg.jsonl
      instances_train2017_odvg_labelmap.json
```

Lustre:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/coco2017_instance_panoptic_v1/
```

The Lustre root has the same relative training paths and also retains the
source archives and hash manifests.

## TAO label maps

`tao/label_map_panoptic.json` is copied byte-for-byte from
`ngc-collaterals` commit
`f8bc9a48aa7dfa976eb396c4158356484411ecce`, path
`cv/resource/notebooks/tao_api_starter_kit/dataset_prepare/coco_panoptic/labelmap.json`.
Its SHA256 is
`4b28b3773f0f8e63d836dc20da77276633da72178453458b79e32be8e892ce56`.
The verifier compares ID, category name, and thing/stuff identity against the
official panoptic JSON and separately validates all label-map colors.

`tao/label_map_instance.json` is copied byte-for-byte from the same commit,
path
`cv/resource/notebooks/tao_launcher_starter_kit/mask2former/specs/labelmap_inst.json`.
Its SHA256 is
`67f15c4dd7d52aa73025da8307dec17e907f13db6d5d82332a670f73da68c306`.
The verifier compares all 80 IDs and names against the official COCO instance
categories, requires each to be a thing, and validates all colors. The
preregistered Mask2Former instance binding uses `type: coco`,
`contiguous_id: true`, and this instance label map.

## Mask Grounding DINO ODVG projection

- Converter repository: TAO Data Services
- Converter commit: `dcea3a39bd3e4709e2325e4b61a4f179efebde4c`
- Exact implementation:
  `nvidia_tao_ds.annotations.conversion.coco_to_odvg.convert_coco_to_odvg`
- `use_all_categories`: `false`
- Output JSONL SHA256:
  `d5deb4f5cfe027786fb1ceb52632ad6d3ef027e95e434525ba715d6841fb2921`
- Output label-map SHA256:
  `02075d96f6bf06d061f9329b4775dc7c3bb5ac140c77bc5c0e465d305c46d6c1`

The installed umbrella CLI imported an unrelated optional `h5py` dependency
through its global command registry. To avoid adding unrelated dependencies,
the exact official conversion function was invoked directly with the same
arguments. No conversion logic was copied or rewritten.

Equivalent invocation:

```python
from types import SimpleNamespace
from nvidia_tao_ds.annotations.conversion.coco_to_odvg import (
    convert_coco_to_odvg,
)

root = "/localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/coco2017_full"
cfg = SimpleNamespace(
    coco=SimpleNamespace(
        ann_file=f"{root}/annotations/instances_train2017.json",
        use_all_categories=False,
    ),
    results_dir=f"{root}/tao/mask_grounding_dino/train",
)
convert_coco_to_odvg(cfg, verbose=False)
```

Two independent invocations produced byte-identical outputs. The verifier
compares every projected image field and every projected instance against the
source JSON, including contiguous label, category, converter-equivalent bbox,
and exact polygon/RLE mask JSON. All 860,001 source instance annotations and
masks are preserved. The official converter omits only images with zero
instance annotations; that policy and count are recorded rather than hidden.

The official train JSON contains two zero-area source annotations, IDs `918`
and `2206849`; validation contains none. They are reported and preserved
exactly. Dataset staging does not silently repair, delete, or relabel them.

## Data-only integrity result

The verifier enforces:

- exact official image and annotation counts;
- unique and valid image, annotation, category, and panoptic-segment IDs;
- exact equality between extracted and JSON-referenced image/mask sets;
- all image and panoptic PNG dimensions;
- every panoptic PNG segment ID and pixel area against JSON;
- identical instance and panoptic thing-category mappings;
- exact official-to-TAO panoptic and instance category identities;
- all source-to-ODVG fields and masks;
- explicit preservation/reporting of official zero-area annotations;
- archive and derived-artifact SHA256 identities.

The output is data-only. It does not establish that OneFormer reports PQ,
Mask2Former reports mask AP, or Mask Grounding DINO reports a task-correct
metric in the eventual pinned SQSH. Those model-side gates remain open.
