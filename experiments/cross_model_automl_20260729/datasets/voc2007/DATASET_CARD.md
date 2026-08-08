# PASCAL VOC2007 full detection dataset gate

Status: **preparation implementation complete; data not downloaded or
prepared by this change**.

This gate is restricted to COCO-style object detection preflight for the
repository model identifiers:

- `dino`
- `deformable_detr`
- `rtdetr`

It does not authorize or prepare data for grounding, semantic segmentation,
instance segmentation, or panoptic segmentation. It also does not launch
training, evaluation, AutoML, or SLURM jobs.

## Dataset identity and rationale

PASCAL VOC2007 is a complete, moderately sized detection dataset with 20
object categories and 9,963 images. The published evaluation statistics count
24,640 **non-difficult** objects; additional objects marked `difficult` are
present in the XML and must also be retained. The official release includes
annotated train, validation, trainval, and test splits. It is small enough for
repeated local correctness checks while exercising actual multi-class object
detection rather than the earlier synthetic subset.

Frozen source metadata is in [`manifest.v1.json`](manifest.v1.json).

| Split | Images | Published non-difficult objects | Intended use |
| --- | ---: | ---: | --- |
| train | 2,501 | 6,301 | AutoML candidate training |
| val | 2,510 | 6,307 | AutoML observations, feasibility, and selection |
| trainval | 5,011 | 12,608 | Optional final retraining after all policies are frozen |
| test | 4,952 | 12,032 | One-time reporting/validation only |
| combined | 9,963 | 24,640 | Completeness audit only |

The test non-difficult count is the difference between the official combined
and trainval evaluation totals. The frozen manifest deliberately leaves total
XML-object counts null rather than misrepresenting non-difficult statistics
as all-object counts. Preparation derives total and difficult counts from the
checksum-verified XML, proves that every source object reaches COCO, and
records both kinds of count. Test annotations must not be exposed to candidate
recommendation, threshold selection, search-space changes, early stopping, or
winner selection.

Authoritative references:

- [VOC2007 dataset and official archive links](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html)
- [VOC2007 development-kit data and annotation contract](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/htmldoc/)
- [Official train/val/trainval statistics](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/dbstats.html)
- [Commit-pinned TorchVision VOC loader publishing archive MD5 identities](https://github.com/pytorch/vision/blob/01dfa8ea81972bb74b52dc01e6a1b43b26b62020/torchvision/datasets/voc.py)

## Rights, terms, and citation

This repository does not assert a dataset-wide SPDX license or redistribution
permission. VOC2007 includes Flickr-sourced images whose rights and terms may
vary. Before downloading or using the data, the operator must review:

- the official VOC2007 database-rights notice;
- the linked Flickr/source-image terms;
- the official VOC2007 citation request;
- the organization's legal and data-governance requirements.

The preparer requires `--accept-dataset-terms` as an explicit record that this
review occurred. The flag is not legal advice and does not grant rights.
Prepared images must not be committed to this repository or redistributed
based on this dataset card.

## Archive integrity

The Oxford dataset page publishes the two archive URLs but does not publish
SHA-256 values. The frozen manifest therefore does not invent SHA-256
identities. It records the MD5 values published by the versioned official
PyTorch TorchVision dataset loader:

| Archive | Published MD5 |
| --- | --- |
| `VOCtrainval_06-Nov-2007.tar` | `c52e279531787c972589f7e41ab4ae64` |
| `VOCtest_06-Nov-2007.tar` | `b6e924de25625d8de591ea690078ad9f` |

MD5 is used only to bind the files to that published dataset identity. After
verification, the preparer computes and records SHA-256 values for both local
archives, the extracted source tree, every COCO JSON output, the converter,
and the frozen manifest. Computed SHA-256 values are local provenance; they
are not represented as publisher-provided checksums.

## Preparation contract

[`prepare_voc2007.py`](prepare_voc2007.py) uses only the Python standard
library and has no download code. It:

1. requires both complete, checksum-matching archives;
2. refuses `.part`, `.partial`, or `.tmp` remnants in place of an archive;
3. rejects absolute paths, `..`, backslashes, links, devices, FIFOs, duplicate
   members, overwrite collisions, oversized members, and archive expansion
   beyond frozen limits;
4. extracts into a newly created staging directory;
5. validates exact train/val/trainval/test membership and disjointness;
6. requires the source JPEG/XML inventory to match the 9,963 split IDs
   exactly;
7. verifies JPEG frame dimensions against every XML annotation;
8. verifies every category, bounding box, difficult flag, published
   non-difficult count, and published trainval non-difficult per-category
   count, while deriving total XML-object counts;
9. converts in ascending six-digit VOC image-ID order and XML object order;
10. atomically publishes a new output directory only after every gate passes;
11. refuses to overwrite an existing output.

VOC boxes are one-based and inclusive. COCO boxes are generated reversibly:

```text
x = xmin - 1
y = ymin - 1
width = xmax - xmin + 1
height = ymax - ymin + 1
```

Each COCO annotation retains the exact source coordinates in `voc_bbox` and
the exact `difficult` value. `difficult` is never silently mapped to
`iscrowd`; `iscrowd` remains zero. Pose, truncation, optional occlusion, and
person-layout part metadata are retained as custom VOC metadata. Unknown
object or part fields fail closed instead of being silently discarded.
Official person-layout part boxes may contain decimal coordinates even though
the detection-object boxes are integral. The converter accepts finite decimal
part coordinates, validates their ordering and image bounds, and preserves
them without rounding in `voc_metadata.parts[].voc_bbox`.

The output is:

```text
<prepared-root>/
  VOCdevkit/VOC2007/
    Annotations/
    ImageSets/
    JPEGImages/
    ...
  coco/annotations/
    instances_train2007.json
    instances_val2007.json
    instances_trainval2007.json
    instances_test2007.json
  integrity.v1.json
  integrity.v1.json.sha256
```

All four annotation files reference the shared
`VOCdevkit/VOC2007/JPEGImages` directory.

## Reproducibility commands

The commands below record the completed preparation workflow and can be
reused by an authorized operator.

Choose explicit locations:

```bash
VOC_ARCHIVE_DIR=/absolute/path/to/voc2007-archives
VOC_PREPARED_DIR=/absolute/path/to/voc2007-coco
VOC_GATE_DIR=/localhome/local-rarunachalam/tao-automl/experiments/cross_model_automl_20260729/datasets/voc2007
install -d -m 0755 "${VOC_ARCHIVE_DIR}"
```

After reviewing and accepting the terms, download to partial filenames using
HTTPS-only redirects:

```bash
curl --fail --location --proto '=https' --proto-redir '=https' \
  --output "${VOC_ARCHIVE_DIR}/VOCtrainval_06-Nov-2007.tar.part" \
  'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
curl --fail --location --proto '=https' --proto-redir '=https' \
  --output "${VOC_ARCHIVE_DIR}/VOCtest_06-Nov-2007.tar.part" \
  'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
```

Verify the published identities before removing the `.part` suffix:

```bash
printf '%s  %s\n' \
  'c52e279531787c972589f7e41ab4ae64' \
  "${VOC_ARCHIVE_DIR}/VOCtrainval_06-Nov-2007.tar.part" | md5sum --check -
printf '%s  %s\n' \
  'b6e924de25625d8de591ea690078ad9f' \
  "${VOC_ARCHIVE_DIR}/VOCtest_06-Nov-2007.tar.part" | md5sum --check -
mv "${VOC_ARCHIVE_DIR}/VOCtrainval_06-Nov-2007.tar.part" \
  "${VOC_ARCHIVE_DIR}/VOCtrainval_06-Nov-2007.tar"
mv "${VOC_ARCHIVE_DIR}/VOCtest_06-Nov-2007.tar.part" \
  "${VOC_ARCHIVE_DIR}/VOCtest_06-Nov-2007.tar"
```

Run the repository verifier, prepare atomically, and independently revalidate:

```bash
PYTHONDONTWRITEBYTECODE=1 python "${VOC_GATE_DIR}/prepare_voc2007.py" \
  --manifest "${VOC_GATE_DIR}/manifest.v1.json" \
  verify-archives \
  --archives-dir "${VOC_ARCHIVE_DIR}"

PYTHONDONTWRITEBYTECODE=1 python "${VOC_GATE_DIR}/prepare_voc2007.py" \
  --manifest "${VOC_GATE_DIR}/manifest.v1.json" \
  prepare \
  --archives-dir "${VOC_ARCHIVE_DIR}" \
  --output "${VOC_PREPARED_DIR}" \
  --accept-dataset-terms

PYTHONDONTWRITEBYTECODE=1 python "${VOC_GATE_DIR}/prepare_voc2007.py" \
  --manifest "${VOC_GATE_DIR}/manifest.v1.json" \
  validate \
  --dataset-root "${VOC_PREPARED_DIR}"
```

Run the contract tests without creating cache artifacts:

```bash
cd /localhome/local-rarunachalam/tao-automl
PYTHONDONTWRITEBYTECODE=1 \
  PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  pytest -p no:cacheprovider -q \
  experiments/cross_model_automl_20260729/datasets/voc2007/test_prepare_voc2007.py
```

## Gate state

The full official archives were downloaded and prepared on 2026-07-29 after
the database-rights and source-image notices were reviewed. The repository
preparer performed no network access. The immutable real-data identities are:

```text
trainval archive SHA-256:
  7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508
test archive SHA-256:
  6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892
prepared integrity SHA-256:
  cc9b9d450a15358ee8033ea8f8aa9a2fb305b060a133e491b3c5205abb20ec39
train COCO SHA-256:
  fbb0c5e68745eba2ab8e849a17e4eb874a38db891aff731cab3948ed7bf2b6e7
validation COCO SHA-256:
  e984d6481ec999735876b3b608c3e8fa3e61b390e7dae4ee7bc84f8f9d4b95f2
test COCO SHA-256:
  9f0fdcc3a76b2f259775965b0bc71f89d7c06af91a84fb7e9c9cdaf077040ce9
```

| Gate | Current state |
| --- | --- |
| Frozen manifest and terms documented | Pass |
| Official archives downloaded | Pass |
| Published archive checksums verified | Pass |
| Full source annotation/inventory validation | Pass |
| Deterministic COCO conversion | Pass |
| Prepared artifact hashes recorded | Pass |
| DINO local batch/epoch/eval preflight | Not run |
| Deformable DETR local preflight | Not run |
| RT-DETR local preflight | Not run |
| SLURM ready | **No** |

No model may proceed to the cross-model SLURM campaign until its local
model/PTM preflight gates pass.
