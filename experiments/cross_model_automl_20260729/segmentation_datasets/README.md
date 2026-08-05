# Segmentation dataset staging

This directory records the data-only preparation of complete public datasets
for the four segmentation models in the cross-model AutoML plan. It does not
contain model, checkpoint, training, evaluation, latency, CPU-smoke, GPU, or
SLURM evidence.

| Model | Staged dataset | Product-path scope |
| --- | --- | --- |
| `segformer` | PASCAL VOC2012 semantic segmentation train/val | 21 indexed classes including background; ignore ID 255 |
| `oneformer` | COCO2017 panoptic train/val | Native 133-category panoptic data |
| `mask2former` | COCO2017 instance and panoptic train/val | Native 80-thing instance task; panoptic assets co-staged for the supported alternate mode |
| `mask_grounding_dino` | COCO2017 instance train/val projected to ODVG | Category-prompted grounded instance segmentation only; not RefCOCOg phrase grounding |

The local source roots are:

```text
/localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/voc2012_segmentation_full
/localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/coco2017_full
```

The immutable Lustre targets are:

```text
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/voc2012_segmentation_v1
/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/cross_model_automl_20260729/coco2017_instance_panoptic_v1
```

See [VOC2012_DATASET_CARD.md](VOC2012_DATASET_CARD.md) and
[COCO2017_SEGMENTATION_DATASET_CARD.md](COCO2017_SEGMENTATION_DATASET_CARD.md)
for the source, license, split, conversion, and model-contract details.
The data-contract fragments are
`voc2012_segformer_dataset_profile.yaml` and
`coco2017_tao_dataset_bindings.yaml`; they contain no training budget,
search-space, PTM, or launch decision.

## Data-only verification

The verifier deliberately imports only Pillow and NumPy. It validates every
VOC image/mask pair, every native COCO reference, every COCO image/panoptic
mask dimension, every panoptic segment ID and area, and every source-to-ODVG
instance field and mask.

```bash
cd /localhome/local-rarunachalam/.tao/worktrees/tao-automl-segmentation-datasets
/localhome/local-rarunachalam/.tao/venvs/segmentation-dataset-validation-py314/bin/python \
  -m experiments.cross_model_automl_20260729.segmentation_datasets.verify_datasets \
  --voc-root /localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/voc2012_segmentation_full \
  --voc-source-archive /localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/public_sources/voc2012_segmentation/VOCtrainval_11-May-2012.tar \
  --coco-root /localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/coco2017_full \
  --output /localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/segmentation_dataset_validation.v1.json \
  --deep-panoptic-pixel-check
```

The canonical committed staging manifest records the content hash of the
machine-readable report and the SHA256 manifests used to verify the Lustre
copies. The report itself records:

```text
data_only_validation = true
model_invoked = false
slurm_job_submitted = false
```

Dataset staging does not automatically authorize model execution. The metric,
PTM, container, and campaign gates in
`docs/cross_model_automl/dataset_preflight_matrix.md` remain independent.
