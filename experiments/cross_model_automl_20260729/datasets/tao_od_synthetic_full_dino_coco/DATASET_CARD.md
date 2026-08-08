# TAO synthetic four-class object-detection dataset

This immutable campaign dataset is the staged copy of:

`s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/`

The campaign-visible root is:

`/lustre/fsw/portfolios/edgeai/users/rarunachalam/data/tao_od_synthetic_full_dino_coco`

It contains complete COCO-format train and validation splits used for the
shared DINO, Deformable DETR, and RT-DETR comparison. The four foreground
category IDs are `1..4`; TAO detector specifications therefore use
`dataset.num_classes: 5` and `dataset.eval_class_ids: [1, 2, 3, 4]`.

The content identities, file counts, byte counts, source URI, and exact
SLURM-visible paths are frozen in `manifest.v1.json`. No label conversion is
performed for these campaigns.
