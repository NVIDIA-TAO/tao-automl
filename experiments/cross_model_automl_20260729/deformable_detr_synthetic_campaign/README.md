# Deformable DETR shared-dataset direct gate

This campaign preserves the earlier VOC2007 qualification unchanged and runs
both official Deformable DETR PTM arms directly on the shared four-class
synthetic COCO dataset. Each arm uses one eight-A100 allocation for ten full
epochs and a standalone evaluation. It contains no CPU, smoke, or mini-step
model execution.

The sealed campaign deliberately reuses the reviewed direct qualification
launcher from `deformable_detr_campaign`; only the immutable dataset and
current fixed SLURM SDK contract differ.
