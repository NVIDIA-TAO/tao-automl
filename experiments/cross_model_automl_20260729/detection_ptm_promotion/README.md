# Detection PTM qualification promotion

This directory contains the evidence-only bridge from the completed,
full-dataset Deformable DETR and RT-DETR qualifications to a reviewable
candidate PTM registry.

The utility does not execute a model, submit a job, select a checkpoint, or
modify the repository registry. It verifies the sealed manifest and completion
hashes, the complete registry/PTM/workflow population, checkpoint identities,
ten-epoch eight-GPU training evidence, standalone evaluation evidence, and all
agent-intervention flags. Exactly the successful workflows are promoted in a
create-only candidate document; failed records remain byte-for-byte unchanged
and are retained in the audit.

RT-DETR consumes `completion.resume.json`. The resumed evidence must reuse all
original completed training jobs, submit zero replacement training jobs, and
preserve the initial completion artifact. During promotion, a preregistered
compatibility projection derives these values only from each frozen registry
`input_contract`:

```yaml
dataset:
  augmentation:
    train_spatial_size: [height, width]
    eval_spatial_size: [height, width]
    preserve_aspect_ratio: <input_contract.preprocessing value>
```

The projection prevents loss of checkpoint-specific 544×960 versus 640×640
contracts. It does not inspect accuracy, latency, or any other observed result.

Example:

```bash
PYTHONPATH=src \
python experiments/cross_model_automl_20260729/detection_ptm_promotion/promotion.py \
  --base-registry src/tao_automl/data/ptm_registry.v1.json \
  --qualification deformable_detr \
    /path/to/deformable-detr-manifest.json \
    /path/to/completion.json \
  --qualification rtdetr \
    experiments/cross_model_automl_20260729/rtdetr_campaign/campaign.v1.json \
    /path/to/completion.resume.json \
  --registry-version detection-qualified-7.1.0-v1 \
  --output-registry /path/to/candidate_registry.json \
  --audit /path/to/promotion_audit.json
```

The output and audit paths are create-only and may not equal the base registry
path. Promotion into the live repository registry remains a separate reviewed
change.
