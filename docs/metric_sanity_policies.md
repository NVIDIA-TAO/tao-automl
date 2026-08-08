# Task-aware metric sanity policies

`tao_automl.metric_sanity` provides a fail-closed validation gate for
cross-model AutoML preflight and campaign evidence. It answers whether a
reported metric is finite, within its verified mathematical scale, and backed
by the preregistered data, training, evaluation, and runtime-contract
evidence.

It does **not** decide whether a candidate is accurate enough for latency
mode. These are separate product concepts:

- **Validation sanity gate:** rejects broken data, invalid metric values,
  missing evaluation evidence, and unverified model/metric contracts.
- **Product latency feasibility:** derives a quality constraint from the
  candidates observed by the AutoML job and is evaluated by the latency
  selector.

Every `MetricSanityDecision` records this separation as:

```text
gate_type = validation_sanity_gate
product_latency_feasibility = not_evaluated
```

There is deliberately no universal `metric >= 0.1` rule. AP, mIoU, PQ, and
IoU have different semantics and may also use fraction or percentage scales.
A low finite metric can pass the mathematical-scale gate if its
model/task/metric contract and required experiment evidence are valid.
Campaigns that require a stronger dataset-specific sanity floor or evidence
of learning must preregister a narrowing override.

## Repository-verified policies

The source inspection below is pinned to TAO PyTorch commit
`2fbd1f1246002e5212e99e864f2713abab060656` and TAO AutoML base commit
`6d64bd44c34f3dc51e09b40bacd4aed3491067ac`.

| Model | Task | Requested metric | Verified scale | Status | Evidence or blocker |
| --- | --- | --- | --- | --- | --- |
| DINO | Object detection | `val_mAP`, `val_mAP50` | `[0, 1]` fraction | Supported | `dino/model/pl_dino_model.py` logs COCO `bbox.stats[0:2]` directly. |
| Deformable DETR | Object detection | `val_mAP`, `val_mAP50` | `[0, 1]` fraction | Supported | `deformable_detr/model/pl_dd_model.py` logs COCO `bbox.stats[0:2]` directly. |
| RT-DETR | Object detection | `val_mAP`, `val_mAP50` | `[0, 1]` fraction | Supported | `rtdetr/model/pl_rtdetr_model.py` logs COCO `bbox.stats[0:2]` directly. |
| Grounding DINO | Category-prompted object detection | `val_mAP`, `val_mAP50` | `[0, 1]` fraction | Supported | The validation loader uses contiguous-ID COCO and the model logs unscaled COCO bbox AP/AP50. This policy does not turn plain category annotations into referring expressions. |
| Grounding DINO | Referring-expression box grounding | `val_Pr@0.5` | Unverified | Blocked | The inspected validation loader uses `CocoDetection`, and the model validation path emits COCO AP rather than a phrase-grounding `Pr@0.5` contract. |
| SegFormer | Semantic segmentation | `val_miou` | `[0, 1]` fraction | Supported | `segformer/utils/iou_metric.py` averages intersection/union ratios and `segformer_pl_model.py` reports `val_miou`. |
| OneFormer | Panoptic segmentation | `PQ` | Unverified | Blocked | The inspected validation path reports semantic mIoU and accuracy, not PQ. |
| Mask2Former | Instance segmentation | `segm_val_mAP` | Unverified | Blocked | The inspected validation path reports semantic mIoU and accuracy, not COCO mask AP. |
| Mask Grounding DINO | Referring-expression segmentation | `val_overall_IoU` | `[0, 100]` percent | Supported | The evaluator explicitly computes `100 * sum(intersection) / sum(union)` and the model prefixes validation output with `val_`. Runtime-contract evidence remains required. |
| Mask Grounding DINO | Referring-expression segmentation | `val_cIoU` | Unverified | Blocked | `val_cIoU` is a legacy enum/docstring name; the inspected runtime evaluator emits `overall_IoU`. The names are not treated as interchangeable. |

A blocked policy is an explicit preflight result, not a guessed metric scale.
The model cannot enter the cross-model campaign for that task/metric until its
TAO evaluation path provides and verifies the requested runtime contract.
Unknown models and metrics raise `UnknownMetricPolicyError` with a
machine-readable reason rather than falling back to generic bounds.

## Evidence contract

The default supported-metric policy requires:

- at least one completed evaluation;
- at least one distinct training step;
- verified dataset annotation and label semantics;
- a passed standalone evaluation;
- a verified runtime metric name and scale.

Campaign profiles can preregister stronger requirements with
`EvidencePolicy`, including more completed evaluations or training steps and
a minimum observed learning delta. Learning evidence compares the first and
best values on the same verified metric scale.

Metric values must be numeric and finite. Python and NumPy booleans are
rejected rather than coerced to zero or one. Evidence counts must be
non-negative integers. Missing evidence and each failed evidence condition
produce structured reason codes.

## Preregistered overrides

An experiment may narrow a repository policy but cannot expand its verified
scale:

```python
from tao_automl.metric_sanity import (
    EvidencePolicy,
    MetricEvidence,
    MetricSanityOverride,
    evaluate_metric_sanity,
)

override = MetricSanityOverride(
    minimum_value=0.02,
    evidence_policy=EvidencePolicy(
        minimum_completed_evaluations=2,
        minimum_distinct_training_steps=100,
        require_observed_improvement=True,
        minimum_improvement=0.01,
    ),
)

decision = evaluate_metric_sanity(
    "segformer",
    "val_miou",
    reported_miou,
    evidence=MetricEvidence(
        completed_evaluations=2,
        distinct_training_steps=100,
        annotation_contract_verified=True,
        standalone_evaluation_passed=True,
        runtime_metric_contract_verified=True,
        first_metric_value=first_miou,
        best_metric_value=best_miou,
    ),
    override=override,
)
```

The override is a validation-only campaign contract. It is not a retained
accuracy threshold and must not be passed to candidate selection as one.
Ranges, evidence thresholds, and learning requirements must be frozen before
campaign results are inspected.

## Integrity and reproducibility

The default registry, each base policy, and each effective policy expose
canonical SHA-256 identities:

```python
from tao_automl.metric_sanity import default_metric_sanity_registry

registry = default_metric_sanity_registry()
print(registry.canonical_sha256)
print(registry.to_json())
```

Canonical serialization sorts policies, aliases, and source evidence, so the
registry hash does not depend on declaration order. A campaign manifest
should record:

- the registry SHA-256;
- the selected base-policy SHA-256;
- the effective-policy SHA-256 after preregistered overrides;
- the serialized evidence and structured decision;
- the AutoML and TAO source commits.

Changing bounds, evidence requirements, source evidence, availability, task,
metric identity, or scale changes the corresponding canonical hash.
