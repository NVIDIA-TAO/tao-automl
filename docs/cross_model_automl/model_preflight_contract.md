# Local model-preflight contract

The local preflight is a fail-closed readiness gate between repository/PTM
qualification and any SLURM reservation. It is an orchestration contract, not
an execution backend: `tao_automl.model_preflight` never downloads a dataset or
checkpoint, starts TAO, selects a GPU, or submits a scheduler job.

A model integration implements the `ModelPreflightAdapter` protocol. The
orchestrator sends an immutable `ModelPreflightStepRequest`, validates the
returned `ModelPreflightStepResult`, derives task-aware metric sanity using
`tao_automl.metric_sanity`, and seals the evidence in an ordered SHA-256 chain.

## Supported identifiers

The contract accepts exactly the repository's eight canonical identifiers.
Aliases must be resolved before constructing the input:

| Model ID | Task contract |
| --- | --- |
| `dino` | `object_detection` |
| `deformable_detr` | `object_detection` |
| `rtdetr` | `object_detection` |
| `grounding_dino` | `referring_expression_box_grounding` |
| `segformer` | `semantic_segmentation` |
| `oneformer` | `panoptic_segmentation` |
| `mask2former` | `instance_segmentation` |
| `mask_grounding_dino` | `referring_expression_segmentation` |

This does not weaken the metric-sanity registry. A model/metric pair whose
runtime contract is unverified or blocked is accepted as a preflight input but
fails the derived `metric_sanity` stage, so it cannot become SLURM-ready.

## Immutable inputs

`ModelPreflightInputs` binds the run to:

- full source commit, wheel/package hash, container hash, and TAO version;
- dataset manifest, annotation contract, and train/validation split hashes;
- the default PTM and the complete eligible PTM inventory, including
  checkpoint, registry-record, and PTM-preflight hashes;
- merged specification, metric, latency protocol/input/scope, and output
  contract identities;
- a preregistered seed and exactly one local GPU.

Eligible PTMs are sorted by stable ID before hashing. Duplicate PTM IDs,
non-finite JSON, truncated hashes, task/model mismatches, and non-single-GPU
inputs are rejected before an adapter is called.

## Required order and evidence

The stage list is exact. Completed reports cannot omit, duplicate, append,
rename, or reorder a stage.

| Index | Stage | Required evidence |
| ---: | --- | --- |
| 0 | `dataset_validation` | Manifest and split hashes, annotation-contract hash, non-empty train/validation sample counts, valid annotations |
| 1 | `default_ptm_load` | Exact default PTM/checkpoint, successful load, input-contract check, merged-spec check |
| 2 | `eligible_ptm_smoke` | Exactly one record for every eligible PTM; load plus train, validation, and inference mini-steps |
| 3 | `default_model_full_epoch` | Default PTM, one local GPU, at least one complete epoch, nonzero batches/steps, final checkpoint hash |
| 4 | `in_epoch_validation` | Finite named primary metric and a completed in-epoch evaluation |
| 5 | `standalone_evaluation` | Finite same primary metric, completed eval, verified runtime metric contract |
| 6 | `metric_sanity` | Orchestrator-derived `MetricSanityDecision`; the adapter is not invoked |
| 7 | `checkpoint_save_reload` | Saved and reloaded default-model checkpoint with identical content, also matching the full-epoch checkpoint |
| 8 | `latency_instrumentation` | Frozen protocol/input/scope, synchronization, warm-up, at least two timed rounds, finite robust statistics, passed quality gates |
| 9 | `output_artifact_validation` | Frozen output contract, non-empty unique artifact identities and hashes, no missing artifact |
| 10 | `interrupted_resume_replay` | Saved interrupted state, deterministic next-request replay, no duplicated or lost trials |

The default's complete epoch and the all-PTM mini-step obligation are
intentionally separate. `all_ptms_smoke_tested=true` cannot substitute for
`default_one_epoch_passed=true`, and a default epoch cannot hide an untested
eligible checkpoint.

## Adapter boundary

An adapter returns only a stage identifier, pass bit, safe machine code, and
the stage's strict evidence object. Unknown evidence fields are rejected.
Successful evidence is canonicalized and order-normalized where applicable.

Adapter exceptions are caught at the boundary. Reports preserve only a safe
exception class name and a fixed message; they never preserve `str(exception)`
or arbitrary diagnostics which might contain credentials. Invalid adapter
evidence is not copied into the report. Only its canonical digest is retained
when it can be computed.

Minimal shape:

```python
def adapter(request: ModelPreflightStepRequest) -> ModelPreflightStepResult:
    evidence = execute_model_specific_step(request)
    return ModelPreflightStepResult.success(request.stage, evidence)

report = run_model_preflight(frozen_inputs, adapter)
assert report["slurm_ready"]
```

The integration behind `execute_model_specific_step` owns framework-specific
training, validation, inference, checkpoint, and latency operations. It must
not mutate the request or use unsealed inputs.

## Metric sanity

The orchestrator builds `MetricEvidence` from:

- verified annotations from `dataset_validation`;
- distinct training steps from `default_model_full_epoch`;
- the in-epoch validation result;
- the standalone evaluation and runtime metric contract.

It evaluates the standalone metric with the repository-owned task-aware
registry. This is an experiment-validity gate only. It neither computes nor
replaces latency mode's archive-relative retained-accuracy policy.

Unknown, blocked, non-finite, out-of-range, or evidence-deficient metric
contracts produce a terminal structured failure.

## Content addressing and resume

Every record contains:

- stage and index;
- deterministic request hash;
- status, fixed reason, and safe code;
- canonical evidence and its hash;
- previous-record hash;
- record hash.

The report additionally seals inputs, provenance, exact stage order, readiness,
failure state, and its own content hash. There are no timestamps in the
canonical identity.

`stop_after_stage` produces a validated `interrupted` strict prefix. Passing it
back through `resume_report` verifies the input identity, report hash, evidence
hashes, record chain, and exact order before the next adapter call. Resuming
with changed inputs or from a terminal failure is rejected. A deterministic
resume produces the same final report hash as an uninterrupted run.

The in-model `interrupted_resume_replay` stage is separate evidence that TAO
checkpoint/search state itself resumes deterministically; orchestrator-prefix
resume cannot stand in for that product behavior.

## Readiness semantics

The report exposes separate booleans for:

- dataset preparation;
- default PTM load;
- complete eligible-PTM smoke coverage;
- default one-epoch training;
- in-epoch validation;
- standalone evaluation;
- task-aware metric sanity;
- checkpoint save/reload;
- stabilized latency;
- output artifacts;
- interrupted/resume replay.

`slurm_ready` is derived and is true only when every field is true, every stage
is present in the required order, and every stage passed. Adapters cannot set
or override readiness.

## Execution restraint

Creating or testing this contract performs no external action. Actual local
preflight evidence must be produced later by a model-specific adapter using
the frozen dataset, PTM, runtime, and protocol identities. A cross-model
campaign manifest must not be submitted until the corresponding completed
report is intact and `slurm_ready=true`.
