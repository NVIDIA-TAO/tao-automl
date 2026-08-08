# DINO local preflight execution

This directory contains the executable single-GPU gate that must pass before
the DINO three-job pilot can reserve SLURM resources. Planning is read-only;
execution uses `DockerSDK` and the DINO skill `build_entrypoint` contract.
Nothing here pulls an image, downloads a checkpoint, submits SLURM, or logs a
credential value.

## Reviewed TAO 7.1 boundary

The authoritative skill is:

```text
/localhome/local-rarunachalam/.tao/worktrees/tao-skills-release-7.1.0/skills/models/tao-train-dino
commit 2e9c1b25f3c7cb1ae444c75652e36c47eace8229
skill tag nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.1.0-rc-245-multiarch
```

Physical execution is pinned to:

```text
nvcr.io/nvstaging/tao/tao-toolkit-pyt@sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667
```

`build_reviewed_runtime_image_contract()` verifies the content-addressed
release/7.1 skill plus the reviewed TAO DINO train/evaluate/inference/schema
source evidence before constructing this tag-to-repository-digest mapping.
The old checked-out 7.0.1 skill is not used for a real plan. The executor
rejects any image other than the exact mapped digest and runs only
`docker image inspect`; it never calls `docker pull`.

The reviewed TAO train source maps `train.is_dry_run` directly to Lightning
`Trainer(fast_dev_run=...)`. Thus the standard skill train action performs one
train and one validation batch for each full-detector PTM smoke. A standard
evaluate action over the executor-created one-batch annotation and a standard
inference action over one real VOC image complete that smoke contract.
Backbone-only PTMs are never passed as full detector checkpoints: the fixed
mounted worker binds `model.pretrained_backbone_path`, constructs the real TAO
DINO model/data module, and performs exactly one train, validation, and
eval-mode inference batch.

## Concrete factories

The plan factory is
`dino_local_factories.DINOAuthoritativePlanFactory`. It needs:

- the absolute frozen VOC manifest path;
- the absolute prepared full VOC2007 root;
- the live `ResolvedPTMRuntimeInventory` returned in the same Python process
  by production checkpoint preflight and `resolve_ptm_runtime_inventory`;
- `DINOPreflightSettings` whose runtime contract comes from
  `build_reviewed_runtime_image_contract()`.

The live typed inventory is intentional. An audit JSON omits the validated
checkpoint-spec documents and is not accepted as a substitute.

A deployment module exposes one zero-argument instance:

```python
from dino_local_factories import (
    DINOAuthoritativePlanFactory,
    build_reviewed_runtime_image_contract,
)
from dino_preflight import DINOPreflightSettings

# `resolved_inventory` is the live production object created immediately
# before this block; no serialized report is rehydrated.
runtime_contract = build_reviewed_runtime_image_contract()
settings = DINOPreflightSettings(
    preflight_id="dino.voc2007.local.v1",
    tao_version="7.1.0-rc-245",
    source_commit="<40-hex-tao-automl-commit>",
    package_sha256="<64-hex-wheel-sha256>",
    container_sha256=runtime_contract.runtime_digest,
    runtime_sha256="<64-hex-runtime-contract-sha256>",
    runtime_image_contract=runtime_contract,
    latency_input_descriptor={
        "shape": [1, 3, 544, 960],
        "dtype": "float32",
        "content": "seeded_preflight_tensor",
    },
    seed=271828,
    batch_size=1,
    precision="fp32",
)
plan = DINOAuthoritativePlanFactory(
    voc_manifest_path=VOC_MANIFEST,
    voc_dataset_root=VOC_ROOT,
    resolved_ptm_inventory=resolved_inventory,
    settings=settings,
)
```

The default hooks factory is
`dino_local_factories:build_default_hooks`; it need not be supplied on the
CLI. Its latency path runs the fixed worker in the same exact image with one
GPU, 50 warm-ups, five rounds of 100 synchronized model forwards, and the
production `run_replica_benchmark` contract. Its replay path uses the
production `Bayesian`, `Controller`, and local `StateStore`, interrupts after
one recommendation/result, reloads state, and proves the next audited request
matches an uninterrupted run.

## External executor configuration

Configuration is YAML and contains environment names only, never their
values. `plan_sha256` is obtained by calling the live plan factory before
execution.

```yaml
schema_version: 1
plan_sha256: <64-hex-plan-sha256>
image: nvcr.io/nvstaging/tao/tao-toolkit-pyt@sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667
results_root: /absolute/preflight/results
mounts:
  - host_path: /absolute/prepared/voc2007
    container_path: /dataset
    read_only: true
  - host_path: /absolute/ptm/cache
    container_path: /ptm
    read_only: true
  - host_path: /absolute/preflight/results
    container_path: /results
    read_only: false
required_environment: []
poll_interval_seconds: 5
max_polls: 720
shm_size: 16g
container_user: "<non-root-uid>:<gid>"
```

Every absolute dataset, checkpoint, and checkpoint-spec path in the live plan
must fall beneath one declared bind. The executor translates paths by the
longest matching bind and rejects an undeclared path before job submission.

The production preparation launcher performs `PTMCheckpointPreflight.run`,
creates the live typed inventory with `resolve_ptm_runtime_inventory`, builds
and freezes the plan/config, and invokes the executor in the same process.
It rejects a dirty source tree and hashes the supplied wheel after verifying
its production contents. After `NGC_KEY` has been exported from
`~/.tao/config.env` without printing it, execute:

```bash
set -euo pipefail
TAO_AUTOML_REPO=/localhome/local-rarunachalam/tao-automl
TAO_AUTOML_PYTHON=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin/python
TAO_AUTOML_COMMIT=$(git -C "$TAO_AUTOML_REPO" rev-parse HEAD)
TAO_AUTOML_BUILD_SRC=$(mktemp -d "/tmp/tao-automl-wheel.${TAO_AUTOML_COMMIT:0:12}.XXXXXX")
TAO_AUTOML_WHEEL_DIR=/localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/wheel/${TAO_AUTOML_COMMIT:0:12}
test ! -e "$TAO_AUTOML_WHEEL_DIR"
mkdir -p "$TAO_AUTOML_WHEEL_DIR"
git -C "$TAO_AUTOML_REPO" archive --format=tar "$TAO_AUTOML_COMMIT" \
  | tar -xf - -C "$TAO_AUTOML_BUILD_SRC"
export SOURCE_DATE_EPOCH
SOURCE_DATE_EPOCH=$(git -C "$TAO_AUTOML_REPO" show -s --format=%ct "$TAO_AUTOML_COMMIT")
PIP_NO_INDEX=1 "$TAO_AUTOML_PYTHON" -m pip wheel \
  --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$TAO_AUTOML_WHEEL_DIR" "$TAO_AUTOML_BUILD_SRC"
TAO_AUTOML_WHEEL=$(find "$TAO_AUTOML_WHEEL_DIR" -maxdepth 1 -type f \
  -name 'nvidia_tao_automl-*.whl' -print -quit)
test -n "$TAO_AUTOML_WHEEL"
sha256sum "$TAO_AUTOML_WHEEL"
cd "$TAO_AUTOML_REPO"
set -a
source /localhome/local-rarunachalam/.tao/config.env
set +a
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  python experiments/cross_model_automl_20260729/dino_preflight/dino_local_launch.py \
  --source-repo /localhome/local-rarunachalam/tao-automl \
  --wheel "$TAO_AUTOML_WHEEL" \
  --registry-path /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/candidate_registry.v2.json \
  --voc-manifest /localhome/local-rarunachalam/tao-automl/experiments/cross_model_automl_20260729/datasets/voc2007/manifest.v1.json \
  --voc-root /localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/voc2007/prepared \
  --ptm-cache /localhome/local-rarunachalam/.tao/cache/cross_model_automl_20260729/dino_ptms \
  --results-root /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_local_preflight \
  --plan /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_local_preflight/dino_preflight_plan.v1.json \
  --executor-config /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_local_preflight/executor.v1.yaml \
  --report /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_local_preflight/report.v1.json \
  --container-user "$(id -u):$(id -g)"
```

This is the primary runnable seam; it never serializes then reconstructs the
PTM runtime inventory. `DINOAuthoritativePlanFactory` remains the
zero-argument plan object used inside that process. The lower-level executor
CLI is available for an already-live deployment module, but an audit JSON is
never accepted as its plan source.

The `--registry-path` input must be the create-only candidate registry
generated from the exact CPU/GPU qualification intersection. The launcher
still uses production `PTMCheckpointPreflight.run`; it does not accept a
qualification report or weaken the supported/runtime-eligible boundary.

## Verification without Docker or GPU execution

```bash
cd /localhome/local-rarunachalam/tao-automl
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  pytest -q \
  experiments/cross_model_automl_20260729/dino_preflight/test_dino_preflight.py \
  experiments/cross_model_automl_20260729/dino_preflight/test_dino_local_executor.py
```

The suite uses fake SDK/process boundaries. It proves one-GPU requests,
SDK-generated YAML, fixed worker commands, no image pull, secret-safe errors,
exact status/metric/checkpoint parsing, immutable artifacts, the real
production state-replay implementation, and the full production latency
record contract without running a model, Docker container, network request,
or GPU job.
