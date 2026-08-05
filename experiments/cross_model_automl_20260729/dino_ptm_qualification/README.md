# DINO PTM qualification contract

This directory contains the executable qualification boundary for the
version-scoped DINO checkpoint metadata projection recorded in the
repository-owned PTM registry. It does not change checkpoint status,
compatibility, or runtime eligibility.

## Artifact projection

`DINOCheckpointMetadataProjectionCallback` accepts only the registered recipe:

```yaml
retain_top_level_keys: [state_dict]
add_top_level_metadata:
  tao_model: dino
tensor_container_key: state_dict
require_exact_tensor_key_set: true
require_exact_tensor_values: true
```

Before any checkpoint deserialization, both the production preflight cache and
the callback verify the official input size and SHA-256. Loading is restricted
to `torch.load(..., map_location="cpu", weights_only=True)`. The projected
checkpoint is reloaded the same way and its sorted tensor keys, dtypes, shapes,
and raw bytes are hashed before and after projection.

The campaign default does not require host PyTorch. It runs the transformation
inside this exact local image:

```text
nvcr.io/nvstaging/tao/tao-toolkit-pyt@sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667
```

Docker is invoked without a shell, with `--pull=never`, networking disabled,
a read-only single-file checkpoint mount, and an isolated read-write output
directory. The module is mounted read-only. Nothing in this contract pulls an
image. The image must already exist locally.

Serialization is staged under the registered wrapper filename because PyTorch
zip serialization incorporates that filename. The callback must reproduce the
registered output byte size and SHA-256. A difference is a structured
`adapted_output_size_mismatch` or `adapted_output_checksum_mismatch`; it is
never accepted as a new wrapper.

## Qualification driver

`run_dino_ptm_qualification` wires:

- `NGCCredential`;
- `NGCHTTPSClient`;
- `AtomicArtifactCache`;
- `PTMCheckpointPreflight`;
- the metadata-projection callback;
- the concrete `TAO71DINOCheckpointLoadSmoke`.

The load smoke uses the same exact TAO 7.1 image identity as the projection
callback and never pulls an image. It safely loads the verified checkpoint on
CPU with `weights_only=True`, constructs the official DINO
`ExperimentConfig` and `DINOPlModel`, and routes the immutable registered
`checkpoint_target`:

- `train.pretrained_model_path` executes the training entrypoint's
  shape-aware full-detector state-loading behavior;
- `model.pretrained_backbone_path` constructs the exact registered backbone
  with that path, while forcing TAO's internal `torch.load` call to
  `map_location="cpu", weights_only=True`.

The callback records the registered target and backbone, TAO path executed,
source, target, matched, missing, shape-mismatched, and unexpected tensor
counts, target coverage fractions, loaded-value matches, and deterministic
key-set digests. A full detector must cover at least 90% of target tensor
entries and 90% of target parameter volume. A backbone may omit small task
heads, but must cover at least 50% of target tensor entries and 90% of target
parameter volume. Every shape-compatible source tensor must be finite and
observed with the same value after TAO loading. These frozen
qualification-safety gates are not AutoML objectives and do not affect winner
selection.

Checkpoint-sidecar values are merged first and registered checkpoint defaults
second for this isolated qualification layer. Host checkpoint paths are
removed before the merged overrides are mounted. The checkpoint, merged
overrides, and worker are mounted as individual read-only files; only a fresh
temporary evidence directory is read-write. GPU visibility and networking are
disabled. Docker receives no NGC credential or other secret argument.

The load-smoke callback returns `CheckpointLoadSmokeResult` and, on success,
must include these exact details:

```text
contract_version = 1
execution_backend = docker
container_identity = sha256:949c0ea8ace09ac91951be4169353cf214daaa3ede7db9eed94070b020361667
tao_version = 7.1.0-rc-245
checkpoint_sha256 = <effective checkpoint SHA-256>
checkpoint_size_bytes = <effective checkpoint bytes>
checkpoint_loaded = true
state_dict_compatible = true
```

The driver creates:

```text
qualification_manifest.v1.json
qualification_completion.v1.json
```

Both are canonical, deterministic, secret-free, and create-only. A fresh run
fails if either frozen target already exists. `resume=True` requires a
byte-identical manifest. When completion exists, resume verifies its hashes,
qualification-isolation flags, and every cached source, adapted checkpoint,
and specification artifact without credential lookup, NGC access, Docker
execution, or file rewriting.

The manifest binds the SHA-256 and byte size of the projection worker, load
smoke worker, qualification driver, and production PTM registry/preflight
modules. Both Docker callbacks mount a private snapshot of the exact
manifest-bound worker bytes, so later source changes cannot silently alter a
frozen qualification run.

The driver records:

```text
qualification_only = true
runtime_eligibility_mutated = false
selection_invoked = false
agent_selected_checkpoint = false
```

The default qualification statuses are the lexicographically canonical
`supported, unverified`. Records classified `unsupported` remain structured
registry exclusions and are never probed or downloaded. In particular, this
keeps the TAO 7.1 GCViT `NotImplementedError` path out of checkpoint I/O.

## Required two-stage qualification

Runtime eligibility is never inferred from registry metadata or the CPU load
check alone.

1. The CPU stage below verifies download identity, metadata projection, TAO
   state loading, and tensor coverage for the complete registry-derived
   qualification population.
2. `train_validation_qualification.py` derives its population exclusively
   from the CPU completion and binds that completion hash. It runs the
   skill-defined TAO train action with `is_dry_run=true`, which the reviewed
   TAO 7.1 source maps to one real VOC2007 train batch and one validation
   batch on one GPU.
3. `registry_promotion.py` promotes only the exact CPU/GPU pass intersection.
   Candidate mode produces a provisional registry for the complete local
   preflight; it is explicitly not distribution-ready.
4. The complete local preflight must then prove train, validation, inference,
   a full epoch, standalone evaluation, checkpoint reload, latency, artifacts,
   and resume over every promoted PTM. Final promotion requires that sealed
   SLURM-ready report and exact registry-record hashes.

No command accepts a checkpoint ID. Failed and excluded records remain in the
evidence and are never replaced.

## CPU qualification command

The command-line driver accepts only non-secret paths and a resume flag. It
reads `NGC_KEY` from the process environment. Source the existing protected
configuration externally; do not put the credential on the command line.

The first immutable CPU attempt is preserved under `cpu/`. It exposed
qualification implementation defects documented in
`docs/cross_model_automl/dino_ptm_qualification_correction.md`. The corrected
run uses the new create-only `cpu_v2/` target; neither run is overwritten.

Run:

```bash
cd /localhome/local-rarunachalam/tao-automl
set -a
source /localhome/local-rarunachalam/.tao/config.env
set +a
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  python \
  experiments/cross_model_automl_20260729/dino_ptm_qualification/qualification_driver.py \
  --output-dir \
  /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/cpu_v2 \
  --cache-dir \
  /localhome/local-rarunachalam/.tao/cache/cross_model_automl_20260729/dino_ptms
```

The exact pinned image must already be present locally. The command does not
pull it. A completed run can be verified without credential lookup, network
access, Docker execution, or artifact rewriting by repeating the command with
`--resume`.

## Real-data train/validation qualification

The second stage uses the same verified cache and the prepared complete
VOC2007 dataset:

```bash
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  python \
  experiments/cross_model_automl_20260729/dino_ptm_qualification/train_validation_qualification.py \
  --output-dir /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/gpu_v2 \
  --cache-dir /localhome/local-rarunachalam/.tao/cache/cross_model_automl_20260729/dino_ptms \
  --runtime-results-dir /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/gpu_runtime_v2 \
  --cpu-qualification-dir /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/cpu_v2 \
  --cpu-cache-dir /localhome/local-rarunachalam/.tao/cache/cross_model_automl_20260729/dino_ptms \
  --voc-manifest /localhome/local-rarunachalam/tao-automl/experiments/cross_model_automl_20260729/datasets/voc2007/manifest.v1.json \
  --voc-root /localhome/local-rarunachalam/.tao/datasets/cross_model_automl_20260729/voc2007/prepared \
  --container-user "$(id -u):$(id -g)"
```

The completion records exact candidate accounting and exits nonzero when no
checkpoint passes, while preserving all structured exclusions.

## Evidence-driven candidate and final registry

Generate the provisional local-preflight registry with a preregistered version
and evidence path:

```bash
python \
  experiments/cross_model_automl_20260729/dino_ptm_qualification/registry_promotion.py \
  --base-registry src/tao_automl/data/ptm_registry.v1.json \
  --cpu-output-dir /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/cpu_v2 \
  --cpu-cache-dir /localhome/local-rarunachalam/.tao/cache/cross_model_automl_20260729/dino_ptms \
  --gpu-output-dir /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/gpu_v2 \
  --gpu-cache-dir /localhome/local-rarunachalam/.tao/cache/cross_model_automl_20260729/dino_ptms \
  --output-registry /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/candidate_registry.v2.json \
  --audit /localhome/local-rarunachalam/.tao/artifacts/cross_model_automl_20260729/dino_ptm_qualification/candidate_promotion_audit.v2.json \
  --registry-version 1.3.0 \
  --validation-evidence experiments/cross_model_automl_20260729/runtime/dino_ptm_qualification/final_promotion_audit.v1.json
```

Pass that candidate registry to the local launcher with `--registry-path`.
After the local report is complete and SLURM-ready, repeat the generator with
the same registry version and evidence string, new create-only output paths,
and:

```text
--local-report <absolute completed local report>
```

Final mode rejects a different PTM population or different registry-record
hashes, so the candidate exercised locally and the generated repository
registry are byte-equivalent in all execution-relevant records.

## Fixture validation

No network, checkpoint download, PyTorch installation, Docker container, or
GPU is needed for the contract tests:

```bash
cd /localhome/local-rarunachalam/tao-automl
PATH=/localhome/local-rarunachalam/.tao/venvs/dino-multiobjective-py314/bin:$PATH \
  pytest -q \
  experiments/cross_model_automl_20260729/dino_ptm_qualification
```

The tests use a fake tensor serializer, fake NGC HTTPS session, and recording
Docker command runners. They validate both the injected test seam and the
concrete production command/evidence contracts without executing a container.
