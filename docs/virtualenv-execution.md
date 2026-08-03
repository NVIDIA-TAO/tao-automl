# Virtual Environment Execution

AutoML actions can run a Python script directly from a virtual environment.
This is an alternative to building an in-container command with
`tao_sdk.script_runner.build_entrypoint`.

Install the direct-execution dependencies with
`pip install "nvidia-tao-automl[virtualenv]"`. The extra includes the SDK's
TOML writer so JSON, YAML, and TOML support does not depend on ambient
packages.

This contract is consumed directly by `AutoMLRunner` and can live in a local
external model directory. Existing packaged TAO model actions remain
container-backed until their skill metadata explicitly adopts
`execution.type: python_script`; the current TAO skill-bank validator and
platform inventory are not changed by this SDK feature.

## Skill Contract

Declare `execution.type: python_script` on the action. Relative script and
working-directory paths are resolved from the model skill directory.

```yaml
network_arch: public_random_forest
actions:
  train:
    config_format: json
    execution:
      type: python_script
      script: scripts/train.py
      args: [--config, "{config_path}"]
      cwd: .
    inputs:
      dataset.path:
        type: file
    outputs:
      results_dir:
        type: folder
```

The script arguments are an argv list, not a shell command. Supported
placeholders are `{config_path}`, `{results_dir}`, and `{job_id}`. A job-local
standard-library supervisor launches the training argv as
`<venv>/bin/python <script> ...` without activating the environment or
invoking a shell, and records its exit status for restart recovery.

The skill must also provide `schemas/train.schema.json`. AutoML uses that
schema as the search-space source, so external models can define their own
parameter names without a `tao_automl.config.<network>` Python package.
The root must have a non-empty `properties` object; `default`, when present,
must be an object. Searchable leaves use TAO AutoML schema types such as
`integer`, `number`, `boolean`, `categorical`, or `ordered`. Numeric ranges use
`minimum` and `maximum`, choices use `enum`, and `automl_enabled: true` enables
a parameter when the caller does not provide an explicit parameter list.

## Metric Contract

The built-in extractor recognizes log lines such as `accuracy: 0.973` or
`loss=0.12`, using the metric name from `automl_settings`. A script can instead
write structured output below a declared result directory. Supported files
include `metrics.json` with a metric-name key, TAO JSONL `status.json` with a
`kpi` object, and `best_score.json`. For a different format, pass
`metric_extractor(log_text, metric_name)` to `AutoMLRunner.run`.

Terminal local logs are scanned in bounded chunks, so an early metric remains
discoverable without loading the complete training log into controller memory.

## Runtime

```python
from tao_automl.runner import AutoMLRunner
from tao_sdk.platforms.virtualenv import VirtualEnvSDK

sdk = VirtualEnvSDK(
    venv_path="/work/venvs/model",
    work_dir="/work/automl-jobs",
)
runner = AutoMLRunner(
    sdk=sdk,
    skill_dir="/work/model-skill",
    action="train",
    poll_interval=1,
)
result = runner.run(
    automl_settings={
        "algorithm": "bayesian",
        "metric": "accuracy",
        "direction": "maximize",
        "automl_max_recommendations": 4,
        "run_baseline": False,
        "run_final_evaluation": False,
    },
    workspace_path="/work/automl-state",
    gpu_count=0,
)
```

Do not pass `image` for Python-script execution. The selected virtual
environment is the runtime.

Each recommendation gets an isolated job directory containing its serialized
config, combined stdout/stderr log, and results directory. Declared output
spec keys are rewritten into that results directory. Declared inputs must be
local paths in this initial implementation; remote URIs are rejected before
the process starts rather than being silently ignored.

Recommendation values are normalized to finite Python scalar, mapping, and
sequence values before persistence and submission. This preserves numeric and
boolean types across live execution and resume, including values produced by
NumPy-based search algorithms. TOML actions reject null values because TOML
has no null representation.

For GPU scripts, pass explicit `gpu_ids` when device ownership matters. The
SDK sets `CUDA_VISIBLE_DEVICES` from those IDs. A count without IDs does not
reserve devices and preserves the process's existing visibility.

## Plan Files

The command-line runner accepts `--platform virtualenv`. Constructor settings
belong in `params.sdk_kwargs`, while per-job settings remain in
`params.platform_kwargs`.

```json
{
  "ready": true,
  "steps": [{
    "params": {
      "skill_dir": "/work/model-skill",
      "train_dataset_uri": "",
      "sdk_kwargs": {
        "venv_path": "/work/venvs/model",
        "work_dir": "/work/automl-jobs"
      },
      "platform_kwargs": {"gpu_count": 0}
    }
  }],
  "automl_settings": {
    "algorithm": "bayesian",
    "metric": "accuracy",
    "direction": "maximize",
    "automl_max_recommendations": 4
  }
}
```
