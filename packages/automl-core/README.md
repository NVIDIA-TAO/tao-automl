# Repository-neutral AutoML core (initial implementation)

`nvidia-automl-core` / `automl_core` is independently buildable. Its controller
does not import TAO, its SDK, a model registry, or any consumer repository.
The existing TAO Bayesian implementation and this controller share the GP/EI
numerical kernel; model-specific parameter and checkpoint logic stays in TAO.
The package is maintained in this repository under `packages/automl-core` and
distributed independently as a wheel. Consumers need the wheel, not this source
checkout. The current delivery is a local wheelhouse; no package index or new
repository is created by this change.

## Install and try it

The 0.1.0 wheel is **not published by this change**. Build/release it before
resolving a consumer's new dependency from a package index:

```bash
python -m build packages/automl-core
python -m pip install packages/automl-core/dist/nvidia_automl_core-0.1.0-py3-none-any.whl
# Existing TAO source development: install this wheel before pip install -e .
```

A consumer only needs the wheel, never this checkout. For a small real CPU
example, copy `examples/random_forest.yaml`, set `execution.python` to an
absolute interpreter with scikit-learn installed, and run from this directory:

```bash
automl plan --manifest examples/random_forest.yaml --workspace /tmp/automl-demo
automl run --manifest examples/random_forest.yaml --workspace /tmp/automl-demo --approve
automl status --manifest examples/random_forest.yaml --workspace /tmp/automl-demo
```

`plan` does not write artifacts or execute workload commands. `run`, `resume`,
and `cancel` require `--approve`. `status`, `logs` and `cancel` also accept
`--experiment-id <id>` instead of `--manifest`, so mutable source files cannot
block inspecting/stopping recorded execution. `logs` returns per-trial log paths. Python
callers use `Experiment(Search(...), workload, Execution(...), workspace)`;
calling `run()` is their explicit launch boundary. `Workload` is a protocol,
not a registry keyed by vendor/model names. `CommandWorkload` provides nested
JSON/YAML/TOML configuration plus argv templates (`{python}`, `{config}`,
`{trial}`), metrics JSON and required artifacts. It never adds a shell.

## Contract and limits

- Explicit metric **and direction**, seed, domains, per-trial timeout, total
  elapsed timeout, and trial count. Failed/invalid trials consume the count.
  Total elapsed time starts at experiment creation and includes downtime.
- Sequential random or Bayesian search. Bayesian search uses three successful
  random observations to initialize GP/EI. No implicit baseline trial. All
  trials start fresh; no scheduler promotion or checkpoint inheritance.
- Numeric domains can be linear/log-scaled; categorical domains are scalar
  choices. Discrete domains may revisit a configuration; budgets still apply.
- Dot paths are only parameter addresses. On-disk workload configs are nested.
  Exit zero is insufficient: the requested finite numeric metric and all
  declared trial-local artifacts must exist. Failed metrics never enter GP.
- Linux local venv and local Docker execution implement submit/status/logs/cancel.
  The venv interpreter is explicit; the controller can use a different Python.
  Commands must stay in the foreground, not daemonize or submit remote jobs.
  Docker uses an explicit **already-local** image, `--pull=never`, read-only
  input mounts, a writable trial mount, and an ownership-labelled container.
  Host/container paths are identical. The container uses the host UID/GID and a
  writable trial-local HOME; images must support that UID. No scheduler or managed-platform claim.
- Credentials are environment **names**, never values in manifests, argv or
  state. Do not put secrets in arbitrary config/commands. Program logs are not
  automatically redacted; workloads remain responsible for safe logging.
  A venv is not a security boundary; run trusted workload commands only.
- One controller lock, atomic fsynced ledger, intent before launch, and durable
  worker receipts. Resume requires the same explicit experiment ID, manifest,
  adapter version, recipe/evaluation content, and environment inventory/image
  ID. Supply immutable data/code revisions in `search.provenance`; the engine
  does not hash datasets or guarantee reproducibility of mutable external files.
- Workers enforce timeout even if the controller dies. Resume reconciles native
  process/container state and collects completed results without relaunching.
  `UNKNOWN` stops the controller: inspect the receipt, logs, PID/start token or
  labelled Docker container. Do not erase the ledger to force a retry. If a
  launch cannot be reconciled, cancel/clean up its native job, then use a new ID.
- `cancel` writes an experiment stop request and terminates native execution;
  it is not just a record update. A canceled experiment cannot launch more
  trials. Containers, logs, checkpoints and state are retained for inspection;
  there is no automatic destructive cleanup.

`experiment.json` is authoritative; `best.json` is a rebuildable projection.
Its best trial contains the exact sampled config and validated artifacts.
The best observed validation score is not a claim about final test performance.

ASHA/Hyperband, PBT, multi-fidelity continuation, parallel/distributed search,
LLM/GEPA suggestions and spend budgets remain in the existing TAO consumer.
They are **not** exposed through fake generic capabilities. Adding them here
requires corresponding workload/executor contracts and lifecycle tests first.

## Packaged skill and validation

The generic skill ships at `automl_core/skills/automl/SKILL.md` in the wheel;
agents can locate it with `importlib.resources.files('automl_core')`. It does
not depend on TAO skills. A repository may instead provide its native adapter
skill, as Cosmos Framework does; do not install competing entrypoints there.

```bash
PYTHONPATH=packages/automl-core/src python -m pytest packages/automl-core/tests
PYTHONPATH=src:packages/automl-core/src python -m pytest tests/test_bayesian_failure_handling.py
python -m ruff check packages/automl-core
```

Tests use real, tiny local CPU processes plus mocked Docker lifecycle checks.
They do not establish GPU training, image compatibility, or cluster support.
