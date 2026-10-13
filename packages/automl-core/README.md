# Repository-neutral AutoML core

`nvidia-automl-core` / `automl_core` is independently buildable. Its controller
does not import TAO, its SDK, a model registry, or any consumer repository.
The existing TAO Bayesian implementation and this controller share the GP/EI
numerical kernel. Algorithm/controller modes use explicit workload contracts;
GEPA is shared through this package. Consumer schemas and checkpoint logic stay
in adapters.
The package is maintained in this repository under `packages/automl-core` and
distributed independently as a wheel. Consumers need the wheel, not this source
checkout. The current delivery is a local wheelhouse; no package index or new
repository is created by this change.

## Install and try it

The 1.0.0.dev0 wheel is **not published by this change**. Build/release it before
resolving a consumer's new dependency from a package index:

```bash
python -m build packages/automl-core
python -m pip install packages/automl-core/dist/nvidia_automl_core-1.0.0.dev0-py3-none-any.whl
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
- Sequential `random`, `bayesian` (GP/EI), `bfbo` (GP/UCB), `hyperband`,
  `hyperband_es` (curve pruning), `bohb` (good/bad KDE), `asha`, `dehb`, `pbt`,
  `llm`, `hybrid`, and `autoresearch`. Bayesian uses three successful random
  observations first. Only autoresearch starts with an explicit base baseline.
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
  writable trial-local HOME; images must support that UID.
- Single-node SLURM execution runs its controller on the cluster using native
  CLIs and shared storage. A candidate gets its own allocation. Configure
  `execution.kind: slurm` and explicit `slurm.partition`, `account`, `gpus`,
  `cpus_per_task`, `time_limit_seconds`, `startup_timeout_seconds` and
  `cleanup_seconds`. A container also needs an absolute cached `.sqsh` image,
  `slurm.image_sha256`, and a consumer `template` sealed in `supporting_files`.
  Optional `bootstrap` and its dependencies are sealed there too. The template
  supports the existing single-node `@@JOB_NAME@@`, `@@NUM_GPUS@@`,
  `@@CPUS_PER_TASK@@`, `@@TIME@@`, `@@LOG_DIR@@`, `@@IMAGE@@`,
  `@@CONTAINER_MOUNTS@@`, `@@COMMAND@@`, `@@SBATCH_EXTRA@@`, `@@ENV_FILE@@`
  and `@@EXTRA_ENV@@` slots. A native CPU workload needs no image/template.
  No site-specific queue, registry pull, SSH tunnel or GPU-job scheduler requeue.
  Automatic infrastructure retries require explicit `search.recovery` caps.
- Kubernetes, Brev and Lepton implement the same four verbs for single-node
  containers. All algorithms and prompt batches use these executors unchanged;
  live cloud/GPU verification is still required. Declare `execution.kind` and a
  matching settings object: `KubernetesSettings`, `BrevSettings` or
  `LeptonSettings` in `automl_core.contracts`. All require explicit `gpus` and
  persistent `volumes` with `path`, `source`, `sub_path` and `read_only`.
  Paths must match on controller and worker and support cross-client POSIX
  locks. No implicit data transfer, provisioning or multi-node scheduling.
  Kubernetes selects a context/namespace/service account, PVCs, CPU/memory,
  optional node labels, GPU resource name and pre-created Secret references.
  Brev selects an existing instance, expected Docker daemon ID, GPU indices
  and CPU/memory limits; it uses `brev exec` with one shell-quoted command.
  Lepton uses an explicit HTTPS workspace `/api/v1` URL and token env name,
  resource shape, dedicated node-group IDs and `node-nfs` mounts via the v1
  Jobs API (wire contract checked against leptonai 0.27.3; no runtime SDK needed).
  Kubernetes/Lepton images require immutable digests and may be pulled using
  pre-created registry Secrets. Brev uses an already-local image and forwards
  environment names from its instance, never values from the CPU controller.
  Kubernetes/Lepton `secrets` map declared `execution.environment` names to
  native Secret references; Kubernetes values are `{name: secret, key: key}`,
  Lepton values are Secret names. The controller's LLM/token credentials stay
  separate. Typed settings reject selectors belonging to another platform.
  Submission intent/spec/scope precede create; uncertain replies enter bounded
  read-only adoption, never another create. Native replacement workers are
  fenced by a shared durable claim. Kubernetes cancellation uses a UID-tested
  suspend patch; Brev stops the immutable owned container ID; Lepton stops the
  owned job ID. Terminal evidence is required, artifacts are retained, and
  no instance/PVC teardown occurs. Brev instance billing continues after stop.
  Worker deadlines preserve runtime and total elapsed budgets. Kubernetes also
  sets a native Job deadline; Lepton/Brev require a live controller for queued
  and pre-worker startup cancellation and do not offer a native allocation cap
  here. Automatic infrastructure training retries remain SLURM-only.
- SLURM queue time counts toward the total elapsed deadline, not the workload
  timeout. Startup + workload runtime + cleanup must fit one allocation. The
  scheduler receives an absolute completion deadline; a standalone supervisor
  enforces startup/runtime bounds even without the controller. Optional
  `record_open` / `record_mark` argv hooks integrate a consumer's existing
  record writer (`{trial}`, `{image}`; and `{id}`, `{state}`, `{backend_ref}`,
  `{message}` for mark). Hooks must not include credentials. Job identity is
  reserved before submit, and ambiguous replies are reconciled by unique name
  and owner. Completion requires matching native/worker/supervisor evidence.
  Catchable controller termination requests cancellation; a hard crash leaves
  jobs bounded by native deadlines. Cluster/GPU end-to-end validation remains
  a separate release gate.
- The controller falls back from unavailable `squeue` queries to `sacct`. If neither
  can establish status, `scheduler_unavailable` is a transient observation,
  not a failed trial. Polls back off from `poll_seconds` to at most 300 seconds
  within the original deadline; only read-only queries repeat. Status reports
  `observed_state: UNKNOWN` and safe query metadata, never cached state as a
  fresh observation. Cancellation interrupts the wait. If cancellation/deadline
  occurs while status is unavailable, durable stop markers and native deadlines
  bound workers; the controller returns `ERROR` requiring reconciliation, not
  false terminal success. Missing/conflicting identities and invalid completion
  receipts still stop at the ambiguity gate. No resubmission or paid API retry.
- `search.stop_on_failure: true` stops on the first unsuccessful logical trial
  after approved infrastructure attempts are exhausted or inapplicable.
  `pause_after_trials: [1]` returns `PAUSED` after one completed trial for
  review. Approved resume acknowledges that boundary without changing the
  identity, budget, history, or elapsed start time. No active trial remains
  while paused; it is an experiment state, not a new native job status.
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

### Bounded CPU-controller continuation

The CPU controller's wall time is independent of GPU trial wall time and the
overall experiment deadline. Opt in on both `plan` and `run` using
`--controller-max-allocations N --controller-allocation-seconds S
--controller-grace-seconds G`. The initial allocation counts toward N. Planning
reports `controller_review`; it does not submit or requeue anything. This is
available to consumer CLIs using the shared `main`, including prompt workflows.

Run one experiment per CPU batch, enable `--requeue`, request a
`--signal=B:USR1@G` warning and `exec` the Python CLI as the batch process. Use
the same sealed manifest, shared interpreter, workspace and available credentials
on each allocation. Grace must fit an in-flight LLM request plus 60 seconds.
At a safe point, the brain persists `SUSPENDED` and requests requeue of the same
CPU job ID. Its verified successor resumes recorded trials/batches and LLM
receipts. No new GPU trial or paid proposal is created by handoff.

`controller.json` binds the original absolute deadline, build, resources, policy
and native restart generations. Queuing/downtime consume time; nothing extends
the experiment. Cap exhaustion stops the brain and requests native cancellation.
Requeue is disabled on the last allowed CPU allocation, and scheduler restarts
also count. Administrative/site-forced extra generations are refused, not
permission for new workload. A lost RPC acknowledgement is
`REQUEUE_UNCONFIRMED`; only an actual native successor can automatically resume.
Do not repeat the RPC or submit another job blindly. `SUSPENDED` may retain
active trials, unlike the manual `PAUSED` gate. Do not unconditionally cancel
trials or delete the credential sidecar on every batch exit.

SIGTERM/SIGHUP still request cancellation and persist observed termination;
unconfirmed native cancellation is an error, not a false terminal claim. Hard
process/node loss still relies on durable intents and independent worker
deadlines. CPU operational fixtures cover handoff, native identity adoption,
GEPA replay and receipt reuse, not live site requeue or cross-node lock behavior.
Existing installed controllers and ledgers require their original builds.

### Bounded infrastructure recovery

`search.recovery` accepts `max_retries_per_trial` and `max_retries_total` (both
default zero), `backoff_seconds` (30, doubles to at most 300),
`cancellation_timeout_seconds` (120), and optional `heartbeat_timeout_seconds`
and `progress_timeout_seconds`. The original elapsed deadline never resets;
each replacement must fit a full allocation. `plan.recovery_review` reports the
worst-case allocation count. Review these additional attempts before approving.

SLURM terminal NODE_FAIL/BOOT_FAIL/PREEMPTED may retry. A watchdog can also
identify a stalled owned allocation on a confirmed DOWN/NOT_RESPONDING node,
but must cancel and confirm terminal scheduler state before replacing it.
DRAIN alone, NCCL text, program/config failures, OOM and timeouts are not inferred
to be transient. Scheduler observation outages and ambiguous submissions never
authorize new work. Unconfirmed recovery cancellation produces an explicit
blocked-recovery error, not an overlapping replacement or indefinite wait.
Automatic infrastructure classification currently supports SLURM only.

### Lost submission replies

`sbatch` timeout, transport/exit errors and invalid acknowledgements are uncertain
submissions, not permission to repeat the RPC. The controller preserves the
candidate and intent and reconciles `squeue`/`sacct` by owner and unique job name.
`execution.slurm.submission_reconcile_seconds` defaults to 120 (1–900); the
window and in-flight bounded scheduler calls stay subject to the original
experiment deadline. An existing job is adopted without another trial or LLM
call. The window is persisted across controller resume, not restarted.

If both queries repeatedly succeed without a match through the window, the
controller atomically closes the attempt's launch fence before classifying it
as retryable. The standalone batch supervisor must claim that same fence before
touching shared execution/worker files or launching a workload. A batch that
already claimed it cannot be fenced or replaced. A closed fence rejects even a
late scheduler acceptance without running the workload or overwriting receipts.
This requires shared POSIX `flock` semantics. Missing/invalid/busy fences, worker
evidence, multiple job IDs or unavailable observations never authorize retry.

Replacement uses the existing per-trial/total retry caps, new record/output paths,
the same candidate and original deadline. An uncertain original submission is
conservatively charged as an attempt even when no job is observed; a late rejected
allocation is not free compute. Retry caps still default to zero. Fencing is not
a claim that the scheduler proved non-submission or terminated a hidden job.
Its possibly queued allocation remains bounded by the original scheduler deadline.
`_submission_reconciliation.json` records safe evidence and decisions. Unresolved
cases return experiment `ERROR` with a blocked reason and preserve the pending
trial; wrappers must not blanket-cancel that result. Old ledgers/staged helpers
are not upgraded. Rebuild the exact local controller wheel without a version bump;
the supervisor/fence are staged from it, not installed into the training image.

Workers publish independent heartbeat and progress timestamps. `TrialPlan` and
`CommandWorkload.progress_files` map command indices to attempt-relative structured
status paths. Every command needs declared watches to enable a progress timeout.
Command transitions reset its progress clock; ordinary logs and worker heartbeats
do not count as progress. Configure thresholds to cover legitimate loading,
export and reporting gaps. Status FAILURE/ERROR is a non-retriable failure signal,
not a regex-based infrastructure diagnosis. Health receipt job IDs must match.

The logical trial ID, config, resource, checkpoint source and LLM proposal remain
unchanged. `attempt_id` names its current sibling execution directory;
`attempts` retains previous failed executions. Retry attempts do not consume
new search slots or LLM calls. The workload must resolve a promotion parent's
successful directory with `recovery.attempt_root(experiment_root, parent)`;
never assume `parent['id']` still names its successful execution. Partial failed
checkpoints are not automatically resumed. Consumer record hooks additionally
provide `execution.slurm.record_retry_open`, an open argv template containing
`{retry_of}` for the previous record ID. Each replacement gets fresh native and
consumer records and fresh writable paths.

`PromptSettings.recovery` applies to GEPA evaluation batches through this same
controller. Its total retry cap spans batches; attempted examples, including
retries, consume `max_evaluated_examples` conservatively. Reflection calls are
never automatically repeated. Existing ledgers/environments are not migrated;
the unchanged `1.0.0.dev0` version still requires exact build selection.

## Resource and proposal contracts

Multi-fidelity modes require `search.fidelity` (positive cumulative `minimum`,
`maximum`, `reduction_factor`) and a workload `plan_stage(config, trial,
execution, stage, parent)`. A generic `CommandWorkload` can declare `stages`
with `resource_path`, `resume_path`, `resume_mode_path` and
`checkpoint_artifact`. These are existing nested-config field addresses, not
new command flags. The checkpoint must be a declared artifact. The workload
must actually implement full-state resume and prove its native semantics; the
generic engine cannot turn model weights into a resumable checkpoint.

Each evaluated stage consumes `max_trials`, including promotions and population
members. Only successful full-maximum stages can win. Parent checkpoints mount
read-only; trial output is separate. ASHA does not imply concurrent execution.
Hyperband variants traverse one set of brackets; PBT traverses population
intervals. `plan` reports allocation details; all schedules are also bounded by
trial/time caps. Failed stages cannot be donors. BOHB and DEHB require adequate
same-resource observations and record explicit random warmup/fallback samplers.
`search.options` exposes their numerical and population settings.

LLM modes require explicit `search.llm` endpoint/model, `api_key_env` name,
`max_calls`, `max_output_tokens`, prompt-size and timeout limits. HTTPS is
required except on loopback. Requests use `/chat/completions`; no provider SDK,
redirects or automatic retries. Journaled requests disclose only parameter
domains/values, scalar results, statuses and research instructions. Responses
can only propose in-domain parameter values, never code/commands. Ambiguous
requests stop rather than repeat spend. `hybrid` restricts numerical phases to
`phase_algorithms`/`phase_trials`, narrows domains, and optionally consumes an
explicit `research_program` (named algorithm/trials/domains/carry_forward phases).
`autoresearch` uses baseline, proposal, validation, prescreen and measured
keep/discard; it does not edit repository code.

Structured requests use `response_format: {type: json_object}`. Parsing accepts
one complete JSON object, bare or enclosed in a single `json`/unlabelled Markdown
code fence. It does not extract arbitrary objects from prose, repair incomplete
JSON, accept duplicate keys/nonfinite constants, or relax domain validation.
GEPA reflection keeps its text-output contract and does not request JSON mode.
Endpoints must support the selected request format; there is no automatic
format fallback or retry. Error receipts distinguish transport/HTTP, envelope,
content parsing, truncation, tool-call and refusal failures. They retain only
allowlisted diagnostics (HTTP status, finish reason, available token usage and
content length/format), never raw error bodies or underlying exception strings.
Validate a bounded controller-only request before allocating training GPUs.
Keep 0.2.0 wheels/receipts for old experiments; 0.2.1 changes engine/request
identity and does not resume ambiguous paid requests from the old client.

`nvidia-automl-core[prompts]` pins GEPA 0.1.2. `PromptExperiment` composes the
same executor with a batch workload implementing `identity`,
`validate_candidate`, `plan_batch`, `collect_batch`, `per_item`, `aggregate`
and `metric_key`. It needs explicit reflection/validation/optional-test data,
candidate choices, LLM disclosure approval, request/token limits, metric-call,
total-example, batch-count, candidate-count and time limits. A native consumer
supplies the CLI manifest loader; generic `automl` remains a command-workload
CLI, not a guessed prompt schema. Results are in `prompt.json` / `best.json`.
Cached batches/requests are reused; upstream GEPA state and joint-config history
are durable. Validation aggregate reranking precedes winner-only test scoring.

The lower-level `automl_core.prompts` API also preserves callback batch runners,
visual reflection hooks, multi-objective feedback and candidate-cost penalties.
The caller owns privacy, execution and budgets when bypassing `PromptExperiment`.
The existing `tao_automl.gepa_autoprompter` names re-export this implementation.

These are capability ports, not binary compatibility with old TAO brain state.
There is no standalone grid or TPE selector; BOHB supplies density proposals.
Parallel search, multi-node trials, arbitrary-code research agents and automatic
workload discovery are not provided. Keep the original wheel for existing 0.1.x
experiments: engine version and runtime-build fingerprint are part of resume identity.

## Development version policy

Keep the initial v1 port at **`1.0.0.dev0`** until an explicitly approved release.
Do not increment package versions or development counters for bug fixes or local
validation wheel builds. The earlier 0.1.x/0.2.x artifacts were experimental
builds, not releases; preserve their original labels and hashes.

Save each build in an immutable, separately named wheelhouse and record the wheel
SHA256 plus source revision. Installing a changed same-version wheel over an
existing installation requires an explicit `--force-reinstall --no-deps` after
resolving dependencies; the version constraint alone cannot select a build.
Do not replace the installed engine of a paused experiment.

SFT and GEPA experiment identities include `engine_build`, the SHA256 fingerprint
of relative Python file names and content hashes under `automl_core`. It is the
same in source and wheel installs, excludes bytecode/docs/installation paths,
and prevents a changed runtime from resuming a same-version ledger. It is not
an authenticity signature or dependency inventory. Existing environment and
consumer provenance checks still apply. Restore exact old wheels for old runs;
no automatic ledger migration is provided.

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
Separate operational CPU runs exercise all twelve modes, checkpoint promotion,
mock LLM calls and actual GEPA with toy inference. No new unit tests were added
for the 0.2.0 port; GPU continuation/PBT and real reflection-service integration
remain verification gaps, not implied by numerical smoke results.
