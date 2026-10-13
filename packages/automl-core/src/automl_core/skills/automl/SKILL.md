---
name: automl
description: Run bounded numerical, multi-fidelity, population or LLM-guided hyperparameter search, or GEPA prompt optimization, through explicit repository-neutral workload contracts. Use for tuning in a venv, Docker, SLURM, Kubernetes, Brev or Lepton; prefer the repository-native AutoML skill when present.
---

# Repository-neutral AutoML

A container or venv describes execution, not the workload. Collect a runnable
training/evaluation command, nested base config, explicit search domains,
held-out metric and direction, seed, required artifacts, and both time/trial
budgets. Do not guess a metric, model schema, or checkpoint-resume capability.

Use the installed `automl` CLI's declarative manifest, or the repository's
`Workload` adapter. Keep skills out of the inner trial loop. Algorithms are
random, bayesian, bfbo, hyperband, hyperband_es, bohb, asha, dehb, pbt, llm,
hybrid and autoresearch. All execution is sequential, including ASHA.

Resource-allocation modes require cumulative fidelity and an explicit native
checkpoint/resume contract, not weights-only warm starts. A command workload
must bind existing resource/resume/resume-mode fields and a checkpoint artifact
through `stages`; a Python workload implements `plan_stage`. Verify the workload
actually restores progress. Each evaluated stage counts as a trial, and only
successful maximum-fidelity results can win. Inspect the plan's stage counts;
do not shorten the scheduler horizon or silently mutate topology for PBT.

LLM modes require an approved endpoint/model, parameter-data disclosure and
call/token/time caps. Proposals change declared in-domain config only, never
commands or repository code. Hybrid phases cannot expand domains or the reviewed
algorithm allowlist. Autoresearch's prescreen is advice, not measured performance.
Incomplete API receipts stop the run; do not repeat possibly paid calls blindly.
For a failed request, inspect the receipt's `error_kind`, HTTP status, finish
reason and usage before reviewing a new bounded call. Structured modes request
JSON output; GEPA reflection requests text. Do not infer structured compatibility
from a successful text call. Diagnose on the CPU controller before allocating GPUs.

GEPA uses the optional prompts extra and `PromptExperiment` with a consumer's
batch evaluator/manifest loader. Do not invent a generic prompt CLI schema.
Separate reflection/validation/test roles and review disclosure before setting
`allow_reflection_data`. Budget final aggregate reranking and optional winner-only
test as well as GEPA metric calls. Lower-level visual/cost callbacks require the
consumer to own privacy, execution and spend contracts explicitly.

Ask between supported installed execution environments if none was selected.
Venv needs an absolute interpreter; Docker needs an explicit already-local image
and read-only asset mounts. Do not pull images or download assets implicitly.
Commands must run in the foreground; they must not submit untracked remote jobs.
Kubernetes, Brev and Lepton use explicit `execution.<kind>` resource settings
and `volumes` (absolute same-path controller/worker storage, source, relative
sub_path and read_only). Require persistent outputs, cross-client POSIX locks
and disjoint input/output roots. This is not automatic data/checkpoint transfer.
Kubernetes needs a context, namespace, service account, PVCs and CPU/memory/GPU
request; Lepton needs a credential-free HTTPS workspace /api/v1 URL, token env
name, resource shape and dedicated node-group IDs. Both use immutable image
digests and pre-created registry/workload Secret references. Brev uses an
existing ready instance, explicit Docker daemon ID and GPU indices, CPU/memory
limits and a pre-pulled image. Workload env names must exist on that instance;
local secrets are not copied. Do not provision, log in, pull images, create
Secrets or delete instances implicitly. Brev billing continues when its trial
container stops. A Brev remote command is a single quoted `brev exec` argument.
These three executors reconcile uncertain submissions by exact native identity,
never repeated creates. Native retries are zero; core training retries remain
SLURM-only. Missing/conflicting identities block new work. Cancellation requires
native terminal evidence, not just a receipt or accepted cancel RPC. Keep the
CPU controller alive for queue/startup cancellation: Lepton/Brev worker deadlines
do not bound a stuck image pull or host startup. Kubernetes also sets a native
Job deadline. Do not claim cloud GPU validation from CPU transport simulations.
SLURM execution uses native CLIs from a controller on the cluster and shared
storage. Collect partition/account and check live limits; never substitute a
site-specific queue. Containers require a prebuilt, hashed squashfs image and
a sealed consumer template. Do not pull an image inside a GPU allocation.
Review startup, workload, cleanup, queue and total elapsed budgets separately.
Treat `PENDING` as nonterminal and unresolved `UNKNOWN` as an investigation gate. Keep
receipts and native job identity together; local PID checks do not establish
whether a process on another node is alive.
The controller distinguishes `scheduler_unavailable` observations from failed jobs
and retries only read-only status queries within the original deadline. Do not
cancel healthy trials or repeat paid work because a query timed out. Unconfirmed
cancellation requires native reconciliation; a stop marker is not terminal proof.
Lost/invalid `sbatch` replies enter automatic bounded reconciliation by owner and
unique job name. An identified job is adopted, never submitted again. Review
`execution.slurm.submission_reconcile_seconds` (default 120). After healthy
absence observations, a shared POSIX-lock launch fence must prevent the old
attempt from starting before it can use the approved retry caps. A late rejected
allocation still counts against those caps. Conflicts, unavailable evidence or
an unavailable fence block replacement. Do not wrap blocked results or submission
timeouts in blanket cancellation; preserve the ledger and original deadline.

For bounded SLURM infrastructure recovery, review `search.recovery` (or GEPA's
`PromptSettings.recovery`) and `plan.recovery_review`: both per-trial and global
retry caps, extra allocation count, backoff and unchanged elapsed deadline.
Choose heartbeat/progress watchdogs to accommodate loading and reporting gaps;
progress timeouts require command-indexed structured `progress_files` watches.
NODE_FAIL/BOOT_FAIL/PREEMPTED can retry; DRAIN alone, NCCL text, OOM and program
errors cannot. Recovery cancellation must be confirmed before replacement.
Preserve `attempts` and resolve artifacts/promotions using the successful
`attempt_id`. Retries reuse candidates, not new LLM calls or partial checkpoints.
Record hooks need `record_retry_open` with `{retry_of}`. Never retrofit a live
controller or change an old manifest to bypass engine identity checks.

For SLURM CPU-controller wall limits, the shared CLI offers opt-in bounded
handoff via `--controller-max-allocations`, `--controller-allocation-seconds`
and `--controller-grace-seconds` on both plan and run. Review CPU resources as
well as GPU and LLM budgets. One experiment owns one requeue-enabled CPU batch;
use `exec` and `--signal=B:USR1@<grace>` so Python receives the warning. Grace
must cover an in-flight LLM request plus 60 seconds. The saved `controller.json`
binds native generations, build, resources and the original deadline. A
cooperative checkpoint records `SUSPENDED`, requeues the same CPU job and adopts
existing GPU jobs; it never creates new candidates or resets spend limits.
`SUSPENDED` can have active trials, unlike `PAUSED`. Cap exhaustion stops the
search and requests cancellation; unconfirmed cancellation stays an error.
Keep credentials available across handoffs and clean them up only after the
experiment terminates. No unconditional cancel/secret-delete EXIT traps.
Never replay an unconfirmed requeue RPC. Check site requeue permissions and
cross-node filesystem locks separately; CPU fixtures are not live rollover
verification. Do not retrofit an installed controller or existing matrix.

1. Run `automl plan --manifest <yaml> --workspace <directory>`.
2. Review objective/direction, domains, data revisions, training command,
   resource choice, output paths, timeouts, maximum trials and known gaps.
3. Obtain approval before `run ... --approve`. Approval covers only that budget;
   do not retry beyond its reviewed caps, increase resources, widen the search
   or spend beyond it.
4. Use `status` and `logs` for progress. `resume ... --approve` requires the same
   manifest and explicit ID. `UNKNOWN` is an investigation gate, not permission
   to resubmit. Use `cancel ... --approve` to stop actual execution.
5. Report `best.json`, metrics, exact trial config and available artifacts.
   State if every trial failed; never manufacture a best result.

Use `search.stop_on_failure` for fail-fast runs and `pause_after_trials: [1]`
when the first trial requires review. `PAUSED` is an experiment state with no
active trial, not a native job state. An approved resume acknowledges that
boundary without resetting elapsed time or restarting completed trials.
Keep the initial port at `1.0.0.dev0` until an explicitly approved release; local
wheel builds do not increment versions. Preserve exact wheel artifacts/hashes for
paused experiments: `engine_build` detects runtime changes even when package
versions match. New algorithms and GEPA have CPU operational coverage, not a
general GPU or external-LLM compatibility guarantee.

Keep credentials in the session environment and pass only their names in
`execution.environment`. Do not store secrets in configs, commands or logs.
Treat validation-set improvement separately from final held-out test evidence.
