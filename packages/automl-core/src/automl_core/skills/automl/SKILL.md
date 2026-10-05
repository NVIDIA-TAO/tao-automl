---
name: automl
description: Run bounded hyperparameter search for a repository with an explicit command/config/metric contract using the repository-neutral AutoML engine. Use for tuning in a local Python environment or container; prefer a repository-native AutoML skill when one exists.
---

# Repository-neutral AutoML

A container or venv describes execution, not the workload. Collect a runnable
training/evaluation command, nested base config, explicit search domains,
held-out metric and direction, seed, required artifacts, and both time/trial
budgets. Do not guess a metric, model schema, or checkpoint-resume capability.

Use the installed `automl` CLI's declarative manifest, or the repository's
`Workload` adapter. Keep skills out of the inner trial loop. Current algorithms
are random and fixed-budget Bayesian search; there is no LLM spend, fidelity
promotion, parallel search or population checkpoint reuse in this contract.

Ask between supported installed execution environments if none was selected.
Venv needs an absolute interpreter; Docker needs an explicit already-local image
and read-only asset mounts. Do not pull images or download assets implicitly.
Commands must run in the foreground; they must not submit untracked remote jobs.

1. Run `automl plan --manifest <yaml> --workspace <directory>`.
2. Review objective/direction, domains, data revisions, training command,
   resource choice, output paths, timeouts, maximum trials and known gaps.
3. Obtain approval before `run ... --approve`. Approval covers only that budget;
   do not retry, increase resources, widen the search or spend beyond it.
4. Use `status` and `logs` for progress. `resume ... --approve` requires the same
   manifest and explicit ID. `UNKNOWN` is an investigation gate, not permission
   to resubmit. Use `cancel ... --approve` to stop actual execution.
5. Report `best.json`, metrics, exact trial config and available artifacts.
   State if every trial failed; never manufacture a best result.

Keep credentials in the session environment and pass only their names in
`execution.environment`. Do not store secrets in configs, commands or logs.
Treat validation-set improvement separately from final held-out test evidence.
