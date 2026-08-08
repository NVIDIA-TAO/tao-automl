# Immutable AutoML campaign manifest

`tao_automl.campaign_manifest` freezes experiment intent before any SLURM
submission. It is a validation and provenance boundary only: it does not
submit a job, inspect a scheduler, resolve an image, or download a dataset or
checkpoint.

## Required identities

Every sealed campaign records:

- the full Git object ID;
- whether a dirty tree is prohibited or explicitly represented by a SHA-256
  of the complete canonical diff;
- the built wheel version and SHA-256;
- the exact Enroot SQSH URI, SQSH SHA-256, and source-container digest;
- the TAO version, precision, per-GPU train/evaluation/latency batch sizes,
  latency-protocol digest, fixed latency-input digest, and timed scope;
- every allowed PTM ID, checkpoint digest, registry-record digest, and
  completed preflight-report digest;
- dataset identity, dataset-manifest digest, conversion digest, and individual
  train/validation split digests;
- algorithm and acquisition implementation versions;
- the complete parameter domain and its canonical hash;
- objective configuration and its canonical hash for each mode;
- candidate, concurrency, wall-clock, terminal-failure, fidelity, resource,
  job-count, and storage budgets.

Signed download URLs, credential-bearing URLs, mutable tags such as
`latest`, and abbreviated Git commits are not suitable manifest identities.
The SQSH URI must identify an exact `.sqsh` file.

For a dirty-tree campaign, `dirty_tree_policy` must be
`allow_with_diff_hash`, `dirty` must be true, and `diff_sha256` must identify
the canonical complete diff used to build the wheel. A clean campaign records
`diff_sha256: null`. The caller is responsible for capturing staged,
unstaged, and untracked source content in that canonical diff; the manifest
does not run Git commands.

## Three independent mode jobs

A valid campaign contains exactly:

1. `accuracy`;
2. `latency`;
3. `multi_objective`.

The modes share campaign-scoped dataset, algorithm, search space, candidate
budget, fidelity, PTM inventory, wheel, image, and one-node/eight-GPU SLURM
resource and runtime contract. They use the same preregistered seed for a
controlled mode comparison.

The SLURM launcher contract is one scheduler task per node. That task runs the
container and TAO/`torchrun` creates eight local distributed workers, one per
GPU. The manifest therefore records:

```text
nodes: 1
gpus_per_node: 8
tasks_per_node: 1
distributed_workers_per_node: 8
```

Each mode has a unique job ID and observation namespace. Cross-mode
observation sharing is false and every mode starts with an empty observation
list. Consequently, the three jobs are independent searches rather than
three selectors over one shared archive.

This also defines the validation boundary.  Accuracy, latency, and
multi-objective invariants are evaluated against the terminal archive of the
mode that made the selection.  Because objective-aware acquisitions may
discover different finite candidate universes, ordering the three active-mode
winners by accuracy or latency is observational evidence, not a production
selector invariant.  A candidate discovered by another mode does not
retroactively make the accuracy selector incorrect.  A read-only union may be
used to diagnose acquisition coverage, but it must never become a production
selection archive or trigger reselection.

PTM identity is not represented as an ordinal or categorical value in the
inner numerical search space. `ptm_search` records deterministic
`hierarchical_nonordinal_arms`; each arm binds a checkpoint ID to its
conditional-search-space, preflight-provenance, and input-contract hashes.

Latency and multi-objective jobs use `ptm_policy: all_qualified`. Accuracy
mode follows its product policy explicitly:

- `registered_default`: exactly the registered default PTM;
- `user_provided`: exactly one qualified user-provided PTM;
- `all_qualified_explicit`: the complete inventory, only when multi-PTM
  accuracy search was explicitly enabled.

The first two are documented fairness exceptions rather than silent PTM
inventory drift. All other fairness inputs remain shared.

Latency mode records an internally calibrated quality constraint:

```text
type: relative_retention
reference: best_observed_within_job
reference_updates: monotonic
terminal_reference: terminal_archive_accuracy_winner
```

Multi-objective mode cannot inherit that constraint. Its constraint is either
null or an independently declared absolute minimum whose source is
`multi_objective_explicit`.

## Staged execution gates

The contract fixes these ordered stages:

1. `single_candidate_gate`: exactly one candidate per mode;
2. `pilot_batch`: a balanced small batch across all modes;
3. `full_search`: only after pilot artifacts and metric sanity pass;
4. `matched_validation`: only after search archives and selections are frozen.

Candidate-stage job counts must equal three times the per-mode candidate
budget. The workload total must equal the sum of all stage counts. Every gate
uses `halt_before_next_stage` on failure.

The cancellation contract includes artifact-integrity, preflight, metric
sanity, failure-budget, storage-budget, and wall-clock triggers. Cancellation
preserves durable records.

## Failure and intervention policy

Retries are finite and capped by the product bound. A terminal failed
recommendation remains a failed record, counts toward the candidate budget,
and cannot be silently replaced.

All agent-intervention fields and all selection-isolation fields are required
and must remain false. Omitting a field is also invalid. Validation-only
matched measurements therefore cannot replace selection-time objectives or
trigger selection or reselection.

## Sealing and resume

Create and seal a new document with:

```python
from tao_automl.campaign_manifest import create_campaign_manifest

campaign = create_campaign_manifest(configuration)
sealed_document = campaign.to_dict()
```

`manifest_sha256` is calculated over canonical finite JSON after normalizing
semantically unordered PTM inventories, mode records, and gate criteria.
Callers cannot supply a hash while creating a new campaign.

Load persisted intent with:

```python
from tao_automl.campaign_manifest import load_campaign_manifest

persisted = load_campaign_manifest(sealed_document)
campaign.assert_resume_compatible(persisted)
```

Any source, wheel, container, PTM, dataset, algorithm, seed, search-space,
objective, budget, fidelity, resource, retry, gate, cancellation, or audit
change produces a different identity and is rejected before resume.

Each independent mode has a derived sealed binding:

```python
latency_manifest = campaign.mode_manifest("latency")
campaign.assert_mode_resume_compatible("latency", latency_manifest)
```

The mode hash binds the mode-specific objective and namespace to every shared
campaign input and to the parent campaign hash.
