# PTM-aware runner contract

`AutoMLRunner.run` accepts a live `ResolvedPTMRuntimeInventory` only after the
caller has completed repository-owned PTM discovery, credential/access
preflight, artifact checksum verification, checkpoint-spec validation, and
model-load smoke testing. The runner never downloads checkpoints, restores a
JSON preflight report into executable state, or infers a PTM inventory.

Enable the path with:

```python
runner.run(
    ...,
    ptm_aware_runtime=True,
    resolved_ptm_inventory=resolved_inventory,
)
```

Passing a live inventory without the flag also enables the path. Explicitly
setting `ptm_aware_runtime=False` while passing an inventory is rejected.
Omitting both preserves legacy direct `BrainFactory` execution and does not
pass a new constructor keyword to `AutoML`.

Before a baseline callback or platform job can run, the runner verifies:

- the value is the live typed inventory, not serialized JSON;
- its content hash and objective-configuration hash;
- exact model, objective mode, and algorithm agreement;
- the complete prepared PTM inventory for latency and multi-objective modes;
- that every runner `spec_overrides` value was already frozen into every PTM
  arm;
- that runner-injected dataset paths were already frozen into every PTM arm;
- that no second `base_checkpoint`, checkpoint target, or checkpoint spec
  override competes with the registry-resolved identity.

This prevents the hierarchical PTM arm specification from silently replacing
dataset or user inputs applied after inventory resolution.

## Mode policy

- Accuracy mode uses the registered default PTM or explicit registered user
  PTM by default. Searching all prepared PTMs requires an explicit `all`
  accuracy policy.
- Latency and multi-objective product profiles must enable PTM-aware runtime
  and use the complete prepared inventory. Neither mode may narrow the
  inventory to a checkpoint selected by a caller or agent.
- The inventory is passed unchanged to `AutoML`; objective and selector
  semantics remain owned by the controller.

## Provenance and resume

The controller exposes an unhashed runtime manifest and its canonical SHA-256.
Before any baseline executes, the runner verifies the hash, rejects
secret-bearing fields or signed URLs, and records this envelope at:

```text
<workspace>/ptm_runtime_manifest.json
```

The returned result exposes the same payload, hash, and record path under
`ptm_runtime`. `query_status(workspace)` also verifies and exposes the
persisted envelope when it is present; a malformed record or mismatched hash
fails closed. Resume requires the existing record to match exactly. A fresh
run never overwrites a conflicting record. JSON reports and manifests remain
audit evidence only; they cannot reconstitute a live preflight inventory.
