# Sensitivity-latency v1 failure audit

The v1 batch produced no latency measurements. All nine allocations stopped
during hardware preflight because the manifest expected the exact string
`2.11.0`, while the pinned SQSH reported
`2.11.0a0+a6c236b9fd.nv26.03.46836102`.

This is deliberately classified as
`preflight_failed_no_latency_measurements`, not as a completed benchmark.
Although every SLURM root row appears as `COMPLETED/0:0`, every `.0`
interactive `srun` step is `FAILED/1:0` and every job-scoped
`allocation_result.json` has terminal status `preflight_failure`.

## Per-allocation evidence

| Allocation | TAO job ID | SLURM job | Node | Root | `srun` step | Result | Profile runs | Rank JSON |
| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: |
| `seed_001234_allocation_0_row_00` | `ee899aa0-1b57-4545-9f78-9030e6ad872a` | 30943464 | `batch-block7-00339` | `COMPLETED/0:0` | `FAILED/1:0` | `preflight_failure` | 0 | 0 |
| `seed_001234_allocation_1_row_03` | `69d3bd6c-a8b7-46b3-9d05-3c414537ec50` | 30943481 | `batch-block7-03401` | `COMPLETED/0:0` | `FAILED/1:0` | `preflight_failure` | 0 | 0 |
| `seed_001234_allocation_2_row_06` | `335fc12b-4291-4a24-99db-eaa6fb75a649` | 30943494 | `batch-block7-02119` | `COMPLETED/0:0` | `FAILED/1:0` | `preflight_failure` | 0 | 0 |
| `seed_271828_allocation_0_row_01` | `36a9f7f8-2291-4a52-bdbb-1988c6f76a61` | 30943509 | `batch-block7-03295` | `COMPLETED/0:0` | `FAILED/1:0` | `preflight_failure` | 0 | 0 |
| `seed_271828_allocation_1_row_04` | `ccf234ae-c6de-4bde-ba5a-ab8929ff3c6d` | 30943521 | `batch-block7-00339` | `COMPLETED/0:0` | `FAILED/1:0` | `preflight_failure` | 0 | 0 |
| `seed_271828_allocation_2_row_07` | `838246c2-a232-44a0-a90b-6f16760021dd` | 30943525 | `batch-block7-03401` | `COMPLETED/0:0` | `FAILED/1:0` | `preflight_failure` | 0 | 0 |
| `seed_314159_allocation_0_row_02` | `37d1ca0e-1163-4b98-9bac-ee756cee016b` | 30943542 | `batch-block7-02119` | `COMPLETED/0:0` | `FAILED/1:0` | `preflight_failure` | 0 | 0 |
| `seed_314159_allocation_1_row_05` | `6fc686ae-b722-4d9d-b2ba-dd83e5b45a78` | 30943558 | `batch-block7-03295` | `COMPLETED/0:0` | `FAILED/1:0` | `preflight_failure` | 0 | 0 |
| `seed_314159_allocation_2_row_08` | `44343f8b-dbc6-4d5b-a300-700cb66ef0ce` | 30943574 | `batch-block7-00670` | `COMPLETED/0:0` | `FAILED/1:0` | `preflight_failure` | 0 | 0 |

For every allocation, the `profiles` directory is absent, `profile_runs` is
empty, recursive `rank_*.json` count is zero, `main.out` is empty, and the
logs contain neither a `torchrun` invocation nor a benchmark-script
invocation. The result was written before `validate_hardware`; the benchmark
subprocess is only reachable after that validation succeeds.

The submission ledger, manifest, and every allocation result set
`feeds_final_selection=false`. The manifest and results also set
`manual_promotion_permitted=false`. The v2 manifest records the v1
disposition as `preflight_failed_no_latency_measurements`. Therefore, no v1
timing data exists to reuse, and this batch is ineligible for latency analysis
or final selection.

SLURM `Start` and `End` fields are retained exactly as returned by the
cluster and are not relabeled as UTC. Job-scoped runner timestamps are UTC.
The machine-readable artifact contains both where relevant.

## Integrity anchors

| Evidence | SHA-256 |
| --- | --- |
| `sensitivity_latency_manifest.v1.json` | `c569f858f4513139292d7189ab5e57f897b8794fdbe5b2dcafc45b0efcd663aa` |
| `runtime/sensitivity_latency/block_submissions.json` | `f227b6123f762091a81b341bd9b824599e399e801dd2d2fb85ce26a948ba2214` |
| `runtime/sensitivity_latency/slurm_state.db` snapshot | `9f78e14247a6cbf68221ee2a923432661a27b230c73dba8ace5d14ab9a522fc6` |
| v1 block runner at commit `e728a46192d8de53407eb1c663c35d590bfe2f73` | `d68002ff5ab19df5dc2d3e4d2d281c69e7d2786caea68964a1a527c68a585320` |
| `sensitivity_latency_manifest.v2.json` at audit | `aedc117414b2691c1a70b73fa4e9e0ac123cb4d20dfd9d25dfe2d4aa490d7655` |
| `sensitivity_latency_v1_failure_audit.json` | `65d096fed00a4857b6b5f7170fa59a289d2ba8e323b08a79991155fdbac0de3f` |

The JSON artifact also pins each remote `allocation_result.json`, `main.err`,
`main.out`, generated sbatch file, and generated entrypoint independently.
No credentials, secret values, login-host names, or environment snapshots are
stored.

## Read-only reproduction commands

Run from the repository root. Credential values are loaded silently and are
never printed:

```bash
set -a
source ~/.tao/config.env >/dev/null 2>&1
set +a
audit_host="${SLURM_HOSTNAME%%,*}"

job_ids='30943464,30943481,30943494,30943509,30943521,30943525,30943542,30943558,30943574'
ssh -o BatchMode=yes "${SLURM_USER}@${audit_host}" \
  "sacct -j ${job_ids} --noheader --parsable2 \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NodeList,Elapsed,Start,End"
```

Recheck the frozen local anchors:

```bash
sha256sum \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_manifest.v1.json \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency/block_submissions.json \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency/slurm_state.db

git show \
  e728a46192d8de53407eb1c663c35d590bfe2f73:experiments/dino_moo_phase2_20260728/sensitivity_latency_block_runner.py \
  | sha256sum
```

Recheck one job-scoped result and its absence of rank measurements. Repeat
with the exact per-job relative paths recorded in the JSON artifact:

```bash
tao_id='ee899aa0-1b57-4545-9f78-9030e6ad872a'
allocation='seed_001234_allocation_0_row_00'
result_base='/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/results'
allocation_root="${result_base}/${tao_id}/dino_moo_phase2_20260728/sensitivity_latency/dino_sensitivity_latency_20260728_v1/seed_001234/${allocation}"

ssh -o BatchMode=yes "${SLURM_USER}@${audit_host}" \
  "sha256sum '${allocation_root}/allocation_result.json'; \
   python3 -m json.tool '${allocation_root}/allocation_result.json'; \
   find '${allocation_root}' -type f -name 'rank_*.json' -print"
```

Recheck the corresponding immutable logs:

```bash
slurm_id='30943464'
log_root='/lustre/fsw/portfolios/edgeai/users/rarunachalam/slurm-logs'
ssh -o BatchMode=yes "${SLURM_USER}@${audit_host}" \
  "sha256sum \
   '${log_root}/tao-job-${tao_id}-${slurm_id}/main.out' \
   '${log_root}/tao-job-${tao_id}-${slurm_id}/main.err'"
```

Validate the tracked-candidate artifact:

```bash
python -m json.tool \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_v1_failure_audit.json \
  >/dev/null
sha256sum \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_v1_failure_audit.json
```
