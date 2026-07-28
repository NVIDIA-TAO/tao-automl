# Expanded DINO search v1 failure audit

## Disposition

The first expanded-search launch is **invalid and excluded from all AutoML
selection evidence**.

It used only DINO ResNet50 and:

```text
s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/
```

The three seed controllers started from the immutable expanded-search manifest
with whole-file SHA-256
`57e331686b8896989263a39f72edb69543fc58833f20a1e6e698c31f34d2e8be`.
The launch preflight pinned the committed v1 runner at AutoML commit
`fae47d3406ea29bfc03893f9808b50958eef70c6`, with source SHA-256
`211c926065ee63a9d7476e312d2e89ee48b9f0189bb6330ce632c3604d4af668`.

All three recommendation-zero training and evaluation jobs completed
successfully. The TAO evaluation status files contained finite mAP50 values,
but serialized them as JSON strings. The v1 result reader rejected those
strings before its log-parser fallback and before launching latency
measurement. Consequently:

- all three rec0 records were reported to AutoML as failures with null metrics;
- no rec0 contains an accepted accuracy objective;
- no selection-time latency job or measurement exists;
- each Bayesian brain has one proposed `X` and zero observed `y` values;
- no valid seed archive or combined archive exists;
- rec1 work was interrupted and canceled during controller shutdown.

No v1 candidate, metric, checkpoint, recommendation history, or partial archive
may be imported into final selection. The corrected v2 experiment must start
from fresh workspaces and fresh SDK state using the same deterministic search
seeds `314159`, `271828`, and `161803`. It must not resume these v1 brain or
controller files.

## Root cause

At the launch-pinned v1 source, `read_status_map50()` reads the final
`test_mAP50` or `val_mAP50` field and sends it directly to:

```python
def finite_number(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ContractError(f"{label} must be a finite number")
    return float(value)
```

The three TAO status files emitted syntactically valid finite values, but the
JSON representation was a string:

```json
{"test_mAP50": "0.499933605841208"}
```

Because a string is not an `int` or `float`, the reader raised:

```text
ContractError: evaluation mAP50 must be a finite number
```

The caller invokes the regex log extractor only when `read_status_map50()`
returns `None`; it does not invoke the fallback when the status reader raises.
The exception therefore escaped the evaluation reader. Candidate measurement
then stopped before `launch_latency_benchmark()` and recorded:

```text
required_eval_fn_failed:evaluation mAP50 must be a finite number
```

This is a v1 orchestration/parser defect. It is not a DINO training failure,
evaluation failure, non-finite model metric, SLURM failure, dataset failure, or
credential failure.

## Rec0 training and evaluation mapping

The candidate parameters below came from the deterministic algorithm
recommendations. No candidate was manually injected.

| Search seed | Candidate | Encoder / decoder depth | Learning rate | Weight decay |
| ---: | --- | --- | ---: | ---: |
| 161803 | `seed_161803_rec_0` | 3 / 5 | 0.0002098775727573059 | 0.0006123641159628601 |
| 271828 | `seed_271828_rec_0` | 6 / 6 | 0.0000459777499171801 | 0.0006077207436969115 |
| 314159 | `seed_314159_rec_0` | 5 / 6 | 0.0002156899238307862 | 0.00010770493619675102 |

Every training and evaluation allocation completed with scheduler exit
`0:0`.

| Candidate | Training TAO job | Training SLURM job | Evaluation TAO job | Evaluation SLURM job | Evaluation status |
| --- | --- | ---: | --- | ---: | --- |
| `seed_161803_rec_0` | `959f0fe6-d0c9-48d9-a0d8-aac26ebd485a` | 30950521 | `04792472-d79e-4eaf-917c-f332c3c9d487` | 30951491 | TAO `Complete`; SLURM `COMPLETED/0:0` |
| `seed_271828_rec_0` | `b3933f47-fa9b-405b-aa0b-2555dec8ced7` | 30950520 | `f50e5737-3a93-4937-8c6a-e39fcf56ec66` | 30951528 | TAO `Complete`; SLURM `COMPLETED/0:0` |
| `seed_314159_rec_0` | `7be04920-5c71-43b1-bdf4-d29c00d56405` | 30950522 | `aec669db-9c32-4e5a-b3b0-4db023d3b62c` | 30951502 | TAO `Complete`; SLURM `COMPLETED/0:0` |

The candidate ledgers retain the terminal checkpoint identities even though
the associated training roots were later removed:

| Candidate | Terminal checkpoint SHA-256 | Size in bytes |
| --- | --- | ---: |
| `seed_161803_rec_0` | `85571846ba8df9479ac1a710369880931cd1c9c9825d39950e3c6c5a6ff490eb` | 497629458 |
| `seed_271828_rec_0` | `f24d8c0b17a3ae8a126cd8f11c85a595e87c29ca808c1bde6cf880828a8be087` | 562439674 |
| `seed_314159_rec_0` | `dca4cb790ff6067ac1565ccbccb52915ea35814cb4cfedd9a48ef80e2a5a5d97` | 547031034 |

## Exact remote mAP50 evidence

The evaluation roots remain present. Read-only inspection of their final TAO
status files produced these exact string values:

| Candidate | Evaluation TAO job | Exact serialized `test_mAP50` | Status-file SHA-256 |
| --- | --- | --- | --- |
| `seed_161803_rec_0` | `04792472-d79e-4eaf-917c-f332c3c9d487` | `"0.499933605841208"` | `8fa3a749c4999423e9bee15f3fbd1cd837515e93a9f0cf73c085a93ebb400faf` |
| `seed_271828_rec_0` | `f50e5737-3a93-4937-8c6a-e39fcf56ec66` | `"0.5175292656942001"` | `070237fe94f85c715f5495d4a938ce71778fd299b4711b7c8f6091929595c66f` |
| `seed_314159_rec_0` | `aec669db-9c32-4e5a-b3b0-4db023d3b62c` | `"0.5728509799562066"` | `1adcf675a0513b7d13eec87447c0deb66225ffc65cfd088b2614d07e66f41c05` |

These values establish that evaluation produced a metric. They are not v1
AutoML observations: the required evaluator rejected them, the candidate
records contain no accepted `mAP50`, and latency was never measured. This audit
does not manually import, reinterpret, or promote them.

## Rec0 failure state

All three candidate ledgers contain the same terminal state:

```text
status=training_or_measurement_failure
automl_result_status=failure
metric=null
failure_reason=required_eval_fn_failed:
               evaluation mAP50 must be a finite number
```

The per-seed controller state likewise records rec0 as `failure` with
`metric_value=null`. The failure occurred after a complete training job and a
complete evaluation job, but before an accuracy value or latency result could
be committed to the candidate record.

## Rec1 interruption and cancellation

Each controller deterministically requested rec1 after reporting rec0 as a
failure. Shutdown then interrupted that work.

| Search seed | Candidate-ledger state | Controller / SDK evidence | TAO job | SLURM job | Terminal state |
| ---: | --- | --- | --- | ---: | --- |
| 161803 | `recommended` | Controller associated the rec with the TAO job | `21acba1e-4cbb-43e4-97dc-fd1d4193decc` | 30952017 | SDK `Canceled`; SLURM `CANCELLED/0:15` |
| 271828 | `recommended` | Controller remained `pending`; SDK created a job before SLURM assignment | `8676986e-4620-4cf4-a9bc-75dc5b77957e` | — | SDK `Canceled`; no SLURM job ID was assigned |
| 314159 | `recommended` | Controller associated the rec with the TAO job | `85c92e03-0e45-41b7-813e-56bd5f7a5686` | 30952020 | SDK `Canceled`; SLURM `CANCELLED/0:15` |

Both assigned rec1 SLURM jobs ran from cluster-local
`2026-07-27T21:06:36` through `2026-07-27T21:07:19` before cancellation.
No rec1 evaluation or latency job was launched.

## Controller stop and quiescence

Every seed result is:

```json
{"status": "failure", "error": "SystemExit: 1", "candidate_count": 2}
```

The local UTC filesystem finalization times are:

| Evidence | Finalized at UTC |
| --- | --- |
| Seed 271828 `result.json` | `2026-07-28T04:07:20.135443261Z` |
| Seed 161803 `result.json` | `2026-07-28T04:07:41.799442496Z` |
| Seed 314159 `result.json` | `2026-07-28T04:07:55.575442009Z` |
| Global `seed_process_status.json` | `2026-07-28T04:07:55.739442003Z` |

The global process-status file records exit code `1` for all three seeds. Its
filesystem time is the audit's controller-group stop/finalization time; the
JSON itself does not embed a timestamp.

At the `2026-07-28T04:14:35.683985530Z` audit snapshot:

- all three `active_jobs.json` files were exactly `[]`;
- all nine SDK records were terminal: six `Complete` and three `Canceled`;
- `squeue` returned no row for any assigned v1 SLURM ID;
- no local `expanded_search_runner.py` process was active.

No cancellation, deletion, relaunch, or other mutation was performed by this
audit.

## Deleted training roots

The AutoML artifact ledgers mark all six SDK-routed training artifacts
`deleted`. Read-only remote checks independently found all six result roots
absent:

| Search seed | Rec | TAO job / routed result root identity | Artifact ledger | Remote root |
| ---: | ---: | --- | --- | --- |
| 161803 | 0 | `959f0fe6-d0c9-48d9-a0d8-aac26ebd485a` | `deleted` | absent |
| 161803 | 1 | `21acba1e-4cbb-43e4-97dc-fd1d4193decc` | `deleted` | absent |
| 271828 | 0 | `b3933f47-fa9b-405b-aa0b-2555dec8ced7` | `deleted` | absent |
| 271828 | 1 | `8676986e-4620-4cf4-a9bc-75dc5b77957e` | `deleted` | absent |
| 314159 | 0 | `7be04920-5c71-43b1-bdf4-d29c00d56405` | `deleted` | absent |
| 314159 | 1 | `85c92e03-0e45-41b7-813e-56bd5f7a5686` | `deleted` | absent |

The three evaluation roots and their status files remain present as failure
evidence. Their presence does not restore the deleted training checkpoints or
create a valid AutoML observation.

## No latency measurement and no usable Bayesian observation

For every seed:

- rec0 has no `mAP50` field;
- rec0 has no `selection_time_latency` field;
- rec0 has no `objective_values` field;
- the SDK job store contains only rec0 training, rec0 evaluation, and rec1
  training records;
- there is no latency TAO job or SLURM job;
- the evaluation result tree contains zero latency-named paths;
- the Bayesian brain has `len(Xs)=1` and `len(ys)=0`.

The single `X` is a proposed parameter vector without a response. It is not a
Bayesian observation and cannot update the surrogate. No candidate has both
accuracy and latency, no candidate has `status=success`, and no candidate can
enter Pareto ranking.

The following required outputs do not exist:

- all three `seed_archive.v1.json` files;
- `expanded_combined_selection.json`;
- the complete expanded candidate JSON/CSV table;
- `expanded_integrity_audit.json`.

The absence of these artifacts is expected fail-closed behavior. It is not
permission to construct a partial archive manually.

## Credential-containment non-impact

The separate SLURM credential-staging containment audit is committed as
`ce0797c4896dd0e79e9e70c2222bcc63217891b3`; its file SHA-256 is
`2664a2826d34e8e2b8b3914ef8d83e7f57b1def56dc0cd81253d2c2609eacfd9`.
It records no credential value or credential-derived hash.

That containment changed filesystem access modes only. It did not change the
search space, deterministic seeds, candidate specifications, training output,
evaluation status files, metric strings, latency measurements, selector input,
or winner. There is no evidence that staging-file replacement or unauthorized
access altered these jobs. The successful `0:0` training/evaluation scheduler
states and the exact status-file hashes further separate the parser failure
from credential handling.

The containment action therefore did not cause the v1 parser failure and does
not salvage v1 as selection evidence. Operational credential response remains
separate from this AutoML disposition.

## Integrity anchors

| Local evidence | SHA-256 |
| --- | --- |
| v1 launch manifest | `57e331686b8896989263a39f72edb69543fc58833f20a1e6e698c31f34d2e8be` |
| v1 launch preflight | `51dbeddbe740d0d631e04caefe5192d7acb3571e5df89ba9dc723a4cfd7fc839` |
| v1 launch-pinned runner | `211c926065ee63a9d7476e312d2e89ee48b9f0189bb6330ce632c3604d4af668` |
| Global seed process status | `e9828355deff98dda9d5808857572226e1b3ac42cc5622ecffa34a744f11287d` |
| Seed 161803 events | `769970581f860ef55bc72866a068301d46e186b110459fb26fb961fa6785f66f` |
| Seed 271828 events | `fe044be2e3cc6df8c8ce80a7e239b4dbcb5a2c14b32a749d46df6507c3412717` |
| Seed 314159 events | `4b375d0f7e805d523984b1295a037d7f65488366f74fba162235d8a43d44ea5c` |
| Seed 161803 candidate ledger | `5f46b5c906b126bfb92151ceece966a400700f8f6b971339ebb74c8b05f918ab` |
| Seed 271828 candidate ledger | `cb0cfa0a7c832f728d51630fde93204d8a6e87744f97ddd20dfa771e02dd1835` |
| Seed 314159 candidate ledger | `a6d7ba549b02f28d770f52d8c22991d318da5b910d92839367e9b54385ce9f4b` |
| Seed 161803 Bayesian brain | `4592d869640d2f99c3792bcf8f389d877c3849f82ff3341eaea3d43a4bd4b5aa` |
| Seed 271828 Bayesian brain | `b938a64d68ee12ef097bdd3b95c3bbb95283142620cf1036516e324a51e5ab58` |
| Seed 314159 Bayesian brain | `3094664b19efcd4d288c5bcf4c618d2e902f292406775a8a1458441441f72394` |
| Seed 161803 controller state | `2cfc463e262690d6bfcd470c334977bd4906fff620263c7c9667777ecc16e18d` |
| Seed 271828 controller state | `f007cdf09304b3370542e96e6df34c58ca668cca10cb172aca13e284fb771024` |
| Seed 314159 controller state | `03128bb47c84814432d6bcf14c25685fad41a9ce94208c046cbe430a47d35c2e` |
| Seed 161803 result | `77891bb923cbff85fe7ba78aba45b690728a83e111ff0fddb8af7a78a2bc8124` |
| Seed 271828 result | `ac78f80ebf57918aa093fb83de0f9129a88fece91cef9ff411d1b5180f4a4bd2` |
| Seed 314159 result | `a46585ab10c8ab749b6d431b49880df4061c1702354a6c4eb12999cde86added` |
| Empty active-job file, identical for all seeds | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| Seed 161803 artifact ledger | `53312586999b8eb223bd0697cd0a2f79945534e6f08f1dae11d2b8124f652aca` |
| Seed 271828 artifact ledger | `f810b1a9d0991f2a0b8112e7fb734459b34d0dd7b3be378db29a4a7d525a5b44` |
| Seed 314159 artifact ledger | `5244baf2ea3bcabf5e1df877771cc107f5bc2ba9e00e72c35fe38fd00c8ee056` |

The remote evaluation-status hashes are listed with the exact mAP50 strings
above. Runtime SQLite databases are not treated as immutable source blobs;
their job identities and terminal states were joined with the read-only
scheduler query instead.

## Required v2 restart semantics

V2 must:

1. bind a corrected, reviewed metric reader to a new immutable launch
   contract;
2. use a new runtime directory, new per-seed SDK databases, and new AutoML
   workspaces;
3. atomically bind that runtime to the exact v2 manifest before spawning seed
   controllers, reject every pre-existing fresh seed state, and require the
   same manifest-bound marker and state allowlist for resume;
4. rerun all three deterministic seeds from rec0 with the same frozen search
   space and 20-recommendation budget;
5. rerun training, accuracy evaluation, and stabilized latency for each
   successful candidate;
6. allow only newly measured complete objective pairs to update each Bayesian
   brain;
7. seal three fresh 20-record seed archives before union selection;
8. keep every v1 record and job ID excluded from the v2 archive and selector.

V2 must not use `--resume` against
`runtime/expanded_search/seed_*/workspace/run_20260728_035441`. Reusing those
workspaces would retain partial controller state and proposed `X` values
without corresponding `y` observations. The deterministic rerun requirement
is satisfied by starting clean with the same seeds, not by continuing the
invalid v1 state.
