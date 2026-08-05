# SLURM credential-staging containment audit

## Scope and conclusion

This is an operational security audit of the SLURM launch path used by the
DINO phase-2 expanded search. It records no credential value and no
credential-derived hash.

The TAO SDK intentionally staged container credentials in per-job files and
embedded credential exports in each generated sbatch script. The individual
files were owner-only (`0600`), but their parent directories were initially
world-writable (`0777`) without a sticky bit. That combination prevented
direct reading of the original files but did not protect their path integrity:
another cluster user could unlink or replace a known-path script or entrypoint.

The user base and observed credential-staging directories were subsequently
hardened to `0700`. The files remained `0600`. This blocks traversal and
replacement by group and other users and contains the immediate shared-path
risk. It does not remove the plaintext credentials, undo prior exposure, or
provide automatic end-of-job cleanup.

No evidence from this audit indicates that an algorithmic candidate,
measurement, objective value, or selection decision was changed. The
containment action changed filesystem access modes only. If later evidence of
staging-file replacement or unauthorized access is found, the affected job
evidence must be treated as untrusted and revalidated.

## Affected staging paths

For every TAO job, the SLURM handler creates these paths beneath the configured
user base:

```text
sbatch/job_<tao-job-id>.sbatch
entrypoints/job_<tao-job-id>.sh
specs/<tao-job-id>.json
env/<tao-job-id>.json
meta/<tao-job-id>.json
```

The credential-bearing locations are:

- the sbatch script, which contains container-environment exports;
- the environment JSON;
- the cloud-metadata JSON.

The entrypoint and specs files are part of the same executable staging
contract. They did not contain the raw container credential mapping observed
in the environment files, but their path integrity is security-sensitive
because the sbatch job executes the entrypoint.

Three first-candidate jobs were in scope when the issue was observed. This
audit intentionally omits credential contents and does not reproduce their
values.

## Initial permissions and risk

The initial metadata snapshot showed:

| Object | Mode | Owner ACL | Group ACL | Other ACL |
| --- | ---: | --- | --- | --- |
| User staging base | `0777` | `rwx` | `rwx` | `rwx` |
| `sbatch/` | `0777` | `rwx` | `rwx` | `rwx` |
| `entrypoints/` | `0777` | `rwx` | `rwx` | `rwx` |
| `specs/` | `0777` | `rwx` | `rwx` | `rwx` |
| `env/` | `0777` | `rwx` | `rwx` | `rwx` |
| `meta/` | `0777` | `rwx` | `rwx` | `rwx` |
| `slurm-logs/` | `0777` | `rwx` | `rwx` | `rwx` |
| Each staged file inspected | `0600` | `rw-` | `---` | `---` |

No extended ACL granted an additional named principal access to the staged
files. The files were owned by the submitting user.

`0600` protected file contents against an ordinary direct read. It did not
make the construction safe because POSIX deletion and rename authorization is
controlled by the parent directory. With `0777` and no sticky bit, another
user could:

1. enumerate or infer the known per-job path;
2. unlink or rename the owner-only file;
3. create a replacement at the same path;
4. have an active or retried job execute the replacement under the submitting
   user's identity and credential-bearing environment.

The initial condition was therefore a high-severity integrity and potential
credential-exfiltration risk, even though the original credential files were
not directly readable by group or other users.

The audit found no metadata evidence that the inspected first-candidate files
had been replaced. That observation does not substitute for credential
rotation after exposure.

## Containment action

Between the initial snapshot and the containment recheck, a concurrent
operator—not this read-only audit—restricted the canonical user staging base
and the observed sensitive staging directories to `0700`.

The following metadata-change times were observed:

| Path | Contained mode | Observed ctime (UTC) |
| --- | ---: | --- |
| Canonical user staging base | `0700` | 2026-07-28 04:00:26 |
| `entrypoints/` | `0700` | 2026-07-28 04:03:28 |
| `sbatch/` | `0700` | 2026-07-28 04:03:29 |
| `env/` | `0700` | 2026-07-28 04:03:34 |
| `meta/` | `0700` | 2026-07-28 04:03:35 |

Representative sbatch, entrypoint, environment, and metadata files remained:

```text
mode: 0600
ACL:  user::rw-, group::---, other::---
owner: submitting user
```

The `0700` canonical parent is the decisive containment boundary. It prevents
group and other users from traversing to nested paths, even if a deeper
directory has a more permissive mode. It also prevents the unlink-and-replace
attack that was possible through the original world-writable parents.

The change preserves access for the submitting user and therefore does not by
itself alter active job commands, candidate configurations, or result
generation.

## Result isolation

Credential-staging files are outside the job-scoped result root:

```text
<user-base>/sbatch/
<user-base>/entrypoints/
<user-base>/env/
<user-base>/meta/

<user-base>/results/<tao-job-id>/
```

The inspected job result roots contained the generated TAO configuration and
training output tree. They did not contain copies of the sbatch, environment,
cloud-metadata, or entrypoint staging files.

This separation also follows from the code:

- `SlurmSDK` routes outputs to the job-scoped Lustre result root.
- `script_runner` persists only declared outputs/configuration generated
  inside that result root.
- The handler stages scheduler files separately beneath the user base.
- `delete_job_artifacts()` targets only
  `<base-results-dir>/results/<tao-job-id>`.

Accordingly, the normal AutoML candidate tables, combined-selection JSON, and
job result artifacts do not publish the credential-staging files. The launch
preflight records loaded credential key names and states that secret values
are not recorded; it does not contain their values.

## Persistence and retry reuse

The current SDK has no terminal-job cleanup for the staging paths above.
Deleting a job's result artifacts removes only its job-scoped result
directory. It does not remove:

- the sbatch script;
- the environment JSON;
- the cloud-metadata JSON;
- the entrypoint;
- the staged specs file.

These files therefore remain indefinitely unless a separate, explicitly
scoped cleanup operation removes them.

The sbatch file also remains operationally significant after initial launch.
The SDK's infrastructure-failure retry path submits the same recorded remote
sbatch path again under the same TAO job identity and a new SLURM job ID.
Removing or rotating the staged command while a retry remains possible can
break recovery; replacing it can change what a retry executes.

`SLURM_USE_REQUEUE=false` disables the sbatch script's timeout requeue tail. It
does not disable the SDK handler's separate automatic resubmission path.

## SDK source provenance

The observed behavior is present in TAO SDK commit:

```text
3d3e1adc1849493d29dc926cb99492417e3a9250
branch: rarunachalam/pre-platform-sdk-removal-20260714
```

Relevant code paths are:

- `tao_sdk/platforms/slurm/sdk.py`
  - `SlurmSDK.create_job()` builds the container environment and injects
    configured S3, NGC, and Hugging Face credentials.
- `tao_sdk/platforms/slurm/handler.py`
  - `_create_job_locked()` serializes the environment and cloud metadata,
    stages the five per-job files, builds the sbatch script, and submits it;
  - `_build_slurm_script()` emits credential exports while temporarily
    suppressing shell xtrace;
  - `_scp_text()` creates the remote parent and copies each staged file;
  - `_resubmit_failed_job()` reuses the recorded remote sbatch path;
  - `delete_job_artifacts()` deletes only the job-scoped result directory.

Git history attributes the core environment/metadata staging and sbatch
construction to commit:

```text
2ce73405
```

Commit:

```text
3edc3ff
```

later suppressed xtrace around credential exports. That change reduces
accidental log disclosure but does not eliminate plaintext staging or provide
cleanup.

## Follow-up requirements

### Credential response

Treat every credential present in the affected staging contract as exposed:

1. rotate or revoke the affected AWS/S3, NGC, and Hugging Face credentials;
2. review their provider-side access history where available;
3. coordinate rotation with active jobs because revocation can break remaining
   input or model access;
4. do not place replacement credential values in experiment reports, logs,
   commits, tickets, or chat.

### Staging cleanup

After all affected jobs are terminal and no retry or forensic preservation is
needed:

1. resolve the exact TAO job IDs from durable state;
2. verify terminal scheduler lineage;
3. remove only the exact per-job staging files;
4. record filenames and deletion status without recording contents;
5. retain algorithmic result artifacts separately.

Cleanup must not run while a job or retry can still reference its sbatch or
entrypoint.

### SDK hardening

The SLURM backend should:

- create the canonical user base and all staging directories as `0700`;
- create files atomically as `0600` and reject symlinks or ownership drift;
- verify owner, mode, path, and immutable command identity before initial
  submission and retry;
- avoid writing duplicate environment/cloud-metadata secret files when they
  are not consumed;
- prefer short-lived or workload-scoped secret delivery over long-lived
  literal exports;
- provide explicit terminal staging cleanup with retry-aware retention;
- test hostile parent permissions and unlink/replace races.

## Algorithmic impact

This containment audit is not an AutoML selection input. It:

- does not change the DINO search space;
- does not alter search or training seeds;
- does not change accuracy or latency measurements;
- does not feed the selector;
- does not promote, demote, or override a candidate;
- does not change Pareto ranks or mode winners.

The observed first-candidate remote training specs matched their persisted
candidate records, and training was active without a detected functional
failure at the audit snapshot. The filesystem hardening changed access control,
not experiment values. Algorithmic evidence remains separate from this
operational containment record.
