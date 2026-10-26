# Resume compatibility

Resume selects an existing search; it is no longer a resume-or-create operation.
Call `AutoMLRunner.run(resume=True, workspace_path="/path/to/run_<timestamp>")`
with the full run directory, not the directory containing several runs.
Missing, unreadable, or ambiguous controller state raises `ValueError` before
recovery or trial launch. Start a fresh search with `resume=False` instead.

An explicit `session_id` must identify an existing controller when resuming.
It may be omitted only when the run contains exactly one controller. Fresh
runs generate an ID when none is supplied. IDs must match
`[A-Za-z0-9][A-Za-z0-9._-]{0,127}`: 1–128 ASCII characters, beginning with a
letter or digit, then letters, digits, dots, underscores, or hyphens. Previously
accepted values containing spaces or beginning with an underscore now fail
validation. Do not rename individual state files to migrate an old run; the
identity is also referenced by other persisted search state.

## Remaining checkpoint limitation

Optimizer state is persisted before controller history and before launching a
trial. This supports recovery after a runner dies during a trial. The two
files are individually replaced atomically, but are not a joint transaction:
a process killed between the writes can leave optimizer state ahead of history
and misalign subsequent Bayesian/BFBO observations. Recovery from this exact
checkpoint-write window is not guaranteed. Restart the search in a new run
directory rather than continuing a run known to have inconsistent state.

Truncating optimizer points to the number of history entries is not a safe
repair: failed trials remain in controller history but are removed from the
optimizer, and optimizer observations and scheduler state must stay aligned
too. A joint atomic checkpoint is needed to eliminate this remaining window.
