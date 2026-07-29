# TAO AutoML — Complete Interface Definitions

Every public class, method, parameter, and return type across all AutoML components.
For use by agent skills, SDK integrations, and team review.

---

## 1. AutoML (top-level entry point)

**File**: `tao_automl/__init__.py`
**Purpose**: Main entry point. Wires SearchSpace → Brain → Controller. Delegates all calls to Controller.

```python
class AutoML:

    def __init__(
        self,
        workspace: str,                       # Local path for state persistence
        network: str,                         # Network name ("cosmos-rl", "dino", etc.)
        train_specs: dict,                    # Base training spec (default config)
        settings: dict,                       # Algorithm config (see AlgorithmParams)
        automl_hyperparameters: list = None,  # Param names to search, or None for schema defaults
        custom_param_ranges: dict = None,     # Per-param range overrides
        resume: bool = False,                 # Resume from persisted state
    ) -> None

    def next_recommendation(self) -> list[Recommendation]
        # Get next batch of hyperparameter recommendations.
        # Returns 1 for Bayesian/BFBO, N for Hyperband/ASHA/PBT. May be empty.

    def report_result(
        self,
        rec_id: int,              # Recommendation ID from rec.id
        metric_value: float,      # The metric value achieved
        best_epoch: int = None,   # Best epoch number (optional)
        status: str = "success",  # "success" or "failure"
    ) -> None
        # Feed back a training result. Thread-safe via file lock.

    def is_complete(self) -> bool
        # Check if optimization is done.

    def get_best(self) -> Recommendation | None
        # Best-performing recommendation so far (by metric direction).

    def get_progress(self) -> dict
        # Returns {"completed": int, "total": int, "best_metric": float|None,
        #          "best_rec_id": int|None, "algorithm": str}

    def get_history(self) -> list[Recommendation]
        # All recommendations generated so far.
```

---

## 2. Types

**File**: `tao_automl/types.py`

### JobStates

```python
class JobStates:
    pending   = "pending"
    started   = "started"
    running   = "running"
    success   = "success"
    failure   = "failure"
    error     = "error"      # alias for failure
    done      = "done"       # alias for success
    canceled  = "canceled"
    canceling = "canceling"
```

### Recommendation

```python
class Recommendation:

    def __init__(
        self,
        identifier: int,    # Must be int
        specs: dict,         # Must be dict — hyperparameter key→value pairs
        metric: str,         # Metric name (e.g. "loss")
    ) -> None

    # Attributes:
    id: int                         # Unique identifier
    specs: dict                     # {"train.optm_lr": 1e-5, "policy.lora.r": 16, ...}
    job_id: str | None              # Caller-assigned via assign_job_id()
    status: str                     # "pending" | "success" | "failure" | "canceled"
    result: float                   # Metric value (0.0 until reported)
    best_epoch_number: str          # Best epoch (if reported)
    metric: str                     # Metric name
    resume_from_job_id: str | None  # PBT/Hyperband: checkpoint to resume from
    early_stop_epoch: int | None    # Hyperband: epoch limit for this config
    created_on: str                 # ISO timestamp
    last_modified: str              # ISO timestamp

    def items(self) -> dict_items
        # Returns specs.items()

    def get(self, key: str) -> any
        # Returns specs.get(key, None)

    def assign_job_id(self, job_id: str) -> None
        # Associates job ID. Must be str.

    def update_result(self, result: float) -> None
        # Update metric value. Casts to float.

    def update_status(self, status: str) -> None
        # Update status. Must be str.

    def __repr__(self) -> str
        # "id: N\njob_id: ...\nresult: ...\nstatus: ..."
```

### ResumeRecommendation

```python
class ResumeRecommendation:

    def __init__(
        self,
        identity: int,                        # Recommendation ID
        specs: dict,                          # Hyperparameter specs
        job_id: str,                          # Job ID
        resume_from_job_id: str | None = None # Checkpoint source job ID (PBT)
    ) -> None

    # Attributes:
    id: int
    specs: dict
    job_id: str
    resume_from_job_id: str | None
```

### AutoMLContext

```python
@dataclass
class AutoMLContext:
    id: str                    # Session/experiment ID
    network: str               # Network architecture name
    action: str = "train"
    workspace_path: str = ""
    metric: str = "loss"
    handler_id: str = ""       # Experiment ID for custom ranges
    num_gpu: int = -1
```

---

## 3. Controller

**File**: `tao_automl/controller/controller.py`
**Purpose**: Manages the brain loop — generates recs, tracks results, persists state.

```python
class Controller:

    def __init__(
        self,
        brain: AutoMLAlgorithmBase,      # Brain instance (Bayesian, Hyperband, etc.)
        context: AutoMLContext,
        state_store: StateStore,
        settings: AlgorithmParams,
        metric: str,                     # "loss", "accuracy", etc.
        algorithm: str,                  # "bayesian", "hyperband", etc.
        parameter_names: list = None,    # List of dotted param names
    ) -> None

    def next_recommendation(self) -> list[Recommendation]
        # Asks brain for recs, wraps as Recommendation objects, persists.

    def report_result(
        self,
        rec_id: int,
        metric_value: float,
        best_epoch: int = None,
        status: str = "success",
    ) -> None
        # Thread-safe: acquires state_store.lock() for read-modify-write.

    def get_best(self) -> Recommendation | None
        # Best by metric. "loss" → lowest. "accuracy" → highest.

    def get_progress(self) -> dict
        # {"completed", "total", "best_metric", "best_rec_id", "algorithm"}

    def get_history(self) -> list[Recommendation]
        # Full recommendation list.

    def is_complete(self) -> bool
        # Bayesian/BFBO: completed >= max_recommendations.
        # Hyperband/BOHB/ASHA/DEHB/PBT/HyperBandES: brain.done()

    def save_state(self) -> None
        # Persist history to state_store.

    @classmethod
    def load_state(
        cls,
        brain: AutoMLAlgorithmBase,
        context: AutoMLContext,
        state_store: StateStore,
        settings: AlgorithmParams,
        metric: str,
        algorithm: str,
        parameter_names: list = None,
    ) -> Controller
        # Restore controller from persisted state.

    # Internal:
    def _find_rec(self, rec_id: int) -> Recommendation | None
    def _estimate_total(self) -> int
    @staticmethod
    def _serialize_rec(rec: Recommendation) -> dict
```

---

## 4. StateStore

**File**: `tao_automl/state/state_store.py`
**Purpose**: JSON file persistence with fcntl.flock concurrency. All files under `workspace/.automl/`.

```python
class StateStore:

    def __init__(self, workspace_path: str) -> None
        # Creates .automl/ directory. Sets up global lock + thread lock.

    def lock(self) -> _FileLock
        # Context manager for compound read-modify-write.
        # Usage: with state_store.lock(): ...

    # --- Job Specs ---
    def get_job_specs(self, job_id: str) -> dict | None
    def save_job_specs(self, job_id: str, specs: dict) -> None

    # --- Brain State ---
    def get_brain_info(self, job_id: str) -> dict | None
    def save_brain_info(self, job_id: str, state: dict) -> None

    # --- Controller History ---
    def get_controller_info(self, job_id: str) -> list[dict] | None
    def save_controller_info(self, job_id: str, recs: list[dict]) -> None

    # --- Current Recommendation Pointer ---
    def get_current_rec(self, job_id: str) -> int | None
    def save_current_rec(self, job_id: str, rec_id: int) -> None

    # --- Custom Parameter Ranges ---
    def get_custom_param_ranges(self, experiment_id: str) -> dict | None
    def save_custom_param_ranges(self, experiment_id: str, ranges: dict) -> None

    # --- Best Recommendation ---
    def get_best_rec_info(self, job_id: str) -> dict | None
        # Returns {"rec_number": int, "rec_data": dict} or None
    def save_best_rec_info(self, job_id: str, rec_number: int, rec_data: dict) -> None
```

**Storage layout:**
```
workspace/.automl/
├── specs/{id}.json
├── brain/{id}.json
├── controller/{id}.json
├── best_rec/{id}.json
├── current_rec/{id}.json
└── custom_ranges/{id}.json
```

**Concurrency**: Reads use `LOCK_SH`, writes use `LOCK_EX`, plus `threading.Lock` for in-process safety. Per-thread unique temp files for atomic writes.

---

## 5. Brain Base Class

**File**: `tao_automl/brain/base.py`
**Purpose**: Base for all 8 HPO algorithms. Handles parameter value generation with type-aware sampling and constraints.

```python
def is_nan_value(val: any) -> bool
    # Check if value (or any element in list/tuple) is NaN.

class AutoMLAlgorithmBase:

    def __init__(
        self,
        context: AutoMLContext,
        state_store: StateStore,
        network: str,              # Network name
        parameters: list[dict],    # List of param config dicts from search space
    ) -> None
        # Loads default_train_spec, custom_ranges, initializes random seed.

    def generate_automl_param_rec_value(self, parameter_config: dict) -> any
        # Generate random value based on parameter type:
        #   float: np.random.uniform(v_min, v_max) + network-specific logic
        #   int/integer: np.random.randint + math_cond (^ 2, / N)
        #   bool: random 0/1
        #   categorical/ordered: np.random.choice(valid_options) + weights
        #   ordered_int: choice from valid_options + weights
        #   subset_list: random subset from valid_options
        #   optional_list: 50% None, 50% all valid options
        #   list_1_backbone/list_1_normal: consecutive number list
        #   list_2/list_3: various list formats
        #   dict/collection: network-specific handler
        #   string: random from valid_options

    def _apply_power_constraint_with_equal_priority(
        self,
        v_min: float,
        v_max: float,
        factor: float,
        fallback_value: float = None,
    ) -> any
        # Sample uniformly from valid powers of factor in range.

    def _apply_relational_constraint(
        self,
        value: any,
        math_cond: str,           # e.g. "> depends_on", "<= depends_on"
        depends_on: str,          # Parent parameter name
        parameter_name: str,
        v_min: float,
        v_max: float,
    ) -> any
        # Enforce relational constraint against parent param value.
```

---

## 6. Brain Factory + AlgorithmParams

**File**: `tao_automl/brain/factory.py`

### AlgorithmParams

```python
@dataclass
class AlgorithmParams:
    automl_max_recommendations: int = 20    # Bayesian/BFBO: max trials
    automl_max_epochs: int = 27             # Multi-fidelity: epoch budget
    automl_reduction_factor: int = 3        # Hyperband/BOHB/ASHA/DEHB halving
    epoch_multiplier: int = 1
    automl_max_concurrent: int = 4          # ASHA: parallel configs
    automl_population_size: int = 10        # PBT: population size
    automl_max_generations: int = 20        # PBT: max generations
    automl_eval_interval: int = 10          # PBT: eval interval epochs
    automl_perturbation_factor: float = 1.2 # PBT: perturbation magnitude
    automl_mutation_factor: float = 0.5     # DEHB
    automl_crossover_prob: float = 0.5      # DEHB
    automl_early_stop_threshold: float = 0.1  # HyperBandES
    automl_min_early_stop_epochs: int = 3     # HyperBandES
    automl_kde_samples: int = 64              # BOHB
    automl_top_n_percent: float = 15.0        # BOHB
    automl_min_points_in_model: int = 10      # BOHB
    automl_max_trials: int | None = None      # ASHA: max total configs
    automl_min_top_configs: int = 5           # ASHA: min at final rung
    automl_delete_intermediate_ckpt: bool = True   # Prune terminal non-best artifacts

    @classmethod
    def from_dict(cls, params_dict: dict) -> AlgorithmParams
        # Create from dict with defaults for missing keys.
```

### AlgorithmType

```python
class AlgorithmType:
    BAYESIAN     = ("bayesian", "b")
    BFBO         = ("bfbo",)
    HYPERBAND    = ("hyperband", "h")
    BOHB         = ("bohb",)
    ASHA         = ("asha",)
    PBT          = ("pbt",)
    DEHB         = ("dehb",)
    HYPERBAND_ES = ("hyperband_es", "hes")
```

### BrainFactory

```python
class BrainFactory:

    @staticmethod
    def create_brain(
        algorithm: str,                  # One of AlgorithmType values
        context: AutoMLContext,
        state_store: StateStore,
        network: str,
        parameters: list[dict],          # From generate_hyperparams_to_search
        params: AlgorithmParams,
        metric: str = "loss",
        resume: bool = False,
    ) -> AutoMLAlgorithmBase
        # Creates the right brain instance. If resume=True, calls brain.load_state().
```

---

## 7. Bayesian Brain

**File**: `tao_automl/brain/bayesian.py`
**Purpose**: Gaussian Process + Expected Improvement acquisition function.

```python
class Bayesian(AutoMLAlgorithmBase):

    def __init__(
        self,
        context: AutoMLContext,
        state_store: StateStore,
        network: str,
        parameters: list[dict],
    ) -> None
        # GP: Matérn 5/2 kernel, alpha=1e-10, 10 restarts.
        # State: Xs (list of [0,1]^d vectors), ys (list of metric values).

    def generate_automl_param_rec_value(
        self,
        parameter_config: dict,
        suggestion: float,        # GP suggestion in [0, 1]
    ) -> any
        # Maps [0,1] suggestion to actual param value.

    def generate_recommendations(self, history: list[Recommendation]) -> list[dict]
        # First call: random [0,1]^d. Subsequent: GP.fit → optimize_ei → map.
        # Returns list with 1 dict of {param_name: value}.

    def save_state(self) -> None
        # Persist Xs, ys to brain JSON.

    @staticmethod
    def load_state(
        context: AutoMLContext,
        state_store: StateStore,
        network: str,
        parameters: list[dict],
    ) -> Bayesian
        # Restore from JSON, re-fit GP.

    def update_gp(self) -> None
        # gp.fit(Xs, ys). Handles inf/nan by replacing with 1e7/0.

    def optimize_ei(self) -> np.ndarray
        # L-BFGS-B on _expected_improvement, 5 restarts. Returns [0,1]^d.

    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> float
        # EI(x) = (μ - f_best - ξ)Φ(Z) + σφ(Z). Returns negated for minimization.
```

---

## 8. Search Space

**File**: `tao_automl/search_space/params.py`

```python
def generate_hyperparams_to_search(
    network: str,                          # Network name
    action: str,                           # "train"
    train_specs: dict,                     # Base training spec
    automl_hyperparameters: list[str],     # Explicit param names, or [] for defaults
    override_automl_disabled_params: bool = False,
) -> tuple[list[dict], list[str]]
    # Returns (param_records, param_names)
    #
    # param_records: list of dicts, each with:
    #   parameter: str          ("train.optm_lr")
    #   value_type: str         ("float", "int", "categorical", ...)
    #   default_value: any
    #   valid_min, valid_max: any
    #   valid_options: str|list
    #   option_weights: list
    #   math_cond: str          ("^ 2", "/ 16", "> depends_on")
    #   parent_param: str|bool
    #   depends_on: str
    #   automl_enabled: bool
    #
    # param_names: list of str  (["train.optm_lr", "policy.lora.r", ...])
```

---

## 9. AutoMLRunner (SDK glue)

**File**: `tao_automl/runner.py`
**Purpose**: Wires AutoML brain to TaoExecutionSDK. Lives in SDK, NOT in standalone wheel.

```python
def _extract_metric_from_logs(logs: str, metric_name: str) -> float | None
    # Extract final metric from logs. Patterns (searched in order):
    #   1. "Step: N/M, Loss: X.XXX" (Cosmos-RL format)
    #   2. "{metric_name}: X.XXX" or "best {metric_name}: X.XXX"
    #   3. "kpi: X.XXX"
    #   4. "Epoch N ... loss: X.XXX"
    # Skips 0.0 values. Returns None if not found.

def _check_execution_status(logs: str) -> str | None
    # Returns "PASS", "FAIL", or None.

class AutoMLRunner:

    def __init__(
        self,
        sdk: TaoExecutionSDK,
        poll_interval: int = 30,     # Seconds between status polls
    ) -> None

    def run(
        self,
        network_arch: str,
        train_dataset_uri: str,
        eval_dataset_uri: str = "",
        base_checkpoint: str = "",
        workspace_id: str = None,
        image: str = None,
        automl_settings: dict = None,
        automl_hyperparameters: list = None,
        custom_param_ranges: dict = None,
        workspace_path: str = "./automl_workspace",
        spec_overrides: dict = None,
        resume: bool = False,
        on_recommendation: callable = None,   # callback(rec)
        on_result: callable = None,           # callback(rec, metric, status)
    ) -> dict
        # Platform is handled entirely by the SDK — the runner is
        # platform-agnostic and has no backend_details parameter.
        # Returns:
        # {
        #   "best": {"rec_id": int, "specs": dict, "metric_value": float},
        #   "progress": {"completed": int, "total": int, "best_metric": float,
        #                "best_rec_id": int, "algorithm": str},
        #   "history": [{"rec_id": int, "metric": float, "status": str}, ...]
        # }

    def _run_one_job(
        self,
        network_arch: str,
        workspace_id: str,
        train_dataset_uri: str,
        eval_dataset_uri: str,
        base_checkpoint: str,
        image: str,
        specs: dict,
        rec: Recommendation,
        metric_name: str,
    ) -> tuple[float | None, str]
        # Launch job, poll status+logs (caching metrics during training),
        # return (metric_value, "success"|"failure").

    @staticmethod
    def _merge_specs(base_specs: dict, rec_specs: dict) -> dict
        # Deep-merge dotted keys: "train.optm_lr" → specs["train"]["optm_lr"]

def run_automl_plan(plan: dict, creds_file: str = None) -> dict
    # Execute an automl_plan.json file via AutoMLRunner.

def main()
    # CLI: python -m tao_automl.runner automl_plan.json [secrets.json]
```

---

## 10. settings Dict Reference

The `settings` dict passed to `AutoML(settings=...)`:

| Key | Type | Default | Used by |
|-----|------|---------|---------|
| `algorithm` | str | **required** | All — selects brain type |
| `metric` | str | `"loss"` | All — "loss" → lower is better |
| `objectives` | list[dict] | unset | Explicit objectives. A two-objective maximize-accuracy/minimize-latency list enables constrained Pareto archive selection. |
| `selection_mode` | str | `"multi_objective"` | Two-objective archives — one of `accuracy`, `latency`, or `multi_objective`. |
| `latency_accuracy_retention` | number or dict | relative 0.98 | Latency mode only — accuracy-winner-relative retained fraction (`relative`) or maximum degradation (`absolute`). A number is relative shorthand. |
| `multi_objective_min_accuracy` | number or dict | unset | Multi-objective mode only — optional eligibility floor. A number is an absolute metric floor; a dict accepts `{"type": "absolute", "value": floor}` or `{"type": "relative", "value": fraction, "reference": "accuracy_winner"}`. Unset means no accuracy floor. |
| `accuracy_constraint` | dict | unset | Deprecated compatibility alias for `latency_accuracy_retention`; it never constrains the multi-objective front. |
| `objective_normalization` | str | `"pareto_front"` | Two-objective archives — front-relative regret normalization. |
| `augmentation_rho` | float | `1e-6` | Multi-objective final selection — strictly positive augmented-Chebyshev tie term. |
| `accuracy_tolerance` | float | `1e-12` | Accuracy-mode equivalence and the strict-improvement threshold in Pareto comparisons; it never makes lower accuracy "no worse." |
| `latency_tolerance` | float | `0.0` | Hard inclusive boundary around latency mode's raw-minimum anchor. In Pareto comparisons, confidence-interval overlap can withhold a latency-only strict claim, but it never widens the latency cohort or makes a slower median "no worse." |
| `selection_score_tolerance` | float | `1e-12` | Multi-objective compromise-score equivalence. |
| `random_seed` | int | stable session-derived | Candidate-generation RNG; explicit values are recorded and reproducible across processes. |
| `require_eval_fn_success` | bool | true for two-objective archives | Fail the candidate when required benchmark evaluation raises or omits a metric; disables fallback to progress-log metrics. |
| `automl_max_recommendations` | int | 20 | Bayesian, BFBO |
| `automl_max_epochs` | int | 27 | Hyperband, BOHB, ASHA, DEHB |
| `automl_reduction_factor` | int | 3 | Hyperband, BOHB, ASHA, DEHB |
| `epoch_multiplier` | int | 1 | Hyperband, BOHB, ASHA, DEHB |
| `automl_max_concurrent` | int | 4 | ASHA |
| `automl_population_size` | int | 10 | PBT |
| `automl_max_generations` | int | 20 | PBT |
| `automl_eval_interval` | int | 10 | PBT |
| `automl_perturbation_factor` | float | 1.2 | PBT |
| `automl_mutation_factor` | float | 0.5 | DEHB |
| `automl_crossover_prob` | float | 0.5 | DEHB |
| `automl_early_stop_threshold` | float | 0.1 | HyperBandES |
| `automl_kde_samples` | int | 64 | BOHB |
| `automl_top_n_percent` | float | 15.0 | BOHB |
| `automl_delete_intermediate_ckpt` | bool | True | All — delete confirmed-terminal failed/non-best job artifacts. During Hyperband-family/PBT searches, retain only the latest promotion-decision window, active resume parents, and current best; completed searches collapse to the winner. Hybrid successes remain retained unless a full-fidelity winner is verified; multi-objective searches retain the Pareto frontier. Cleanup-aware SDKs reject unreclaimable output routes before launch. Set false to retain all artifacts for debugging. |
| `automl_checkpoint_retention_strategy` | string | `"auto"` | Training jobs when `automl_delete_intermediate_ckpt=true` — bounds checkpoint files inside each retained job. `auto` uses `best` when the merged spec exposes `train.checkpointer`, otherwise `terminal`. `best` enables top-1 checkpointing with the trainer-declared monitor/mode (falling back to the AutoML objective) and requests replacement of periodic saves; it requires a trainer whose `train.checkpointer` contract honors `replace_periodic`, so use `terminal` with older additive-only checkpointers. `terminal` sets the epoch checkpoint interval to the recommendation's effective `train.num_epochs`, preserving ASHA/Hyperband rung budgets. Allowed values: `auto`, `best`, `terminal`. Ignored when intermediate-checkpoint deletion is disabled. |
| `session_id` | str | auto | All — override session ID |
| `experiment_id` | str | auto | All — override experiment ID |

The repository-wide latency retention default remains `0.98`. Product profiles
that permit a larger accuracy reduction must opt in explicitly. The DINO
latency validation profile uses:

```python
automl_settings = {
    "selection_mode": "latency",
    "latency_accuracy_retention": {
        "type": "relative",
        "retained_fraction": 0.90,
        "reference": "accuracy_winner",
    },
    # Independent: latency retention never becomes a multi-objective floor.
    "multi_objective_min_accuracy": None,
}
```

This means `minimum_accuracy = 0.90 * accuracy_mode_winner_accuracy`; it does
not subtract ten percentage points from the accuracy metric. Relative retention
must be finite and satisfy `0 < retained_fraction <= 1`. The complete checked-in
profile is
[`dino_latency_90_policy_profile.v1.json`](../experiments/dino_moo_phase2_20260728/dino_latency_90_policy_profile.v1.json).
Unknown retention-map keys, selector settings without a resolvable
maximize-accuracy/minimize-latency objective pair, and explicitly named
metrics absent from `objectives` are rejected rather than silently defaulted
or routed through legacy scalarization.

---

## 11. Algorithm Behaviors

| Algorithm | `next_recommendation()` returns | `is_complete()` when | Parallel? |
|---|---|---|---|
| `bayesian` | 1 rec | completed >= max_recs | No |
| `bfbo` | 1 rec | completed >= max_recs | No |
| `hyperband` | Batch (per rung) | All brackets exhausted | Yes (within rung) |
| `bohb` | Batch | All brackets exhausted | Yes |
| `asha` | Up to max_concurrent | brain.done() | Yes (async) |
| `pbt` | population_size recs | generations done | Yes (full pop) |
| `dehb` | Batch | All brackets exhausted | Yes |
| `hyperband_es` | Batch | All brackets exhausted | Yes |
