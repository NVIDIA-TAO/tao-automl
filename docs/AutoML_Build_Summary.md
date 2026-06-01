# TAO AutoML — Build Summary & Knowledge Base

**Date:** 2026-04-03
**Author:** Tejas Anand
**Repo:** `git@github.com:NVIDIA-TAO/tao-automl.git`

---

## 1. What We Built

A standalone Python wheel (`nvidia-tao-automl`) that extracts the hyperparameter optimization (HPO) brain from NVIDIA's FTMS monolith into an independent, portable package.

**Two-wheel architecture:**

| Wheel | Scope | Depends on |
|---|---|---|
| `nvidia-tao-automl` | HPO brain, search space, controller, state, runner | `tao-sdk` |
| `tao-sdk` | Job execution, specs, monitoring | `nvidia-tao-core` |

The dependency direction is **automl → SDK** (not vice versa), because automl is useless without job execution capability, but SDK users may never need AutoML.

---

## 2. What Came From FTMS vs What's New

### Ported from FTMS (algorithm logic identical)
- `brain/base.py` — Base algorithm with value generation (float, int, categorical, power-of-2, relational constraints)
- `brain/bayesian.py` — GP with Matérn 5/2 kernel + Expected Improvement acquisition
- `brain/hyperband.py`, `bohb.py`, `asha.py`, `bfbo.py`, `dehb.py`, `pbt.py`, `hyperband_es.py` — All 8 HPO algorithms
- `search_space/` — Parameter discovery from TAO Core network config dataclasses (`automl_enabled="TRUE"` metadata)
- `schema/` — Network-specific parameter schemas (cosmos-rl LoRA constraints, multi-part LR, etc.)
- `config/` — AlgorithmParams, algorithm factory
- `types.py` — Recommendation, ResumeRecommendation (verbatim from FTMS)
- `utils/` — Parameter encoding/decoding utilities

### Rewritten
- `controller/controller.py` — 87KB FTMS monolith → 315-line pure brain loop manager
  - Removed: Flask routes, MongoDB calls, FTMS job orchestration, Docker management
  - Kept: recommendation lifecycle, completion logic, state transitions
- `state/state_store.py` — 100% new, replaces 15+ MongoDB functions from `stateless_handler_utils.py`
  - JSON file I/O with `fcntl.flock` concurrency protection
  - Per-thread unique temp files for atomic writes
  - `lock()` context manager for compound read-modify-write operations

### Brand new (not in FTMS)
- `__init__.py` — `AutoML` class: the public API (`next_recommendation()`, `report_result()`, `is_complete()`, `get_best()`, `get_progress()`, `get_history()`)
- `types.py` — `AutoMLContext` dataclass (replaces FTMS's 18-field `JobContext` with 7 fields)
- `runner.py` — `AutoMLRunner`: wires AutoML brain to SDK execution, handles full HPO loops
  - Cosmos-RL config auto-fixes (batch size, model_max_length, dp_shard_size, validation)
  - Log caching during polling (Lepton expires logs within seconds of job completion)
  - 4-pattern metric extraction (Step format, generic, KPI, epoch)
  - `spec_overrides` and `resume` support

---

## 3. Architecture Flow

```
User / Agent
     │
     ▼
AutoMLRunner.run(network_arch, train_dataset_uri, automl_settings={...})
     │
     ├──▶ AutoML(workspace, network, train_specs, settings)
     │         │
     │         ├──▶ SearchSpace  (discover tunable params from TAO Core schemas)
     │         ├──▶ BrainFactory (select algorithm: bayesian, hyperband, etc.)
     │         ├──▶ Controller   (manage recommendation lifecycle)
     │         └──▶ StateStore   (JSON file persistence under workspace/.automl/)
     │
     │    Loop: while not automl.is_complete()
     │         │
     │         ├──▶ automl.next_recommendation() → [Recommendation]
     │         │
     │         ├──▶ runner._run_one_job(specs)
     │         │         │
     │         │         ├──▶ sdk.create_job(...)
     │         │         ├──▶ poll: sdk.get_job_status() + sdk.get_job_logs()
     │         │         │         └──▶ cache metrics from logs during polling
     │         │         └──▶ return (metric_value, status)
     │         │
     │         └──▶ automl.report_result(rec_id, metric_value, status)
     │
     └──▶ return {best: {...}, progress: {...}, history: [...]}
```

---

## 4. Key Design Decisions

1. **File-based state, not MongoDB** — The standalone wheel has no external dependencies for persistence. State lives in `workspace/.automl/` as JSON files with file-level locking (`fcntl.flock`).

2. **Algorithm logic untouched** — Every brain algorithm (Bayesian GP, Hyperband brackets, BOHB, etc.) was ported with identical logic. Only imports and state I/O changed.

3. **Runner owns network-specific fixes** — Cosmos-RL quirks (batch_size divisibility, model_max_length=40960, dp_shard_size=1, validation.enable=True) are applied in the runner before AutoML sees the specs.

4. **Log caching** — Lepton expires job logs within seconds of completion. The runner reads logs every poll cycle and caches the latest metric value, so results aren't lost.

5. **Clean public API** — FTMS had no single entry point; callers had to wire Flask routes → controller → brain → MongoDB manually. The `AutoML` class provides a clean 6-method API.

---

## 5. Bugs Found & Fixed During E2E Testing

All discovered during real Cosmos-RL training on DGX Cloud Lepton (H100 GPUs):

| Bug | Root Cause | Fix |
|---|---|---|
| `train_batch_per_replica(1) must be divisible by mini_batch(4)` | Default spec has batch=1, mini_batch=4 | Runner auto-sets batch = mini_batch for cosmos-rl |
| `vision_embeds.shape[0] != n_tokens` token overflow | model_max_length=4096, needs 40960 for video | Runner sets `policy.model_max_length = 40960` |
| `FileNotFoundError: data/sft/annotations.json` | SDK dataset injection only handled `mapping` dict, not cosmos-rl flat `data_sources` format | Added flat format injection to SDK |
| Downloaded `images.tar.gz` instead of `videos.tar.gz` | `path_from_format` defaulted to images | Changed to prefer `llava` format paths |
| Metric extraction returns 0.0 | Lepton expires logs before runner reads them post-completion | Added log caching during polling |
| Wrong metric (validation 0.0 vs training 8.27) | Regex picked first match, not the meaningful one | Reordered patterns, skip 0.0 values |
| `UnboundLocalError` when `validation.enable = false` | Container bug in cosmos-rl | Runner forces `validation.enable = True` |
| `aws://bucket/data//annotations.json` double slash | URIs with trailing `/` | `rstrip("/")` on all URIs |
| Concurrent `report_result` crash (8 threads) | `FileNotFoundError` on temp file race | `fcntl.flock` + `threading.Lock` + per-thread temp files |

---

## 6. Test Results

### Unit Tests (21 tests)
`/Users/tejasanand/tao-automl/tests/test_wheel.py` — All pass:
- Import verification (all subpackages)
- Type construction (Recommendation, AutoMLContext)
- StateStore CRUD (write/read/update/delete, file locking)
- Controller lifecycle (init, recommendation, report, completion)
- Concurrency (8 threads, no data corruption)
- AlgorithmParams (all 8 algorithms)
- Wheel metadata

### E2E Test (Cosmos-RL on Lepton)
- 3 Bayesian recommendations, H100 GPU
- nvidia/Cosmos-Reason1-7B, LoRA SFT, video dataset
- Results: loss values 0.799 and 4.196
- Full trace in `docs/AutoML_E2E_Proof_Report.md`

---

## 7. Package Structure

```
tao-automl/
├── pyproject.toml              # nvidia-tao-automl wheel config
├── src/
│   └── tao_automl/
│       ├── __init__.py         # AutoML class (public API)
│       ├── runner.py           # AutoMLRunner (wires brain → SDK execution)
│       ├── types.py            # Recommendation, AutoMLContext
│       ├── brain/
│       │   ├── base.py         # AutoMLAlgorithmBase
│       │   ├── bayesian.py     # Gaussian Process + EI
│       │   ├── hyperband.py    # Hyperband
│       │   ├── bohb.py         # BOHB
│       │   ├── asha.py         # ASHA
│       │   ├── bfbo.py         # BFBO
│       │   ├── dehb.py         # DEHB
│       │   ├── pbt.py          # PBT
│       │   └── hyperband_es.py # HyperBand with early stopping
│       ├── controller/
│       │   └── controller.py   # 315-line brain loop manager
│       ├── state/
│       │   └── state_store.py  # JSON file persistence with flock
│       ├── search_space/       # Parameter discovery from TAO Core schemas
│       ├── schema/             # Network-specific param schemas
│       ├── config/             # AlgorithmParams, BrainFactory
│       └── utils/              # Parameter encoding/decoding
├── tests/
│   └── test_wheel.py           # 21 unit tests
└── docs/
    ├── AutoML_Build_Summary.md             # This file
    ├── AutoML_Complete_Interface_Definitions.md
    ├── AutoML_E2E_Proof_Report.md
    ├── automl_runner_interface.excalidraw
    └── automl_runner_interface.png
```

---

## 8. Dependencies

```toml
dependencies = [
    "numpy",
    "scikit-learn",       # Gaussian Process for Bayesian optimization
    "scipy",              # Optimization (minimize for EI acquisition)
    "pandas",             # Tabular data handling
    "omegaconf",          # Config management
    "requests",
    "tao-sdk @ git+https://github.com/NVIDIA-TAO/tao-sdks.git",
]
```

---

## 9. How to Use

### Programmatic
```python
from tao_sdk import TaoExecutionSDK
from tao_automl.runner import AutoMLRunner

sdk = TaoExecutionSDK(creds_file="secrets.json")
runner = AutoMLRunner(sdk)
result = runner.run(
    network_arch="cosmos-rl",
    train_dataset_uri="aws://bucket/data/subset",
    automl_settings={
        "algorithm": "bayesian",
        "metric": "loss",
        "automl_max_recommendations": 5,
    },
    spec_overrides={"train.epoch": 2},
)
print(result["best"])
```

### CLI
```bash
python -m tao_automl.runner automl_plan.json secrets.json
```

### Agentic (via skill bank)
The AutoML skill lives in `tao-skills-external/applications/tao-automl/SKILL.md`.
It guides an LLM agent through: parse intent → select algorithm → configure & run → monitor → interpret results.

---

## 10. Open Items

- **`list_searchable_params()`** — Runner method to let agents discover tunable parameters programmatically. Discussed but not yet implemented.
- **SDK dataset injection hardcoding** — The flat format injection and video preference logic in `sdk.py` may be too cosmos-rl specific. Needs review for generalization.
- **Multi-node concurrency** — Current runner runs recommendations sequentially. ASHA algorithm supports `automl_max_concurrent` but the runner doesn't launch parallel jobs yet.
