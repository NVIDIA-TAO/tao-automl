# TAO AutoML + TAO SDK — End-to-End Proof of Execution

**Date**: 2026-03-30
**Run by**: Tejas Anand
**Model**: Cosmos-RL (nvidia/Cosmos-Reason1-7B)
**Platform**: DGX Cloud Lepton (H100 SXM)
**Algorithm**: Bayesian Optimization (Gaussian Process + Expected Improvement)

---

## 1. Experiment Configuration

```python
from tao_sdk import TaoExecutionSDK
from tao_automl.runner import AutoMLRunner

sdk = TaoExecutionSDK(creds_file="secrets.json")
runner = AutoMLRunner(sdk)
result = runner.run(
    network_arch="cosmos-rl",
    train_dataset_uri="aws://nvcf-storage-handling/data/cosmos_rl_wts_train_subset",
    image="nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl",
    automl_settings={
        "algorithm": "bayesian",
        "metric": "loss",
        "automl_max_recommendations": 3,
    },
)
```

**Dataset**: `cosmos_rl_wts_train_subset` — 248KB annotations + 624MB videos (LLaVA format)
**Search space**: 7 hyperparameters auto-discovered from network schema

| Parameter | Type | Range | Constraint |
|---|---|---|---|
| `train.epoch` | int | 1–20 | — |
| `train.optm_lr` | float | 0–∞ | Log-uniform sampling |
| `train.optm_decay_type` | categorical | linear, sqrt, cosine, none | Weighted |
| `policy.lora.r` | int | 1–256 | Power of 2 |
| `policy.lora.lora_alpha` | int | 1–1024 | Power of 2 |
| `policy.lora.lora_dropout` | float | 0.0–0.1 | — |
| `custom.vision.fps` | int | 1–3 | — |

---

## 2. Execution Timeline

```
23:02:31  AutoML initialized: algorithm=bayesian, metric=loss, params=7
          Brain seed: random, GP kernel: Matérn 5/2

── Recommendation 0 ──────────────────────────────────────────────────
23:02:31  Brain generates first recommendation (random sampling)
          train.epoch = 12
          train.optm_lr = 1.416e-07
          train.optm_decay_type = none
          policy.lora.r = 4        (power of 2)
          policy.lora.lora_alpha = 512  (power of 2)
          policy.lora.lora_dropout = 0.0472
          custom.vision.fps = 1

23:02:32  Job 6005e65b submitted to Lepton (H100 SXM)
          → Container downloads 624MB videos.tar.gz from S3
          → Container downloads Cosmos-Reason1-7B from HuggingFace (~15GB)
          → Training starts: 12 epochs × 107 steps/epoch
          → Loss: 6.685 → decreasing over steps

00:15:07  Job completed — Execution status: PASS
          Loss extracted from logs: 0.79879
          ✅ Reported to brain: metric=0.798790, status=success

── Recommendation 1 ──────────────────────────────────────────────────
00:15:07  Brain updates GP: gp.fit(Xs=[rec0_params], ys=[0.79879])
          Brain optimizes Expected Improvement → generates rec 1
          train.epoch = 8
          train.optm_lr = 5.728e-08    (GP explored lower LR)
          train.optm_decay_type = cosine
          policy.lora.r = 128          (GP explored higher rank)
          policy.lora.lora_alpha = 512
          policy.lora.lora_dropout = 0.0494
          custom.vision.fps = 1

00:15:09  Job 101c4e31 submitted to Lepton (H100 SXM)
          → Training: 8 epochs

01:18:34  Job completed — Execution status: PASS
          Loss extracted: 4.19579
          ✅ Reported to brain: metric=4.195790, status=success

── Recommendation 2 ──────────────────────────────────────────────────
01:18:34  Brain updates GP: gp.fit(Xs=[rec0, rec1], ys=[0.799, 4.196])
          GP learned: lower LR with small lora_r → better
          Brain generates rec 2 via EI optimization
          train.epoch = 8
          train.optm_lr = 2.206e-07
          train.optm_decay_type = none
          policy.lora.r = 64
          policy.lora.lora_alpha = 4
          custom.vision.fps = 3

01:18:35  Job e6635585 submitted to Lepton
01:28:01  Job failed (transient Lepton issue)
          ❌ Reported: metric=0.0, status=failure

── Loop Complete ─────────────────────────────────────────────────────
01:28:01  AutoML complete: 3 recommendations, best metric=0.798790 (rec 0)
```

**Total wall time**: ~2.5 hours (includes model download from HuggingFace on each cold start)

---

## 3. Results

### Best Configuration Found

```json
{
    "rec_id": 0,
    "metric_value": 0.79879,
    "specs": {
        "train.epoch": 12,
        "train.optm_lr": 1.416e-07,
        "train.optm_decay_type": "none",
        "policy.lora.r": 4,
        "policy.lora.lora_alpha": 512,
        "policy.lora.lora_dropout": 0.0472,
        "custom.vision.fps": 1
    }
}
```

### All Recommendations

| Rec | epoch | lr | decay | lora_r | lora_alpha | dropout | fps | Loss | Status |
|-----|-------|----|-------|--------|------------|---------|-----|------|--------|
| **0** | **12** | **1.42e-7** | **none** | **4** | **512** | **0.047** | **1** | **0.799** | **success** |
| 1 | 8 | 5.73e-8 | cosine | 128 | 512 | 0.049 | 1 | 4.196 | success |
| 2 | 8 | 2.21e-7 | none | 64 | 4 | 0.019 | 3 | — | failure |

### GP Learning Trace

- **After Rec 0** (loss=0.799): GP has 1 data point. EI optimization explores different region — higher lora_r (128), even lower lr (5.73e-8), cosine decay.
- **After Rec 1** (loss=4.196): GP now has 2 data points. Learns that Rec 0's config (small rank, moderate lr, no decay) is better. EI steers Rec 2 toward a compromise — medium rank (64), medium lr (2.21e-7), but tries fps=3 and low alpha=4.
- **After Rec 2** (failed): Brain records failure, marks experiment complete (3/3 recs done).

---

## 4. System Trace — What Each Component Did

### tao_automl (brain wheel)

```
1. SearchSpace: Generated schema from cosmos-rl config dataclass
   → Found 7 automl_enabled params: epoch, lr, decay, lora_r, lora_alpha, dropout, fps

2. BrainFactory: Created Bayesian brain
   → GP kernel: ConstantKernel(1.0) * Matérn(length_scale=[1,1,1,1,1,1,1], nu=2.5)
   → Acquisition: Expected Improvement (ξ=0.01)
   → Optimizer: L-BFGS-B with 5 random restarts

3. Controller: Managed 3 recommendations
   → Rec 0: random sampling in [0,1]^7, mapped to param values
   → Rec 1: GP.fit([rec0_X], [0.799]) → optimize_ei() → new [0,1]^7 → mapped
   → Rec 2: GP.fit([rec0_X, rec1_X], [0.799, 4.196]) → optimize_ei() → mapped

4. StateStore: Persisted all state to JSON files
   → ./automl_workspace/.automl/brain/{id}.json     (GP's Xs, ys)
   → ./automl_workspace/.automl/controller/{id}.json (recommendation history)
   → ./automl_workspace/.automl/best_rec/{id}.json   (best config)
```

### tao_sdk (executor wheel)

```
1. get_default_specs("cosmos-rl", "train")
   → Loaded 178 spec fields from config dataclass

2. _inject_datasets_into_specs()
   → annotation_path: aws://…/cosmos_rl_wts_train_subset/annotations.json
   → media_path: aws://…/cosmos_rl_wts_train_subset/videos.tar.gz
   (Fixed: detects videos.tar.gz via path_from_format llava format)

3. create_job() × 3
   → Resolved resource_shape=gpu.h100-sxm on gcp-iad-lepton-002-vnbwicri
   → Built cloud_metadata with S3 credentials
   → Launched Lepton jobs with TAO container image 6.26.3-cosmos-rl

4. get_job_status() + get_job_logs() (polled every 30s)
   → Cached loss values from logs during training
   → Extracted final metric after job completion
```

### Runner (glue)

```
1. Applied cosmos-rl config fixes:
   → train_batch_per_replica: 1 → 4 (must be ≥ mini_batch)
   → model_max_length: 4096 → 40960 (video token overflow prevention)
   → dp_shard_size: 1 (single GPU)
   → validation.enable: true (container bug if false)

2. For each recommendation:
   → Merged AutoML hyperparams into base specs
   → Submitted job via SDK
   → Polled status + cached log metrics during training
   → Reported result back to AutoML brain

3. On completion: returned best config + full history
```

### Lepton Container (TAO Toolkit 6.26.3-cosmos-rl)

```
Per job:
1. Downloaded dataset from S3 (624MB videos + 248KB annotations) — ~30s
2. Detected SFT mode, used /opt/cosmos_rl/tao_sft_example.py
3. Launched cosmos-rl with config → torchrun on 1×H100
4. Downloaded Cosmos-Reason1-7B from HuggingFace (~15GB) — ~2-3 min
5. Loaded model with LoRA adapters (r, alpha, dropout from AutoML)
6. Trained for N epochs (107 steps/epoch, ~3s/step on H100)
7. Saved checkpoint to /results/{job_id}/output/
8. Uploaded status.json to S3
9. Execution status: PASS
```

---

## 5. Persisted State (Resumable)

The AutoML state is fully persisted. If interrupted, `AutoML(..., resume=True)` would continue from the last completed recommendation.

```
./automl_workspace/.automl/
├── brain/f3e6d1f94c1a.json        ← GP state: Xs (7D vectors), ys (loss values)
├── controller/f3e6d1f94c1a.json   ← 3 recommendations with specs, results, timestamps
├── specs/f3e6d1f94c1a.json        ← Base training spec (178 fields)
├── best_rec/f3e6d1f94c1a.json     ← Best: rec 0, loss=0.79879
└── custom_ranges/                 ← (empty — no custom ranges used)
```

---

## 6. Bugs Found and Fixed During Testing

| Issue | Root Cause | Fix |
|---|---|---|
| Container crash: `train_batch_per_replica(1) must be divisible by mini_batch(4)` | Default spec has batch=1, mini_batch=4 | Runner auto-fixes to batch=mini_batch |
| Container crash: token overflow `vision_embeds.shape[0] != n_tokens` | `model_max_length: 4096` too small for video | Runner sets to 40960 per cosmos docs |
| No media downloaded: `FileNotFoundError: annotations.json` | `_inject_datasets_into_specs` didn't handle cosmos-rl flat data_sources format | Added flat format injection to SDK |
| Wrong media file: downloaded `images.tar.gz` (doesn't exist) instead of `videos.tar.gz` | `path_from_format` defaulted to `"*": "images.tar.gz"` | Changed to prefer `llava` format which includes `videos.tar.gz` |
| Metric extraction fails: loss=0.0 for all recs | Logs expire on Lepton before runner reads them post-completion | Added log caching during polling — reads logs every 30s while job runs |
| Validation bug: `UnboundLocalError` if `validation.enable = false` | Container bug | Runner forces `validation.enable = true` |

---

## 7. JSON Output

```json
{
  "best": {
    "rec_id": 0,
    "specs": {
      "train.epoch": 12,
      "train.optm_lr": 1.4159942779014233e-07,
      "train.optm_decay_type": "none",
      "policy.lora.r": 4,
      "policy.lora.lora_alpha": 512,
      "policy.lora.lora_dropout": 0.0472483786746007,
      "custom.vision.fps": 1
    },
    "metric_value": 0.79879
  },
  "progress": {
    "completed": 3,
    "total": 3,
    "best_metric": 0.79879,
    "best_rec_id": 0,
    "algorithm": "bayesian"
  },
  "history": [
    {"rec_id": 0, "metric": 0.79879, "status": "success"},
    {"rec_id": 1, "metric": 4.19579, "status": "success"},
    {"rec_id": 2, "metric": 0.0,     "status": "failure"}
  ]
}
```
