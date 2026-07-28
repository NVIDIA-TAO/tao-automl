# DINO phase-2 latency-sensitivity report

## Decision

This study is a preregistered one-factor screen, not an AutoML winner
selection. It uses only DINO ResNet50 and
`s3://nvcf-storage-handling/data/tao_od_synthetic_full_dino_coco/`.

The frozen qualification rule retains:

- `model.enc_layers`: qualified non-reference values `3`, `4`, and `5`,
  with reference value `6` retained in the expanded domain;
- `model.dec_layers`: qualified non-reference values `3`, `4`, and `5`,
  with reference value `6` retained in the expanded domain.

It excludes the complete `model.num_queries` and `model.num_select` axes from
the expanded latency-sensitive search. This decision follows only the
preregistered latency-effect rule: all tested encoder- and decoder-depth
reductions have hierarchical paired-effect confidence intervals wholly beyond
the practical noise band, while every query-count and selection-count interval
overlaps that band. Accuracy retention is reported separately and did not
qualify or disqualify an architecture value. No final candidate or winner was
selected, promoted, or overridden by this study.

The 98% result must therefore be read narrowly. It is an annotation for the
constrained latency mode, not a gate for the future shared multi-objective
archive. In particular, low encoder-depth accuracy does not invalidate the
measured latency sensitivity; the expanded search combines the retained
architecture axes with accuracy-influencing training parameters and lets the
algorithm evaluate the resulting trade-offs.

## Frozen protocol

The measurement and qualification contracts were frozen in
`sensitivity_latency_manifest.v2.json` before aggregation.

| Item | Frozen value |
| --- | --- |
| Reference architecture | `num_queries=594`, `num_select=300`, `enc_layers=6`, `dec_layers=6` |
| Fixed training parameters | learning rate `0.00045`; weight decay `0.00026967723799334445` |
| One-factor query levels | `300`, `450`, `750`, `900`; constraint `num_queries >= num_select` |
| One-factor encoder-depth levels | `3`, `4`, `5` |
| One-factor decoder-depth levels | `3`, `4`, `5` |
| One-factor selection-count levels | `50`, `100`, `200`; same-seed reference checkpoint reused |
| Training seeds | `1234`, `271828`, `314159` |
| Training/evaluation budget | 10 epochs; validation every epoch; checkpoint at epoch 10 |
| Training execution | 8 GPUs, DDP, batch size 4/GPU, FP32, TF32 off, deterministic cuDNN |
| Benchmark allocations | 9 independent 8-GPU A100 allocations; 3 per training seed |
| Ordering | all 14 profiles per allocation in nine balanced partial Williams rows |
| Benchmark input | model input `[1,4,800,1333]`; RGB `[1,3,800,1333]`; mask `[1,1,800,1333]` |
| Benchmark sampling | 16 preloaded inputs; 50 warm-ups; 5 rounds × 100 timed iterations |
| Synchronization | CUDA synchronization for every sample plus NCCL barriers |
| Timed scope | model forward plus DINO GPU postprocessing |
| Excluded scope | loading, disk I/O, decoding, resize/normalize, H2D, COCO accumulation, distributed gather |
| Primary latency statistic | median of device-round medians; pooled-sample p95 also recorded |
| Benchmark seed | `20260727` |
| Bootstrap | 5,000 resamples, 95% CI; deterministic seed rule pinned in the artifact |
| Precision/runtime | FP32; TF32 off; pinned TAO 7.0.1 SQSH; CUDA 13.2; cuDNN 92000 |

The screen contains 14 profiles, 33 independently trained checkpoints, 42
accuracy evaluations, and 126 valid matched allocation/profile measurements.
Each allocation/profile contributes 8 ranks and 4,000 timed raw samples. All
126 measurements passed the frozen hardware, runtime, protocol, input,
checkpoint, and evidence-integrity checks.

## Accuracy observations

The values below are generated from the immutable 42-entry training-accuracy
artifact. “98% by seed” compares each value with 98% of the same seed's
reference mAP50; it is not the latency-effect qualification gate.

| Profile | Axis value | mAP50 seed 1234 | mAP50 seed 271828 | mAP50 seed 314159 | Across-seed median | 98% by seed |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `reference` | reference | 0.616349 | 0.594082 | 0.637878 | 0.616349 | reference |
| `num_queries_300` | 300 | 0.603047 | 0.589169 | 0.592356 | 0.592356 | N/Y/N |
| `num_queries_450` | 450 | 0.563076 | 0.624094 | 0.602008 | 0.602008 | N/Y/N |
| `num_queries_750` | 750 | 0.557525 | 0.649796 | 0.594984 | 0.594984 | N/Y/N |
| `num_queries_900` | 900 | 0.621423 | 0.616138 | 0.655693 | 0.621423 | Y/Y/Y |
| `num_select_50` | 50 | 0.615510 | 0.592934 | 0.636935 | 0.615510 | Y/Y/Y |
| `num_select_100` | 100 | 0.616029 | 0.593654 | 0.637465 | 0.616029 | Y/Y/Y |
| `num_select_200` | 200 | 0.616087 | 0.594012 | 0.637857 | 0.616087 | Y/Y/Y |
| `enc_layers_3` | 3 | 0.519272 | 0.562078 | 0.518955 | 0.519272 | N/N/N |
| `enc_layers_4` | 4 | 0.512927 | 0.583577 | 0.551196 | 0.551196 | N/Y/N |
| `enc_layers_5` | 5 | 0.529423 | 0.587381 | 0.591078 | 0.587381 | N/Y/N |
| `dec_layers_3` | 3 | 0.637137 | 0.651348 | 0.631784 | 0.637137 | Y/Y/Y |
| `dec_layers_4` | 4 | 0.643576 | 0.682005 | 0.646883 | 0.646883 | Y/Y/Y |
| `dec_layers_5` | 5 | 0.625369 | 0.576455 | 0.598474 | 0.598474 | Y/N/N |

The 10-epoch accuracy values are screening measurements, not claims about
full-budget convergence. They show why latency qualification and a latency
mode's retained-accuracy constraint must remain separate: encoder depth has a
large, stable graph-cost effect even though these fixed-training-parameter
runs often miss 98% retention.

## Latency and dispersion

Every row below summarizes the same nine matched allocations. Bracketed values
are the minimum and maximum allocation statistics. Robust CV, round range,
device range, and bootstrap-CI width are the medians of the nine
allocation-level values and quantify within-allocation dispersion; the
allocation median and p95 ranges quantify between-allocation variability.

| Profile | Allocation median ms, median [min, max]; range | Allocation p95 ms, median [min, max]; range | Robust CV | Round-median range ms | Device-median range ms | Median bootstrap-CI width ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reference` | 77.850480 [77.675432, 78.138263]; 0.462831 | 78.221278 [78.008426, 78.408494]; 0.400069 | 0.002465 | 0.071756 | 0.490268 | 0.180408 |
| `num_queries_300` | 77.839945 [77.745868, 78.078378]; 0.332510 | 78.229924 [78.023935, 78.404316]; 0.380381 | 0.001845 | 0.027163 | 0.438119 | 0.123847 |
| `num_queries_450` | 77.846599 [77.654939, 78.178153]; 0.523214 | 78.028248 [77.901645, 78.419266]; 0.517621 | 0.001831 | 0.054036 | 0.344987 | 0.093317 |
| `num_queries_750` | 77.856575 [77.778955, 78.172055]; 0.393100 | 78.103423 [77.962866, 78.491139]; 0.528273 | 0.001950 | 0.051418 | 0.456944 | 0.119180 |
| `num_queries_900` | 77.975581 [77.856863, 78.177543]; 0.320680 | 78.229157 [78.124557, 78.396042]; 0.271485 | 0.001908 | 0.043493 | 0.425614 | 0.103891 |
| `num_select_50` | 77.829609 [77.678319, 78.092528]; 0.414209 | 78.122627 [77.977861, 78.272152]; 0.294291 | 0.002346 | 0.060281 | 0.459609 | 0.199186 |
| `num_select_100` | 77.791613 [77.645191, 78.167030]; 0.521840 | 78.129567 [77.974935, 78.430027]; 0.455092 | 0.002070 | 0.053481 | 0.441444 | 0.130648 |
| `num_select_200` | 77.855110 [77.736421, 78.182713]; 0.446292 | 78.192842 [78.015246, 78.611795]; 0.596549 | 0.002127 | 0.049018 | 0.505804 | 0.138930 |
| `enc_layers_3` | 63.616600 [63.424290, 63.834087]; 0.409797 | 63.865575 [63.579640, 64.206095]; 0.626455 | 0.002264 | 0.024283 | 0.368871 | 0.069892 |
| `enc_layers_4` | 68.451955 [68.251379, 68.605489]; 0.354110 | 68.745372 [68.429252, 69.024749]; 0.595497 | 0.001880 | 0.034951 | 0.365479 | 0.096634 |
| `enc_layers_5` | 73.101013 [72.874398, 73.371647]; 0.497249 | 73.422016 [73.153734, 73.729958]; 0.576225 | 0.002480 | 0.040096 | 0.570166 | 0.155023 |
| `dec_layers_3` | 65.662855 [65.548677, 65.801854]; 0.253177 | 65.925439 [65.714845, 66.139269]; 0.424424 | 0.001970 | 0.023326 | 0.385259 | 0.074462 |
| `dec_layers_4` | 69.715108 [69.678782, 69.846359]; 0.167576 | 69.962947 [69.831032, 70.175989]; 0.344957 | 0.001802 | 0.037998 | 0.439388 | 0.085859 |
| `dec_layers_5` | 73.748757 [73.677204, 74.036833]; 0.359629 | 74.074877 [73.879684, 74.333874]; 0.454190 | 0.001843 | 0.026028 | 0.344270 | 0.093417 |

## Matched effects, noise floor, and qualification

For each candidate profile, an allocation effect is:

```text
candidate median latency - reference median latency
```

using the same training seed and allocation. The hierarchical bootstrap
resamples the three training seeds, then the three matched allocations within
each sampled seed; it computes a within-seed median and then a median across
sampled seeds. It uses 5,000 deterministic resamples and a 95% interval.

The preregistered practical floor is:

```text
max(historical floor, maximum same-seed reference allocation range)
= max(0.73553775 ms, 0.46283150 ms)
= 0.73553775 ms
```

The reference allocation ranges are 0.21178750 ms for seed 1234,
0.46283150 ms for seed 271828, and 0.28123175 ms for seed 314159. A level
qualifies only when its entire hierarchical CI is below -0.73553775 ms or
above +0.73553775 ms. The rule is direction-agnostic; all effects that actually
qualified in this sweep were faster.

| Profile | Seed effects ms: 1234 / 271828 / 314159 | Median effect ms | Hierarchical 95% CI ms | Direction | Effect qualified | 98% suitable | Expanded-space action |
| --- | ---: | ---: | ---: | --- | :---: | :---: | --- |
| `num_queries_300` | -0.058378 / +0.013447 / -0.009623 | -0.009623 | [-0.198996, +0.111202] | uncertain/practical band | No | No | Exclude query axis |
| `num_queries_450` | -0.076109 / -0.020493 / -0.097944 | -0.076109 | [-0.151213, +0.039890] | uncertain/practical band | No | No | Exclude query axis |
| `num_queries_750` | -0.020133 / +0.033792 / +0.052647 | +0.033792 | [-0.025487, +0.124537] | uncertain/practical band | No | No | Exclude query axis |
| `num_queries_900` | +0.132980 / +0.143963 / +0.097262 | +0.132980 | [+0.025521, +0.181431] | uncertain/practical band | No | Yes | Exclude query axis |
| `enc_layers_3` | -14.277178 / -14.248272 / -14.283632 | -14.277178 | [-14.304176, -14.214302] | faster | Yes | No | Retain value 3 |
| `enc_layers_4` | -9.446314 / -9.457763 / -9.535503 | -9.457763 | [-9.535503, -9.386882] | faster | Yes | No | Retain value 4 |
| `enc_layers_5` | -4.783854 / -4.766617 / -4.770424 | -4.770424 | [-4.863874, -4.712530] | faster | Yes | No | Retain value 5 |
| `dec_layers_3` | -12.195899 / -12.196516 / -12.169440 | -12.195899 | [-12.336410, -12.119992] | faster | Yes | Yes | Retain value 3 |
| `dec_layers_4` | -8.144672 / -8.151055 / -8.108099 | -8.144672 | [-8.262732, -8.023165] | faster | Yes | Yes | Retain value 4 |
| `dec_layers_5` | -4.122297 / -4.101430 / -4.093055 | -4.101430 | [-4.137750, -3.998228] | faster | Yes | No | Retain value 5 |
| `num_select_50` | -0.059954 / -0.045735 / +0.108790 | -0.045735 | [-0.097175, +0.108790] | uncertain/practical band | No | Yes | Exclude select axis |
| `num_select_100` | -0.036462 / -0.030241 / -0.055941 | -0.036462 | [-0.072900, +0.004732] | uncertain/practical band | No | Yes | Exclude select axis |
| `num_select_200` | -0.050978 / +0.044449 / +0.020197 | +0.020197 | [-0.113545, +0.060989] | uncertain/practical band | No | Yes | Exclude select axis |

The query and selection effects are not merely smaller than the historical
floor: their CIs are close to zero and, except for query count 900, cross zero.
Query count 900's CI is wholly positive but remains far inside the
preregistered practical-equivalence band, so it does not qualify. Conversely,
the smallest qualified architecture effect is decoder depth 5 at about
-4.10 ms, over five times the effective noise floor in absolute magnitude.

## Search-space rationale and implications

### Supported and retained

`model.enc_layers` and `model.dec_layers` are supported by the current TAO DINO
ResNet50 configuration mapping, alter the inference graph directly, succeeded
at every tested seed, and produced repeatable effects well beyond both
within-allocation and between-allocation variability. The two axes are
therefore retained with the complete preregistered supported domain `3–6`:
depths 3–5 are the qualified non-reference levels, and depth 6 remains the
reference/control.

The final expanded AutoML archive may combine these axes with learning rate and
weight decay, which influence trained accuracy but not inference graph cost.
Those training parameters do not themselves satisfy the latency-sensitivity
screen and must not be described as latency axes.

### Supported but excluded

`model.num_queries` is supported under `num_queries >= num_select`, and
`model.num_select` is supported under `num_select <= num_queries`. All tested
profiles completed. Nevertheless, neither axis produced a repeatable,
practically meaningful latency effect under the fixed deployment shape and
timed scope. They are excluded from the expanded latency-sensitive search even
though query count 900 and all three selection-count values pass the separate
98% accuracy annotation. Keeping them because of favorable accuracy would
violate the preregistered latency qualification rule.

Selection-count profiles reused the same-seed reference checkpoint and
required no extra training. Their exclusion is therefore an evidence decision,
not a training-cost decision.

### Compatibility boundaries

The study did not broaden the model family. ResNet50 remained the only
compatible DINO PTM in the established TAO 7.0.1 evidence; GCViT was
unsupported and the FAN variants retained legacy projection incompatibilities.
Those unrelated PTM problems were not repaired.

Other apparent architecture controls were not admitted without evidence:

- hidden dimension, attention heads, feed-forward dimension, and attention
  point counts can change PTM tensor shapes;
- feature-level count is coupled to the feature mapping and is not exposed as
  an independent supported search axis by the current schema;
- input resolution is part of the controlled deployment/input profile here,
  rather than an interchangeable user-searchable parameter;
- precision/deployment variants were held fixed and are not treated as DINO
  architecture candidates.

This prevents an unsupported or deployment-specific parameter from entering
the search merely because it might theoretically affect FLOPs.

### Training-cost implications

The complete screen used 33 training jobs: 11 effective trained profiles
(reference plus query, encoder, and decoder levels) for each of three seeds.
It evaluated 42 profile/seed combinations because the three selection-count
levels reused each seed's reference checkpoint. Encoder depths 3–5 and decoder
depths 3–5 accounted for 18 of the training jobs. The expanded domain will
raise combinatorial cost when crossed with learning rate and weight decay, so
its budget and deterministic seeds must be frozen before results are observed.

The depth effects justify that cost. Query/select axes would enlarge the
archive without a latency separation that exceeds the measurement noise and
are therefore omitted.

## Integrity and provenance

This report reads the committed artifacts; it does not rewrite measurements or
feed repeat data into a frozen historical selection.

| Evidence | SHA256 |
| --- | --- |
| `sensitivity_latency_analysis.v2.json` (whole file) | `33aea1c13ece0ce632587abd16ed6020ecc88c63220f89891a5f30183322eaea` |
| v2 analysis internal `report_sha256` | `40a8bccb6e43b8238c2cf6b47eaf3253e735d82fd160212d12915b3137a3fa79` |
| `sensitivity_training_accuracy.v1.json` | `459da2ebe557ec26947dc723b2864f2bc31880ae3181ad1216c3a47825ec466b` |
| `sensitivity_training_checkpoints.v1.json` | `20188a8858a9329ce4b861730ad3b0b2f6185389c8af1b02ad29284e5ed1b012` |
| `sensitivity_latency_manifest.v2.json` | `aedc117414b2691c1a70b73fa4e9e0ac123cb4d20dfd9d25dfe2d4aa490d7655` |
| `sensitivity_latency_analysis_erratum.v1.json` | `8e19287bf2ffd674f62b21cdaf11e000b0eae1ed8af9d0ada1238491588993f2` |
| `sensitivity_latency_aggregate_erratum.py` | `9209e748093e0555fe5cba339327a8216744ec9ca6b9dae276c7041703a409c6` |
| Original manifest-pinned aggregator | `5f5aebd4274c746ec9674f28f978af5d228d98c6ba0af8d76cff8b1742dab967` |
| Immutable nine-job submission ledger | `b1c170c0d4697463d171cbeca3e4adcbd34cc1cb7429c236f48b58c46c3b6d54` |
| Verified 1,017-file remote inventory | `a0527c5f687b7660e208a009972cc4c2de5a0f684b1e62316cd7671e9de15021` |
| Pinned SQSH | `88ba75e3a8eb9524fc0dbf026f2ea5da2c68696ae8d918b0afde5e0384ca641e` |
| DINO ResNet50 PTM | `7a391fb84a18714b60258becdb512594ec54faff5dccbf17ca53c5d902137512` |
| Validation annotation | `9b715b689e9a17588805faad26ed94597886d28ac687438dcb778de433f997af` |
| Training annotation | `7401a1245dc0b691c40f8f53cf4f46f9b96a3e0bc3dcfd357de038074acc1994` |

Relevant repository identities are:

- AutoML branch: `rarunachalam/pre-platform-sdk-removal-20260714`;
- measurement launch commit:
  `cb62ef447704b95980b17aa82604992564b4e71f`;
- corrected analysis commit:
  `6472954ec6996f3d7872c6dcb6217f7c3b228a61`;
- committed v2 evidence commit:
  `211d8fd6a5d4e718fdb28a5f57f0483f8bbf4c40`;
- TAO SDK commit:
  `3d3e1adc1849493d29dc926cb99492417e3a9250`;
- TAO skills commit:
  `18f831c7c83b424861a60353fb735dd80efcfded`;
- TAO 7.0.1 dereferenced source commit:
  `1ac00f8e9c511591e6e1cfb048c1bad9101b3d32`.

The nine SLURM allocations each used one eight-GPU A100 node. The SDK database
rows remained `Pending` only because the launch path did not run the SDK
monitor; exact read-only `sacct` reconciliation found every job
`COMPLETED/0:0`. The analysis did not retry, mutate, or reinterpret those jobs.

The artifact policy explicitly records:

```text
winner_selected=false
feeds_final_selection=false
manual_promotion_permitted=false
```

## Exact reproduction

Run the following from the AutoML repository at the recorded branch/commit.
Secrets are loaded only into the process environment and are never printed or
stored in an artifact.

First verify the immutable inputs:

```bash
cd /localhome/local-rarunachalam/tao-automl
git rev-parse HEAD
git merge-base --is-ancestor \
  cb62ef447704b95980b17aa82604992564b4e71f \
  6472954ec6996f3d7872c6dcb6217f7c3b228a61
sha256sum \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_analysis.v2.json \
  experiments/dino_moo_phase2_20260728/sensitivity_training_accuracy.v1.json \
  experiments/dino_moo_phase2_20260728/sensitivity_training_checkpoints.v1.json \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_manifest.v2.json \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_analysis_erratum.v1.json \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_aggregate_erratum.py
```

Regenerate the analysis into a new path so the committed evidence is not
overwritten:

```bash
set -a
source /localhome/local-rarunachalam/.tao/config.env
set +a
python experiments/dino_moo_phase2_20260728/sensitivity_latency_aggregate_erratum.py \
  --analysis-erratum \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_analysis_erratum.v1.json \
  --analysis-erratum-sha256 \
  8e19287bf2ffd674f62b21cdaf11e000b0eae1ed8af9d0ada1238491588993f2 \
  --manifest \
  experiments/dino_moo_phase2_20260728/sensitivity_latency_manifest.v2.json \
  --checkpoint-artifact \
  experiments/dino_moo_phase2_20260728/sensitivity_training_checkpoints.v1.json \
  --checkpoint-artifact-sha256 \
  20188a8858a9329ce4b861730ad3b0b2f6185389c8af1b02ad29284e5ed1b012 \
  --accuracy-artifact \
  experiments/dino_moo_phase2_20260728/sensitivity_training_accuracy.v1.json \
  --accuracy-artifact-sha256 \
  459da2ebe557ec26947dc723b2864f2bc31880ae3181ad1216c3a47825ec466b \
  --submission-ledger \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency_v2/block_submissions.json \
  --submission-ledger-sha256 \
  b1c170c0d4697463d171cbeca3e4adcbd34cc1cb7429c236f48b58c46c3b6d54 \
  --sdk-state \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency_v2/slurm_state.json \
  --evidence-snapshot \
  experiments/dino_moo_phase2_20260728/runtime/sensitivity_latency_v2/evidence_snapshot \
  --output /tmp/sensitivity_latency_analysis.reproduced.json
```

The aggregator revalidates all 1,017 exact remote/local evidence paths, job and
scheduler identities, raw rank-file hashes, and the training-accuracy join.
Wall-clock metadata and local output paths can differ on regeneration; samples,
statistics, qualification decisions, and the six qualified profiles must not.

Run the focused integrity tests:

```bash
python -m pytest -q \
  experiments/dino_moo_phase2_20260728/test_sensitivity_latency_analysis_erratum.py \
  experiments/dino_moo_phase2_20260728/test_sensitivity_latency_runtime_contract.py
```

The report's numeric tables were independently checked by joining every
analysis allocation record to the exact `(profile_id, seed)` mAP50 entry:
126/126 allocation rows matched, yielding 42 unique accuracy observations.
The qualified-profile list read directly from the committed analysis is:

```text
enc_layers_3
enc_layers_4
enc_layers_5
dec_layers_3
dec_layers_4
dec_layers_5
```

No query-count or selection-count profile appears in that list.
