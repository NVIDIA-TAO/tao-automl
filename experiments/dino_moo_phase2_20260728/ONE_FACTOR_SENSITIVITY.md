# DINO one-factor latency-sensitivity preregistration

This package is separate from the phase-2 matched-latency replay harness. It
tests four compatible DINO axes around the phase-1 algorithm-selected accuracy
winner, with all other model, data, training, hardware, and benchmark controls
frozen.

The study is validation-only: `feeds_final_selection=false`. Its deterministic
promotion rule can identify levels worth admitting to a future AutoML search,
but it cannot select or replace an AutoML winner. Results never change profile
generation, job order, promotion thresholds, or submission order.

## Frozen design

- Reference: `seed_271828_rec_5` model/optimizer profile.
- Axes: `model.num_queries`, `model.enc_layers`, `model.dec_layers`, and
  postprocess-only `model.num_select`.
- Seeds: `1234`, `271828`, and `314159`.
- Unique profiles: 14; full training jobs: 33; evaluation/benchmark profile
  instances: 42.
- Every full training spec sets `train.activation_checkpoint=false`.
- Every job uses one node and all eight A100 GPUs through the pinned TAO SDK
  SLURM backend and the pinned TAO 7.0.1 SQSH image.
- Each non-reference result is compared only with the same seed's reference.
- Same-seed 98% accuracy retention is reported as a separate annotation for
  constrained-latency suitability. It does not qualify or disqualify an axis
  for the shared multi-objective search.
- Each latency level must have all nine valid measurements: three matched
  allocations for each of the three training seeds.
- A latency effect qualifies only when its seed-stratified hierarchical 95%
  confidence interval lies wholly outside the practical-equivalence band.
  Both reliably faster and reliably slower effects establish that an axis
  changes inference cost; the direction is recorded separately.
- The effective noise floor is the maximum of the historical
  `0.73553775 ms` floor and a three-independent-allocation reference
  recalibration.
- If any level on an architecture axis qualifies, the future shared search
  admits that axis's complete preregistered domain. It does not fit a smaller
  range after seeing which level happened to qualify.

Unsupported, coupled, training-only, and deployment-resolution axes are listed
explicitly in the manifest and cannot enter profile generation.

The authoritative latency protocol and qualification rules are frozen in
`sensitivity_latency_manifest.v2.json`. Its predecessor v1 stopped at runtime
preflight and produced no latency measurements.

## Dry run

The launcher is fail-closed and dry-runs when no mode flag is supplied:

```bash
cd /localhome/local-rarunachalam/tao-automl
python experiments/dino_moo_phase2_20260728/one_factor_sensitivity_launcher.py \
  --dry-run \
  --report /tmp/dino_one_factor_sensitivity_plan.json
```

The report contains the complete resolved model mapping, its SHA256 digest, the
full resolved train-spec digest, seed, execution class, and checkpoint reuse
source for every profile instance. It contains no secret values.

## Submission guard

No jobs were submitted while creating this preregistration. A later authorized
launch must supply both `--submit-training` and the exact acknowledgement
frozen in the manifest:

```bash
python experiments/dino_moo_phase2_20260728/one_factor_sensitivity_launcher.py \
  --submit-training \
  --verify-remote \
  --acknowledgement USER_AUTHORIZED_CONCURRENT_8GPU_SLURM_DINO_ONE_FACTOR_20260728
```

This queues the complete deterministic 33-job training plan. It does not wait,
evaluate, benchmark, promote levels, or pick winners. Evaluation and matched
latency stages must replay each entry's full resolved model spec and verify its
digest before loading its checkpoint.
