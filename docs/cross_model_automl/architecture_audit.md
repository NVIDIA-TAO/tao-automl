# Cross-model objective-aware AutoML architecture audit

Status: phase-1 audit, frozen before PTM downloads, dataset downloads, or GPU
campaigns.

Source commit: `6d64bd44c34f3dc51e09b40bacd4aed3491067ac`

## Post-audit implementation status

The findings below describe the audited source commit, not the subsequent
working-tree implementation. The corrective implementation now:

* routes Bayesian accuracy, constrained-latency, and multi-objective jobs to
  raw-objective acquisition rather than an archive-selector proxy;
* uses separate accuracy and latency response surfaces for constrained
  latency and deterministic changing-weight ParEGO for multi-objective search;
* persists acquisition counters, RNG state, design points, and one immutable
  issuance audit per recommendation before returning it to the launcher;
* fails unsupported algorithm/mode combinations explicitly and labels
  numerical scalarized alternatives as non-native fallbacks;
* prevents failed or non-finite trials from entering multi-fidelity promotion
  and exploitation decisions;
* provides a versioned repository-owned PTM registry, exact-member
  checksum-aware preflight, and a non-ordinal hierarchical PTM scheduler;
* injects the previously ignored `base_checkpoint` through the skill-declared
  spec input and rejects ambiguous or conflicting checkpoint configuration.

These changes are implementation evidence only. They do not make the DINO
pilot or any cross-model campaign pass; those stages remain gated on
end-to-end PTM integration and local model/dataset preflight.

## Executive finding

At the audited source commit, the production selector was mode-separated and
Pareto-safe, but the recommendation layer was not yet a native constrained or
multi-objective optimizer.

The current controller recomputes one scalar acquisition utility after every
successful observation:

* accuracy mode uses observed accuracy;
* latency mode uses normalized latency utility for candidates that pass the
  internally resolved accuracy constraint and a constraint-violation penalty
  otherwise;
* multi-objective mode uses the negative sum of Pareto rank and normalized
  augmented-Chebyshev regret.

Bayesian and BFBO fit one scalar surrogate to that changing archive-relative
utility. They do not fit separate accuracy and latency response surfaces, do
not use constrained expected improvement, and do not use Pareto-aware
hypervolume acquisition. The previous DINO campaign therefore proves
algorithmic candidate generation and deterministic archive selection, but it
does not prove three independent, natively objective-aware searches.

The selector should not be redesigned based on this finding. The correctness
and product gaps are in recommendation, PTM resolution, failure handling, and
resume/audit fidelity.

## Recommendation and objective flow

1. `AutoMLRunner` builds a schema-derived, fixed-dimensional search space.
2. `BrainFactory` constructs the configured search algorithm.
3. The brain proposes a candidate from prior recommendation results.
4. Training returns the configured metric. Multi-objective jobs additionally
   collect structured accuracy and latency objective values.
5. The controller calls archive analysis after each successful result and
   writes the mode-specific scalar acquisition utility back to the
   recommendation result.
6. The next recommendation is generated from that scalar history.
7. At terminal selection, accuracy, latency, and multi-objective policies route
   through the production selector independently.

The final selection is not merely enumeration-order based: canonical
specification fingerprints and candidate IDs provide deterministic final
tie-breaking. Recommendation order remains history-dependent by design.

## Supported algorithms and compatibility at the audited commit

“Partial” means that the algorithm can consume the current mode-specific scalar
proxy, but does not model the stated constrained or multi-objective problem
natively.

| AutoML algorithm | Accuracy objective | Latency objective | Constrained latency | Native multi-objective acquisition | Categorical PTM variable | Conditional spaces | Resume deterministic | Required changes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bayesian | EI on scalar accuracy utility | Scalar proxy | Partial | No | Ordered-bin encoding only | Relational numeric bounds only | No | Separate objective/constraint models, categorical encoding, persisted RNG/acquisition state |
| BFBO | UCB on scalar utility | Scalar proxy | Partial | No | Ordered-bin encoding only | Relational numeric bounds only | No | Same as Bayesian |
| HyperBand | Promotion plus random configurations | Direction defect for raw latency | Scalar promotions only | No | Random initial coverage | Relational numeric bounds only | No | Generic direction and failure-safe promotion |
| HyperBandES | HyperBand plus curve stopping | Direction defect for raw latency | Scalar promotions only | No | Random initial coverage | Relational numeric bounds only | No | Generic direction and failure-safe promotion |
| BOHB | TPE-like scalar modelling for floats | Direction defect for raw latency | Partial | No | Non-floats collapse to one encoded value | Relational numeric bounds only | No | Mixed-variable encoding, direction, and failure-safe promotion |
| ASHA | Promotion plus random configurations | Direction defect for raw latency | Scalar promotions only | No | Random initial coverage | Relational numeric bounds only | No | Generic direction handling |
| PBT | Scalar exploit/explore | Direction defect for raw latency | Partial | No | Unsafe when PTM changes while resuming another member | Relational numeric bounds only | No | Direction/failure handling; PTM-change prohibition or explicit support |
| DEHB | Scalar DE plus promotions | Direction defect for raw latency | Partial | No | Categories collapse in normalized DE representation | Relational numeric bounds only | No | Mixed-variable representation, direction, and failure-safe promotion |
| LLM | Best-effort scalar prompt | Direction defect for raw latency | Scalar prompt only | No | Schema proposal only; no PTM contract | Prompt validation only | No | Raw objectives/policy in prompt, provider replay contract |
| Hybrid | Depends on active sub-brain | Direction defect for raw latency | Partial | No | Depends on sub-brain | Subset/range edits, not conditional topology | No | Raw objective context, direction, persisted sub-brain/RNG state |
| Autoresearch | Scalar LLM reasoning | Direction defect for raw latency | Scalar reasoning only | No | Schema proposal only; no PTM contract | Prompt validation only | No | Raw objective context and deterministic replay support |

## Precise answer: can Bayesian search learn latency?

Yes, but with an important limitation.

When the frozen search space contains parameters that affect inference cost,
the Bayesian brain receives different scalar feedback when their measured
latencies differ. It can therefore learn to favor latency-improving values
without an agent selecting individual candidates. It learns a discontinuous,
archive-relative scalar proxy, however, rather than independent accuracy and
latency surfaces. The feasible boundary also moves whenever a better accuracy
reference is observed.

The previous expanded DINO candidate `seed_271828_rec_19` was generated by the
algorithm and selected without candidate injection or winner override. The
encoder- and decoder-depth axes used by that campaign were introduced after a
separate latency-sensitivity study, and the three mode winners were selected
from one shared multi-objective archive. That evidence does not establish that
the default unfamiliar-dataset workflow autonomously discovers latency-relevant
dimensions, nor that three independent mode-specific jobs explore different
regions.

## Search-space representation

The schema layer supports fixed-dimensional integer, float, ordered integer,
boolean, string, categorical, list, collection, and dictionary values.
`depends_on` can adjust a numeric bound using an already generated parent. It
does not activate or deactivate PTM-specific branches and cannot express
variable-dimensional hierarchical spaces.

Current target-model schema inspection found:

| Model identifier | Default searchable dimensions | Clear inference-cost dimensions |
| --- | ---: | --- |
| `dino` | 19 | query count, encoder layers, decoder layers, selected queries |
| `deformable_detr` | 18 | query count, encoder layers, decoder layers, selected queries |
| `rtdetr` | 17 | query count, encoder layers, decoder layers, selected queries |
| `grounding_dino` | 15 | encoder layers, decoder layers, selected queries |
| `segformer` | 18 | No clear architecture-capacity axis in the default space |
| `oneformer` | 9 | train/test input size |
| `mask2former` | 12 | object queries, hidden dimension, decoder layers, input size |
| `mask_grounding_dino` | 13 | selected queries; default depth is fixed |

`grounding_dino` and `mask_grounding_dino` expose a selected-query dependency
whose parent query-count parameter is absent from the default selected list.
That topology must fail preflight or be completed explicitly before search.

## Failure and infeasibility handling

Terminal selection already rejects unsuccessful candidates and non-finite or
missing objectives.

Acquisition and promotion are inconsistent:

* Bayesian, BFBO, and ASHA exclude failed observations;
* HyperBand, BOHB, and DEHB can sort failed recommendations in promotion
  windows;
* PBT can treat a failed member as complete and copy or rank its result.

Failed recommendations commonly retain result `0.0`. Valid multi-objective
scalar utilities are non-positive, so a failed `0.0` can rank above every valid
candidate. This is a production correctness blocker for multi-fidelity
campaigns and must be fixed before any DINO pilot.

Failed recommendations must remain immutable failed records. They may not be
silently replaced to achieve a requested count of successful candidates.

## PTM architecture gap

No repository-owned checkpoint registry, checksum-aware resolver, access
preflight, checkpoint-YAML merge stage, compatibility exclusion contract, or
categorical PTM search integration exists.

`AutoMLRunner.run(base_checkpoint=...)` accepts and documents a checkpoint but
does not consume it. The SDK runtime can download a complete NGC model version
and substitute the first checkpoint-like file it finds, but it does not:

* select a declared member deterministically;
* verify a checksum;
* load checkpoint-specific YAML;
* validate TAO, task, architecture, or input compatibility;
* smoke-load every checkpoint;
* return structured exclusion reasons.

PTM identity must not be represented as an ordinal scalar in the current GP.
Until mixed categorical kernels and inactive conditional masks exist, the safe
design is hierarchical or staged:

1. resolve and smoke-test every registered PTM;
2. allocate an equal-fidelity deterministic initial quota per compatible PTM;
3. model or optimize within each PTM-specific conditional space;
4. allocate subsequent budget algorithmically across PTM arms;
5. combine only directly comparable objective records in the terminal archive.

## Reproducibility and recommendation audit

Seeds are process-stable when a job starts from scratch. The controller writes
recommendations and result state atomically, and terminal winner selection is
archive-order invariant.

Exact interrupted-run replay is not yet guaranteed:

* Python and NumPy RNG state are not persisted;
* surrogate/acquisition state is reconstructed rather than recorded;
* Hybrid does not persist all sub-brain state;
* provider-backed LLM recommendations are externally nondeterministic;
* the state-store lock explicitly does not provide an NFS guarantee.

Every future campaign must preserve a recommendation-time ledger containing:

* algorithm, mode, search seed, and source commit;
* complete frozen search space and hash;
* ordered observation snapshot visible at recommendation time;
* raw objectives and constraint state exposed to acquisition;
* acquisition function and configuration;
* generated parameter values and canonical fingerprint;
* PTM arm and conditional-space identity;
* immutable intervention flags.

Required intervention flags, all initialized to false:

```text
agent_selected_candidate = false
agent_injected_candidate = false
agent_modified_search_space_after_results = false
agent_changed_seed_after_results = false
agent_changed_budget_after_results = false
agent_changed_threshold_after_results = false
agent_changed_ptm_after_results = false
agent_overrode_winner = false
```

## Required implementation gate

The DINO three-job pilot is blocked until all of the following pass:

1. repository-owned PTM registry and checksum-aware preflight;
2. an explicit supported objective-aware acquisition contract;
3. failure-safe promotion/exploitation for every enabled multi-fidelity brain;
4. generic optimization-direction handling;
5. recommendation-time audit records and deterministic resume/replay tests;
6. categorical PTM and conditional-space capability gates;
7. a deterministic exploration/calibration stage before latency constraints
   steer acquisition;
8. local dataset and model preflight, including a complete epoch and reload.

No SLURM work was launched during this audit.
