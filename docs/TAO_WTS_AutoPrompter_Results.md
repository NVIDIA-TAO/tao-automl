# Full WTS Auto-Prompter integration result

Run date: 2026-07-15. This experiment exercised exact GEPA prompt evolution
through real TAO Cosmos `evaluate` jobs with fixed model weights and inference
settings. It validates the batch-action contract now owned by
`tao_automl.gepa_autoprompter`; it is not a VANTAGE benchmark.

## Before and after

All results use normalized exact-match accuracy. The final evaluation contains
all 2,676 WTS QA records from 171 videos; no evaluation subset or limit was
used.

| Prompt | Selection validation (530) | Held-out questions (1,070) | Full corpus (2,676) |
|---|---:|---:|---:|
| Default | 298 / 530 (56.23%) | 579 / 1,070 (54.11%) | 1,450 / 2,676 (54.19%) |
| Best of four hand-written prompts | 303 / 530 (57.17%) | 592 / 1,070 (55.33%) | 1,479 / 2,676 (55.27%) |
| GEPA Auto-Prompter | **310 / 530 (58.49%)** | **640 / 1,070 (59.81%)** | **1,574 / 2,676 (58.82%)** |

Compared with the default prompt, Auto-Prompter gained 5.70 percentage points
on held-out questions and 4.63 points over the full corpus. Compared with the
best hand-written prompt, it gained 4.49 and 3.55 points, respectively. Across
the complete corpus it corrected 229 baseline errors and regressed 105 baseline
correct answers, a net gain of 124.

## Controlled settings

- Dataset SHA-256:
  `f828a63f1bbdd45197e1f3393fb94f76ebfdfc785402617aa8c1397b0b47c555`.
- Question split, seed 42: 1,076 train, 530 validation, 1,070 test.
- Target: local Cosmos3 Nano VLM in
  `nvcr.io/nvstaging/tao/tao-toolkit:7.0.1-cosmos-rl-fix`, one A100 80 GB.
- Fixed config: 8 frames, 256 output tokens, temperature 0, seed 1.
- GEPA: commit `d750388`, budget 1,200, reflection minibatch 16, maximum four
  proposals; Nemotron 3 Super V3 reflector.
- Reflection: query, generated output, and generic failure mode only; no gold
  answer or dataset/media identifier.

Record IDs are disjoint, but questions from the same videos occur across split
roles. The held-out result therefore measures unseen questions, not unseen
scenes. One of 530 validation outputs changed between candidate selection and
the later full rerun despite temperature zero: selection scored 310/530 and the
full-run validation slice scored 309/530.

Local artifacts are under
`/localhome/local-rarunachalam/tao_automl_runs/cosmos3_wts_eval_autoprompt_full/gepa_run_20260715_002721/`.

## Fine-tuned checkpoint follow-up

A second experiment tested whether Auto-Prompter adds value after supervised
fine-tuning. TAO job `cb3491f4-052c-4e12-bb67-bb74d01f9f3b` trained Cosmos3
Nano for one epoch on all 5,555 WTS training records from 341 videos. Slurm job
`29072640` completed on eight GPUs in 38 minutes 12 seconds. The 171 evaluation
videos have zero overlap with the training videos.

Prompt development used a video-disjoint split: 1,079 records / 68 videos for
reflection, 550 / 35 for validation selection, and 1,047 / 68 for untouched
test. The expanded GEPA run allowed equal minibatch candidates to reach complete
validation so a high-accuracy seed could be challenged fairly.

| Fine-tuned checkpoint candidate | Validation (550) | Video-disjoint test (1,047) | Full eval (2,676) |
|---|---:|---:|---:|
| Seed prompt, 8 frames | 506 / 550 (92.00%) | 970 / 1,047 (92.65%) | 2,489 / 2,676 (93.01%) |
| TAO prompt/config selected, 16 frames | **507 / 550 (92.18%)** | **971 / 1,047 (92.74%)** | **2,493 / 2,676 (93.16%)** |

Four generated prompts reached full validation. Two tied the seed at 506/550
and two scored 505/550, so TAO correctly retained the default prompt. The
initial prompt-only incremental gain was therefore 0.00 percentage points.

The follow-up then exercised the config half of the original Auto-Prompter
recipe. Complete validation scored 504/550 at 4 frames, 506/550 at 8, 507/550 at
16, and 506/550 at 32. All four GEPA prompts were reranked at the winning
16-frame setting and scored 505, 505, 506, and 505, so the candidate was frozen
as the seed prompt plus 16 frames before test. It added **0.10 percentage point**
on untouched test (three corrections, two regressions) and **0.15 point** over
the complete eval (six corrections, two regressions).

For context, the base checkpoint scored 1,456/2,676 (54.41%) under the same
full-eval family, so full-data SFT itself supplied a 38.60-point gain. The
post-SFT optimizer recovered a smaller additional gain by increasing visual
coverage; test corrections were all upper/lower-body clothing-color questions.
This demonstrates compatibility and measurable incremental value without
misrepresenting the original zero-shot VANTAGE lift as a post-SFT result.

Artifacts are under
`/localhome/local-rarunachalam/tao_automl_runs/cosmos3_wts_sft_autoprompt/`.
The frozen/full comparison is in
`sft_frame_policy_20260716/final_comparison.json`; complete predictions and
per-shard TAO logs are in
`sft_candidate_selected_seed_nframes16_full_sharded_20260716_005604/`.

The runtime tradeoff is material: comparable 8-frame validation jobs took
319-347 seconds, while 16 frames took 855 seconds and 32 took 1,160 seconds.
FPS-based full validation and one monolithic 16-frame full action exhausted host
memory because the evaluator materializes processed inputs. The successful full
result used eight video-disjoint execution shards and rejected missing or
duplicate prediction IDs before scoring.

## Remaining production validation

WTS uses accuracy and this run tuned prompt plus evenly sampled frame count.
Production acceptance for Metropolis event verification still requires the
full VANTAGE corpus, a
video-disjoint train/validation/test protocol where applicable, official
validation/test Macro-F1, and model-level VK plus Alerts microservice AB results
reported side by side. The TAO aggregate reranker now selects on true validation
Macro-F1; executing that benchmark still requires the VANTAGE data and Alerts
evaluation environment.
