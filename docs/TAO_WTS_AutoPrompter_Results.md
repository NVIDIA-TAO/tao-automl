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

## Remaining production validation

WTS uses accuracy and this run tuned only the prompt. Production acceptance for
Metropolis event verification still requires the full VANTAGE corpus, a
video-disjoint train/validation/test protocol where applicable, official
validation/test Macro-F1, and model-level VK plus Alerts microservice AB results
reported side by side. The TAO aggregate reranker now selects on true validation
Macro-F1; executing that benchmark still requires the VANTAGE data and Alerts
evaluation environment.
