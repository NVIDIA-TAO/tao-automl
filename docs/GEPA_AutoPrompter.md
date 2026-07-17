# TAO GEPA Auto-Prompter

TAO owns the integration layer for reflective prompt optimization. The upstream
Auto-Tuner and VLMEvalKit repositories are read-only references and metric
providers; TAO does not require feature changes in either repository.

## When to use it

Use `GEPAutoPrompter` for a non-train TAO action when the model weights stay
fixed and the goal is to improve one or more text prompt fields, optionally
together with bounded inference-config components. It:

1. launches a batch evaluate action for a candidate prompt;
2. retains aligned per-example scores for GEPA;
3. gives the reflector only the query, generated output, and sanitized feedback;
4. evaluates each accepted candidate on validation with the official set-level
   metric; and
5. runs the selected candidate once on the untouched test set.

For joint optimization, pass `config_choices` to `TAOGEPAAdapter` and include
those component values in the seed candidate. GEPA then owns one candidate pool
over prompt and config values and updates components round-robin. Free-form text
is proposed by the reflector; config proposals remain inside their declared
discrete choices and use score-history UCB. The same minibatch and complete-
validation acceptance gates score the resulting prompt/config pair. Use TAO's
`autoresearch` controller instead for continuous or broad schema-derived HPO.

For an already fine-tuned checkpoint, establish the checkpoint's unchanged
prompt/config score first, then search only on a separate validation role. A
practical bounded composition is:

1. use `autoresearch` or an exhaustive action-native grid to select live
   perception settings such as `vision.nframes`;
2. rerank the complete GEPA prompt pool under the selected setting because
   prompt and perception settings can interact; and
3. freeze the prompt/config pair before launching the untouched test or full
   evaluation.

Cosmos evaluate exposes 4, 8, and 16 evenly sampled frames. Eight remains the
default. Sixteen should be treated as a higher-cost option: the evaluator
currently materializes processed video inputs before inference, so large
corpora may need video-disjoint execution shards even when GPU memory is
sufficient. `vision.fps` is mutually exclusive with `vision.nframes` and is not
a bounded substitute for frame-count sampling on variable-duration videos.

## Installation

```bash
pip install 'nvidia-tao-automl[autoprompter]'
```

The extra pins GEPA to the exact validated upstream commit, `d750388`.

## Action integration

```python
from tao_automl import (
    GEPAutoPrompter,
    GEPAReflectionLM,
    TAOActionBatchRunner,
    TAOGEPAAdapter,
)
from tao_automl.brain.llm_client import LLMClient, LLMConfig
from vlmeval.vss import binary_aggregate, binary_metric


def evaluate_action(specs, items):
    # Launch one TAO Cosmos evaluate job with `specs` and an annotation made
    # from `items`; wait for completion and return predictions in item order.
    return launch_and_read_aligned_predictions(specs, items)


base_specs = {
    "dataset": {"system_prompt": ""},
    "vision": {"nframes": 8},
    "generation": {"max_tokens": 256, "temperature": 0.0},
}
runner = TAOActionBatchRunner(
    base_specs,
    evaluate_action,
    value_coercers={"vision.nframes": int},
)
adapter = TAOGEPAAdapter(runner, binary_metric)
reflection_lm = GEPAReflectionLM(LLMClient(LLMConfig(
    endpoint=llm_endpoint,
    model=llm_model,
    api_key=llm_api_key,
)))
prompter = GEPAutoPrompter(
    adapter,
    reflection_lm=reflection_lm,
    aggregate_metric_fn=binary_aggregate,
    aggregate_metric_key="macro_f1",
    reflection_minibatch_size=16,
    seed=42,
)
result = prompter.optimize(
    {"dataset.system_prompt": seed_prompt},
    train_items,
    validation_items,
    budget=3000,
    testset=test_items,
)
print(result.to_dict())
```

For a vision-capable reflector, pass a platform-owned evidence callback and
mark the prompt components that consume visual input:

```python
from gepa.image import Image

def failure_frames(item, candidate):
    return {
        timestamp: Image(path=path)
        for timestamp, path in sample_candidate_frames(item, candidate).items()
    }

adapter = TAOGEPAAdapter(
    runner,
    binary_metric,
    config_choices={"vision.nframes": [4, 8, 16]},
    reflection_evidence_fn=failure_frames,
    vision_components=["dataset.system_prompt"],
)
```

Evidence is requested only for failed reflection examples. It is attached to
the selected vision components without adding gold answers, item IDs, or media
paths to the rendered reflection record. Use a strong system message on
`GEPAReflectionLM` that permits only reusable visual/temporal strategy and
forbids facts or answers from any particular example. The reflector proposes;
the task metric still decides whether GEPA accepts the proposal.

The joint seed for that adapter must carry both components:

```python
result = prompter.optimize(
    {
        "dataset.system_prompt": seed_prompt,
        "vision.nframes": "8",
    },
    train_items,
    validation_items,
    budget=3000,
    testset=test_items,
)
```

Every config choice is normalized to a string for GEPA and should have a
corresponding `TAOActionBatchRunner.value_coercers` entry when the action spec
requires a numeric value. Fixed settings belong in `fixed_candidate`, not in
`config_choices` or the seed.

`budget` is GEPA's maximum metric-call count, not its number of prompt
proposals. A proposal consumes metric calls for both the incumbent and proposed
prompt on the reflection minibatch; every accepted proposal also consumes a
complete validation pass. If `MaxCandidateProposalsStopper` is configured, set
it high enough that it does not terminate a larger metric budget prematurely,
or omit it and rely on the metric budget plus a wall-time guard. The original
WTS 1,200-call run also had a four-proposal cap, so it was not a controlled
budget comparison with the uncapped 3,000-call run. A later comparison used one
shared trajectory: the best candidate available by call 1,200 scored 64.31% on
all 2,676 records, while the best candidate by call 3,000 scored 65.55%, under
identical final inference settings. The shared search attempted 11 proposals
and admitted four candidates to complete validation.

`evaluate_action` is the platform boundary. It may use Docker, Lepton, SLURM,
Kubernetes, Brev, or a virtual environment, but it must return exactly one
output for every input item in the original order. A missing or extra output
fails the candidate instead of silently corrupting its score.

Candidate keys use dotted TAO spec paths. `TAOActionBatchRunner` deep-copies the
base action spec and applies each candidate value without mutating the base.
Use `value_coercers` when a string candidate must become a typed TAO spec value.

## Metric selection

GEPA requires decomposable per-example feedback while metrics such as Macro-F1
are defined over a complete set. TAO therefore uses the per-item metric for
reflection and GEPA's proposal gate, then reranks every accepted candidate using
`aggregate_metric_fn` over the complete validation set. Ties fall back to the
GEPA proxy score and then the earlier candidate. The test set is not evaluated
until after this selection.

For VANTAGE event verification, pass VLMEvalKit's `binary_metric` and
`binary_aggregate` with `aggregate_metric_key="macro_f1"`. This removes the
previous balanced-accuracy-proxy versus reported-Macro-F1 selection mismatch.

## Data contract

Each item must contain:

- `query`: task input shown to the model and reflector;
- `gold`: used only by the metric; and
- any media/action fields consumed by `evaluate_action`.

Do not put dataset IDs, media paths, or gold answers in metric feedback. TAO
removes common private fields from structured feedback and never copies the
complete item into GEPA's reflection record, but free-form feedback still must
be written without answers.

Use three disjoint roles: reflect on train, select once on validation, and
report once on test. For datasets with multiple questions per clip, group the
split by video when the required claim is generalization to unseen scenes.
