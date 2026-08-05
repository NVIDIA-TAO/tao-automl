# AutoML algorithm and objective capabilities

TAO AutoML distinguishes final archive selection from the acquisition or
promotion policy that generates candidates. A selector can produce a valid
winner from an archive without making the search algorithm itself natively
multi-objective.

The production source of truth is
`BrainFactory.objective_capability_matrix()`. Registry construction derives the
algorithm names and aliases from `AlgorithmType` and fails when a new factory
algorithm has not been classified.

| Algorithm | Aliases | Accuracy mode | Latency mode | Multi-objective mode |
| --- | --- | --- | --- | --- |
| Bayesian | `bayesian`, `b` | Native raw-accuracy EI | Native constrained latency EI | Native raw-objective ParEGO EI |
| BFBO | `bfbo` | Archive accuracy-score fallback | Scalarized constrained-score fallback | Scalarized Pareto-score fallback |
| Hyperband | `hyperband`, `h` | Archive accuracy-score promotion | Scalarized constrained-score promotion | Scalarized Pareto-score promotion |
| BOHB | `bohb` | Archive accuracy-score search/promotion | Scalarized constrained-score search/promotion | Scalarized Pareto-score search/promotion |
| ASHA | `asha` | Archive accuracy-score promotion | Scalarized constrained-score promotion | Scalarized Pareto-score promotion |
| PBT | `pbt` | Archive accuracy-score exploitation | Scalarized constrained-score exploitation | Scalarized Pareto-score exploitation |
| DEHB | `dehb` | Archive accuracy-score search/promotion | Scalarized constrained-score search/promotion | Scalarized Pareto-score search/promotion |
| Hyperband ES | `hyperband_es`, `hes` | Archive accuracy-score promotion | Scalarized constrained-score promotion | Scalarized Pareto-score promotion |
| LLM | `llm` | Supported scalar accuracy feedback | Unsupported | Unsupported |
| Hybrid | `hybrid` | Supported scalar accuracy feedback | Unsupported | Unsupported |
| Autoresearch | `autoresearch` | Supported scalar accuracy feedback | Unsupported | Unsupported |

## Capability levels

- `native`: the brain consumes and models the requested raw objective values.
- `scalarized_fallback`: the controller converts the current raw archive into
  a deterministic maximize-oriented acquisition score. The brain searches,
  ranks, promotes, or exploits using that score, but does not model accuracy
  and latency independently.
- `unsupported`: the brain lacks the raw objective context required to make the
  requested mode-specific search claim. Construction fails with a clear error.

Accuracy mode is supported broadly because its archive acquisition score is the
raw valid accuracy metric. Agentic algorithms remain unsupported for latency
and multi-objective modes because their prompt and history currently expose
only a scalar result.

For strict product validation of native mode-specific acquisition, use
Bayesian. Numerical fallback algorithms can still test deterministic selection
and promotion behavior, but must be reported as scalarized fallbacks rather
than native multi-objective optimization.

Legacy single-objective searches remain supported by every factory algorithm.
Generic multi-objective configurations without the accuracy/latency archive
selector use the existing weighted-scalar fallback for numerical algorithms
and are rejected for scalar-only agentic algorithms.
