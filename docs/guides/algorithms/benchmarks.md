# Benchmarks: which algorithm to use

Every algorithm in AAA is tested with scripted models, which proves it does what it says but not that it **helps**. This page is the result of running them against real (small, local) models on three kinds of task. Read the caveats first: the numbers are a guide to *shape*, not a ranking to quote.

## What was measured

| | |
|---|---|
| Models | Qwen3-4B-Instruct-2507 (4-bit) and Qwen3.5-2B (8-bit), served locally by [omlx](https://github.com/jundot/omlx) on a 16 GB Apple-silicon laptop |
| Tasks | **Arithmetic**: 18 seeded multi-step word problems. **Tools**: 15 questions over an invented table of cities, answered with `lookup`, `add`, `subtract` and `multiply` tools. **Retrieval**: 12 questions over 12 invented documents (some need two hops) plus 4 that the documents cannot answer, where the right behaviour is to say so. Everything is invented or computed, so a model cannot have memorised it. |
| Runs | Each case twice (`repeats=2`), one case at a time, temperature 0.2 (sampling algorithms set their own) |
| Reported | **Accuracy** (the final answer is right; on unanswerable questions, a refusal is right), mean **model calls**, mean **tokens** and mean **latency** per case, and accuracy per 1,000 tokens |
| Code | `benchmarks/` in the repository; `uv run python -m benchmarks.run --model <id>` repeats it on any OpenAI-compatible server |

## Caveats

* **Small models, small sets, one machine.** 32 to 36 runs per cell: a difference of a few points is noise (one run is about 3 points). Latency depends on this laptop, and on the local server swapping models in and out.
* **Easy arithmetic.** Both models already get these right in one call, so there was little for the extra reasoning to fix. This says nothing about harder problems, where the cost may buy accuracy. Do not read "all 100%" as "these algorithms are equal".
* **Structured output.** The server fell back to best-effort prompting instead of strict JSON-schema decoding during part of the runs (an `xgrammar` component was not installed). Algorithms that parse structured model output (the planners, the graders, `adaptive_rag`, `self_rag`) were affected more than the others, and a different server or a larger model may do better.
* **Prompts are the defaults.** Each algorithm ran with its default prompts. Tuning them (see `aaa.optimize`) would change the ranking.
* Runs lost to the server (connection refused, a model being swapped out) are retried, not counted as failures. Failures of the algorithm itself (an invalid plan, a ReAct loop that never answers) are counted as wrong.

## What the numbers say

**Arithmetic: a plain call is hard to beat on easy problems.** For the 4B model the one-call baseline was already at 100% for about 190 tokens. `self_consistency`, `debate`, `best_of_n`, `least_to_most` and `graph_of_thoughts` also reached 100% but spent 5 to 14 times the tokens. `self_refine`, `tree_of_thoughts` and `chain_of_verification` were at 97%, one miss in 36. `codeact` matched the baseline for ~280 tokens, because the model computes in Python instead of by hand. The smaller 2B model shows the same shape, with `tree_of_thoughts`, `chain_of_verification`, `least_to_most` and `codeact` a little lower (86 to 89%).
*Use the baseline first. Reach for `self_consistency` (the cheapest of the search algorithms: 5 calls, no judge) only when you measure errors, and for `codeact` when the answer is a computation.*

**Tool use: the loop that matches the model's tool-calling wins; planners need a model that can plan.** With the 4B model `react` answered every question (100%) at about 7 calls, `tool_loop` 80%, and the plan-first algorithms did worst: `rewoo` 60%, `llm_compiler` 40%, `plan_and_execute` 33% and by far the most expensive (about 17 calls and 5,400 tokens). Their failures were invalid plans and a mis-wired step (for example "step E2 (multiply) failed"), not wrong arithmetic. With the 2B model the order changed: `rewoo` 87% and `llm_compiler` 80% were best, `tool_loop` 83%, and `react` failed (3%: it ran out of turns, spending 10,000 tokens per case). So the result depends on the model more than on the algorithm.
*Start with `tool_loop`; try `react` if the model has no native tool calling; use `rewoo` or `llm_compiler` when calls are independent and you want fewer model calls (about 4 against 6 to 7), after checking on your own model that it can write the plan. `plan_and_execute` is the costly one: use it for long tasks, not for lookups.*

**Retrieval: how the algorithm decides to retrieve matters more than how it grades.** With 4B, `adaptive_rag` scored 97% for 3.5 calls and 560 tokens; `corrective_rag` 81% for 9 calls (it declines unanswerable questions correctly, at about twice the cost); `rag_agent` 50%, because the model often answered from memory without searching; and `self_rag` 6%, because the model judged that retrieval was not needed and made up an answer. With 2B the order flipped: `rag_agent` 97%, `corrective_rag` 81%, `self_rag` 56%, `adaptive_rag` 12%. The pattern is that **an algorithm that asks the model whether to retrieve inherits that model's habits**: a model that overrates its own knowledge skips retrieval, and a model that over-retrieves does not.
*If your questions always need the documents, `corrective_rag` is the dependable choice (it grades what it found and can refuse); if you use `adaptive_rag` or `self_rag`, measure how often your model chooses to skip retrieval, or force retrieval for everything in your domain.*

## When to use which

| You have | Start with | Add only if measured | Avoid on small models |
|---|---|---|---|
| A reasoning question the model usually gets right | one call | `self_consistency` | multi-stage pipelines (`least_to_most`, `graph_of_thoughts`, `debate`): 5 to 14 times the tokens for no gain here |
| A question that is really a computation | `codeact` | `self_consistency` | |
| Tools with a native tool-calling model | `tool_loop` | `react` for non-native models | `plan_and_execute` for simple lookups |
| Independent tool calls, cost matters | `rewoo`, `llm_compiler` | check plan validity on your model first | |
| Questions that need your documents | `corrective_rag` | `adaptive_rag` if you measured that it retrieves | `self_rag` where the model skips retrieval |

## Per-algorithm results

Sorted by accuracy, then by tokens. "Accuracy per 1k tokens" is a rough value-for-money figure, not a quality score.

### Qwen3-4B-Instruct-2507 (4-bit)

#### Arithmetic word problems (36 runs per algorithm)

| Algorithm | Accuracy | Model calls | Tokens | Latency (s) | Accuracy per 1k tokens | What it does |
|---|---:|---:|---:|---:|---:|---|
| `direct` | 100% | 1.0 | 192 | 3.3 | 5.20 | one call, think step by step |
| `codeact` | 100% | 1.0 | 277 | 3.8 | 3.62 | writes and runs Python |
| `self_consistency` | 100% | 5.0 | 979 | 6.8 | 1.02 | 5 samples, majority vote |
| `lats` | 100% | 7.2 | 1804 | 14.4 | 0.55 | tree search, 3 iterations |
| `debate` | 100% | 7.0 | 1814 | 26.5 | 0.55 | 3 agents, 1 round, judge |
| `least_to_most` | 100% | 7.2 | 2357 | 20.4 | 0.42 | sub-questions first, then the answer |
| `best_of_n` | 100% | 10.0 | 2399 | 9.6 | 0.42 | 5 samples, model judge picks |
| `graph_of_thoughts` | 100% | 11.0 | 2714 | 14.1 | 0.37 | default generate/score/aggregate graph |
| `self_refine` | 97% | 2.6 | 968 | 13.4 | 1.00 | draft, critique, rewrite |
| `tree_of_thoughts` | 97% | 4.0 | 985 | 5.6 | 0.99 | depth 2, width 2 |
| `chain_of_verification` | 97% | 6.0 | 1254 | 14.9 | 0.78 | draft, check questions, revise |

#### Tool use over a lookup table (30 runs per algorithm)

| Algorithm | Accuracy | Model calls | Tokens | Latency (s) | Accuracy per 1k tokens | What it does |
|---|---:|---:|---:|---:|---:|---|
| `react` | 100% | 7.4 | 1940 | 7.8 | 0.52 | text Thought/Action/Observation |
| `tool_loop` | 80% | 5.1 | 1340 | 3.0 | 0.60 | native tool calling |
| `rewoo` | 60% | 5.4 | 751 | 4.1 | 0.80 | plan once with placeholders, run, solve |
| `llm_compiler` | 40% | 6.1 | 1104 | 4.8 | 0.36 | plan a DAG, run in parallel |
| `plan_and_execute` | 33% | 17.0 | 5438 | 12.7 | 0.06 | plan, run each step with a tool loop |

#### Retrieval-grounded questions (32 runs per algorithm)

| Algorithm | Accuracy | Model calls | Tokens | Latency (s) | Accuracy per 1k tokens | What it does |
|---|---:|---:|---:|---:|---:|---|
| `adaptive_rag` | 97% | 3.5 | 560 | 2.7 | 1.73 | classify, then pick the retrieval depth |
| `corrective_rag` | 81% | 9.2 | 1317 | 9.0 | 0.62 | grade chunks, rewrite, answer |
| `rag_agent` | 50% | 1.6 | 394 | 2.3 | 1.27 | retrieve with a tool loop |
| `self_rag` | 6% | 2.0 | 211 | 1.5 | 0.30 | reflect on support and usefulness |

### Qwen3.5-2B (8-bit)

#### Arithmetic word problems (36 runs per algorithm)

| Algorithm | Accuracy | Model calls | Tokens | Latency (s) | Accuracy per 1k tokens | What it does |
|---|---:|---:|---:|---:|---:|---|
| `direct` | 100% | 1.0 | 329 | 5.8 | 3.04 | one call, think step by step |
| `self_refine` | 100% | 2.7 | 1178 | 11.2 | 0.85 | draft, critique, rewrite |
| `self_consistency` | 100% | 5.0 | 1824 | 10.4 | 0.55 | 5 samples, majority vote |
| `debate` | 100% | 7.0 | 2786 | 24.0 | 0.36 | 3 agents, 1 round, judge |
| `best_of_n` | 100% | 10.0 | 3064 | 9.1 | 0.33 | 5 samples, model judge picks |
| `lats` | 94% | 8.2 | 2875 | 9.5 | 0.33 | tree search, 3 iterations |
| `graph_of_thoughts` | 92% | 11.0 | 3425 | 16.5 | 0.27 | default generate/score/aggregate graph |
| `tree_of_thoughts` | 89% | 4.1 | 1190 | 5.3 | 0.75 | depth 2, width 2 |
| `chain_of_verification` | 89% | 5.9 | 1554 | 10.8 | 0.57 | draft, check questions, revise |
| `least_to_most` | 89% | 6.9 | 2819 | 14.3 | 0.32 | sub-questions first, then the answer |
| `codeact` | 86% | 1.0 | 290 | 3.3 | 2.96 | writes and runs Python |

#### Tool use over a lookup table (30 runs per algorithm)

| Algorithm | Accuracy | Model calls | Tokens | Latency (s) | Accuracy per 1k tokens | What it does |
|---|---:|---:|---:|---:|---:|---|
| `rewoo` | 87% | 4.2 | 854 | 4.4 | 1.02 | plan once with placeholders, run, solve |
| `tool_loop` | 83% | 6.3 | 2144 | 3.7 | 0.39 | native tool calling |
| `llm_compiler` | 80% | 4.1 | 1103 | 6.0 | 0.73 | plan a DAG, run in parallel |
| `plan_and_execute` | 37% | 13.2 | 5424 | 14.1 | 0.07 | plan, run each step with a tool loop |
| `react` | 3% | 7.3 | 10660 | 46.4 | 0.00 | text Thought/Action/Observation |

#### Retrieval-grounded questions (32 runs per algorithm)

| Algorithm | Accuracy | Model calls | Tokens | Latency (s) | Accuracy per 1k tokens | What it does |
|---|---:|---:|---:|---:|---:|---|
| `rag_agent` | 97% | 3.0 | 984 | 1.9 | 0.98 | retrieve with a tool loop |
| `corrective_rag` | 81% | 6.9 | 1104 | 5.1 | 0.74 | grade chunks, rewrite, answer |
| `self_rag` | 56% | 5.9 | 1128 | 3.4 | 0.50 | reflect on support and usefulness |
| `adaptive_rag` | 12% | 2.0 | 328 | 1.7 | 0.38 | classify, then pick the retrieval depth |


## Run it on your own model

```bash
uv run python -m benchmarks.run --model <model id> --base-url http://localhost:8001/v1 --repeats 2 --concurrency 1
uv run python -m benchmarks.report
```

Use `--concurrency 1` with a local server. The tasks, the algorithm lineup and the metrics are in `benchmarks/tasks.py` and `benchmarks/families.py`; to benchmark your own task, add a `Dataset` and a metric there (the harness is `aaa.eval`, see [Evaluation](../evals.md)).
