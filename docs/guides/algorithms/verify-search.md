# Verification, decomposition and search

Seven ways to spend more model calls on one question. Like every [algorithm](custom.md), each is a fold of the run's log: parallel calls may finish in any order without changing the result, a recorded run replays to the same outcome, and every one is bounded at construction (`max_calls` tells you the most model calls a run can make) and by the run's [`Budget`](../../concepts/runs-and-the-engine.md). Text from the task, drafts and candidates is shown to the model as quoted data (`<draft>...</draft>`), never as instructions, and the closing tag cannot be forged from inside it.

```python
--8<-- "examples/verification_algorithms.py"
```

```text
self-refine: 42 km (units added) after 1 rewrite; accepted
verification: The capital of Australia is Canberra. | draft was: The capital of Australia is Sydney.
best-of-n: the longest answer of all | scores [5.0, 10.0, 15.0, 25.0]
least-to-most: 12 via ['Legs per dog?', 'How many dogs?']
self-ask: 1969. after 1 follow-up
graph-of-thoughts: a+b from 11 thoughts; 9 calls at most
lats: 24 | path ['step 1: 8 / (3 - 8 / 3)', 'FINAL: 24'] | solved True
```

## Checking and improving a draft

**`SelfRefine(max_rounds=3)`**: one model drafts, critiques its own draft (`{"ok", "feedback"}`), and rewrites, until the critique accepts, a rewrite changes nothing, or `max_rounds` rewrites are made. It is cheaper than [`GeneratorCritic`](refine.md) (no second model) and weaker for the same reason: the critic shares the writer's blind spots. Result `{"answer", "rounds", "stopped"}` with `stopped` one of `accepted`, `unchanged`, `max_rounds`, `unchecked` (a critique call failed; the draft is returned as it is). Up to `2 * max_rounds + 2` calls.

**`ChainOfVerification(max_questions=5)`**: draft, plan up to `max_questions` fact-checking questions, answer each **without seeing the draft** (in parallel), then revise the draft against those answers. Answering blind is the point: a claim the draft got wrong is not repeated by the check. Result `{"answer", "draft", "checks": [{"question", "answer"}], "revised"}`. If no questions come back the draft is returned unchanged.

**`BestOfN(n, score=None, verifier_tool=None, threshold=None)`**: `n` samples in parallel, each scored, the best returned. The verifier is one of

- `score=fn(task, candidate) -> float`: pure and free of I/O, because it runs inside the algorithm (no extra model calls);
- `verifier_tool="name"`: a tool called with `{"task", "candidate"}` that returns a number or `{"score": number}`, so approval, timeouts and guards apply as for any tool;
- neither: a separate model call rates each candidate 0 to 10 (`judge_model=`).

With `threshold`, the **lowest-numbered** candidate reaching it is returned as soon as every earlier candidate has been scored, so the answer never depends on which call finished first. A candidate whose scoring fails is passed over; if none can be scored the run fails. Result `{"answer", "score", "scores", "selected", "n", "early_stop"}`. [`SelfConsistency`](ensembles-rag.md) remains the voting algorithm for exact answers.

## Breaking a question down

**`LeastToMost(max_subquestions=6)`**: the model splits the question into at most `max_subquestions` easier sub-questions, which are answered in order (each seeing the earlier answers); the last step is the original question. An unusable split falls back to answering the question directly. Result `{"answer", "steps": [{"question", "answer"}]}`. Much cheaper than [`PlanAndExecute`](plan-and-execute.md): no executor runs and no plan object exists.

**`SelfAsk(max_followups=4, lookup_tool=None)`**: the model answers or writes `Follow up: <question>`; the follow-up is answered by a separate call or, with `lookup_tool`, by that tool called with `{"query": question}` (a search tool is the natural fit). After `max_followups` follow-ups the model must answer. A reply in neither form is taken as the final answer. Result `{"answer", "followups", "turns"}`.

## Searching over thoughts

**`GraphOfThoughts(ops=None)`**: a tree of thoughts can only branch; a graph can also merge and refine. You give a list of `Op(id, kind, inputs, n, instruction)`:

| kind | does |
|---|---|
| `generate` | writes `n` new thoughts (from the task, or building on its `inputs`) |
| `aggregate` | merges all input thoughts into one |
| `refine` | improves each input thought (a failed refinement keeps the original) |
| `score` | rates each input thought 0 to 10 |
| `keep_best` | keeps the `n` highest scored (code, no model call) |

Operations run in the listed order, calls inside one operation in parallel. The graph is validated at construction (unique ids, inputs defined earlier, at most 16 operations, 30 thoughts per operation, 200 calls) and `max_calls` follows from its structure. The default graph is generate 4, score, keep 2, merge, refine, score, keep 1. Result `{"answer", "score", "thoughts", "operations"}`.

**`LATS(iterations=6, branching=3, depth_limit=5)`**: Monte Carlo tree search over reasoning steps. Each iteration picks the most promising node that can still be expanded (UCT with `exploration`), writes `branching` next steps from it, scores them, and backs the scores up. A step starting with `FINAL:` ends a path; a finished path scoring below `reflect_below` gets a one-line **lesson** that later expansions see. It stops when a finished path reaches `solved_score`, when `iterations` run out, or when nothing can be expanded. Identical sibling steps count once. Result `{"answer", "score", "path", "solved", "iterations", "explored", "reflections"}`; with no finished path the outcome is `NO_SOLUTION` carrying the best partial path. At most `3 * iterations * branching` calls.

## Choosing

- You can write the check as code or a tool: `BestOfN` with `score=` / `verifier_tool=`, or [`GenerateTestRepair`](refine.md).
- You cannot, and the failure is a wrong fact: `ChainOfVerification`.
- A single answer that is probably fine but could be polished: `SelfRefine`.
- Multi-step questions where later steps depend on earlier answers: `LeastToMost`, or `SelfAsk` if the model should decide what to look up.
- Problems where partial solutions can be scored and combined: `GraphOfThoughts`; where they need exploring with backtracking: `LATS` (or the cheaper [`TreeOfThoughts`](reasoning.md)).

None of these makes a model correct; they trade calls for fewer mistakes, and a model judge or critic is still a model.
