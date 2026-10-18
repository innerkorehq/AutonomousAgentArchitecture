# Plan-first and delegating algorithms

Three ways to spend the model's attention on **planning** and then run the plan with as little model involvement as possible, plus a review option for [`PlanAndExecute`](plan-and-execute.md). As for every [algorithm](custom.md), state is a fold of the run's log, parallel completions cannot change the result, a recorded run replays to the same outcome, and each algorithm tells you its most model calls (`max_calls`) up front.

```python
--8<-- "examples/planning_algorithms.py"
```

```text
rewoo: Paris has 2.1 million people. | evidence {'E1': 'Paris', 'E2': '2.1 million', 'E3': '4'} | model calls: 2
llm-compiler: Paris has 2.1 million people; 2 + 2 = 4. | rounds 2
manager: Report: Paris is the capital of France. | sub-goals ['research', 'write']
plan-and-execute with review: ['gather', 'r1-check']
```

## ReWOO and LLMCompiler: a checked plan of tool calls

`ReWOO(registry, tool_names=None, max_steps=8, on_step_error="fail")` asks the model once for **every** tool call up front:

```json
{"steps": [
  {"id": "E1", "tool": "lookup", "arguments": {"query": "capital of France"}},
  {"id": "E2", "tool": "lookup", "arguments": {"query": "population of #E1"}}
]}
```

`#E1` in an argument means "the result of step E1". Steps whose inputs are ready run in parallel through the normal tool pipeline (approval, timeouts, guards, budgets all apply), and one **solver** call writes the answer from the evidence. Two model calls however many tools run, against ReAct's one per step. The price is that the plan cannot react to a surprising result.

`LLMCompiler(registry, ..., max_replans=1)` runs the same plan, then a **joiner** call decides: answer, or `replan` with feedback. A replan asks for *additional* steps (numbered after the last one, free to use earlier results), at most `max_replans` times; costs `2 * (max_replans + 1)` calls at most. If it still wants to replan when the limit is reached, its answer is used when it gave one, else the run ends `NO_SOLUTION` with the evidence.

What is checked **before any tool runs**, because a plan is model output:

- every tool name must be in `tool_names` (default: the registry); a plan naming anything else is refused, and nothing has run;
- step ids are `E1`, `E2`, ... and unique; a step may refer only to **earlier** steps, so cycles cannot exist;
- arguments must be a JSON object under 20 000 characters; the number of steps is capped by `max_steps`;
- `#E1` substitution is plain string work (a whole-string `#E1` becomes the raw result, a mention inside text becomes the result as text cut at `max_evidence_chars`); nothing is evaluated.

A tool result that is substituted into a later call's arguments is *data in that argument*, and the solver and joiner see results only inside a quoted `<evidence>` block. A failed step ends the run (`on_step_error="fail"`, with a fixed reason) or is recorded as `(step failed)` and the plan goes on (`"continue"`).

Result `{"text", "plan": [{"id", "tool", "arguments"}], "evidence": {id: text}, "rounds"}`.

## HierarchicalAgent: a manager with specialist workers

```python
team = HierarchicalAgent(
    {"researcher": ToolLoop(tools, name="researcher"), "writer": ToolLoop(no_tools, name="writer")},
    descriptions={"researcher": "finds facts", "writer": "writes prose"},
    max_subgoals=6, max_parallel=3, max_depth=3,
    child_budget=Budget(max_steps=50, deadline_s=300),
)
```

The manager writes sub-goals `{id, worker, goal, depends_on}`. Ids, worker names and the dependency graph are validated before any child starts. Ready sub-goals run as **child runs**, at most `max_parallel` at a time, each under `child_budget` (so one worker cannot spend the others' budget) and each seeing only its own goal and the results of the sub-goals it depends on; then one call combines the results (`synthesize=False` returns the last sub-goal's result instead). Cancelling the parent cancels the children.

- **Nesting.** A worker can itself be a `HierarchicalAgent` (give each a distinct `name=`). Every child is told its depth, and a manager at `max_depth` refuses to delegate (the run ends `FAILED`), so a model cannot recurse without limit.
- **Failure.** A failed child fails the run (`on_child_failure="fail"`) or is passed on as `(this sub-goal failed)` (`"continue"`).
- Compared with `PlanAndExecute`, which uses one executor for every step, and `Supervisor`, where the model delegates turn by turn through tool calls, the manager here commits to a validated DAG and picks a specialist per sub-goal.

Result `{"text", "subgoals": [{"id", "worker", "goal", "result"}], "depth"}`.

## PlanAndExecute(review_progress=True)

By default `PlanAndExecute` replans only when a step fails. With `review_progress=True` the planner is also asked, after each finished step that leaves steps pending, whether the **remaining** steps still hold; it returns them unchanged or corrected, and the pending steps are replaced (finished and running steps are kept; new ids are prefixed `r1-`, `r2-`, ...). Each review counts against `max_replans`, so cost stays bounded. A review that fails, is not valid JSON, or is not a valid plan is ignored and the plan carries on (`aaa.plan.kept` is logged); only an invalid plan after a *failure* ends the run.

## Choosing

| You want | Use |
|---|---|
| The fewest model calls, tools known up front | `ReWOO` |
| The same, but able to correct course once or twice | `LLMCompiler` |
| Different specialists for different parts of the work | `HierarchicalAgent` |
| One executor for every step, plan may drift | `PlanAndExecute` (with `review_progress` if it may) |
| The model should decide each next step from what it just saw | [`ToolLoop`](tool-loop-react.md) |
