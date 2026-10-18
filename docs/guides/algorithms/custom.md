# Writing your own algorithm

An algorithm is a small object with three members, and **no state of its own**:

```python
class MyAlgorithm:
    name = "my_algorithm"  # unique; how runs are routed to you

    def subscriptions(self) -> list[str]:  # event-type globs you react to
        return ["aaa.run.started", "aaa.llm.completed"]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:  # (event, run state) -> new events
        ...
```

Here is a complete one: translate a text into several languages **in parallel**, then collect the results.

```python
--8<-- "examples/custom_algorithm.py"
```

```text
{'French': 'Bonjour', 'German': 'Hallo', 'Spanish': 'Hola'}
```

## The recipe

1. **Describe the run state as a projection** (a Pydantic model plus a fold function). See
   [Projections and snapshots](../../concepts/projections-and-snapshots.md).
2. **React to `aaa.run.started`** by emitting the first request(s).
3. **React to results** by projecting state and deciding. Return `[]` when you are still waiting.
4. **Finish** with `finish(e, OutcomeKind.SUCCESS, value)`.

## Helpers (`aaa.algos.base`)

| Helper | |
|---|---|
| `llm_request(parent, LLMRequest, tag="main", prev=None)` | An `aaa.llm.requested` event. `tag` correlates the reply; `prev` continues a conversation as a delta (see below). |
| `tool_request(parent, call_id, name, arguments)` | An `aaa.tool.requested` event. |
| `finish(parent, kind, value, reason)` | The terminal `aaa.run.completed` event. |
| `parse_json_obj(text)` | Best-effort JSON object from model text (strips code fences, tolerates prose). |
| `strip_fence(text)`, `render_result(x)`, `task_messages(input, system)` | Small utilities. |

## `Ctx`

| | |
|---|---|
| `await ctx.project(PROJECTION)` | The run's current state: snapshot + tail. |
| `await ctx.started()` | The `aaa.run.started` event (input, budget). |
| `await ctx.history()` | Every event (use `project` instead for anything on the hot path). |
| `ctx.ledger()` | Spend so far. |
| `ctx.engine`, `ctx.run_id` | For starting child runs or emitting events directly. |

## Rules that keep it correct

- **Decide from state, not from the triggering event.** The same event may be seen twice; a pure function of the projected
  state gives the same answer both times.
- **Derive events with `e.caused(...)`** (every helper does). That gives them deterministic ids, so re-deriving after a
  crash is absorbed by the store instead of duplicated.
- **No I/O and no clocks in `on_event`.** Anything with a side effect belongs in a tool or a handler. That is what makes a
  run replayable.
- **Tag parallel requests** and key your projection by tag, as above. Return `[]` until all have reported.
- **Child runs**: `await ctx.engine.start(algo, input, parent=ctx.run_id, tag=..., run_id=stable_id(ctx.run_id, tag))`.
  A deterministic `run_id` makes the start idempotent after a crash; the result arrives as `aaa.child.completed`.
- **Emitting more than one event in a defined order** where a later step reads the earlier one: call
  `await ctx.engine.emit(event)` for the first, then project again (`PlanAndExecute` does this when it creates a plan then
  schedules from it).
- **Unique names.** Two instances with the same `name` cannot both run in one cluster. Pass distinct names.

## Continuing a conversation cheaply

For multi-turn loops, send only the *new* messages and point at the previous request:

```python
first = llm_request(e, LLMRequest(messages=[system, user]))  # full messages
second = llm_request(e, LLMRequest(messages=[tool_result_message]), prev=state.request_id)
```

Store `request_id` (the id of the last `main` request) in your projection instead of the transcript. The `LLMHandler` reconstructs
the full conversation for the provider. This keeps log size and state linear in the number of turns.

## A checklist before you ship an algorithm

- **Bound every loop** at construction (validate arguments, expose `max_calls`) and test the bound at its edge.
- **Order-independence:** run it with a client that completes parallel calls in a random order (three seeds) and assert the same outcome.
- **Replay:** re-fold the stored events and assert the projected state equals the live state.
- **Budget:** run with a tiny `Budget` and assert a prompt, partial outcome rather than a hang.
- **Data is not instruction:** put "ignore previous instructions" in every channel the algorithm reads (retrieval, tools, memory, human answers) and assert nothing it controls changes.
- **No I/O or clocks in `on_event`**, no unseeded randomness.
- Register an offline example and add the algorithm to the [choosing guide](index.md).

## Testing it

Use `FakeLLMClient` with a function of the request, and assert on the outcome and the event list. See [Testing your agents](../testing.md).
