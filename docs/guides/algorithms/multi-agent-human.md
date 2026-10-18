# Multi-agent coordination and asking a person

Five ways for several model calls (or a model and a person) to build one answer. Like every [algorithm](custom.md), each is a fold of the run's log: parallel calls may finish in any order without changing the result, a recorded run replays to the same outcome, and `max_calls` states the most model calls a run can make. Whatever one participant wrote is shown to the next only as quoted data (`<answers>`, `<board>`, `<document_from_previous_role>`, `<conversation>`), and a participant cannot close its own quote.

```python
--8<-- "examples/multi_agent_algorithms.py"
```

```text
mixture: Paris, the capital of France (3 proposers agreed) | layers [3, 3]
blackboard: Launch on Tuesday, after the fix. | board ['researcher', 'critic']
auction: winner expert | Done by the best-value bidder.
protocol: {'approved': True, 'notes': 'three steps, fine'} | analyst document {'title': 'Migrate', 'steps': ['freeze', 'copy', 'switch']}
human: the run is waiting on 'For how many people?'
human: Table booked for 4. | asked 1
```

## MixtureOfAgents

`MixtureOfAgents(proposers=3, layers=2, personas=None, models=None)`. Layer 0: independent answers. Every later layer: each proposer reads **all** the previous layer's answers and writes an improved one. An aggregator writes the final answer from the last layer. Unlike [`Debate`](reasoning.md), which ends in a vote or a judge's ruling, the answers are rewritten layer by layer and then merged; `models` gives each proposer its own model, which is where mixtures earn their keep. A proposer whose call fails drops out of that layer. Result `{"answer", "layers"}`; `proposers * layers + 1` calls.

## Blackboard

`Blackboard(experts={"name": "what they do"}, max_rounds=6)`. A **chair** reads the board and names who should contribute next (or `done`); that expert adds one contribution (capped at `max_entry_chars`); finally a solver writes the answer from the board. The chair can only pick among the experts you listed, an unknown name ends the session, and a failed expert call is skipped. Result `{"answer", "board", "rounds"}`; `2 * max_rounds + 1` calls.

**Knowledge sources.** An expert can be an `Expert(description, when=..., auto=..., repeat=...)`. `when` is a plain function of the board (a list of `(author, text)`; no I/O, an exception counts as "not triggered"); ready-made ones are `empty_board()`, `after(author)` and `mentions(word, by=None)`.

```python
Blackboard({
    "drafter": Expert("writes the draft.", when=empty_board(), auto=True),
    "legal": Expert("checks legal issues.", when=mentions("legal review"), auto=True),
    "scribe": Expert("takes notes when asked.", when=after("legal")),
    "editor": "polishes the final text.",                 # no trigger: the chair may pick it any time
})
```

- With `auto=True` the expert contributes **as soon as its trigger holds**, without the chair being asked (the first such expert in the order given, if several are ready). The chair is consulted only when no automatic expert is ready, so a reactive board needs fewer model calls.
- Without `auto`, `when` limits *when the chair may pick* the expert: the chair is shown only the experts that are eligible right now, and naming any other ends the session.
- A triggered expert contributes **once per session** unless `repeat=True`, which lets it fire again after each change of the board (never twice on the same board). Rounds are always capped by `max_rounds`, so triggers cannot loop.

## Auction

`Auction(workers, rule="best_value", min_confidence=0.2, max_cost=1000, fallback=True, child_budget=...)`. Each worker bids `{"cost", "confidence", "approach"}` **blind** (bids are made in parallel and no bidder sees another's). Bids are parsed, clamped and filtered; the award is plain code: `best_value` = confidence / (1 + cost), or `lowest_cost`, or `highest_confidence`, with ties broken by worker name. The winner runs the task as a child run under `child_budget`; if it fails and `fallback` is on, the next-ranked bidder gets it. No acceptable bid ends `NO_SOLUTION`. A bidder can lie, but its only reward is getting the work and then being judged on what it does. Result `{"text", "winner", "tried", "bids"}`.

## RolePlaySOP

```python
class Spec(BaseModel): title: str; steps: list[str]
class Review(BaseModel): approved: bool; notes: str
RolePlaySOP([Role("analyst", "Write a spec.", Spec), Role("reviewer", "Review the spec.", Review)])
```

Roles run in order. Each receives the task and the previous role's **validated** document, and must reply with JSON that validates against its own output model (which is also sent as the response schema). An invalid reply goes back with the validation errors for up to `max_repairs` repairs, then the run fails. Only the declared fields travel on, so a role cannot slip extra fields, or instructions, into the next one's input. Result `{"document", "documents", "roles", "repairs"}`.

## AskHuman

`AskHuman(max_questions=3)` makes asking a person part of the protocol. Each turn the model answers or asks **one** question. A question is logged as `aaa.human.requested` `{question_id, question}` and the run waits. A person answers by signalling the run:

```python
handle = await engine.start(AskHuman(), {"task": "Book a table."})
asked = await engine.await_event(handle.run_id, lambda e: e.type == "aaa.human.requested")
await handle.signal("aaa.human.answered", {"question_id": asked.data["question_id"], "answer": "4"})
```

- Only an `aaa.human.answered` event counts, and model output is never an event of that type. An answer counts for the **outstanding** question only, **once**; stray, repeated or early answers are ignored. Answers are capped at 2000 characters and are data in the next prompt, never instructions.
- After `max_questions` the model must answer. Result `{"answer", "questions", "asked"}`.
- There is no timer inside an algorithm: bound the wait with `Budget(deadline_s=...)`; the run then ends `TIMEOUT`.
- **Who may signal a run is the host's decision.** In your own code it is whoever holds the handle. Behind `aaa-server`, use `POST /v1/runs/{id}/answers` with `{"question_id": "q0", "answer": "4"}`: it needs the separate `answer` scope, applies to your tenant's runs only, accepts the outstanding question once (`409` otherwise), and records who answered. The question itself arrives on the event stream as `human.requested`; the answer is not broadcast.

## Choosing

| You want | Use |
|---|---|
| Several drafts that improve each other | `MixtureOfAgents` |
| Specialists adding to one shared piece of work | `Blackboard` |
| The best-suited worker for a task, picked fairly | `Auction` |
| A fixed process with typed hand-offs | `RolePlaySOP` |
| The agent to ask when it is unsure | `AskHuman` |
