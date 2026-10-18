# Memory and self-improvement

Four ways for an agent to get better with use: keep **skills** it has written and verified, **reflect** on what it has experienced, **manage its own memory**, and have its **prompt improved from evaluation results**. All of them write something that later runs read, so they share one rule: what is written is scoped to the run's tenant and user (from the host, never from model output), carries its provenance, can be listed and deleted, and is shown to the model as quoted data. The threats are in the [algorithms threat model](../../security/algorithms.md) (T2, T3, T11).

```python
--8<-- "examples/memory_algorithms.py"
```

```text
skills: stored add_numbers | a failing test is refused | AssertionError
skills: run -> 42 | other tenant sees []
curriculum: [('add 20 and 22', True), ('add 1 and 2', True)] | skills ['add_numbers'] | done
reflection: Ann is building a bakery in Lisbon. | sources ['m1', 'm0'] | stored 1
self-managed: True (the next run starts with what the model chose to keep)
optimise: 'Think step by step.' | dev 0.0 -> 1.0, held-out 0.0 -> 1.0 | improved: True
```

## Skills: SkillLibrary and VoyagerAgent

A **skill** is a small Python function `def skill(**arguments)` with a description and a **test** (Python that calls `skill` and raises if it is wrong).

```python
library = SkillLibrary(memory, DockerSandbox())
agent = VoyagerAgent(library)            # a ToolLoop; attach ToolHandler(agent.tools)
```

What the code enforces, not the prompt:

- **Verified before stored.** `SkillLibrary.add` runs the test in the sandbox and stores the skill only if it passes. An entry that lacks the `verified` mark (a forged or damaged one) is never offered or run.
- **Scoped.** Skills live in the run's tenant/user scope; another tenant finds nothing. Each records who wrote it, when and from which run.
- **Running is a separate, gated step.** `find_skill` returns names and descriptions only. `run_skill` and `save_skill` execute code, so they **wait for approval** by default (`requires_approval=False` only for a sandbox you would trust with the worst code the model could write).
- **Arguments are data.** A skill runs from a fixed template; its arguments arrive as a JSON file, never as text spliced into code.
- **Bounded.** Code, test and description sizes, `max_skills`, and the sandbox `Limits` (time, memory, output).

`VoyagerAgent` is a tool loop that finds a fitting skill, otherwise writes one with `save_skill` (and repairs it if the test fails), then runs it and answers. The memory backend must keep `meta` as given (the native and fake memories do); `SubprocessSandbox` is for development, not isolation.

### VoyagerCurriculum

`VoyagerCurriculum(agent, goal, max_tasks=5, child_budget=...)` is Voyager's automatic curriculum. Each round it lists the skills the library holds (`list_skills`), asks the model for the **next task** (slightly harder than what the agent can already do, building on its skills; the goal, the skills and the tasks tried so far are shown as quoted data), and runs the agent on it as a **child run under `child_budget`**. It stops when the model says `done`, when it proposes nothing new (a task is never attempted twice, compared ignoring case), or after `max_tasks`.

It adds no new power: everything the agent does stays gated as above (tests before storing, approval before saving or running unless you turned it off, the sandbox's limits). It only chooses *what to attempt*, a bounded number of times. Skills are written under the run's tenant and user; children inherit their parent's user. Result `{"attempted": [{"task", "ok", "answer"}], "skills": [...], "stopped": "done" | "max_tasks" | "no_new_task"}`.

## MemoryReflection

`MemoryReflection(memory, window=20, threshold=40, questions=3, insights_per_question=2)` follows Generative Agents. It lists the scope's `window` most recent memories, scores any unrated ones 1 to 10 for importance, and **only if the total reaches `threshold`** asks for the most salient questions, retrieves memories for each (ranked by recency, importance and relevance together with `rank_memories`), asks for insights, and writes them back.

- An insight must **cite memory ids that were shown to the model for that question**; others are dropped. What is stored is only the insight text (at most 400 characters), marked `reflection`, with `sources`, `importance` and `run_id`, so it can be audited and deleted.
- Below the threshold nothing but the scoring calls is spent. Result `{"reflected", "importance", "questions", "insights", "written"}`; at most `window + 1 + questions` model calls.
- Needs a memory that can list (`Listable`); attach `ToolHandler(reflection.tools)`. Run it after conversations or on a schedule.

## SelfManagedMemory

`SelfManagedMemory(memory, blocks={"persona": 2000, "human": 2000})` is MemGPT's idea as a tool loop. **Core memory** blocks are small and always in front of the model: at the start of each run they are read and put at the top of the system prompt (quoted). The model edits them with `core_memory_append` / `core_memory_replace` (size-capped, `old` must appear exactly once, edits to one block are serialised) and uses **archival memory** for the rest (`archival_insert`, `archival_search`). Every edit is a tool call on the log. Block labels and limits are fixed when you construct it; the memory must be able to list and update.

Keeping a long *conversation* inside the model's window is the context policy's job: `LLMHandler(client, context=ContextPolicy(...))` summarises or drops old turns when each request is sent, while the log keeps everything. Edits made during a run show up in the tool results, and the next run starts with them in the prompt.

## aaa.optimize: improve a prompt from evals

```python
result = await optimize_prompt(engine, build, dataset, [ExactMatch()], proposer,
                               baseline="You answer questions.", rounds=2, variants=3)
result.improved, result.best, result.best_holdout, result.baseline_holdout, result.diff
```

`build(prompt)` returns the algorithm to evaluate. The dataset is split **deterministically** into dev and held-out parts (by a hash of the case id, so file order does not matter). Each round the `proposer` (any `LLMClient`) sees the best prompt, its dev score and some failing cases, and proposes variants; every variant is scored on the dev cases with an [`EvalSuite`](../evals.md). At the end the baseline and the winner are both scored on the **held-out cases**, and `improved` is true only if the winner beats the baseline there. A prompt that only wins on the dev cases is reported as not improved.

- It **never changes your configuration**: you get a candidate prompt and a unified diff to review.
- It is **bounded before it starts**: more than `max_evaluations` case runs is an error. Rounds (6), variants (8) and prompt length are capped.
- Failing cases shown to the proposer are quoted data; a proposer that fails costs a round, not the search.
- The result carries the dev and held-out sizes: on a handful of cases a win can be luck.

## Choosing

| You want | Use |
|---|---|
| An agent that accumulates reusable, tested code | `VoyagerAgent` |
| Higher-level conclusions from many small memories | `MemoryReflection` |
| The model to decide what stays in view and what is archived | `SelfManagedMemory` |
| A measurably better prompt, with the evidence | `optimize_prompt` |
