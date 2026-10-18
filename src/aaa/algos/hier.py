"""Hierarchical delegation: a manager splits a goal into sub-goals, hands each to a named worker as a **child run with its own budget**,
and writes the final answer from what came back. A worker can itself be a `HierarchicalAgent`, so teams nest; `max_depth` stops that
at construction time of the run, not just by convention.

Compared with `PlanAndExecute` (one executor for every step) and `Supervisor` (the model delegates through tool calls turn by turn), the
manager here commits to a validated sub-goal DAG first, picks a *specialist* per sub-goal, and every child is capped separately, so one worker cannot
use up the budget of the others."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from aaa.algos.base import (
    Calls,
    _apply_calls,
    ask,
    data_block,
    finish,
    llm_failed,
    parse_json_obj,
    render_result,
    task_text,
)
from aaa.core.ids import stable_id
from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.llm.types import Message
from aaa.runtime import Algorithm, Budget, Ctx

MAX_SUBGOALS, MAX_DEPTH, MAX_PARALLEL = 12, 5, 6
_ID = re.compile(r"^[a-z][a-z0-9_]{0,19}$")
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "worker": {"type": "string"},
                    "goal": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "worker", "goal"],
            },
        }
    },
    "required": ["subgoals"],
}


class _H(Calls):
    started: list[str] = Field(default_factory=list)  # sub-goal ids whose child run was started
    outcomes: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )  # sub-goal id -> child outcome


def _apply(st: _H, e: Event) -> _H:
    _apply_calls(st, e)
    if e.type == "aaa.task.started" and "subgoal" in e.data:
        st.started.append(e.data["subgoal"])
    elif e.type == "aaa.child.completed" and e.data.get("tag") in st.started:
        st.outcomes[e.data["tag"]] = e.data["outcome"]
    return st


HIER = Projection(_H, _apply, name="aaa.hier", version=1, state_type=_H)


def _text_of(value: Any) -> str:
    if isinstance(value, dict):
        for k in ("text", "answer", "candidate"):
            if isinstance(value.get(k), str):
                return value[k]
    return render_result(value)[:4000]


class HierarchicalAgent:
    """Manager over named workers.

    `workers` maps a name to the algorithm that does that kind of work (`descriptions` tells the manager what each is for). The manager
    writes sub-goals `{id, worker, goal, depends_on}`; ids, worker names and the dependency graph are checked before any child starts.
    Ready sub-goals run as child runs (at most `max_parallel` at once), each under `child_budget`, and each sees only its own goal and the results of
    the sub-goals it depends on. `max_depth` bounds nesting: a manager at that depth refuses to delegate.

    Result: `{"text", "subgoals": [{"id", "worker", "goal", "result"}], "depth"}`."""

    def __init__(
        self,
        workers: Mapping[str, Algorithm],
        *,
        descriptions: Mapping[str, str] | None = None,
        max_subgoals: int = 6,
        max_parallel: int = 3,
        max_depth: int = 3,
        child_budget: Budget | None = None,
        on_child_failure: str = "fail",
        synthesize: bool = True,
        model: str | None = None,
        system: str = "You are a manager who delegates work to specialists.",
        name: str = "hierarchical",
    ) -> None:
        if not workers:
            raise ValueError("a manager needs at least one worker")
        if not all(_ID.match(k) for k in workers):
            raise ValueError("worker names must be lowercase letters, digits and '_' (at most 20)")
        if not 1 <= max_subgoals <= MAX_SUBGOALS:
            raise ValueError(f"max_subgoals must be between 1 and {MAX_SUBGOALS}")
        if not 1 <= max_parallel <= MAX_PARALLEL:
            raise ValueError(f"max_parallel must be between 1 and {MAX_PARALLEL}")
        if not 1 <= max_depth <= MAX_DEPTH:
            raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}")
        if on_child_failure not in ("fail", "continue"):
            raise ValueError("on_child_failure must be 'fail' or 'continue'")
        self.workers, self.descriptions = dict(workers), dict(descriptions or {})
        self.max_subgoals, self.max_parallel, self.max_depth = max_subgoals, max_parallel, max_depth
        self.child_budget = child_budget or Budget(max_steps=50, max_tokens=200_000, deadline_s=300)
        self.on_child_failure, self.synthesize = on_child_failure, synthesize
        self.model, self.system, self.name = model, system, name

    @property
    def max_children(self) -> int:
        return self.max_subgoals

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed", "aaa.child.completed"]

    def _parse(self, st: _H) -> tuple[list[dict[str, Any]], str | None]:
        obj = parse_json_obj(st.text("decompose")) or {}
        raw = obj.get("subgoals")
        if not isinstance(raw, list) or not raw:
            return [], "the manager produced no sub-goals"
        if len(raw) > self.max_subgoals:
            return [], f"the manager produced {len(raw)} sub-goals (at most {self.max_subgoals})"
        seen: list[str] = []
        out: list[dict[str, Any]] = []
        for g in raw:
            if not isinstance(g, dict):
                return [], "a sub-goal is not an object"
            gid, worker, goal = g.get("id"), g.get("worker"), g.get("goal")
            deps = g.get("depends_on") or []
            if not isinstance(gid, str) or not _ID.match(gid) or gid in seen:
                return [], f"bad or repeated sub-goal id {gid!r}"
            if not isinstance(worker, str) or worker not in self.workers:
                return [], f"sub-goal {gid} names a worker that does not exist"
            if not isinstance(goal, str) or not goal.strip():
                return [], f"sub-goal {gid} has no goal"
            if not isinstance(deps, list) or any(d not in seen for d in deps):
                return [], f"sub-goal {gid} depends on something that is not an earlier sub-goal"
            seen.append(gid)
            out.append(
                {"id": gid, "worker": worker, "goal": goal.strip()[:2000], "depends_on": list(deps)}
            )
        return out, None

    def _manager_prompt(self, st: _H) -> list[Message]:
        team = "\n".join(f"- {n}: {self.descriptions.get(n, 'a worker')}" for n in self.workers)
        return [
            Message(
                role="system",
                content=(
                    f"{self.system} Split the task into at most {self.max_subgoals} sub-goals, each for one worker. Reply with JSON "
                    '{"subgoals": [{"id": "a", "worker": "<name>", "goal": "...", "depends_on": []}]}. '
                    f"depends_on lists earlier sub-goal ids whose results are needed.\n\nWorkers:\n{team}"
                ),
            ),
            Message(role="user", content=data_block("task", task_text(st.input))),
        ]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        if e.type == "aaa.llm.failed":
            return [llm_failed(e)]
        st = await ctx.project(HIER)
        depth = int(st.input.get("_depth", 0)) if isinstance(st.input.get("_depth", 0), int) else 0
        if e.type == "aaa.run.started":
            if depth >= self.max_depth:
                return [
                    finish(e, OutcomeKind.FAILED, reason="the delegation depth limit was reached")
                ]
            return [
                ask(
                    e,
                    self.name,
                    "decompose",
                    self._manager_prompt(st),
                    model=self.model,
                    schema=_SCHEMA,
                )
            ]
        if not st.done("decompose"):
            return []
        goals, why = self._parse(st)
        if why:
            return [finish(e, OutcomeKind.FAILED, reason=f"invalid delegation: {why}")]
        status: dict[str, str] = {}
        results: dict[str, str] = {}
        for g in goals:
            out = st.outcomes.get(g["id"])
            if out is None:
                status[g["id"]] = "running" if g["id"] in st.started else "pending"
            elif out["kind"] == "success":
                status[g["id"]], results[g["id"]] = "done", _text_of(out["value"])
            elif self.on_child_failure == "continue":
                status[g["id"]], results[g["id"]] = "done", "(this sub-goal failed)"
            else:
                return [finish(e, OutcomeKind.FAILED, reason=f"sub-goal {g['id']} failed")]
        running = sum(1 for s in status.values() if s == "running")
        for g in goals:
            if status[g["id"]] != "pending" or running >= self.max_parallel:
                continue
            if not all(status[d] == "done" for d in g["depends_on"]):
                continue
            prerequisites = {d: results[d] for d in g["depends_on"]}
            prompt = (
                f"Overall task: {task_text(st.input)}\nYour sub-goal ({g['id']}): {g['goal']}\n"
                f"Results of the sub-goals you depend on: {render_result(prerequisites)}"
            )
            child = stable_id(ctx.run_id, g["id"])
            await ctx.engine.emit(
                e.caused(
                    "aaa.task.started",
                    {"subgoal": g["id"], "worker": g["worker"], "child_run_id": child},
                )
            )
            await ctx.engine.start(
                self.workers[g["worker"]],
                {"task": prompt, "_depth": depth + 1},
                parent=ctx.run_id,
                tag=g["id"],
                run_id=child,
                budget=self.child_budget,
            )
            status[g["id"]] = "running"
            running += 1
        if any(s != "done" for s in status.values()):
            return []
        value = {
            "subgoals": [
                {
                    "id": g["id"],
                    "worker": g["worker"],
                    "goal": g["goal"],
                    "result": results[g["id"]],
                }
                for g in goals
            ],
            "depth": depth,
        }
        if not self.synthesize:
            return [finish(e, OutcomeKind.SUCCESS, {"text": results[goals[-1]["id"]], **value})]
        if not st.done("synth"):
            listing = "\n".join(f"[{g['id']}] {g['goal']}\n{results[g['id']]}" for g in goals)
            msgs = [
                Message(
                    role="system",
                    content="Combine the sub-goal results into one final answer to the task. Reply with the answer only.",
                ),
                Message(
                    role="user",
                    content=data_block("task", task_text(st.input))
                    + "\n\n"
                    + data_block("results", listing),
                ),
            ]
            return [ask(e, self.name, "synth", msgs, model=self.model)]
        return [finish(e, OutcomeKind.SUCCESS, {"text": st.text("synth"), **value})]
