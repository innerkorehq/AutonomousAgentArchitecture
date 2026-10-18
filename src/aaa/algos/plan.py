from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aaa.algos.base import finish, llm_failed, llm_request, parse_json_obj, render_result
from aaa.core.ids import stable_id
from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.llm.types import LLMRequest, LLMResponse, Message
from aaa.runtime import Algorithm, Ctx


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str
    depends_on: list[str]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[Step]


_PLAN_SYSTEM = (
    "You are a planner. Break the task into the smallest set of concrete steps. "
    'Reply with JSON {"steps":[{"id":"s1","description":"...","depends_on":[]}]}. '
    "Use depends_on for steps that need earlier results; independent steps run in parallel."
)


class _Step(BaseModel):
    description: str
    deps: list[str]
    status: str = "pending"  # pending | running | done | failed
    result: Any = None


class _P(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    steps: dict[str, _Step] = Field(default_factory=dict)
    replans: int = 0
    replanning: bool = False
    synth: bool = False
    plans: int = 0


def _apply(st: _P, e: Event) -> _P:
    d = e.data
    if e.type == "aaa.run.started":
        st.input = d["input"]
    elif e.type == "aaa.plan.created":
        st.steps = {
            s["id"]: _Step(description=s["description"], deps=s["depends_on"]) for s in d["steps"]
        }
        st.plans += 1
    elif e.type == "aaa.plan.revised":
        st.steps = {k: v for k, v in st.steps.items() if v.status in ("done", "running")}
        st.steps.update(
            {s["id"]: _Step(description=s["description"], deps=s["depends_on"]) for s in d["steps"]}
        )
        st.replans, st.replanning = st.replans + 1, False
    elif e.type == "aaa.plan.kept":  # a progress review that changed nothing (or could not be used)
        st.replans, st.replanning = st.replans + 1, False
    elif e.type == "aaa.llm.requested":
        st.replanning = st.replanning or d.get("tag") in ("replan", "review")
        st.synth = st.synth or d.get("tag") == "synth"
    elif e.type == "aaa.task.started" and d["step_id"] in st.steps:
        st.steps[d["step_id"]].status = "running"
    elif e.type == "aaa.child.completed" and d.get("tag") in st.steps:
        s, out = st.steps[d["tag"]], d["outcome"]
        s.status = "done" if out["kind"] == "success" else "failed"
        s.result = out["value"] if s.status == "done" else out.get("reason") or out["kind"]
    return st


PLAN = Projection(_P, _apply, name="aaa.plan", version=2, state_type=_P)


def _valid(steps: list[Step], known: set[str]) -> str | None:
    ids = [s.id for s in steps]
    if not steps:
        return "plan has no steps"
    if len(set(ids)) != len(ids):
        return "duplicate step ids"
    allowed = set(ids) | known
    for s in steps:
        if bad := [d for d in s.depends_on if d not in allowed]:
            return f"step {s.id} depends on unknown {bad}"
    seen: set[str] = set(known)
    todo = list(steps)
    while todo:  # Kahn-style: every pass must make progress or there is a cycle
        ready = [s for s in todo if all(d in seen for d in s.depends_on)]
        if not ready:
            return "plan has a dependency cycle"
        seen |= {s.id for s in ready}
        todo = [s for s in todo if s not in ready]
    return None


class PlanAndExecute:
    """Plan once (LLM, structured), execute steps as child runs of `executor` honoring the
    dependency DAG with bounded parallelism, replan on step failure (bounded), then synthesize.
    The plan lives on the log as events, so it is inspectable, resumable and replayable."""

    name = "plan_and_execute"

    def __init__(
        self,
        executor: Algorithm,
        *,
        model: str | None = None,
        max_parallel: int = 3,
        max_replans: int = 1,
        synthesize: bool = True,
        planner_system: str = _PLAN_SYSTEM,
        review_progress: bool = False,
    ) -> None:
        self.executor, self.model, self.max_parallel = executor, model, max_parallel
        self.max_replans, self.synthesize, self.planner_system = (
            max_replans,
            synthesize,
            planner_system,
        )
        self.review_progress = review_progress  # after each finished step, let the planner revise what is left (counts against max_replans)

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed", "aaa.child.completed"]

    def _req(self, e: Event, system: str, user: str, tag: str, schema: bool = True) -> Event:
        req = LLMRequest(
            model=self.model,
            messages=[Message(role="system", content=system), Message(role="user", content=user)],
            response_schema=Plan.model_json_schema() if schema else None,
        )
        return llm_request(e, req, tag)

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(PLAN)
        tag = e.data.get("tag")
        if e.type == "aaa.run.started":
            return [self._req(e, self.planner_system, f"Task: {st.input.get('task', '')}", "plan")]
        if e.type == "aaa.llm.failed":
            if tag == "review":  # a review that could not be made leaves the plan as it is
                return await self._keep(e, ctx)
            return [llm_failed(e)]
        if e.type == "aaa.llm.completed":
            text = LLMResponse.model_validate(e.data["response"]).message.content
            if tag in ("plan", "replan"):
                return await self._accept_plan(e, ctx, st, text, revised=tag == "replan")
            if tag == "review":
                return await self._accept_plan(e, ctx, st, text, revised=True, soft=True)
            if tag == "synth":
                return [
                    finish(e, OutcomeKind.SUCCESS, {"text": text, "results": self._results(st)})
                ]
            return []
        if e.type == "aaa.child.completed":
            if e.data["outcome"]["kind"] != "success":
                if st.replans >= self.max_replans or st.replanning:
                    return (
                        []
                        if st.replanning
                        else [
                            finish(
                                e,
                                OutcomeKind.FAILED,
                                reason=f"step {e.data['tag']} failed: {st.steps[e.data['tag']].result}",
                            )
                        ]
                    )
                return [self._req(e, self.planner_system, self._replan_prompt(st), "replan")]
            if (
                self.review_progress
                and not st.replanning
                and st.replans < self.max_replans
                and any(s.status == "pending" for s in st.steps.values())
            ):
                return [self._req(e, self.planner_system, self._review_prompt(st), "review")]
            return await self._advance(e, ctx, st)
        return []

    def _results(self, st: _P) -> dict[str, Any]:
        return {k: s.result for k, s in st.steps.items() if s.status == "done"}

    def _replan_prompt(self, st: _P) -> str:
        done = {k: s.result for k, s in st.steps.items() if s.status == "done"}
        failed = {k: s.result for k, s in st.steps.items() if s.status == "failed"}
        return (
            f"Task: {st.input.get('task', '')}\nCompleted steps and results: {render_result(done)}\n"
            f"Failed steps and reasons: {render_result(failed)}\n"
            "Return ONLY the remaining steps needed (new ids; depends_on may reference completed ids)."
        )

    def _review_prompt(self, st: _P) -> str:
        done = {k: s.result for k, s in st.steps.items() if s.status == "done"}
        left = {k: s.description for k, s in st.steps.items() if s.status == "pending"}
        return (
            f"Task: {st.input.get('task', '')}\nCompleted steps and results: {render_result(done)}\n"
            f"Steps still planned: {render_result(left)}\n"
            "Return the steps still needed, unchanged if the plan still holds or corrected in the light of the results "
            "(new ids; depends_on may reference completed ids)."
        )

    async def _keep(self, e: Event, ctx: Ctx) -> list[Event]:
        await ctx.engine.emit(e.caused("aaa.plan.kept", {}))
        return await self._advance(e, ctx, await ctx.project(PLAN))

    async def _accept_plan(
        self, e: Event, ctx: Ctx, st: _P, text: str | None, revised: bool, soft: bool = False
    ) -> list[Event]:
        obj = parse_json_obj(text)
        try:
            plan = Plan.model_validate(obj)
        except ValidationError:
            if soft:
                return await self._keep(e, ctx)
            return [finish(e, OutcomeKind.FAILED, reason="planner returned an invalid plan")]
        known = (
            {k for k, s in st.steps.items() if s.status in ("done", "running")}
            if revised
            else set()
        )
        if revised:  # namespace new ids so they never collide with earlier steps
            pre = f"r{st.replans + 1}-"
            plan = Plan(
                steps=[
                    Step(
                        id=pre + s.id,
                        description=s.description,
                        depends_on=[
                            pre + d if d in {x.id for x in plan.steps} else d for d in s.depends_on
                        ],
                    )
                    for s in plan.steps
                ]
            )
        if why := _valid(plan.steps, known):
            if soft:
                return await self._keep(e, ctx)
            return [finish(e, OutcomeKind.FAILED, reason=f"invalid plan: {why}")]
        ev = e.caused(
            "aaa.plan.revised" if revised else "aaa.plan.created",
            {"steps": [s.model_dump() for s in plan.steps]},
        )
        await ctx.engine.emit(ev)
        return await self._advance(e, ctx, await ctx.project(PLAN))

    async def _advance(self, e: Event, ctx: Ctx, st: _P) -> list[Event]:
        if st.replanning:
            return []
        running = sum(1 for s in st.steps.values() if s.status == "running")
        ready = [
            k
            for k, s in st.steps.items()
            if s.status == "pending"
            and all(st.steps[d].status == "done" for d in s.deps if d in st.steps)
        ]
        for k in ready[: max(0, self.max_parallel - running)]:
            s, child = st.steps[k], stable_id(ctx.run_id, k)
            ctx_txt = {d: st.steps[d].result for d in s.deps if d in st.steps}
            prompt = (
                f"Overall task: {st.input.get('task', '')}\nYour step ({k}): {s.description}\n"
                f"Results of prerequisite steps: {render_result(ctx_txt)}"
            )
            await ctx.engine.emit(
                e.caused("aaa.task.started", {"step_id": k, "child_run_id": child})
            )
            await ctx.engine.start(
                self.executor, {"task": prompt}, parent=ctx.run_id, tag=k, run_id=child
            )
            s.status = "running"
            running += 1
        if st.steps and all(s.status == "done" for s in st.steps.values()) and not st.synth:
            if not self.synthesize:
                return [finish(e, OutcomeKind.SUCCESS, {"results": self._results(st)})]
            user = f"Task: {st.input.get('task', '')}\nStep results: {render_result(self._results(st))}\nWrite the final answer."
            return [
                self._req(
                    e,
                    "Combine the step results into one final answer.",
                    user,
                    "synth",
                    schema=False,
                )
            ]
        return []


__all__ = ["Plan", "PlanAndExecute", "Step"]
