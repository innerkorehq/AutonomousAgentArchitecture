"""Anytime: try the cheapest approach first and escalate only while the budget allows.

`Anytime` wraps an ordered ladder of algorithms (cheapest first). Each rung runs as a child run under its own `Budget`. A rung's successful result is accepted when
`accept(value)` says so (default: always); otherwise, or if the rung fails, the next rung is tried **if what is left of the run's budget can afford it**. When it cannot,
the run ends with the best result so far, marked `degraded`, instead of starting something it cannot finish. Remaining budget is read from the run's own log
(steps, tokens, cost, elapsed time), so the decision is the same on replay."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from aaa.algos.base import Calls, _apply_calls, finish
from aaa.core.ids import stable_id
from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.runtime import Algorithm, Budget, Ctx
from aaa.runtime.engine import RUNMETA

MAX_RUNGS = 6


@dataclass(frozen=True)
class Rung:
    name: str
    algo: Algorithm
    budget: Budget | None = None  # the rung's own cap
    needs_steps: int = 1  # steps the run must still have left to be worth starting this rung


class _A(Calls):
    started: list[str] = Field(default_factory=list)
    outcomes: dict[str, dict[str, Any]] = Field(default_factory=dict)


def _apply(st: _A, e: Event) -> _A:
    _apply_calls(st, e)
    if e.type == "aaa.task.started" and "rung" in e.data:
        st.started.append(e.data["rung"])
    elif e.type == "aaa.child.completed" and e.data.get("tag") in st.started:
        st.outcomes[e.data["tag"]] = e.data["outcome"]
    return st


ANYTIME = Projection(_A, _apply, name="aaa.anytime", version=1, state_type=_A)


class Anytime:
    """Escalate along a ladder while the budget allows.

    Result: `{"value", "rung", "tried", "degraded", "stopped"}` where `stopped` is `accepted` (a rung's result was good enough), `exhausted` (the ladder ran out) or
    `budget` (the next rung could not be afforded). `degraded` is true unless the accepted rung was the first choice."""

    name = "anytime"

    def __init__(self, ladder: list[Rung], *, accept: Callable[[Any], bool] | None = None) -> None:
        if not 2 <= len(ladder) <= MAX_RUNGS:
            raise ValueError(f"a ladder needs between 2 and {MAX_RUNGS} rungs")
        names = [r.name for r in ladder]
        if len(set(names)) != len(names):
            raise ValueError("rung names must be unique")
        self.ladder, self.accept = list(ladder), accept or (lambda _v: True)

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.child.completed"]

    async def _affordable(self, e: Event, ctx: Ctx, rung: Rung) -> bool:
        started = await ctx.started()
        budget = Budget.model_validate(started.data.get("budget") or {})
        meta = await ctx.project(RUNMETA)
        spent = meta.ledger
        if budget.max_steps is not None and budget.max_steps - spent.steps < rung.needs_steps:
            return False
        if budget.max_tokens is not None and budget.max_tokens - spent.tokens <= 0:
            return False
        if budget.max_cost is not None and budget.max_cost - spent.cost <= 0:
            return False
        if budget.deadline_s is not None:
            left = budget.deadline_s - (e.time - started.time).total_seconds()
            if left <= 0:
                return False
        return True

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(ANYTIME)
        best: tuple[str, Any] | None = None
        tried: list[str] = []
        for i, rung in enumerate(self.ladder):
            out = st.outcomes.get(rung.name)
            if out is None:
                if rung.name in st.started:
                    return []
                if i > 0 and not await self._affordable(e, ctx, rung):
                    return [self._end(e, best, tried, "budget")]
                child = stable_id(ctx.run_id, f"rung-{rung.name}")
                await ctx.engine.emit(
                    e.caused("aaa.task.started", {"rung": rung.name, "child_run_id": child})
                )
                await ctx.engine.start(
                    rung.algo,
                    dict(st.input),
                    parent=ctx.run_id,
                    tag=rung.name,
                    run_id=child,
                    budget=rung.budget,
                )
                return []
            tried.append(rung.name)
            if out["kind"] == "success":
                best = (rung.name, out["value"])
                if self.accept(out["value"]):
                    return [self._end(e, best, tried, "accepted")]
        return [self._end(e, best, tried, "exhausted")]

    def _end(self, e: Event, best: tuple[str, Any] | None, tried: list[str], stopped: str) -> Event:
        if best is None:
            return finish(
                e,
                OutcomeKind.FAILED,
                reason=f"no rung produced a result ({', '.join(tried) or 'none tried'}; stopped: {stopped})",
            )
        rung, value = best
        return finish(
            e,
            OutcomeKind.SUCCESS,
            {
                "value": value,
                "rung": rung,
                "tried": tried,
                "degraded": rung != self.ladder[0].name,
                "stopped": stopped,
            },
        )


__all__ = ["Anytime", "Rung"]
