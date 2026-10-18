"""An auction: workers bid for a task, a deterministic rule awards it, the winner runs it as a child run under its own budget.

Bids are collected independently and in parallel (no bidder sees another's bid), parsed into numbers and clamped, and the award rule is plain code with fixed
tie-breaking (worker name), so a bidder cannot talk its way into winning: the worst it can do is bid high confidence and then fail, in which case
the next-best bidder gets the task (`fallback`). Only workers you list can bid."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from aaa.algos.base import ask, data_block, finish, parse_json_obj, task_text
from aaa.algos.hier import HIER, _text_of
from aaa.core.ids import stable_id
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import Message
from aaa.runtime import Algorithm, Budget, Ctx

MAX_WORKERS = 8
RULES = ("best_value", "lowest_cost", "highest_confidence")
_BID_SCHEMA = {
    "type": "object",
    "properties": {
        "cost": {"type": "number"},
        "confidence": {"type": "number"},
        "approach": {"type": "string"},
    },
    "required": ["cost", "confidence"],
}


class Auction:
    """Award a task to the best bidder.

    Each worker is asked for `{"cost": estimated effort >= 0, "confidence": 0 to 1, "approach"}`. Bids below `min_confidence` or above `max_cost` are
    rejected. The rule ranks the rest: `best_value` = confidence / (1 + cost), `lowest_cost`, or `highest_confidence`. The winner runs the task under
    `child_budget`; if it fails and `fallback` is on, the next-ranked bidder runs it.

    Result: `{"text", "winner", "tried": [names], "bids": {name: {"cost", "confidence"}}}`. Calls: one bid per worker plus the winner's own."""

    name = "auction"

    def __init__(
        self,
        workers: Mapping[str, Algorithm],
        *,
        descriptions: Mapping[str, str] | None = None,
        rule: str = "best_value",
        min_confidence: float = 0.2,
        max_cost: float = 1000.0,
        fallback: bool = True,
        child_budget: Budget | None = None,
        model: str | None = None,
    ) -> None:
        if not 2 <= len(workers) <= MAX_WORKERS:
            raise ValueError(f"an auction needs between 2 and {MAX_WORKERS} workers")
        if not all(re.match(r"^[a-z][a-z0-9_]{0,19}$", n) for n in workers):
            raise ValueError("worker names must be lowercase letters, digits and '_' (at most 20)")
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if not 0.0 <= min_confidence <= 1.0 or max_cost < 0:
            raise ValueError("min_confidence must be in [0, 1] and max_cost must not be negative")
        self.workers, self.descriptions = dict(workers), dict(descriptions or {})
        self.rule, self.min_confidence, self.max_cost, self.fallback = (
            rule,
            min_confidence,
            max_cost,
            fallback,
        )
        self.child_budget = child_budget or Budget(max_steps=50, max_tokens=200_000, deadline_s=300)
        self.model = model

    @property
    def max_bids(self) -> int:
        return len(self.workers)

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed", "aaa.child.completed"]

    @staticmethod
    def _bid(text: str) -> tuple[float, float] | None:
        obj = parse_json_obj(text) or {}
        try:
            cost, conf = float(obj["cost"]), float(obj["confidence"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(cost) and math.isfinite(conf)) or cost < 0:
            return None
        return cost, max(0.0, min(1.0, conf))

    def _ranking(self, bids: dict[str, tuple[float, float]]) -> list[str]:
        ok = {
            n: b for n, b in bids.items() if b[1] >= self.min_confidence and b[0] <= self.max_cost
        }
        key = {
            "best_value": lambda b: -(b[1] / (1.0 + b[0])),
            "lowest_cost": lambda b: b[0],
            "highest_confidence": lambda b: -b[1],
        }[self.rule]
        return sorted(ok, key=lambda n: (key(ok[n]), n))

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(HIER)
        task = task_text(st.input)
        tags = {n: f"b.{n}" for n in self.workers}
        if not st.all_done(tags.values()):
            out = []
            for n, t in tags.items():
                if st.done(t):
                    continue
                msgs = [
                    Message(
                        role="system",
                        content=(
                            f"You are {n}: {self.descriptions.get(n, 'a worker')}. A task is up for auction. Bid honestly. "
                            'Reply with JSON {"cost": estimated effort, 0 or more, "confidence": 0 to 1 that you can do it well, "approach": "one line"}.'
                        ),
                    ),
                    Message(role="user", content=data_block("task", task)),
                ]
                out.append(ask(e, self.name, t, msgs, model=self.model, schema=_BID_SCHEMA))
            return out
        bids = {
            n: b
            for n in self.workers
            if tags[n] in st.texts and (b := self._bid(st.text(tags[n]))) is not None
        }
        order = self._ranking(bids)
        if not order:
            return [
                finish(
                    e,
                    OutcomeKind.NO_SOLUTION,
                    {"bids": {n: {"cost": b[0], "confidence": b[1]} for n, b in bids.items()}},
                    reason="no acceptable bid",
                )
            ]
        tried: list[str] = []
        for name in order:
            tried.append(name)
            out_ = st.outcomes.get(name)
            if out_ is None:
                if name not in st.started:
                    child = stable_id(ctx.run_id, f"win-{name}")
                    await ctx.engine.emit(
                        e.caused(
                            "aaa.task.started",
                            {"subgoal": name, "worker": name, "child_run_id": child},
                        )
                    )
                    await ctx.engine.start(
                        self.workers[name],
                        {"task": task},
                        parent=ctx.run_id,
                        tag=name,
                        run_id=child,
                        budget=self.child_budget,
                    )
                return []
            if out_["kind"] == "success":
                value: dict[str, Any] = {
                    "text": _text_of(out_["value"]),
                    "winner": name,
                    "tried": tried,
                    "bids": {n: {"cost": b[0], "confidence": b[1]} for n, b in bids.items()},
                }
                return [finish(e, OutcomeKind.SUCCESS, value)]
            if not self.fallback:
                break
        return [
            finish(e, OutcomeKind.FAILED, reason=f"the winning worker failed ({', '.join(tried)})")
        ]


__all__ = ["Auction"]
