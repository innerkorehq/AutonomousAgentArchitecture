"""LLMCompiler: plan a DAG of tool calls, run the independent ones in parallel, then let a **joiner** either answer or ask for one more round
of planning with what was learned.

Built on the same checked plan as `ReWOO` (see `toolplan.py`): the planner can only name the tools it was given, references go to earlier steps only,
and every call goes through the normal tool pipeline. The difference is the joiner, which makes the plan able to react to a surprising result
(at most `max_replans` times) at the price of one more call per round."""

from __future__ import annotations

from typing import Any

from aaa.algos.base import Calls, ask, data_block, finish, parse_json_obj, task_text
from aaa.algos.toolplan import PlanStep, _ToolPlanner
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import Message
from aaa.tools import ToolRegistry

MAX_REPLANS = 3
_JOIN_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["finish", "replan"]},
        "answer": {"type": "string"},
        "feedback": {"type": "string"},
    },
    "required": ["action"],
}


class LLMCompiler(_ToolPlanner):
    """Plan, execute in parallel, join.

    Result: `{"text", "plan", "evidence", "rounds"}`. If the joiner keeps asking to replan after `max_replans` rounds, its last answer is used when
    it gave one, otherwise the run ends `NO_SOLUTION` with the evidence."""

    name = "llm_compiler"

    def __init__(self, registry: ToolRegistry, *, max_replans: int = 1, **kw: Any) -> None:
        super().__init__(registry, **kw)
        if not 0 <= max_replans <= MAX_REPLANS:
            raise ValueError(f"max_replans must be between 0 and {MAX_REPLANS}")
        self.max_replans = max_replans

    @property
    def max_calls(self) -> int:
        return 2 * (self.max_replans + 1)

    def _conclude(
        self, e: Event, st: Calls, r: int, steps: dict[str, PlanStep], evidence: dict[str, str]
    ) -> tuple[list[Event], str | None]:
        tag = f"join{r}"
        if not st.done(tag):
            msgs = [
                Message(
                    role="system",
                    content=(
                        f"{self.system} Decide whether the evidence answers the task. Reply with JSON "
                        '{"action": "finish", "answer": "..."} or, if more tool calls are needed, {"action": "replan", "feedback": "what is missing"}.'
                    ),
                ),
                Message(
                    role="user",
                    content=data_block("task", task_text(st.input))
                    + "\n\n"
                    + data_block("evidence", self._evidence_text(steps, evidence)),
                ),
            ]
            return [ask(e, self.name, tag, msgs, model=self.model, schema=_JOIN_SCHEMA)], None
        obj = parse_json_obj(st.text(tag)) or {}
        answer = str(obj.get("answer") or "").strip()
        if obj.get("action") == "replan" and r < self.max_replans:
            return [], str(obj.get("feedback") or "more information is needed")[:500]
        value = {
            "text": answer,
            "plan": [
                {"id": s.id, "tool": s.tool, "arguments": s.arguments} for s in steps.values()
            ],
            "evidence": evidence,
            "rounds": r + 1,
        }
        if not answer:
            return [
                finish(e, OutcomeKind.NO_SOLUTION, value, reason="the joiner gave no answer")
            ], None
        return [finish(e, OutcomeKind.SUCCESS, value)], None
