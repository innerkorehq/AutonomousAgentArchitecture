"""Plan the whole tool-call graph first, run it without asking the model again, then answer: `ReWOO` and `LLMCompiler`.

The planner writes steps like `{"id": "E2", "tool": "calc", "arguments": {"expr": "#E1 * 2"}}`. A `#E1` in an argument stands for the
result of step E1; independent steps run in parallel through the normal tool pipeline (approval, timeouts, guards, budgets). Because the
plan is data checked before anything runs, a model cannot name a tool it was not given, build an argument by evaluating text, or create a cycle
(a step may only refer to earlier steps). A tool result that is substituted into a later step's arguments is plain data in that argument.

All state is the answers so far (`Calls`): the plan is parsed again from its text on each event, which is cheap and keeps replay exact."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from aaa.algos.base import (
    CALLS,
    Calls,
    ask,
    data_block,
    finish,
    llm_failed,
    parse_json_obj,
    render_result,
    stable,
    task_text,
    tool_request,
)
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import Message
from aaa.runtime import Ctx
from aaa.tools import ToolRegistry

MAX_STEPS, MAX_ARG_CHARS = 24, 20_000
_ID = re.compile(r"^E\d{1,3}$")
_REF = re.compile(r"#(E\d{1,3})")
_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "tool", "arguments"],
            },
        }
    },
    "required": ["steps"],
}


@dataclass(frozen=True)
class PlanStep:
    id: str
    tool: str
    arguments: dict[str, Any]
    depends_on: tuple[str, ...] = field(default_factory=tuple)


def references(value: Any) -> set[str]:
    """Every `#E<n>` mentioned in the strings of a JSON value."""
    if isinstance(value, str):
        return set(_REF.findall(value))
    if isinstance(value, list):
        return set().union(*(references(v) for v in value)) if value else set()
    if isinstance(value, dict):
        return set().union(*(references(v) for v in value.values())) if value else set()
    return set()


def substitute(value: Any, results: dict[str, Any], limit: int) -> Any:
    """`value` with each `#E<n>` replaced by that step's result: a string that is exactly `#E1` becomes the raw result, a mention inside
    longer text becomes the result rendered as text (cut at `limit` characters). Pure string work: nothing is evaluated."""
    if isinstance(value, str):
        whole = _REF.fullmatch(value)
        if whole:
            return results[whole.group(1)]
        return _REF.sub(lambda m: render_result(results[m.group(1)])[:limit], value)
    if isinstance(value, list):
        return [substitute(v, results, limit) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, results, limit) for k, v in value.items()}
    return value


def parse_plan(
    text: str, allowed: set[str], known: set[str], room: int
) -> tuple[list[PlanStep], str | None]:
    """(steps, None) when the plan is acceptable, else ([], why). `known` are ids from earlier rounds a step may refer to."""
    obj = parse_json_obj(text)
    raw = (obj or {}).get("steps")
    if not isinstance(raw, list) or not raw:
        return [], "the plan has no steps"
    if len(raw) > room:
        return [], f"the plan has {len(raw)} steps (at most {room} more are allowed)"
    seen = set(known)
    out: list[PlanStep] = []
    for item in raw:
        if not isinstance(item, dict):
            return [], "a step is not an object"
        sid, tool, args = item.get("id"), item.get("tool"), item.get("arguments", {})
        if not isinstance(sid, str) or not _ID.match(sid) or sid in seen:
            return [], f"bad or repeated step id {sid!r} (use E1, E2, ...)"
        if not isinstance(tool, str) or tool not in allowed:
            return [], f"step {sid} names a tool that is not available"
        if not isinstance(args, dict) or len(json.dumps(args, default=str)) > MAX_ARG_CHARS:
            return [], f"step {sid} has unusable arguments"
        explicit = item.get("depends_on") or []
        if not isinstance(explicit, list) or not all(isinstance(x, str) for x in explicit):
            return [], f"step {sid} has bad depends_on"
        deps = references(args) | set(explicit)
        if bad := sorted(deps - seen):
            return [], f"step {sid} refers to {bad}, which is not an earlier step"
        out.append(PlanStep(sid, tool, args, tuple(sorted(deps))))
        seen.add(sid)
    return out, None


class _ToolPlanner:
    """Shared driver. Subclasses say how a finished plan is concluded (`_conclude`)."""

    name = "tool_planner"

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        tool_names: list[str] | None = None,
        max_steps: int = 8,
        on_step_error: str = "fail",
        max_evidence_chars: int = 4000,
        model: str | None = None,
        system: str = "",
    ) -> None:
        names = tool_names if tool_names is not None else registry.names()
        if missing := [n for n in names if n not in registry]:
            raise ValueError(f"tool_names not in the registry: {missing}")
        if not names:
            raise ValueError("a tool planner needs at least one tool")
        if not 1 <= max_steps <= MAX_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_STEPS}")
        if on_step_error not in ("fail", "continue"):
            raise ValueError("on_step_error must be 'fail' or 'continue'")
        self.registry, self.tool_names, self.max_steps = registry, list(names), max_steps
        self.on_step_error, self.limit = on_step_error, max_evidence_chars
        self.model, self.system = model, system

    def subscriptions(self) -> list[str]:
        return [
            "aaa.run.started",
            "aaa.llm.completed",
            "aaa.llm.failed",
            "aaa.tool.completed",
            "aaa.tool.failed",
        ]

    # -- prompts -----------------------------------------------------------------------------
    def _catalog(self) -> str:
        lines = []
        for n in self.tool_names:
            t = self.registry.get(n)
            assert t is not None
            lines.append(f"- {n}: {t.description} arguments: {json.dumps(t.parameters)[:600]}")
        return "\n".join(lines)

    def _plan_prompt(
        self, st: Calls, known: dict[str, PlanStep], evidence: dict[str, str], feedback: str
    ) -> list[Message]:
        extra = ""
        if known:
            extra = (
                "\n\nThe plan so far ran these steps:\n"
                + "\n".join(f"{k}: {s.tool} -> {evidence.get(k, '')}" for k, s in known.items())
                + f"\n\nFeedback: {feedback}\nWrite ONLY the additional steps needed, numbered after {max(known)}."
            )
        system = (
            f"{self.system} Plan the tool calls needed to answer the task. Reply with JSON "
            '{"steps": [{"id": "E1", "tool": "<name>", "arguments": {...}}]}. Write #E1 in an argument to use the result of step E1 '
            f"(only earlier steps). Independent steps run in parallel. At most {self.max_steps} steps in all.\n\nTools:\n{self._catalog()}"
        )
        return [
            Message(role="system", content=system),
            Message(role="user", content=data_block("task", task_text(st.input)) + extra),
        ]

    # -- the walk ----------------------------------------------------------------------------
    def _evidence_text(self, steps: dict[str, PlanStep], evidence: dict[str, str]) -> str:
        return "\n".join(f"{k} ({s.tool}): {evidence.get(k, '')}" for k, s in steps.items())

    def _conclude(
        self, e: Event, st: Calls, r: int, steps: dict[str, PlanStep], evidence: dict[str, str]
    ) -> tuple[list[Event], str | None]:
        """(events, None) to finish or wait; ([], feedback) to plan another round."""
        raise NotImplementedError

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        if e.type == "aaa.llm.failed":
            return [llm_failed(e)]
        st = await ctx.project(CALLS)
        known: dict[str, PlanStep] = {}
        results: dict[str, Any] = {}
        evidence: dict[str, str] = {}
        feedback = ""
        r = 0
        while True:
            ptag = "plan" if r == 0 else f"plan{r}"
            if not st.done(ptag):
                msgs = self._plan_prompt(st, known, evidence, feedback)
                return [ask(e, self.name, ptag, msgs, model=self.model, schema=_PLAN_SCHEMA)]
            steps, why = parse_plan(
                st.text(ptag), set(self.tool_names), set(known), self.max_steps - len(known)
            )
            if why:
                return [finish(e, OutcomeKind.FAILED, reason=f"invalid plan: {why}")]
            known.update({s.id: s for s in steps})
            out: list[Event] = []
            for s in steps:
                if s.id in st.failed:
                    if self.on_step_error == "fail":
                        return [
                            finish(e, OutcomeKind.FAILED, reason=f"step {s.id} ({s.tool}) failed")
                        ]
                    results[s.id], evidence[s.id] = "(step failed)", "(step failed)"
                elif s.id in st.results:
                    results[s.id] = st.results[s.id]
                    evidence[s.id] = render_result(st.results[s.id])[: self.limit]
            for s in steps:
                if s.id in results or s.id in st.failed:
                    continue
                if all(d in results for d in s.depends_on):
                    args = substitute(s.arguments, results, self.limit)
                    out.append(stable(tool_request(e, s.id, s.tool, args), f"{self.name}:{s.id}"))
            if any(s.id not in results for s in steps):
                return out
            events, feedback_next = self._conclude(e, st, r, known, evidence)
            if feedback_next is None:
                return events
            feedback, r = feedback_next, r + 1


class ReWOO(_ToolPlanner):
    """Reasoning WithOut Observation: one planning call writes every tool call up front (with `#E1`-style placeholders for earlier
    results), the tools run without the model in the loop, and one solver call writes the answer from the evidence. Two model calls
    however many tools there are, against ReAct's one per step; the price is that the plan cannot react to a surprising result.

    Result: `{"text", "plan": [{"id", "tool", "arguments"}], "evidence": {id: result}, "rounds": 1}`."""

    name = "rewoo"

    @property
    def max_calls(self) -> int:
        return 2

    def _conclude(
        self, e: Event, st: Calls, r: int, steps: dict[str, PlanStep], evidence: dict[str, str]
    ) -> tuple[list[Event], str | None]:
        if not st.done("solve"):
            msgs = [
                Message(
                    role="system",
                    content=f"{self.system} Answer the task using only the evidence below. Reply with the answer only.",
                ),
                Message(
                    role="user",
                    content=data_block("task", task_text(st.input))
                    + "\n\n"
                    + data_block("evidence", self._evidence_text(steps, evidence)),
                ),
            ]
            return [ask(e, self.name, "solve", msgs, model=self.model)], None
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "text": st.text("solve"),
                    "plan": [
                        {"id": s.id, "tool": s.tool, "arguments": s.arguments}
                        for s in steps.values()
                    ],
                    "evidence": evidence,
                    "rounds": r + 1,
                },
            )
        ], None
