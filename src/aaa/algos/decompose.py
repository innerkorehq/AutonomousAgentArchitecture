"""Question-decomposition loops: solve easier sub-questions first (`LeastToMost`), or let the model ask itself what it needs (`SelfAsk`).

Cheaper than a full plan (`PlanAndExecute`): no executor runs, no plan object, a hard cap on how many sub-questions can exist.
Model calls are tagged by position (`dec`, `s<i>` for sub-question `i`; `t<k>` for turn `k` and `f<k>` for its follow-up), and
every decision is a pure function of the answers so far."""

from __future__ import annotations

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

MAX_SUBQUESTIONS, MAX_FOLLOWUPS = 12, 10
_SUBS = ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]
_DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {"subquestions": {"type": "array", "items": {"type": "string"}}},
    "required": ["subquestions"],
    "additionalProperties": False,
}
_FINAL = "so the final answer is:"
_FOLLOW = "follow up:"


def _listing(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs) or "(none yet)"


class LeastToMost:
    """Least-to-most prompting: the model splits the question into at most `max_subquestions` easier ones, then answers them in order,
    each answer seeing the earlier ones; the last step is the original question. If the split fails or comes back empty the
    original question is answered directly.

    Result: `{"answer", "steps": [{"question", "answer"}]}` (the final step is the original question)."""

    name = "least_to_most"

    def __init__(
        self,
        *,
        max_subquestions: int = 6,
        model: str | None = None,
        system: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        if not 1 <= max_subquestions <= MAX_SUBQUESTIONS:
            raise ValueError(f"max_subquestions must be between 1 and {MAX_SUBQUESTIONS}")
        self.max_subquestions, self.model, self.system, self.params = (
            max_subquestions,
            model,
            system,
            params,
        )

    @property
    def max_calls(self) -> int:
        return 1 + self.max_subquestions + 1

    def subscriptions(self) -> list[str]:
        return list(_SUBS)

    def _subquestions(self, st: Calls) -> list[str]:
        raw = (parse_json_obj(st.text("dec")) or {}).get("subquestions")
        out: list[str] = []
        for q in raw if isinstance(raw, list) else []:
            if isinstance(q, str) and q.strip() and q.strip() not in out:
                out.append(q.strip()[:500])
        return out[: self.max_subquestions]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        task = task_text(st.input)
        if e.type == "aaa.run.started":
            msgs = [
                Message(
                    role="system",
                    content=(
                        f"Break the question into at most {self.max_subquestions} simpler sub-questions, easiest first, so that answering "
                        'them in order leads to the answer. Reply with JSON {"subquestions": [...]}.'
                    ),
                ),
                Message(role="user", content=data_block("question", task)),
            ]
            return [ask(e, self.name, "dec", msgs, model=self.model, schema=_DECOMPOSE_SCHEMA)]
        if not st.done("dec"):
            return []
        steps = [*self._subquestions(st), task]
        done: list[tuple[str, str]] = []
        for i, q in enumerate(steps):
            tag = f"s{i}"
            if not st.done(tag):
                msgs = [
                    Message(
                        role="system",
                        content=self.system
                        or "Answer the question, using the earlier answers where they help. Be concise.",
                    ),
                    Message(
                        role="user",
                        content="\n\n".join(
                            [
                                data_block("original_question", task),
                                data_block("earlier_answers", _listing(done)),
                                data_block("question", q),
                            ]
                        ),
                    ),
                ]
                return [ask(e, self.name, tag, msgs, model=self.model, params=self.params)]
            if tag in st.failed:
                if i == len(steps) - 1:
                    return (
                        [llm_failed(e)]
                        if e.type == "aaa.llm.failed"
                        else [
                            finish(
                                e,
                                OutcomeKind.FAILED,
                                reason="the final question could not be answered",
                            )
                        ]
                    )
                done.append((q, "(no answer)"))
            else:
                done.append((q, st.text(tag)))
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "answer": done[-1][1],
                    "steps": [{"question": q, "answer": a} for q, a in done],
                },
            )
        ]


class SelfAsk:
    """Self-ask: the model answers directly or writes `Follow up: <question>`; the follow-up is answered (by a separate model call, or
    by `lookup_tool` called with `{"query": question}`) and the loop continues, for at most `max_followups` follow-ups, then the
    model must give `So the final answer is: ...`. A reply in neither form is taken as the final answer.

    Result: `{"answer", "followups": [{"question", "answer"}], "turns"}`."""

    name = "self_ask"

    def __init__(
        self,
        *,
        max_followups: int = 4,
        lookup_tool: str | None = None,
        model: str | None = None,
        system: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        if not 1 <= max_followups <= MAX_FOLLOWUPS:
            raise ValueError(f"max_followups must be between 1 and {MAX_FOLLOWUPS}")
        self.max_followups, self.lookup_tool = max_followups, lookup_tool
        self.model, self.system, self.params = model, system, params

    @property
    def max_calls(self) -> int:
        return (self.max_followups + 1) + (0 if self.lookup_tool else self.max_followups)

    def subscriptions(self) -> list[str]:
        subs = list(_SUBS)
        return [*subs, "aaa.tool.completed", "aaa.tool.failed"] if self.lookup_tool else subs

    @staticmethod
    def _parse(reply: str) -> tuple[str, str]:
        """("final", answer) or ("follow", question) from one turn's reply."""
        lines = [ln.strip() for ln in reply.strip().splitlines() if ln.strip()]
        for ln in reversed(lines):
            if ln.lower().startswith(_FINAL):
                return ("final", ln[len(_FINAL) :].strip())
        for ln in lines:
            if ln.lower().startswith(_FOLLOW) and ln[len(_FOLLOW) :].strip():
                return ("follow", ln[len(_FOLLOW) :].strip()[:500])
        return ("final", reply.strip())

    def _turn(self, e: Event, st: Calls, k: int, trail: list[tuple[str, str]]) -> Event:
        last = k >= self.max_followups
        system = (
            self.system
            or "Answer the question. If you need a fact first, reply with one line 'Follow up: <question>' and nothing else. "
            "When you can answer, reply 'So the final answer is: <answer>'."
        ) + (" You must answer now." if last else "")
        msgs = [
            Message(role="system", content=system),
            Message(
                role="user",
                content=f"{data_block('question', task_text(st.input))}\n\n{data_block('follow_ups', _listing(trail))}",
            ),
        ]
        return ask(e, self.name, f"t{k}", msgs, model=self.model, params=self.params)

    def _followup(self, e: Event, k: int, question: str) -> Event:
        if self.lookup_tool is not None:
            return stable(
                tool_request(e, f"f{k}", self.lookup_tool, {"query": question}),
                f"{self.name}:f{k}",
            )
        msgs = [
            Message(role="system", content="Answer the question briefly and factually."),
            Message(role="user", content=question),
        ]
        return ask(e, self.name, f"f{k}", msgs, model=self.model)

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        trail: list[tuple[str, str]] = []
        for k in range(self.max_followups + 1):
            t = f"t{k}"
            if not st.done(t):
                return [self._turn(e, st, k, trail)]
            if t in st.failed:
                return (
                    [llm_failed(e)]
                    if e.type == "aaa.llm.failed"
                    else [finish(e, OutcomeKind.FAILED, reason="the model call failed")]
                )
            kind, body = self._parse(st.text(t))
            if kind == "final" or k >= self.max_followups:
                return [
                    finish(
                        e,
                        OutcomeKind.SUCCESS,
                        {
                            "answer": body,
                            "followups": [{"question": q, "answer": a} for q, a in trail],
                            "turns": k + 1,
                        },
                    )
                ]
            f = f"f{k}"
            if not st.done(f):
                return [self._followup(e, k, body)]
            if f in st.failed:
                trail.append((body, "(no answer)"))
            elif f in st.results:
                trail.append((body, render_result(st.results[f])[:2000]))
            else:
                trail.append((body, st.text(f)))
        return []
