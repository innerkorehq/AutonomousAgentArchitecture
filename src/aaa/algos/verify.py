"""Draft-then-check algorithms: one model that critiques itself, independent fact-checking questions, and best-of-N with a verifier.

All three keep no state of their own. Each model call is tagged by its place (`d` the draft, `c<r>`/`r<r>` critique and rewrite of round
`r`, `a<i>` the i-th check answer, `g<i>`/`v<i>` the i-th sample and its verification), and every decision is a pure function of
what has been answered so far, so a replay and a different completion order give the same run."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aaa.algos.base import (
    CALLS,
    SCORE_SCHEMA,
    Calls,
    ask,
    clamp_score,
    data_block,
    finish,
    llm_failed,
    parse_json_obj,
    stable,
    task_messages,
    task_text,
    tool_request,
)
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import Message
from aaa.runtime import Ctx

MAX_ROUNDS, MAX_QUESTIONS, MAX_SAMPLES = 8, 12, 32
_SUBS = ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]
_CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "feedback": {"type": "string"}},
    "required": ["ok", "feedback"],
    "additionalProperties": False,
}
_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}


class SelfRefine:
    """One model drafts, critiques its own draft, and rewrites it, until the critique accepts, the rewrite changes nothing, or
    `max_rounds` rewrites have been made. Cheaper than `GeneratorCritic` (no second model) and weaker for the same reason: the
    critic shares the writer's blind spots, so prefer a real check (`GenerateTestRepair`) when you can write one.

    Result: `{"answer", "rounds", "stopped"}` where `stopped` is `accepted`, `unchanged`, `max_rounds` or `unchecked` (a critique call failed)."""

    name = "self_refine"

    def __init__(
        self,
        *,
        max_rounds: int = 3,
        model: str | None = None,
        system: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        if not 1 <= max_rounds <= MAX_ROUNDS:
            raise ValueError(f"max_rounds must be between 1 and {MAX_ROUNDS}")
        self.max_rounds, self.model, self.system, self.params = max_rounds, model, system, params

    @property
    def max_calls(self) -> int:
        return 1 + (self.max_rounds + 1) + self.max_rounds

    def subscriptions(self) -> list[str]:
        return list(_SUBS)

    def _critique(self, e: Event, st: Calls, tag: str, cand: str) -> Event:
        msgs = [
            Message(
                role="system",
                content=(
                    "You are reviewing your own draft. Judge whether it fully and correctly does the task. "
                    'Reply with JSON {"ok": bool, "feedback": "specific, actionable fixes"}.'
                ),
            ),
            Message(
                role="user",
                content=f"{data_block('task', task_text(st.input))}\n\n{data_block('draft', cand)}",
            ),
        ]
        return ask(e, self.name, tag, msgs, model=self.model, schema=_CRITIQUE_SCHEMA)

    def _rewrite(self, e: Event, st: Calls, tag: str, cand: str, feedback: str) -> Event:
        msgs = [
            Message(
                role="system",
                content="Revise the draft to address the feedback. Reply with the complete revised answer only.",
            ),
            Message(
                role="user",
                content="\n\n".join(
                    [
                        data_block("task", task_text(st.input)),
                        data_block("draft", cand),
                        data_block("feedback", feedback),
                    ]
                ),
            ),
        ]
        return ask(e, self.name, tag, msgs, model=self.model, params=self.params)

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        if e.type == "aaa.run.started":
            msgs = task_messages(st.input, self.system)
            return [ask(e, self.name, "d", msgs, model=self.model, params=self.params)]
        if not st.done("d"):
            return []
        if "d" in st.failed:
            return [llm_failed(e)] if e.type == "aaa.llm.failed" else []
        cand, r = st.text("d"), 0

        def end(stopped: str) -> list[Event]:
            return [
                finish(e, OutcomeKind.SUCCESS, {"answer": cand, "rounds": r, "stopped": stopped})
            ]

        while True:
            c = f"c{r}"
            if not st.done(c):
                return [self._critique(e, st, c, cand)]
            if c in st.failed:
                return end("unchecked")
            obj = parse_json_obj(st.text(c)) or {}
            if obj.get("ok") is True:
                return end("accepted")
            if r >= self.max_rounds:
                return end("max_rounds")
            w = f"r{r + 1}"
            if not st.done(w):
                return [self._rewrite(e, st, w, cand, str(obj.get("feedback", "")))]
            if w in st.failed or not st.text(w).strip():
                return end("unchecked")
            if st.text(w).strip() == cand.strip():
                return end("unchanged")
            cand, r = st.text(w), r + 1


class ChainOfVerification:
    """Chain-of-Verification: draft an answer, plan fact-checking questions, answer each question **without seeing the draft** (so
    a mistake in the draft is not repeated), then revise the draft against those answers.

    Result: `{"answer", "draft", "checks": [{"question", "answer"}], "revised"}`."""

    name = "chain_of_verification"

    def __init__(
        self,
        *,
        max_questions: int = 5,
        model: str | None = None,
        system: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        if not 1 <= max_questions <= MAX_QUESTIONS:
            raise ValueError(f"max_questions must be between 1 and {MAX_QUESTIONS}")
        self.max_questions, self.model, self.system, self.params = (
            max_questions,
            model,
            system,
            params,
        )

    @property
    def max_calls(self) -> int:
        return 3 + self.max_questions

    def subscriptions(self) -> list[str]:
        return list(_SUBS)

    def _questions(self, st: Calls) -> list[str]:
        obj = parse_json_obj(st.text("q")) or {}
        raw = obj.get("questions")
        out: list[str] = []
        for q in raw if isinstance(raw, list) else []:
            if isinstance(q, str) and q.strip() and q.strip() not in out:
                out.append(q.strip()[:500])
        return out[: self.max_questions]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        if e.type == "aaa.run.started":
            msgs = task_messages(st.input, self.system)
            return [ask(e, self.name, "d", msgs, model=self.model, params=self.params)]
        if not st.done("d"):
            return []
        if "d" in st.failed:
            return [llm_failed(e)] if e.type == "aaa.llm.failed" else []
        draft = st.text("d")

        def end(checks: list[dict[str, str]], answer: str) -> list[Event]:
            return [
                finish(
                    e,
                    OutcomeKind.SUCCESS,
                    {
                        "answer": answer,
                        "draft": draft,
                        "checks": checks,
                        "revised": answer.strip() != draft.strip(),
                    },
                )
            ]

        if not st.done("q"):
            msgs = [
                Message(
                    role="system",
                    content=(
                        f"List up to {self.max_questions} short, self-contained questions whose answers would check the factual claims "
                        'in the draft. Each question must make sense without the draft. Reply with JSON {"questions": [...]}.'
                    ),
                ),
                Message(
                    role="user",
                    content=f"{data_block('task', task_text(st.input))}\n\n{data_block('draft', draft)}",
                ),
            ]
            return [ask(e, self.name, "q", msgs, model=self.model, schema=_QUESTIONS_SCHEMA)]
        questions = self._questions(st)
        if "q" in st.failed or not questions:
            return end([], draft)
        tags = [f"a{i}" for i in range(len(questions))]
        if not st.all_done(tags):
            return [
                ask(
                    e,
                    self.name,
                    t,
                    [
                        Message(
                            role="system", content="Answer the question briefly and factually."
                        ),
                        Message(role="user", content=q),
                    ],
                    model=self.model,
                )
                for t, q in zip(tags, questions, strict=True)
                if not st.done(t)
            ]
        checks = [
            {"question": q, "answer": st.text(t)}
            for t, q in zip(tags, questions, strict=True)
            if t in st.texts
        ]
        if not checks:
            return end([], draft)
        if not st.done("f"):
            listing = "\n".join(f"Q: {c['question']}\nA: {c['answer']}" for c in checks)
            msgs = [
                Message(
                    role="system",
                    content=(
                        "Revise the draft so it agrees with the verified answers; keep what they support, fix or remove what they "
                        "contradict. Reply with the complete final answer only."
                    ),
                ),
                Message(
                    role="user",
                    content="\n\n".join(
                        [
                            data_block("task", task_text(st.input)),
                            data_block("draft", draft),
                            data_block("verified", listing),
                        ]
                    ),
                ),
            ]
            return [ask(e, self.name, "f", msgs, model=self.model, params=self.params)]
        return end(checks, draft if "f" in st.failed else st.text("f"))


class BestOfN:
    """Sample `n` candidates in parallel, score each with a verifier, return the best.

    The verifier is one of: a pure `score(task, candidate) -> float` (deterministic and free of I/O, because it runs inside the
    algorithm), a tool named by `verifier_tool` (called with `{"task", "candidate"}`; it returns a number or `{"score": number}`), or by
    default a separate model call that rates 0 to 10. With `threshold`, the **lowest-numbered** candidate that reaches it is returned as soon as
    every earlier candidate has been scored (so the answer never depends on which call finished first).

    Result: `{"answer", "score", "scores", "selected", "n", "early_stop"}`."""

    name = "best_of_n"

    def __init__(
        self,
        n: int = 5,
        *,
        score: Callable[[str, str], float] | None = None,
        verifier_tool: str | None = None,
        threshold: float | None = None,
        judge_model: str | None = None,
        model: str | None = None,
        system: str = "",
        temperature: float = 0.8,
        extract: Callable[[str], str] = str.strip,
    ) -> None:
        if not 1 <= n <= MAX_SAMPLES:
            raise ValueError(f"n must be between 1 and {MAX_SAMPLES}")
        if score is not None and verifier_tool is not None:
            raise ValueError("give either score= or verifier_tool=, not both")
        self.n, self.score, self.verifier_tool, self.threshold = n, score, verifier_tool, threshold
        self.judge_model, self.model, self.system = judge_model, model, system
        self.temperature, self.extract = temperature, extract

    @property
    def max_calls(self) -> int:
        return self.n * (1 if self.score is not None else 2)

    def subscriptions(self) -> list[str]:
        subs = list(_SUBS)
        return [*subs, "aaa.tool.completed", "aaa.tool.failed"] if self.verifier_tool else subs

    def _cand(self, st: Calls, i: int) -> str | None:
        return self.extract(st.text(f"g{i}")) if f"g{i}" in st.texts else None

    def _scored(self, st: Calls, i: int) -> tuple[bool, float | None]:
        """(settled, score): a candidate is settled once it failed to generate or its verification has an answer."""
        cand = self._cand(st, i)
        if cand is None:
            return (f"g{i}" in st.failed, None)
        if self.score is not None:
            try:
                return (True, float(self.score(task_text(st.input), cand)))
            except Exception:  # noqa: BLE001 - a verifier that raises scores that candidate as unusable
                return (True, None)
        v = f"v{i}"
        if not st.done(v):
            return (False, None)
        if self.verifier_tool is not None:
            r = st.results.get(v)
            r = r.get("score") if isinstance(r, dict) else r
            try:
                return (True, float(r))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return (True, None)
        return (True, clamp_score(st.text(v)))

    def _verify(self, e: Event, st: Calls, i: int, cand: str) -> Event:
        task = task_text(st.input)
        if self.verifier_tool is not None:
            return stable(
                tool_request(e, f"v{i}", self.verifier_tool, {"task": task, "candidate": cand}),
                f"{self.name}:v{i}",
            )
        msgs = [
            Message(
                role="system",
                content='Rate how well the candidate answers the task, from 0 (wrong) to 10 (excellent). Reply with JSON {"score": number}.',
            ),
            Message(
                role="user",
                content=f"{data_block('task', task)}\n\n{data_block('candidate', cand)}",
            ),
        ]
        return ask(
            e, self.name, f"v{i}", msgs, model=self.judge_model or self.model, schema=SCORE_SCHEMA
        )

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        if e.type == "aaa.run.started":
            return [
                ask(
                    e,
                    self.name,
                    f"g{i}",
                    task_messages(st.input, self.system),
                    model=self.model,
                    params={"temperature": self.temperature},
                )
                for i in range(self.n)
            ]
        out: list[Event] = []
        if self.score is None:
            for i in range(self.n):
                cand = self._cand(st, i)
                if cand is not None and not st.done(f"v{i}"):
                    out.append(self._verify(e, st, i, cand))
        scored = [self._scored(st, i) for i in range(self.n)]
        for i, (settled, sc) in enumerate(scored):
            if not settled:
                return out
            if self.threshold is not None and sc is not None and sc >= self.threshold:
                return [self._end(e, st, scored, i, early=True)]
        usable = [(sc, -i) for i, (_, sc) in enumerate(scored) if sc is not None]
        if not usable:
            return [finish(e, OutcomeKind.FAILED, reason="no candidate could be scored")]
        return [self._end(e, st, scored, -max(usable)[1], early=False)]

    def _end(
        self,
        e: Event,
        st: Calls,
        scored: list[tuple[bool, float | None]],
        i: int,
        *,
        early: bool,
    ) -> Event:
        return finish(
            e,
            OutcomeKind.SUCCESS,
            {
                "answer": self._cand(st, i),
                "score": scored[i][1],
                "scores": [sc for _, sc in scored],
                "selected": i,
                "n": self.n,
                "early_stop": early,
            },
        )
