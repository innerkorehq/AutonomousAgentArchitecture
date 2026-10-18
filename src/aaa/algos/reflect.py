"""Reflection over memory (Generative Agents style): turn a pile of recent memories into a few higher-level insights, and remember those.

`MemoryReflection` is an algorithm you run now and then (after a conversation, on a schedule): it lists the scope's recent memories, scores how important
each one is, and **only if enough has happened** asks what the most salient questions are, retrieves memories for each (ranked by recency, importance and
relevance together), asks for insights that cite their sources, and writes the insights back.

The controls, since reflections are read by later runs and so are an attack path (a poisoned page becoming a "lesson"):

* the scope is the run's (tenant, user, agent), from the host, never from model output; the tools reach no other scope;
* memories are shown to the model as quoted data; an insight must cite memory ids that were shown to it for that question, and is dropped otherwise;
* what is written is only the insight text (length-capped), marked `reflection`, with its sources and the run that made it, so it can be audited and deleted;
* the number of insights, questions and recent memories is capped at construction."""

from __future__ import annotations

import math
from datetime import UTC, datetime
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
    tool_request,
)
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import Message
from aaa.memory.base import Listable, Memory, require
from aaa.memory.tools import scope_from_tool_context
from aaa.memory.types import MemoryEntry, MemoryItem
from aaa.runtime import Ctx
from aaa.tools import Tool, ToolRegistry
from aaa.tools.tool import ToolContext, tool

MAX_WINDOW, MAX_QUESTIONS, MAX_INSIGHTS, MAX_INSIGHT_CHARS = 100, 5, 20, 400
_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}
_INSIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "sources"],
            },
        }
    },
    "required": ["insights"],
    "additionalProperties": False,
}


def rank_memories(
    items: list[MemoryItem],
    importance: dict[str, float],
    now: datetime,
    *,
    half_life_hours: float = 72.0,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[MemoryItem]:
    """Order memories by recency + importance + relevance (each scaled to 0..1 across `items`), best first; ties keep the input order.

    Recency halves every `half_life_hours`; importance is the 1 to 10 score in `importance[id]` (5 when unknown); relevance is the memory's own search
    `score`. Pure, so it can be tested and tuned."""
    if not items:
        return []

    def scale(values: list[float]) -> list[float]:
        lo, hi = min(values), max(values)
        return (
            [1.0] * len(values) if math.isclose(lo, hi) else [(v - lo) / (hi - lo) for v in values]
        )

    def age(m: MemoryItem) -> float:
        when = m.created_at or m.valid_at
        return max(0.0, (now - when).total_seconds() / 3600.0) if when else half_life_hours

    recency = scale([0.5 ** (age(m) / half_life_hours) for m in items])
    imp = scale([float(importance.get(m.id, 5.0)) for m in items])
    rel = scale([float(m.score or 0.0) for m in items])
    wr, wi, wl = weights
    scored = [(wr * recency[i] + wi * imp[i] + wl * rel[i], -i) for i in range(len(items))]
    order = sorted(range(len(items)), key=lambda i: scored[i], reverse=True)
    return [items[i] for i in order]


def _importance_of(m: dict[str, Any]) -> float | None:
    v = (m.get("meta") or {}).get("importance")
    return float(v) if isinstance(v, (int, float)) else None


def reflection_tools(memory: Memory, *, window: int, k: int = 8) -> list[Tool]:
    """The three tools `MemoryReflection` calls. They act only on the scope of the run that calls them."""
    require(memory, Listable)

    @tool
    async def reflect_recent(ctx: ToolContext) -> list[dict[str, Any]]:
        """The most recent memories of this scope."""
        items = await memory.list(scope_from_tool_context(ctx), limit=window)  # type: ignore[attr-defined]
        items = sorted(
            items, key=lambda m: m.created_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )[:window]
        return [
            {
                "id": m.id,
                "text": m.text[:600],
                "meta": {"importance": m.meta.get("importance")} if "importance" in m.meta else {},
            }
            for m in items
        ]

    @tool
    async def reflect_retrieve(query: str, ctx: ToolContext) -> list[dict[str, Any]]:
        """The memories most relevant to a question, ranked by recency, importance and relevance."""
        found = await memory.search(query, scope_from_tool_context(ctx), k=k * 2)
        imp = {
            m.id: float(m.meta["importance"])
            for m in found
            if isinstance(m.meta.get("importance"), (int, float))
        }
        ranked = rank_memories(found, imp, datetime.now(UTC))[:k]
        return [{"id": m.id, "text": m.text[:600]} for m in ranked]

    @tool(delivery="at_most_once")
    async def reflect_write(insights: list[dict[str, Any]], ctx: ToolContext) -> dict[str, Any]:
        """Remember insights. Each is {"text", "sources": [memory ids], "importance"}."""
        entries = []
        for item in insights[:MAX_INSIGHTS]:
            body = str(item.get("text", "")).strip()[:MAX_INSIGHT_CHARS]
            if not body:
                continue
            sources = [str(s)[:100] for s in (item.get("sources") or [])][:10]
            try:
                importance = max(1.0, min(10.0, float(item.get("importance", 7))))
            except (TypeError, ValueError):
                importance = 7.0
            entries.append(
                MemoryEntry(
                    text=body,
                    kind="fact",
                    meta={
                        "reflection": True,
                        "sources": sources,
                        "importance": importance,
                        "run_id": ctx.run_id,
                    },
                )
            )
        res = (
            await memory.add(entries, scope_from_tool_context(ctx), key=ctx.idempotency_key)
            if entries
            else None
        )
        return {"written": len(entries), "ids": res.ids if res else []}

    return [reflect_recent, reflect_retrieve, reflect_write]


class MemoryReflection:
    """Reflect on the recent memories of the run's scope.

    `threshold` is the summed importance (each memory 1 to 10) the recent window must reach before anything is asked; below it the run ends
    `{"reflected": False}` having spent only the scoring calls. Attach `ToolHandler(reflection.tools)`.

    Result: `{"reflected", "importance", "questions", "insights": [{"text", "sources"}], "written"}`. At most `window + 1 + questions` model calls."""

    name = "memory_reflection"

    def __init__(
        self,
        memory: Memory,
        *,
        window: int = 20,
        threshold: float = 40.0,
        questions: int = 3,
        insights_per_question: int = 2,
        k: int = 8,
        model: str | None = None,
        scorer_model: str | None = None,
    ) -> None:
        if not 1 <= window <= MAX_WINDOW:
            raise ValueError(f"window must be between 1 and {MAX_WINDOW}")
        if not 1 <= questions <= MAX_QUESTIONS:
            raise ValueError(f"questions must be between 1 and {MAX_QUESTIONS}")
        if not 1 <= insights_per_question * questions <= MAX_INSIGHTS:
            raise ValueError(
                f"questions * insights_per_question must be between 1 and {MAX_INSIGHTS}"
            )
        self.window, self.threshold, self.questions = window, threshold, questions
        self.per_question, self.k, self.model, self.scorer_model = (
            insights_per_question,
            k,
            model,
            scorer_model,
        )
        self.tools = ToolRegistry(reflection_tools(memory, window=window, k=k))

    @property
    def max_calls(self) -> int:
        return self.window + 1 + self.questions

    def subscriptions(self) -> list[str]:
        return [
            "aaa.run.started",
            "aaa.llm.completed",
            "aaa.llm.failed",
            "aaa.tool.completed",
            "aaa.tool.failed",
        ]

    def _score(self, e: Event, i: int, text: str) -> Event:
        msgs = [
            Message(
                role="system",
                content='Rate how important this memory is to remember, from 1 (mundane) to 10 (life-changing or decisive). Reply with JSON {"score": number}.',
            ),
            Message(role="user", content=data_block("memory", text)),
        ]
        return ask(
            e,
            self.name,
            f"imp.{i}",
            msgs,
            model=self.scorer_model or self.model,
            schema=SCORE_SCHEMA,
        )

    def _end(
        self,
        e: Event,
        reflected: bool,
        total: float,
        questions: list[str],
        insights: list[dict[str, Any]],
        written: int,
    ) -> list[Event]:
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "reflected": reflected,
                    "importance": total,
                    "questions": questions,
                    "insights": insights,
                    "written": written,
                },
            )
        ]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        if e.type == "aaa.llm.failed":
            return [llm_failed(e)]
        st: Calls = await ctx.project(CALLS)
        if not st.done("recent"):
            return [stable(tool_request(e, "recent", "reflect_recent", {}), f"{self.name}:recent")]
        recent = st.results.get("recent") or []
        recent = [m for m in recent if isinstance(m, dict) and m.get("id") and m.get("text")][
            : self.window
        ]
        if not recent:
            return self._end(e, False, 0.0, [], [], 0)
        known = {m["id"]: _importance_of(m) for m in recent}
        todo = [i for i, m in enumerate(recent) if known[m["id"]] is None]
        if not st.all_done([f"imp.{i}" for i in todo]):
            return [self._score(e, i, recent[i]["text"]) for i in todo if not st.done(f"imp.{i}")]
        total = 0.0
        for i, m in enumerate(recent):
            v = known[m["id"]]
            total += v if v is not None else (clamp_score(st.text(f"imp.{i}"), 1.0, 10.0) or 5.0)
        if total < self.threshold:
            return self._end(e, False, total, [], [], 0)
        if not st.done("questions"):
            listing = "\n".join(f"[{m['id']}] {m['text']}" for m in recent)
            msgs = [
                Message(
                    role="system",
                    content=f'Given these memories, what are the {self.questions} most salient high-level questions they raise? Reply with JSON {{"questions": [...]}}.',
                ),
                Message(role="user", content=data_block("memories", listing)),
            ]
            return [
                ask(e, self.name, "questions", msgs, model=self.model, schema=_QUESTIONS_SCHEMA)
            ]
        raw = (parse_json_obj(st.text("questions")) or {}).get("questions")
        questions = (
            [q.strip()[:300] for q in raw if isinstance(q, str) and q.strip()][: self.questions]
            if isinstance(raw, list)
            else []
        )
        if not questions:
            return self._end(e, False, total, [], [], 0)
        retrievals = [f"ret.{j}" for j in range(len(questions))]
        if not st.all_done(retrievals):
            return [
                stable(tool_request(e, r, "reflect_retrieve", {"query": q}), f"{self.name}:{r}")
                for r, q in zip(retrievals, questions, strict=True)
                if not st.done(r)
            ]
        asks = []
        for j, q in enumerate(questions):
            if st.done(f"ins.{j}"):
                continue
            shown = st.results.get(f"ret.{j}") or []
            listing = "\n".join(
                f"[{m['id']}] {m['text']}"
                for m in shown
                if isinstance(m, dict) and "id" in m and "text" in m
            )
            msgs = [
                Message(
                    role="system",
                    content=(
                        f"Write up to {self.per_question} high-level insights that answer the question from these memories. Each insight must list the ids of the memories it rests on. "
                        'Reply with JSON {"insights": [{"text": "...", "sources": ["id"]}]}.'
                    ),
                ),
                Message(
                    role="user",
                    content=f"{data_block('question', q)}\n\n{data_block('memories', listing or '(none)')}",
                ),
            ]
            asks.append(
                ask(e, self.name, f"ins.{j}", msgs, model=self.model, schema=_INSIGHTS_SCHEMA)
            )
        if asks:
            return asks
        insights: list[dict[str, Any]] = []
        for j in range(len(questions)):
            shown_ids = {
                m["id"]
                for m in (st.results.get(f"ret.{j}") or [])
                if isinstance(m, dict) and "id" in m
            }
            raw_i = (parse_json_obj(st.text(f"ins.{j}")) or {}).get("insights")
            for item in raw_i[: self.per_question] if isinstance(raw_i, list) else []:
                if not isinstance(item, dict):
                    continue
                body = str(item.get("text", "")).strip()[:MAX_INSIGHT_CHARS]
                src = [
                    s for s in (item.get("sources") or []) if isinstance(s, str) and s in shown_ids
                ]
                if body and src and body not in {x["text"] for x in insights}:
                    insights.append({"text": body, "sources": src})
        if not insights:
            return self._end(e, True, total, questions, [], 0)
        if not st.done("write"):
            return [
                stable(
                    tool_request(e, "write", "reflect_write", {"insights": insights}),
                    f"{self.name}:write",
                )
            ]
        written = (
            int((st.results.get("write") or {}).get("written", 0)) if "write" in st.results else 0
        )
        return self._end(e, True, total, questions, insights, written)


__all__ = ["MemoryReflection", "rank_memories", "reflection_tools"]
