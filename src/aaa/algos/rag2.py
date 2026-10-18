"""Retrieval algorithms that check what they retrieved: `CorrectiveRAG`, `SelfRAG` and `AdaptiveRAG`.

`RAGAgent` lets the model decide when to search. These three put the *checking* in the algorithm instead: each retrieved passage is graded
before it is used, a poor result triggers a rewritten query (and, for `CorrectiveRAG`, an operator-chosen fallback tool), and the
amount of retrieval is matched to the question. Retrieval goes through the normal tool pipeline (`rag.tools` is attached with a `ToolHandler`, like
`RAGAgent`), so tenancy, guards and budgets apply. Retrieved text is always shown to the model as quoted data."""

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
from aaa.data.index import Index
from aaa.events import Event
from aaa.llm.types import Message
from aaa.runtime import Ctx
from aaa.tools import Tool, ToolRegistry, search_tool

MAX_K, MAX_ROUNDS, MAX_HOPS = 10, 3, 5
Passage = tuple[str, str]
_GRADE_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
    "additionalProperties": False,
}
_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}
_ANSWER_SYSTEM = (
    "Answer the question using only the passages. Cite passage ids like [id]. "
    "If the passages do not contain the answer, say so instead of guessing."
)
_SUBS = [
    "aaa.run.started",
    "aaa.llm.completed",
    "aaa.llm.failed",
    "aaa.tool.completed",
    "aaa.tool.failed",
]


class _Rag:
    """Retrieval plumbing shared by the three algorithms."""

    name = "rag"

    def __init__(
        self,
        index: Index,
        *,
        k: int = 5,
        max_chars: int = 800,
        model: str | None = None,
        grader_model: str | None = None,
        system: str = _ANSWER_SYSTEM,
        extra_tools: list[Tool] | None = None,
    ) -> None:
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k must be between 1 and {MAX_K}")
        self.index, self.k, self.max_chars = index, k, max_chars
        self.model, self.grader_model, self.system = model, grader_model, system
        search = search_tool(index, k=k, max_chars=max_chars)
        self.search_name = search.name
        self.tools = ToolRegistry([search, *(extra_tools or [])])

    def subscriptions(self) -> list[str]:
        return list(_SUBS)

    # -- passages ----------------------------------------------------------------------------
    def _passages(self, raw: Any, prefix: str = "") -> list[Passage]:
        items = raw if isinstance(raw, list) else [raw]
        out: list[Passage] = []
        for n, it in enumerate(items):
            if isinstance(it, dict):
                body = str(it.get("text") or it.get("content") or it.get("snippet") or "")
                pid = str(it.get("id") or it.get("url") or f"p{n + 1}")
            else:
                body, pid = render_result(it), f"p{n + 1}"
            if body.strip():
                out.append((f"{prefix}{pid}"[:80], body.strip()[: self.max_chars]))
        return out[: self.k]

    def _search(self, e: Event, call_id: str, tool: str, query: str) -> Event:
        return stable(tool_request(e, call_id, tool, {"query": query}), f"{self.name}:{call_id}")

    def _grade(self, e: Event, st: Calls, tag: str, passage: Passage) -> Event:
        msgs = [
            Message(
                role="system",
                content='Judge whether the passage helps answer the question. Reply with JSON {"relevant": true|false}.',
            ),
            Message(
                role="user",
                content=f"{data_block('question', task_text(st.input))}\n\n{data_block('passage', f'[{passage[0]}] {passage[1]}')}",
            ),
        ]
        return ask(
            e, self.name, tag, msgs, model=self.grader_model or self.model, schema=_GRADE_SCHEMA
        )

    def _relevant(self, st: Calls, prefix: str, passages: list[Passage]) -> list[Passage]:
        return [
            p
            for i, p in enumerate(passages)
            if (parse_json_obj(st.text(f"{prefix}.{i}")) or {}).get("relevant") is True
        ]

    def _answer(self, e: Event, st: Calls, tag: str, passages: list[Passage]) -> Event:
        listing = "\n".join(f"[{pid}] {body}" for pid, body in passages) or "(no passages)"
        msgs = [
            Message(role="system", content=self.system),
            Message(
                role="user",
                content=f"{data_block('question', task_text(st.input))}\n\n{data_block('passages', listing)}",
            ),
        ]
        return ask(e, self.name, tag, msgs, model=self.model)

    def _rewrite(self, e: Event, st: Calls, tag: str, queries: list[str]) -> Event:
        msgs = [
            Message(
                role="system",
                content='The searches so far did not find what is needed. Write one better search query. Reply with JSON {"query": "..."}.',
            ),
            Message(
                role="user",
                content=f"{data_block('question', task_text(st.input))}\n\n{data_block('queries_tried', chr(10).join(queries))}",
            ),
        ]
        return ask(e, self.name, tag, msgs, model=self.model, schema=_QUERY_SCHEMA)

    @staticmethod
    def _new_query(text: str, tried: list[str]) -> str | None:
        q = str((parse_json_obj(text) or {}).get("query") or "").strip()[:300]
        return q if q and q not in tried else None


class CorrectiveRAG(_Rag):
    """Corrective RAG: retrieve, **grade every passage**, and act on the grades.

    Enough relevant passages (`min_relevant`) and the question is answered from them. Otherwise the query is rewritten and retrieval repeated
    (up to `max_rewrites` times); if that still fails and a `web_tool` was given, it is called with the last query and its results are graded too.
    With nothing relevant anywhere the run ends `NO_SOLUTION` instead of guessing.

    `web_tool` is the operator's fallback (a web or enterprise search tool taking a `query`); it is added to `rag.tools`, so attach
    `ToolHandler(rag.tools)`. Result: `{"text", "sources", "queries", "route", "graded"}` with `route` one of `index`, `rewritten`, `web`."""

    name = "corrective_rag"

    def __init__(
        self,
        index: Index,
        *,
        web_tool: Tool | None = None,
        min_relevant: int = 1,
        max_rewrites: int = 1,
        **kw: Any,
    ) -> None:
        if web_tool is not None and "query" not in (web_tool.parameters.get("properties") or {}):
            raise ValueError("web_tool must take a 'query' argument")
        super().__init__(index, extra_tools=[web_tool] if web_tool else None, **kw)
        if not 1 <= min_relevant <= self.k:
            raise ValueError("min_relevant must be between 1 and k")
        if not 0 <= max_rewrites <= MAX_ROUNDS:
            raise ValueError(f"max_rewrites must be between 0 and {MAX_ROUNDS}")
        self.web_tool, self.min_relevant, self.max_rewrites = web_tool, min_relevant, max_rewrites

    @property
    def max_calls(self) -> int:
        rounds = self.max_rewrites + 1
        return rounds * self.k + self.max_rewrites + (self.k if self.web_tool else 0) + 1

    def _end(
        self,
        e: Event,
        st: Calls,
        passages: list[Passage],
        queries: list[str],
        route: str,
        graded: int,
    ) -> list[Event]:
        if not st.done("answer"):
            return [self._answer(e, st, "answer", passages)]
        if "answer" in st.failed:
            return [llm_failed(e)] if e.type == "aaa.llm.failed" else []
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "text": st.text("answer"),
                    "sources": [pid for pid, _ in passages],
                    "queries": queries,
                    "route": route,
                    "graded": graded,
                },
            )
        ]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        if e.type == "aaa.llm.failed" and e.data.get("tag") == "answer":
            return [llm_failed(e)]
        queries = [task_text(st.input)]
        graded = 0
        r = 0
        while True:
            rid = f"r{r}"
            if not st.done(rid):
                return [self._search(e, rid, self.search_name, queries[-1])]
            passages = self._passages(st.results.get(rid, []))
            grades = [f"g{r}.{i}" for i in range(len(passages))]
            if not st.all_done(grades):
                return [
                    self._grade(e, st, g, p)
                    for g, p in zip(grades, passages, strict=True)
                    if not st.done(g)
                ]
            good = self._relevant(st, f"g{r}", passages)
            graded += len(passages)
            if len(good) >= self.min_relevant:
                return self._end(e, st, good, queries, "index" if r == 0 else "rewritten", graded)
            if r < self.max_rewrites:
                tag = f"rw{r}"
                if not st.done(tag):
                    return [self._rewrite(e, st, tag, queries)]
                new = self._new_query(st.text(tag), queries)
                if new is not None:
                    queries.append(new)
                    r += 1
                    continue
            break
        return self._web(e, st, queries, good, graded)

    def _web(
        self, e: Event, st: Calls, queries: list[str], good: list[Passage], graded: int
    ) -> list[Event]:
        if self.web_tool is None:
            return [
                finish(
                    e,
                    OutcomeKind.NO_SOLUTION,
                    {"sources": [], "queries": queries, "graded": graded},
                    reason="no relevant evidence was found",
                )
            ]
        if not st.done("w"):
            return [self._search(e, "w", self.web_tool.name, queries[-1])]
        web = self._passages(st.results.get("w", []), "web:")
        grades = [f"gw.{i}" for i in range(len(web))]
        if not st.all_done(grades):
            return [
                self._grade(e, st, g, p) for g, p in zip(grades, web, strict=True) if not st.done(g)
            ]
        keep = [*good, *self._relevant(st, "gw", web)]
        graded += len(web)
        if not keep:
            return [
                finish(
                    e,
                    OutcomeKind.NO_SOLUTION,
                    {"sources": [], "queries": queries, "graded": graded},
                    reason="no relevant evidence was found",
                )
            ]
        return self._end(e, st, keep, queries, "web", graded)


_RETRIEVE_SCHEMA = {
    "type": "object",
    "properties": {"retrieve": {"type": "boolean"}},
    "required": ["retrieve"],
    "additionalProperties": False,
}
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "supported": {"type": "string", "enum": ["full", "partial", "none"]},
        "useful": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["relevant", "supported", "useful"],
    "additionalProperties": False,
}
_SUPPORT = {"full": 2, "partial": 1}


class SelfRAG(_Rag):
    """Self-RAG as an algorithm: the model decides whether the question needs retrieval at all, then writes **one candidate answer per retrieved
    passage** (so no answer mixes sources) and judges each candidate with reflection fields: is the passage `relevant`, is the answer
    `supported` by it (`full`, `partial`, `none`) and how `useful` is it (1 to 5). The best supported candidate wins. If none is supported the query
    is rewritten and the round repeated (`max_rewrites`), after which the run ends `NO_SOLUTION` carrying the best candidate rather than presenting
    an unsupported answer as grounded.

    Result: `{"text", "source", "supported", "useful", "queries", "rounds", "route"}` (`route` is `retrieved` or `direct`)."""

    name = "self_rag"

    def __init__(
        self,
        index: Index,
        *,
        adaptive: bool = True,
        max_candidates: int = 3,
        max_rewrites: int = 1,
        **kw: Any,
    ) -> None:
        super().__init__(index, **kw)
        if not 1 <= max_candidates <= self.k:
            raise ValueError("max_candidates must be between 1 and k")
        if not 0 <= max_rewrites <= MAX_ROUNDS:
            raise ValueError(f"max_rewrites must be between 0 and {MAX_ROUNDS}")
        self.adaptive, self.max_candidates, self.max_rewrites = (
            adaptive,
            max_candidates,
            max_rewrites,
        )

    @property
    def max_calls(self) -> int:
        rounds = self.max_rewrites + 1
        return (1 if self.adaptive else 0) + rounds * 2 * self.max_candidates + self.max_rewrites

    def _candidate(self, e: Event, st: Calls, tag: str, passage: Passage) -> Event:
        return self._answer(e, st, tag, [passage])

    def _judge(self, e: Event, st: Calls, tag: str, passage: Passage, answer: str) -> Event:
        msgs = [
            Message(
                role="system",
                content=(
                    "Judge the candidate answer. relevant: does the passage bear on the question? supported: is every claim in the answer "
                    'backed by the passage (full, partial, none)? useful: 1 (useless) to 5 (fully answers). Reply with JSON {"relevant", "supported", "useful"}.'
                ),
            ),
            Message(
                role="user",
                content="\n\n".join(
                    [
                        data_block("question", task_text(st.input)),
                        data_block("passage", f"[{passage[0]}] {passage[1]}"),
                        data_block("answer", answer),
                    ]
                ),
            ),
        ]
        return ask(
            e, self.name, tag, msgs, model=self.grader_model or self.model, schema=_JUDGE_SCHEMA
        )

    @staticmethod
    def _score(judgment: dict[str, Any]) -> tuple[float, bool]:
        sup = _SUPPORT.get(str(judgment.get("supported")), 0)
        try:
            useful = max(1, min(5, int(judgment.get("useful", 1))))
        except (TypeError, ValueError):
            useful = 1
        relevant = judgment.get("relevant") is True
        return (float(relevant) + 2.0 * sup + useful / 5.0, relevant and sup > 0)

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        question = task_text(st.input)
        if self.adaptive:
            if not st.done("ret"):
                msgs = [
                    Message(
                        role="system",
                        content='Does answering this question need passages from the knowledge base? Reply with JSON {"retrieve": true|false}.',
                    ),
                    Message(role="user", content=data_block("question", question)),
                ]
                return [ask(e, self.name, "ret", msgs, model=self.model, schema=_RETRIEVE_SCHEMA)]
            if (parse_json_obj(st.text("ret")) or {}).get("retrieve") is False:
                if not st.done("direct"):
                    msgs = [
                        Message(
                            role="system",
                            content="Answer the question briefly. If you are not sure, say so.",
                        ),
                        Message(role="user", content=data_block("question", question)),
                    ]
                    return [ask(e, self.name, "direct", msgs, model=self.model)]
                if "direct" in st.failed:
                    return [llm_failed(e)] if e.type == "aaa.llm.failed" else []
                return [
                    finish(
                        e,
                        OutcomeKind.SUCCESS,
                        {
                            "text": st.text("direct"),
                            "source": None,
                            "supported": None,
                            "useful": None,
                            "queries": [],
                            "rounds": 0,
                            "route": "direct",
                        },
                    )
                ]
        queries = [question]
        r = 0
        while True:
            rid = f"r{r}"
            if not st.done(rid):
                return [self._search(e, rid, self.search_name, queries[-1])]
            passages = self._passages(st.results.get(rid, []))[: self.max_candidates]
            cands = [f"c{r}.{i}" for i in range(len(passages))]
            if not st.all_done(cands):
                return [
                    self._candidate(e, st, c, p)
                    for c, p in zip(cands, passages, strict=True)
                    if not st.done(c)
                ]
            judged = [f"j{r}.{i}" for i in range(len(passages))]
            ready = [(i, p) for i, p in enumerate(passages) if cands[i] in st.texts]
            todo = [
                self._judge(e, st, judged[i], p, st.text(cands[i]))
                for i, p in ready
                if not st.done(judged[i])
            ]
            if todo:
                return todo
            options = []
            for i, p in ready:
                score, ok = self._score(parse_json_obj(st.text(judged[i])) or {})
                options.append((score, -i, ok, i, p))
            best = max(options) if options else None
            if best is not None and best[2]:
                _, _, _, i, p = best
                j = parse_json_obj(st.text(judged[i])) or {}
                return [
                    finish(
                        e,
                        OutcomeKind.SUCCESS,
                        {
                            "text": st.text(cands[i]),
                            "source": p[0],
                            "supported": j.get("supported"),
                            "useful": j.get("useful"),
                            "queries": queries,
                            "rounds": r + 1,
                            "route": "retrieved",
                        },
                    )
                ]
            if r < self.max_rewrites:
                tag = f"rw{r}"
                if not st.done(tag):
                    return [self._rewrite(e, st, tag, queries)]
                new = self._new_query(st.text(tag), queries)
                if new is not None:
                    queries.append(new)
                    r += 1
                    continue
            value: dict[str, Any] = {
                "text": None,
                "source": None,
                "queries": queries,
                "rounds": r + 1,
                "route": "retrieved",
            }
            if best is not None:
                _, _, _, i, p = best
                value |= {"text": st.text(cands[i]), "source": p[0]}
            return [
                finish(
                    e,
                    OutcomeKind.NO_SOLUTION,
                    value,
                    reason="no answer was supported by the evidence",
                )
            ]


class AdaptiveRAG(_Rag):
    """Adaptive RAG: match the retrieval to the question. A first call routes it: `direct` (answer from the model's own knowledge, no retrieval),
    `single` (one retrieval, then answer) or `iterative` (up to `max_hops` rounds of "search again or answer", for questions that need
    several facts). An unusable routing answer falls back to `single`.

    Result: `{"text", "route", "hops", "sources", "queries"}`."""

    name = "adaptive_rag"

    def __init__(self, index: Index, *, max_hops: int = 3, **kw: Any) -> None:
        super().__init__(index, **kw)
        if not 1 <= max_hops <= MAX_HOPS:
            raise ValueError(f"max_hops must be between 1 and {MAX_HOPS}")
        self.max_hops = max_hops

    @property
    def max_calls(self) -> int:
        return 1 + self.max_hops + 1

    def _evidence(self, st: Calls, upto: int) -> list[Passage]:
        seen: set[str] = set()
        out: list[Passage] = []
        for h in range(upto):
            for pid, body in self._passages(st.results.get(f"r{h}", [])):
                if pid not in seen:
                    seen.add(pid)
                    out.append((pid, body))
        return out

    def _end(
        self,
        e: Event,
        st: Calls,
        route: str,
        hops: int,
        passages: list[Passage],
        queries: list[str],
    ) -> list[Event]:
        if not st.done("answer"):
            return [self._answer(e, st, "answer", passages)]
        if "answer" in st.failed:
            return [llm_failed(e)] if e.type == "aaa.llm.failed" else []
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "text": st.text("answer"),
                    "route": route,
                    "hops": hops,
                    "sources": [p for p, _ in passages],
                    "queries": queries,
                },
            )
        ]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        question = task_text(st.input)
        if not st.done("route"):
            msgs = [
                Message(
                    role="system",
                    content=(
                        "Choose how to answer. direct: common knowledge, no lookup. single: one lookup in the knowledge base will do. "
                        'iterative: several facts must be found and combined. Reply with JSON {"route": "direct"|"single"|"iterative"}.'
                    ),
                ),
                Message(role="user", content=data_block("question", question)),
            ]
            return [
                ask(
                    e,
                    self.name,
                    "route",
                    msgs,
                    model=self.model,
                    schema={
                        "type": "object",
                        "properties": {
                            "route": {"type": "string", "enum": ["direct", "single", "iterative"]}
                        },
                        "required": ["route"],
                        "additionalProperties": False,
                    },
                )
            ]
        route = str((parse_json_obj(st.text("route")) or {}).get("route"))
        if route not in ("direct", "iterative"):
            route = "single"
        if route == "direct":
            if not st.done("direct"):
                msgs = [
                    Message(
                        role="system",
                        content="Answer the question briefly from your own knowledge. If you are not sure, say so.",
                    ),
                    Message(role="user", content=data_block("question", question)),
                ]
                return [ask(e, self.name, "direct", msgs, model=self.model)]
            if "direct" in st.failed:
                return [llm_failed(e)] if e.type == "aaa.llm.failed" else []
            return [
                finish(
                    e,
                    OutcomeKind.SUCCESS,
                    {
                        "text": st.text("direct"),
                        "route": "direct",
                        "hops": 0,
                        "sources": [],
                        "queries": [],
                    },
                )
            ]
        if route == "single":
            if not st.done("r0"):
                return [self._search(e, "r0", self.search_name, question)]
            return self._end(e, st, "single", 1, self._evidence(st, 1), [question])
        queries: list[str] = []
        hops = 0
        for h in range(self.max_hops):
            tag = f"h{h}"
            if not st.done(tag):
                evidence = (
                    "\n".join(f"[{pid}] {body}" for pid, body in self._evidence(st, hops))
                    or "(nothing yet)"
                )
                msgs = [
                    Message(
                        role="system",
                        content='You may search the knowledge base. Reply with JSON {"action": "search", "query": "..."} to look something up, or {"action": "answer"} when the passages are enough.',
                    ),
                    Message(
                        role="user",
                        content=f"{data_block('question', question)}\n\n{data_block('passages', evidence)}",
                    ),
                ]
                return [
                    ask(
                        e,
                        self.name,
                        tag,
                        msgs,
                        model=self.model,
                        schema={
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["search", "answer"]},
                                "query": {"type": "string"},
                            },
                            "required": ["action"],
                        },
                    )
                ]
            d = parse_json_obj(st.text(tag)) or {}
            q = str(d.get("query") or "").strip()[:300]
            if d.get("action") != "search" or not q or q in queries:
                break
            rid = f"r{h}"
            if not st.done(rid):
                return [self._search(e, rid, self.search_name, q)]
            queries.append(q)
            hops += 1
        return self._end(e, st, "iterative", hops, self._evidence(st, hops), queries)
