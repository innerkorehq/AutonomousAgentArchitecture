"""Graph-of-thoughts: a small, validated graph of operations over a pool of thoughts, so thoughts from different branches can be
scored, kept, merged and refined (a tree of thoughts can only branch).

You describe the graph as a list of `Op`s; the algorithm runs them in the listed order, all calls inside one operation in parallel,
and moves on when the operation is complete. The number of model calls is known before anything runs (`max_calls`), because every
operation's output size follows from its inputs and its `n`. Calls are tagged `<op id>:<index>`, so the run is a pure function of the log."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from aaa.algos.base import (
    CALLS,
    SCORE_SCHEMA,
    Calls,
    ask,
    clamp_score,
    data_block,
    finish,
    task_text,
)
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import Message
from aaa.runtime import Ctx

MAX_OPS, MAX_THOUGHTS, MAX_CALLS = 16, 30, 200
Kind = Literal["generate", "aggregate", "refine", "score", "keep_best"]
_ID = re.compile(r"^[a-z][a-z0-9_]{0,19}$")
Thought = tuple[str, float | None]


@dataclass(frozen=True)
class Op:
    """One operation. `generate` writes `n` new thoughts (from the task, or building on its `inputs`); `aggregate` merges all input thoughts
    into one; `refine` improves each input thought; `score` rates each input thought 0 to 10; `keep_best` keeps the `n` highest scored."""

    id: str
    kind: Kind
    inputs: tuple[str, ...] = ()
    n: int = 1
    instruction: str = ""


def default_graph(width: int = 4, keep: int = 2) -> tuple[Op, ...]:
    """generate -> score -> keep the best -> merge them -> refine -> score -> keep the best."""
    return (
        Op("gen", "generate", n=width),
        Op("s1", "score", ("gen",)),
        Op("top", "keep_best", ("s1",), n=keep),
        Op("agg", "aggregate", ("top",)),
        Op("ref", "refine", ("agg",)),
        Op("s2", "score", ("ref",)),
        Op("best", "keep_best", ("s2",), n=1),
    )


def _sizes(ops: tuple[Op, ...]) -> tuple[dict[str, int], int]:
    """(output size per op, total model calls), derived from the structure alone."""
    size: dict[str, int] = {}
    calls = 0
    for op in ops:
        got = sum(size[i] for i in op.inputs)
        if op.kind == "generate":
            size[op.id], c = op.n, op.n
        elif op.kind == "aggregate":
            size[op.id], c = (1 if got else 0), (1 if got else 0)
        elif op.kind == "refine" or op.kind == "score":
            size[op.id], c = got, got
        else:  # keep_best
            size[op.id], c = min(op.n, got), 0
        if size[op.id] > MAX_THOUGHTS:
            raise ValueError(
                f"operation {op.id!r} would hold {size[op.id]} thoughts (limit {MAX_THOUGHTS})"
            )
        calls += c
    return size, calls


class GraphOfThoughts:
    """Run a graph of operations over thoughts and return the best final thought.

    Result: `{"answer", "score", "thoughts", "operations"}` (`thoughts` is how many thoughts the operations hold in total)."""

    name = "graph_of_thoughts"

    def __init__(
        self,
        ops: tuple[Op, ...] | list[Op] | None = None,
        *,
        model: str | None = None,
        scorer_model: str | None = None,
        system: str = "You solve problems by producing and improving thoughts.",
        temperature: float = 0.8,
    ) -> None:
        self.ops = tuple(ops) if ops is not None else default_graph()
        if not 1 <= len(self.ops) <= MAX_OPS:
            raise ValueError(f"a graph needs between 1 and {MAX_OPS} operations")
        seen: set[str] = set()
        for op in self.ops:
            if not _ID.match(op.id):
                raise ValueError(
                    f"operation id {op.id!r} must be lowercase letters, digits and '_'"
                )
            if op.id in seen:
                raise ValueError(f"two operations are called {op.id!r}")
            if op.kind not in ("generate", "aggregate", "refine", "score", "keep_best"):
                raise ValueError(f"operation {op.id!r}: unknown kind {op.kind!r}")
            if any(i not in seen for i in op.inputs):
                raise ValueError(
                    f"operation {op.id!r} reads an operation that is not defined before it"
                )
            if op.kind != "generate" and not op.inputs:
                raise ValueError(f"operation {op.id!r} ({op.kind}) needs inputs")
            if not 1 <= op.n <= MAX_THOUGHTS:
                raise ValueError(f"operation {op.id!r}: n must be between 1 and {MAX_THOUGHTS}")
            seen.add(op.id)
        _, self._calls = _sizes(self.ops)
        if self._calls > MAX_CALLS:
            raise ValueError(f"this graph can make {self._calls} model calls (limit {MAX_CALLS})")
        self.model, self.scorer_model, self.system = model, scorer_model, system
        self.temperature = temperature

    @property
    def max_calls(self) -> int:
        return self._calls

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]

    # -- prompts -----------------------------------------------------------------------------
    def _prompt(self, op: Op, st: Calls, ins: list[Thought], j: int) -> list[Message]:
        task = data_block("task", task_text(st.input))
        extra = f" {op.instruction}" if op.instruction else ""
        if op.kind == "generate":
            seeds = (
                "\n\n" + data_block("thoughts", "\n---\n".join(t for t, _ in ins)) if ins else ""
            )
            return [
                Message(
                    role="system",
                    content=f"{self.system} Write one new thought (a candidate answer or a partial solution). Reply with the thought only.{extra}",
                ),
                Message(role="user", content=f"{task}{seeds}\n\nThis is variant {j + 1}."),
            ]
        if op.kind == "aggregate":
            return [
                Message(
                    role="system",
                    content=f"{self.system} Combine the thoughts into one better thought that keeps what is right in each. Reply with the thought only.{extra}",
                ),
                Message(
                    role="user",
                    content=f"{task}\n\n{data_block('thoughts', chr(10).join(f'[{i + 1}] {t}' for i, (t, _) in enumerate(ins)))}",
                ),
            ]
        if op.kind == "refine":
            return [
                Message(
                    role="system",
                    content=f"{self.system} Improve the thought: fix errors and gaps. Reply with the improved thought only.{extra}",
                ),
                Message(role="user", content=f"{task}\n\n{data_block('thought', ins[j][0])}"),
            ]
        return [  # score
            Message(
                role="system",
                content='Rate how well the thought solves the task, from 0 (wrong) to 10 (solved). Reply with JSON {"score": number}.',
            ),
            Message(role="user", content=f"{task}\n\n{data_block('thought', ins[j][0])}"),
        ]

    # -- evaluation: a pure function of the answers so far -----------------------------------
    def _walk(self, e: Event, st: Calls) -> tuple[list[Event], dict[str, list[Thought]] | None]:
        out: dict[str, list[Thought]] = {}
        for op in self.ops:
            ins = [t for i in op.inputs for t in out[i]]
            if op.kind == "keep_best":
                ranked = sorted(
                    enumerate(ins),
                    key=lambda p: (-(p[1][1] if p[1][1] is not None else -1.0), p[0]),
                )
                out[op.id] = [t for _, t in ranked[: op.n]]
                continue
            count = op.n if op.kind == "generate" else (1 if op.kind == "aggregate" else len(ins))
            if op.kind == "aggregate" and not ins:
                count = 0
            tags = [f"{op.id}:{j}" for j in range(count)]
            if not st.all_done(tags):
                model = (self.scorer_model or self.model) if op.kind == "score" else self.model
                return (
                    [
                        ask(
                            e,
                            self.name,
                            t,
                            self._prompt(op, st, ins, j),
                            model=model,
                            params=None
                            if op.kind == "score"
                            else {"temperature": self.temperature},
                            schema=SCORE_SCHEMA if op.kind == "score" else None,
                        )
                        for j, t in enumerate(tags)
                        if not st.done(t)
                    ],
                    None,
                )
            if op.kind == "score":
                out[op.id] = [
                    (ins[j][0], clamp_score(st.text(t)) or 0.0) for j, t in enumerate(tags)
                ]
            elif op.kind == "refine":  # a failed refinement keeps the thought it was refining
                out[op.id] = [
                    (st.text(t).strip(), None) if st.text(t).strip() else ins[j]
                    for j, t in enumerate(tags)
                ]
            else:
                out[op.id] = [(st.text(t).strip(), None) for t in tags if st.text(t).strip()]
        return [], out

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        requests, out = self._walk(e, st)
        if out is None:
            return requests
        final = out[self.ops[-1].id]
        if not final:
            return [finish(e, OutcomeKind.FAILED, reason="no thought survived the graph")]
        best = max(
            enumerate(final), key=lambda p: (p[1][1] if p[1][1] is not None else -1.0, -p[0])
        )[1]
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "answer": best[0],
                    "score": best[1],
                    "thoughts": sum(len(v) for v in out.values()),
                    "operations": [op.id for op in self.ops],
                },
            )
        ]
