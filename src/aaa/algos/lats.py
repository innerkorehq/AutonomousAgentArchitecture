"""LATS: language-agent tree search. Monte Carlo tree search over reasoning steps, with model-scored values and written reflections
on attempts that finished but scored poorly.

Each iteration selects the most promising node that can still be expanded (UCT), asks for `branching` next steps from it in
parallel, scores each, and backs the scores up the tree. A step that begins with `FINAL:` ends a path. Iterations run one after the
other and every choice (selection, ties, the result) is a pure function of the answers so far, so the run is the same whatever order parallel
calls finish in. Calls are tagged `x<it>.<child>` (expand), `v<it>.<child>` (value) and `r<it>.<child>` (reflection)."""

from __future__ import annotations

import math
from dataclasses import dataclass

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

MAX_ITERATIONS, MAX_BRANCHING, MAX_DEPTH = 20, 6, 8
FINAL = "FINAL:"
_ROOT = "root"


@dataclass
class _Node:
    parent: str | None
    text: str
    depth: int
    terminal: bool = False
    value: float = 0.0
    visits: int = 0
    total: float = 0.0


class _Tree:
    def __init__(self, depth_limit: int, exploration: float) -> None:
        self.nodes = {_ROOT: _Node(None, "", 0)}
        self.kids: dict[str, list[str]] = {_ROOT: []}
        self.dead: set[str] = set()  # nodes whose expansion produced nothing usable
        self.order: list[str] = []
        self.limit, self.c = depth_limit, exploration

    def alive(self, nid: str) -> bool:
        n = self.nodes[nid]
        if n.terminal or n.depth >= self.limit or nid in self.dead:
            return False
        return not self.kids[nid] or any(self.alive(k) for k in self.kids[nid])

    def select(self) -> str | None:
        nid = _ROOT
        while True:
            if not self.kids[nid]:
                return nid if self.alive(nid) else None
            alive = [k for k in self.kids[nid] if self.alive(k)]
            if not alive:
                return None
            parent = self.nodes[nid]

            def uct(k: str, parent: _Node = parent) -> float:
                n = self.nodes[k]
                return n.total / n.visits + self.c * math.sqrt(
                    math.log(max(parent.visits, 1)) / n.visits
                )

            nid = max(alive, key=uct)  # first of equals wins: creation order

    def add(self, parent: str, nid: str, text: str, terminal: bool, value: float) -> None:
        self.nodes[nid] = _Node(parent, text, self.nodes[parent].depth + 1, terminal, value)
        self.kids[nid] = []
        self.kids[parent].append(nid)
        self.order.append(nid)
        cur: str | None = nid
        while cur is not None:  # back the value up to the root
            n = self.nodes[cur]
            n.visits, n.total = n.visits + 1, n.total + value
            cur = n.parent

    def path(self, nid: str) -> list[str]:
        out: list[str] = []
        cur: str | None = nid
        while cur is not None and cur != _ROOT:
            out.append(self.nodes[cur].text)
            cur = self.nodes[cur].parent
        return out[::-1]


class LATS:
    """Tree search over reasoning steps with reflections.

    Stops when a finished path scores at least `solved_score`, when `iterations` are used, or when nothing can be expanded.
    A finished path scoring below `reflect_below` gets a short written lesson that later expansions see. Costs at most
    `iterations * branching * 3` model calls.

    Result: `{"answer", "score", "path", "solved", "iterations", "explored", "reflections"}`. Without any finished path the
    outcome is `NO_SOLUTION` carrying the best partial path."""

    name = "lats"

    def __init__(
        self,
        *,
        iterations: int = 6,
        branching: int = 3,
        depth_limit: int = 5,
        exploration: float = 1.4,
        solved_score: float = 9.0,
        reflect_below: float = 4.0,
        model: str | None = None,
        scorer_model: str | None = None,
        system: str = "You solve problems one reasoning step at a time.",
        temperature: float = 0.8,
    ) -> None:
        if not 1 <= iterations <= MAX_ITERATIONS:
            raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")
        if not 1 <= branching <= MAX_BRANCHING:
            raise ValueError(f"branching must be between 1 and {MAX_BRANCHING}")
        if not 1 <= depth_limit <= MAX_DEPTH:
            raise ValueError(f"depth_limit must be between 1 and {MAX_DEPTH}")
        if exploration < 0:
            raise ValueError("exploration must not be negative")
        self.iterations, self.branching, self.depth_limit = iterations, branching, depth_limit
        self.exploration, self.solved_score, self.reflect_below = (
            exploration,
            solved_score,
            reflect_below,
        )
        self.model, self.scorer_model, self.system = model, scorer_model, system
        self.temperature = temperature

    @property
    def max_calls(self) -> int:
        return self.iterations * self.branching * 3

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]

    # -- prompts -----------------------------------------------------------------------------
    def _expand(self, e: Event, st: Calls, tag: str, path: list[str], lessons: list[str]) -> Event:
        steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(path)) or "(none yet)"
        parts = [data_block("task", task_text(st.input)), data_block("reasoning_so_far", steps)]
        if lessons:
            parts.append(data_block("lessons", "\n".join(f"- {x}" for x in lessons[-3:])))
        parts.append(
            f"Write the next step only. If it completes the solution, start it with '{FINAL}' followed by the final answer."
        )
        msgs = [
            Message(role="system", content=self.system),
            Message(role="user", content="\n\n".join(parts)),
        ]
        return ask(
            e, self.name, tag, msgs, model=self.model, params={"temperature": self.temperature}
        )

    def _value(self, e: Event, st: Calls, tag: str, path: list[str]) -> Event:
        msgs = [
            Message(
                role="system",
                content='Rate how likely this line of reasoning is to be correct and complete, from 0 (wrong or stuck) to 10 (solved). Reply with JSON {"score": number}.',
            ),
            Message(
                role="user",
                content=f"{data_block('task', task_text(st.input))}\n\n{data_block('reasoning', chr(10).join(path))}",
            ),
        ]
        return ask(
            e, self.name, tag, msgs, model=self.scorer_model or self.model, schema=SCORE_SCHEMA
        )

    def _reflect(self, e: Event, st: Calls, tag: str, path: list[str]) -> Event:
        msgs = [
            Message(
                role="system",
                content="This attempt finished but is probably wrong. In one or two sentences say what went wrong and what to try differently.",
            ),
            Message(
                role="user",
                content=f"{data_block('task', task_text(st.input))}\n\n{data_block('attempt', chr(10).join(path))}",
            ),
        ]
        return ask(e, self.name, tag, msgs, model=self.model)

    # -- the search, replayed from the answers so far ------------------------------------------
    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        tree = _Tree(self.depth_limit, self.exploration)
        lessons: list[str] = []
        used = 0
        for it in range(self.iterations):
            leaf = tree.select()
            if leaf is None:
                break
            used = it + 1
            path = tree.path(leaf)
            xs = [f"x{it}.{c}" for c in range(self.branching)]
            if not st.all_done(xs):
                return [self._expand(e, st, t, path, lessons) for t in xs if not st.done(t)]
            kids = [(c, st.text(f"x{it}.{c}").strip()) for c in range(self.branching)]
            seen: set[str] = (
                set()
            )  # identical sibling steps are one node: no point scoring the same text twice
            kids = [(c, t) for c, t in kids if t and not (t in seen or seen.add(t))]
            if not kids:
                tree.dead.add(leaf)
                continue
            vs = [f"v{it}.{c}" for c, _ in kids]
            if not st.all_done(vs):
                return [
                    self._value(e, st, f"v{it}.{c}", [*path, t])
                    for c, t in kids
                    if not st.done(f"v{it}.{c}")
                ]
            made: list[str] = []
            for c, t in kids:
                value = clamp_score(st.text(f"v{it}.{c}")) or 0.0
                nid = f"{it}.{c}"
                tree.add(leaf, nid, t, t.startswith(FINAL), value)
                made.append(nid)
            solved = [
                n
                for n in made
                if tree.nodes[n].terminal and tree.nodes[n].value >= self.solved_score
            ]
            if solved:
                return [self._end(e, tree, lessons, used, solved[0])]
            low = [
                n
                for n in made
                if tree.nodes[n].terminal and tree.nodes[n].value < self.reflect_below
            ]
            if not st.all_done([f"r{n}" for n in low]):
                return [
                    self._reflect(e, st, f"r{n}", tree.path(n)) for n in low if not st.done(f"r{n}")
                ]
            lessons += [st.text(f"r{n}").strip() for n in low if st.text(f"r{n}").strip()]
        return [self._end(e, tree, lessons, used, None)]

    def _end(self, e: Event, tree: _Tree, lessons: list[str], used: int, pick: str | None) -> Event:
        if pick is None:
            done = [n for n in tree.order if tree.nodes[n].terminal]
            pool = done or tree.order
            if not pool:
                return finish(e, OutcomeKind.FAILED, reason="the search produced no usable step")
            pick = max(pool, key=lambda n: (tree.nodes[n].value, -tree.order.index(n)))
        node = tree.nodes[pick]
        value = {
            "answer": node.text.removeprefix(FINAL).strip(),
            "score": node.value,
            "path": tree.path(pick),
            "solved": node.terminal and node.value >= self.solved_score,
            "iterations": used,
            "explored": len(tree.order),
            "reflections": lessons,
        }
        if not node.terminal:
            return finish(
                e, OutcomeKind.NO_SOLUTION, value, reason="no path reached a final answer"
            )
        return finish(e, OutcomeKind.SUCCESS, value)
