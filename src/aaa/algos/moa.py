"""Mixture-of-agents and a blackboard: two ways for several model calls to build on each other's work.

`MixtureOfAgents` is layered: every proposer answers, the next layer's proposers each read all the previous layer's answers and write a better one, and an
aggregator writes the final answer. `Blackboard` is shared state: a controller picks which expert contributes next to a common board, until it says
it is done. Both are bounded at construction (`max_calls`), run as a fold of the log, and show earlier work to the model only as quoted data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aaa.algos.base import (
    CALLS,
    ask,
    data_block,
    finish,
    llm_failed,
    parse_json_obj,
    task_messages,
    task_text,
)
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import Message
from aaa.runtime import Ctx

MAX_LAYERS, MAX_PROPOSERS, MAX_ROUNDS = 4, 6, 10
_SUBS = ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]


class MixtureOfAgents:
    """Layered proposers with an aggregator.

    Layer 0: `proposers` independent answers to the task (each may have its own `models[i]` and `personas[i]`). Each later layer: every proposer
    sees all the previous layer's answers and writes an improved one. Finally the aggregator combines the last layer. A proposer whose call fails
    drops out of that layer; a layer with no answers fails the run. At most `proposers * layers + 1` calls.

    Result: `{"answer", "layers": [[answers...]]}`."""

    name = "mixture_of_agents"

    def __init__(
        self,
        *,
        proposers: int = 3,
        layers: int = 2,
        models: Sequence[str | None] | None = None,
        personas: Sequence[str] | None = None,
        aggregator_model: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        system: str = "",
    ) -> None:
        if not 2 <= proposers <= MAX_PROPOSERS:
            raise ValueError(f"proposers must be between 2 and {MAX_PROPOSERS}")
        if not 1 <= layers <= MAX_LAYERS:
            raise ValueError(f"layers must be between 1 and {MAX_LAYERS}")
        if models is not None and len(models) != proposers:
            raise ValueError("models needs one entry per proposer")
        if personas is not None and len(personas) != proposers:
            raise ValueError("personas needs one entry per proposer")
        self.proposers, self.layers, self.model, self.system = proposers, layers, model, system
        self.models, self.personas = list(models or []), list(personas or [])
        self.aggregator_model, self.temperature = aggregator_model, temperature

    @property
    def max_calls(self) -> int:
        return self.proposers * self.layers + 1

    def subscriptions(self) -> list[str]:
        return list(_SUBS)

    def _propose(self, e: Event, st: Any, layer: int, i: int, previous: list[str]) -> Event:
        persona = f" {self.personas[i]}" if self.personas else ""
        model = (self.models[i] if self.models else None) or self.model
        if layer == 0:
            msgs = task_messages(st.input, f"{self.system}{persona}".strip())
        else:
            listing = "\n\n".join(f"[{n + 1}] {a}" for n, a in enumerate(previous))
            msgs = [
                Message(
                    role="system",
                    content=f"{self.system}{persona} Several assistants answered the task. Use the best of their ideas, fix their mistakes, and write one improved answer. Reply with the answer only.".strip(),
                ),
                Message(
                    role="user",
                    content=f"{data_block('task', task_text(st.input))}\n\n{data_block('answers', listing)}",
                ),
            ]
        return ask(
            e,
            self.name,
            f"l{layer}.{i}",
            msgs,
            model=model,
            params={"temperature": self.temperature},
        )

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        answers: list[list[str]] = []
        for layer in range(self.layers):
            tags = [f"l{layer}.{i}" for i in range(self.proposers)]
            if not st.all_done(tags):
                previous = answers[-1] if answers else []
                return [
                    self._propose(e, st, layer, i, previous)
                    for i, t in enumerate(tags)
                    if not st.done(t)
                ]
            got = [st.text(t).strip() for t in tags if st.text(t).strip()]
            if not got:
                return (
                    [llm_failed(e)]
                    if e.type == "aaa.llm.failed"
                    else [
                        finish(
                            e,
                            OutcomeKind.FAILED,
                            reason=f"no proposer answered in layer {layer + 1}",
                        )
                    ]
                )
            answers.append(got)
        if not st.done("agg"):
            listing = "\n\n".join(f"[{n + 1}] {a}" for n, a in enumerate(answers[-1]))
            msgs = [
                Message(
                    role="system",
                    content=f"{self.system} Combine the answers into one final answer that is accurate and complete. Reply with the answer only.".strip(),
                ),
                Message(
                    role="user",
                    content=f"{data_block('task', task_text(st.input))}\n\n{data_block('answers', listing)}",
                ),
            ]
            return [ask(e, self.name, "agg", msgs, model=self.aggregator_model or self.model)]
        if "agg" in st.failed:
            return (
                [llm_failed(e)]
                if e.type == "aaa.llm.failed"
                else [finish(e, OutcomeKind.FAILED, reason="the aggregator call failed")]
            )
        return [finish(e, OutcomeKind.SUCCESS, {"answer": st.text("agg"), "layers": answers})]


Board = list[tuple[str, str]]
Trigger = Callable[[Board], bool]


@dataclass(frozen=True)
class Expert:
    """One participant of a `Blackboard`. `when` is a pure function of the board (a list of `(author, text)`) saying when this expert has something to
    add; it must not do I/O. With `auto=True` the expert contributes **as soon as** `when` holds (a knowledge source reacting to a change on the board),
    without the chair being asked. Without `auto`, `when` only limits when the chair may pick it. A triggered expert contributes once per session (`repeat=True`: again after each change of the board, never twice on the same board)."""

    description: str
    when: Trigger | None = None
    auto: bool = False
    repeat: bool = False  # a triggered expert contributes once per session unless it may fire again each time the board changes


def after(author: str) -> Trigger:
    """True when the most recent contribution is by `author`."""
    return lambda board: bool(board) and board[-1][0] == author


def mentions(word: str, *, by: str | None = None) -> Trigger:
    """True when some contribution (by `by`, if given) contains `word`, ignoring case."""
    w = word.lower()
    return lambda board: any(w in t.lower() and (by is None or a == by) for a, t in board)


def empty_board() -> Trigger:
    """True while nobody has contributed yet (a way to start the session)."""
    return lambda board: not board


class Blackboard:
    """Experts build a shared answer on a common board, one contribution per round.

    Who contributes next is decided in one of two ways. An expert with `Expert(..., when=..., auto=True)` is a **knowledge source**: as soon as its trigger
    holds on the current board it contributes, without the chair being asked (the first such expert, in the order given, if several are ready). When no
    automatic expert is ready, the chair reads the board and names who goes next, or `done`; it may only pick experts that have no trigger or whose trigger
    holds now. A named expert writes one contribution (capped at `max_entry_chars`) appended under their name; at the end a solver writes the answer
    from the board.

    Everything that matters is bounded and checked in code: triggers are plain functions of the board (an exception counts as "not triggered"), an expert
    a triggered expert contributes once unless it may repeat, an unknown or ineligible name ends the session, a failed expert call is skipped, and rounds are capped. At most
    `2 * max_rounds + 1` model calls (automatic rounds need no chair call).

    Result: `{"answer", "board": [{"author", "text"}], "rounds"}`."""

    name = "blackboard"

    def __init__(
        self,
        experts: Mapping[str, str | Expert],
        *,
        max_rounds: int = 6,
        max_entry_chars: int = 2000,
        model: str | None = None,
        controller_model: str | None = None,
        system: str = "",
    ) -> None:
        import re

        if not experts:
            raise ValueError("a blackboard needs at least one expert")
        if not all(re.match(r"^[a-z][a-z0-9_]{0,19}$", n) for n in experts):
            raise ValueError("expert names must be lowercase letters, digits and '_' (at most 20)")
        if not 1 <= max_rounds <= MAX_ROUNDS:
            raise ValueError(f"max_rounds must be between 1 and {MAX_ROUNDS}")
        self.specs = {n: (x if isinstance(x, Expert) else Expert(x)) for n, x in experts.items()}
        for n, x in self.specs.items():
            if x.auto and x.when is None:
                raise ValueError(f"expert {n!r} is automatic but has no trigger (when=)")
        self.experts = {n: x.description for n, x in self.specs.items()}
        self.max_rounds, self.max_entry_chars = max_rounds, max_entry_chars
        self.model, self.controller_model, self.system = model, controller_model, system

    @property
    def max_calls(self) -> int:
        return 2 * self.max_rounds + 1

    def subscriptions(self) -> list[str]:
        return list(_SUBS)

    def _board(self, board: Board) -> str:
        return "\n".join(f"[{a}] {t}" for a, t in board) or "(empty)"

    def _fresh(self, name: str, fired: dict[str, int], board: Board) -> bool:
        spec = self.specs[name]
        if spec.when is None:
            return True
        return fired.get(name, -1) < len(board) if spec.repeat else name not in fired

    @staticmethod
    def _holds(spec: Expert, board: Board) -> bool:
        if spec.when is None:
            return True
        try:
            return bool(spec.when(list(board)))
        except Exception:  # noqa: BLE001 - a trigger that raises is not triggered
            return False

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        task = task_text(st.input)
        board: Board = []
        fired: dict[str, int] = {}  # expert -> size of the board when it last contributed
        rounds = 0
        for r in range(self.max_rounds):
            fresh = [n for n in self.specs if self._fresh(n, fired, board)]
            auto = [n for n in fresh if self.specs[n].auto and self._holds(self.specs[n], board)]
            ctl, ex = f"c{r}", f"x{r}"
            if auto:
                who = auto[0]
            else:
                eligible = [n for n in fresh if self._holds(self.specs[n], board)]
                if not eligible:
                    break
                if not st.done(ctl):
                    roster = "\n".join(
                        f"- {n}: {self.specs[n].description}"
                        + (" (triggered by the board)" if self.specs[n].when else "")
                        for n in eligible
                    )
                    msgs = [
                        Message(
                            role="system",
                            content=(
                                f"{self.system} You chair a group of experts working on a shared board. Decide who should contribute next, or finish when the board answers the task. "
                                f'Reply with JSON {{"next": "<expert name>" | "done"}}.\n\nExperts you may pick now:\n{roster}'
                            ).strip(),
                        ),
                        Message(
                            role="user",
                            content=f"{data_block('task', task)}\n\n{data_block('board', self._board(board))}",
                        ),
                    ]
                    return [
                        ask(
                            e,
                            self.name,
                            ctl,
                            msgs,
                            model=self.controller_model or self.model,
                            schema={
                                "type": "object",
                                "properties": {"next": {"type": "string"}},
                                "required": ["next"],
                            },
                        )
                    ]
                who = str((parse_json_obj(st.text(ctl)) or {}).get("next", "done"))
                if who not in eligible:
                    break
            if not st.done(ex):
                msgs = [
                    Message(
                        role="system",
                        content=f"{self.system} You are the {who}: {self.specs[who].description} Add one contribution to the board that moves the task forward. Reply with the contribution only.".strip(),
                    ),
                    Message(
                        role="user",
                        content=f"{data_block('task', task)}\n\n{data_block('board', self._board(board))}",
                    ),
                ]
                return [ask(e, self.name, ex, msgs, model=self.model)]
            rounds = r + 1
            body = st.text(ex).strip()[: self.max_entry_chars]
            if body:
                board.append((who, body))
            fired[who] = len(board)  # a triggered expert fires again only after the board changes
        if not board:
            return [finish(e, OutcomeKind.FAILED, reason="no expert contributed to the board")]
        if not st.done("answer"):
            msgs = [
                Message(
                    role="system",
                    content=f"{self.system} Write the final answer to the task from the board. Reply with the answer only.".strip(),
                ),
                Message(
                    role="user",
                    content=f"{data_block('task', task)}\n\n{data_block('board', self._board(board))}",
                ),
            ]
            return [ask(e, self.name, "answer", msgs, model=self.model)]
        if "answer" in st.failed:
            return (
                [llm_failed(e)]
                if e.type == "aaa.llm.failed"
                else [finish(e, OutcomeKind.FAILED, reason="the solver call failed")]
            )
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "answer": st.text("answer"),
                    "board": [{"author": a, "text": t} for a, t in board],
                    "rounds": rounds,
                },
            )
        ]
