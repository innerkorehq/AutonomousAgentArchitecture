"""Role-play with a fixed protocol (CAMEL / MetaGPT style): roles in sequence, passing **typed documents**.

Each `Role` has instructions and an output model (a pydantic class). A role receives the task and the previous role's *validated* document, and must
reply with JSON that validates against its own output model; an invalid document is sent back with the validation errors for up to `max_repairs` repairs,
then the run fails. So what travels between roles is always data of a known shape, never free text another role could be talked into trusting."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from aaa.algos.base import CALLS, ask, data_block, finish, llm_failed, parse_json_obj, task_text
from aaa.core.outcome import OutcomeKind
from aaa.events import Event
from aaa.llm.types import Message
from aaa.runtime import Ctx

MAX_ROLES, MAX_REPAIRS = 8, 3
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,19}$")


@dataclass(frozen=True)
class Role:
    name: str
    instructions: str
    output: type[BaseModel]
    model: str | None = None


class RolePlaySOP:
    """A pipeline of roles passing validated documents.

    Result: `{"document": <last role's document>, "documents": {role: document}, "roles": [...], "repairs": n}`. At most
    `len(roles) * (max_repairs + 1)` model calls."""

    name = "role_play_sop"

    def __init__(self, roles: Sequence[Role], *, max_repairs: int = 1, system: str = "") -> None:
        if not 1 <= len(roles) <= MAX_ROLES:
            raise ValueError(f"a protocol needs between 1 and {MAX_ROLES} roles")
        if not 0 <= max_repairs <= MAX_REPAIRS:
            raise ValueError(f"max_repairs must be between 0 and {MAX_REPAIRS}")
        names = [r.name for r in roles]
        if len(set(names)) != len(names) or not all(_NAME.match(n) for n in names):
            raise ValueError("role names must be unique, lowercase letters, digits and '_'")
        self.roles, self.max_repairs, self.system = list(roles), max_repairs, system

    @property
    def max_calls(self) -> int:
        return len(self.roles) * (self.max_repairs + 1)

    def subscriptions(self) -> list[str]:
        return ["aaa.run.started", "aaa.llm.completed", "aaa.llm.failed"]

    async def on_event(self, e: Event, ctx: Ctx) -> list[Event]:
        st = await ctx.project(CALLS)
        docs: dict[str, dict[str, Any]] = {}
        previous: dict[str, Any] | None = None
        repairs = 0
        for role in self.roles:
            problem = ""
            for attempt in range(self.max_repairs + 1):
                tag = f"{role.name}.{attempt}"
                if not st.done(tag):
                    parts = [data_block("task", task_text(st.input))]
                    if previous is not None:
                        parts.append(
                            data_block(
                                "document_from_previous_role",
                                json.dumps(previous, ensure_ascii=False),
                            )
                        )
                    if problem:
                        parts.append(data_block("your_previous_reply_was_invalid", problem))
                    msgs = [
                        Message(
                            role="system",
                            content=f"{self.system} You are the {role.name}. {role.instructions} Reply with one JSON object that matches the schema, and nothing else.".strip(),
                        ),
                        Message(role="user", content="\n\n".join(parts)),
                    ]
                    return [
                        ask(
                            e,
                            self.name,
                            tag,
                            msgs,
                            model=role.model,
                            schema=role.output.model_json_schema(),
                        )
                    ]
                if tag in st.failed:
                    return (
                        [llm_failed(e)]
                        if e.type == "aaa.llm.failed"
                        else [finish(e, OutcomeKind.FAILED, reason=f"the {role.name} call failed")]
                    )
                obj = parse_json_obj(st.text(tag))
                try:
                    doc = role.output.model_validate(obj if obj is not None else {})
                except ValidationError as err:
                    problem = "; ".join(
                        f"{'.'.join(map(str, x['loc'])) or 'document'}: {x['msg']}"
                        for x in err.errors()
                    )[:600]
                    repairs += 1
                    continue
                previous = doc.model_dump(mode="json")
                docs[role.name] = previous
                break
            else:
                return [
                    finish(
                        e,
                        OutcomeKind.FAILED,
                        reason=f"the {role.name} could not produce a valid document: {problem}",
                    )
                ]
        return [
            finish(
                e,
                OutcomeKind.SUCCESS,
                {
                    "document": previous,
                    "documents": docs,
                    "roles": [r.name for r in self.roles],
                    "repairs": max(0, repairs),
                },
            )
        ]
