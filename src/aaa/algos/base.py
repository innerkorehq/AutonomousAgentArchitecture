from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from aaa.core.outcome import OutcomeKind
from aaa.events import Event, Projection
from aaa.llm.types import LLMRequest, LLMResponse, Message


def llm_request(
    parent: Event, req: LLMRequest, tag: str = "main", *, prev: str | None = None
) -> Event:
    """Build an `aaa.llm.requested` event.

    With `prev` (the id of the previous request event of the SAME conversation), `req.messages`
    must hold only the messages appended since that request's reply; the event then stores just
    that delta plus a pointer. The LLM handler reconstructs the full transcript the provider
    needs, so the log (and algorithm state) grows linearly with turns, not quadratically."""
    body = req.model_dump(mode="json")
    data: dict[str, Any] = {"request": body, "tag": tag}
    if prev is not None:
        data["conv"] = {"prev": prev, "delta": body["messages"]}
        body["messages"] = []
    return parent.caused("aaa.llm.requested", data)


def stable(event: Event, key: str) -> Event:
    """The same event with an id fixed by the run and `key`. Fan-in steps ("every sample is in, start the next round") can be
    reached from more than one triggering event when completions land together; a fixed id makes the store drop the repeat."""
    return event.model_copy(update={"id": f"{event.run_id}:{key}"})


def finish(parent: Event, kind: OutcomeKind, value: Any = None, reason: str | None = None) -> Event:
    return parent.caused(
        "aaa.run.completed", {"kind": kind.value, "value": value, "reason": reason}
    )


def llm_failed(e: Event) -> Event:
    """End the run because its model call failed: `BLOCKED` when a guardrail did it, else `FAILED`."""
    err = e.data["error"]
    kind = OutcomeKind.BLOCKED if err["type"] == "GuardrailBlocked" else OutcomeKind.FAILED
    return finish(e, kind, reason=f"{err['type']}: {err['message']}")


def tool_request(parent: Event, call_id: str, name: str, arguments: dict[str, Any]) -> Event:
    return parent.caused(
        "aaa.tool.requested", {"call_id": call_id, "name": name, "arguments": arguments}
    )


_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```\s*$", re.DOTALL)


def strip_fence(text: str) -> str:
    t = text.strip()
    m = _FENCE.match(t)
    return m.group(1) if m else t


def parse_json_obj(text: str | None) -> dict[str, Any] | None:
    """Best-effort: a JSON object from raw model text (fenced or surrounded by prose)."""
    if not text:
        return None
    t = strip_fence(text)
    for cand in (t, t[t.find("{") : t.rfind("}") + 1] if "{" in t else ""):
        try:
            v = json.loads(cand)
        except ValueError:
            continue
        if isinstance(v, dict):
            return v
    return None


def render_result(payload: Any) -> str:
    return payload if isinstance(payload, str) else json.dumps(payload, default=str)


def first_started(hist: list[Event]) -> Event:
    return next(e for e in hist if e.type == "aaa.run.started")


def task_messages(inp: dict[str, Any], system: str) -> list[Message]:
    if "messages" in inp:
        return [Message.model_validate(m) for m in inp["messages"]]
    msgs = [Message(role="system", content=system)] if system else []
    user: dict[str, Any] = {"role": "user", "content": str(inp.get("task", ""))}
    if inp.get(
        "attachments"
    ):  # images/audio/files for the task: parts or their JSON form (see aaa.llm.parts)
        user["parts"] = [
            p.model_dump(mode="json", exclude_none=True) if hasattr(p, "model_dump") else p
            for p in inp["attachments"]
        ]
    msgs.append(Message.model_validate(user))
    return msgs


# -- tagged calls: the fold most multi-call algorithms need ------------------------------------------
class Calls(BaseModel):
    """What the model and the tools have answered so far, by tag (model calls) or call id (tool calls).

    An algorithm built on this keeps no state of its own: every decision is a pure function of `Calls`, so a replay and a
    different completion order give the same run."""

    input: dict[str, Any] = Field(default_factory=dict)
    texts: dict[str, str] = Field(default_factory=dict)  # tag -> reply text
    results: dict[str, Any] = Field(default_factory=dict)  # tool call id -> result
    failed: list[str] = Field(default_factory=list)  # tags / call ids that failed

    def done(self, key: str) -> bool:
        return key in self.texts or key in self.results or key in self.failed

    def all_done(self, keys: Iterable[str]) -> bool:
        return all(self.done(k) for k in keys)

    def text(self, key: str) -> str:
        return self.texts.get(key, "")


def _apply_calls(st: Calls, e: Event) -> Calls:
    d = e.data
    if e.type == "aaa.run.started":
        st.input = d["input"]
    elif e.type == "aaa.llm.completed":
        st.texts[str(d.get("tag", ""))] = (
            LLMResponse.model_validate(d["response"]).message.content or ""
        )
    elif e.type == "aaa.llm.failed":
        st.failed.append(str(d.get("tag", "")))
    elif e.type == "aaa.tool.completed":
        st.results[d["call_id"]] = d["result"]
    elif e.type == "aaa.tool.failed":
        st.failed.append(d["call_id"])
    return st


CALLS = Projection(Calls, _apply_calls, name="aaa.calls", version=1, state_type=Calls)


def ask(
    e: Event,
    algo: str,
    tag: str,
    messages: list[Message],
    *,
    model: str | None = None,
    params: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> Event:
    """One tagged model call with an id fixed by the run, `algo` and `tag`: asking again for the same tag is dropped by the store,
    so a driver may re-issue every call whose inputs are ready without ever making a call twice."""
    req = LLMRequest(model=model, messages=messages, params=params or {}, response_schema=schema)
    return stable(llm_request(e, req, tag), f"{algo}:{tag}")


def task_text(inp: Mapping[str, Any]) -> str:
    """The task as plain text (the last user message when `messages` were given)."""
    return task_messages(dict(inp), "")[-1].content or ""


def clamp_score(text: str, lo: float = 0.0, hi: float = 10.0) -> float | None:
    """A `{"score": number}` reply as a number in [lo, hi]; `None` when the reply is not usable."""
    obj = parse_json_obj(text)
    try:
        return max(lo, min(hi, float((obj or {})["score"])))
    except (KeyError, TypeError, ValueError):
        return None


SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
    "additionalProperties": False,
}


def data_block(label: str, text: str) -> str:
    """Text the model should treat as quoted material, not as instructions: `<label> ... </label>`. Text is stripped of the closing tag so
    it cannot end its own quote."""
    return f"<{label}>\n{text.replace(f'</{label}>', '')}\n</{label}>"
