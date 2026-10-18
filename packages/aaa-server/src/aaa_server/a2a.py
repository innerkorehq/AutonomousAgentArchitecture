"""Agent2Agent (A2A) protocol, EXPERIMENTAL: serve one algorithm as an A2A agent.

Targets **A2A 1.0**, the JSON-RPC binding (`SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`,
`SubscribeToTask`) and an agent card at `/.well-known/agent-card.json`. The wire shapes were taken from the protocol's own
generated types and checked against the official `a2a-sdk` client in the tests. Out of scope: push notifications, the extended
card, non-text parts, gRPC and REST bindings. A message naming an existing task is accepted in one case only: the task is waiting for a person's answer.

A2A tasks ARE runs: they go through the same authentication, tenant scope, admission and budget as `POST /v1/runs`, and the
tenant comes only from the credential. A task that parks on a human approval is reported as `TASK_STATE_INPUT_REQUIRED`; the
approval itself is given through the REST API by a principal with the `approve` scope. A task whose algorithm (`AskHuman`) is waiting for a person's answer is also
`TASK_STATE_INPUT_REQUIRED`, carrying the question; the client answers by sending a message with that `taskId`, which needs the `answer` scope."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from aaa.algos.human import HUMAN_ANSWERED, HUMAN_REQUESTED
from aaa.core.ids import new_id
from aaa.core.outcome import OutcomeKind
from aaa.runtime.engine import RUN_COMPLETED, RUNMETA
from aaa.tools.handler import (
    APPROVAL_DENIED,
    APPROVAL_GRANTED,
    APPROVAL_REQUESTED,
    COMPLETED,
    FAILED,
)

from .auth import SCOPE_ANSWER, Principal

if TYPE_CHECKING:
    from .app import ServerApp

VERSION = "1.0"
TASK_NOT_FOUND, NOT_CANCELABLE, UNSUPPORTED_OP = -32001, -32002, -32004
CONTENT_TYPE, VERSION_UNSUPPORTED = -32005, -32009
INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL, PARSE_ERROR = (
    -32600,
    -32601,
    -32602,
    -32603,
    -32700,
)
MAX_TEXT = 100_000


def _default_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in (
            "text",
            "answer",
        ):  # the result field of ToolLoop-style and answer-style algorithms
            if isinstance(value.get(key), str):
                return value[key]
    return value if isinstance(value, str) else json.dumps(value, default=str)


@dataclass
class A2AConfig:
    """`url` is this server's public address (the card must carry an absolute URL)."""

    algorithm: str  # a key of ServerConfig.algorithms
    url: str
    name: str = "AAA agent"
    description: str = "An agent served by aaa-server."
    version: str = "0.1"
    skills: tuple[dict[str, Any], ...] = ()
    to_input: Callable[[str], dict[str, Any]] = lambda text: {"task": text}
    result_text: Callable[[Any], str] = _default_text
    wait_s: float = (
        300.0  # how long a blocking SendMessage waits before answering with the working task
    )
    extra: dict[str, Any] = field(default_factory=dict)

    def card(self) -> dict[str, Any]:
        skills = list(self.skills) or [
            {"id": self.algorithm, "name": self.name, "description": self.description, "tags": []}
        ]
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "supportedInterfaces": [
                {
                    "url": self.url.rstrip("/") + "/a2a",
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": VERSION,
                }
            ],
            "capabilities": {"streaming": True, "pushNotifications": False},
            "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
            "securityRequirements": [{"schemes": {"bearer": {}}}],
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": skills,
            **self.extra,
        }


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _agent_message(task_id: str, text: str) -> dict[str, Any]:
    return {
        "messageId": new_id(),
        "taskId": task_id,
        "contextId": task_id,
        "role": "ROLE_AGENT",
        "parts": [{"text": text}],
    }


class A2AEndpoint:
    def __init__(self, app: ServerApp, cfg: A2AConfig) -> None:
        self.app, self.cfg = app, cfg

    @property
    def algo(self) -> Any:
        return self.app.cfg.algorithms[self.cfg.algorithm]

    # -- task views -------------------------------------------------------------------------
    async def _pending_approval(self, rid: str) -> bool:
        asked, settled = set(), set()
        for e in await self.app.engine.store.read(rid):
            cid = e.data.get("call_id")
            if e.type == APPROVAL_REQUESTED:
                asked.add(cid)
            elif e.type in (APPROVAL_GRANTED, APPROVAL_DENIED, COMPLETED, FAILED):
                settled.add(cid)
        return bool(asked - settled)

    async def _pending_question(self, rid: str) -> tuple[str, str] | None:
        """(question id, question) of the question a person has not yet answered, if any."""
        asked: dict[str, str] = {}
        for e in await self.app.engine.store.read(rid):
            if e.type == HUMAN_REQUESTED:
                asked[str(e.data.get("question_id"))] = str(e.data.get("question", ""))
            elif e.type == HUMAN_ANSWERED:
                asked.pop(str(e.data.get("question_id")), None)
        return next(iter(asked.items()), None)

    async def task(self, p: Principal, rid: str, *, artifacts: bool = True) -> dict[str, Any]:
        tid = self.app._public(p, rid)
        meta = await self.app.engine.project(RUNMETA, rid)
        done, message = meta.completed, None
        if done is None:
            question = await self._pending_question(rid)
            if await self._pending_approval(rid):
                state, message = "TASK_STATE_INPUT_REQUIRED", "waiting for a human approval"
            elif question is not None:
                state, message = (
                    "TASK_STATE_INPUT_REQUIRED",
                    f"waiting for an answer: {question[1]}",
                )
            else:
                state = "TASK_STATE_WORKING"
        else:
            kind = str(done.get("kind"))
            state = {
                OutcomeKind.SUCCESS.value: "TASK_STATE_COMPLETED",
                OutcomeKind.CANCELLED.value: "TASK_STATE_CANCELED",
                OutcomeKind.BLOCKED.value: "TASK_STATE_REJECTED",
            }.get(kind, "TASK_STATE_FAILED")
            if state != "TASK_STATE_COMPLETED":
                message = done.get("reason") or str(kind)
        out: dict[str, Any] = {
            "id": tid,
            "contextId": tid,
            "status": {
                "state": state,
                "timestamp": _now(),
                **({"message": _agent_message(tid, message)} if message else {}),
            },
        }
        if artifacts and state == "TASK_STATE_COMPLETED":
            out["artifacts"] = [
                {
                    "artifactId": "result",
                    "name": "result",
                    "parts": [{"text": self.cfg.result_text(done.get("value") if done else None)}],
                }
            ]
        return out

    async def _owned(self, p: Principal, params: dict[str, Any]) -> Any:
        tid = params.get("id")
        if not isinstance(tid, str) or not tid:
            raise RpcError(INVALID_PARAMS, "id is required")
        try:
            h = await self.app.engine.tenant(p.tenant).handle(tid)
            started = await self.app.engine._started_of(h.run_id)
        except KeyError:
            raise RpcError(TASK_NOT_FOUND, "task not found") from None
        if started.data.get("algo") != self.algo.name:
            raise RpcError(TASK_NOT_FOUND, "task not found")
        return h

    # -- methods ----------------------------------------------------------------------------
    def _text_of(self, params: dict[str, Any], *, follow_up: bool = False) -> str:
        msg = params.get("message")
        if not isinstance(msg, dict) or msg.get("role") not in (None, "ROLE_USER"):
            raise RpcError(INVALID_PARAMS, "message with role ROLE_USER is required")
        if msg.get("taskId") and not follow_up:
            raise RpcError(UNSUPPORTED_OP, "continuing an existing task is not supported")
        parts = msg.get("parts")
        if not isinstance(parts, list) or not parts:
            raise RpcError(INVALID_PARAMS, "message.parts is required")
        texts = []
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise RpcError(CONTENT_TYPE, "only text parts are supported")
            texts.append(part["text"])
        text = "\n".join(texts)
        if len(text) > MAX_TEXT:
            raise RpcError(INVALID_PARAMS, "message is too long")
        return text

    async def _answer(self, p: Principal, params: dict[str, Any]) -> str:
        """A message that names an existing task: accepted only as the answer to the question the task is waiting on. Returns the run id."""
        task_id = params["message"]["taskId"]
        h = await self._owned(p, {"id": task_id})
        question = (
            None
            if (await self.app.engine.project(RUNMETA, h.run_id)).completed
            else await self._pending_question(h.run_id)
        )
        if question is None:
            raise RpcError(UNSUPPORTED_OP, "this task is not waiting for an answer from you")
        if not p.can(SCOPE_ANSWER):
            raise RpcError(UNSUPPORTED_OP, "answering a task's question needs the 'answer' scope")
        text = self._text_of(params, follow_up=True)
        await h.signal(
            HUMAN_ANSWERED, {"question_id": question[0], "answer": text[:2000], "by": p.subject}
        )
        return h.run_id

    async def send(self, p: Principal, params: dict[str, Any]) -> dict[str, Any]:
        msg = params.get("message")
        if isinstance(msg, dict) and msg.get("taskId"):
            rid = await self._answer(p, params)
            await asyncio.sleep(0)  # let the run take the answer before the task is described
            return {"task": await self.task(p, rid)}
        text = self._text_of(params)
        rid, _ = await self.app.launch(p, self.algo, self.cfg.to_input(text))
        cfgd = params.get("configuration") or {}
        if not (isinstance(cfgd, dict) and cfgd.get("returnImmediately")):
            try:
                await asyncio.wait_for(self.app.engine.handle(rid).result(), self.cfg.wait_s)
            except TimeoutError:
                pass
        return {"task": await self.task(p, rid)}

    async def get(self, p: Principal, params: dict[str, Any]) -> dict[str, Any]:
        h = await self._owned(p, params)
        return await self.task(p, h.run_id)

    async def cancel(self, p: Principal, params: dict[str, Any]) -> dict[str, Any]:
        h = await self._owned(p, params)
        if (await self.app.engine.project(RUNMETA, h.run_id)).completed is not None:
            raise RpcError(NOT_CANCELABLE, "task is already finished")
        await h.cancel(f"cancelled by {p.subject}")
        try:
            await asyncio.wait_for(h.result(), 5)
        except TimeoutError:
            pass
        return await self.task(p, h.run_id)

    async def list(self, p: Principal, params: dict[str, Any]) -> dict[str, Any]:
        infos = [
            r
            for r in await self.app.engine.tenant(p.tenant).runs()
            if r.started.data.get("algo") == self.algo.name
        ][-50:]
        tasks = [await self.task(p, r.stream, artifacts=False) for r in infos]
        return {
            "tasks": tasks,
            "nextPageToken": "",
            "pageSize": len(tasks),
            "totalSize": len(tasks),
        }

    async def frames(
        self, p: Principal, rid: str, *, snapshot: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """StreamResponse objects: the task, then text as it is written, then the terminal status."""
        tid = snapshot["id"]
        yield {"task": snapshot}
        streamed = False
        start = (
            (await self.app.engine.store.last_seq(rid)) + 1
            if snapshot["status"]["state"] != "TASK_STATE_WORKING"
            else 0
        )
        async for e in self.app.engine.store.subscribe(rid, start):
            if e.type == "aaa.llm.delta" and e.data.get("tag") in (None, "main"):
                yield {
                    "artifactUpdate": {
                        "taskId": tid,
                        "contextId": tid,
                        "artifact": {
                            "artifactId": "stream",
                            "parts": [{"text": e.data.get("text", "")}],
                        },
                        "append": streamed,
                    }
                }
                streamed = True
            elif e.type == RUN_COMPLETED:
                final = await self.task(p, rid)
                if final.get("artifacts"):
                    yield {
                        "artifactUpdate": {
                            "taskId": tid,
                            "contextId": tid,
                            "artifact": final["artifacts"][0],
                            "lastChunk": True,
                        }
                    }
                yield {"statusUpdate": {"taskId": tid, "contextId": tid, "status": final["status"]}}
                return

    # -- HTTP entry --------------------------------------------------------------------------
    async def rpc(self, request: Request, p: Principal) -> Response:
        ver = request.headers.get("a2a-version")
        raw = await self.app._raw(request)
        try:
            msg = json.loads(raw)
        except ValueError:
            return JSONResponse(self._err(None, PARSE_ERROR, "parse error"), 400)
        if (
            not isinstance(msg, dict)
            or msg.get("jsonrpc") != "2.0"
            or not isinstance(msg.get("method"), str)
        ):
            return JSONResponse(self._err(None, INVALID_REQUEST, "invalid JSON-RPC request"), 400)
        id_, method, params = msg.get("id"), msg["method"], msg.get("params") or {}
        if ver and ver.split(".")[0] != "1":
            return JSONResponse(self._err(id_, VERSION_UNSUPPORTED, "only A2A 1.x is supported"))
        if not isinstance(params, dict):
            return JSONResponse(self._err(id_, INVALID_PARAMS, "params must be an object"))
        try:
            if method in ("SendStreamingMessage", "SubscribeToTask"):
                return await self._stream(p, id_, method, params)
            fn = {
                "SendMessage": self.send,
                "GetTask": self.get,
                "CancelTask": self.cancel,
                "ListTasks": self.list,
            }.get(method)
            if fn is None:
                return JSONResponse(
                    self._err(id_, METHOD_NOT_FOUND, f"method {method!r} is not supported")
                )
            return JSONResponse({"jsonrpc": "2.0", "id": id_, "result": await fn(p, params)})
        except RpcError as e:
            return JSONResponse(self._err(id_, e.code, e.message))

    async def _stream(
        self, p: Principal, id_: Any, method: str, params: dict[str, Any]
    ) -> Response:
        from contextlib import AsyncExitStack

        if method == "SendStreamingMessage":
            msg = params.get("message")
            if isinstance(msg, dict) and msg.get("taskId"):
                rid = await self._answer(p, params)
            else:
                text = self._text_of(params)
                rid, _ = await self.app.launch(p, self.algo, self.cfg.to_input(text))
        else:
            h = await self._owned(p, params)
            rid = h.run_id
            if (await self.app.engine.project(RUNMETA, rid)).completed is not None:
                raise RpcError(UNSUPPORTED_OP, "task is already finished")
        snapshot = await self.task(p, rid)
        stack = AsyncExitStack()
        await stack.enter_async_context(self.app.streams.hold(p.tenant))

        async def body() -> AsyncIterator[bytes]:
            async for frame in self.frames(p, rid, snapshot=snapshot):
                yield f"data: {json.dumps({'jsonrpc': '2.0', 'id': id_, 'result': frame})}\n\n".encode()

        return self.app._stream_response(p, body, stack, trailer=False)

    @staticmethod
    def _err(id_: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}
