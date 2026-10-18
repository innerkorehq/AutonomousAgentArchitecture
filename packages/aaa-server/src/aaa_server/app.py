"""The ASGI application. Every route is: authenticate -> authorise (scope) -> rate-limit -> validate -> act, and
every act goes through a `TenantScope` / `TenantSessions` bound to the authenticated principal's tenant."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from aaa.algos.human import HUMAN_ANSWERED, HUMAN_REQUESTED
from aaa.core.ids import new_id
from aaa.events import Event
from aaa.llm.parts import Part
from aaa.mcp.server import MCPToolServer
from aaa.runtime.budget import Budget
from aaa.runtime.engine import RUN_COMPLETED, RUNMETA, Algorithm, Engine
from aaa.session import SessionBusy, SessionClosed, SessionManager
from aaa.tools.handler import (
    APPROVAL_DENIED,
    APPROVAL_GRANTED,
    APPROVAL_REQUESTED,
    COMPLETED,
    FAILED,
)

from . import mcp_http
from .a2a import A2AConfig, A2AEndpoint
from .admission import ConcurrencyLimiter, Denied, Quota, QuotaStore, RateLimiter
from .auth import (
    SCOPE_ADMIN,
    SCOPE_ANSWER,
    SCOPE_APPROVE,
    SCOPE_RUN,
    SCOPE_SESSION,
    Authenticator,
    AuthError,
    Principal,
)

log = logging.getLogger("aaa_server")

_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_TURN_RUN = re.compile(r"^.+:t\d+$")
EventView = Callable[[Event], "dict[str, Any] | None"]


# -- what leaves the process ---------------------------------------------------------------------
def _clip(value: Any, limit: int = 4096) -> Any:
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else {"truncated": True, "preview": text[:limit]}


def default_event_view(e: Event) -> dict[str, Any] | None:
    """The projection of the log a client may see. Anything not listed here is not sent: system prompts, memory
    contents, guard internals and raw model requests stay inside."""
    d = e.data
    match e.type:
        case "aaa.run.started":
            return {"type": "run.started", "algo": d.get("algo")}
        case "aaa.run.completed":
            return {"type": "run.completed", "kind": d.get("kind"), "reason": d.get("reason")}
        case "aaa.llm.delta" if d.get("tag") in (None, "main"):
            return {"type": "delta", "text": d.get("text", "")}
        case "aaa.tool.requested":
            return {"type": "tool.requested", "name": d.get("name"), "call_id": d.get("call_id")}
        case "aaa.tool.completed":
            return {"type": "tool.completed", "name": d.get("name"), "call_id": d.get("call_id")}
        case "aaa.tool.failed":
            err = d.get("error") or {}
            return {
                "type": "tool.failed",
                "name": d.get("name"),
                "call_id": d.get("call_id"),
                "error": err.get("type"),
            }
        case "aaa.approval.requested":
            return {
                "type": "approval.requested",
                "name": d.get("name"),
                "call_id": d.get("call_id"),
                "arguments": _clip(d.get("arguments", {})),
            }
        case "aaa.human.requested":
            return {
                "type": "human.requested",
                "question_id": d.get("question_id"),
                "question": _clip(d.get("question", "")),
            }
        case "aaa.session.reply":
            return {"type": "session.reply", "turn": d.get("turn"), "text": d.get("text")}
    return None


# -- configuration -------------------------------------------------------------------------------
@dataclass
class ServerConfig:
    engine: Engine
    authenticator: Authenticator
    algorithms: Mapping[str, Algorithm] = field(default_factory=dict)
    sessions: Mapping[str, SessionManager] = field(default_factory=dict)
    budget: Budget = field(
        default_factory=lambda: Budget(max_steps=50, max_tokens=200_000, deadline_s=300.0)
    )
    rate_limiter: RateLimiter | None = None
    quota: QuotaStore | None = None
    max_open_runs: int | None = None
    max_streams: int = 8
    max_stream_s: float = 900.0
    keepalive_s: float = 15.0
    turn_timeout_s: float = 300.0
    max_body_bytes: int = 1 << 20
    cors_origins: tuple[str, ...] = ()
    event_view: EventView = default_event_view
    mcp: MCPToolServer | None = None  # serve these tools at POST /mcp
    mcp_origins: tuple[str, ...] = ()  # browser Origins allowed to call /mcp; none by default
    a2a: A2AConfig | None = None  # EXPERIMENTAL: serve one algorithm as an A2A agent


# -- request models (unknown fields are an error: a client cannot smuggle in a tenant) -----------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BudgetRequest(_Strict):
    max_steps: int | None = Field(None, ge=1)
    max_tokens: int | None = Field(None, ge=1)
    max_cost: float | None = Field(None, gt=0)
    deadline_s: float | None = Field(None, gt=0)


class RunRequest(_Strict):
    algo: str = Field(max_length=128)
    input: dict[str, Any] = {}
    budget: BudgetRequest | None = None


class ApprovalRequest(_Strict):
    call_id: str = Field(max_length=256)
    decision: Literal["grant", "deny"]
    reason: str | None = Field(None, max_length=500)


class AnswerRequest(_Strict):
    question_id: str = Field(max_length=64)
    answer: str = Field(max_length=2000)


class SessionRequest(_Strict):
    agent: str | None = Field(None, max_length=128)
    title: str | None = Field(None, max_length=200)


class MessageRequest(_Strict):
    text: str = Field(max_length=100_000)
    attachments: list[dict[str, Any]] = []
    stream: bool = False


_PARTS = TypeAdapter(Part)


def _forbid_blob_refs(value: Any) -> None:
    """A client names data inline or by URL, never by blob reference: those are the server's own storage."""
    if isinstance(value, dict):
        if "blob" in value:
            raise ValueError("blob references are not accepted from clients")
        for v in value.values():
            _forbid_blob_refs(v)
    elif isinstance(value, list):
        for v in value:
            _forbid_blob_refs(v)


def clamp_budget(server: Budget, asked: BudgetRequest | None) -> Budget:
    """The server's budget is a ceiling: a request may lower a limit or fill one the server left open."""
    a = asked.model_dump() if asked else {}
    out: dict[str, Any] = {}
    for name in ("max_steps", "max_tokens", "max_cost", "deadline_s"):
        cap, want = getattr(server, name), a.get(name)
        out[name] = cap if want is None else (want if cap is None else min(cap, want))
    return Budget(**out)


# -- the app -------------------------------------------------------------------------------------
class _Reply(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def _error(status: int, code: str, message: str, rid: str, headers: dict[str, str] | None = None):
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": rid}},
        status_code=status,
        headers={"X-Request-Id": rid, **(headers or {})},
    )


def _sse(event: str, data: Any, id: int | None = None) -> bytes:
    head = f"id: {id}\n" if id is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


class ServerApp:
    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.engine = cfg.engine
        self.streams = ConcurrencyLimiter(cfg.max_streams)
        self.a2a = A2AEndpoint(self, cfg.a2a) if cfg.a2a else None
        if cfg.quota is not None:
            self.engine.subscribe(RUN_COMPLETED, self._settle, group="aaa-server.quota")

    # -- plumbing ---------------------------------------------------------------------------
    def _managers(self, name: str | None) -> list[SessionManager]:
        if name is None:
            return list(self.cfg.sessions.values())
        m = self.cfg.sessions.get(name)
        return [m] if m else []

    @staticmethod
    def _public(p: Principal, rid: str) -> str:
        return rid.removeprefix(f"{p.tenant}/")

    async def _raw(self, request: Request) -> bytes:
        limit = self.cfg.max_body_bytes
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise _Reply(413, "too_large", "request body too large")
        chunks, size = [], 0
        async for c in request.stream():
            size += len(c)
            if size > limit:
                raise _Reply(413, "too_large", "request body too large")
            chunks.append(c)
        return b"".join(chunks)

    async def _body(self, request: Request, model: type[Any]) -> Any:
        raw = await self._raw(request)
        try:
            return model.model_validate(json.loads(raw or b"{}"))
        except (ValueError, ValidationError) as e:
            detail = (
                "; ".join(
                    f"{'.'.join(map(str, x['loc']))}: {x['msg']}"
                    for x in e.errors(include_input=False, include_url=False, include_context=False)
                )
                if isinstance(e, ValidationError)
                else "body is not valid JSON"
            )
            raise _Reply(422, "invalid_request", detail) from None

    def route(
        self, scope: str | None, fn: Callable[..., Awaitable[Response]]
    ) -> Callable[[Request], Awaitable[Response]]:
        async def endpoint(request: Request) -> Response:
            rid = new_id()
            try:
                try:
                    p = await self.cfg.authenticator.authenticate(
                        {k.lower(): v for k, v in request.headers.items()}
                    )
                except AuthError as e:
                    log.info("auth refused (request %s): %s", rid, e)
                    raise _Reply(
                        401,
                        "unauthorized",
                        "authentication required",
                        {"WWW-Authenticate": "Bearer"},
                    ) from None
                if scope is not None and not p.can(scope):
                    raise _Reply(403, "forbidden", "this credential may not do that")
                if self.cfg.rate_limiter is not None:
                    d = await self.cfg.rate_limiter.take(f"{p.tenant}/{p.subject}")
                    if not d.allowed:
                        raise Denied(429, "rate_limited", "rate limit exceeded", d.retry_after)
                resp = await fn(request, p)
                resp.headers["X-Request-Id"] = rid
                return resp
            except _Reply as r:
                return _error(r.status, r.code, r.message, rid, r.headers)
            except Denied as d:
                h = (
                    {"Retry-After": str(max(1, int(d.retry_after + 0.999)))}
                    if d.retry_after
                    else {}
                )
                return _error(d.status, d.code, d.message, rid, h)
            except KeyError:
                return _error(404, "not_found", "not found", rid)
            except Exception:
                log.exception("unhandled error (request %s)", rid)
                return _error(500, "internal", "internal error", rid)

        return endpoint

    # -- quota settlement -------------------------------------------------------------------
    async def _settle(self, e: Event) -> list[Event]:
        q = self.cfg.quota
        if q is None or not e.tenant or not e.run_id:
            return []
        try:
            started = await self.engine._started_of(e.run_id)
        except KeyError:
            return []
        parent, algo = started.data.get("parent"), str(started.data.get("algo", ""))
        # Count each unit of work once: top-level runs and session turns (a session's own ledger already
        # contains its turns, and a run's ledger contains its children).
        if algo.startswith("aaa.session:") or (parent and not _TURN_RUN.match(e.run_id)):
            return []
        meta = await self.engine.project(RUNMETA, e.run_id)
        await q.settle(e.tenant, e.run_id, tokens=meta.ledger.tokens, cost=meta.ledger.cost)
        return []

    # -- endpoints: health --------------------------------------------------------------------
    async def healthz(self, request: Request) -> Response:
        return JSONResponse({"ok": True})

    # -- endpoints: runs ------------------------------------------------------------------------
    async def _summary(self, p: Principal, rid: str) -> dict[str, Any]:
        started = await self.engine._started_of(rid)
        meta = await self.engine.project(RUNMETA, rid)
        done = meta.completed
        return {
            "run_id": self._public(p, rid),
            "algo": started.data.get("algo"),
            "status": "completed" if done is not None else "running",
            "outcome": (
                {
                    "kind": done.get("kind"),
                    "value": done.get("value"),
                    "reason": done.get("reason"),
                }
                if done is not None
                else None
            ),
            "usage": meta.ledger.model_dump(),
        }

    async def launch(
        self,
        p: Principal,
        algo: Algorithm,
        input: dict[str, Any],
        *,
        key: str | None = None,
        asked: BudgetRequest | None = None,
    ) -> tuple[str, bool]:
        """Admit and start a run for `p`: (namespaced run id, whether it was newly started). Every entry point (REST,
        A2A) goes through here, so the checks cannot differ between them."""
        try:
            _forbid_blob_refs(input)
        except ValueError as e:
            raise _Reply(422, "invalid_request", str(e)) from None
        scope = self.engine.tenant(p.tenant)
        if key is not None and not _ID.match(key):
            raise _Reply(422, "invalid_request", "Idempotency-Key must match [A-Za-z0-9._-]{1,128}")
        if key is not None:
            rid = scope.run_id(key)
            try:
                started = await self.engine._started_of(rid)
            except KeyError:
                started = None
            if started is not None:  # a replay: same request -> same run; different request -> 409
                same = started.data.get("algo") == algo.name and started.data.get(
                    "input"
                ) == to_jsonable_python(input, serialize_unknown=True)
                if not same:
                    raise _Reply(
                        409, "idempotency_conflict", "that key was used for another request"
                    )
                return rid, False
        if self.cfg.max_open_runs is not None:
            open_now = len(await scope.runs(open_only=True))
            if open_now >= self.cfg.max_open_runs:
                raise Denied(429, "too_many_runs", "too many open runs", retry_after=1.0)
        if self.cfg.quota is not None:
            await self.cfg.quota.admit(p.tenant, scope.run_id(key) if key else new_id())
        handle = await scope.start(
            algo,
            input,
            run_id=key,
            budget=clamp_budget(self.cfg.budget, asked),
            user=p.subject,
        )
        return handle.run_id, True

    async def start_run(self, request: Request, p: Principal) -> Response:
        req: RunRequest = await self._body(request, RunRequest)
        algo = self.cfg.algorithms.get(req.algo)
        if algo is None:
            raise _Reply(404, "unknown_algorithm", "no such algorithm")
        rid, created = await self.launch(
            p, algo, req.input, key=request.headers.get("idempotency-key"), asked=req.budget
        )
        return JSONResponse(await self._summary(p, rid), status_code=202 if created else 200)

    async def list_runs(self, request: Request, p: Principal) -> Response:
        scope = self.engine.tenant(p.tenant)
        open_only = request.query_params.get("open") == "true"
        try:
            limit = min(max(int(request.query_params.get("limit", "50")), 1), 200)
        except ValueError:
            raise _Reply(422, "invalid_request", "limit must be an integer") from None
        infos = (await scope.runs(open_only=open_only))[-limit:]
        return JSONResponse(
            {
                "runs": [
                    {
                        "run_id": self._public(p, r.stream),
                        "algo": r.started.data.get("algo"),
                        "status": "running" if r.completed is None else "completed",
                    }
                    for r in infos
                ]
            }
        )

    async def get_run(self, request: Request, p: Principal) -> Response:
        scope = self.engine.tenant(p.tenant)
        h = await scope.handle(request.path_params["run_id"])
        return JSONResponse(await self._summary(p, h.run_id))

    async def cancel_run(self, request: Request, p: Principal) -> Response:
        h = await self.engine.tenant(p.tenant).handle(request.path_params["run_id"])
        await h.cancel(f"cancelled by {p.subject}")
        return JSONResponse({"run_id": self._public(p, h.run_id), "status": "cancelling"}, 202)

    async def resolve_approval(self, request: Request, p: Principal) -> Response:
        h = await self.engine.tenant(p.tenant).handle(request.path_params["run_id"])
        req: ApprovalRequest = await self._body(request, ApprovalRequest)
        events = await self.engine.store.read(h.run_id)
        asked, settled = set(), set()
        for e in events:
            cid = e.data.get("call_id")
            if e.type == APPROVAL_REQUESTED:
                asked.add(cid)
            elif e.type in (APPROVAL_GRANTED, APPROVAL_DENIED, COMPLETED, FAILED):
                settled.add(cid)
        if req.call_id not in asked or req.call_id in settled:
            raise _Reply(409, "no_pending_approval", "no approval is pending for that call")
        await h.signal(
            APPROVAL_GRANTED if req.decision == "grant" else APPROVAL_DENIED,
            {
                "call_id": req.call_id,
                "by": p.subject,
                **({"reason": req.reason} if req.reason else {}),
            },
        )
        return JSONResponse({"call_id": req.call_id, "decision": req.decision}, 202)

    async def answer_question(self, request: Request, p: Principal) -> Response:
        """A person's answer to the question an `AskHuman` run is waiting on. Like approvals: its own scope, the run's own tenant, and only the
        outstanding question, once, is accepted."""
        h = await self.engine.tenant(p.tenant).handle(request.path_params["run_id"])
        req: AnswerRequest = await self._body(request, AnswerRequest)
        asked, answered = set(), set()
        for e in await self.engine.store.read(h.run_id):
            if e.type == HUMAN_REQUESTED:
                asked.add(str(e.data.get("question_id")))
            elif e.type == HUMAN_ANSWERED:
                answered.add(str(e.data.get("question_id")))
        if req.question_id not in asked or req.question_id in answered:
            raise _Reply(
                409, "no_pending_question", "no question is waiting for an answer with that id"
            )
        await h.signal(
            HUMAN_ANSWERED, {"question_id": req.question_id, "answer": req.answer, "by": p.subject}
        )
        return JSONResponse({"question_id": req.question_id}, 202)

    # -- endpoints: event stream -------------------------------------------------------------
    def _stream_response(
        self,
        p: Principal,
        body: Callable[[], AsyncIterator[bytes]],
        stack: AsyncExitStack,
        *,
        trailer: bool = True,
    ) -> StreamingResponse:
        async def guarded() -> AsyncIterator[bytes]:
            try:
                with anyio.move_on_after(self.cfg.max_stream_s):
                    async for chunk in body():
                        yield chunk
                if trailer:  # protocols with their own framing (A2A) must not see an extra event
                    yield _sse("end", {"reason": "max_duration_or_done"})
            finally:
                await stack.aclose()

        return StreamingResponse(
            guarded(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            background=BackgroundTask(stack.aclose),
        )

    async def _paced(self, source: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Items from `source`, with `None` yielded whenever `keepalive_s` passes quietly."""
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def pump() -> None:
            try:
                async for item in source:
                    await queue.put(("item", item))
                await queue.put(("end", None))
            except Exception as e:  # noqa: BLE001
                await queue.put(("err", e))

        task = asyncio.create_task(pump())
        try:
            while True:
                try:
                    kind, item = await asyncio.wait_for(queue.get(), self.cfg.keepalive_s)
                except TimeoutError:
                    yield None
                    continue
                if kind == "end":
                    return
                if kind == "err":
                    raise item
                yield item
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def run_events(self, request: Request, p: Principal) -> Response:
        h = await self.engine.tenant(p.tenant).handle(request.path_params["run_id"])
        raw = request.headers.get("last-event-id") or request.query_params.get("after")
        try:
            start = int(raw) + 1 if raw is not None else 0
        except ValueError:
            raise _Reply(
                422, "invalid_request", "Last-Event-ID must be an event sequence number"
            ) from None
        if start < 0:
            raise _Reply(422, "invalid_request", "Last-Event-ID must be >= 0")
        stack = AsyncExitStack()
        await stack.enter_async_context(self.streams.hold(f"{p.tenant}"))

        async def body() -> AsyncIterator[bytes]:
            async def finished() -> AsyncIterator[Event]:
                for e in await self.engine.store.read(h.run_id, start):
                    yield e

            # A finished run is read, not tailed: resuming after its last event must end, not wait forever.
            done = (await self.engine.project(RUNMETA, h.run_id)).completed is not None
            src = finished() if done else self.engine.store.subscribe(h.run_id, start)
            async for e in self._paced(src.__aiter__()):
                if e is None:
                    yield b": keepalive\n\n"
                    continue
                view = self.cfg.event_view(await self.engine.hydrate(e))
                if view is not None:
                    yield _sse(view["type"], view, e.seq)
                if e.type == RUN_COMPLETED:
                    return

        return self._stream_response(p, body, stack)

    # -- endpoints: sessions ---------------------------------------------------------------------
    async def _session(self, p: Principal, sid: str) -> Any:
        for m in self.cfg.sessions.values():
            try:
                return await m.tenant(p.tenant).get(sid)
            except KeyError:
                continue
        raise KeyError(sid)

    async def create_session(self, request: Request, p: Principal) -> Response:
        req: SessionRequest = await self._body(request, SessionRequest)
        ms = self._managers(req.agent)
        if len(ms) != 1 or (req.agent is None and len(self.cfg.sessions) != 1):
            raise _Reply(422, "invalid_request", "name one of the configured agents")
        if self.cfg.quota is not None:
            await self.cfg.quota.admit(p.tenant, new_id())
        s = await ms[0].tenant(p.tenant).create(user=p.subject, title=req.title)
        return JSONResponse({"session_id": self._public(p, s.id)}, status_code=201)

    async def list_sessions(self, request: Request, p: Principal) -> Response:
        out = []
        for m in self.cfg.sessions.values():
            for i in await m.tenant(p.tenant).list(user=p.subject):
                out.append({"session_id": self._public(p, i.id), "open": i.open})
        return JSONResponse({"sessions": out})

    async def get_session(self, request: Request, p: Principal) -> Response:
        s = await self._session(p, request.path_params["session_id"])
        turns = [
            {"turn": t.n, "user": t.user, "reply": t.assistant, "status": t.status}
            for t in await s.history()
        ]
        return JSONResponse({"session_id": self._public(p, s.id), "turns": turns})

    async def close_session(self, request: Request, p: Principal) -> Response:
        s = await self._session(p, request.path_params["session_id"])
        await s.close(f"closed by {p.subject}")
        return JSONResponse({"session_id": self._public(p, s.id), "status": "closed"})

    async def send_message(self, request: Request, p: Principal) -> Response:
        s = await self._session(p, request.path_params["session_id"])
        req: MessageRequest = await self._body(request, MessageRequest)
        try:
            _forbid_blob_refs(req.attachments)
            parts = [_PARTS.validate_python(a) for a in req.attachments]
        except ValueError as e:
            raise _Reply(422, "invalid_request", f"bad attachment: {e}") from None
        if self.cfg.quota is not None:
            await self.cfg.quota.admit(p.tenant, new_id())
        try:
            if not req.stream:
                r = await s.send(req.text, attachments=parts, timeout_s=self.cfg.turn_timeout_s)
                return JSONResponse(
                    {"turn": r.turn, "text": r.text, "kind": r.kind, "reason": r.reason}
                )
            ts = await s.stream(req.text, attachments=parts)
        except (SessionBusy, SessionClosed) as e:
            raise _Reply(
                409, type(e).__name__.lower(), str(e).replace(s.id, "this session")
            ) from None
        except TimeoutError:
            raise _Reply(504, "timeout", "the turn did not finish in time") from None
        stack = AsyncExitStack()
        await stack.enter_async_context(self.streams.hold(f"{p.tenant}"))

        async def body() -> AsyncIterator[bytes]:
            async for piece in ts:
                yield _sse("delta", {"type": "delta", "text": piece})
            r = await ts.result()
            yield _sse(
                "session.reply",
                {"type": "session.reply", "turn": r.turn, "text": r.text, "kind": r.kind},
            )

        return self._stream_response(p, body, stack)

    # -- endpoints: A2A -----------------------------------------------------------------------
    async def agent_card(self, request: Request) -> Response:
        assert self.cfg.a2a is not None
        return JSONResponse(self.cfg.a2a.card())

    # -- endpoints: MCP ---------------------------------------------------------------------------
    async def mcp_post(self, request: Request, p: Principal) -> Response:
        core = self.cfg.mcp
        if core is None:
            raise _Reply(404, "not_found", "not found")
        origin = request.headers.get("origin")
        if (
            origin is not None and origin not in self.cfg.mcp_origins
        ):  # DNS-rebinding defence (MCP spec)
            raise _Reply(403, "forbidden", "origin not allowed")
        try:
            message = json.loads(await self._raw(request))
        except ValueError:
            return JSONResponse(mcp_http._error(None, mcp_http.PARSE_ERROR, "parse error"), 400)
        answer = await mcp_http.handle(
            core, message, tenant=p.tenant, user=p.subject, scopes=p.scopes
        )
        return Response(status_code=202) if answer is None else JSONResponse(answer)

    async def mcp_other(self, request: Request, p: Principal) -> Response:
        return Response(status_code=405, headers={"Allow": "POST"})

    # -- endpoints: usage --------------------------------------------------------------------------
    async def usage(self, request: Request, p: Principal) -> Response:
        q = self.cfg.quota
        if q is None:
            raise _Reply(404, "not_found", "not found")
        u = await q.usage(p.tenant)
        lim: Quota = q.limits(p.tenant)
        return JSONResponse({"usage": u.model_dump(), "limits": lim.model_dump()})


def create_app(cfg: ServerConfig, *, manage_engine: bool = False) -> Starlette:
    """The ASGI app. With `manage_engine=True` the app starts and stops `cfg.engine` with the server (what
    `aaa-server serve` does); otherwise you own the engine's lifetime."""
    app = ServerApp(cfg)
    r = app.route
    routes = [
        Route("/healthz", app.healthz, methods=["GET"]),
        Route("/v1/runs", r(SCOPE_RUN, app.start_run), methods=["POST"]),
        Route("/v1/runs", r(SCOPE_RUN, app.list_runs), methods=["GET"]),
        Route("/v1/runs/{run_id}", r(SCOPE_RUN, app.get_run), methods=["GET"]),
        Route("/v1/runs/{run_id}/events", r(SCOPE_RUN, app.run_events), methods=["GET"]),
        Route("/v1/runs/{run_id}/cancel", r(SCOPE_RUN, app.cancel_run), methods=["POST"]),
        Route(
            "/v1/runs/{run_id}/approvals", r(SCOPE_APPROVE, app.resolve_approval), methods=["POST"]
        ),
        Route("/v1/runs/{run_id}/answers", r(SCOPE_ANSWER, app.answer_question), methods=["POST"]),
        Route("/v1/sessions", r(SCOPE_SESSION, app.create_session), methods=["POST"]),
        Route("/v1/sessions", r(SCOPE_SESSION, app.list_sessions), methods=["GET"]),
        Route("/v1/sessions/{session_id}", r(SCOPE_SESSION, app.get_session), methods=["GET"]),
        Route("/v1/sessions/{session_id}", r(SCOPE_SESSION, app.close_session), methods=["DELETE"]),
        Route(
            "/v1/sessions/{session_id}/messages",
            r(SCOPE_SESSION, app.send_message),
            methods=["POST"],
        ),
        *(
            [
                Route("/a2a", r(SCOPE_RUN, app.a2a.rpc), methods=["POST"]),
                Route("/.well-known/agent-card.json", app.agent_card, methods=["GET"]),
            ]
            if app.a2a and cfg.a2a
            else []
        ),
        Route("/mcp", r(SCOPE_RUN, app.mcp_post), methods=["POST"]),
        Route("/mcp", r(SCOPE_RUN, app.mcp_other), methods=["GET", "DELETE"]),
        Route("/v1/usage", r(SCOPE_ADMIN, app.usage), methods=["GET"]),
    ]
    middleware = []
    if cfg.cors_origins:
        if "*" in cfg.cors_origins:
            raise ValueError("list CORS origins explicitly; a wildcard is not accepted")
        middleware.append(
            Middleware(
                CORSMiddleware,
                allow_origins=list(cfg.cors_origins),
                allow_methods=["GET", "POST", "DELETE"],
                allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Last-Event-ID"],
                allow_credentials=False,
            )
        )

    @asynccontextmanager
    async def engine_lifespan(_: Starlette) -> AsyncIterator[None]:
        async with cfg.engine:
            yield

    return Starlette(
        routes=routes, middleware=middleware, lifespan=engine_lifespan if manage_engine else None
    )


__all__ = [
    "ApprovalRequest",
    "BudgetRequest",
    "MessageRequest",
    "RunRequest",
    "ServerConfig",
    "clamp_budget",
    "create_app",
    "default_event_view",
]
