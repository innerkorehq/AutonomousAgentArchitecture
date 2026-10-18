# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""Another MCP server as a source: its tools join this server's, under this server's policy.

The upstream is untrusted input (see docs/security/mcp-generator.md, T10 to T14):

* a stdio upstream is launched from an argument vector (no shell) with a MINIMAL environment: only `env` and the variables named in
  `env_from`, so a secret in this process's environment is not inherited by accident;
* an http upstream is reached through a transport that resolves the host once, requires a public address (and `allowed_hosts`),
  connects to that address with the original name as `Host`/SNI, and follows no redirects (the SDK's own client is replaced by ours);
* each connection is owned by ONE background task, so it is opened and closed in the same task whoever calls `aclose()` (the SDK's
  transports cannot be closed from another task);
* its tool descriptions and results are capped, and its tools require approval unless it marks them read-only."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from aaa.mcp.client import MCPClient, MCPToolInfo, _attr
from aaa.mcp.config import Secret, SpecError
from aaa.mcp.model import MCPAnnotations
from aaa.mcp.sources.mcp_passthrough import UpstreamPrompt, UpstreamResources, paged
from aaa.mcp.spec import McpSource
from aaa.tools.builtin.net import FetchBlocked, FetchPolicy, SafeFetcher
from aaa.tools.tool import Tool

log = logging.getLogger("aaa.mcp.upstream")
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def subprocess_allowed() -> bool:
    return os.environ.get("AAA_MCP_ALLOW_SUBPROCESS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


@dataclass
class Loaded:
    tools: list[Tool] = field(default_factory=list)
    annotations: dict[str, MCPAnnotations] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    upstream: Upstream | None = None
    resources: Any = None  # an UpstreamResources provider, if the source asked for resources
    prompts: dict[str, Any] = field(default_factory=dict)  # exposed name -> UpstreamPrompt


# -- the http guard ----------------------------------------------------------------------------
def _policy(src: McpSource) -> FetchPolicy:
    from urllib.parse import urlsplit

    u = urlsplit(src.url or "")
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or not host or u.username or u.password:
        raise SpecError(f"mcp {src.name}: url must be http(s) with a host and no credentials")
    if u.scheme == "http" and not (src.allow_insecure_loopback and host in LOOPBACK):
        raise SpecError(
            f"mcp {src.name}: plain http is only allowed to loopback with allow_insecure_loopback = true; use https"
        )
    import fnmatch as fn

    if not any(
        fn.fnmatchcase(host, h.lower()) or (h.startswith("*.") and host == h[2:].lower())
        for h in src.allowed_hosts
    ):
        raise SpecError(
            f"mcp {src.name}: the host {host!r} is not in allowed_hosts {src.allowed_hosts}"
        )
    port = u.port or (443 if u.scheme == "https" else 80)
    return FetchPolicy(
        allowed_hosts=tuple(src.allowed_hosts),
        allowed_ports=(port,),
        allow_http=u.scheme == "http",
        max_redirects=0,
        timeout_s=src.call_timeout_s,
        connect_timeout_s=src.connect_timeout_s,
        allow_private_networks=src.allow_private_networks
        or (u.scheme == "http" and host in LOOPBACK),
    )


def guarded_client(hx: Any, policy: FetchPolicy, headers: dict[str, str], src: McpSource) -> Any:
    """An `AsyncClient` of the given httpx flavour (`httpx` for SDK 1.x, `httpx2` for 2.x) whose every request is validated and pinned
    exactly as `SafeFetcher` does for a fetch: allowed host, public address, connection to the checked address, no redirects."""
    checker = SafeFetcher(policy)

    class Guard(hx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.inner = hx.AsyncHTTPTransport()

        async def handle_async_request(self, request: Any) -> Any:
            try:
                connect, host, host_header = await checker._target(str(request.url))
            except FetchBlocked as e:
                raise hx.ConnectError(f"blocked: {e}") from None
            headers = hx.Headers(request.headers)
            headers["host"] = host_header
            ext = dict(request.extensions)
            if connect.scheme == "https":
                ext["sni_hostname"] = host
            pinned = hx.Request(
                request.method,
                hx.URL(str(connect)),
                headers=headers,
                stream=request.stream,
                extensions=ext,
            )
            return await self.inner.handle_async_request(pinned)

        async def aclose(self) -> None:
            await self.inner.aclose()

    return hx.AsyncClient(
        transport=Guard(),
        headers=headers,
        follow_redirects=False,
        trust_env=False,
        timeout=hx.Timeout(src.call_timeout_s, connect=src.connect_timeout_s, read=300.0),
    )


def _httpx_flavour() -> Any:
    import mcp

    if hasattr(mcp, "Client"):  # mcp >= 2 ships its own HTTP stack
        try:
            import httpx2 as hx

            return hx
        except ImportError:
            pass
    import httpx as hx

    return hx


def resolve_headers(src: McpSource) -> dict[str, str]:
    out = {}
    for name, ref in src.headers.items():
        prefix, _, rest = ref.partition("${")
        secret = Secret.parse(
            ("env:" + rest.rstrip("}")) if rest else ref, field=f"mcp {src.name} header {name}"
        )
        out[name] = prefix + secret.value()
    return out


def http_client(src: McpSource) -> MCPClient:
    policy = _policy(src)
    headers = resolve_headers(src)  # a missing variable is a startup error
    url = src.url or ""

    async def connect(stack: AsyncExitStack) -> Any:
        from mcp.client import streamable_http as transport

        from aaa.mcp.client import _open_session

        hx = _httpx_flavour()
        if hasattr(transport, "streamable_http_client"):
            client = guarded_client(hx, policy, headers, src)
            stack.push_async_callback(client.aclose)
            streams = await stack.enter_async_context(
                transport.streamable_http_client(url, http_client=client)
            )
        else:  # early 1.x: the SDK builds the client, from a factory we supply

            def factory(
                headers: dict[str, str] | None = None, timeout: Any = None, auth: Any = None
            ) -> Any:
                return guarded_client(hx, policy, dict(headers or {}), src)

            streams = await stack.enter_async_context(
                transport.streamablehttp_client(
                    url, headers=headers or None, httpx_client_factory=factory
                )
            )
        return await _open_session(stack, streams[0], streams[1])

    return MCPClient(name=src.name, _connect=connect)


def stdio_client(src: McpSource) -> MCPClient:
    env = dict(src.env)
    for name in src.env_from:
        v = os.environ.get(name)
        if v is None:
            raise SpecError(
                f"mcp {src.name}: env_from names {name}, which is not set in this environment"
            )
        env[name] = v
    return MCPClient.stdio(src.command or "", src.args, env=env or None, cwd=src.cwd, name=src.name)


# -- one connection, owned by one task ---------------------------------------------------------------
class Upstream:
    def __init__(self, src: McpSource, make: Any) -> None:
        self.src, self._make = src, make
        self.client: MCPClient | None = None
        self.infos: list[MCPToolInfo] = []
        self.resource_infos: list[Any] = []
        self.prompt_infos: list[Any] = []
        self._ready: asyncio.Future[None] | None = None
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._task = asyncio.create_task(self._run(), name=f"mcp-upstream-{self.src.name}")
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), self.src.connect_timeout_s)
        except BaseException:
            await self.aclose()
            raise

    async def _run(self) -> None:
        assert self._ready is not None
        try:
            async with self._make() as client:
                self.client = client
                self.infos = await client.list_tools()
                if self.src.resources:
                    self.resource_infos = await paged(client.session.list_resources, "resources")
                if self.src.prompts:
                    self.prompt_infos = await paged(client.session.list_prompts, "prompts")
                self._ready.set_result(None)
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 - reported to start(), or logged if it happens later
            if not self._ready.done():
                self._ready.set_exception(
                    e if isinstance(e, Exception) else RuntimeError("cancelled")
                )
            else:
                log.warning("upstream %s ended: %s", self.src.name, type(e).__name__)
        finally:
            self.client = None

    async def aclose(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, 5.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            except Exception:
                log.debug("upstream %s close error", self.src.name, exc_info=True)


def _wrap(tool: Tool, upstream: Upstream, src: McpSource) -> Tool:
    """Make a call fail cleanly if the connection is gone, with a message that carries nothing from the upstream's internals."""
    inner = tool.fn  # the MCPClient caller: async (arguments) -> result

    async def call(arguments: dict[str, Any]) -> Any:
        if upstream.client is None:
            from aaa.tools.tool import ToolError

            raise ToolError(f"the upstream server {src.name!r} is not available")
        return await inner(arguments)

    tool.fn = call
    return tool


async def start_upstream(src: McpSource) -> Loaded:
    """Connect to the upstream and return its (filtered, capped) tools. The caller must `aclose()` `Loaded.upstream`."""
    if src.transport == "stdio":
        make: Any = lambda: stdio_client(src)
    else:
        make = lambda: http_client(src)
        _policy(src)  # fail on a bad URL / host before any connection
        resolve_headers(src)
    if src.transport == "stdio":
        stdio_client(src)  # validate env_from before spawning anything
    up = Upstream(src, make)
    try:
        await up.start()
    except SpecError:
        raise
    except Exception as e:  # noqa: BLE001
        if src.optional:
            return Loaded(
                notes=[
                    f"WARNING: upstream {src.name!r} is not available ({type(e).__name__}); its tools are skipped"
                ]
            )
        raise SpecError(
            f"mcp {src.name}: cannot connect to the upstream ({type(e).__name__}: {str(e)[:200]})"
        ) from None
    assert up.client is not None
    infos = [
        i
        for i in up.infos
        if any(fnmatch.fnmatchcase(i.name, p) for p in src.include)
        and not any(fnmatch.fnmatchcase(i.name, p) for p in src.exclude)
    ]
    tools = up.client.tools(
        prefix="",
        include=[i.name for i in infos],
        approval=src.approval,
        trust_annotations=src.trust_annotations,
        timeout_s=src.call_timeout_s,
        max_result_chars=src.max_result_chars,
        infos=infos,
    )
    out = Loaded(upstream=up)
    for t, info in zip(tools, infos, strict=True):
        t.description = (src.description_overrides.get(info.name) or t.description)[
            : src.description_max_chars
        ]
        out.tools.append(_wrap(t, up, src))
        if src.trust_annotations:
            out.annotations[t.name] = MCPAnnotations(
                read_only=info.read_only,
                destructive=info.destructive,
                idempotent=info.idempotent,
                open_world=info.open_world,
            )
    out.notes.append(f"upstream {src.name!r}: {len(tools)} of {len(up.infos)} tool(s) taken")
    if src.resources:
        out.resources = UpstreamResources(src, up, up.resource_infos)
        out.notes.append(
            f"upstream {src.name!r}: {len(out.resources.items)} resource(s) re-exposed"
        )
    if src.prompts:
        for info in up.prompt_infos:
            remote = str(_attr(info, "name"))
            if any(fnmatch.fnmatchcase(remote, p) for p in src.exclude) or not any(
                fnmatch.fnmatchcase(remote, p) for p in src.include
            ):
                continue
            exposed = (src.prefix + remote)[:64]
            out.prompts[exposed] = UpstreamPrompt(src, up, info, exposed)
        out.notes.append(f"upstream {src.name!r}: {len(out.prompts)} prompt(s) re-exposed")
    return out
