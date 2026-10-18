# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
# (the `mcp` SDK is an optional dependency with two incompatible major versions)
"""Expose AAA tools (and, optionally, whole algorithms) as an MCP server.

    server = MCPToolServer(registry)                         # tools only
    server = MCPToolServer(registry, engine=engine, algorithms={"react": ReAct(registry)})
    await serve_stdio(server)                                # or: aaa mcp serve mymodule:build

The defaults assume the caller is not the person who wrote the tools:

* **Tools that need human approval are not listed and cannot be called** (a call is answered exactly like a call to a
  tool that does not exist). MCP has no approval channel, so the only ways to expose one are an explicit
  `approval=` callback that makes the decision, or `expose_approval_tools=True`, which says in its name that you accept the risk.
  Nothing is ever approved automatically.
* **Arguments are validated by the tool's own schema** before it runs; failures and exceptions become `isError` results with a
  short message (a tool's `ToolError` text is shown, any other exception is logged and reported generically).
* **The tenant and user are fixed by whoever builds the server** (or, behind `aaa-server`, by the authenticated principal);
  nothing a client sends can change them.
* **`run_agent` (algorithms) is off unless you pass `algorithms=`**, always runs under a `Budget` with a deadline, and returns
  the outcome only.

`MCPToolServer` is the SDK-free core (testable on its own); `build_server` and `serve_stdio` bind it to the official `mcp` SDK
(1.x or 2.x)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import jsonschema

from aaa.mcp.model import (
    MCPAnnotations,
    MCPCallResult,
    MCPPrompt,
    MCPPromptResult,
    MCPResource,
    MCPResourceContents,
    MCPResourceTemplate,
    MCPToolDef,
    MCPToolMeta,
)
from aaa.mcp.policy import Gates, Limiter, ProfileSet, ToolRule
from aaa.mcp.sources.prompts import PromptDef, PromptError
from aaa.mcp.sources.resources import ResourceNotFound, ResourceSet
from aaa.runtime.budget import Budget
from aaa.tools.tool import Tool, ToolArgsError, ToolContext, ToolError, ToolRegistry

if TYPE_CHECKING:
    from aaa.runtime.engine import Algorithm, Engine

log = logging.getLogger("aaa.mcp.server")

RUN_AGENT = "run_agent"
ApprovalCallback = Callable[[str, dict[str, Any], str | None], Awaitable[bool]]
"""`(tool name, arguments, tenant) -> allowed`. Called for every call to a tool that requires approval."""


class MCPToolServer:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        name: str = "aaa",
        engine: Engine | None = None,
        algorithms: Mapping[str, Algorithm] | None = None,
        approval: ApprovalCallback | None = None,
        expose_approval_tools: bool = False,
        tenant: str = "mcp",
        user: str | None = None,
        agent_budget: Budget | None = None,
        max_result_chars: int = 100_000,
        title: str | None = None,
        version: str = "0.1",
        instructions: str | None = None,
        tool_meta: Mapping[str, MCPToolMeta] | None = None,
        structured_output: bool = False,
        resources: ResourceSet | None = None,
        prompts: Mapping[str, PromptDef] | None = None,
        tool_policy: Mapping[str, ToolRule] | None = None,
        default_scopes: Iterable[str] | None = None,
        clock: Callable[[], float] | None = None,
        profiles: ProfileSet | None = None,
    ) -> None:
        if algorithms and engine is None:
            raise ValueError("algorithms= needs engine=")
        if expose_approval_tools and approval is not None:
            raise ValueError("pass either approval= or expose_approval_tools=True, not both")
        self.registry, self.name, self.engine = registry, name, engine
        self.algorithms = dict(algorithms or {})
        self.approval, self.expose_approval_tools = approval, expose_approval_tools
        self.tenant, self.user, self.max_result_chars = tenant, user, max_result_chars
        self.title, self.version, self.instructions = title, version, instructions
        self.tool_meta = dict(tool_meta or {})
        self.structured_output = (
            structured_output  # send structuredContent for tools with an output schema
        )
        self.resources = resources
        self.prompts = dict(prompts or {})
        self.tool_policy = dict(tool_policy or {})
        self.profiles = (
            profiles if profiles else None
        )  # who may see what, by audience (None: nobody is restricted)
        self.default_scopes = None if default_scopes is None else frozenset(default_scopes)
        self._limiter, self._gates = Limiter(clock), Gates()
        self.agent_budget = agent_budget or Budget(
            max_steps=50, max_tokens=200_000, deadline_s=120.0
        )

    # -- resources and prompts -------------------------------------------------------------
    @property
    def has_resources(self) -> bool:
        return bool(self.resources)

    @property
    def has_prompts(self) -> bool:
        return bool(self.prompts)

    def _profile_ok(self, kind: str, scopes: Iterable[str] | None, *candidates: str) -> bool:
        """Whether the caller's profile (from the authenticated `scopes`, or the server's default for an unrestricted caller) allows this."""
        if self.profiles is None:
            return True
        return self.profiles.allows(
            kind, None if scopes is None else frozenset(scopes), *candidates
        )

    def list_resources(
        self, tenant: str | None = None, scopes: Iterable[str] | None = None
    ) -> list[MCPResource]:
        if not self.resources:
            return []
        return [
            r
            for r in self.resources.list(tenant or self.tenant)
            if self._profile_ok("resources", scopes, r.uri, r.name)
        ]

    def list_resource_templates(
        self, scopes: Iterable[str] | None = None
    ) -> list[MCPResourceTemplate]:
        if not self.resources:
            return []
        return [
            t
            for t in self.resources.templates()
            if self._profile_ok("resources", scopes, t.uri_template, t.name)
        ]

    async def read_resource(
        self,
        uri: str,
        *,
        tenant: str | None = None,
        user: str | None = None,
        scopes: Iterable[str] | None = None,
    ) -> MCPResourceContents:
        """Read one resource as the host's tenant/user. Raises `ResourceNotFound` for anything that cannot be served
        (including what the caller's profile does not allow: that is indistinguishable from a missing one)."""
        if not self.resources:
            raise ResourceNotFound
        tenant = tenant or self.tenant
        if self.profiles is not None:
            names = [r.name for r in self.resources.list(tenant) if r.uri == uri]
            if not self._profile_ok("resources", scopes, uri, *names):
                raise ResourceNotFound
        return await self.resources.read(uri, tenant, user or self.user)

    def list_prompts(self, scopes: Iterable[str] | None = None) -> list[MCPPrompt]:
        return [
            p.describe() for n, p in self.prompts.items() if self._profile_ok("prompts", scopes, n)
        ]

    def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        scopes: Iterable[str] | None = None,
    ) -> MCPPromptResult:
        """Fill a prompt. Raises `PromptError` for an unknown prompt or arguments that do not fit."""
        p = self.prompts.get(name)
        if p is None or not self._profile_ok("prompts", scopes, name):
            raise PromptError(f"unknown prompt {name!r}")
        return p.get(arguments)

    async def aget_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        scopes: Iterable[str] | None = None,
    ) -> MCPPromptResult:
        """Like `get_prompt`, for hosts that can wait: also fills prompts that live on another server (see `McpSource.prompts`)."""
        p = self.prompts.get(name)
        if p is None or not self._profile_ok("prompts", scopes, name):
            raise PromptError(f"unknown prompt {name!r}")
        fetch = getattr(p, "aget", None)
        return await fetch(arguments) if fetch is not None else p.get(arguments)

    # -- what is offered -------------------------------------------------------------------
    def _exposed(self, t: Tool) -> bool:
        return (not t.requires_approval) or self.approval is not None or self.expose_approval_tools

    def _visible(self, name: str, tenant: str, scopes: frozenset[str] | None) -> bool:
        rule = self.tool_policy.get(name)
        return rule is None or rule.visible_to(tenant, scopes)

    def _scopes(self, scopes: Iterable[str] | None) -> frozenset[str] | None:
        return self.default_scopes if scopes is None else frozenset(scopes)

    def list_tools(
        self, *, tenant: str | None = None, scopes: Iterable[str] | None = None
    ) -> list[MCPToolDef]:
        """The tools this caller may see. `tenant`/`scopes` come from a host that authenticated its caller; `scopes=None` means the
        server's `default_scopes` (None there too: unrestricted)."""
        tenant, held = tenant or self.tenant, self._scopes(scopes)
        out = [
            self._tool_def(t)
            for n in self.registry.names()
            if (t := self.registry.get(n)) is not None
            and self._exposed(t)
            and self._visible(n, tenant, held)
            and self._profile_ok("tools", scopes, n)
        ]
        if self.algorithms and self._profile_ok("tools", scopes, RUN_AGENT):
            out.append(
                MCPToolDef(
                    RUN_AGENT,
                    "Run an agent algorithm to completion and return its outcome.",
                    {
                        "type": "object",
                        "properties": {
                            "algo": {"type": "string", "enum": sorted(self.algorithms)},
                            "input": {"type": "object"},
                        },
                        "required": ["algo"],
                        "additionalProperties": False,
                    },
                )
            )
        return out

    def _output_schema(self, t: Tool) -> dict[str, Any] | None:
        meta = self.tool_meta.get(t.name)
        if meta is not None and meta.output_schema is not None:
            return meta.output_schema
        return t.output_schema if self.structured_output else None

    def _tool_def(self, t: Tool) -> MCPToolDef:
        meta = self.tool_meta.get(t.name) or MCPToolMeta()
        ann = meta.annotations or _default_annotations(t)
        return MCPToolDef(
            t.name,
            t.description,
            t.parameters,
            read_only=bool(ann and ann.read_only),
            title=meta.title or (ann.title if ann else None),
            annotations=ann,
            output_schema=self._output_schema(t),
        )

    # -- calling ---------------------------------------------------------------------------
    def _render(self, value: Any) -> str:
        text = (
            value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        )
        if len(text) > self.max_result_chars:
            text = (
                text[: self.max_result_chars]
                + f"\n[truncated at {self.max_result_chars} characters]"
            )
        return text

    async def call(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        tenant: str | None = None,
        user: str | None = None,
        scopes: Iterable[str] | None = None,
    ) -> MCPCallResult:
        """Run one tool. `tenant`/`user` are for a host that has authenticated its caller (aaa-server); the
        MCP client never supplies them."""
        args = dict(arguments or {})
        tenant, user = tenant or self.tenant, user or self.user
        if name == RUN_AGENT and self.algorithms:
            if not self._profile_ok("tools", scopes, name):
                return MCPCallResult(f"unknown tool {name!r}", True)
            return await self._run_agent(args, tenant, user)
        t = self.registry.get(name)
        if (
            t is None
            or not self._exposed(t)
            or not self._visible(name, tenant, self._scopes(scopes))
            or not self._profile_ok("tools", scopes, name)
        ):
            return MCPCallResult(f"unknown tool {name!r}", True)
        rule = self.tool_policy.get(name)
        if (
            rule is not None
            and rule.rate_per_minute is not None
            and not self._limiter.allow(
                f"{tenant}/{user or ''}/{name}", rule.rate_per_minute, rule.burst
            )
        ):
            return MCPCallResult(f"{name} is being called too often; retry shortly", True)
        if t.requires_approval and self.approval is not None:
            try:
                allowed = await self.approval(name, args, tenant)
            except Exception:
                log.exception("approval callback failed for %s", name)
                allowed = False
            if not allowed:
                return MCPCallResult(f"{name} was not approved", True)
        ctx = ToolContext(
            engine=self.engine,
            run_id=f"mcp:{uuid.uuid4().hex}",
            call_id=uuid.uuid4().hex,
            tenant=tenant,
            meta={"user": user} if user else {},
        )
        limit = rule.max_concurrency if rule is not None else None
        if limit is not None and not self._gates.enter(name, limit):
            return MCPCallResult(f"{name} is busy; retry shortly", True)
        timeout = rule.timeout_s if rule is not None and rule.timeout_s else t.timeout_s
        try:
            async with asyncio.timeout(timeout):
                value = await t.invoke(args, ctx)
            return self._result(t, value)
        except TimeoutError:
            return MCPCallResult(f"{name} timed out after {timeout:g}s", True)
        except ToolArgsError as e:
            return MCPCallResult(str(e), True)
        except ToolError as e:
            return MCPCallResult(str(e) or f"{name} failed", True)
        except Exception:
            log.exception("tool %s failed", name)
            return MCPCallResult(f"{name} failed", True)
        finally:
            if limit is not None:
                self._gates.leave(name)

    def _result(self, t: Tool, value: Any) -> MCPCallResult:
        schema = self._output_schema(t)
        structured = None
        if schema is not None and isinstance(value, dict):
            problem = _violates(schema, value)
            if problem:
                log.error(
                    "tool %s returned data that does not match its output schema: %s",
                    t.name,
                    problem,
                )
                return MCPCallResult(
                    f"{t.name} returned data that does not match its output schema", True
                )
            structured = value
        text = self._render(value)
        if (
            structured is not None
            and len(json.dumps(structured, default=str)) > self.max_result_chars
        ):
            structured = None  # never send a structure that was cut short: text only, with its truncation marker
        return MCPCallResult(text, False, structured)

    async def _run_agent(
        self, args: dict[str, Any], tenant: str, user: str | None
    ) -> MCPCallResult:
        assert self.engine is not None
        extra = set(args) - {"algo", "input"}
        algo = self.algorithms.get(str(args.get("algo")))
        inp = args.get("input", {})
        if extra or algo is None or not isinstance(inp, dict):
            return MCPCallResult("run_agent needs {algo: <one of the offered>, input: {...}}", True)
        try:
            out = await self.engine.tenant(tenant).run(
                algo, inp, budget=self.agent_budget, user=user
            )
        except Exception:
            log.exception("run_agent failed")
            return MCPCallResult("the agent run failed", True)
        body = {"kind": out.kind.value, "value": out.value, "reason": out.reason}
        return MCPCallResult(self._render(body), not out.ok)


def _default_annotations(t: Tool) -> MCPAnnotations | None:
    """Hints derivable from what the tool already says: approval-gated means potentially destructive, and a tool whose delivery
    is `at_least_once` is safe to repeat (idempotent). Nothing is claimed about read-only-ness: only a config can say so."""
    destructive = True if t.requires_approval else None
    idempotent = True if t.delivery == "at_least_once" else None
    if destructive is None and idempotent is None:
        return None
    return MCPAnnotations(destructive=destructive, idempotent=idempotent)


def _violates(schema: dict[str, Any], value: Any) -> str | None:
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as e:
        return e.message[:200]
    except jsonschema.SchemaError as e:
        return f"invalid output schema: {e.message[:120]}"
    return None


# -- binding to the official SDK ---------------------------------------------------------------
def build_server(core: MCPToolServer) -> Any:
    """An SDK server object (low-level `Server`) exposing `core`. Works with mcp 1.x and 2.x.

    The SDK's own types are built from the shared wire dictionaries (`aaa.mcp.wire`), so what a tool looks like is decided in one
    place for stdio, the official client and the HTTP handler alike."""
    import mcp

    from aaa.mcp import wire

    if hasattr(
        mcp, "Client"
    ):  # mcp >= 2 (not "can mcp_types be imported": it can outlive a downgrade to 1.x)
        import mcp_types as types
        from mcp.server.lowlevel import Server

        async def on_list(ctx: Any, params: Any) -> Any:
            return types.ListToolsResult.model_validate(wire.tools_list_result(core))

        async def on_call(ctx: Any, params: Any) -> Any:
            r = await core.call(params.name, params.arguments)
            return types.CallToolResult.model_validate(wire.call_result_to_wire(r))

        extra: dict[str, Any] = {}
        if core.has_resources:

            async def on_list_resources(ctx: Any, params: Any) -> Any:
                return types.ListResourcesResult.model_validate(
                    wire.resources_list_result(core.list_resources())
                )

            async def on_list_templates(ctx: Any, params: Any) -> Any:
                return types.ListResourceTemplatesResult.model_validate(
                    wire.resource_templates_list_result(core.list_resource_templates())
                )

            async def on_read(ctx: Any, params: Any) -> Any:
                try:
                    c = await core.read_resource(str(params.uri))
                except ResourceNotFound:
                    raise _protocol_error(-32002, "resource not found") from None
                return types.ReadResourceResult.model_validate(wire.resource_read_result(c))

            extra |= {
                "on_list_resources": on_list_resources,
                "on_list_resource_templates": on_list_templates,
                "on_read_resource": on_read,
            }
        if core.has_prompts:

            async def on_list_prompts(ctx: Any, params: Any) -> Any:
                return types.ListPromptsResult.model_validate(
                    wire.prompts_list_result(core.list_prompts())
                )

            async def on_get_prompt(ctx: Any, params: Any) -> Any:
                try:
                    r = await core.aget_prompt(params.name, params.arguments)
                except PromptError as e:
                    raise _protocol_error(-32602, str(e)) from None
                return types.GetPromptResult.model_validate(wire.prompt_get_result(r))

            extra |= {"on_list_prompts": on_list_prompts, "on_get_prompt": on_get_prompt}

        return Server(
            core.name,
            version=core.version,
            title=core.title,
            instructions=core.instructions,
            on_list_tools=on_list,
            on_call_tool=on_call,
            **extra,
        )
    from mcp import types  # mcp 1.x
    from mcp.server.lowlevel import Server

    server = Server(core.name, version=core.version, instructions=core.instructions)

    @server.list_tools()
    async def _list() -> list[Any]:
        return [types.Tool.model_validate(w) for w in wire.tools_list_result(core)["tools"]]

    @server.call_tool(validate_input=False)
    async def _call(name: str, arguments: dict[str, Any] | None) -> Any:
        r = await core.call(name, arguments)
        if r.is_error:
            raise ToolError(
                r.text
            )  # the SDK turns an exception into an isError result carrying this text
        content = [types.TextContent(type="text", text=r.text)]
        return (content, r.structured) if r.structured is not None else content

    if core.has_resources:

        @server.list_resources()
        async def _list_resources() -> list[Any]:
            return [
                types.Resource.model_validate(r)
                for r in wire.resources_list_result(core.list_resources())["resources"]
            ]

        @server.list_resource_templates()
        async def _list_templates() -> list[Any]:
            return [
                types.ResourceTemplate.model_validate(t)
                for t in wire.resource_templates_list_result(core.list_resource_templates())[
                    "resourceTemplates"
                ]
            ]

        @server.read_resource()
        async def _read(uri: Any) -> Any:
            from mcp.server.lowlevel.helper_types import ReadResourceContents

            try:
                c = await core.read_resource(str(uri))
            except ResourceNotFound:
                raise ValueError("resource not found") from None
            return [ReadResourceContents(content=c.text, mime_type=c.mime_type)]

    if core.has_prompts:

        @server.list_prompts()
        async def _list_prompts() -> list[Any]:
            return [
                types.Prompt.model_validate(p)
                for p in wire.prompts_list_result(core.list_prompts())["prompts"]
            ]

        @server.get_prompt()
        async def _get_prompt(name: str, arguments: dict[str, str] | None) -> Any:
            try:
                r = await core.aget_prompt(name, arguments)
            except PromptError as e:
                raise ValueError(str(e)) from None
            return types.GetPromptResult.model_validate(wire.prompt_get_result(r))

    return server


def _protocol_error(code: int, message: str) -> Exception:
    """An exception the 2.x SDK turns into a JSON-RPC error with this code."""
    from mcp.shared.exceptions import MCPError

    return MCPError(code, message)


async def serve_stdio(core: MCPToolServer) -> None:
    """Serve over stdio until the client disconnects (stdout carries the protocol: never print to it)."""
    from mcp.server.stdio import stdio_server

    server = build_server(core)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


__all__ = [
    "RUN_AGENT",
    "ApprovalCallback",
    "MCPAnnotations",
    "MCPCallResult",
    "MCPToolDef",
    "MCPToolMeta",
    "MCPToolServer",
    "build_server",
    "serve_stdio",
]
