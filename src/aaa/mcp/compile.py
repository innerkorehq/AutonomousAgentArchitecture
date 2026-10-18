"""Compiling an `MCPSpec` into a running server core.

    spec = load_spec("server.yaml")
    core = await acompile_spec(spec)          # or compile_spec(spec) outside an event loop
    await serve_stdio(core)                   # or ServerConfig(mcp=core, ...) behind aaa-server

Importing code is the one dangerous thing a spec can ask for, so it has a kill-switch: `AAA_MCP_ALLOW_IMPORT=0` (or
`allow_import=False`) turns any source that imports a module into an error before anything is imported."""

from __future__ import annotations

import asyncio
import fnmatch
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from aaa.mcp.config import SpecError
from aaa.mcp.filters import Entry, rename, select, with_name
from aaa.mcp.model import MCPAnnotations, MCPToolMeta
from aaa.mcp.policy import Profile, ProfileSet, ToolRule
from aaa.mcp.server import ApprovalCallback, MCPToolServer
from aaa.mcp.sources.agents import AgentTool, check_spec
from aaa.mcp.sources.mcp_upstream import Upstream, start_upstream, subprocess_allowed
from aaa.mcp.sources.objects import load_ref, normalize, resolve_ref
from aaa.mcp.sources.openapi import load_openapi
from aaa.mcp.sources.prompts import PromptDef
from aaa.mcp.sources.pymodule import load_module
from aaa.mcp.sources.resources import (
    FilesProvider,
    IndexProvider,
    MemoryProvider,
    ResourceSet,
    StaticProvider,
)
from aaa.mcp.spec import (
    AnnotationsSpec,
    McpSource,
    MCPSpec,
    ModuleSource,
    ObjectsSource,
    ToolLimits,
    ToolOverride,
)
from aaa.tools.tool import ToolRegistry

if TYPE_CHECKING:
    from aaa.runtime.engine import Engine


def import_allowed() -> bool:
    return os.environ.get("AAA_MCP_ALLOW_IMPORT", "1").strip().lower() not in ("0", "false", "no")


class CompiledServer(MCPToolServer):
    """An `MCPToolServer` built from a spec, plus what was decided: the spec itself and a human-readable `report`."""

    spec: MCPSpec
    report: list[str]
    hidden: list[str]
    _upstreams: list[Upstream]
    owns_engine: bool = False  # the spec built this engine (runtime.engine): whoever serves it must enter and leave it

    async def aclose(self) -> None:
        """Close the connections to upstream MCP servers (and stop the processes this server launched)."""
        ups, self._upstreams = getattr(self, "_upstreams", []), []
        await asyncio.gather(*(u.aclose() for u in ups), return_exceptions=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


async def _object(ref: str, live: Mapping[str, Any] | None, where: str) -> Any:
    if ":" in ref:
        return await resolve_ref(ref)
    if live is not None and ref in live:
        return live[ref]
    raise SpecError(f"{where}: no live object {ref!r} (use module:attribute or pass live={{...}})")


def _rules(spec: MCPSpec, names: list[str]) -> dict[str, ToolRule]:
    """Per-tool rules: a tool's own override beats the first matching policy, setting by setting."""
    for pol in spec.tools.policies:
        if not any(fnmatch.fnmatchcase(n, g) for g in pol.match for n in names):
            raise SpecError(f"tools.policies: match {pol.match} selects no exposed tool")
    out: dict[str, ToolRule] = {}
    for n in names:
        layers: list[ToolLimits] = [
            *([spec.tools.overrides[n]] if n in spec.tools.overrides else []),
            *(p for p in spec.tools.policies if any(fnmatch.fnmatchcase(n, g) for g in p.match)),
        ]

        def first(attr: str, layers: list[ToolLimits] = layers) -> Any:
            return next((v for lim in layers if (v := getattr(lim, attr)) is not None), None)

        rl, scopes, tenants = first("rate_limit"), first("scopes"), first("tenants")
        rule = ToolRule(
            timeout_s=first("timeout_s"),
            rate_per_minute=rl.per_minute if rl else None,
            burst=rl.burst if rl else None,
            max_concurrency=first("max_concurrency"),
            scopes=frozenset(scopes) if scopes else None,
            tenants=frozenset(tenants) if tenants else None,
        )
        if rule != ToolRule():
            out[n] = rule
    return out


def _profiles(spec: MCPSpec, names: list[str]) -> ProfileSet | None:
    """Profiles as the server enforces them. A tool glob that selects nothing is an error (a typo would otherwise hide a tool from the audience that should see it)."""
    if not spec.profiles:
        return None
    for pname, prof in spec.profiles.items():
        for g in prof.tools:
            if not any(fnmatch.fnmatchcase(n, g) for n in names):
                raise SpecError(f"profiles.{pname}: tools glob {g!r} selects no exposed tool")
    return ProfileSet(
        {
            n: Profile(tuple(p.tools), tuple(p.resources), tuple(p.prompts))
            for n, p in spec.profiles.items()
        },
        default=spec.tenancy.profile,
        required=spec.tenancy.profiles_required,
    )


async def _resources(
    spec: MCPSpec, live: Mapping[str, Any] | None, extra: list[Any] | None = None
) -> ResourceSet | None:
    if not spec.resources and not extra:
        return None
    lim = spec.limits
    providers: list[Any] = [*(extra or [])]
    for r in spec.resources:
        if r.type == "static":
            providers.append(
                StaticProvider(
                    r.name,
                    r.text,
                    mime_type=r.mime_type,
                    title=r.title,
                    description=r.description,
                    limit=lim.max_resource_bytes,
                )
            )
        elif r.type == "files":
            if not Path(r.root).is_dir():
                raise SpecError(f"resources.{r.name}: root {r.root!r} is not a directory")
            providers.append(
                FilesProvider(
                    r.name,
                    r.root,
                    pattern=r.pattern,
                    shared=r.shared,
                    allow_hidden=r.allow_hidden,
                    mime_type=r.mime_type,
                    title=r.title,
                    description=r.description,
                    limit=lim.max_resource_bytes,
                    max_listed=lim.max_resources_listed,
                )
            )
        elif r.type == "index":
            obj = await _object(r.ref, live, f"resources.{r.name}")
            if not hasattr(obj, "search"):
                raise SpecError(
                    f"resources.{r.name}: {r.ref} is not an Index (it has no search method)"
                )
            providers.append(
                IndexProvider(
                    r.name,
                    obj,
                    k=r.k,
                    max_chars=r.max_chars,
                    tenant_key=None if r.shared else r.tenant_key,
                    title=r.title,
                    description=r.description,
                )
            )
        else:
            obj = await _object(r.ref, live, f"resources.{r.name}")
            if not hasattr(obj, "search"):
                raise SpecError(
                    f"resources.{r.name}: {r.ref} is not a memory backend (it has no search method)"
                )
            providers.append(
                MemoryProvider(
                    r.name,
                    obj,
                    k=r.k,
                    max_chars=r.max_chars,
                    title=r.title,
                    description=r.description,
                )
            )
    try:
        return ResourceSet(providers, max_listed=lim.max_resources_listed)
    except ValueError as e:
        raise SpecError(f"resources: {e}; give each a distinct name within its type") from None


def _annotations(a: AnnotationsSpec | None) -> MCPAnnotations | None:
    if a is None:
        return None
    return MCPAnnotations(a.title, a.read_only, a.destructive, a.idempotent, a.open_world)


def _override(entry: Entry, o: ToolOverride) -> Entry:
    import copy

    tool = copy.copy(entry.tool)
    if o.description is not None:
        tool.description = o.description
    if o.requires_approval is not None:
        tool.requires_approval = o.requires_approval
    return Entry(tool, entry.origin, entry.tags, entry.annotations)


async def acompile_spec(
    spec: MCPSpec,
    *,
    live: Mapping[str, Any] | None = None,
    engine: Engine | None = None,
    approval: ApprovalCallback | None = None,
    allow_import: bool | None = None,
    allow_subprocess: bool | None = None,
) -> CompiledServer:
    """Build the server. If the spec has `mcp` sources the returned server holds live connections: close them with `await core.aclose()`
    (or `async with core:`). If compilation fails, anything already connected is closed before the error propagates."""
    holders: list[Upstream] = []
    try:
        core = await _compile(
            spec,
            holders,
            live=live,
            engine=engine,
            approval=approval,
            allow_import=allow_import,
            allow_subprocess=allow_subprocess,
        )
    except BaseException:
        await asyncio.gather(*(h.aclose() for h in holders), return_exceptions=True)
        raise
    core._upstreams = holders
    return core


async def _compile(
    spec: MCPSpec,
    holders: list[Upstream],
    *,
    live: Mapping[str, Any] | None,
    engine: Engine | None,
    approval: ApprovalCallback | None,
    allow_import: bool | None,
    allow_subprocess: bool | None,
) -> CompiledServer:
    if spec.launches_processes() and not (
        subprocess_allowed() if allow_subprocess is None else allow_subprocess
    ):
        names = [
            s.name for s in spec.sources if isinstance(s, McpSource) and s.transport == "stdio"
        ]
        raise SpecError(
            f"upstream(s) {names} launch processes, but launching is disabled (AAA_MCP_ALLOW_SUBPROCESS=0 / --no-subprocess-sources)"
        )
    allowed = import_allowed() if allow_import is None else allow_import
    if spec.imports_code() and not allowed:
        refs = [
            s.ref if isinstance(s, ObjectsSource) else s.module
            for s in spec.sources
            if isinstance(s, ObjectsSource | ModuleSource)
        ]
        raise SpecError(
            f"sources {refs} import code, but importing is disabled (AAA_MCP_ALLOW_IMPORT=0 / --no-import-sources)"
        )
    owns_engine = False
    if engine is None and spec.runtime.engine is not None:
        engine = await resolve_ref(spec.runtime.engine)
        from aaa.runtime.engine import Engine as _Engine

        if not isinstance(engine, _Engine):
            raise SpecError(
                f"runtime.engine: {spec.runtime.engine} is not an Engine (got {type(engine).__name__})"
            )
        owns_engine = True
    notes: list[str] = []
    entries: list[Entry] = []
    if spec.agents:
        if engine is None:
            raise SpecError(
                "agents need an engine to run on: set runtime.engine = 'module:attribute', or pass engine= to compile_spec "
                "(behind aaa-server, the server's engine)"
            )
        gate = asyncio.Semaphore(spec.limits.max_concurrent_agents)
        for ag in spec.agents:
            check_spec(ag)
            if ":" in ag.algorithm:
                algo = await resolve_ref(ag.algorithm)
            elif live is not None and ag.algorithm in live:
                algo = live[ag.algorithm]
            else:
                raise SpecError(
                    f"agents.{ag.name}: no algorithm {ag.algorithm!r} (use module:attribute or pass live={{...}})"
                )
            if not hasattr(algo, "on_event") or not hasattr(algo, "subscriptions"):
                raise SpecError(
                    f"agents.{ag.name}: {ag.algorithm!r} is not an algorithm (got {type(algo).__name__})"
                )
            entries.append(
                Entry(
                    AgentTool(ag, algo, engine, gate),
                    f"agent {ag.name}",
                    frozenset(ag.tags),
                    _annotations(ag.annotations),
                )
            )
    upstream_providers: list[Any] = []
    upstream_prompts: dict[str, Any] = {}
    for src in spec.sources:
        if src.type == "mcp":
            up = await start_upstream(src)
            notes += up.notes
            if up.upstream is not None:
                holders.append(up.upstream)
            if up.resources is not None:
                upstream_providers.append(up.resources)
            for pname, pdef in up.prompts.items():
                if pname in upstream_prompts:
                    raise SpecError(
                        f"prompts: {pname!r} is offered by two upstream sources; give one a prefix"
                    )
                upstream_prompts[pname] = pdef
            for t in up.tools:
                entries.append(
                    Entry(
                        with_name(t, src.prefix + t.name),
                        f"mcp {src.name}",
                        frozenset(src.tags),
                        up.annotations.get(t.name),
                    )
                )
            continue
        if src.type == "module":
            loaded = load_module(src)
            notes += loaded.notes
            for t in loaded.tools:
                entries.append(
                    Entry(
                        with_name(t, src.prefix + t.name),
                        f"module {src.module}",
                        frozenset(src.tags),
                        loaded.annotations.get(t.name),
                    )
                )
            continue
        if src.type == "openapi":
            api = await load_openapi(src)
            notes += api.notes
            for t in api.tools:
                entries.append(
                    Entry(
                        with_name(t, src.prefix + t.name),
                        f"openapi {src.document}",
                        frozenset(src.tags),
                        api.annotations.get(t.name),
                    )
                )
            continue
        if src.type == "objects":
            tools, origin = await load_ref(src.ref), f"objects {src.ref}"
        else:
            if live is None or src.key not in live:
                raise SpecError(
                    f"sources: no live object was given for key {src.key!r} (compile_spec(..., live={{...}}))"
                )
            tools, origin = normalize(live[src.key], f"live {src.key}"), f"live {src.key}"
        for t in tools:
            entries.append(Entry(with_name(t, src.prefix + t.name), origin, frozenset(src.tags)))

    chosen = rename(select(entries, spec.tools), spec.tools)
    unknown = set(spec.tools.overrides) - {e.tool.name for e in chosen}
    if unknown:
        raise SpecError(
            f"tools.overrides: no exposed tool is named {sorted(unknown)} (overrides use the final exposed name)"
        )
    meta: dict[str, MCPToolMeta] = {
        e.tool.name: MCPToolMeta(None, e.annotations, None)
        for e in chosen
        if e.annotations is not None
    }
    final: list[Entry] = []
    for e in chosen:
        o = spec.tools.overrides.get(e.tool.name)
        if o is not None:
            e = _override(e, o)
            prior = meta.get(e.tool.name)
            meta[e.tool.name] = MCPToolMeta(
                o.title,
                _annotations(o.annotations) or (prior.annotations if prior else None),
                o.output_schema,
            )
        final.append(e)
    if len(final) > spec.limits.max_tools:
        raise SpecError(
            f"{len(final)} tools selected but limits.max_tools is {spec.limits.max_tools}: "
            "narrow tools.include/exclude/tags or raise the limit deliberately"
        )
    registry = ToolRegistry([e.tool for e in final])
    rules = _rules(spec, [e.tool.name for e in final])
    profiles = _profiles(spec, [e.tool.name for e in final])
    resources = await _resources(spec, live, upstream_providers)
    prompts: dict[str, Any] = {p.name: PromptDef(p) for p in spec.prompts}
    if len(prompts) != len(spec.prompts):
        raise SpecError("prompts: two prompts share a name")
    if clash := sorted(set(prompts) & set(upstream_prompts)):
        raise SpecError(
            f"prompts: {clash} are defined here and offered by an upstream source; rename one"
        )
    prompts |= upstream_prompts
    core = CompiledServer(
        registry,
        name=spec.server.name,
        title=spec.server.title,
        version=spec.server.version,
        instructions=spec.server.instructions,
        engine=engine,
        approval=approval,
        expose_approval_tools=spec.tools.expose_approval_tools and approval is None,
        tenant=spec.tenancy.tenant,
        user=spec.tenancy.user,
        max_result_chars=spec.limits.max_result_chars,
        tool_meta=meta,
        structured_output=spec.limits.structured_output,
        resources=resources,
        prompts=prompts,
        tool_policy=rules,
        default_scopes=spec.tenancy.scopes,
        profiles=profiles,
    )
    core.spec = spec
    core.owns_engine = owns_engine
    offered = {t.name for t in core.list_tools()}
    core.hidden = [e.tool.name for e in final if not core._exposed(e.tool)]
    core.report = [
        f"{len(offered)} tool(s) exposed from {len(spec.sources)} source(s)",
        *(f"note: {n}" for n in notes),
    ]
    if resources:
        core.report.append(
            f"{len(resources.providers)} resource provider(s): "
            + ", ".join(f"{k}/{n}" for k, n in resources.providers)
        )
    if prompts:
        core.report.append(f"{len(prompts)} prompt(s): {', '.join(prompts)}")
    if core.hidden:
        core.report.append(
            f"{len(core.hidden)} tool(s) hidden because they require approval "
            f"(tools.expose_approval_tools, or an override with requires_approval: false): {', '.join(core.hidden)}"
        )
    if not offered:
        core.report.append("WARNING: the server exposes no tools")
    return core


def compile_spec(spec: MCPSpec, **kw: Any) -> CompiledServer:
    """`acompile_spec` for code that is not already in an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(acompile_spec(spec, **kw))
    raise RuntimeError(
        "compile_spec() was called inside a running event loop: use `await acompile_spec(...)`"
    )


def describe(core: CompiledServer) -> dict[str, Any]:
    """Exactly what a client (and so a model) would be shown, for review: tool names, titles, descriptions and hints."""
    from aaa.mcp import wire

    return {
        "server": wire.server_info(core)
        | ({"instructions": core.instructions} if core.instructions else {}),
        "tools": wire.tools_list_result(core)["tools"],
        "resources": wire.resources_list_result(core.list_resources())["resources"],
        "resource_templates": wire.resource_templates_list_result(core.list_resource_templates())[
            "resourceTemplates"
        ],
        "prompts": wire.prompts_list_result(core.list_prompts())["prompts"],
        "hidden": core.hidden,
        "profiles": {
            n: {"tools": list(p.tools), "resources": list(p.resources), "prompts": list(p.prompts)}
            for n, p in (core.profiles.profiles.items() if core.profiles else [])
        },
        "policies": {
            n: {
                k: (sorted(v) if isinstance(v, frozenset) else v)
                for k, v in vars(r).items()
                if v is not None
            }
            for n, r in sorted(core.tool_policy.items())
        },
        "report": core.report,
    }


__all__ = ["CompiledServer", "acompile_spec", "compile_spec", "describe", "import_allowed"]
