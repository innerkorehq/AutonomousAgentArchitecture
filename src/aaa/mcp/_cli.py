"""`aaa mcp serve|check|inspect`: the command line of the MCP server generator.

    aaa mcp serve --config server.yaml                    # a generated server over stdio
    aaa mcp serve --source myapp.tools:registry --exclude 'admin_*' --tenant acme
    aaa mcp serve myapp.tools:build                       # the older form: a factory returning a ToolRegistry/MCPToolServer
    aaa mcp check --config server.yaml                    # validate and compile; report what would be exposed
    aaa mcp inspect --config server.yaml                  # print exactly what a client (and its model) will be shown
    aaa mcp init my-server                                # interactive: write a project (config, server.py, Dockerfile ...)
    aaa mcp init my-server --yes --source openapi=api.json --allowed-host api.example.com
    aaa mcp export my-server --check                      # have the generated files drifted from their templates?

Flags override the environment (`AAA_MCP__*`), which overrides the file. stdout carries the protocol while serving, so every
message from `serve` goes to stderr."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from aaa.mcp.config import SpecError, load_spec


def add_arguments(mc: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="a .yaml, .yml, .json or .toml server configuration")
        p.add_argument(
            "--overlay",
            action="append",
            metavar="FILE",
            help="a configuration file layered on top of --config (repeatable; also AAA_MCP_OVERLAY)",
        )
        p.add_argument(
            "--source",
            action="append",
            metavar="MODULE:ATTR",
            help="tools from AAA objects (repeatable); replaces the file's sources",
        )
        p.add_argument(
            "--include",
            action="append",
            metavar="GLOB",
            help="expose only matching tool names (repeatable)",
        )
        p.add_argument(
            "--exclude",
            action="append",
            metavar="GLOB",
            help="never expose matching tool names (repeatable)",
        )
        p.add_argument(
            "--tag", action="append", help="expose only tools carrying this source tag (repeatable)"
        )
        p.add_argument("--prefix", help="prefix every exposed tool name")
        p.add_argument("--name", help="server name")
        p.add_argument("--title", help="server title")
        p.add_argument("--instructions", help="instructions shown to the client")
        p.add_argument("--tenant", help="tenant the calls are for")
        p.add_argument(
            "--profile",
            help="the profile this stdio server serves (see `profiles` in the configuration)",
        )
        p.add_argument("--user", help="user the calls are for")
        p.add_argument(
            "--expose-approval-tools",
            action="store_true",
            default=None,
            help="also offer tools that require approval (nothing is approved for you)",
        )
        p.add_argument("--max-result-chars", type=int)
        p.add_argument(
            "--no-import-sources", action="store_true", help="refuse any source that imports code"
        )
        p.add_argument(
            "--no-subprocess-sources",
            action="store_true",
            help="refuse any upstream MCP server that this one would launch",
        )

    def project(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", help="project name (lowercase letters, digits, '-', '_')")
        p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
        p.add_argument("--no-docker", action="store_true", help="do not write a Dockerfile")
        p.add_argument(
            "--format",
            choices=["toml", "json", "yaml"],
            default="toml",
            help="configuration file format",
        )
        p.add_argument(
            "--force",
            action="store_true",
            help="write into a non-empty directory, overwriting files",
        )

    ini = mc.add_parser("init", help="generate a project: interactively, or with --yes and flags")
    ini.add_argument("dir", nargs="?", help="where to write the project")
    ini.add_argument("--yes", action="store_true", help="do not ask questions; use the flags")
    ini.add_argument(
        "--source",
        action="append",
        metavar="TYPE=VALUE",
        help="objects=module:attr | module=pkg.mod[:fn,fn] | openapi=PATH_OR_URL (repeatable)",
    )
    ini.add_argument(
        "--allowed-host",
        action="append",
        help="API host(s) an openapi source may call (required with openapi)",
    )
    ini.add_argument(
        "--bearer-env", help="environment variable holding a bearer token for the openapi source"
    )
    ini.add_argument(
        "--vendor-spec",
        action="store_true",
        help="copy OpenAPI documents into the project (reproducible builds)",
    )
    project(ini)
    ex = mc.add_parser(
        "export", help="write a project from an existing configuration, or check/update one"
    )
    ex.add_argument("dir")
    ex.add_argument("--config", help="create a project at DIR from this configuration")
    ex.add_argument(
        "--check",
        action="store_true",
        help="report generated files that drifted from their templates (exit 1 if any)",
    )
    ex.add_argument(
        "--update", action="store_true", help="regenerate generated files you have not edited"
    )
    project(ex)

    sv = mc.add_parser("serve", help="serve tools over MCP stdio")
    sv.add_argument(
        "--reload",
        action="store_true",
        help="reload the configuration on SIGHUP (a failed or unsafe reload leaves the running server alone)",
    )
    sv.add_argument(
        "--watch",
        nargs="?",
        const=5.0,
        type=float,
        metavar="SECONDS",
        help="also reload when the configuration files change (checked every SECONDS, default 5); implies --reload",
    )
    sv.add_argument(
        "factory",
        nargs="?",
        help="older form: module:function returning a ToolRegistry or MCPToolServer",
    )
    common(sv)
    ck = mc.add_parser("check", help="validate a configuration and report what it would expose")
    common(ck)
    ck.add_argument(
        "--no-compile", action="store_true", help="only validate the file; do not import anything"
    )
    common(mc.add_parser("inspect", help="print what clients will be shown"))
    cc = mc.add_parser(
        "client-config", help="print the entry an MCP client needs to connect to this server"
    )
    common(cc)
    cc.add_argument("--client", choices=["claude-desktop", "cursor", "generic"], default="generic")
    cc.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    cc.add_argument("--url", help="where aaa-server serves /mcp (with --transport http)")
    cc.add_argument("--server-name", help="the name to register under (default: the server's name)")
    fl = mc.add_parser(
        "flatten",
        help="write the configuration with its includes and overlays composed into one file",
    )
    common(fl)
    fl.add_argument(
        "-o", "--output", help="write here (format from the extension); default: stdout as JSON"
    )


def overlay_of(a: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, Any] = {}

    def put(path: str, value: Any) -> None:
        if value is None:
            return
        node = o
        *parents, last = path.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value

    if a.source:
        put("sources", [{"type": "objects", "ref": r} for r in a.source])
    put("tools.include", a.include)
    put("tools.exclude", a.exclude)
    put("tools.tags", a.tag)
    put("tools.prefix", a.prefix)
    put("tenancy.profile", a.profile)
    put("tools.expose_approval_tools", True if a.expose_approval_tools else None)
    put("server.name", a.name)
    put("server.title", a.title)
    put("server.instructions", a.instructions)
    put("tenancy.tenant", a.tenant)
    put("tenancy.user", a.user)
    put("limits.max_result_chars", a.max_result_chars)
    return o


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


async def _legacy_serve(spec: str) -> int:
    import importlib
    import inspect

    from aaa.mcp.server import MCPToolServer, serve_stdio
    from aaa.tools import ToolRegistry

    module, _, attr = spec.partition(":")
    if not module or not attr:
        return _fail("expected module:factory, e.g. myapp.tools:build")
    sys.path.insert(0, "")
    obj = getattr(importlib.import_module(module), attr)
    obj = obj() if callable(obj) else obj
    if inspect.isawaitable(obj):
        obj = await obj
    if isinstance(obj, ToolRegistry):
        obj = MCPToolServer(obj)
    if not isinstance(obj, MCPToolServer):
        return _fail(f"{spec} must return a ToolRegistry or MCPToolServer")
    await serve_stdio(obj)
    return 0


async def _vendor(data: dict[str, Any]) -> dict[str, str]:
    """Replace each OpenAPI document with a copy inside the project (`openapi/api.json`); returns the files to write."""
    from aaa.mcp.sources.openapi import read_document
    from aaa.mcp.spec import OpenAPISource

    files: dict[str, str] = {}
    for i, src in enumerate(data["sources"]):
        if src["type"] != "openapi":
            continue
        doc = await read_document(OpenAPISource.model_validate(src))
        rel = f"openapi/{'api' if len(data['sources']) == 1 else f'api{i}'}.json"
        files[rel] = json.dumps(doc, indent=2) + "\n"
        src["document"] = rel
    return files


def _sources_from_flags(a: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in a.source or []:
        kind, _, value = item.partition("=")
        if kind == "objects":
            out.append({"type": "objects", "ref": value})
        elif kind == "module":
            mod, _, fns = value.partition(":")
            out.append({"type": "module", "module": mod, "allow": [f for f in fns.split(",") if f]})
        elif kind == "openapi":
            if not a.allowed_host:
                raise SpecError(
                    "an openapi source needs --allowed-host (the only host(s) the server may call)"
                )
            doc = (
                value
                if value.startswith("https://") or a.vendor_spec
                else str(Path(value).resolve())
            )
            src: dict[str, Any] = {
                "type": "openapi",
                "document": doc,
                "allowed_hosts": list(a.allowed_host),
            }
            if a.bearer_env:
                src["auth"] = {"scheme": "bearer", "secret": f"env:{a.bearer_env}"}
            out.append(src)
        else:
            raise SpecError(f"unknown --source type {kind!r}; use objects=, module= or openapi=")
    return out


async def _init(a: argparse.Namespace) -> int:
    from aaa.mcp import scaffold
    from aaa.mcp.config import load_spec

    if a.yes:
        if not a.dir:
            return _fail("--yes needs a directory")
        name = a.name or Path(a.dir).resolve().name.lower().replace(" ", "-")
        data: dict[str, Any] = {"server": {"name": name}, "sources": _sources_from_flags(a)}
        opts = scaffold.Options(name, a.transport, not a.no_docker, a.format).validate()
        dest = Path(a.dir)
    else:
        spec0, opts, dest = scaffold.prompt_spec(lambda q: input(q), print)
        if a.dir:
            dest = Path(a.dir)
        data = spec0.model_dump(mode="json", exclude_defaults=True)
    vendored = await _vendor(data) if a.vendor_spec else {}
    spec = load_spec(data, environ={})
    files = scaffold.render(spec, scaffold.config_text_for(spec, opts), opts) | vendored
    for rel in scaffold.write_project(dest, files, opts, force=a.force):
        print(f"wrote {dest / rel}")
    print(
        f"\nNext: cd {dest} && pip install -r requirements.txt && aaa mcp check --config {opts.config_name}"
    )
    return 0


async def _export(a: argparse.Namespace) -> int:
    from aaa.mcp import scaffold
    from aaa.mcp.config import load_spec

    dest = Path(a.dir)
    if a.check or a.update:
        if a.check:
            problems = scaffold.check_project(dest)
            for p in problems:
                print(f"DRIFT: {p}")
            print(
                "ok: the generated files match their templates"
                if not problems
                else f"{len(problems)} file(s) differ"
            )
            return 1 if problems else 0
        rewritten, skipped = scaffold.update_project(dest)
        for r in rewritten:
            print(f"rewrote {r}")
        for sk in skipped:
            print(
                f"kept {sk}: you edited it (compare with a fresh `aaa mcp init` to merge by hand)"
            )
        print("up to date" if not rewritten and not skipped else "done")
        return 0
    if not a.config:
        return _fail(
            "give --config FILE to create a project, or --check / --update for an existing one"
        )
    cfg_path = Path(a.config)
    spec = load_spec(cfg_path)  # validates it (and reports problems) before anything is written
    fmt = {".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml"}.get(
        cfg_path.suffix.lower()
    )
    if fmt is None:
        return _fail(f"{cfg_path}: unknown configuration format (use .toml, .json or .yaml)")
    name = a.name or dest.resolve().name.lower().replace(" ", "-")
    opts = scaffold.Options(name, a.transport, not a.no_docker, fmt).validate()
    # the configuration is copied as written (comments, relative paths and all): it is the user's file, not ours to re-dump
    files = scaffold.render(spec, cfg_path.read_text(), opts)
    for rel in scaffold.write_project(dest, files, opts, force=a.force):
        print(f"wrote {dest / rel}")
    return 0


def _flatten(spec: Any, output: str | None) -> int:
    """Write the composed configuration. With -o, file paths are made relative to the output's directory so the file can move with its data."""
    from aaa.mcp.config import dump_spec

    if output:
        base = Path(output).resolve().parent

        def rel(v: str | None) -> str | None:
            if v is None or v.startswith(("http://", "https://")):
                return v
            return os.path.relpath(v, base) if Path(v).is_absolute() else v

        for src in spec.sources:
            if src.type == "openapi":
                src.document = rel(src.document)
            elif src.type == "module":
                src.path = rel(src.path)
        for res in spec.resources:
            if res.type == "files":
                res.root = rel(res.root)
    text = dump_spec(spec, output)
    if not output:
        print(text, end="")
    return 0


async def run(a: argparse.Namespace) -> int:
    from aaa.mcp.compile import acompile_spec, describe
    from aaa.mcp.server import serve_stdio

    sys.path.insert(0, "")
    if a.sub in ("init", "export"):
        try:
            return await (_init(a) if a.sub == "init" else _export(a))
        except SpecError as e:
            return _fail(str(e))
    if a.sub == "serve" and a.factory and not a.config:
        return await _legacy_serve(a.factory)
    try:
        spec = load_spec(a.config, overlay=overlay_of(a), overlays=a.overlay or ())
        if a.sub == "client-config":
            from aaa.mcp import clientconfig

            if not a.config:
                raise SpecError("client-config needs --config")
            entry = clientconfig.build(
                spec,
                a.config,
                client=a.client,
                transport=a.transport,
                url=a.url,
                name=a.server_name,
                overlays=a.overlay or [],
            )
            print(json.dumps(entry, indent=2))
            return 0
        if a.sub == "flatten":
            return _flatten(spec, a.output)
        if a.sub == "check" and a.no_compile:
            print(
                f"ok: configuration is valid ({len(spec.sources)} source(s); nothing was imported)"
            )
            return 0
        core = await acompile_spec(
            spec,
            allow_import=False if a.no_import_sources else None,
            allow_subprocess=False if a.no_subprocess_sources else None,
        )
    except SpecError as e:
        return _fail(str(e))
    if a.sub == "inspect":
        print(json.dumps(describe(core), indent=2, ensure_ascii=False))
        await core.aclose()
        return 0
    if a.sub == "check":
        print("ok: configuration is valid")
        for line in core.report:
            print(f"  {line}")
        await core.aclose()
        return 0
    for line in core.report:  # stderr: stdout is the protocol
        print(f"aaa mcp: {line}", file=sys.stderr)
    if a.sub == "serve" and (a.reload or a.watch):
        return await _serve_reloadable(a, core)
    if core.owns_engine and core.engine is not None:
        async with (
            core.engine
        ):  # the engine this configuration built runs for as long as the server does
            await serve_stdio(core)
    else:
        await serve_stdio(core)
    return 0


async def _serve_reloadable(a: argparse.Namespace, first: Any) -> int:
    """Serve with hot reload: SIGHUP (and, with --watch, a change to the configuration files) rebuilds the server; see `aaa.mcp.reload` for what is refused."""
    import signal

    from aaa.mcp.compile import acompile_spec
    from aaa.mcp.include import compose, overlays_from_env
    from aaa.mcp.reload import ReloadableServer, ReloadResult, watch
    from aaa.mcp.server import serve_stdio

    def read() -> Any:
        return load_spec(a.config, overlay=overlay_of(a), overlays=a.overlay or ())

    async def compile_(spec: Any) -> Any:
        return await acompile_spec(
            spec,
            allow_import=False if a.no_import_sources else None,
            allow_subprocess=False if a.no_subprocess_sources else None,
        )

    def say(r: ReloadResult) -> None:
        print(f"aaa mcp: reload {r.status}: {r.reason}", file=sys.stderr)

    server = ReloadableServer(first, read, compile_)
    loop = asyncio.get_running_loop()

    def on_hup() -> None:
        async def go() -> None:
            say(await server.reload())

        loop.create_task(go())

    if hasattr(signal, "SIGHUP"):
        loop.add_signal_handler(signal.SIGHUP, on_hup)
    watcher = None
    if a.watch:
        watcher = asyncio.create_task(
            watch(
                server,
                lambda: (
                    compose(a.config, [*overlays_from_env(os.environ), *(a.overlay or [])]).files
                ),
                interval_s=a.watch,
                on_result=say,
            )
        )
    try:
        await serve_stdio(server)
    finally:
        if watcher is not None:
            watcher.cancel()
        await server.aclose()
    return 0


def main_run(a: argparse.Namespace) -> int:
    return asyncio.run(run(a))
