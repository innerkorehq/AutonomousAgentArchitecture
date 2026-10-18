"""Reload a running MCP server's configuration without restarting it.

    server = await ReloadableServer.start(load, compile)  # load() reads the spec from the files as they are now; compile(spec) builds the server
    ...
    await server.reload()                                 # or on SIGHUP / when a file changes: `aaa mcp serve --watch`

`ReloadableServer` is an `MCPToolServer` that forwards every call to the server compiled last. A reload builds the **whole new server first** and swaps it in only if that
worked, so a bad edit never takes down a good server. What it will and will not do:

* **Nothing new runs.** A reload may reconfigure (tools, policies, profiles, prompts, resources, OpenAPI documents, upstream connections) but the new configuration may only
  refer to code that was already running at startup (`MCPSpec.code_refs()`): no new module import, algorithm, engine or launched command. This is checked on the
  configuration *before* it is compiled, because compiling is what imports; a refused reload has run nothing. Code that changed on disk is not reloaded either
  (Python keeps the module it already imported): restart for that.
* **Identity is fixed.** The server's name, title, version, instructions and its set of capabilities (resources? prompts?) are decided at startup; a reload that would
  change them is refused with "restart required". A configuration that builds its own engine (`runtime.engine`) cannot be hot-reloaded.
* **Same safeguards as startup**: the kill-switches and the permission checks run again on every reload (they live in `build`).
* **In-flight calls finish on the server they started on.** The old server's upstream connections close after its calls are done, or after `grace_s`.
* **Rate-limited** (`min_interval_s`), one reload at a time, and a failure is logged with a generic reason and leaves the running server alone.

Clients that cached the tool list see the new one the next time they list it (the MCP list-changed notification is not sent)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from aaa.mcp.compile import CompiledServer
from aaa.mcp.model import (
    MCPCallResult,
    MCPPrompt,
    MCPPromptResult,
    MCPResource,
    MCPResourceContents,
    MCPResourceTemplate,
    MCPToolDef,
)
from aaa.mcp.server import MCPToolServer
from aaa.mcp.spec import MCPSpec

log = logging.getLogger("aaa.mcp.reload")


@dataclass(frozen=True)
class ReloadResult:
    status: str  # applied | unchanged | refused | failed | skipped
    reason: str = ""

    @property
    def applied(self) -> bool:
        return self.status == "applied"


class ReloadableServer(MCPToolServer):
    """Serves whatever the last successful reload compiled. Construct with `await ReloadableServer.start(load, compile)`."""

    def __init__(
        self,
        first: CompiledServer,
        load: Callable[[], MCPSpec],
        compile: Callable[[MCPSpec], Awaitable[CompiledServer]],
        *,
        min_interval_s: float = 2.0,
        grace_s: float = 30.0,
    ) -> None:
        # deliberately not MCPToolServer.__init__: everything is forwarded to the current server (see __getattr__)
        self._current = first
        self._load, self._compile, self._min_interval, self._grace = (
            load,
            compile,
            min_interval_s,
            grace_s,
        )
        self._lock = asyncio.Lock()
        self._last = -1e9
        self._inflight: dict[int, int] = {}
        self._retired: dict[int, CompiledServer] = {}
        self._baseline_refs = first.spec.code_refs()
        self.reloads = 0

    @classmethod
    async def start(
        cls,
        load: Callable[[], MCPSpec],
        compile: Callable[[MCPSpec], Awaitable[CompiledServer]],
        **kw: Any,
    ) -> ReloadableServer:
        return cls(await compile(load()), load, compile, **kw)

    def __getattr__(
        self, name: str
    ) -> Any:  # only reached for attributes this class does not define: name, version, title, spec, report, engine, ...
        if name.startswith("__") or name == "_current":
            raise AttributeError(name)
        return getattr(self._current, name)

    @property
    def current(self) -> CompiledServer:
        return self._current

    # -- the surface hosts use, forwarded ---------------------------------------------------------
    @property
    def has_resources(self) -> bool:
        return self._current.has_resources

    @property
    def has_prompts(self) -> bool:
        return self._current.has_prompts

    def list_tools(
        self, *, tenant: str | None = None, scopes: Iterable[str] | None = None
    ) -> list[MCPToolDef]:
        return self._current.list_tools(tenant=tenant, scopes=scopes)

    def list_resources(
        self, tenant: str | None = None, scopes: Iterable[str] | None = None
    ) -> list[MCPResource]:
        return self._current.list_resources(tenant, scopes)

    def list_resource_templates(
        self, scopes: Iterable[str] | None = None
    ) -> list[MCPResourceTemplate]:
        return self._current.list_resource_templates(scopes)

    def list_prompts(self, scopes: Iterable[str] | None = None) -> list[MCPPrompt]:
        return self._current.list_prompts(scopes)

    def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        scopes: Iterable[str] | None = None,
    ) -> MCPPromptResult:
        return self._current.get_prompt(name, arguments, scopes)

    @contextlib.asynccontextmanager
    async def _hold(self) -> AsyncIterator[CompiledServer]:
        """The current server, kept alive until the caller is done with it even if a reload replaces it meanwhile."""
        inner = self._current
        key = id(inner)
        self._inflight[key] = self._inflight.get(key, 0) + 1
        try:
            yield inner
        finally:
            self._inflight[key] -= 1
            if self._inflight[key] == 0 and key in self._retired:
                await self._close(key)

    async def call(
        self, name: str, arguments: Mapping[str, Any] | None = None, **kw: Any
    ) -> MCPCallResult:
        async with self._hold() as inner:
            return await inner.call(name, arguments, **kw)

    async def read_resource(self, uri: str, **kw: Any) -> MCPResourceContents:
        async with self._hold() as inner:
            return await inner.read_resource(uri, **kw)

    async def aget_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        scopes: Iterable[str] | None = None,
    ) -> MCPPromptResult:
        async with self._hold() as inner:
            return await inner.aget_prompt(name, arguments, scopes)

    # -- reloading ----------------------------------------------------------------------------------
    def _check(self, old: CompiledServer, new: CompiledServer) -> str | None:
        """Why `new` may not replace `old`, or None."""
        for attr in ("name", "title", "version", "instructions"):
            if getattr(old, attr) != getattr(new, attr):
                return f"the server's {attr} changed: restart required"
        if (old.has_resources, old.has_prompts) != (new.has_resources, new.has_prompts):
            return "the set of capabilities (resources, prompts) changed: restart required"
        if old.owns_engine or new.owns_engine:
            return "a configuration that builds its own engine (runtime.engine) cannot be hot-reloaded: restart required"
        return None

    def _new_code(self, spec: MCPSpec) -> str | None:
        if extra := sorted(spec.code_refs() - self._baseline_refs):
            return f"the new configuration would run code that was not running at startup ({', '.join(extra)[:300]}): restart required"
        return None

    async def reload(self, *, force: bool = False) -> ReloadResult:
        """Build the configuration again and swap it in if it is sound. Never raises for a bad configuration."""
        async with self._lock:
            now = time.monotonic()
            if not force and now - self._last < self._min_interval:
                return ReloadResult("skipped", "reloaded a moment ago")
            self._last = now
            try:
                spec = self._load()
            except Exception as e:  # noqa: BLE001 - whatever is wrong with the files, the running server stays
                log.warning("reload failed: %s", type(e).__name__)
                return ReloadResult(
                    "failed",
                    f"the new configuration could not be read ({type(e).__name__}); the running server was not changed",
                )
            if why := self._new_code(spec):  # before compiling: compiling is what imports
                log.warning("reload refused: %s", why)
                return ReloadResult("refused", why)
            try:
                new = await self._compile(spec)
            except Exception as e:  # noqa: BLE001
                log.warning("reload failed: %s", type(e).__name__)
                return ReloadResult(
                    "failed",
                    f"the new configuration could not be built ({type(e).__name__}); the running server was not changed",
                )
            old = self._current
            if why := self._check(old, new):
                await new.aclose()
                log.warning("reload refused: %s", why)
                return ReloadResult("refused", why)
            self._current = new
            self.reloads += 1
            key = id(old)
            self._retired[key] = old
            if self._inflight.get(key, 0) == 0:
                await self._close(key)
            else:
                asyncio.get_running_loop().call_later(
                    self._grace, lambda: asyncio.ensure_future(self._close(key))
                )
            log.info("reloaded: %s", "; ".join(new.report[:1]))
            return ReloadResult("applied", new.report[0] if new.report else "")

    async def _close(self, key: int) -> None:
        old = self._retired.pop(key, None)
        if old is not None:
            await old.aclose()

    async def aclose(self) -> None:
        for key in list(self._retired):
            await self._close(key)
        await self._current.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def fingerprint(files: Iterable[Path]) -> str:
    """A digest of the contents of the configuration files (and which ones they are): what the watcher compares."""
    h = hashlib.sha256()
    for f in sorted({Path(x) for x in files}):
        h.update(str(f).encode())
        with contextlib.suppress(OSError):
            h.update(f.read_bytes())
    return h.hexdigest()


async def watch(
    server: ReloadableServer,
    files: Callable[[], Iterable[Path]],
    *,
    interval_s: float = 5.0,
    on_result: Callable[[ReloadResult], None] | None = None,
) -> None:
    """Reload when the configuration files' contents change. A change must hold still for one more poll first (so a half-written file is not read), and a
    file that cannot be listed or read is simply not a change. Runs until cancelled."""
    try:
        seen = fingerprint(files())
    except Exception:  # noqa: BLE001
        seen = ""
    pending: str | None = None
    while True:
        await asyncio.sleep(interval_s)
        try:
            now = fingerprint(files())
        except Exception:
            log.debug("watch: the configuration could not be listed", exc_info=True)
            continue
        if now == seen:
            pending = None
            continue
        if pending != now:
            pending = now  # changed, but wait for it to settle
            continue
        seen, pending = now, None
        result = await server.reload(force=True)
        if on_result is not None:
            on_result(result)


__all__ = ["ReloadResult", "ReloadableServer", "fingerprint", "watch"]
