"""MCP resources: read-only context a client can pull in, from four kinds of provider.

    static   inline text                         aaa://static/<name>
    files    a directory (confined, per tenant)  aaa://files/<name>/<path>
    index    search over an `Index`              aaa://index/<name>/search?q=<text>
    memory   search over a memory backend        aaa://memory/<name>/search?q=<text>

Every failure to read (unknown name, bad path, a path outside the root, another tenant's document, a provider error) raises the same
`ResourceNotFound`, so a client cannot tell "forbidden" from "missing". Index and file providers are tenant-scoped unless the
operator says `shared`; memory is scoped by its own `Scope(tenant, user)`."""

from __future__ import annotations

import fnmatch
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import parse_qs, unquote, urlsplit

from aaa.mcp.model import MCPResource, MCPResourceContents, MCPResourceTemplate
from aaa.tools.builtin.files import FileAccess
from aaa.tools.tool import ToolError

if TYPE_CHECKING:
    from aaa.data.index import Index

log = logging.getLogger("aaa.mcp.resources")
SCHEME = "aaa"
_SLUG = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TENANT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MAX_QUERY = 500
SCAN_LIMIT = 5000  # directory entries visited when listing, so a huge tree cannot stall a list call


class ResourceNotFound(Exception):
    """Unknown, forbidden or unreadable: deliberately one error, so existence does not leak."""


class Provider(Protocol):
    kind: str
    name: str

    def list(self, tenant: str) -> list[MCPResource]: ...
    def templates(self) -> list[MCPResourceTemplate]: ...
    async def read(
        self, tenant: str, user: str | None, rest: str, query: dict[str, str]
    ) -> MCPResourceContents: ...


def parse_uri(uri: str) -> tuple[str, str, str, dict[str, str]]:
    """`aaa://<kind>/<name>/<rest>?q=..` -> (kind, name, rest, query). Anything else raises `ResourceNotFound`."""
    try:
        u = urlsplit(uri)
    except ValueError:
        raise ResourceNotFound from None
    if u.scheme != SCHEME or u.fragment or u.username or u.port:
        raise ResourceNotFound
    kind = u.netloc
    parts = u.path.split("/", 2)  # ['', name, rest]
    if (
        len(parts) < 2
        or not _SLUG.match(parts[1])
        or kind not in ("static", "files", "index", "memory", "upstream")
    ):
        raise ResourceNotFound
    q = {k: v[0] for k, v in parse_qs(u.query, keep_blank_values=False, max_num_fields=4).items()}
    return kind, parts[1], unquote(parts[2]) if len(parts) > 2 else "", q


def _glob(rel: str, pattern: str) -> bool:
    """Shell-style match on a relative path where `*` crosses directories and a leading `**/` also matches files at the top."""
    return fnmatch.fnmatch(rel, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:])
    )


def _clip(text: str, limit: int) -> str:
    if len(text.encode()) <= limit:
        return text
    return (
        text.encode()[:limit].decode("utf-8", errors="ignore") + f"\n[truncated at {limit} bytes]"
    )


def _safe_tenant(tenant: str) -> str:
    if not _TENANT.match(tenant) or ".." in tenant:
        raise ResourceNotFound
    return tenant


# -- providers ---------------------------------------------------------------------------------
class StaticProvider:
    kind = "static"

    def __init__(
        self,
        name: str,
        text: str,
        *,
        mime_type: str,
        title: str | None,
        description: str | None,
        limit: int,
    ) -> None:
        self.name, self.text, self.mime, self.title, self.description, self.limit = (
            name,
            text,
            mime_type,
            title,
            description,
            limit,
        )

    @property
    def uri(self) -> str:
        return f"{SCHEME}://static/{self.name}"

    def list(self, tenant: str) -> list[MCPResource]:
        return [
            MCPResource(
                self.uri,
                self.name,
                self.title,
                self.description,
                self.mime,
                len(self.text.encode()),
            )
        ]

    def templates(self) -> list[MCPResourceTemplate]:
        return []

    async def read(
        self, tenant: str, user: str | None, rest: str, query: dict[str, str]
    ) -> MCPResourceContents:
        if rest or query:
            raise ResourceNotFound
        return MCPResourceContents(self.uri, _clip(self.text, self.limit), self.mime)


class FilesProvider:
    kind = "files"

    def __init__(
        self,
        name: str,
        root: str | os.PathLike[str],
        *,
        pattern: str = "**/*",
        shared: bool = False,
        allow_hidden: bool = False,
        mime_type: str | None = None,
        title: str | None = None,
        description: str | None = None,
        limit: int,
        max_listed: int,
    ) -> None:
        self.name, self.root, self.pattern, self.shared = name, Path(root), pattern, shared
        self.allow_hidden, self.mime, self.title, self.description = (
            allow_hidden,
            mime_type,
            title,
            description,
        )
        self.limit, self.max_listed = limit, max_listed

    def _access(self, tenant: str) -> FileAccess:
        base = self.root.resolve()
        root = base if self.shared else (base / _safe_tenant(tenant)).resolve()
        try:
            root.relative_to(base)
        except ValueError:
            raise ResourceNotFound from None
        if not root.is_dir():
            raise ResourceNotFound
        return FileAccess(root, max_bytes=self.limit, allow_hidden=self.allow_hidden)

    def _mime(self, path: str) -> str:
        return self.mime or mimetypes.guess_type(path)[0] or "text/plain"

    def list(self, tenant: str) -> list[MCPResource]:
        try:
            fa = self._access(tenant)
        except ResourceNotFound:
            return []
        out: list[MCPResource] = []
        visited = 0
        for dirpath, dirnames, filenames in os.walk(fa.root):
            dirnames[:] = sorted(d for d in dirnames if self.allow_hidden or not d.startswith("."))
            for fname in sorted(filenames):
                visited += 1
                if visited > SCAN_LIMIT or len(out) >= self.max_listed:
                    return out
                full = Path(dirpath) / fname
                rel = full.relative_to(fa.root).as_posix()
                if not _glob(rel, self.pattern):
                    continue
                try:
                    fa.resolve(rel)  # hidden/denied/escaping entries are not listed
                except ToolError:
                    continue
                out.append(
                    MCPResource(
                        f"{SCHEME}://files/{self.name}/{rel}",
                        rel,
                        self.title,
                        self.description,
                        self._mime(rel),
                        full.stat().st_size,
                    )
                )
        return out

    def templates(self) -> list[MCPResourceTemplate]:
        return [
            MCPResourceTemplate(
                f"{SCHEME}://files/{self.name}/{{path}}",
                self.name,
                self.title,
                self.description,
                self.mime,
            )
        ]

    async def read(
        self, tenant: str, user: str | None, rest: str, query: dict[str, str]
    ) -> MCPResourceContents:
        if not rest or query or not _glob(rest, self.pattern):
            raise ResourceNotFound
        try:
            got = self._access(tenant).read(rest)
        except ToolError:
            raise ResourceNotFound from None
        return MCPResourceContents(
            f"{SCHEME}://files/{self.name}/{rest}", str(got["text"]), self._mime(rest)
        )


class IndexProvider:
    kind = "index"

    def __init__(
        self,
        name: str,
        index: Index,
        *,
        k: int,
        max_chars: int,
        tenant_key: str | None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        self.name, self.index, self.k, self.max_chars, self.tenant_key = (
            name,
            index,
            k,
            max_chars,
            tenant_key,
        )
        self.title, self.description = title, description

    def list(self, tenant: str) -> list[MCPResource]:
        return []

    def templates(self) -> list[MCPResourceTemplate]:
        return [
            MCPResourceTemplate(
                f"{SCHEME}://index/{self.name}/search?q={{query}}",
                self.name,
                self.title,
                self.description or "Search results as JSON.",
                "application/json",
            )
        ]

    async def read(
        self, tenant: str, user: str | None, rest: str, query: dict[str, str]
    ) -> MCPResourceContents:
        q = query.get("q", "")
        if rest != "search" or not q or len(q) > MAX_QUERY or set(query) - {"q"}:
            raise ResourceNotFound
        flt = (lambda d: d.meta.get(self.tenant_key) == tenant) if self.tenant_key else None
        hits = await self.index.search(q, self.k, filter=flt)
        body = [
            {
                "id": h.doc.id,
                "score": h.score,
                "text": h.doc.text[: self.max_chars],
                "meta": h.doc.meta,
            }
            for h in hits
        ]
        return MCPResourceContents(
            f"{SCHEME}://index/{self.name}/search?q={q}",
            json.dumps(body, default=str),
            "application/json",
        )


class MemoryProvider:
    kind = "memory"

    def __init__(
        self,
        name: str,
        memory: Any,
        *,
        k: int,
        max_chars: int,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        self.name, self.memory, self.k, self.max_chars = name, memory, k, max_chars
        self.title, self.description = title, description

    def list(self, tenant: str) -> list[MCPResource]:
        return []

    def templates(self) -> list[MCPResourceTemplate]:
        return [
            MCPResourceTemplate(
                f"{SCHEME}://memory/{self.name}/search?q={{query}}",
                self.name,
                self.title,
                self.description or "Memories matching a query, as JSON.",
                "application/json",
            )
        ]

    async def read(
        self, tenant: str, user: str | None, rest: str, query: dict[str, str]
    ) -> MCPResourceContents:
        from aaa.memory.types import Scope

        q = query.get("q", "")
        if rest != "search" or not q or len(q) > MAX_QUERY or set(query) - {"q"}:
            raise ResourceNotFound
        items = await self.memory.search(q, Scope(tenant=tenant, user=user), k=self.k)
        body = [{"id": m.id, "text": m.text[: self.max_chars]} for m in items]
        return MCPResourceContents(
            f"{SCHEME}://memory/{self.name}/search?q={q}",
            json.dumps(body, default=str),
            "application/json",
        )


class ResourceSet:
    """The providers of a server, addressed by (kind, name)."""

    def __init__(self, providers: list[Provider], *, max_listed: int) -> None:
        self.providers = {(p.kind, p.name): p for p in providers}
        if len(self.providers) != len(providers):
            raise ValueError("two resources share a kind and a name")
        self.max_listed = max_listed

    def __bool__(self) -> bool:
        return bool(self.providers)

    def list(self, tenant: str) -> list[MCPResource]:
        out: list[MCPResource] = []
        for p in self.providers.values():
            out += p.list(tenant)
            if len(out) >= self.max_listed:
                break
        return out[: self.max_listed]

    def templates(self) -> list[MCPResourceTemplate]:
        return [t for p in self.providers.values() for t in p.templates()]

    async def read(self, uri: str, tenant: str, user: str | None) -> MCPResourceContents:
        kind, name, rest, query = parse_uri(uri)
        p = self.providers.get((kind, name))
        if p is None:
            raise ResourceNotFound
        try:
            return await p.read(tenant, user, rest, query)
        except ResourceNotFound:
            raise
        except Exception:
            log.warning("resource provider %s/%s failed", kind, name, exc_info=True)
            raise ResourceNotFound from None
