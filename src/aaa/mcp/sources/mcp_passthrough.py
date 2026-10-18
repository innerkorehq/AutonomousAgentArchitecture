"""Resources and prompts of an upstream MCP server, re-exposed by this server (opt-in per source: `resources = true`, `prompts = true`).

What crosses is narrower than what the upstream offers, on purpose:

* **Resources** are listed once at startup. Each is re-exposed under a namespaced URI, `aaa://upstream/<source>/<token>`, where the token encodes the upstream's own URI.
  A read is accepted only for a token that was **listed**: a caller cannot ask the upstream for any other URI, and resource *templates* are not passed on at all. Only text
  is passed (a binary resource is left out), cut to `max_resource_bytes`, and every failure is the uniform `resource not found`.
* **Prompts** keep only their text messages (roles user and assistant), with the number of messages, their size, names, descriptions and argument values capped. Only the
  arguments the upstream declared are accepted, and any failure on the upstream's side is a generic message.

What comes back is the upstream's words and they reach your callers' models: the same untrusted-content rule as for tool results. One connection serves every caller (it
is not tenant-isolated)."""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import inspect
import logging
from typing import Any

from aaa.mcp.client import _attr
from aaa.mcp.model import (
    MCPPrompt,
    MCPPromptArgument,
    MCPPromptResult,
    MCPResource,
    MCPResourceContents,
    MCPResourceTemplate,
)
from aaa.mcp.sources.prompts import PromptError
from aaa.mcp.sources.resources import SCHEME, ResourceNotFound, _clip
from aaa.mcp.spec import McpSource

log = logging.getLogger("aaa.mcp.upstream")
MAX_PAGES, MAX_ITEMS = 10, 500
MAX_PROMPT_MESSAGES, MAX_PROMPT_CHARS, MAX_ARG_CHARS = 20, 20_000, 2_000
TEXT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/toml",
)


def token_of(uri: str) -> str:
    return base64.urlsafe_b64encode(uri.encode()).decode().rstrip("=")


def uri_of(token: str) -> str | None:
    try:
        return base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
    except (ValueError, UnicodeDecodeError):
        return None


async def paged(call: Any, field: str) -> list[Any]:
    """Every item of a paginated list call (a bounded number of pages and items), whichever way this SDK takes the cursor."""
    out: list[Any] = []
    cursor: str | None = None
    takes_cursor = "cursor" in inspect.signature(call).parameters
    for _ in range(MAX_PAGES):
        if cursor is None:
            res = await call()
        elif takes_cursor:
            res = await call(cursor=cursor)
        else:
            from mcp.types import PaginatedRequestParams  # pyright: ignore[reportMissingImports]

            res = await call(params=PaginatedRequestParams(cursor=cursor))
        out += list(_attr(res, field, default=[]))
        cursor = _attr(res, "next_cursor", "nextCursor")
        if not cursor or len(out) >= MAX_ITEMS:
            break
    return out[:MAX_ITEMS]


def _text_ok(mime: str | None) -> bool:
    return mime is None or any(mime.startswith(t) for t in TEXT_TYPES)


class UpstreamResources:
    kind = "upstream"

    def __init__(self, src: McpSource, upstream: Any, listed: list[Any]) -> None:
        self.src, self.upstream, self.name = src, upstream, src.name
        self.items: dict[str, MCPResource] = {}
        for r in listed:
            orig = str(_attr(r, "uri"))
            mime = _attr(r, "mime_type", "mimeType")
            rname = str(_attr(r, "name", default=orig))
            if not _text_ok(mime) or any(
                fnmatch.fnmatchcase(x, p) for p in src.resource_exclude for x in (rname, orig)
            ):
                continue
            tok = token_of(orig)
            desc = src.description_overrides.get(rname) or _attr(r, "description")
            self.items[tok] = MCPResource(
                f"{SCHEME}://upstream/{self.name}/{tok}",
                rname[:200],
                (_attr(r, "title") or None) and str(_attr(r, "title"))[:200],
                str(desc)[: src.description_max_chars] if desc else None,
                mime,
                None,
            )
            if len(self.items) >= src.max_listed:
                break

    def list(self, tenant: str) -> list[MCPResource]:
        return list(self.items.values())

    def templates(self) -> list[MCPResourceTemplate]:
        return []

    async def read(
        self, tenant: str, user: str | None, rest: str, query: dict[str, str]
    ) -> MCPResourceContents:
        item = self.items.get(rest)
        client = self.upstream.client
        if item is None or query or client is None:
            raise ResourceNotFound
        orig = uri_of(rest)
        if orig is None:
            raise ResourceNotFound
        try:
            res = await asyncio.wait_for(
                client.session.read_resource(orig), self.src.call_timeout_s
            )
        except Exception:
            log.warning("upstream %s: reading a resource failed", self.name, exc_info=True)
            raise ResourceNotFound from None
        texts = [
            str(t)
            for c in _attr(res, "contents", default=[])
            if (t := _attr(c, "text")) is not None
        ]
        if not texts:
            raise ResourceNotFound  # nothing textual came back
        return MCPResourceContents(
            item.uri,
            _clip("\n".join(texts), self.src.max_resource_bytes),
            item.mime_type or "text/plain",
        )


class UpstreamPrompt:
    def __init__(self, src: McpSource, upstream: Any, info: Any, name: str) -> None:
        self.src, self.upstream, self.name = src, upstream, name
        self.remote = str(_attr(info, "name"))
        desc = src.description_overrides.get(self.remote) or _attr(info, "description")
        self.description = str(desc)[: src.description_max_chars] if desc else None
        self.title = (str(_attr(info, "title"))[:200]) if _attr(info, "title") else None
        self.args = [
            MCPPromptArgument(
                str(_attr(a, "name"))[:64],
                (str(_attr(a, "description"))[:500] if _attr(a, "description") else None),
                bool(_attr(a, "required", default=False)),
            )
            for a in (_attr(info, "arguments", default=None) or [])
        ]

    def describe(self) -> MCPPrompt:
        return MCPPrompt(self.name, self.description, self.title, tuple(self.args))

    def get(
        self, arguments: Any = None
    ) -> MCPPromptResult:  # the synchronous path exists only for prompts that need no I/O
        raise PromptError(
            "this prompt is fetched from another server and must be requested asynchronously"
        )

    async def aget(self, arguments: Any = None) -> MCPPromptResult:
        given = dict(arguments or {})
        known = {a.name: a for a in self.args}
        if unknown := sorted(set(given) - set(known)):
            raise PromptError(f"unknown argument(s) {unknown}; this prompt takes {sorted(known)}")
        for a in self.args:
            if a.required and a.name not in given:
                raise PromptError(f"missing required argument {a.name!r}")
        values = {
            k: (v if isinstance(v, str) else str(v))[:MAX_ARG_CHARS] for k, v in given.items()
        }
        client = self.upstream.client
        if client is None:
            raise PromptError("the upstream server is not available")
        try:
            res = await asyncio.wait_for(
                client.session.get_prompt(self.remote, values), self.src.call_timeout_s
            )
        except Exception:
            log.warning(
                "upstream %s: filling prompt %s failed", self.src.name, self.remote, exc_info=True
            )
            raise PromptError("the upstream server could not fill this prompt") from None
        messages: list[tuple[str, str]] = []
        for m in list(_attr(res, "messages", default=[]))[:MAX_PROMPT_MESSAGES]:
            role = str(_attr(m, "role", default=""))
            content = _attr(m, "content")
            text = _attr(content, "text")
            if role in ("user", "assistant") and isinstance(text, str) and text:
                messages.append((role, text[:MAX_PROMPT_CHARS]))
        if not messages:
            raise PromptError("the upstream server returned nothing usable for this prompt")
        desc = _attr(res, "description")
        return MCPPromptResult(
            str(desc)[: self.src.description_max_chars] if desc else self.description,
            tuple(messages),
        )
