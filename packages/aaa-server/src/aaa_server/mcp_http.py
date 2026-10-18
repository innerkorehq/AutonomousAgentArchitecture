"""MCP over Streamable HTTP, served by `aaa-server` so the same authentication, tenant and rate-limit rules apply.

This is the stateless subset of the transport: every POST carries one JSON-RPC message and gets one JSON answer (no sessions, no
server-initiated messages, so `GET` is `405`). That is enough for `initialize`, `ping`, `tools/list` and `tools/call`, which is
all a tool server needs, and it is what lets the tenant of every call come from the authenticated principal instead of from
anything the MCP client says."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from aaa.mcp import wire
from aaa.mcp.server import MCPToolServer
from aaa.mcp.sources.prompts import PromptError
from aaa.mcp.sources.resources import ResourceNotFound

SUPPORTED = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602
RESOURCE_NOT_FOUND = -32002


def _error(id_: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


async def handle(
    core: MCPToolServer,
    message: Any,
    *,
    tenant: str,
    user: str,
    scopes: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Answer one JSON-RPC message, or `None` for a notification / client response (which get `202`)."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, INVALID_REQUEST, "expected a JSON-RPC 2.0 object")
    method, id_, params = message.get("method"), message.get("id"), message.get("params") or {}
    if method is None:  # a response to something we never sent
        return None
    if id_ is None:  # a notification
        return None
    if not isinstance(method, str) or not isinstance(params, dict):
        return _error(id_, INVALID_REQUEST, "malformed request")
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in SUPPORTED else SUPPORTED[0]
        return {"jsonrpc": "2.0", "id": id_, "result": wire.initialize_result(core, version)}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": id_, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": id_, "result": wire.tools_list_result(core, tenant, scopes)}
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            return _error(id_, INVALID_PARAMS, "tools/call needs a name and an arguments object")
        r = await core.call(name, args, tenant=tenant, user=user, scopes=scopes)
        return {"jsonrpc": "2.0", "id": id_, "result": wire.call_result_to_wire(r)}
    if method.startswith("resources/") and core.has_resources:
        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": wire.resources_list_result(core.list_resources(tenant, scopes)),
            }
        if method == "resources/templates/list":
            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": wire.resource_templates_list_result(core.list_resource_templates(scopes)),
            }
        if method == "resources/read":
            uri = params.get("uri")
            if not isinstance(uri, str):
                return _error(id_, INVALID_PARAMS, "resources/read needs a uri")
            try:
                c = await core.read_resource(uri, tenant=tenant, user=user, scopes=scopes)
            except ResourceNotFound:
                return _error(id_, RESOURCE_NOT_FOUND, "resource not found")
            return {"jsonrpc": "2.0", "id": id_, "result": wire.resource_read_result(c)}
    if method.startswith("prompts/") and core.has_prompts:
        if method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": wire.prompts_list_result(core.list_prompts(scopes)),
            }
        if method == "prompts/get":
            name, args = params.get("name"), params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(args, dict):
                return _error(
                    id_, INVALID_PARAMS, "prompts/get needs a name and an arguments object"
                )
            try:
                r = await core.aget_prompt(name, args, scopes)
            except PromptError as e:
                return _error(id_, INVALID_PARAMS, str(e))
            return {"jsonrpc": "2.0", "id": id_, "result": wire.prompt_get_result(r)}
    return _error(id_, METHOD_NOT_FOUND, f"method {method!r} is not supported")
