"""MCP over HTTP in aaa-server: tenant comes from the credential, and the official SDK client can use it."""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest
import uvicorn
from aaa_server import create_app

from aaa.mcp import MCPToolServer
from aaa.tools import ToolRegistry, tool
from helpers.server_env import make_env


@tool
def whoami(ctx) -> str:
    """Report the caller as the server sees it."""
    return f"{ctx.tenant}/{ctx.meta.get('user')}"


@tool(requires_approval=True)
def wire_money(amount: int) -> str:
    """Send money."""
    return "sent"


def rpc(method, params=None, id=1):
    return {"jsonrpc": "2.0", "id": id, "method": method, **({"params": params} if params else {})}


async def make(**over):
    core = MCPToolServer(ToolRegistry([whoami, wire_money]))
    env, engine = await make_env(mcp=core, **over)
    return env, engine


async def test_tenant_and_user_are_the_principals_whatever_the_client_says():
    env, engine = await make()
    async with engine, env.client("key-acme") as a, env.client("key-globex") as g:
        init = (
            await a.post("/mcp", json=rpc("initialize", {"protocolVersion": "2025-06-18"}))
        ).json()
        assert init["result"]["protocolVersion"] == "2025-06-18"
        call = rpc("tools/call", {"name": "whoami", "arguments": {}})
        assert (await a.post("/mcp", json=call)).json()["result"]["content"][0][
            "text"
        ] == "acme/ann"
        assert (await g.post("/mcp", json=call)).json()["result"]["content"][0][
            "text"
        ] == "globex/gus"
        spoof = rpc("tools/call", {"name": "whoami", "arguments": {"tenant": "globex"}})
        assert (await a.post("/mcp", json=spoof)).json()["result"]["isError"] is True
        tools = (await a.post("/mcp", json=rpc("tools/list"))).json()["result"]["tools"]
        assert [t["name"] for t in tools] == ["whoami"]  # the approval tool is not offered
        hidden = (
            await a.post(
                "/mcp", json=rpc("tools/call", {"name": "wire_money", "arguments": {"amount": 1}})
            )
        ).json()
        assert hidden["result"]["isError"] is True


async def test_protocol_edges():
    env, engine = await make()
    async with engine, env.client("key-acme") as c:
        assert (
            await c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        ).status_code == 202
        assert (
            await c.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "result": {}})
        ).status_code == 202
        assert (await c.post("/mcp", json=rpc("ping"))).json()["result"] == {}
        assert (await c.post("/mcp", json=rpc("resources/list"))).json()["error"]["code"] == -32601
        assert (await c.post("/mcp", json=rpc("tools/call", {"name": 5}))).json()["error"][
            "code"
        ] == -32602
        assert (await c.post("/mcp", content=b"{nope")).status_code == 400
        assert (await c.post("/mcp", json=[rpc("ping")])).json()["error"]["code"] == -32600
        assert (await c.get("/mcp")).status_code == 405
        old = (
            await c.post("/mcp", json=rpc("initialize", {"protocolVersion": "1999-01-01"}))
        ).json()
        assert old["result"]["protocolVersion"] == "2025-11-25"


async def test_authentication_scope_origin_and_size_apply_to_mcp_too():
    env, engine = await make()
    async with engine:
        async with env.client(None) as anon:
            assert (await anon.post("/mcp", json=rpc("ping"))).status_code == 401
        async with env.client("key-acme") as c:
            assert (
                await c.post("/mcp", json=rpc("ping"), headers={"Origin": "https://evil.example"})
            ).status_code == 403
            assert (await c.post("/mcp", content=b" " * (2 << 20))).status_code == 413
    ok_env, ok_engine = await make(mcp_origins=("https://app.example",))
    async with ok_engine, ok_env.client("key-acme") as c:
        assert (
            await c.post("/mcp", json=rpc("ping"), headers={"Origin": "https://app.example"})
        ).status_code == 200


async def test_without_a_configured_server_there_is_no_endpoint():
    env, engine = await make_env()
    async with engine, env.client("key-acme") as c:
        assert (await c.post("/mcp", json=rpc("ping"))).status_code == 404


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_the_official_sdk_client_works_against_it_over_a_real_socket():
    mcp = pytest.importorskip("mcp")
    env, engine = await make()
    port = free_port()
    async with engine:
        server = uvicorn.Server(
            uvicorn.Config(create_app(env.cfg), host="127.0.0.1", port=port, log_level="error")
        )
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.02)
        try:
            from aaa.mcp import MCPClient

            url = f"http://127.0.0.1:{port}/mcp"
            async with MCPClient.http(
                url, headers={"Authorization": "Bearer key-acme"}, name="srv"
            ) as client:
                assert [t.name for t in await client.list_tools()] == ["whoami"]
                out, is_error = await client.call("whoami", {})
                assert out == "acme/ann" and not is_error
            with pytest.raises(Exception):  # noqa: B017 - the SDK's own error type varies by version
                async with MCPClient.http(
                    url, headers={"Authorization": "Bearer nope"}, name="srv"
                ) as bad:
                    await bad.list_tools()
        finally:
            server.should_exit = True
            await task
    _ = mcp, httpx


async def test_tool_scopes_follow_the_authenticated_principal():
    from aaa.mcp.policy import ToolRule

    core = MCPToolServer(
        ToolRegistry([whoami]),
        tool_policy={"whoami": ToolRule(scopes=frozenset({"admin"}))},
    )
    env, engine = await make_env(mcp=core)
    async with engine, env.client("key-acme") as admin, env.client("key-acme-approver") as plain:
        listed = lambda r: [t["name"] for t in r.json()["result"]["tools"]]
        assert listed(await admin.post("/mcp", json=rpc("tools/list"))) == ["whoami"]
        assert listed(await plain.post("/mcp", json=rpc("tools/list"))) == []
        call = rpc("tools/call", {"name": "whoami", "arguments": {}})
        assert (await admin.post("/mcp", json=call)).json()["result"]["isError"] is False
        a = (await plain.post("/mcp", json=call)).json()["result"]
        b = (
            await plain.post("/mcp", json=rpc("tools/call", {"name": "nope", "arguments": {}}))
        ).json()["result"]
        assert a["isError"] and a["content"][0]["text"].replace("whoami", "X") == b["content"][0][
            "text"
        ].replace("nope", "X")


async def test_profile_scopes_of_the_api_key_decide_what_a_caller_sees():
    from aaa.mcp.policy import Profile, ProfileSet

    @tool
    def internal() -> str:
        """Internal only."""
        return "internal"

    core = MCPToolServer(
        ToolRegistry([whoami, internal]),
        profiles=ProfileSet({"support": Profile(tools=("whoami",))}),
    )
    env, engine = await make_env(mcp=core)
    async with (
        engine,
        env.client("key-acme") as plain,
        env.client("key-acme-support") as support,
        env.client("key-acme-ghost") as ghost,
    ):
        listed = lambda r: [t["name"] for t in r.json()["result"]["tools"]]
        assert sorted(listed(await plain.post("/mcp", json=rpc("tools/list")))) == [
            "internal",
            "whoami",
        ]  # no profile scope: unrestricted
        assert listed(await support.post("/mcp", json=rpc("tools/list"))) == ["whoami"]
        assert (
            listed(await ghost.post("/mcp", json=rpc("tools/list"))) == []
        )  # a misspelt profile fails closed
        call = rpc("tools/call", {"name": "internal", "arguments": {}})
        a = (await support.post("/mcp", json=call)).json()["result"]
        b = (
            await support.post("/mcp", json=rpc("tools/call", {"name": "nope", "arguments": {}}))
        ).json()["result"]
        assert a["isError"] and a["content"][0]["text"].replace("internal", "X") == b["content"][0][
            "text"
        ].replace("nope", "X")
        assert (await plain.post("/mcp", json=call)).json()["result"]["isError"] is False


def test_a_token_keeps_profile_scopes_and_drops_ones_the_server_does_not_know():
    from aaa_server.auth import _known

    assert _known(frozenset({"run", "profile:support", "profile:Bad", "root", "profile:"})) == {
        "run",
        "profile:support",
    }
