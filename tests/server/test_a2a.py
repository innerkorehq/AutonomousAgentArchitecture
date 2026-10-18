"""A2A (experimental): protocol behaviour over ASGI, and the official a2a-sdk client over a real socket."""

from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest
import uvicorn
from aaa_server import A2AConfig, create_app

from helpers.server_env import make_env

CARD_URL = "http://127.0.0.1:0"


def cfg():
    return A2AConfig(
        algorithm="agent", url="http://agent.example", name="Helper", description="helps"
    )


def rpc(method, params=None, id=1):
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}}


def msg(text, **extra):
    return {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": text}], **extra}}


async def make():
    return await make_env(a2a=cfg())


async def test_card_is_public_and_describes_the_binding():
    env, engine = await make()
    async with engine, env.client(None) as c:
        card = (await c.get("/.well-known/agent-card.json")).json()
    assert card["supportedInterfaces"][0] == {
        "url": "http://agent.example/a2a",
        "protocolBinding": "JSONRPC",
        "protocolVersion": "1.0",
    }
    assert card["capabilities"]["streaming"] is True and card["skills"][0]["id"] == "agent"


async def test_send_get_list_and_the_tenant_boundary():
    env, engine = await make()
    async with engine, env.client("key-acme") as a, env.client("key-globex") as g:
        sent = (await a.post("/a2a", json=rpc("SendMessage", msg("hi")))).json()["result"]["task"]
        assert sent["status"]["state"] == "TASK_STATE_COMPLETED"
        assert sent["artifacts"][0]["parts"][0]["text"] == "hello hi"
        tid = sent["id"]
        got = (await a.post("/a2a", json=rpc("GetTask", {"id": tid}))).json()["result"]
        assert got["status"]["state"] == "TASK_STATE_COMPLETED"
        listed = (await a.post("/a2a", json=rpc("ListTasks"))).json()["result"]["tasks"]
        assert [t["id"] for t in listed] == [tid]
        # another tenant sees nothing, and a foreign id is indistinguishable from a missing one
        foreign = (await g.post("/a2a", json=rpc("GetTask", {"id": tid}))).json()["error"]
        missing = (await g.post("/a2a", json=rpc("GetTask", {"id": "nope"}))).json()["error"]
        assert foreign == missing and foreign["code"] == -32001
        assert (await g.post("/a2a", json=rpc("ListTasks"))).json()["result"]["tasks"] == []
        assert (await g.post("/a2a", json=rpc("CancelTask", {"id": tid}))).json()["error"][
            "code"
        ] == -32001
    started = (await env.engine.store.run_index())[0].started
    assert started.tenant == "acme" and started.meta == {"user": "ann"}


async def test_refusals():
    env, engine = await make()
    async with engine, env.client("key-acme") as c:

        async def err(method, params):
            return (await c.post("/a2a", json=rpc(method, params))).json()["error"]["code"]

        assert (
            await err("SendMessage", msg("x", taskId="t-1")) == -32001
        )  # naming a task that does not exist
        non_text = {
            "message": {
                "messageId": "m",
                "role": "ROLE_USER",
                "parts": [{"url": "https://x/y.png"}],
            }
        }
        assert await err("SendMessage", non_text) == -32005
        assert (
            await err("SendMessage", {"message": {"role": "ROLE_AGENT", "parts": [{"text": "x"}]}})
            == -32602
        )
        assert await err("SendMessage", {}) == -32602
        assert await err("Nope", {}) == -32601
        assert await err("GetTask", {}) == -32602
        r = await c.post("/a2a", json=rpc("GetTask", {"id": "x"}), headers={"A2A-Version": "2.0"})
        assert r.json()["error"]["code"] == -32009
        assert (await c.post("/a2a", content=b"{bad")).status_code == 400
        assert (await c.post("/a2a", json=[1])).status_code == 400


async def test_authentication_and_scope_apply():
    env, engine = await make()
    async with engine, env.client(None) as anon:
        assert (await anon.post("/a2a", json=rpc("ListTasks"))).status_code == 401


async def test_a_task_waiting_for_approval_is_input_required_and_cancellable():
    env, engine = await make()
    async with engine, env.client("key-acme") as c:
        t = (
            await c.post(
                "/a2a",
                json=rpc(
                    "SendMessage", {**msg("wire it"), "configuration": {"returnImmediately": True}}
                ),
            )
        ).json()["result"]["task"]
        tid = t["id"]
        for _ in range(200):
            state = (await c.post("/a2a", json=rpc("GetTask", {"id": tid}))).json()["result"][
                "status"
            ]
            if state["state"] == "TASK_STATE_INPUT_REQUIRED":
                break
            await asyncio.sleep(0.02)
        assert (
            state["state"] == "TASK_STATE_INPUT_REQUIRED"
            and "approval" in state["message"]["parts"][0]["text"]
        )
        cancelled = (await c.post("/a2a", json=rpc("CancelTask", {"id": tid}))).json()["result"]
        assert cancelled["status"]["state"] == "TASK_STATE_CANCELED"
        again = (await c.post("/a2a", json=rpc("CancelTask", {"id": tid}))).json()["error"]
        assert again["code"] == -32002


async def test_quota_and_budget_apply_to_a2a_tasks():
    from aaa_server import Quota, QuotaStore

    from aaa.events import InMemoryEventStore

    env, engine = await make_env(
        a2a=cfg(), quota=QuotaStore(InMemoryEventStore(), Quota(max_runs=1))
    )
    async with engine, env.client("key-acme") as c:
        assert (await c.post("/a2a", json=rpc("SendMessage", msg("one")))).status_code == 200
        second = await c.post("/a2a", json=rpc("SendMessage", msg("two")))
        assert second.status_code == 429
    run = (await env.engine.store.run_index())[0].started
    assert run.data["budget"]["max_steps"] == 20  # the server's ceiling, not the client's choice


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_the_official_a2a_sdk_client_end_to_end():
    pytest.importorskip("a2a")
    from a2a.client import ClientConfig, create_client
    from a2a.types import a2a_pb2 as pb
    from google.protobuf import json_format

    env, engine = await make_env(a2a=A2AConfig(algorithm="agent", url="PLACEHOLDER", name="Helper"))
    port = free_port()
    env.cfg.a2a.url = f"http://127.0.0.1:{port}"  # type: ignore[union-attr]
    app = create_app(env.cfg)
    async with engine:
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.02)
        try:
            base = f"http://127.0.0.1:{port}"
            # the card parses as the protocol's own AgentCard
            async with httpx.AsyncClient() as plain:
                card_json = (await plain.get(f"{base}/.well-known/agent-card.json")).json()
            json_format.ParseDict(card_json, pb.AgentCard())

            async with httpx.AsyncClient(
                headers={"Authorization": "Bearer key-acme"}, timeout=20
            ) as http:
                request = pb.SendMessageRequest(
                    message=pb.Message(
                        message_id="m1", role=pb.ROLE_USER, parts=[pb.Part(text="hello there")]
                    )
                )
                for streaming in (False, True):
                    client = await create_client(
                        base, ClientConfig(httpx_client=http, streaming=streaming)
                    )
                    events = [ev async for ev in client.send_message(request)]
                    texts = []
                    state = None
                    for ev in events:
                        kind = ev.WhichOneof("payload")
                        if kind == "task":
                            state = ev.task.status.state
                            for art in ev.task.artifacts:
                                texts += [p.text for p in art.parts]
                        elif kind == "status_update":
                            state = ev.status_update.status.state
                        elif kind == "artifact_update":
                            texts += [p.text for p in ev.artifact_update.artifact.parts]
                    assert state == pb.TASK_STATE_COMPLETED, (streaming, events)
                    assert "hello hello there" in "".join(texts), (streaming, texts)
                got = await client.get_task(pb.GetTaskRequest(id=events[0].task.id))
                assert got.status.state == pb.TASK_STATE_COMPLETED
            # without the credential the SDK client is refused
            async with httpx.AsyncClient(timeout=20) as anon:
                bad = await create_client(base, ClientConfig(httpx_client=anon, streaming=False))
                with pytest.raises(Exception):  # noqa: B017
                    _ = [ev async for ev in bad.send_message(request)]
        finally:
            server.should_exit = True
            await task
    _ = json
