from __future__ import annotations

import httpx
from aaa_server import ServerConfig, create_app
from aaa_server.auth import ApiKeyAuthenticator

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text, tool_call
from aaa.runtime import Budget, Engine
from aaa.session import SessionManager
from aaa.tools import ToolHandler, ToolRegistry, tool

KEYS = {
    "key-acme": ("acme", "ann", ["run", "session", "admin"]),
    "key-acme-approver": ("acme", "root", ["run", "session", "approve"]),
    "key-globex": ("globex", "gus", ["run", "session", "approve", "admin"]),
    "key-acme-answerer": ("acme", "amy", ["run", "answer"]),
    "key-globex-answerer": ("globex", "gina", ["run", "answer"]),
}


@tool(requires_approval=True)
def wire_money(amount: int) -> str:
    """Send money."""
    return f"sent {amount}"


def script(req):
    last = req.messages[-1]
    if last.role == "user" and "wire" in (last.content or ""):
        return tool_call("wire_money", amount=250)
    if last.role == "tool":
        return text(f"done: {last.content}")
    return text("hello " + (last.content or ""))


class Env:
    def __init__(self, engine: Engine, cfg: ServerConfig, algo: ToolLoop) -> None:
        self.engine, self.cfg, self.algo = engine, cfg, algo
        self.app = create_app(cfg)

    def client(self, key: str | None = "key-acme") -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://test",
            headers=headers,
        )


async def make_env(**over) -> tuple[Env, Engine]:
    engine = Engine()
    registry = ToolRegistry([wire_money])
    LLMHandler(FakeLLMClient(script)).attach(engine)
    ToolHandler(registry).attach(engine)
    algo = ToolLoop(registry, system="be brief")
    manager = SessionManager(engine, ToolLoop(ToolRegistry([])), system="be brief")
    cfg = ServerConfig(
        engine=engine,
        authenticator=ApiKeyAuthenticator.from_keys(KEYS),
        algorithms={"agent": algo},
        sessions={"chat": manager},
        budget=Budget(max_steps=20, max_tokens=10_000, deadline_s=30.0),
        keepalive_s=0.2,
        **over,
    )
    return Env(engine, cfg, algo), engine
