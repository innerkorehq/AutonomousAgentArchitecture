"""Hot reload: build first, swap atomically, refuse anything that would change identity or bring in new code, and keep in-flight calls whole."""

import asyncio
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aaa.mcp.compile import CompiledServer, acompile_spec
from aaa.mcp.config import load_spec
from aaa.mcp.include import compose
from aaa.mcp.reload import ReloadableServer, fingerprint, watch

HELPERS = Path(__file__).parent.parent
RICH = "helpers.aaa_mcp_server:rich_registry"


def write(path: Path, **over) -> None:
    cfg = {"server": {"name": "live"}, "sources": [{"type": "objects", "ref": RICH}], **over}
    path.write_text(json.dumps(cfg))
    path.chmod(0o644)


def loaders(path: Path):
    return (lambda: load_spec(path, environ={})), acompile_spec


def names(server) -> list[str]:
    return sorted(t.name for t in server.list_tools())


async def start(tmp_path, **over):
    cfg = tmp_path / "s.json"
    write(cfg, **over)
    return cfg, await ReloadableServer.start(*loaders(cfg), min_interval_s=0.0)


# ================================================================ a reload that works
async def test_a_changed_configuration_replaces_the_tools_a_caller_sees(tmp_path):
    cfg, server = await start(tmp_path)
    assert "add" in names(server) and "shout" in names(server)
    write(cfg, tools={"exclude": ["shout"]})
    result = await server.reload()
    assert result.applied and "shout" not in names(server) and "add" in names(server)
    assert (await server.call("add", {"a": 1, "b": 2})).text == "3"
    assert (await server.call("shout", {"text": "x"})).is_error  # gone
    assert server.reloads == 1 and server.name == "live"
    await server.aclose()


async def test_the_server_object_forwards_what_hosts_read(tmp_path):
    cfg, server = await start(tmp_path, prompts=[{"name": "p", "messages": [{"text": "hi"}]}])
    assert server.has_prompts and not server.has_resources and server.title is None
    assert [p.name for p in server.list_prompts()] == ["p"] and server.get_prompt("p").messages == (
        ("user", "hi"),
    )
    assert (await server.aget_prompt("p")).messages == (("user", "hi"),)
    write(cfg, prompts=[{"name": "p", "messages": [{"text": "changed"}]}])
    assert (await server.reload()).applied
    assert server.get_prompt("p").messages == (("user", "changed"),)
    await server.aclose()


# ================================================================ a reload that must not hurt
async def test_a_bad_file_leaves_the_running_server_alone(tmp_path):
    cfg, server = await start(tmp_path)
    before = names(server)
    cfg.write_text("{ not json")
    r = await server.reload()
    assert r.status == "failed" and "could not be read" in r.reason and names(server) == before
    write(cfg, tools={"include": ["nothing_matches_*"]}, profiles={"x": {"tools": ["typo*"]}})
    assert (await server.reload()).status == "failed" and names(server) == before
    write(cfg, tools={"exclude": ["shout"]})
    assert (await server.reload()).applied  # and it recovers when the file is fixed
    await server.aclose()


async def test_a_reload_cannot_change_who_the_server_is_or_what_it_can_do(tmp_path, monkeypatch):
    closed = []
    real = CompiledServer.aclose

    async def spy(self):
        closed.append(self.name)
        await real(self)

    monkeypatch.setattr(CompiledServer, "aclose", spy)
    cfg, server = await start(tmp_path)
    for over, why in (
        ({"server": {"name": "renamed"}}, "name changed"),
        ({"server": {"name": "live", "version": "9"}}, "version changed"),
        ({"server": {"name": "live", "instructions": "new"}}, "instructions changed"),
        ({"resources": [{"type": "static", "name": "r", "text": "t"}]}, "capabilities"),
        ({"prompts": [{"name": "p", "messages": [{"text": "t"}]}]}, "capabilities"),
    ):
        write(cfg, **over)
        r = await server.reload()
        assert r.status == "refused" and why in r.reason and "restart required" in r.reason, (
            over,
            r,
        )
    assert len(closed) == 5  # each refused server was built and then closed again, not leaked
    assert names(server) and server.name == "live" and server.reloads == 0
    await server.aclose()


async def test_a_reload_cannot_bring_in_code_that_was_not_running_at_startup(tmp_path):
    cfg, server = await start(tmp_path)
    for new in (
        [
            {"type": "objects", "ref": RICH},
            {"type": "objects", "ref": "helpers.aaa_mcp_agents:algo"},
        ],
        [{"type": "module", "module": "json", "allow": ["dumps"]}],
        [{"type": "objects", "ref": "os:environ"}],
    ):
        write(cfg, sources=new)
        r = await server.reload()
        assert r.status == "refused" and "not running at startup" in r.reason, new
    write(cfg, sources=[{"type": "objects", "ref": RICH}], tools={"exclude": ["shout"]})
    assert (await server.reload()).applied  # the same code, reconfigured, is fine
    write(cfg, sources=[])
    assert (await server.reload()).applied and names(server) == []  # taking code OUT is fine too
    await server.aclose()


async def test_the_permission_check_runs_again_on_every_reload(tmp_path):
    cfg, server = await start(tmp_path)
    cfg.chmod(0o666)  # world-writable and it imports code: refused at startup, and refused now
    r = await server.reload()
    assert r.status == "failed" and "tools" not in r.reason and names(server)
    await server.aclose()


async def test_a_configuration_that_builds_its_own_engine_is_not_hot_reloadable(tmp_path):
    cfg = tmp_path / "e.json"
    write(
        cfg,
        runtime={"engine": "helpers.aaa_mcp_agents:engine"},
        agents=[
            {
                "name": "ask",
                "algorithm": "helpers.aaa_mcp_agents:algo",
                "description": "d",
                "budget": {"deadline_s": 5},
            }
        ],
    )
    server = await ReloadableServer.start(*loaders(cfg), min_interval_s=0.0)
    r = await server.reload()
    assert r.status == "refused" and "runtime.engine" in r.reason
    await server.aclose()


# ================================================================ in-flight calls, timing, concurrency
async def test_a_call_in_flight_finishes_on_the_server_it_started_on(tmp_path):
    from aaa.tools import ToolRegistry, tool

    @tool
    async def slow(ms: int) -> str:
        """Wait."""
        await asyncio.sleep(ms / 1000)
        return "old server answered"

    spec_a = {"sources": [{"type": "live", "key": "t"}], "server": {"name": "live"}}
    versions = iter([ToolRegistry([slow]), ToolRegistry([])])

    from aaa.mcp.spec import MCPSpec

    async def compile_(spec):
        return await acompile_spec(spec, live={"t": next(versions)})

    server = await ReloadableServer.start(
        lambda: MCPSpec.model_validate(spec_a), compile_, min_interval_s=0.0, grace_s=5
    )
    first = server.current
    inflight = asyncio.create_task(server.call("slow", {"ms": 300}))
    await asyncio.sleep(0.05)
    assert (await server.reload()).applied and names(server) == []  # the new server has no `slow`
    assert (await inflight).text == "old server answered"  # but the call that was running finished
    await asyncio.sleep(0.05)
    assert first._upstreams == [] and id(first) not in server._retired
    await server.aclose()


async def test_reloads_are_rate_limited_and_serialised(tmp_path):
    cfg = tmp_path / "s.json"
    write(cfg)
    server = await ReloadableServer.start(*loaders(cfg), min_interval_s=60.0)
    assert (await server.reload()).applied
    again = await server.reload()
    assert again.status == "skipped"
    assert (await server.reload(force=True)).applied  # an explicit force passes
    many = await asyncio.gather(*[server.reload(force=True) for _ in range(4)])
    assert all(r.status == "applied" for r in many) and server.reloads == 6
    await server.aclose()


# ================================================================ the watcher
async def test_the_watcher_reloads_when_files_change_and_only_after_the_change_settles(tmp_path):
    cfg, server = await start(tmp_path)
    results = []
    task = asyncio.create_task(
        watch(server, lambda: compose(cfg).files, interval_s=0.05, on_result=results.append)
    )
    try:
        await asyncio.sleep(0.2)
        assert results == []  # nothing changed, nothing happened
        write(cfg, tools={"exclude": ["shout"]})
        for _ in range(100):
            if results:
                break
            await asyncio.sleep(0.05)
        assert [r.status for r in results] == ["applied"] and "shout" not in names(server)
        await asyncio.sleep(0.3)
        assert len(results) == 1  # an unchanged file is not reloaded again
        cfg.write_text("{ half written")
        await asyncio.sleep(0.04)
        write(
            cfg, tools={"exclude": ["shout", "boom"]}
        )  # fixed before the next poll confirmed the bad content
        await asyncio.sleep(0.4)
        assert all(r.status == "applied" for r in results)
    finally:
        task.cancel()
        await server.aclose()


def test_the_fingerprint_follows_content_and_survives_missing_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("1")
    first = fingerprint([a, b])
    assert first == fingerprint([b, a]) and fingerprint([a, b]) == first
    a.write_text("2")
    assert fingerprint([a, b]) != first


# ================================================================ the real thing over stdio
@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="needs SIGHUP")
def test_sighup_reloads_a_real_stdio_server(tmp_path):
    pytest.importorskip("mcp")
    cfg = tmp_path / "s.json"
    write(cfg)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": f"{HELPERS}:{HELPERS.parent / 'src'}"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "aaa.cli", "mcp", "serve", "--config", str(cfg), "--reload"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=tmp_path,
    )

    def rpc(method, params=None, id_=1):
        proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}})
            + "\n"
        )
        proc.stdin.flush()
        ready, _, _ = select.select([proc.stdout], [], [], 30)
        assert ready, "no answer"
        return json.loads(proc.stdout.readline())

    try:
        rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        )
        proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        proc.stdin.flush()
        before = {t["name"] for t in rpc("tools/list", id_=2)["result"]["tools"]}
        assert "shout" in before
        write(cfg, tools={"exclude": ["shout"]})
        proc.send_signal(signal.SIGHUP)
        for i in range(60):
            after = {t["name"] for t in rpc("tools/list", id_=10 + i)["result"]["tools"]}
            if "shout" not in after:
                break
            time.sleep(0.1)
        assert "shout" not in after and "add" in after
    finally:
        proc.kill()
        proc.wait()
