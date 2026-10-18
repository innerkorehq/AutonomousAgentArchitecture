"""Another MCP server as a source: approval defaults, minimal environment, kill-switches, a dead upstream, cleanup, and the HTTP guard."""

import asyncio
import json
import os
import signal
import socket
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from aaa.mcp.compile import acompile_spec
from aaa.mcp.config import SpecError, load_spec
from aaa.mcp.spec import MCPSpec

HELPERS = Path(__file__).parent.parent / "helpers"
NOTES = [str(HELPERS / "mcp_server.py")]
ENVSRV = [str(HELPERS / "mcp_env_server.py")]


def stdio(name="notes", args=None, **kw):
    return {
        "type": "mcp",
        "name": name,
        "transport": "stdio",
        "command": sys.executable,
        "args": args or NOTES,
        **kw,
    }


async def build(*sources, **over):
    return await acompile_spec(MCPSpec.model_validate({"sources": list(sources), **over}))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


async def gone(pid: int, seconds: float = 8.0) -> bool:
    end = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < end:
        if not alive(pid):
            return True
        await asyncio.sleep(0.1)
    return False


def names(core):
    return sorted(t.name for t in core.list_tools())


# ===================================================================== approval and filtering
async def test_an_upstreams_tools_require_approval_unless_it_says_they_are_read_only():
    strict = await build(stdio())
    try:
        assert (
            names(strict) == [] and len(strict.hidden) >= 4
        )  # nothing is trusted by default: all hidden
        assert any("hidden because they require approval" in line for line in strict.report)
    finally:
        await strict.aclose()
    trusted = await build(stdio(trust_annotations=True))
    try:
        assert names(trusted) == ["search_notes"]  # only the tool the upstream marks read-only
        assert set(trusted.hidden) >= {"delete_note", "set_title", "explode"}
        t = trusted.list_tools()[0]
        assert t.annotations is not None and t.annotations.read_only is True
    finally:
        await trusted.aclose()
    open_ = await build(stdio(approval="none"))
    try:
        assert {"search_notes", "delete_note", "set_title"} <= set(names(open_))
    finally:
        await open_.aclose()


async def test_calls_go_through_prefix_filters_caps_and_overrides():
    core = await build(
        stdio(
            approval="none",
            prefix="n_",
            exclude=["delete_*", "explode"],
            description_overrides={"search_notes": "Search the team's notes."},
            max_result_chars=1_000,
        ),
    )
    try:
        assert "n_delete_note" not in names(core) and "n_search_notes" in names(core)
        d = {t.name: t.description for t in core.list_tools()}
        assert d["n_search_notes"] == "Search the team's notes."
        r = await core.call("n_search_notes", {"query": "roadmap", "limit": 2})
        assert not r.is_error and "note about roadmap #0" in r.text
        big = await core.call("n_big", {"n": 50_000})
        assert (
            not big.is_error and len(big.text) < 1_300 and "truncated" in big.text
        )  # an upstream cannot flood the caller
        bad = await core.call("n_search_notes", {"query": 5})
        assert (
            bad.is_error and "invalid arguments" in bad.text
        )  # validated against the upstream's schema before it is called
    finally:
        await core.aclose()


async def test_upstream_tools_join_other_sources_under_one_policy():
    objs = {"type": "objects", "ref": "helpers.aaa_mcp_server:rich_registry"}
    core = await build(
        objs,
        stdio(approval="none", prefix="n_", include=["search_notes"]),
        tools={"exclude": ["broken_*"]},
    )
    try:
        assert {"add", "locate", "n_search_notes"} <= set(names(core))
        with pytest.raises(SpecError, match="two tools are both called"):
            await build(
                stdio(approval="none", include=["search_notes"]),
                stdio("other", approval="none", include=["search_notes"]),
            )
    finally:
        await core.aclose()


# ===================================================================== the environment and the process
async def test_the_upstream_gets_only_the_environment_you_name(monkeypatch):
    monkeypatch.setenv("PARENT_SECRET", "must-not-leak")
    monkeypatch.setenv("WANTED_TOKEN", "wanted-value")
    core = await build(
        stdio("envsrv", ENVSRV, approval="none", env={"LITERAL": "1"}, env_from=["WANTED_TOKEN"])
    )
    try:
        listed = (await core.call("env_names")).text.split(",")
        assert "LITERAL" in listed and "WANTED_TOKEN" in listed and "PARENT_SECRET" not in listed
        assert "must-not-leak" not in (await core.call("env_value", {"name": "PARENT_SECRET"})).text
        assert "wanted-value" in (await core.call("env_value", {"name": "WANTED_TOKEN"})).text
    finally:
        await core.aclose()
    with pytest.raises(SpecError, match="env_from names NOT_SET_ANYWHERE"):
        await build(stdio("envsrv", ENVSRV, env_from=["NOT_SET_ANYWHERE"]))


async def test_the_subprocess_kill_switch_stops_everything_before_a_process_starts(
    tmp_path, monkeypatch
):
    marker = tmp_path / "spawned.txt"
    src = {
        "type": "mcp",
        "name": "x",
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-c", f"open({str(marker)!r}, 'w').write('ran')"],
    }
    monkeypatch.setenv("AAA_MCP_ALLOW_SUBPROCESS", "0")
    with pytest.raises(SpecError, match="launching is disabled"):
        await build(src)
    monkeypatch.delenv("AAA_MCP_ALLOW_SUBPROCESS")
    with pytest.raises(SpecError, match="launching is disabled"):
        await acompile_spec(MCPSpec.model_validate({"sources": [src]}), allow_subprocess=False)
    assert not marker.exists()


async def test_an_unavailable_upstream_is_an_error_unless_it_is_optional():
    missing = {
        "type": "mcp",
        "name": "ghost",
        "transport": "stdio",
        "command": "definitely-not-a-command-xyz",
        "connect_timeout_s": 5,
    }
    with pytest.raises(SpecError, match="cannot connect to the upstream"):
        await build(missing)
    core = await build(
        {**missing, "optional": True},
        {"type": "objects", "ref": "helpers.aaa_mcp_server:rich_registry"},
    )
    try:
        assert "add" in names(core) and any(
            "ghost" in line and "not available" in line for line in core.report
        )
    finally:
        await core.aclose()


async def test_closing_stops_the_process_and_a_failed_compile_leaves_nothing_running():
    core = await build(stdio("envsrv", ENVSRV, approval="none"))
    pid = int((await core.call("pid")).text)
    assert alive(pid)
    await core.aclose()
    assert await gone(pid)

    with pytest.raises(SpecError):
        await build(
            stdio("envsrv", ENVSRV, approval="none"),
            {"type": "objects", "ref": "no_such_module_xyz:thing"},
        )
    # the first upstream was started and then the compile failed: it must have been closed (nothing to look up its pid with,
    # so look for leftover processes running the helper)
    import subprocess

    out = subprocess.run(  # noqa: ASYNC221
        ["ps", "-ax", "-o", "command="], capture_output=True, text=True, check=False
    ).stdout
    assert out.count("mcp_env_server.py") == 0, "an upstream process was left running"


async def test_a_dead_upstream_gives_clean_errors_and_the_server_keeps_serving():
    core = await build(
        stdio("envsrv", ENVSRV, approval="none", prefix="u_"),
        {"type": "objects", "ref": "helpers.aaa_mcp_server:rich_registry"},
    )
    try:
        pid = int((await core.call("u_pid")).text)
        os.kill(pid, signal.SIGKILL)
        assert await gone(pid)
        await asyncio.sleep(0.5)
        r = await core.call("u_echo", {"text": "hi"})
        assert r.is_error and "hunter" not in r.text and len(r.text) < 300
        assert (
            await core.call("add", {"a": 1, "b": 2})
        ).text == "3"  # the rest of the server is unaffected
        assert "u_echo" in names(core)
    finally:
        await core.aclose()


# ===================================================================== configuration is checked
@pytest.mark.parametrize(
    ("src", "why"),
    [
        ({"transport": "stdio"}, "needs a command"),
        (
            {"transport": "stdio", "command": "x", "url": "https://h/x"},
            "apply to http upstreams only",
        ),
        ({"transport": "http"}, "needs a url"),
        ({"transport": "http", "url": "https://h/x"}, "needs allowed_hosts"),
        (
            {"transport": "http", "url": "https://h/x", "allowed_hosts": ["h"], "command": "x"},
            "apply to stdio upstreams only",
        ),
        (
            {
                "transport": "http",
                "url": "https://h/x",
                "allowed_hosts": ["h"],
                "headers": {"Authorization": "Bearer sk-live-abc"},
            },
            "never the value itself",
        ),
        (
            {
                "transport": "http",
                "url": "https://h/x",
                "allowed_hosts": ["h"],
                "headers": {"Host": "env:X"},
            },
            "cannot be set",
        ),
        (
            {"transport": "stdio", "command": "x", "env_from": ["bad name"]},
            "not a valid environment variable name",
        ),
        ({"transport": "stdio", "command": "x", "env": {"A": "x" * 600}}, "limited to 500"),
    ],
)
def test_upstream_configuration_mistakes_are_refused(src, why):
    with pytest.raises(Exception, match=why):
        MCPSpec.model_validate({"sources": [{"type": "mcp", "name": "u", **src}]})


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_a_configuration_that_launches_a_process_must_not_be_world_writable(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"sources": [stdio()]}))
    f.chmod(0o666)
    with pytest.raises(SpecError, match="writable by everyone"):
        load_spec(f, environ={})


# ===================================================================== http upstreams
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def http_upstream():
    import uvicorn

    from helpers.mcp_server import build as build_notes

    notes, _ = build_notes()
    inner = notes.streamable_http_app()
    seen: list[dict[str, str]] = []

    async def spy(scope, receive, send):
        if scope["type"] == "http":
            seen.append({k.decode(): v.decode() for k, v in scope["headers"]})
        await inner(scope, receive, send)

    # run the app's own lifespan (the SDK needs it for its session manager) by letting uvicorn drive `inner`'s lifespan
    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            await inner(scope, receive, send)
        else:
            await spy(scope, receive, send)

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    yield port, seen
    server.should_exit = True
    await task


def http_src(port, **kw):
    return {
        "type": "mcp",
        "name": "web",
        "transport": "http",
        "url": f"http://127.0.0.1:{port}/mcp",
        "allowed_hosts": ["127.0.0.1"],
        "allow_insecure_loopback": True,
        "approval": "none",
        "connect_timeout_s": 10,
        **kw,
    }


async def test_an_http_upstream_works_through_the_guarded_transport(http_upstream, monkeypatch):
    port, seen = http_upstream
    monkeypatch.setenv("UPSTREAM_TOKEN", "tok-from-env")
    core = await build(http_src(port, headers={"Authorization": "Bearer ${UPSTREAM_TOKEN}"}))
    try:
        r = await core.call("search_notes", {"query": "x", "limit": 1})
        assert not r.is_error and "note about x" in r.text
        assert any(
            h.get("authorization") == "Bearer tok-from-env" for h in seen
        )  # the credential reached the upstream
        assert "tok-from-env" not in json.dumps(core.report) + json.dumps(
            core.spec.model_dump(mode="json")
        )
    finally:
        await core.aclose()


async def test_http_upstream_mistakes_are_refused_before_any_connection(http_upstream, monkeypatch):
    port, seen = http_upstream
    monkeypatch.delenv("UNSET_VAR", raising=False)
    with pytest.raises(SpecError, match="plain http is only allowed to loopback"):
        await build(http_src(port, allow_insecure_loopback=False))
    with pytest.raises(SpecError, match="not in allowed_hosts"):
        await build(http_src(port, allowed_hosts=["api.example.com"]))
    with pytest.raises(SpecError, match="UNSET_VAR is not set"):
        await build(http_src(port, headers={"Authorization": "env:UNSET_VAR"}))
    assert seen == []  # nothing connected


async def test_a_private_address_needs_an_explicit_opt_in(http_upstream):
    port, seen = http_upstream
    src = http_src(
        port, url=f"https://127.0.0.1:{port}/mcp", allow_insecure_loopback=False
    )  # https to a loopback address, no opt-in
    with pytest.raises(SpecError, match="cannot connect to the upstream"):
        await build(src)
    assert seen == []  # the guard refused before a single byte reached the upstream


async def test_an_http_upstream_that_redirects_is_not_followed():
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import RedirectResponse
    from starlette.routing import Route

    hits: list[str] = []
    other_port, port = free_port(), free_port()

    async def leak(request):
        hits.append("second host reached")
        return RedirectResponse("/")

    async def redirect(request):
        return RedirectResponse(f"http://127.0.0.1:{other_port}/leak", status_code=307)

    servers = [
        uvicorn.Server(
            uvicorn.Config(
                Starlette(routes=[Route("/mcp", redirect, methods=["GET", "POST"])]),
                host="127.0.0.1",
                port=port,
                log_level="error",
            )
        ),
        uvicorn.Server(
            uvicorn.Config(
                Starlette(routes=[Route("/leak", leak, methods=["GET", "POST"])]),
                host="127.0.0.1",
                port=other_port,
                log_level="error",
            )
        ),
    ]
    tasks = [asyncio.create_task(s.serve()) for s in servers]
    while not all(s.started for s in servers):
        await asyncio.sleep(0.02)
    try:
        with pytest.raises(SpecError, match="cannot connect to the upstream"):
            await build(http_src(port, allowed_hosts=["127.0.0.1"], connect_timeout_s=5))
        assert hits == []
    finally:
        for s in servers:
            s.should_exit = True
        await asyncio.gather(*tasks)


async def test_the_guard_is_the_same_on_every_sdk_generation():
    from aaa.mcp.sources.mcp_upstream import _httpx_flavour

    hx = _httpx_flavour()
    assert hx.__name__ in ("httpx", "httpx2")
    assert (hx.__name__ == "httpx2") == hasattr(
        mcp, "Client"
    )  # SDK 2.x uses its own HTTP stack; 1.x uses httpx


def test_the_command_line_can_refuse_subprocess_sources(tmp_path):
    import subprocess

    cfg = tmp_path / "s.json"
    cfg.write_text(json.dumps({"sources": [stdio(approval="none")]}))
    cfg.chmod(0o644)
    env = {**os.environ, "PYTHONPATH": f"{HELPERS.parent}:{HELPERS.parent.parent / 'src'}"}
    base = [sys.executable, "-m", "aaa.cli", "mcp", "check", "--config", str(cfg)]
    ok = subprocess.run(base, capture_output=True, text=True, env=env, timeout=60, check=False)
    assert ok.returncode == 0 and "upstream 'notes'" in ok.stdout
    refused = subprocess.run(
        [*base, "--no-subprocess-sources"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert refused.returncode == 2 and "launching is disabled" in refused.stderr


# ===================================================================== resources and prompts of an upstream
async def test_nothing_of_an_upstream_but_its_tools_is_passed_on_unless_asked():
    core = await build(stdio(approval="none"))
    try:
        assert core.list_resources() == [] and core.list_prompts() == []
    finally:
        await core.aclose()


async def test_listed_text_resources_are_re_exposed_under_a_namespaced_uri_and_read_through():
    from aaa.mcp.sources.resources import ResourceNotFound

    core = await build(stdio(resources=True, approval="none"))
    try:
        listed = {r.name: r for r in core.list_resources()}
        assert {"welcome", "readme", "huge"} <= set(listed)
        assert all(r.uri.startswith("aaa://upstream/notes/") for r in listed.values())
        assert core.list_resource_templates() == []  # the upstream's template is not passed on
        welcome = await core.read_resource(listed["welcome"].uri)
        assert (
            welcome.text == "Welcome to the notes server" and welcome.uri == listed["welcome"].uri
        )
        readme = await core.read_resource(listed["readme"].uri)
        assert readme.mime_type == "text/markdown" and "# Notes" in readme.text
    finally:
        await core.aclose()
    # the upstream's own URI cannot be asked for, directly or by forging a token
    from aaa.mcp.sources.mcp_passthrough import token_of

    core = await build(stdio(resources=True, approval="none"))
    try:
        for uri in (
            f"aaa://upstream/notes/{token_of('note://item/1')}",  # a real upstream URI that was never listed
            f"aaa://upstream/notes/{token_of('file:///etc/passwd')}",
            "aaa://upstream/notes/not-a-token!",
            "aaa://upstream/other/anything",
            f"aaa://upstream/notes/{token_of('note://welcome')}?x=1",
            "note://welcome",
        ):
            with pytest.raises(ResourceNotFound):
                await core.read_resource(uri)
    finally:
        await core.aclose()


async def test_resource_size_exclusion_and_a_dead_upstream():
    from aaa.mcp.sources.resources import ResourceNotFound

    core = await build(
        stdio(resources=True, approval="none", max_resource_bytes=1000, resource_exclude=["readme"])
    )
    try:
        listed = {r.name: r for r in core.list_resources()}
        assert "readme" not in listed
        huge = await core.read_resource(listed["huge"].uri)
        assert len(huge.text) < 1200 and "[truncated at 1000 bytes]" in huge.text
        # the upstream goes away: a read is the same "not found" as any other failure
        await core._upstreams[0].aclose()
        with pytest.raises(ResourceNotFound):
            await core.read_resource(listed["welcome"].uri)
    finally:
        await core.aclose()


async def test_upstream_prompts_keep_text_validate_arguments_and_fail_generically():
    from aaa.mcp.sources.prompts import PromptError

    core = await build(stdio(prompts=True, approval="none", prefix="n_"))
    try:
        prompts = {p.name: p for p in core.list_prompts()}
        assert set(prompts) == {"n_summarize", "n_review"}
        assert {(a.name, a.required) for a in prompts["n_summarize"].arguments} == {
            ("text", True),
            ("style", False),
        }
        got = await core.aget_prompt("n_summarize", {"text": "the cat sat", "style": "long"})
        assert got.messages == (("user", "Summarize in a long style: the cat sat"),)
        with pytest.raises(PromptError, match="unknown argument"):
            await core.aget_prompt("n_summarize", {"text": "x", "evil": "y"})
        with pytest.raises(PromptError, match="missing required"):
            await core.aget_prompt("n_review", {})
        with pytest.raises(PromptError, match="unknown prompt"):
            await core.aget_prompt("summarize", {})  # the unprefixed name is not offered
        with pytest.raises(PromptError, match="asynchronously"):
            core.get_prompt("n_summarize", {"text": "x"})
        big = await core.aget_prompt("n_review", {"code": "c" * 50_000})
        assert len(big.messages[0][1]) <= 20_000
        await core._upstreams[0].aclose()
        with pytest.raises(PromptError, match="not available"):
            await core.aget_prompt("n_review", {"code": "x"})
    finally:
        await core.aclose()


async def test_prompt_name_clashes_are_refused_and_exclusions_apply():
    with pytest.raises(SpecError, match="defined here and offered by an upstream"):
        await build(
            stdio(prompts=True, approval="none"),
            prompts=[{"name": "review", "messages": [{"text": "mine"}]}],
        )
    with pytest.raises(SpecError, match="offered by two upstream sources"):
        await build(
            stdio(prompts=True, approval="none"), stdio("other", prompts=True, approval="none")
        )
    core = await build(stdio(prompts=True, approval="none", exclude=["review"]))
    try:
        assert [p.name for p in core.list_prompts()] == ["summarize"]
    finally:
        await core.aclose()


async def test_upstream_prompts_are_served_over_the_http_handler_too():
    from aaa_server.mcp_http import handle

    core = await build(stdio(prompts=True, resources=True, approval="none"))
    try:
        r = await handle(
            core,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "prompts/get",
                "params": {"name": "review", "arguments": {"code": "x = 1"}},
            },
            tenant="acme",
            user="ann",
        )
        assert r["result"]["messages"][0]["content"]["text"] == "Review this:\nx = 1"
        listed = await handle(
            core, {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}, tenant="acme", user="ann"
        )
        assert any(
            x["uri"].startswith("aaa://upstream/notes/") for x in listed["result"]["resources"]
        )
        bad = await handle(
            core,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompts/get",
                "params": {"name": "review", "arguments": {}},
            },
            tenant="acme",
            user="ann",
        )
        assert "error" in bad
    finally:
        await core.aclose()
