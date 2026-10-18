"""The builder, TOML/YAML/JSON round trips, and algorithms exposed as MCP tools."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from aaa.algos import ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.mcp import MCPServerBuilder
from aaa.mcp.compile import acompile_spec
from aaa.mcp.config import SpecError, dump_spec, load_spec
from aaa.mcp.sources import agents as agents_mod
from aaa.mcp.spec import MCPSpec
from aaa.mcp.templating import render, template_fields
from aaa.runtime import Engine
from aaa.tools import ToolRegistry, tool
from helpers.aaa_mcp_server import rich_registry

HELPERS = Path(__file__).parent.parent / "helpers"


async def make_engine(script=None) -> Engine:
    eng = Engine()
    LLMHandler(FakeLLMClient(script or (lambda r: text("42")))).attach(eng)
    return eng


# ===================================================================== builder and round trips
def builder():
    return (
        MCPServerBuilder("desk")
        .title("Support desk")
        .instructions("Be careful.")
        .tenant("acme", "ann")
        .add_ref("helpers.aaa_mcp_server:rich_registry", tags=["core"], prefix="")
        .include("add", "loc*", "shout")
        .exclude("shout")
        .rename("add", "plus")
        .prefix("d_")
        .override("d_plus", title="Plus", read_only=True)
        .limits(max_result_chars=5000)
        .agent(
            "research",
            algorithm="helpers.aaa_mcp_agents:algo",
            description="Research it.",
            deadline_s=30,
            max_steps=5,
            read_only=True,
        )
        .engine("helpers.aaa_mcp_agents:engine")
    )


def test_the_builder_produces_a_validated_spec():
    s = builder().spec()
    assert (
        s.server.title == "Support desk"
        and s.tenancy.user == "ann"
        and s.tools.include == ["add", "loc*", "shout"]
    )
    assert (
        s.tools.rename == {"add": "plus"}
        and s.agents[0].budget.max_steps == 5
        and s.runtime.engine == "helpers.aaa_mcp_agents:engine"
    )
    assert (
        s.tools.overrides["d_plus"].annotations is not None
        and s.tools.overrides["d_plus"].annotations.read_only is True
    )


@pytest.mark.parametrize("ext", [".json", ".toml", ".yaml"])
def test_a_spec_survives_writing_and_reading(tmp_path, ext):
    if ext == ".yaml":
        pytest.importorskip("yaml")
    s = builder().spec()
    path = tmp_path / f"s{ext}"
    path.write_text(dump_spec(s, path))
    path.chmod(0o644)
    assert load_spec(path, environ={}) == s


def test_toml_writer_handles_awkward_keys_and_values(tmp_path):
    s = MCPSpec.model_validate(
        {
            "server": {"name": "x", "instructions": 'quote " and \\ and\nnewline and ünicode'},
            "tools": {
                "overrides": {
                    "d_x": {
                        "output_schema": {
                            "type": "object",
                            "properties": {"$ref-ish": {"type": "string"}},
                        }
                    }
                }
            },
            "agents": [
                {
                    "name": "a",
                    "algorithm": "m:a",
                    "description": "d",
                    "budget": {"deadline_s": 1.5},
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                    "input_template": {"task": "{q}", "n": [1, 2]},
                }
            ],
        }
    )
    p = tmp_path / "s.toml"
    p.write_text(dump_spec(s, p))
    assert load_spec(p, environ={}) == s


def test_a_builder_holding_live_objects_cannot_be_written_but_still_compiles():
    b = MCPServerBuilder("x").add_tools(rich_registry())
    with pytest.raises(SpecError, match="live Python objects"):
        b.dump()
    assert "add" in {t.name for t in b.compile().list_tools()}


async def test_the_builder_compiles_what_the_file_would():
    core = await (
        MCPServerBuilder("desk")
        .add_tools(rich_registry())
        .exclude("b*", "shout")
        .prefix("t_")
        .override("t_add", title="Add")
        .acompile()
    )
    assert sorted(t.name for t in core.list_tools()) == ["t_add", "t_locate"]


# ===================================================================== templates
def test_templates_only_allow_plain_placeholders():
    assert template_fields({"task": "{q}", "n": ["{a}", {"x": "b {c}"}], "k": 3}) == {"q", "a", "c"}
    for bad in ("{q.__class__}", "{q[0]}", "{q!r}", "{q:>10}", "{0}", "{}"):
        with pytest.raises(SpecError, match="not allowed"):
            template_fields({"task": bad})


def test_rendering_keeps_the_type_of_a_lone_placeholder():
    args = {"q": "why?", "n": 3, "tags": ["a", "b"], "obj": {"k": 1}}
    assert render({"task": "{q}", "limit": "{n}", "tags": "{tags}", "o": "{obj}"}, args) == {
        "task": "why?",
        "limit": 3,
        "tags": ["a", "b"],
        "o": {"k": 1},
    }
    assert render("Q: {q} / {n} / {tags}", args) == 'Q: why? / 3 / ["a", "b"]'
    assert render({"nested": [{"x": "{q}!"}], "plain": 7}, args) == {
        "nested": [{"x": "why?!"}],
        "plain": 7,
    }
    assert render("{{literal}} {q}", args) == "{literal} why?"


# ===================================================================== agents as tools
async def agent_core(**agent_kw):
    eng = await make_engine()
    algo = ToolLoop(ToolRegistry([]))
    b = (
        MCPServerBuilder("desk")
        .engine(eng)
        .agent("ask", algorithm=algo, description="Ask the agent.", deadline_s=10, **agent_kw)
    )
    return await b.acompile(), eng


async def test_an_algorithm_becomes_a_tool_with_its_own_schema_and_hints():
    core, eng = await agent_core(read_only=True, open_world=True)
    async with eng:
        (t,) = core.list_tools()
        assert (
            t.name == "ask"
            and t.input_schema["required"] == ["input"]
            and t.annotations is not None
        )
        assert t.annotations.read_only is True and t.annotations.open_world is True
        r = await core.call("ask", {"input": "what is 6 x 7?"})
        assert r.text == "42" and not r.is_error
        bad = await core.call("ask", {"nope": 1})
        assert bad.is_error and "invalid arguments" in bad.text


async def test_the_json_output_mode_returns_the_whole_outcome():
    core, eng = await agent_core(output="json")
    async with eng:
        r = json.loads((await core.call("ask", {"input": "x"})).text)
    assert r["kind"] == "success" and r["value"]["text"] == "42"


async def test_calls_run_as_the_hosts_tenant_and_user_with_the_configured_budget():
    core, eng = await agent_core(max_steps=7)
    async with eng:
        await core.call("ask", {"input": "x"}, tenant="acme", user="ann")
        (info,) = await eng.store.run_index()
    assert info.started.tenant == "acme" and info.started.meta == {"user": "ann"}
    assert (
        info.started.data["budget"]["max_steps"] == 7
        and info.started.data["budget"]["deadline_s"] == 10
    )


async def test_a_custom_schema_and_template_shape_the_algorithms_input():
    seen = []

    def script(req):
        seen.append(next(m.content for m in reversed(req.messages) if m.role == "user"))
        return text("ok")

    eng = await make_engine(script)
    schema = {
        "type": "object",
        "properties": {"topic": {"type": "string"}, "depth": {"type": "integer"}},
        "required": ["topic"],
        "additionalProperties": False,
    }
    core = await (
        MCPServerBuilder()
        .engine(eng)
        .agent(
            "brief",
            algorithm=ToolLoop(ToolRegistry([])),
            description="d",
            deadline_s=5,
            input_schema=schema,
            input_template={"task": "Brief me on {topic} (depth {depth})"},
        )
        .acompile()
    )
    async with eng:
        assert not (await core.call("brief", {"topic": "tides", "depth": 2})).is_error
        assert not (await core.call("brief", {"topic": "tides"})).is_error
    assert seen == ["Brief me on tides (depth 2)", "Brief me on tides (depth )"]


async def test_a_failing_agent_is_an_error_not_a_crash():
    eng = await make_engine(lambda r: text("never ends"))
    from aaa.llm.fake import tool_call

    loop = ToolLoop(ToolRegistry([]), max_turns=2)
    eng2 = await make_engine(lambda r: tool_call("missing_tool"))
    core = (
        await MCPServerBuilder()
        .engine(eng2)
        .agent("loopy", algorithm=loop, description="d", deadline_s=5, max_steps=3)
        .acompile()
    )
    async with eng2:
        r = await core.call("loopy", {"input": "x"})
    assert r.is_error and "the agent ended" in r.text
    _ = eng


async def test_the_concurrency_cap_refuses_instead_of_piling_up(monkeypatch):
    monkeypatch.setattr(agents_mod, "BUSY_WAIT_S", 0.05)
    sys.path.insert(0, str(HELPERS.parent))
    from helpers.aaa_mcp_agents import slow

    eng = Engine()
    core = await (
        MCPServerBuilder()
        .engine(eng)
        .limits(max_concurrent_agents=1)
        .agent("slow", algorithm=slow, description="d", deadline_s=5)
        .acompile()
    )
    async with eng:
        first = asyncio.create_task(core.call("slow", {"input": "a"}))
        await asyncio.sleep(0.05)
        second = await core.call("slow", {"input": "b"})
        assert second.is_error and "as many agents as it allows" in second.text
        assert (await first).text == "slow done"
        assert (
            await core.call("slow", {"input": "c"})
        ).text == "slow done"  # the slot was released


async def test_agent_mistakes_are_loud_at_compile_time():
    eng = await make_engine()
    algo = ToolLoop(ToolRegistry([]))
    base = {"name": "a", "algorithm": "k", "description": "d", "budget": {"deadline_s": 5}}

    async def build(**over):
        spec = MCPSpec.model_validate({"agents": [{**base, **over}]})
        return await acompile_spec(spec, live={"k": algo}, engine=eng)

    with pytest.raises(SpecError, match="not properties of its input_schema"):
        await build(input_template={"task": "{missing}"})
    with pytest.raises(SpecError, match="not allowed"):
        await build(input_template={"task": "{input.__class__}"})
    with pytest.raises(SpecError, match="must be a JSON schema of type object"):
        await build(input_schema={"type": "string"})
    with pytest.raises(SpecError, match="not an algorithm"):
        await acompile_spec(
            MCPSpec.model_validate({"agents": [base]}), live={"k": object()}, engine=eng
        )
    with pytest.raises(SpecError, match="no algorithm 'k'"):
        await acompile_spec(MCPSpec.model_validate({"agents": [base]}), engine=eng)
    with pytest.raises(SpecError, match="need an engine"):
        await acompile_spec(MCPSpec.model_validate({"agents": [base]}), live={"k": algo})
    with pytest.raises(
        Exception, match="deadline_s"
    ):  # a budget without a deadline is refused by the model itself
        MCPSpec.model_validate({"agents": [{**base, "budget": {"max_steps": 3}}]})
    with pytest.raises(Exception, match="Extra inputs"):
        MCPSpec.model_validate(
            {"agents": [{**base, "tenant": "x"}]}
        )  # a tenant is never part of an agent tool


async def test_agent_tools_follow_the_same_naming_and_filtering_rules():
    core, eng = await agent_core()
    async with eng:
        spec = core.spec.model_copy(
            update={"tools": core.spec.tools.model_copy(update={"prefix": "x_"})}
        )
        again = await acompile_spec(
            spec, live={"agent_ask": ToolLoop(ToolRegistry([]))}, engine=eng
        )
        assert [t.name for t in again.list_tools()] == ["x_ask"]


# ===================================================================== timeouts and the stdio path
async def test_a_tool_that_outlives_its_timeout_is_reported_not_awaited_forever():
    @tool
    async def hang() -> str:
        """Never returns in time."""
        await asyncio.sleep(5)
        return "late"

    hang.timeout_s = 0.05
    core = await MCPServerBuilder().add_tools(ToolRegistry([hang])).acompile()
    r = await core.call("hang")
    assert r.is_error and "timed out" in r.text


async def test_agents_run_over_real_stdio_with_an_engine_the_configuration_builds(tmp_path):
    pytest.importorskip("mcp")
    from aaa.mcp import MCPClient

    cfg = tmp_path / "s.json"
    cfg.write_text(
        json.dumps(
            {
                "runtime": {"engine": "helpers.aaa_mcp_agents:engine"},
                "agents": [
                    {
                        "name": "ask",
                        "algorithm": "helpers.aaa_mcp_agents:algo",
                        "description": "Ask.",
                        "budget": {"deadline_s": 20},
                    }
                ],
            }
        )
    )
    cfg.chmod(0o644)
    env = {"PYTHONPATH": f"{HELPERS.parent}:{HELPERS.parent.parent / 'src'}"}
    async with MCPClient.stdio(
        sys.executable,
        ["-m", "aaa.cli", "mcp", "serve", "--config", str(cfg)],
        name="g",
        cwd=str(HELPERS.parent),
        env=env,
    ) as c:
        assert [t.name for t in await c.list_tools()] == ["ask"]
        out, is_error = await c.call("ask", {"input": "hello"})
        assert out == "answer to: hello" and not is_error


async def test_algorithms_from_the_newer_families_can_be_exposed_as_tools():
    spec = MCPSpec.model_validate(
        {
            "runtime": {"engine": "helpers.aaa_mcp_agents:engine"},
            "agents": [
                {
                    "name": "pick_best",
                    "algorithm": "helpers.aaa_mcp_agents:best_of",
                    "description": "Best of three.",
                    "budget": {"deadline_s": 20},
                },
                {
                    "name": "blend",
                    "algorithm": "helpers.aaa_mcp_agents:mixture",
                    "description": "A mixture.",
                    "budget": {"deadline_s": 20},
                },
            ],
        }
    )
    core = await acompile_spec(spec)
    assert {t.name for t in core.list_tools()} == {"pick_best", "blend"}
    async with (
        core.engine
    ):  # the engine the configuration built runs for as long as the server does
        for name in ("pick_best", "blend"):
            r = await core.call(name, {"input": "hello"})
            assert not r.is_error and "answer to: hello" in r.text, (name, r.text)
