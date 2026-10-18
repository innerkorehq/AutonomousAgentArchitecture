"""Profiles: which tools, resources and prompts an audience may see, chosen only by authenticated scopes (or the stdio operator), failing closed."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aaa.mcp.compile import acompile_spec, describe
from aaa.mcp.config import SpecError
from aaa.mcp.policy import Profile, ProfileSet
from aaa.mcp.spec import MCPSpec
from helpers.aaa_mcp_server import rich_registry

BASE = {
    "sources": [{"type": "live", "key": "t"}],
    "resources": [
        {"type": "static", "name": "readme", "text": "public docs"},
        {"type": "static", "name": "runbook", "text": "internal runbook"},
    ],
    "prompts": [
        {"name": "greet", "messages": [{"text": "Hello"}]},
        {"name": "escalate", "messages": [{"text": "Escalate"}]},
    ],
    "profiles": {
        "support": {"tools": ["add", "shout"], "resources": ["*readme*"], "prompts": ["greet"]},
        "admin": {"tools": ["*"], "resources": ["*"], "prompts": ["*"]},
    },
}


async def build(**over):
    spec = MCPSpec.model_validate({**BASE, **over})
    return await acompile_spec(spec, live={"t": rich_registry()})


def names(core, scopes):
    return sorted(t.name for t in core.list_tools(scopes=scopes))


# ================================================================ the set itself
def test_active_profiles_come_from_scopes_and_fail_closed():
    ps = ProfileSet({"a": Profile(tools=("x*",)), "b": Profile(tools=("y",))})
    assert ps.active(frozenset({"run"})) is None  # no profile scope: unrestricted
    assert ps.active(frozenset({"profile:a", "run"})) == {"a"}
    assert ps.active(frozenset({"profile:a", "profile:b"})) == {"a", "b"}  # the union
    assert (
        ps.active(frozenset({"profile:ghost"})) == frozenset()
    )  # a misspelt profile sees nothing, not everything
    assert ps.allows("tools", frozenset({"profile:a"}), "xray") and not ps.allows(
        "tools", frozenset({"profile:a"}), "y"
    )
    assert not ps.allows(
        "resources", frozenset({"profile:a"}), "anything"
    )  # a kind that is not listed is not visible
    required = ProfileSet({"a": Profile()}, required=True)
    assert required.active(frozenset({"run"})) == frozenset() and not required.allows(
        "tools", frozenset({"run"}), "x"
    )
    default = ProfileSet({"a": Profile(tools=("*",))}, default="a")
    assert default.active(None) == {"a"} and default.active(frozenset({"run"})) is None
    assert ProfileSet({}).active(frozenset({"profile:a"})) is None and not ProfileSet({})


# ================================================================ a compiled server
async def test_a_profile_limits_tools_resources_and_prompts_and_the_rest_is_indistinguishable_from_missing():
    core = await build()
    support, admin = ["profile:support"], ["profile:admin"]
    assert names(core, support) == ["add", "shout"]
    assert set(names(core, admin)) >= {"add", "locate", "shout"}
    assert names(core, ["run"]) == names(
        core, admin
    )  # no profile scope and not required: no restriction
    assert [r.name for r in core.list_resources(scopes=support)] == ["readme"]
    assert [p.name for p in core.list_prompts(scopes=support)] == ["greet"]
    hidden = await core.call("locate", {"name": "x"}, scopes=support)
    missing = await core.call("nothing_here", {"name": "x"}, scopes=support)
    assert hidden.is_error and hidden.text.replace("locate", "X") == missing.text.replace(
        "nothing_here", "X"
    )
    assert (await core.call("add", {"a": 1, "b": 2}, scopes=support)).text == "3"
    from aaa.mcp.sources.prompts import PromptError
    from aaa.mcp.sources.resources import ResourceNotFound

    runbook = next(r for r in core.list_resources(scopes=admin) if r.name == "runbook")
    with pytest.raises(ResourceNotFound):
        await core.read_resource(runbook.uri, scopes=support)
    assert (await core.read_resource(runbook.uri, scopes=admin)).text == "internal runbook"
    with pytest.raises(PromptError, match="unknown prompt"):
        await core.aget_prompt("escalate", {}, scopes=support)
    assert (await core.aget_prompt("escalate", {}, scopes=admin)).messages[0][1] == "Escalate"
    assert core.list_resource_templates(scopes=support) == []


async def test_required_profiles_and_a_misspelt_one_leave_the_caller_with_nothing():
    core = await build(tenancy={"profiles_required": True})
    assert (
        names(core, ["run"]) == []
        and core.list_resources(scopes=["run"]) == []
        and core.list_prompts(scopes=["run"]) == []
    )
    assert names(core, ["profile:support"]) == ["add", "shout"]
    assert (
        names(await build(), ["profile:suport"]) == []
    )  # the typo fails closed even when profiles are optional


async def test_the_stdio_operator_picks_the_profile_with_tenancy_profile_and_a_request_cannot():
    core = await build(tenancy={"profile": "support"})
    assert names(core, None) == [
        "add",
        "shout",
    ]  # the default for a caller the host did not restrict
    assert (await core.call("locate", {"name": "x"})).is_error
    assert names(core, ["profile:admin"]) != names(
        core, None
    )  # an authenticated caller's own scopes win
    # nothing in a tool call's arguments can name a profile
    r = await core.call("add", {"a": 1, "b": 2, "profile": "admin"})
    assert r.is_error  # the argument is rejected by the tool's schema, it does not switch anything


async def test_describe_lists_the_profiles():
    d = describe(await build())
    assert d["profiles"]["support"]["tools"] == ["add", "shout"]


def test_the_configuration_is_validated():
    with pytest.raises(ValueError, match="not defined"):
        MCPSpec.model_validate({**BASE, "tenancy": {"profile": "ghost"}})
    with pytest.raises(ValueError, match="needs at least one"):
        MCPSpec.model_validate({"tenancy": {"profiles_required": True}})
    with pytest.raises(ValueError):
        MCPSpec.model_validate({**BASE, "profiles": {"Bad Name": {}}})
    with pytest.raises(ValueError):
        MCPSpec.model_validate({**BASE, "profiles": {"x": {"tool": ["*"]}}})  # unknown field


async def test_a_tool_glob_that_selects_nothing_is_an_error():
    with pytest.raises(SpecError, match="selects no exposed tool"):
        await build(profiles={"support": {"tools": ["ad*", "sho*", "typo*"]}})


# ================================================================ the command line
def test_the_profile_flag_limits_what_inspect_shows(tmp_path):
    cfg = tmp_path / "s.json"
    cfg.write_text(
        json.dumps(
            {
                **{k: v for k, v in BASE.items() if k != "sources"},
                "sources": [{"type": "objects", "ref": "helpers.aaa_mcp_server:rich_registry"}],
            }
        )
    )
    cfg.chmod(0o644)
    helpers = Path(__file__).parent.parent
    env = {"PYTHONPATH": f"{helpers}:{helpers.parent / 'src'}", "PATH": "/usr/bin:/bin"}

    def inspect(*args):
        r = subprocess.run(
            [sys.executable, "-m", "aaa.cli", "mcp", "inspect", "--config", str(cfg), *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)

    assert {t["name"] for t in inspect("--profile", "support")["tools"]} == {"add", "shout"}
    assert len(inspect()["tools"]) > 2
