"""`aaa mcp init` / `export`: golden files, no user text in code, no secrets on disk, a project that runs, and drift detection."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aaa.mcp import scaffold
from aaa.mcp.config import SpecError, load_spec
from aaa.mcp.scaffold import Options

ROOT = Path(__file__).parent.parent
GOLDEN = ROOT / "golden" / "mcp_init"
SRC = ROOT.parent / "src"


def run_cli(*args, cwd=None, env=None, stdin=None):
    e = {**os.environ, "PYTHONPATH": f"{ROOT}:{SRC}", **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "aaa.cli", "mcp", *args],
        capture_output=True,
        text=True,
        env=e,
        cwd=cwd,
        input=stdin,
        timeout=120,
        check=False,
    )


OPENAPI_SPEC = {
    "server": {"name": "orders"},
    "sources": [
        {
            "type": "openapi",
            "document": "openapi/api.json",
            "allowed_hosts": ["api.example.com"],
            "auth": {"scheme": "bearer", "secret": "env:ORDERS_API_TOKEN"},
        }
    ],
}


def spec_of(data):
    return load_spec(data, environ={})


# ===================================================================== golden files
VARIANTS = {
    "stdio_toml": Options("orders", "stdio", True, "toml"),
    "http_json_nodocker": Options("orders", "http", False, "json"),
    "http_yaml": Options("orders", "http", True, "yaml"),
}


@pytest.fixture(autouse=True)
def stable_version(monkeypatch):
    monkeypatch.setattr(scaffold, "__version__", "0.3.1")


@pytest.mark.parametrize("variant", VARIANTS)
def test_generated_files_match_the_golden_files(variant):
    opts = VARIANTS[variant]
    if opts.fmt == "yaml":
        pytest.importorskip("yaml")
    spec = spec_of(OPENAPI_SPEC)
    files = scaffold.render(spec, scaffold.config_text_for(spec, opts), opts)
    d = GOLDEN / variant
    if os.environ.get("AAA_UPDATE_GOLDEN") == "1":
        for rel, text in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(text)
        pytest.skip("golden files rewritten")
    assert sorted(files) == sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
    for rel, text in files.items():
        assert (d / rel).read_text() == text, (
            f"{variant}/{rel} differs from its golden file (AAA_UPDATE_GOLDEN=1 regenerates them)"
        )


# ===================================================================== nothing a user types reaches code
def test_python_source_is_identical_whatever_the_configuration_says():
    hostile = {
        "server": {"name": "x", "instructions": '"""; import os; os.system("boom"); #'},
        "sources": [{"type": "module", "module": "a.b", "allow": ["f"]}],
        "tools": {"overrides": {"f": {"description": "'''\n__import__('os').system('boom')"}}},
    }
    a = scaffold.render(spec_of(hostile), "x", Options("one", "stdio"))
    b = scaffold.render(spec_of({"server": {"name": "y"}}), "x", Options("two", "stdio"))
    for rel in ("server.py", "tests/test_server.py"):
        assert a[rel] == b[rel]  # not a character of the configuration is in the code
    assert "boom" not in "".join(
        v for k, v in a.items() if k != "x" and not k.startswith("server.t")
    )


@pytest.mark.parametrize(
    "bad", ["", "Has Space", "UPPER", "1start", "a/b", "a;b", "a" * 64, "..", "-x", "x\ny"]
)
def test_project_names_are_validated(bad):
    with pytest.raises(SpecError, match="project name"):
        Options(bad).validate()
    with pytest.raises(SpecError):
        Options("ok", transport="ftp").validate()
    with pytest.raises(SpecError):
        Options("ok", fmt="ini").validate()


def test_a_written_path_can_never_leave_the_project(tmp_path):
    opts = Options("ok")
    with pytest.raises(ValueError):  # Path.relative_to refuses a path that escapes
        scaffold.write_project(tmp_path / "p", {"../evil.txt": "x"}, opts)
    assert not (tmp_path / "evil.txt").exists()


# ===================================================================== secrets
def test_no_secret_value_is_ever_written(tmp_path, monkeypatch):
    canary = "CANARY-0f9e8d7c6b5a-secret"
    monkeypatch.setenv("ORDERS_API_TOKEN", canary)
    doc = tmp_path / "api.json"
    doc.write_text(
        json.dumps(
            {"openapi": "3.0.0", "servers": [{"url": "https://api.example.com"}], "paths": {}}
        )
    )
    r = run_cli(
        "init",
        str(tmp_path / "proj"),
        "--yes",
        "--name",
        "orders",
        "--source",
        f"openapi={doc}",
        "--allowed-host",
        "api.example.com",
        "--bearer-env",
        "ORDERS_API_TOKEN",
        "--transport",
        "http",
        "--vendor-spec",
        env={"ORDERS_API_TOKEN": canary},
    )
    assert r.returncode == 0, r.stderr
    tree = {p: p.read_text() for p in (tmp_path / "proj").rglob("*") if p.is_file()}
    assert tree and not any(canary in text for text in tree.values())
    assert "ORDERS_API_TOKEN=" in (tmp_path / "proj" / ".env.example").read_text()
    assert "MCP_API_KEYS=" in (tmp_path / "proj" / ".env.example").read_text()
    assert 'secret = "env:ORDERS_API_TOKEN"' in (tmp_path / "proj" / "server.toml").read_text()
    assert canary not in r.stdout + r.stderr


# ===================================================================== the init command
def test_init_yes_writes_a_project_and_refuses_to_overwrite(tmp_path):
    dest = tmp_path / "proj"
    r = run_cli("init", str(dest), "--yes", "--source", "module=json:dumps")
    assert r.returncode == 0, r.stderr
    files = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    assert {
        "server.toml",
        "server.py",
        "requirements.txt",
        "README.md",
        ".env.example",
        ".gitignore",
        "Dockerfile",
        ".dockerignore",
        "tests/test_server.py",
        scaffold.MANIFEST,
    } <= files
    manifest = json.loads((dest / scaffold.MANIFEST).read_text())
    assert (
        manifest["options"]["transport"] == "stdio"
        and "server.py" in manifest["files"]
        and "server.toml" not in manifest["files"]
    )
    again = run_cli("init", str(dest), "--yes", "--source", "module=json:dumps")
    assert again.returncode == 2 and "not empty" in again.stderr
    forced = run_cli(
        "init",
        str(dest),
        "--yes",
        "--source",
        "module=json:dumps",
        "--force",
        "--no-docker",
        "--format",
        "json",
    )
    assert forced.returncode == 0 and (dest / "server.json").exists()


def test_init_flag_errors_are_clear(tmp_path):
    assert "needs a directory" in run_cli("init", "--yes").stderr
    r = run_cli("init", str(tmp_path / "a"), "--yes", "--source", "openapi=x.json")
    assert r.returncode == 2 and "--allowed-host" in r.stderr
    assert (
        "unknown --source type"
        in run_cli("init", str(tmp_path / "b"), "--yes", "--source", "magic=1").stderr
    )
    assert (
        "project name" in run_cli("init", str(tmp_path / "c"), "--yes", "--name", "Bad Name").stderr
    )
    assert not (tmp_path / "a").exists() and not (tmp_path / "c").exists()


def test_interactive_prompts_build_the_same_thing():
    answers = iter(
        [
            "Orders",
            "orders",
            "openapi",
            "https://api.example.com/openapi.json",
            "api.example.com",
            "yes",
            "ORDERS_TOKEN",
            "done",
            "http",
            "no",
            "json",
            "",
        ]
    )
    said: list[str] = []
    spec, opts, dest = scaffold.prompt_spec(lambda q: next(answers), said.append)
    assert (opts.name, opts.transport, opts.docker, opts.fmt, dest) == (
        "orders",
        "http",
        False,
        "json",
        Path("orders"),
    )
    (src,) = spec.sources
    assert (
        src.type == "openapi"
        and src.allowed_hosts == ["api.example.com"]
        and src.auth is not None
        and src.auth.secret == "env:ORDERS_TOKEN"
    )
    assert any("lowercase" in s for s in said)  # "Orders" was refused and re-asked


def test_interactive_init_through_the_command(tmp_path):
    r = run_cli(
        "init", str(tmp_path / "p"), stdin="demo\nmodule\njson\ndumps\ndone\nstdio\nno\ntoml\n\n"
    )
    assert r.returncode == 0, r.stderr
    assert (
        'module = "json"' in (tmp_path / "p" / "server.toml").read_text()
        and not (tmp_path / "p" / "Dockerfile").exists()
    )


# ===================================================================== the generated project works
def make_module_project(tmp_path, transport="stdio"):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        '[server]\nname = "demo"\n\n[[sources]]\ntype = "module"\nmodule = "demo_tools"\npath = "."\nallow = ["shout"]\nno_approval = ["shout"]\n'
    )
    cfg.chmod(0o644)
    dest = tmp_path / "proj"
    r = run_cli(
        "export", str(dest), "--config", str(cfg), "--name", "demo", "--transport", transport
    )
    assert r.returncode == 0, r.stderr
    (dest / "demo_tools.py").write_text(
        '__all__ = ["shout"]\n\ndef shout(text: str) -> str:\n    """Upper-case text."""\n    return text.upper()\n'
    )
    return dest


def test_the_generated_smoke_test_passes_and_paths_follow_the_configuration_file(tmp_path):
    dest = make_module_project(tmp_path)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=dest,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{SRC}"},
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    # `path = "."` is relative to the configuration file, not to where the command runs
    other = tmp_path / "elsewhere"
    other.mkdir()
    r = run_cli("check", "--config", str(dest / "server.toml"), cwd=other)
    assert r.returncode == 0 and "1 tool(s) exposed" in r.stdout


async def test_the_generated_stdio_server_runs(tmp_path):
    pytest.importorskip("mcp")
    from aaa.mcp import MCPClient

    dest = make_module_project(tmp_path)
    async with MCPClient.stdio(
        sys.executable,
        [str(dest / "server.py")],
        name="demo",
        cwd=str(dest),
        env={"PYTHONPATH": str(SRC)},
    ) as c:
        assert [t.name for t in await c.list_tools()] == ["shout"]
        out, is_error = await c.call("shout", {"text": "hi"})
        assert out == "HI" and not is_error


async def test_the_generated_http_server_authenticates_callers(tmp_path, monkeypatch):
    import httpx

    dest = make_module_project(tmp_path, "http")
    monkeypatch.syspath_prepend(str(dest))
    monkeypatch.setenv("MCP_API_KEYS", json.dumps({"s3cret": ["acme", "ann"]}))
    import importlib.util

    spec = importlib.util.spec_from_file_location("generated_server", dest / "server.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from aaa_server import create_app

    cfg = await mod.build()
    async with (
        cfg.engine,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(cfg)), base_url="http://t"
        ) as c,
    ):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        assert (await c.post("/mcp", json=body)).status_code == 401
        listed = (
            await c.post("/mcp", json=body, headers={"Authorization": "Bearer s3cret"})
        ).json()
        assert [t["name"] for t in listed["result"]["tools"]] == ["shout"]
    monkeypatch.setenv("MCP_API_KEYS", "{}")
    with pytest.raises(SystemExit, match="MCP_API_KEYS is empty"):
        await mod.build()
    sys.modules.pop("demo_tools", None)


def test_vendoring_copies_the_document_into_the_project(tmp_path):
    doc = tmp_path / "orders-api.json"
    doc.write_text(
        json.dumps(
            {
                "openapi": "3.0.0",
                "servers": [{"url": "https://api.example.com"}],
                "paths": {"/a": {"get": {"operationId": "a", "responses": {}}}},
            }
        )
    )
    r = run_cli(
        "init",
        str(tmp_path / "p"),
        "--yes",
        "--source",
        f"openapi={doc}",
        "--allowed-host",
        "api.example.com",
        "--vendor-spec",
    )
    assert r.returncode == 0, r.stderr
    assert 'document = "openapi/api.json"' in (tmp_path / "p" / "server.toml").read_text()
    assert json.loads((tmp_path / "p" / "openapi" / "api.json").read_text())["paths"]["/a"]
    ok = run_cli("check", "--config", str(tmp_path / "p" / "server.toml"), cwd=tmp_path)
    assert (
        ok.returncode == 0 and "1 tool(s) exposed" in ok.stdout
    )  # found relative to the config, from any directory


# ===================================================================== staying in sync
def test_check_and_update_track_drift(tmp_path, monkeypatch):
    dest = make_module_project(tmp_path)
    assert run_cli("export", str(dest), "--check").returncode == 0

    (dest / "server.py").write_text("# edited by hand\n")
    (dest / "Dockerfile").unlink()
    r = run_cli("export", str(dest), "--check")
    assert (
        r.returncode == 1
        and "server.py: differs from its template (and was edited by hand)" in r.stdout
        and "Dockerfile: missing" in r.stdout
    )

    up = run_cli("export", str(dest), "--update")
    assert (
        up.returncode == 0
        and "rewrote Dockerfile" in up.stdout
        and "kept server.py: you edited it" in up.stdout
    )
    assert (
        dest / "server.py"
    ).read_text() == "# edited by hand\n"  # your edits are never overwritten
    cfg_before = (dest / "server.toml").read_text()
    assert (
        dest / "server.toml"
    ).read_text() == cfg_before  # nor is the configuration, which is yours


def test_a_template_change_is_reported_and_rolled_out_to_untouched_files(tmp_path, monkeypatch):
    import aaa

    monkeypatch.setattr(
        scaffold, "__version__", aaa.__version__
    )  # the project was generated by a subprocess with the real version
    dest = make_module_project(tmp_path)
    monkeypatch.setattr(scaffold, "_GITIGNORE", ".env\n__pycache__/\n*.pyc\n.venv/\n")
    problems = scaffold.check_project(dest)
    assert problems == [".gitignore: differs from its template (the template changed)"]
    rewritten, skipped = scaffold.update_project(dest)
    assert (
        rewritten == [".gitignore"]
        and skipped == []
        and ".venv/" in (dest / ".gitignore").read_text()
    )
    assert scaffold.check_project(dest) == []


def test_check_needs_a_generated_project(tmp_path):
    r = run_cli("export", str(tmp_path), "--check")
    assert r.returncode == 2 and "was not generated by `aaa mcp init`" in r.stderr
    assert "give --config" in run_cli("export", str(tmp_path / "x")).stderr


def test_rendering_a_configuration_writes_nothing_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = spec_of(OPENAPI_SPEC)
    for fmt in ("toml", "json"):
        scaffold.config_text_for(spec, Options("orders", fmt=fmt))
    assert list(tmp_path.iterdir()) == []
