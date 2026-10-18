"""Loading an `MCPSpec`: file -> environment overlay -> flag overlay -> one validation, with errors that name the field.

Formats: JSON and TOML need nothing; YAML needs PyYAML (`pip install "autonomous-agent-architecture[mcp-config]"`), read with
`safe_load` only. Secrets never live in the spec: a field that holds one takes an `env:NAME` / `${NAME}` reference, resolved into a
`Secret` whose `repr` is redacted."""

from __future__ import annotations

import json
import os
import re
import stat
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aaa.mcp.spec import MCPSpec

ENV_PREFIX = "AAA_MCP__"


class SpecError(ValueError):
    """The configuration is invalid or cannot be used. The message names the field and says how to fix it."""


def deep_merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Dictionaries merge key by key; anything else (scalars, lists) in `over` replaces what `base` had."""
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else v
        )
    return out


def _coerce(raw: str) -> Any:
    s = raw.strip()
    if s[:1] in "[{" or s in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", s):
        try:
            return json.loads(s)
        except ValueError:
            return raw
    return raw


def env_overlay(environ: Mapping[str, str]) -> dict[str, Any]:
    """`AAA_MCP__SERVER__NAME=x` -> `{"server": {"name": "x"}}` (double underscore separates levels; JSON for lists/numbers/bools)."""
    out: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        if not path:
            continue
        node = out
        for p in path[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise SpecError(f"{key}: conflicts with another AAA_MCP__ variable")
        node[path[-1]] = _coerce(raw)
    return out


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise SpecError(f"{path}: cannot read the configuration file ({e.strerror})") from None
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(raw)
        elif suffix == ".toml":
            data = tomllib.loads(raw.decode())
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml  # pyright: ignore[reportMissingModuleSource]
            except ImportError:
                raise SpecError(
                    f"{path}: YAML needs PyYAML: pip install 'autonomous-agent-architecture[mcp-config]' "
                    "(or write the configuration as JSON or TOML)"
                ) from None
            data = yaml.safe_load(raw)
        else:
            raise SpecError(
                f"{path}: unknown configuration format {suffix!r} (use .yaml, .yml, .json or .toml)"
            )
    except SpecError:
        raise
    except Exception as e:  # noqa: BLE001 - a parser error is a configuration error, whatever the parser
        raise SpecError(f"{path}: not valid {suffix.lstrip('.').upper()}: {str(e)[:200]}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SpecError(f"{path}: the top level must be a mapping of settings")
    return data


def _anchor(data: dict[str, Any], base: Path) -> None:
    """Make the file paths a configuration names relative to the configuration file, not to wherever the server was started from."""

    def fix(d: dict[str, Any], key: str) -> None:
        v = d.get(key)
        if (
            isinstance(v, str)
            and v
            and not v.startswith(("http://", "https://"))
            and not Path(v).is_absolute()
        ):
            d[key] = str((base / v).resolve())

    for src in data.get("sources", []) if isinstance(data.get("sources"), list) else []:
        if isinstance(src, dict) and src.get("type") == "openapi":
            fix(src, "document")
        if isinstance(src, dict) and src.get("type") == "module":
            fix(src, "path")
    for res in data.get("resources", []) if isinstance(data.get("resources"), list) else []:
        if isinstance(res, dict) and res.get("type") == "files":
            fix(res, "root")


def format_errors(err: ValidationError, where: str = "") -> str:
    lines = []
    for e in err.errors(include_input=False, include_url=False, include_context=False):
        loc = ".".join(str(p) for p in e["loc"]) or "(top level)"
        lines.append(f"{loc}: {e['msg']}")
    head = (
        f"{where}: invalid MCP server configuration"
        if where
        else "invalid MCP server configuration"
    )
    return head + "\n  " + "\n  ".join(lines)


def load_spec(
    source: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    overlay: Mapping[str, Any] | None = None,
    overlays: Sequence[str | os.PathLike[str]] = (),
) -> MCPSpec:
    """Build the spec: the file (with its `include`s) or mapping (lowest), then overlay files (`overlays=` and `AAA_MCP_OVERLAY`),
    then `AAA_MCP__*` environment variables, then `overlay` (command-line flags)."""
    from aaa.mcp.include import compose, merge, overlays_from_env

    env = os.environ if environ is None else environ
    where = ""
    files: list[Path] = []
    files_overlays = [*overlays_from_env(env), *overlays]
    if source is None:
        data: dict[str, Any] = {}
    elif isinstance(source, Mapping):
        data = dict(source)
        if "include" in data:
            raise SpecError("include needs a configuration file to be relative to")
    else:
        path = Path(source)
        where = str(path)
        c = compose(path, files_overlays)
        data, files = c.data, c.files
        files_overlays = []
    if files_overlays:  # a mapping or no file at all: overlays still apply
        base: dict[str, Any] = data
        for o in files_overlays:
            c = compose(o)
            base, files = merge(base, c.data), [*files, *c.files]
        data = base
    data = deep_merge(data, env_overlay(env))
    if overlay:
        data = deep_merge(data, overlay)
    try:
        spec = MCPSpec.model_validate(data)
    except ValidationError as e:
        raise SpecError(format_errors(e, where)) from None
    if spec.imports_code() or spec.launches_processes():
        for f in files:
            if f.stat().st_mode & stat.S_IWOTH:
                raise SpecError(
                    f"{f}: this configuration imports code or launches processes and {'this file is' if str(f) != where else 'it is'} "
                    f"writable by everyone; restrict it (chmod o-w) or the server would run whatever someone else puts in it"
                )
    return spec


# -- writing a spec ------------------------------------------------------------------------------
_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(k: str) -> str:
    return k if _BARE.match(k) else json.dumps(k)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)  # a JSON string is a valid TOML basic string
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise SpecError(f"cannot write {type(v).__name__} to TOML")


def to_toml(data: dict[str, Any]) -> str:
    """The spec as TOML: scalars first, then one `[table]` per nested mapping, `[[table]]` per list of mappings; mappings inside those are inline."""
    out: list[str] = []

    def section(path: list[str], d: dict[str, Any]) -> None:
        for k, v in d.items():
            if not isinstance(v, dict) and not (
                isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
            ):
                out.append(f"{_toml_key(k)} = {_toml_value(v)}")
        for k, v in d.items():
            full = [*path, k]
            if isinstance(v, dict):
                simple = {a: b for a, b in v.items()}
                out.append(f"\n[{'.'.join(_toml_key(p) for p in full)}]")
                section(full, simple)
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                for item in v:
                    out.append(f"\n[[{'.'.join(_toml_key(p) for p in full)}]]")
                    for a, b in item.items():
                        out.append(f"{_toml_key(a)} = {_toml_value(b)}")

    section([], data)
    return "\n".join(out).strip() + "\n"


def dump_spec(spec: MCPSpec, path: str | os.PathLike[str] | None = None) -> str:
    """The spec as text in the format `path`'s extension names (JSON if no path), written to `path` when one is given. Defaults are left
    out, so the file says only what you chose. Specs with `live` sources refer to Python objects a file cannot carry, so they cannot be written."""
    text = spec_text(spec, Path(path).suffix if path else ".json")
    if path is not None:
        Path(path).write_text(text)
    return text


def spec_text(spec: MCPSpec, suffix: str = ".json") -> str:
    """The spec as text in the format a file extension (`.toml`, `.yaml`, `.yml`, `.json`) names, without writing anything."""
    if any(src.type == "live" for src in spec.sources):
        raise SpecError(
            "this configuration holds live Python objects (type = 'live'), which a file cannot carry"
        )
    data = spec.model_dump(mode="json", exclude_defaults=True, exclude_none=True)
    data = {"version": 1, **{k: v for k, v in data.items() if k != "version"}}
    suffix = suffix.lower()
    if suffix == ".toml":
        text = to_toml(data)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # pyright: ignore[reportMissingModuleSource]
        except ImportError:
            raise SpecError(
                "writing YAML needs PyYAML: pip install 'autonomous-agent-architecture[mcp-config]'"
            ) from None
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    elif suffix == ".json":
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    else:
        raise SpecError(
            f"unknown configuration format {suffix!r} (use .yaml, .yml, .json or .toml)"
        )
    return text


# -- secrets -------------------------------------------------------------------------------------
_REF = re.compile(r"^(?:env:|\$\{)([A-Za-z_][A-Za-z0-9_]*)\}?$")


class Secret:
    """A value read from the environment when needed. Its `repr` and `str` never show it."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    @classmethod
    def parse(cls, ref: str, *, field: str = "secret") -> Secret:
        m = _REF.match(ref.strip())
        if not m:
            raise SpecError(
                f"{field}: give a reference to an environment variable ('env:NAME' or '${{NAME}}'), never the value itself"
            )
        return cls(m.group(1))

    @property
    def name(self) -> str:
        return self._name

    def value(self, environ: Mapping[str, str] | None = None) -> str:
        v = (os.environ if environ is None else environ).get(self._name)
        if not v:
            raise SpecError(
                f"the environment variable {self._name} is not set (it holds a secret this server needs)"
            )
        return v

    def __repr__(self) -> str:
        return f"Secret(env:{self._name})"

    __str__ = __repr__


__all__ = [
    "ENV_PREFIX",
    "Secret",
    "SpecError",
    "deep_merge",
    "env_overlay",
    "format_errors",
    "load_spec",
]
