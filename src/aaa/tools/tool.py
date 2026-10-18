from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from jsonschema import validators
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic_core import to_jsonable_python

from aaa.llm.types import ToolSpec
from aaa.tools.guard import GuardConfig, check_overrides


class ToolError(Exception):
    """Raise from a tool to report a clean, model-readable failure."""


class ToolArgsError(ToolError):
    pass


@dataclass
class ToolContext:
    """Injected into tools that declare a `ctx: ToolContext` parameter.

    `idempotency_key` is stable across every attempt and every worker for the same logical call
    (default `<run>/<call id>`, or whatever the tool's `idempotency_key=` function returns). Pass it
    to the system you are calling so a repeated attempt cannot repeat the effect. `attempt`/`fence`
    is the 1-based, monotonically increasing execution number of this call."""

    engine: Any
    run_id: str
    call_id: str
    attempt: int = 1
    fence: int = 1
    idempotency_key: str = ""
    tenant: str | None = None  # the run's tenant (from the log, never from the model)
    meta: dict[str, Any] = field(default_factory=dict)  # the run's metadata, e.g. {"user": ...}


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            k: _strip_titles(v) for k, v in schema.items() if k != "title" or not isinstance(v, str)
        }
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def _output_schema_of(ret: Any) -> dict[str, Any] | None:
    """A JSON schema for a tool's return type, when it is an OBJECT type (a pydantic model, TypedDict or dataclass): that is what
    MCP structured output can carry. Anything else (str, list, int, None, unions) has no output schema."""
    import dataclasses
    import typing

    if ret is None or ret is type(None):
        return None
    try:
        if isinstance(ret, type) and issubclass(ret, BaseModel):
            schema = ret.model_json_schema()
        elif typing.is_typeddict(ret) or (isinstance(ret, type) and dataclasses.is_dataclass(ret)):
            from pydantic import TypeAdapter

            schema = TypeAdapter(ret).json_schema()
        else:
            return None
    except Exception:  # noqa: BLE001 - a return type pydantic cannot describe simply has no output schema
        return None
    schema = _strip_titles(schema)
    return schema if schema.get("type") == "object" else None


class Tool:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        requires_approval: bool = False,
        timeout_s: float = 60.0,
        pool: str = "default",
        delivery: str | None = None,
        idempotency_key: Callable[[dict[str, Any]], str] | None = None,
        guard: dict[str, Any] | GuardConfig | None = None,
    ) -> None:
        """`delivery`, `idempotency_key` and `guard` configure the stalled-worker guard for this
        tool (see aaa.tools.guard); anything unset inherits the ToolHandler's default."""
        self.fn, self.name = fn, name or fn.__name__
        self.description = (description or inspect.getdoc(fn) or "").strip()
        self.requires_approval, self.timeout_s, self.pool = requires_approval, timeout_s, pool
        self.idempotency_key = idempotency_key
        if isinstance(guard, GuardConfig):
            self.guard: dict[str, Any] | GuardConfig = guard
        else:
            self.guard = {**(guard or {}), **({"delivery": delivery} if delivery else {})}
            check_overrides(self.guard)  # field-level errors surface at definition time, not at 3am
        sig, hints = inspect.signature(fn), get_type_hints(fn, include_extras=True)
        self.wants_ctx = "ctx" in sig.parameters
        fields: dict[str, Any] = {}
        for pname, p in sig.parameters.items():
            if pname == "ctx":
                continue
            fields[pname] = (hints.get(pname, Any), ... if p.default is p.empty else p.default)
        self.input_model: type[BaseModel] = create_model(
            f"{self.name}_args", __config__=ConfigDict(extra="forbid"), **fields
        )
        schema = _strip_titles(self.input_model.model_json_schema())
        self.parameters = schema
        self.output_schema: dict[str, Any] | None = _output_schema_of(hints.get("return"))

    @property
    def delivery(self) -> str | None:
        """This tool's own delivery override (`at_least_once` / `at_most_once`), or None to inherit the
        ToolHandler's default."""
        g = self.guard
        return g.get("delivery") if isinstance(g, dict) else getattr(g, "delivery", None)

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext | None = None) -> Any:
        try:
            parsed = self.input_model.model_validate(arguments)
        except ValidationError as e:
            msgs = "; ".join(
                f"{'.'.join(map(str, x['loc'])) or 'arguments'}: {x['msg']}" for x in e.errors()
            )
            raise ToolArgsError(f"invalid arguments for {self.name}: {msgs}") from e
        kwargs = {k: getattr(parsed, k) for k in type(parsed).model_fields}
        if self.wants_ctx:
            kwargs["ctx"] = ctx
        if inspect.iscoroutinefunction(self.fn):
            result = await self.fn(**kwargs)
        else:
            result = await asyncio.to_thread(self.fn, **kwargs)
        return to_jsonable_python(result)


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Check `arguments` against a JSON Schema (the draft the schema declares, else the latest); returns human-readable problems (empty = valid).
    `jsonschema` is a required dependency: ranges, enums, patterns and nested schemas are enforced, never skipped."""
    validator = validators.validator_for(schema)
    return [
        f"{'.'.join(map(str, e.absolute_path)) or 'arguments'}: {e.message}"
        for e in sorted(
            validator(schema).iter_errors(arguments), key=lambda e: list(e.absolute_path)
        )
    ]


class SchemaTool(Tool):
    """A tool defined by an explicit JSON Schema and an async callable taking the argument dict,
    instead of a Python signature. This is how tools that arrive from outside Python (MCP servers,
    OpenAPI operations, ...) join a `ToolRegistry`; they get the same argument validation,
    approval, timeout and execution-guard behavior as `@tool` functions."""

    def __init__(
        self,
        call: Callable[[dict[str, Any]], Any],
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
        **opts: Any,
    ) -> None:
        async def _unused() -> None: ...

        super().__init__(_unused, name=name, description=description, **opts)
        self.fn = call
        self.output_schema = _strip_titles(output_schema) if output_schema else None
        self.parameters = _strip_titles(parameters or {"type": "object", "properties": {}})

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext | None = None) -> Any:
        if problems := validate_arguments(self.parameters, arguments):
            raise ToolArgsError(f"invalid arguments for {self.name}: {'; '.join(problems)}")
        result = self.fn(arguments)
        return to_jsonable_python(await result if inspect.isawaitable(result) else result)


def tool(fn: Callable[..., Any] | None = None, **opts: Any) -> Any:
    """Decorator: `@tool` or `@tool(requires_approval=True)`. Schema comes from type hints,
    description from the docstring."""
    if fn is not None:
        return Tool(fn, **opts)
    return lambda f: Tool(f, **opts)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool | Callable[..., Any]) -> Tool:
        t = t if isinstance(t, Tool) else Tool(t)
        if t.name in self._tools:
            raise ValueError(f"duplicate tool {t.name!r}")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        return [t.spec() for n, t in self._tools.items() if names is None or n in names]

    def extend(self, other: ToolRegistry) -> None:
        for t in other._tools.values():
            if self._tools.get(t.name) is t:
                continue
            self.add(t)

    def merged(self, other: ToolRegistry) -> ToolRegistry:
        return ToolRegistry([*self._tools.values(), *other._tools.values()])
