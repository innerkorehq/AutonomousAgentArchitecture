"""Tool arguments are validated against the full JSON Schema (ranges, enums, patterns, nesting), with no weaker fallback."""

from typing import Annotated, Literal

import pytest
from pydantic import Field

from aaa.tools import SchemaTool, tool, validate_arguments
from aaa.tools.tool import ToolArgsError

SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        "kind": {"type": "string", "enum": ["a", "b"]},
        "code": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "nested": {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 0}},
            "required": ["n"],
        },
    },
    "required": ["limit"],
    "additionalProperties": False,
}


def test_what_a_basic_check_would_miss_is_rejected():
    assert (
        validate_arguments(
            SCHEMA, {"limit": 5, "kind": "a", "code": "ABC", "tags": ["x"], "nested": {"n": 1}}
        )
        == []
    )
    for bad in (
        {"limit": 0},
        {"limit": 11},
        {"limit": 1, "kind": "c"},
        {"limit": 1, "code": "abc"},
        {"limit": 1, "tags": ["a", "b", "c"]},
        {"limit": 1, "nested": {"n": -1}},
        {"limit": 1, "nested": {}},
        {"limit": 1, "extra": 1},
        {},
    ):
        assert validate_arguments(SCHEMA, bad), bad


async def test_a_decorated_tool_enforces_its_own_constraints():
    @tool
    def pick(n: Annotated[int, Field(ge=1, le=3)], mode: Literal["x", "y"]) -> str:
        """Pick."""
        return f"{n}{mode}"

    assert await pick.invoke({"n": 2, "mode": "x"}, None) == "2x"  # type: ignore[arg-type]
    for bad in ({"n": 0, "mode": "x"}, {"n": 4, "mode": "x"}, {"n": 1, "mode": "z"}):
        with pytest.raises(ToolArgsError):
            await pick.invoke(bad, None)  # type: ignore[arg-type]


async def test_a_schema_tool_enforces_the_schema_it_was_given():
    async def call(args):
        return args

    t = SchemaTool(call, name="t", description="d", parameters=SCHEMA)
    with pytest.raises(ToolArgsError):
        await t.invoke({"limit": 99}, None)  # type: ignore[arg-type]
    assert await t.invoke({"limit": 3}, None) == {"limit": 3}  # type: ignore[arg-type]
