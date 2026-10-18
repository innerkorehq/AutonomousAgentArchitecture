"""Environments an agent can act in: a small protocol (observe, act, reset) so a loop like `ComputerUse` does not care whether it drives a browser, an
emulator or a test double.

    env = BrowserEnvironment(allowed_hosts=["docs.example.com"], start_url="https://docs.example.com/")   # needs the `browser` extra
    agent = ComputerUse(env)
    ToolHandler(agent.tools).attach(engine)

The rules every environment follows, because what it shows is written by someone else (a web page, a document, another program):

* **What it shows is data.** `Observation.text` is untrusted; the algorithm quotes it and the model is told so.
* **Each action declares whether it has side effects** (`ActionSpec.side_effect`, default true). Actions without side effects run freely; the others run only through a tool
  that **waits for approval**. An environment can also refuse an action outright (`ActionRefused`), for example typing into a password field.
* **One environment is one screen**: use it from one run at a time (the tools serialise calls, they cannot give two runs two screens)."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from aaa.tools import Tool
from aaa.tools.tool import ToolError, tool, validate_arguments

MAX_TEXT = 8000


class ActionRefused(Exception):
    """The environment will not perform this action (policy, not failure)."""


class ActionSpec(BaseModel):
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    side_effect: bool = (
        True  # true unless the environment says the action only reads or moves the view
    )


class Observation(BaseModel):
    """What the environment shows after an action. `fingerprint` changes when the state does (used to notice a stuck loop)."""

    text: str
    url: str | None = None
    title: str | None = None
    fingerprint: str = ""

    @model_validator(mode="after")
    def _finish(self) -> Observation:
        self.text = self.text[:MAX_TEXT]
        if not self.fingerprint:
            self.fingerprint = hashlib.sha256(f"{self.url}|{self.text}".encode()).hexdigest()[:16]
        return self


@runtime_checkable
class Environment(Protocol):
    @property
    def actions(self) -> Mapping[str, ActionSpec]: ...

    async def reset(self) -> Observation: ...
    async def observe(self) -> Observation: ...
    async def act(self, action: str, arguments: dict[str, Any]) -> Observation: ...
    async def aclose(self) -> None: ...


def env_tools(env: Environment) -> list[Tool]:
    """`env_reset`, `env_observe`, `env_act` and `env_act_confirm` bound to `env`.

    `env_act` runs only actions that declare no side effect; `env_act_confirm` runs the others and **waits for approval**. Calls are serialised."""
    lock = asyncio.Lock()

    async def do(action: str, arguments: dict[str, Any], *, effect: bool) -> dict[str, Any]:
        spec = env.actions.get(action)
        if spec is None:
            raise ToolError(f"unknown action {action!r}; the actions are {sorted(env.actions)}")
        if spec.side_effect != effect:
            raise ToolError(
                f"action {action!r} must be run with {'env_act_confirm' if spec.side_effect else 'env_act'}"
            )
        if problems := validate_arguments(spec.parameters, arguments):
            raise ToolError(f"bad arguments for {action}: {'; '.join(problems)[:300]}")
        try:
            async with lock:
                return (await env.act(action, arguments)).model_dump()
        except ActionRefused as e:
            raise ToolError(f"refused: {str(e)[:200]}") from None

    @tool
    async def env_reset() -> dict[str, Any]:
        """Return the environment to its starting state and show it."""
        async with lock:
            return (await env.reset()).model_dump()

    @tool
    async def env_observe() -> dict[str, Any]:
        """Show the environment as it is now."""
        async with lock:
            return (await env.observe()).model_dump()

    @tool
    async def env_act(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Perform an action that has no side effects (looking, scrolling, going back)."""
        return await do(action, arguments, effect=False)

    @tool(requires_approval=True, delivery="at_most_once")
    async def env_act_confirm(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Perform an action that changes something (clicking, typing, submitting). Waits for approval."""
        return await do(action, arguments, effect=True)

    return [env_reset, env_observe, env_act, env_act_confirm]


class FakePage(BaseModel):
    text: str
    on: dict[str, str] = Field(default_factory=dict)  # "click:pricing" -> next page name


class FakeEnvironment:
    """A deterministic environment for tests and offline examples: named pages, and actions that move between them.

    Actions: `click(target)` and `type(target, text)` (side effects), `scroll(direction)` and `read()` (none). `click:<target>` or `type:<target>` in a
    page's `on` map names the page it leads to; anything else leaves the page as it is. `log` records every action performed."""

    def __init__(self, pages: Mapping[str, FakePage | str], start: str) -> None:
        self.pages = {
            k: (v if isinstance(v, FakePage) else FakePage(text=v)) for k, v in pages.items()
        }
        if start not in self.pages:
            raise ValueError(f"unknown start page {start!r}")
        self.start, self.current = start, start
        self.log: list[tuple[str, dict[str, Any]]] = []
        self.actions = {
            "click": ActionSpec(
                description="Click the element named by target.",
                parameters={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                    "additionalProperties": False,
                },
            ),
            "type": ActionSpec(
                description="Type text into the element named by target.",
                parameters={
                    "type": "object",
                    "properties": {"target": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["target", "text"],
                    "additionalProperties": False,
                },
            ),
            "scroll": ActionSpec(
                description="Scroll the page.",
                parameters={
                    "type": "object",
                    "properties": {"direction": {"type": "string", "enum": ["up", "down"]}},
                    "required": ["direction"],
                    "additionalProperties": False,
                },
                side_effect=False,
            ),
            "read": ActionSpec(description="Read the page again.", side_effect=False),
        }

    def _look(self) -> Observation:
        return Observation(
            text=self.pages[self.current].text, url=f"fake://{self.current}", title=self.current
        )

    async def reset(self) -> Observation:
        self.current = self.start
        return self._look()

    async def observe(self) -> Observation:
        return self._look()

    async def act(self, action: str, arguments: dict[str, Any]) -> Observation:
        self.log.append((action, dict(arguments)))
        if action == "type" and "password" in str(arguments.get("target", "")).lower():
            raise ActionRefused("typing into a password field is not allowed")
        key = f"{action}:{arguments.get('target', '')}"
        self.current = self.pages[self.current].on.get(key, self.current)
        return self._look()

    async def aclose(self) -> None:
        return None


__all__ = [
    "ActionRefused",
    "ActionSpec",
    "Environment",
    "FakeEnvironment",
    "FakePage",
    "Observation",
    "env_tools",
]
