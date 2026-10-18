"""Calling tools from inside sandboxed code, without giving the sandbox a network or a way to reach the host.

The sandbox runs a program to completion and has no channel back, so the bridge is **replay**: a program that calls `tools.lookup(query="x")` runs until its first call that
has no recorded answer, prints a request and stops. The host performs that call through the normal tool pipeline (approval, timeouts, guards, budgets, the tool's own schema),
records the answer, and runs the program again from the start with the answers so far; calls the program already made return their recorded answers instantly. When the program runs
to the end, its output is the result.

What that gives, and what it asks of the code:

* **Every tool call is an ordinary tool call** on the run's log, made by the host, for the run's tenant and user. The code cannot name a tool the host did not allow, cannot
  reach the host's memory or network, and cannot make a call the host did not see.
* **The program must be deterministic up to its calls** (no randomness, clock or unrecorded input deciding which calls it makes), because it is run again. The prelude checks this:
  if replay reaches a call that differs from the recorded one, the code gets an error instead of a wrong answer.
* **Printing is not an effect**: output is taken from the final, complete run only.
* **Approval covers the program once.** Running new code waits for approval (`run_program`); running the *same, already approved* code again with more answers (`resume_program`) does not,
  and refuses to run code that was not approved. A tool that needs approval of its own still asks, every time."""

from __future__ import annotations

import json
from typing import Any

from aaa.sandbox.base import Limits, Sandbox, SandboxError
from aaa.tools import Tool
from aaa.tools.tool import ToolContext, ToolError, tool

MARK = "__AAA_TOOL_CALL__"
PENDING_EXIT = 87
MAX_RESULTS_BYTES = 200_000
PRELUDE = f'''\
import json as _json, os as _os, sys as _sys

class ToolCallError(Exception):
    """A tool call made by this program failed, was refused, or was not approved."""

_recorded = _json.load(open("tool_results.json"))
_next = 0

def _call(_name, **_kwargs):
    global _next
    _sig = _json.dumps([_name, _kwargs], sort_keys=True, default=str)
    if _next < len(_recorded):
        _r = _recorded[_next]
        if _r["sig"] != _sig:
            raise ToolCallError("the program is not deterministic: call %d is not the one recorded" % _next)
        _next += 1
        if _r["ok"]:
            return _r["value"]
        raise ToolCallError(_r["error"])
    _sys.stdout.write("\\n{MARK}" + _json.dumps({{"tool": _name, "arguments": _kwargs, "index": _next}}, default=str) + "\\n")
    _sys.stdout.flush()
    _os._exit({PENDING_EXIT})

class _Tools:
    def __getattr__(self, _name):
        if _name.startswith("_"):
            raise AttributeError(_name)
        return lambda **_kwargs: _call(_name, **_kwargs)

tools = _Tools()
'''


def signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """The same string the prelude computes for a call, so a replayed call can be matched to its record."""
    return json.dumps([tool_name, arguments], sort_keys=True, default=str)


def record(
    tool_name: str, arguments: dict[str, Any], ok: bool, value: Any = None, error: str = ""
) -> dict[str, Any]:
    entry: dict[str, Any] = {"sig": signature(tool_name, arguments), "ok": ok}
    entry |= {"value": value} if ok else {"error": error}
    return entry


def _pending(stdout: str, exit_code: int | None) -> tuple[str, dict[str, Any] | None]:
    """(the program's output without the request, the request it stopped on or None)."""
    if exit_code != PENDING_EXIT:
        return stdout, None
    lines = stdout.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(MARK):
            try:
                req = json.loads(lines[i][len(MARK) :])
            except ValueError:
                return stdout, None
            if (
                isinstance(req, dict)
                and isinstance(req.get("tool"), str)
                and isinstance(req.get("arguments"), dict)
                and isinstance(req.get("index"), int)
            ):
                return "\n".join(lines[:i]), req
    return stdout, None


async def _execute(
    sandbox: Sandbox, limits: Limits, code: str, results: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = json.dumps(results, default=str)
    if len(payload) > MAX_RESULTS_BYTES:
        raise ToolError("the recorded tool results are too large to hand to the program")
    try:
        r = await sandbox.run(
            f"{PRELUDE}\n{code}\n", "python", files={"tool_results.json": payload}, limits=limits
        )
    except SandboxError as e:
        raise ToolError(f"the sandbox could not run the code: {e}") from None
    stdout, request = _pending(r.stdout, r.exit_code)
    return {
        "exit_code": None if request else r.exit_code,
        "stdout": stdout,
        "stderr": r.stderr,
        "timed_out": r.timed_out,
        "killed": r.killed,
        "output_truncated": r.truncated,
        "pending_call": request,
    }


def program_tools(sandbox: Sandbox, *, limits: Limits, requires_approval: bool) -> list[Tool]:
    """`run_program` (new code: waits for approval) and `resume_program` (the same approved code again, with tool answers)."""

    @tool(
        requires_approval=requires_approval,
        name="run_program",
        timeout_s=limits.timeout_s + 60,
        delivery="at_most_once",
    )
    async def run_program(code: str, ctx: ToolContext) -> dict[str, Any]:
        """Run a program that may call tools as `tools.<name>(**arguments)`; stops at its first tool call that has no recorded answer."""
        return await _execute(sandbox, limits, code, [])

    @tool(name="resume_program", timeout_s=limits.timeout_s + 60, delivery="at_most_once")
    async def resume_program(
        origin: str, results: list[dict[str, Any]], ctx: ToolContext
    ) -> dict[str, Any]:
        """Run an already approved program again from the start with the tool answers so far. `origin` is the call id of its `run_program`."""
        code, granted, denied = None, False, False
        for e in await ctx.engine.store.read(ctx.run_id):
            e = await ctx.engine.hydrate(e)
            if (
                e.type == "aaa.tool.requested"
                and e.data.get("call_id") == origin
                and e.data.get("name") == "run_program"
            ):
                code = e.data["arguments"].get("code")
            elif e.type == "aaa.approval.granted" and e.data.get("call_id") == origin:
                granted = True
            elif e.type == "aaa.approval.denied" and e.data.get("call_id") == origin:
                denied = True
        if not isinstance(code, str):
            raise ToolError("there is no program with that origin in this run")
        if denied or (requires_approval and not granted):
            raise ToolError("that program was not approved")
        return await _execute(sandbox, limits, code, results)

    return [run_program, resume_program]


__all__ = ["MARK", "PENDING_EXIT", "PRELUDE", "program_tools", "record", "signature"]
