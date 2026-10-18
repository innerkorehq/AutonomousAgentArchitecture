"""Acting in the world: a computer-use loop over an environment, code as the action, and an anytime ladder that escalates only while the budget allows.
Scripted model, offline. (The sandbox is the local-development one; use DockerSandbox for anything untrusted. Every action that changes something waits for approval, which this
example grants itself.)"""

import asyncio
import json

from aaa.algos import Anytime, CodeAct, ComputerUse, Rung, ToolLoop
from aaa.algos.base import finish
from aaa.core import OutcomeKind
from aaa.env import FakeEnvironment, FakePage
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.sandbox import SubprocessSandbox
from aaa.tools import ToolHandler, ToolRegistry, tool


async def run_approving(algorithm, model, task):
    """Run to the end, saying yes to every approval request (a real application asks a person)."""
    engine = Engine()
    LLMHandler(FakeLLMClient(model)).attach(engine)
    ToolHandler(algorithm.tools).attach(engine)
    async with engine:
        handle = await engine.start(algorithm, {"task": task})
        waiting = asyncio.ensure_future(handle.result())
        granted: set[str] = set()
        while not waiting.done():
            for event in await engine.store.read(handle.run_id):
                if event.type == "aaa.approval.requested" and event.data["call_id"] not in granted:
                    granted.add(event.data["call_id"])
                    await handle.signal("aaa.approval.granted", {"call_id": event.data["call_id"]})
            await asyncio.sleep(0.01)
        return await waiting, len(granted)


async def computer() -> None:
    env = FakeEnvironment(
        {
            "home": FakePage(text="Welcome. Pricing | Login", on={"click:pricing": "pricing"}),
            "pricing": FakePage(text="Pro costs 20 euros a month."),
        },
        "home",
    )
    steps = iter(
        [
            {"action": "click", "arguments": {"target": "pricing"}},
            {"action": "finish", "answer": "20 euros a month"},
        ]
    )
    out, approvals = await run_approving(
        ComputerUse(env), lambda request: text(json.dumps(next(steps))), "What does Pro cost?"
    )
    print(
        "computer:", out.value["answer"], "| actions", len(env.log), "| approvals asked", approvals
    )


async def code() -> None:
    replies = iter(
        [
            "```python\nprint(sum(i * i for i in range(1, 11)))\n```",
            "The sum of the first ten squares is 385.",
        ]
    )
    out, approvals = await run_approving(
        CodeAct(SubprocessSandbox(unsafe=True)),
        lambda request: text(next(replies)),
        "Sum of squares 1..10?",
    )
    print(
        "codeact:",
        out.value["answer"],
        "| it ran:",
        out.value["steps"][0]["stdout"].strip(),
        "| approvals asked",
        approvals,
    )


@tool
def lookup(query: str) -> str:
    """Look a fact up."""
    return {"capital of France": "Paris", "population of Paris": "2100000"}[query]


async def code_with_tools() -> None:
    program = """```python
city = tools.lookup(query="capital of France")
people = int(tools.lookup(query="population of " + city))
print(city, people + 1)
```"""
    replies = iter([program, "Paris has 2100001 people with one more."])
    algorithm = CodeAct(SubprocessSandbox(unsafe=True), tools=ToolRegistry([lookup]))
    out, approvals = await run_approving(
        algorithm, lambda request: text(next(replies)), "People in the capital of France, plus one?"
    )
    step = out.value["steps"][0]
    print(
        "codeact+tools:",
        step["stdout"].strip(),
        "| tool calls",
        [c["tool"] for c in step["tool_calls"]],
        "| approvals asked",
        approvals,
    )


class Fixed(ToolLoop):
    """A rung that answers at once (a stand-in for 'a cheap model' or 'a quick heuristic')."""

    def __init__(self, name: str, answer: str, good: bool) -> None:
        super().__init__(ToolRegistry([]), name=name)
        self.answer, self.good = answer, good

    async def on_event(self, event, ctx):
        if event.type == "aaa.run.started":
            return [finish(event, OutcomeKind.SUCCESS, {"text": self.answer, "good": self.good})]
        return []


async def anytime() -> None:
    ladder = [
        Rung("quick", Fixed("quick", "maybe 40", False)),
        Rung("careful", Fixed("careful", "42", True)),
    ]
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(lambda request: text("unused"))).attach(engine)
        out = await engine.run(
            Anytime(ladder, accept=lambda value: value["good"]), {"task": "Answer."}
        )
    print(
        "anytime:",
        out.value["value"]["text"],
        "| tried",
        out.value["tried"],
        "| stopped:",
        out.value["stopped"],
    )


async def main() -> None:
    for demo in (computer, code, code_with_tools, anytime):
        await demo()


asyncio.run(main())
