"""Agents that improve themselves: a verified skill library, reflection over memory, memory the model manages, and prompt optimisation that reports
its held-out score. Scripted model, offline. (The sandbox here is the local-development one; use DockerSandbox for anything untrusted.)"""

import asyncio
import json
from datetime import UTC, datetime, timedelta

from aaa.algos import MemoryReflection, SelfManagedMemory, Skill, SkillLibrary, ToolLoop
from aaa.eval import Case, Dataset, ExactMatch
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.llm.types import LLMResponse, Message, ToolCall
from aaa.memory.fake import FakeMemory
from aaa.memory.types import MemoryEntry, Scope
from aaa.optimize import optimize_prompt
from aaa.runtime import Engine
from aaa.sandbox import SubprocessSandbox
from aaa.tools import ToolHandler, ToolRegistry

SCOPE = Scope(tenant="acme", user="ann")


def system_of(request) -> str:
    return next((m.content or "" for m in request.messages if m.role == "system"), "")


def user_of(request) -> str:
    return next((m.content or "" for m in reversed(request.messages) if m.role == "user"), "")


async def skills() -> None:
    library = SkillLibrary(FakeMemory(), SubprocessSandbox(unsafe=True))
    good = Skill(
        "add_numbers",
        "Add two numbers.",
        "def skill(a, b):\n    return a + b\n",
        "assert skill(2, 3) == 5\n",
    )
    wrong = Skill(
        "bad_add",
        "Add, wrongly.",
        "def skill(a, b):\n    return a - b\n",
        "assert skill(2, 3) == 5\n",
    )
    stored, _ = await library.add(good, SCOPE)
    refused, why = await library.add(wrong, SCOPE)
    run = await library.run("add_numbers", {"a": 20, "b": 22}, SCOPE)
    print(
        "skills:",
        "stored" if stored else "refused",
        good.name,
        "| a failing test is",
        "refused" if not refused else "stored",
        "|",
        why.splitlines()[-1],
    )
    print(
        "skills: run ->",
        run.result,
        "| other tenant sees",
        await library.search("add numbers", Scope(tenant="globex")),
    )


async def reflection() -> None:
    memory = FakeMemory()
    now = datetime(2026, 1, 10, tzinfo=UTC)
    facts = [
        "Ann moved to Lisbon",
        "Ann opened a bakery",
        "The bakery lost money in April",
        "Ann hired a baker",
    ]
    for n, fact in enumerate(facts):
        await memory.add([MemoryEntry(text=fact, id=f"m{n}", time=now - timedelta(hours=n))], SCOPE)

    def model(request):
        s = system_of(request)
        if "Rate how important" in s:
            return text('{"score": 9}')
        if "salient high-level questions" in s:
            return text('{"questions": ["What is Ann building?"]}')
        return text(
            '{"insights": [{"text": "Ann is building a bakery in Lisbon.", "sources": ["m1", "m0", "invented"]}]}'
        )

    algorithm = MemoryReflection(memory, window=4, threshold=20)
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(algorithm.tools).attach(engine)
        out = await engine.run(algorithm, {"task": "reflect"}, tenant="acme", meta={"user": "ann"})
    stored = [m for m in await memory.list(SCOPE) if m.meta.get("reflection")]
    print(
        "reflection:",
        out.value["insights"][0]["text"],
        "| sources",
        out.value["insights"][0]["sources"],
        "| stored",
        len(stored),
    )


async def self_managed() -> None:
    memory = FakeMemory()
    agent = SelfManagedMemory(memory)

    def call(name, **arguments):
        return LLMResponse(
            message=Message(
                role="assistant", tool_calls=[ToolCall(id="c1", name=name, arguments=arguments)]
            ),
            model="fake",
            finish_reason="tool_calls",
        )

    async def session(task, replies):
        replies = iter(replies)
        seen = []

        def model(request):
            seen.append(system_of(request))
            return next(replies)

        async with Engine() as engine:
            LLMHandler(FakeLLMClient(model)).attach(engine)
            ToolHandler(agent.tools).attach(engine)
            await engine.run(agent, {"task": task}, tenant="acme", meta={"user": "ann"})
        return seen[0]

    await session(
        "I live in Lisbon.",
        [call("core_memory_append", label="human", text="Ann lives in Lisbon."), text("Noted.")],
    )
    first_prompt = await session("Where do I live?", [text("In Lisbon.")])
    print(
        "self-managed:",
        "Ann lives in Lisbon." in first_prompt,
        "(the next run starts with what the model chose to keep)",
    )


async def optimisation() -> None:
    cases = [Case(id=f"q{i}", input={"task": f"question {i}"}, expected="42") for i in range(10)]

    def build(prompt):
        return ToolLoop(ToolRegistry([]), system=prompt)

    def answerer(request):
        return text("42" if "step by step" in system_of(request) else "not sure")

    proposer = FakeLLMClient(
        lambda request: text(json.dumps({"prompts": ["Think step by step.", "Be brief."]}))
    )
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(answerer)).attach(engine)
        result = await optimize_prompt(
            engine,
            build,
            Dataset(cases, "qa"),
            [ExactMatch()],
            proposer,
            baseline="You answer questions.",
            rounds=1,
        )
    print(
        "optimise:",
        repr(result.best),
        f"| dev {result.baseline_dev:.1f} -> {result.best_dev:.1f}, held-out {result.baseline_holdout:.1f} -> {result.best_holdout:.1f}",
        "| improved:",
        result.improved,
    )


async def main() -> None:
    for demo in (skills, reflection, self_managed, optimisation):
        await demo()


asyncio.run(main())
