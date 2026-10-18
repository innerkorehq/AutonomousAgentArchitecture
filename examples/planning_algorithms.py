"""Plan first, then run: ReWOO, LLMCompiler, a manager with specialist workers, and PlanAndExecute with a progress review. Scripted model, offline."""

import asyncio
import json

from aaa.algos import HierarchicalAgent, LLMCompiler, PlanAndExecute, ReWOO, ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.tools import ToolHandler, ToolRegistry, tool


def system_of(request) -> str:
    return next((m.content or "" for m in request.messages if m.role == "system"), "")


def user_of(request) -> str:
    return next((m.content or "" for m in reversed(request.messages) if m.role == "user"), "")


@tool
def lookup(query: str) -> str:
    """Look a fact up."""
    return {"capital of France": "Paris", "population of Paris": "2.1 million"}.get(
        query, "unknown"
    )


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


REGISTRY = ToolRegistry([lookup, add])
PLAN = {
    "steps": [
        {"id": "E1", "tool": "lookup", "arguments": {"query": "capital of France"}},
        {"id": "E2", "tool": "lookup", "arguments": {"query": "population of #E1"}},
        {"id": "E3", "tool": "add", "arguments": {"a": 2, "b": 2}},
    ]
}


async def run(algorithm, model, task, registry=REGISTRY):
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(registry).attach(engine)
        return await engine.run(algorithm, {"task": task})


async def rewoo() -> None:
    def model(request):
        if "Plan the tool calls" in system_of(request):
            return text(json.dumps(PLAN))
        return text("Paris has 2.1 million people.")

    out = await run(ReWOO(REGISTRY), model, "How many people live in the capital of France?")
    print("rewoo:", out.value["text"], "| evidence", out.value["evidence"], "| model calls: 2")


async def llm_compiler() -> None:
    answers = iter(
        [
            {"action": "replan", "feedback": "also add the numbers"},
            {"action": "finish", "answer": "Paris has 2.1 million people; 2 + 2 = 4."},
        ]
    )
    plans = iter([{"steps": PLAN["steps"][:2]}, {"steps": [{**PLAN["steps"][2], "id": "E3"}]}])

    def model(request):
        if "Plan the tool calls" in system_of(request):
            return text(json.dumps(next(plans)))
        return text(json.dumps(next(answers)))

    out = await run(LLMCompiler(REGISTRY), model, "Population of the capital, and 2 + 2?")
    print("llm-compiler:", out.value["text"], "| rounds", out.value["rounds"])


async def manager() -> None:
    def model(request):
        s = system_of(request)
        if "Split the task" in s:
            goals = [
                {"id": "research", "worker": "researcher", "goal": "find the facts"},
                {
                    "id": "write",
                    "worker": "writer",
                    "goal": "write them up",
                    "depends_on": ["research"],
                },
            ]
            return text(json.dumps({"subgoals": goals}))
        if "Combine the sub-goal results" in s:
            return text("Report: Paris is the capital of France.")
        if "Your sub-goal (research)" in user_of(request):
            return text("Paris is the capital of France.")
        return text("A short note on Paris.")

    empty = ToolRegistry([])
    team = HierarchicalAgent(
        {
            "researcher": ToolLoop(empty, name="researcher"),
            "writer": ToolLoop(empty, name="writer"),
        },
        descriptions={"researcher": "finds facts", "writer": "writes prose"},
    )
    out = await run(team, model, "Write about France.", empty)
    print("manager:", out.value["text"], "| sub-goals", [g["id"] for g in out.value["subgoals"]])


async def review() -> None:
    def model(request):
        s = system_of(request)
        if "planner" in s and "Steps still planned" in user_of(request):
            step = {
                "id": "check",
                "description": "verify the draft against the facts",
                "depends_on": ["gather"],
            }
            return text(json.dumps({"steps": [step]}))
        if "planner" in s:
            steps = [
                {"id": "gather", "description": "gather facts", "depends_on": []},
                {"id": "write", "description": "write the answer", "depends_on": ["gather"]},
            ]
            return text(json.dumps({"steps": steps}))
        if "Combine the step results" in s:
            return text("done")
        return text("ok")

    empty = ToolRegistry([])
    algorithm = PlanAndExecute(ToolLoop(empty), review_progress=True, max_parallel=1)
    out = await run(algorithm, model, "Answer.", empty)
    print("plan-and-execute with review:", sorted(out.value["results"]))


async def main() -> None:
    for demo in (rewoo, llm_compiler, manager, review):
        await demo()


asyncio.run(main())
