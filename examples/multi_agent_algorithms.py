"""Several model calls working together: a mixture of agents, a blackboard, an auction, a role-play protocol with typed documents, and a loop that asks a person.
Scripted model, offline."""

import asyncio
import json

from pydantic import BaseModel

from aaa.algos import AskHuman, Auction, Blackboard, MixtureOfAgents, Role, RolePlaySOP, ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.tools import ToolRegistry


def system_of(request) -> str:
    return next((m.content or "" for m in request.messages if m.role == "system"), "")


def user_of(request) -> str:
    return next((m.content or "" for m in reversed(request.messages) if m.role == "user"), "")


def between(source: str, tag: str) -> str:
    return source.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()


async def run(algorithm, model, task):
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        return await engine.run(algorithm, {"task": task})


async def mixture() -> None:
    def model(request):
        s = system_of(request)
        if "Combine the answers" in s:
            return text("Paris, the capital of France (3 proposers agreed)")
        if "Several assistants answered" in s:
            return text("Paris")
        return text("Paris, probably")

    out = await run(MixtureOfAgents(proposers=3, layers=2), model, "Capital of France?")
    print(
        "mixture:", out.value["answer"], "| layers", [len(layer) for layer in out.value["layers"]]
    )


async def blackboard() -> None:
    turns = iter(["researcher", "critic", "done"])

    def model(request):
        s = system_of(request)
        if "You chair a group" in s:
            return text(json.dumps({"next": next(turns)}))
        if "Write the final answer" in s:
            return text("Launch on Tuesday, after the fix.")
        role = s.split("You are the ")[1].split(":")[0]
        return text(
            {
                "researcher": "Tuesday has the fewest conflicts.",
                "critic": "The fix is not merged yet.",
            }[role]
        )

    experts = {"researcher": "finds facts.", "critic": "finds flaws."}
    out = await run(Blackboard(experts), model, "When should we launch?")
    print("blackboard:", out.value["answer"], "| board", [b["author"] for b in out.value["board"]])


async def auction() -> None:
    bids = {
        "quick": {"cost": 1, "confidence": 0.3},
        "expert": {"cost": 4, "confidence": 0.95},
        "slow": {"cost": 30, "confidence": 0.9},
    }

    def model(request):
        if "up for auction" in system_of(request):
            return text(json.dumps(bids[system_of(request).split("You are ")[1].split(":")[0]]))
        return text("Done by the best-value bidder.")

    empty = ToolRegistry([])
    workers = {name: ToolLoop(empty, name=name) for name in bids}
    out = await run(Auction(workers), model, "Fix the bug.")
    print("auction: winner", out.value["winner"], "|", out.value["text"])


async def protocol() -> None:
    class Spec(BaseModel):
        title: str
        steps: list[str]

    class Review(BaseModel):
        approved: bool
        notes: str

    def model(request):
        if "You are the analyst" in system_of(request):
            return text(
                json.dumps(
                    {"title": "Migrate", "steps": ["freeze", "copy", "switch"], "extra": "ignored"}
                )
            )
        spec = json.loads(between(user_of(request), "document_from_previous_role"))
        return text(json.dumps({"approved": len(spec["steps"]) == 3, "notes": "three steps, fine"}))

    roles = [Role("analyst", "Write a spec.", Spec), Role("reviewer", "Review the spec.", Review)]
    out = await run(RolePlaySOP(roles), model, "Plan the migration.")
    print(
        "protocol:", out.value["document"], "| analyst document", out.value["documents"]["analyst"]
    )


async def human() -> None:
    replies = iter(
        [
            {"action": "ask", "question": "For how many people?"},
            {"action": "answer", "answer": "Table booked for 4."},
        ]
    )
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(lambda request: text(json.dumps(next(replies))))).attach(engine)
        handle = await engine.start(AskHuman(), {"task": "Book a table."})
        asked = await engine.await_event(handle.run_id, lambda e: e.type == "aaa.human.requested")
        print("human: the run is waiting on", repr(asked.data["question"]))
        await handle.signal(
            "aaa.human.answered", {"question_id": asked.data["question_id"], "answer": "4"}
        )
        out = await handle.result()
    print("human:", out.value["answer"], "| asked", out.value["asked"])


async def main() -> None:
    for demo in (mixture, blackboard, auction, protocol, human):
        await demo()


asyncio.run(main())
