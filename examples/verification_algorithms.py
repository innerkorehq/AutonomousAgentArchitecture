"""Seven ways to get a better answer from the same model: self-critique, fact-check questions, best-of-N, decomposition, self-ask,
a graph of thoughts and tree search. Scripted model, offline."""

import asyncio
import json

from aaa.algos import (
    LATS,
    BestOfN,
    ChainOfVerification,
    GraphOfThoughts,
    LeastToMost,
    Op,
    SelfAsk,
    SelfRefine,
)
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine


def system_of(request) -> str:
    return next((m.content or "" for m in request.messages if m.role == "system"), "")


def user_of(request) -> str:
    return next((m.content or "" for m in reversed(request.messages) if m.role == "user"), "")


def between(source: str, tag: str) -> str:
    return source.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()


async def run(algorithm, model, task):
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        return (await engine.run(algorithm, {"task": task})).value


async def self_refine() -> None:
    def model(request):
        if "reviewing your own draft" in system_of(request):
            ok = "units" in between(user_of(request), "draft")
            return text(json.dumps({"ok": ok, "feedback": "say the unit"}))
        if "Revise the draft" in system_of(request):
            return text("42 km (units added)")
        return text("42")

    v = await run(SelfRefine(), model, "How far?")
    print("self-refine:", v["answer"], "after", v["rounds"], "rewrite;", v["stopped"])


async def chain_of_verification() -> None:
    def model(request):
        s = system_of(request)
        if "self-contained questions" in s:
            return text(json.dumps({"questions": ["What is the capital of Australia?"]}))
        if "briefly and factually" in s:
            return text("Canberra")
        if "Revise the draft so it agrees" in s:
            return text("The capital of Australia is Canberra.")
        return text("The capital of Australia is Sydney.")

    v = await run(ChainOfVerification(), model, "Capital of Australia?")
    print("verification:", v["answer"], "| draft was:", v["draft"])


async def best_of_n() -> None:
    drafts = iter(["short", "a longer answer", "the longest answer of all", "mid answer"])
    v = await run(
        BestOfN(4, score=lambda task, answer: len(answer)), lambda r: text(next(drafts)), "Explain."
    )
    print("best-of-n:", v["answer"], "| scores", sorted(v["scores"]))


async def least_to_most() -> None:
    def model(request):
        if "Break the question" in system_of(request):
            return text(json.dumps({"subquestions": ["Legs per dog?", "How many dogs?"]}))
        question = between(user_of(request), "question")
        return text({"Legs per dog?": "4", "How many dogs?": "3"}.get(question, "12"))

    v = await run(LeastToMost(), model, "Legs of 3 dogs?")
    print("least-to-most:", v["answer"], "via", [s["question"] for s in v["steps"][:-1]])


async def self_ask() -> None:
    def model(request):
        if "briefly and factually" in system_of(request):
            return text("1969")
        if "Q: " in user_of(request):
            return text("So the final answer is: 1969.")
        return text("Follow up: When did Apollo 11 land?")

    v = await run(SelfAsk(), model, "When did people reach the moon?")
    print("self-ask:", v["answer"], "after", len(v["followups"]), "follow-up")


async def graph_of_thoughts() -> None:
    pool = iter("abcd")

    def model(request):
        s = system_of(request)
        if "Rate how well the thought" in s:
            return text(json.dumps({"score": len(between(user_of(request), "thought").split("+"))}))
        if "Combine the thoughts" in s:
            lines = between(user_of(request), "thoughts").splitlines()
            return text("+".join(sorted(ln.split("] ")[1] for ln in lines)))
        return text(next(pool))

    graph = [
        Op("gen", "generate", n=4),
        Op("s1", "score", ("gen",)),
        Op("top", "keep_best", ("s1",), n=2),
        Op("merge", "aggregate", ("top",)),
    ]
    algorithm = GraphOfThoughts(graph)
    v = await run(algorithm, model, "Combine.")
    print(
        "graph-of-thoughts:",
        v["answer"],
        "from",
        v["thoughts"],
        "thoughts;",
        algorithm.max_calls,
        "calls at most",
    )


async def lats() -> None:
    def model(request):
        s = system_of(request)
        if "Rate how likely" in s:
            last = between(user_of(request), "reasoning").splitlines()[-1]
            return text(json.dumps({"score": 10 if last.startswith("FINAL") else 5}))
        so_far = between(user_of(request), "reasoning_so_far")
        return text("FINAL: 24" if "step" in so_far else "step 1: 8 / (3 - 8 / 3)")

    v = await run(LATS(iterations=4, branching=2), model, "Make 24 from 3 3 8 8.")
    print("lats:", v["answer"], "| path", v["path"], "| solved", v["solved"])


async def main() -> None:
    for demo in (
        self_refine,
        chain_of_verification,
        best_of_n,
        least_to_most,
        self_ask,
        graph_of_thoughts,
        lats,
    ):
        await demo()


asyncio.run(main())
