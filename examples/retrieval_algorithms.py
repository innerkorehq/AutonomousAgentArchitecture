"""Retrieval that checks itself: CorrectiveRAG grades passages and falls back to a web tool, SelfRAG answers per passage and judges support,
AdaptiveRAG matches retrieval to the question. Scripted model, offline."""

import asyncio
import json
import re

from aaa.algos import AdaptiveRAG, CorrectiveRAG, SelfRAG
from aaa.data import Doc, InMemoryIndex
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.tools import ToolHandler, tool


def system_of(request) -> str:
    return next((m.content or "" for m in request.messages if m.role == "system"), "")


def user_of(request) -> str:
    return next((m.content or "" for m in reversed(request.messages) if m.role == "user"), "")


def between(source: str, tag: str) -> str:
    return source.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()


async def index_of(*docs: Doc) -> InMemoryIndex:
    index = InMemoryIndex()
    await index.upsert(list(docs))
    return index


FRANCE = Doc(id="fr", text="Paris is the capital of France.")
GERMANY = Doc(id="de", text="Berlin is the capital of Germany.")
BANANAS = Doc(id="bn", text="Bananas are yellow.")


@tool
def web_search(query: str) -> list:
    """Search the web."""
    return [{"id": "w1", "text": "France's capital city is Paris."}]


async def run(algorithm, model, task):
    async with Engine() as engine:
        LLMHandler(FakeLLMClient(model)).attach(engine)
        ToolHandler(algorithm.tools).attach(engine)
        return await engine.run(algorithm, {"task": task})


def model(request):
    s = system_of(request)
    if "Judge whether the passage helps" in s:
        return text(json.dumps({"relevant": "France" in between(user_of(request), "passage")}))
    if "Judge the candidate answer" in s:
        ok = "France" in between(user_of(request), "passage")
        return text(
            json.dumps(
                {"relevant": ok, "supported": "full" if ok else "none", "useful": 5 if ok else 1}
            )
        )
    if "Does answering this question need passages" in s:
        return text(json.dumps({"retrieve": True}))
    if "Choose how to answer" in s:
        return text(json.dumps({"route": "iterative"}))
    if "You may search the knowledge base" in s:
        evidence = user_of(request)
        if "Berlin" in evidence:
            action = {"action": "answer"}
        elif "Paris" in evidence:
            action = {"action": "search", "query": "capital of Germany"}
        else:
            action = {"action": "search", "query": "capital of France"}
        return text(json.dumps(action))
    ids = re.findall(r"\[([\w:]+)\]", between(user_of(request), "passages"))
    return text("Answered from " + ", ".join(ids))


async def corrective() -> None:
    out = await run(
        CorrectiveRAG(await index_of(GERMANY, BANANAS), web_tool=web_search, max_rewrites=0),
        model,
        "What is the capital of France?",
    )
    print(
        "corrective:",
        out.value["text"],
        "| route",
        out.value["route"],
        "| sources",
        out.value["sources"],
    )
    out = await run(
        CorrectiveRAG(await index_of(FRANCE, GERMANY, BANANAS), k=3),
        model,
        "What is the capital of France?",
    )
    print(
        "corrective:",
        out.value["text"],
        "| route",
        out.value["route"],
        "| graded",
        out.value["graded"],
        "passages",
    )


async def self_rag() -> None:
    out = await run(
        SelfRAG(await index_of(FRANCE, GERMANY), adaptive=True, max_candidates=2),
        model,
        "What is the capital of France?",
    )
    print(
        "self-rag:",
        out.value["text"],
        "| supported",
        out.value["supported"],
        "| from",
        out.value["source"],
    )


async def adaptive() -> None:
    out = await run(
        AdaptiveRAG(await index_of(FRANCE, GERMANY), k=1),
        model,
        "Compare the capitals of France and Germany.",
    )
    print(
        "adaptive:", out.value["route"], "with", out.value["hops"], "searches;", out.value["text"]
    )


async def main() -> None:
    await corrective()
    await self_rag()
    await adaptive()


asyncio.run(main())
