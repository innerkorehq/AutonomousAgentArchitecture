"""An engine with a scripted model and two algorithms, for agent-as-tool tests (referenced from configurations as module:attr)."""

import asyncio

from aaa.algos import BestOfN, MixtureOfAgents, ToolLoop
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Engine
from aaa.tools import ToolRegistry

SLOW_SECONDS = 0.5


def _script(req):
    last = next((m.content or "" for m in reversed(req.messages) if m.role == "user"), "")
    return text(f"answer to: {last}")


async def engine() -> Engine:
    eng = Engine()
    LLMHandler(FakeLLMClient(_script)).attach(eng)
    return eng


algo = ToolLoop(ToolRegistry([]), system="Answer briefly.")


class Slow:
    """An algorithm that takes a while (to hold a concurrency slot)."""

    name = "slow"

    def subscriptions(self):
        return ["aaa.run.started"]

    async def on_event(self, e, ctx):
        await asyncio.sleep(SLOW_SECONDS)
        from aaa.algos.base import finish
        from aaa.core.outcome import OutcomeKind

        return [finish(e, OutcomeKind.SUCCESS, {"text": "slow done"})]


slow = Slow()


# -- algorithms from the newer families, to show any algorithm can be exposed as a tool ----------------
best_of = BestOfN(3, score=lambda task, answer: len(answer))
mixture = MixtureOfAgents(proposers=2, layers=1)
