"""Algorithm lineups per task family. Each entry is a factory so every run gets a fresh instance (and a fresh engine)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from aaa.algos import (
    LATS,
    AdaptiveRAG,
    BestOfN,
    ChainOfVerification,
    CodeAct,
    CorrectiveRAG,
    Debate,
    GraphOfThoughts,
    LeastToMost,
    LLMCompiler,
    PlanAndExecute,
    RAGAgent,
    ReAct,
    ReWOO,
    SelfConsistency,
    SelfRAG,
    SelfRefine,
    ToolLoop,
    TreeOfThoughts,
)
from aaa.eval import CallableMetric, Dataset, Metric
from aaa.sandbox import SubprocessSandbox
from aaa.tools import Tool, ToolRegistry

from .tasks import (
    CITIES,
    is_refusal,
    kb_lookup,
    last_number,
    math_cases,
    rag_cases,
    rag_index,
    tool_cases,
)

COT = "Think step by step, then finish with a line 'Answer: <integer>'."


def _add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def _subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


def _multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def kb_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                kb_lookup,
                name="lookup",
                description="Look up a fact. Ask like 'population of Veldor' or 'area of Veldor in km2'. Returns a number.",
            ),
            Tool(_add, name="add", description="Add two integers."),
            Tool(_subtract, name="subtract", description="Subtract b from a."),
            Tool(_multiply, name="multiply", description="Multiply two integers."),
        ]
    )


def numeric_metric() -> Metric:
    def correct(r: Any) -> tuple[bool, str]:
        got = last_number(r.text or "")
        return (
            got is not None and abs(got - float(r.case.expected)) < 1e-6
        ), f"got {got}, want {r.case.expected}"

    return CallableMetric(correct, "correct")


def rag_metric() -> Metric:
    def correct(r: Any) -> tuple[bool, str]:
        text = r.text or ""
        if r.case.expected == "REFUSE":
            declined = (
                r.outcome is not None
                and not r.outcome.ok
                and "no relevant evidence" in str(r.outcome.reason)
            )  # CorrectiveRAG's refusal
            return (is_refusal(text) or declined), "expected a refusal"
        return str(r.case.expected).lower() in text.lower(), f"want {r.case.expected!r}"

    return CallableMetric(correct, "correct")


@dataclass
class Entry:
    name: str
    build: Callable[[], Any]  # -> algorithm
    needs_tools: Callable[[], ToolRegistry | Any] | None = (
        None  # attach ToolHandler(<this>) when set
    )
    about: str = ""


@dataclass
class Family:
    name: str
    dataset: Callable[[], Dataset]
    metric: Callable[[], Metric]
    entries: list[Entry]
    prepare: Callable[[], Awaitable[Any]] | None = None


def math_family() -> Family:
    sandbox = SubprocessSandbox(
        unsafe=True
    )  # benchmark only: invented arithmetic problems, no untrusted input
    return Family(
        "math",
        math_cases,
        numeric_metric,
        [
            Entry(
                "direct",
                lambda: ToolLoop(ToolRegistry([]), system=COT),
                about="one call, think step by step",
            ),
            Entry("self_consistency", lambda: SelfConsistency(5), about="5 samples, majority vote"),
            Entry("best_of_n", lambda: BestOfN(5), about="5 samples, model judge picks"),
            Entry(
                "self_refine", lambda: SelfRefine(max_rounds=2), about="draft, critique, rewrite"
            ),
            Entry(
                "chain_of_verification",
                lambda: ChainOfVerification(max_questions=3),
                about="draft, check questions, revise",
            ),
            Entry(
                "least_to_most", lambda: LeastToMost(), about="sub-questions first, then the answer"
            ),
            Entry("debate", lambda: Debate(3, 1), about="3 agents, 1 round, judge"),
            Entry(
                "graph_of_thoughts",
                lambda: GraphOfThoughts(),
                about="default generate/score/aggregate graph",
            ),
            Entry(
                "tree_of_thoughts",
                lambda: TreeOfThoughts(depth=2, width=2, branching=2),
                about="depth 2, width 2",
            ),
            Entry(
                "lats", lambda: LATS(iterations=3, branching=2), about="tree search, 3 iterations"
            ),
            Entry(
                "codeact",
                lambda: CodeAct(sandbox, requires_approval=False, max_steps=4),
                about="writes and runs Python",
            ),
        ],
    )


def tools_family() -> Family:
    reg = kb_registry
    return Family(
        "tools",
        tool_cases,
        numeric_metric,
        [
            Entry("tool_loop", lambda: ToolLoop(reg()), reg, "native tool calling"),
            Entry("react", lambda: ReAct(reg()), reg, "text Thought/Action/Observation"),
            Entry("rewoo", lambda: ReWOO(reg()), reg, "plan once with placeholders, run, solve"),
            Entry("llm_compiler", lambda: LLMCompiler(reg()), reg, "plan a DAG, run in parallel"),
            Entry(
                "plan_and_execute",
                lambda: PlanAndExecute(ToolLoop(reg())),
                reg,
                "plan, run each step with a tool loop",
            ),
        ],
    )


def rag_family() -> Family:
    state: dict[str, Any] = {}

    async def prepare() -> None:
        state["index"] = await rag_index()

    def idx() -> Any:
        return state["index"]

    return Family(
        "rag",
        rag_cases,
        rag_metric,
        [
            Entry(
                "rag_agent",
                lambda: RAGAgent(idx()),
                lambda: RAGAgent(idx()).tools,
                "retrieve with a tool loop",
            ),
            Entry(
                "corrective_rag",
                lambda: CorrectiveRAG(idx()),
                lambda: CorrectiveRAG(idx()).tools,
                "grade chunks, rewrite, answer",
            ),
            Entry(
                "self_rag",
                lambda: SelfRAG(idx()),
                lambda: SelfRAG(idx()).tools,
                "reflect on support and usefulness",
            ),
            Entry(
                "adaptive_rag",
                lambda: AdaptiveRAG(idx()),
                lambda: AdaptiveRAG(idx()).tools,
                "classify, then pick the retrieval depth",
            ),
        ],
        prepare,
    )


FAMILIES = {"math": math_family, "tools": tools_family, "rag": rag_family}
_ = CITIES  # keep the knowledge base importable from here for the docs
