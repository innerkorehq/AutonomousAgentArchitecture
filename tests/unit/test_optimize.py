"""aaa.optimize: dev/held-out split, the search, the honest report, and its limits."""

import json

import pytest

from aaa.algos import ToolLoop
from aaa.eval import Case, Dataset, ExactMatch
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.errors import FatalError
from aaa.llm.fake import text
from aaa.optimize import default_objective, optimize_prompt, split_dataset
from aaa.runtime import Engine
from aaa.tools import ToolRegistry

CASES = [Case(id=f"c{i}", input={"task": f"question {i}"}, expected="42") for i in range(10)]
DATA = Dataset(CASES, "qa")


def sys_of(req):
    return next((m.content or "" for m in req.messages if m.role == "system"), "")


def user_of(req):
    return next((m.content or "" for m in reversed(req.messages) if m.role == "user"), "")


def build(prompt):
    return ToolLoop(ToolRegistry([]), system=prompt)


def answerer(good):
    """The system answers 42 only when its prompt contains `good`."""

    def script(req):
        return text("42" if good in sys_of(req) else "I am not sure")

    return script


def proposer(*prompts):
    return FakeLLMClient(lambda req: text(json.dumps({"prompts": list(prompts)})))


async def run(good="step by step", proposals=("Think step by step.", "Be brief."), **kw):
    async with Engine() as eng:
        LLMHandler(FakeLLMClient(answerer(good))).attach(eng)
        return await optimize_prompt(
            eng,
            build,
            DATA,
            [ExactMatch()],
            proposer(*proposals),
            baseline="You answer questions.",
            **kw,
        )


def test_the_split_is_deterministic_disjoint_and_independent_of_file_order():
    dev, held = split_dataset(DATA, 0.3, seed=1)
    assert len(dev) == 7 and len(held) == 3
    assert {c.id for c in dev}.isdisjoint({c.id for c in held})
    _again_dev, again_held = split_dataset(Dataset(list(reversed(CASES))), 0.3, seed=1)
    assert [c.id for c in again_held] == [c.id for c in held]
    assert [c.id for c in split_dataset(DATA, 0.3, seed=2)[1]] != [c.id for c in held]
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            split_dataset(DATA, bad)
    with pytest.raises(ValueError):
        split_dataset(Dataset(CASES[:1]), 0.5)


async def test_a_better_prompt_is_found_and_confirmed_on_cases_it_never_saw():
    r = await run()
    assert r.best == "Think step by step." and r.improved
    assert (r.baseline_dev, r.best_dev) == (0.0, 1.0) and (r.baseline_holdout, r.best_holdout) == (
        0.0,
        1.0,
    )
    assert (r.dev_cases, r.holdout_cases) == (7, 3)
    assert [c.round for c in r.candidates] == [
        0,
        1,
        1,
    ]  # round 2 only repeated what was already tried
    assert "-You answer questions." in r.diff and "+Think step by step." in r.diff
    assert r.evaluations <= 7 * (1 + 2 * 3) + 2 * 3


async def test_a_prompt_that_only_wins_on_the_dev_cases_is_not_called_an_improvement():
    _dev, held = split_dataset(DATA, 0.3, seed=0)
    held_ids = {c.id for c in held}

    def script(req):
        # the "overfit" prompt answers right on the dev cases only
        overfit = "memorise" in sys_of(req)
        task = user_of(req).split()[-1]
        return text("42" if overfit and f"c{task}" not in held_ids else "no")

    async with Engine() as eng:
        LLMHandler(FakeLLMClient(script)).attach(eng)
        r = await optimize_prompt(
            eng,
            build,
            DATA,
            [ExactMatch()],
            proposer("memorise the answers"),
            baseline="You answer questions.",
            rounds=1,
        )
    assert r.best == "memorise the answers" and r.best_dev == 1.0
    assert r.best_holdout == 0.0 and r.improved is False  # the held-out number is what counts


async def test_nothing_better_keeps_the_baseline_and_a_failing_proposer_costs_only_a_round():
    r = await run(good="never matches", proposals=("a", "b"))
    assert r.best == r.baseline and r.improved is False and r.diff == ""

    class Down:
        async def complete(self, request):
            raise FatalError("down")

    async with Engine() as eng:
        LLMHandler(FakeLLMClient(answerer("x"))).attach(eng)
        r = await optimize_prompt(eng, build, DATA, [ExactMatch()], Down(), baseline="b", rounds=2)  # type: ignore[arg-type]
    assert r.best == "b" and len(r.candidates) == 1


async def test_duplicates_empty_and_oversized_proposals_are_ignored_and_failures_are_shown_as_data():
    seen = []

    class Spy:
        async def complete(self, request):
            seen.append(user_of(request))
            return text(
                json.dumps(
                    {
                        "prompts": [
                            "You answer questions.",
                            "  ",
                            "x" * 5000,
                            "Think step by step.",
                            "Think step by step.",
                        ]
                    }
                )
            )

    async with Engine() as eng:
        LLMHandler(FakeLLMClient(answerer("step by step"))).attach(eng)
        r = await optimize_prompt(
            eng, build, DATA, [ExactMatch()], Spy(), baseline="You answer questions.", rounds=1
        )  # type: ignore[arg-type]
    assert [c.prompt for c in r.candidates] == ["You answer questions.", "Think step by step."]
    assert (
        "<failing_cases>" in seen[0] and "expected: 42" in seen[0] and "<current_prompt>" in seen[0]
    )


async def test_the_search_is_bounded_before_it_starts():
    async with Engine() as eng:
        LLMHandler(FakeLLMClient(answerer("x"))).attach(eng)
        with pytest.raises(ValueError, match="case runs"):
            await optimize_prompt(
                eng,
                build,
                DATA,
                [ExactMatch()],
                proposer("a"),
                baseline="b",
                rounds=6,
                variants=8,
                max_evaluations=100,
            )
        for kw in (
            {"rounds": 0},
            {"rounds": 7},
            {"variants": 0},
            {"variants": 9},
            {"baseline": " "},
            {"baseline": "x" * 5000},
        ):
            args = {"baseline": "b", **kw}
            with pytest.raises(ValueError):
                await optimize_prompt(eng, build, DATA, [ExactMatch()], proposer("a"), **args)


def test_the_default_objective_counts_unrunnable_cases_as_failures():
    from aaa.eval.report import EvalReport
    from aaa.eval.result import CaseResult
    from aaa.score import Score

    ok = CaseResult(case=CASES[0], outcome=None, scores=[Score(name="m", value=True)])
    bad = CaseResult(
        case=CASES[1], outcome=None, error="boom", scores=[Score(name="m", value=True)]
    )
    assert (
        default_objective(EvalReport("r", [ok, bad])) == 0.5
        and default_objective(EvalReport("r", [])) == 0.0
    )
