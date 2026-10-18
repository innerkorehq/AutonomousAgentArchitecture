"""Improve a prompt from evaluation results (OPRO / APE style), and say honestly whether it helped.

    result = await optimize_prompt(engine, build, dataset, metrics, proposer, baseline="You answer questions.")
    result.improved, result.best, result.best_holdout, result.baseline_holdout

The loop: score the baseline prompt on a **dev** part of the dataset with an `EvalSuite`; ask a model (`proposer`) for improved variants, shown the current best
prompt, its score and some failing cases; score each variant on the dev part; keep the best. At the end, baseline and best are both scored on a **held-out**
part the search never saw, and that is the number to believe: `improved` is true only if the best prompt beats the baseline there.

What it does not do: it never changes your configuration (it returns a candidate and a diff for you to review), it never runs more than `max_evaluations`
case runs (checked before anything starts), and the examples it shows the proposer are quoted as data. A prompt that wins on a small dataset can still be a
coincidence: `dev` and `holdout` sizes are in the result so you can judge."""

from __future__ import annotations

import difflib
import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from aaa.algos.base import data_block, parse_json_obj
from aaa.eval.dataset import Dataset
from aaa.eval.metrics import Metric
from aaa.eval.report import EvalReport
from aaa.eval.suite import EvalSuite
from aaa.llm.client import LLMClient
from aaa.llm.types import LLMRequest, Message
from aaa.runtime import Algorithm, Budget, Engine

log = logging.getLogger("aaa.optimize")
MAX_ROUNDS, MAX_VARIANTS, MAX_PROMPT_CHARS = 6, 8, 4000


@dataclass
class Candidate:
    prompt: str
    dev: float
    round: int  # 0 is the baseline


@dataclass
class OptimizeResult:
    baseline: str
    best: str
    baseline_dev: float
    best_dev: float
    baseline_holdout: float
    best_holdout: float
    improved: bool  # the best prompt beat the baseline on the held-out cases
    candidates: list[Candidate] = field(default_factory=list)
    dev_cases: int = 0
    holdout_cases: int = 0
    evaluations: int = 0  # case runs used
    diff: str = ""  # unified diff, baseline -> best


def split_dataset(dataset: Dataset, holdout: float = 0.3, seed: int = 0) -> tuple[Dataset, Dataset]:
    """A deterministic dev / held-out split: cases are ordered by a hash of (seed, id), so it does not depend on file order."""
    if not 0.0 < holdout < 1.0:
        raise ValueError("holdout must be between 0 and 1")
    ordered = sorted(
        dataset.cases, key=lambda c: hashlib.sha256(f"{seed}:{c.id}".encode()).hexdigest()
    )
    n_hold = max(1, round(len(ordered) * holdout))
    if len(ordered) - n_hold < 1:
        raise ValueError(
            "the dataset is too small to split: need at least one dev case and one held-out case"
        )
    return Dataset(ordered[n_hold:], f"{dataset.name}-dev"), Dataset(
        ordered[:n_hold], f"{dataset.name}-holdout"
    )


def default_objective(report: EvalReport) -> float:
    """The mean of the metrics' mean scores (booleans count as 0 or 1); a case the harness could not run counts as a failure."""
    if not report.results:
        return 0.0
    per_case = []
    for r in report.results:
        nums = [s.numeric for s in r.scores if s.numeric is not None]
        per_case.append(sum(nums) / len(nums) if nums and r.error is None else 0.0)
    return sum(per_case) / len(per_case)


def _failing(report: EvalReport, limit: int = 4) -> str:
    rows = []
    for r in report.failures[:limit]:
        rows.append(
            f"input: {str(r.case.input)[:300]}\nexpected: {str(r.case.expected)[:200]}\ngot: {r.text[:300]}"
        )
    return "\n---\n".join(rows) or "(none: every dev case passed)"


async def optimize_prompt(
    engine: Engine,
    build: Callable[[str], Algorithm],
    dataset: Dataset,
    metrics: Sequence[Metric],
    proposer: LLMClient,
    *,
    baseline: str,
    rounds: int = 2,
    variants: int = 3,
    holdout: float = 0.3,
    seed: int = 0,
    objective: Callable[[EvalReport], float] = default_objective,
    max_evaluations: int = 500,
    margin: float = 0.0,
    instructions: str = "",
    proposer_model: str | None = None,
    case_budget: Budget | None = None,
    concurrency: int = 4,
) -> OptimizeResult:
    """Search for a better system prompt for the algorithm `build(prompt)` makes. See the module docstring for the guarantees.

    `build` must return a fresh algorithm each call; give each a distinct `name` if your engine is distributed."""
    if not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"rounds must be between 1 and {MAX_ROUNDS}")
    if not 1 <= variants <= MAX_VARIANTS:
        raise ValueError(f"variants must be between 1 and {MAX_VARIANTS}")
    if not baseline.strip() or len(baseline) > MAX_PROMPT_CHARS:
        raise ValueError(f"the baseline prompt must be 1 to {MAX_PROMPT_CHARS} characters")
    dev, held = split_dataset(dataset, holdout, seed)
    needed = len(dev) * (1 + rounds * variants) + 2 * len(held)
    if needed > max_evaluations:
        raise ValueError(
            f"this search could need {needed} case runs (limit {max_evaluations}): lower rounds, variants or the dataset size, or raise max_evaluations deliberately"
        )
    used = 0

    async def score(prompt: str, part: Dataset) -> tuple[float, EvalReport]:
        nonlocal used
        report = await EvalSuite(
            part, build(prompt), metrics, budget=case_budget, concurrency=concurrency
        ).run(engine)
        used += len(part)
        return objective(report), report

    base_dev, base_report = await score(baseline, dev)
    candidates = [Candidate(baseline, base_dev, 0)]
    best, best_dev, best_report = baseline, base_dev, base_report
    seen = {baseline.strip()}
    for r in range(1, rounds + 1):
        msgs = [
            Message(
                role="system",
                content=(
                    f"You improve prompts for an AI system. Write {variants} improved variants of the current prompt that fix the failures without making other cases worse. "
                    f'Reply with JSON {{"prompts": ["...", ...]}}. {instructions}'.strip()
                ),
            ),
            Message(
                role="user",
                content="\n\n".join(
                    [
                        data_block("current_prompt", best),
                        f"Its score on the development cases: {best_dev:.3f}",
                        data_block("failing_cases", _failing(best_report)),
                    ]
                ),
            ),
        ]
        try:
            reply = await proposer.complete(LLMRequest(model=proposer_model, messages=msgs))
            raw = (parse_json_obj(reply.message.content) or {}).get("prompts")
        except Exception:
            log.warning("optimize: the proposer failed in round %d", r, exc_info=True)
            continue
        new = []
        for p in raw if isinstance(raw, list) else []:
            if (
                isinstance(p, str)
                and p.strip()
                and len(p) <= MAX_PROMPT_CHARS
                and p.strip() not in seen
            ):
                seen.add(p.strip())
                new.append(p.strip())
        for p in new[:variants]:
            s, report = await score(p, dev)
            candidates.append(Candidate(p, s, r))
            if s > best_dev:
                best, best_dev, best_report = p, s, report
    base_hold, _ = await score(baseline, held)
    best_hold = base_hold if best == baseline else (await score(best, held))[0]
    diff = "".join(
        difflib.unified_diff(
            baseline.splitlines(keepends=True),
            best.splitlines(keepends=True),
            "baseline",
            "best",
        )
    )
    return OptimizeResult(
        baseline=baseline,
        best=best,
        baseline_dev=base_dev,
        best_dev=best_dev,
        baseline_holdout=base_hold,
        best_holdout=best_hold,
        improved=best != baseline and best_hold > base_hold + margin,
        candidates=candidates,
        dev_cases=len(dev),
        holdout_cases=len(held),
        evaluations=used,
        diff=diff,
    )


__all__ = ["Candidate", "OptimizeResult", "default_objective", "optimize_prompt", "split_dataset"]
