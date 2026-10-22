"""Run the algorithm benchmark against a local OpenAI-compatible server (omlx, llama.cpp, vLLM, Ollama ...).

    uv run python -m benchmarks.run --model mlx-community--Qwen3-4B-Instruct-2507-4bit --family math --repeats 2

Results are cached per (model, family, algorithm) under benchmarks/results/, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from aaa.eval import EvalSuite
from aaa.llm import LLMConfig, LLMHandler, OpenAICompatClient
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler

from .families import FAMILIES, Entry, Family

RESULTS = Path(__file__).parent / "results"


def _transient(r) -> bool:
    """A run lost to the server (connection refused, model being swapped out), not to the algorithm."""
    if "no relevant evidence" in str(getattr(r.outcome, "reason", "")):
        return False  # CorrectiveRAG declining to answer is a result, not an outage
    if r.tokens == 0 and not r.ok:  # nothing was spent: the first model call already failed
        return True
    why = f"{r.error or ''} {getattr(r.outcome, 'reason', '') or ''}"
    return any(
        k in why
        for k in ("TransientError", "transport error", "HTTP 409", "HTTP 5", "ConnectError")
    )


async def run_entry(
    fam: Family,
    entry: Entry,
    model: str,
    base_url: str,
    repeats: int,
    concurrency: int,
    limit: int | None,
) -> dict:
    client = OpenAICompatClient(
        LLMConfig(
            model=model,
            base_url=base_url,
            timeout_s=300,
            params={"temperature": 0.2, "max_tokens": 900},
        )
    )
    algo = entry.build()
    async with Engine() as engine:
        LLMHandler(client).attach(engine)
        if entry.needs_tools:
            ToolHandler(entry.needs_tools()).attach(engine)
        ds = fam.dataset()
        if limit:
            ds = type(ds).from_list(
                [
                    {
                        "id": c.id,
                        "input": c.input["task"] if isinstance(c.input, dict) else c.input,
                        "expected": c.expected,
                    }
                    for c in list(ds)[:limit]
                ],
                ds.name,
            )
        suite = EvalSuite(
            ds,
            algo,
            [fam.metric()],
            concurrency=concurrency,
            repeats=repeats,
            budget=Budget(max_steps=60, deadline_s=300),
        )
        t0 = time.monotonic()
        report = await suite.run(engine)
        wall = time.monotonic() - t0
    await client.aclose()
    rs = report.results
    ok = [
        bool(r.scores and all(s.value for s in r.scores if not s.metadata.get("error")))
        and r.error is None
        for r in rs
    ]
    errors = sum(1 for r in rs if r.error or not r.ok)
    transient = sum(1 for r in rs if _transient(r))
    n = len(rs)
    mean = lambda xs: statistics.fmean(xs) if xs else 0.0
    tokens = mean([r.tokens for r in rs])
    acc = sum(ok) / n if n else 0.0
    return {
        "algorithm": entry.name,
        "about": entry.about,
        "model": model,
        "family": fam.name,
        "runs": n,
        "accuracy": acc,
        "calls": mean([r.steps for r in rs]),
        "tokens": tokens,
        "latency_s": mean([r.latency_s for r in rs]),
        "wall_s": wall,
        "acc_per_1k_tokens": (acc / (tokens / 1000)) if tokens else 0.0,
        "errors": errors,
        "transient_errors": transient,
        "failures": [
            {
                "case": r.case.id,
                "text": (r.text or r.error or "")[-200:],
                "reason": str(getattr(r.outcome, "reason", None))[:200],
            }
            for r, k in zip(rs, ok, strict=True)
            if not k
        ][:6],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--family", action="append", choices=list(FAMILIES), help="default: all")
    ap.add_argument("--algo", action="append", help="only these algorithm names")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="first N cases only (smoke test)")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    a = ap.parse_args()
    for fname in a.family or list(FAMILIES):
        fam = FAMILIES[fname]()
        if fam.prepare:
            await fam.prepare()
        for entry in fam.entries:
            if a.algo and entry.name not in a.algo:
                continue
            out = RESULTS / a.model / fname / f"{entry.name}.json"
            if out.exists() and not a.force and not a.limit:
                print(f"skip {fname}/{entry.name} (cached)", flush=True)
                continue
            print(f"run  {fname}/{entry.name} ...", flush=True)
            res = await run_entry(
                fam, entry, a.model, a.base_url, a.repeats, a.concurrency, a.limit
            )
            print(
                f"     acc {res['accuracy']:.0%}  calls {res['calls']:.1f}  tokens {res['tokens']:.0f}  latency {res['latency_s']:.1f}s  errors {res['errors']}",
                flush=True,
            )
            if res["transient_errors"] > res["runs"] * 0.05:
                print(
                    "     still losing runs to the server after retries; not saving",
                    flush=True,
                )
            elif not a.limit:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
