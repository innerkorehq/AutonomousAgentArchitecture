# Algorithm benchmark

Runs the algorithms in `aaa.algos` against a real model served by any OpenAI-compatible endpoint (written against [omlx](https://github.com/jundot/omlx) on Apple silicon) and reports accuracy, model calls, tokens and latency per algorithm. The results and how to read them are in the docs: `docs/guides/algorithms/benchmarks.md`.

```bash
uv run python -m benchmarks.run --model <model id> --repeats 2 --concurrency 1
uv run python -m benchmarks.report      # markdown tables from benchmarks/results/
```

* Three task families (`tasks.py`): arithmetic word problems (seeded, answer computed), tool use over an invented lookup table, and retrieval over invented documents with some unanswerable questions. Nothing can be memorised.
* Results are cached per (model, family, algorithm) in `benchmarks/results/`, so an interrupted run resumes. `--force` redoes an algorithm.
* Use `--concurrency 1`: a small local server queues parallel requests, which inflates latency and drops requests.
* A run that spent no tokens, or that failed with a transport error, counts as "lost to the server". The runner waits and retries the algorithm, and refuses to save the result if it keeps happening, so an outage cannot be mistaken for a bad algorithm.
* Not part of CI (it needs a local model).
