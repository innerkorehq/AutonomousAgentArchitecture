# Corrective, Self- and Adaptive RAG

[`RAGAgent`](ensembles-rag.md) lets the model decide when to search and trusts what comes back. These three algorithms put the **checking** in the algorithm: passages are graded or judged before they are used, a poor result triggers a rewritten query, and the amount of retrieval is matched to the question. All three read your documents through the normal tool pipeline, exactly like `RAGAgent`:

```python
rag = CorrectiveRAG(index, k=5, web_tool=my_web_search)
ToolHandler(rag.tools).attach(engine)      # the search tool (and the fallback tool) live in rag.tools
```

Like every [algorithm](custom.md), each is a fold of the run's log: parallel grading calls may finish in any order without changing the result, a recorded run replays to the same outcome, and `max_calls` is the most model calls one run can make. Retrieved text is shown to the model only as quoted data (`<passage>...</passage>`), and a passage cannot close its own quote.

```python
--8<-- "examples/retrieval_algorithms.py"
```

```text
corrective: Answered from web:w1 | route web | sources ['web:w1']
corrective: Answered from fr | route index | graded 2 passages
self-rag: Answered from fr | supported full | from fr
adaptive: iterative with 2 searches; Answered from fr, de
```

## CorrectiveRAG

`CorrectiveRAG(index, k=5, web_tool=None, min_relevant=1, max_rewrites=1)`:

1. Search the index; **grade each passage** (`{"relevant": bool}`, one call per passage, in parallel).
2. At least `min_relevant` relevant: answer from those passages only, with citations.
3. Otherwise **rewrite the query** and search again (up to `max_rewrites`).
4. Still not enough and a `web_tool` was given: call it with the last query, grade its results the same way (their ids are prefixed `web:`), and answer from the relevant ones plus any relevant index passages.
5. Nothing relevant anywhere: the run ends `NO_SOLUTION` with the queries tried. It never answers without evidence.

`web_tool` is *your* fallback: any [`Tool`](../tools.md) that takes a `query` argument and returns a list of `{"id", "text"}` dicts (or text). It is called through the tool pipeline, so approval, timeouts and guards apply, and it only ever receives the rewritten question as its query. Result `{"text", "sources", "queries", "route", "graded"}` with `route` `index`, `rewritten` or `web`.

## SelfRAG

`SelfRAG(index, adaptive=True, max_candidates=3, max_rewrites=1)`:

1. With `adaptive`, a first call decides whether the question needs retrieval at all (`route: "direct"` answers without it).
2. Search; write **one candidate answer per passage**, so no answer mixes sources.
3. Judge every candidate with reflection fields: is the passage `relevant`, is the answer `supported` by it (`full`, `partial`, `none`), how `useful` is it (1 to 5). The best supported, relevant candidate wins (ties go to the earlier passage).
4. Nothing supported: rewrite the query and repeat (`max_rewrites`), then end `NO_SOLUTION` carrying the best candidate, so an unsupported answer is never passed off as grounded.

Result `{"text", "source", "supported", "useful", "queries", "rounds", "route"}`.

## AdaptiveRAG

`AdaptiveRAG(index, k=5, max_hops=3)`: a routing call picks how much retrieval the question deserves.

- `direct`: common knowledge, answered without searching.
- `single`: one search, then answer from those passages.
- `iterative`: up to `max_hops` rounds of "search again or answer" for questions that need several facts; passages accumulate (de-duplicated by id), and a repeated query ends the search.

An unusable routing reply falls back to `single`. Result `{"text", "route", "hops", "sources", "queries"}`.

## Choosing

| Situation | Use |
|---|---|
| The index is good but sometimes misses, and you have a second source | `CorrectiveRAG` with a `web_tool` |
| Answers must be traceable to one passage and you want unsupported ones refused | `SelfRAG` |
| Questions vary from trivia to multi-hop | `AdaptiveRAG` |
| You just want the model to search as it sees fit | [`RAGAgent`](ensembles-rag.md) |

Grading and judging are model calls: they reduce, not remove, wrong answers. Use a separate `grader_model` if the answering model is the one you distrust.
