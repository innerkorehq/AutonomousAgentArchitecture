# Changelog

## Unreleased

- `aaa-server`, A2A: a task whose `AskHuman` run is waiting for a person is `TASK_STATE_INPUT_REQUIRED` with the question in the status message, and a message naming that `taskId` answers it (needs the `answer` scope; the task's own tenant; one outstanding question, once). A message naming an unknown task is now `TaskNotFound` (it was `UnsupportedOperation`); the default A2A result text now uses an `answer` field as well as `text`.
- `Blackboard` knowledge sources: `Expert(description, when=..., auto=..., repeat=...)` with pure trigger functions of the board (`empty_board()`, `after(author)`, `mentions(word)`). An automatic expert contributes as soon as its trigger holds, without a chair call; otherwise the chair is shown only the experts whose triggers hold now. Plain `{"name": "description"}` experts behave as before.
- Computer use, vision: `Observation` can carry a screenshot (`image`, `image_type`); with one, its fingerprint is the picture's digest, so `ComputerUse`'s stall rule stops on a screen that stopped changing. `ComputerUse(vision=True)` attaches the current screenshot to the prompt (quoted as untrusted); `BrowserEnvironment(screenshots=True)` takes them (JPEG, animations off, caret hidden).
- Browser, DNS rebinding closed: `BrowserEnvironment` sends its traffic through a new `aaa.env.egress.EgressProxy` that resolves each name once, requires public addresses and connects to the address it checked; redirects, scripts, frames and WebSockets are held to the allow-list because they all pass through it. `pin_dns=False` keeps the older in-page checks. Also fixed: a click that started a navigation could make the next observation fail with "Execution context was destroyed".
- CI: GitHub Actions moved to their Node 24 releases (`checkout` 5, `setup-uv` 7, `upload-artifact` 7, `download-artifact` 8, `upload-pages-artifact` 5).
- Fixed: parameters of a `@tool` function written as `Annotated[int, Field(ge=1, le=3)]` lost their constraints (the type hints were read without their metadata), so the schema the model saw and the validation applied had no range, pattern or length. They are kept now.
- `jsonschema` is a required dependency: tool arguments and structured outputs are always validated against their full JSON Schema (ranges, enums, patterns, nested schemas). The weaker fallback that applied when it was not installed is gone, so there is no configuration in which validation is silently reduced.

## 0.5.0 (2026-10-18)

- New algorithms (programme A1): `SelfRefine`, `ChainOfVerification`, `BestOfN` (verifier: pure scorer, tool or model judge), `LeastToMost`, `SelfAsk` (optionally with a lookup tool), `GraphOfThoughts` (a validated graph of generate/aggregate/refine/score/keep_best operations) and `LATS` (UCT tree search with reflections). All are event-sourced, bounded at construction with a known `max_calls`, and independent of the order parallel calls finish in. `aaa.algos.base` gains the shared `Calls` projection and `ask`, `task_text`, `clamp_score`, `data_block` helpers (additive).
- New algorithms (programme A2): `ReWOO` (plan every tool call up front with `#E1` placeholders, run them without the model, one solver call), `LLMCompiler` (the same checked plan plus a joiner that may replan, bounded by `max_replans`) and `HierarchicalAgent` (a manager delegates a validated sub-goal DAG to named workers as child runs, each under its own `Budget`, with `max_depth`). `PlanAndExecute(review_progress=True)` lets the planner revise the remaining steps after each finished step (additive; counts against `max_replans`; a review that fails or is unusable keeps the plan).
- New algorithms (programme A3): `CorrectiveRAG` (grades every retrieved passage, rewrites the query, falls back to an operator-supplied web/enterprise search tool, and answers `NO_SOLUTION` rather than guess), `SelfRAG` (decides whether to retrieve, answers per passage, judges relevance/support/usefulness, rewrites and retries, never presents an unsupported answer as grounded) and `AdaptiveRAG` (routes `direct`/`single`/`iterative` retrieval). Retrieval goes through the tool pipeline (`rag.tools` with a `ToolHandler`, like `RAGAgent`); retrieved text is always quoted data.
- New (programme A4): `SkillLibrary` / `skill_tools` / `VoyagerAgent` (code skills stored only after their own test passes in a sandbox, tenant scoped, run only through approval-gated sandbox tools with arguments passed as data), `MemoryReflection` (Generative-Agents style: importance threshold, salient questions, insights that must cite memories they were shown, written back with provenance; `rank_memories` for recency/importance/relevance), `SelfManagedMemory` / `self_memory_tools` (MemGPT-style core blocks and archival memory as tools) and `aaa.optimize.optimize_prompt` (OPRO/APE-style prompt search over an `EvalSuite` that always reports a held-out score and a diff, never writes configuration, and is bounded before it starts).
- New (programme A5): `MixtureOfAgents` (layered proposers and an aggregator), `Blackboard` (a chair picks which of your experts adds to a shared board), `Auction` (blind parallel bids, a deterministic award rule, the winner runs as a child run under its own budget, fallback to the next bidder), `RolePlaySOP` (roles passing pydantic-validated documents, with bounded repair) and `AskHuman` (answer or ask a person and wait for an `aaa.human.answered` signal for the outstanding question; the wait is bounded by the run's `Budget` deadline).
- New (programme A6): `ComputerUse` (an observe-act loop over an `aaa.env.Environment`, with step, repeat and stall stop rules; actions with side effects wait for approval), `aaa.env` (`Environment` protocol, `FakeEnvironment`, `env_tools`) and `aaa.env.browser.BrowserEnvironment` (Playwright, new `browser` extra: every request and every redirect hop checked against an allow-list before it is sent, a clean context, downloads off, credential fields refused), `CodeAct` (Python as the action, only in a `Sandbox`, every run approved by default, output capped and quoted) and `Anytime` (a ladder of algorithms, escalating only while the run's remaining budget can afford the next rung).
- `aaa-server`: `POST /v1/runs/{id}/answers` (new `answer` scope) answers the outstanding question of an `AskHuman` run; the question is on the event stream as `human.requested`. `SCOPE_ANSWER` is part of `ALL_SCOPES`.
- `CodeAct(tools=...)`: sandboxed code can call tools as `tools.<name>(**arguments)`. The host makes each call through the normal tool pipeline and runs the program again with the recorded answers (replay), so the sandbox keeps no network or channel to the host; only listed tools, validated arguments and a call limit; approval covers the program once and each tool keeps its own. New `aaa.algos.codebridge`. The code must be deterministic up to its calls (checked on replay).
- `VoyagerCurriculum`: Voyager's automatic curriculum (propose a task from the goal, the skills held and the tasks tried; a `VoyagerAgent` attempts it as a child run under its own budget; never the same task twice; bounded by `max_tasks`). `skill_tools` gains a read-only `list_skills`.
- Fixed: a child run started with `engine.start(..., parent=...)` and no `meta` of its own now inherits its parent's `meta` (user, agent), as it already inherited the tenant. Before, tools in child runs (memory, skills) saw only the tenant, so work done on behalf of one user was filed tenant-wide.
- Fixed: `aaa mcp init` / `export` (and `scaffold.config_text_for`) left a stray `x.<format>` file in the working directory, because `dump_spec` was given a file name only to choose a format. New `aaa.mcp.config.spec_text(spec, suffix)` renders without writing; `dump_spec` is unchanged for callers.
- Docs: [threat model for the additional algorithms](../security/algorithms.md) (written ahead of the code) and a pre-ship checklist for custom algorithms.

## 0.4.0 (2026-10-18)

**Generated MCP servers** (see [the guide](../guides/mcp-generator.md) and [threat model](../security/mcp-generator.md))

- `aaa mcp serve --config server.toml|json|yaml`, `aaa mcp check`, `aaa mcp inspect`: an MCP server compiled from a validated `MCPSpec` (file, `AAA_MCP__*` environment, flags). Choose, rename, prefix, tag-filter and describe tools; per-tool titles and hints; approval-gated tools stay hidden unless you say otherwise; an import kill-switch.
- Structured output: tools returning pydantic models, TypedDicts or dataclasses advertise an `outputSchema` and return validated `structuredContent`; titles, annotations, server instructions and version are now sent by stdio and by `aaa-server`'s `POST /mcp` from one set of wire shapes (`aaa.mcp.wire`).
- `Tool.output_schema`, `SchemaTool(output_schema=)`, `MCPToolServer(title=, version=, instructions=, tool_meta=, structured_output=)` (additive; defaults unchanged).
- Algorithms as tools: `[[agents]]` in the configuration (or `MCPServerBuilder.agent(...)`) exposes an algorithm as its own tool with its own input schema, a mandatory budget with a deadline, an input template restricted to validated arguments, a concurrency cap, and tenant/user from the host. `runtime.engine` lets `aaa mcp serve` build the engine.
- Other MCP servers as a source (`type = "mcp"`, stdio or https): their tools join the one server under the same policy, untrusted by default (approval unless read-only and trusted), minimal subprocess environment, host-bound credentials from `env:` references, a pinned public-address-only HTTP transport on both SDK generations, and `CompiledServer.aclose()` for the connections. `AAA_MCP_ALLOW_SUBPROCESS=0` / `--no-subprocess-sources` kill-switch.
- `aaa mcp client-config --client claude-desktop|cursor|generic [--transport stdio|http]`: the `mcpServers` entry for the single server, with environment variable names as placeholders and never values. `.env.example` of a generated project now also lists the variables an upstream MCP source reads.
- Configuration `include = [...]` (contained in the top file's directory, depth/size capped, cycles refused), `--overlay FILE` / `AAA_MCP_OVERLAY` layers, and `aaa mcp flatten`. `sources`, `agents`, `resources`, `prompts` and `tools.policies` concatenate; everything else merges or is replaced by the later file. `load_spec(..., overlays=)` is additive.
- Per-tool policy (`[[tools.policies]]`, or the same fields in a tool override): `timeout_s`, `rate_limit` (token bucket per caller and tool), `max_concurrency`, and `scopes` / `tenants` visibility. A tool hidden from a caller answers exactly like an unknown one; `aaa-server` applies the API key's scopes. `tenancy.scopes` for stdio. Additive: `MCPToolServer(tool_policy=, default_scopes=, clock=)`, `list_tools(tenant=, scopes=)`, `call(..., scopes=)`.
- `aaa mcp init` / `aaa mcp export`: generate a project (configuration, `server.py`, requirements, Dockerfile, `.env.example`, smoke test) around a server configuration, interactively or with flags; `--check` and `--update` track drift from the templates. No user text in generated code, no secret value on disk. Relative paths in a configuration now resolve against the configuration file.
- OpenAPI 3.0/3.1 documents as tools (`type = "openapi"`): one tool per operation through the SSRF-safe fetcher with a mandatory host allow-list, credentials from `env:` references sent only to the API's host, state-changing operations requiring approval, and unsupported features skipped with a note. `SafeFetcher.request()` (any method, body, host-bound credentials, same-host redirects) is new and additive.
- Python modules as tools (`type = "module"`, `MCPServerBuilder.add_module`): an explicit allow-list or `__all__`, schemas from type hints, docstring descriptions, approval required by default, never called during inspection.
- Resources (`static`, tenant-scoped `files`, `index`, `memory`) and prompts (typed arguments), served over stdio and `aaa-server`'s `POST /mcp` (`resources/list`, `resources/templates/list`, `resources/read`, `prompts/list`, `prompts/get`); every unreadable resource gives the same `resource not found`.
- `MCPServerBuilder`: the same configuration as typed, chainable Python; `dump_spec`/`load_spec` write and read JSON, TOML and YAML (round-trip tested).
- **Behaviour change:** `MCPToolServer.call` now enforces each tool's `timeout_s` (default 60 s), as `ToolHandler` already did; a tool that needs longer must set a longer `timeout_s`.
- **The `mcp` extra now requires `mcp>=1.21.2`.** Older 1.x releases no longer import with current pydantic. New optional extra `mcp-config` (PyYAML) for YAML configurations.

## 0.3.1 (renamed)

**The project is now "Autonomous Agent Architecture" (AAA).** It is much more than a set of algorithms, and the name now says so. What changed:

- The PyPI distribution is **`autonomous-agent-architecture`** (it was `autonomous-agent-algos`, whose last release is 0.3.0). `pip install autonomous-agent-architecture`.
- The GitHub repository is `innerkorehq/AutonomousAgentArchitecture` and the documentation is at <https://innerkorehq.github.io/AutonomousAgentArchitecture/>. The old repository address redirects; the old documentation address does not.
- **The import name is unchanged: `import aaa`.** No code changes are needed beyond the package name in your requirements. The code is otherwise identical to 0.3.0.

## aaa-server 0.2.0 (2026-10-18)

Requires `autonomous-agent-architecture>=0.5.0`.

- `POST /v1/runs/{id}/answers` and the new `answer` scope: a person answers the question an [`AskHuman`](../guides/algorithms/multi-agent-human.md) run is waiting on. Tenant-scoped like every run endpoint, only the outstanding question is accepted (once), the answerer is recorded, and the event stream shows `human.requested` (the question, never the answer).
- `POST /mcp` applies the caller's API-key scopes to per-tool policy: a tool restricted by `scopes` in an MCP server configuration is listed and callable only for keys that hold one of them, and answers like an unknown tool otherwise.
- `SCOPE_ANSWER` is part of `ALL_SCOPES`.

## aaa-server 0.1.0 (2026-10-16)

First release of the separate [`aaa-server`](../server/index.md) package (it requires `autonomous-agent-architecture>=0.3.1`). Everything in the "HTTP server" and "MCP server and A2A" entries under 0.3.0 below ships here, plus: `POST /mcp` and the experimental A2A endpoint, the `Dockerfile`/`compose.yaml` deployment, and the fix that builds an `async` factory inside the server's own event loop.

## 0.3.0 (2026-10-16)

A large release: everything below is additive to the 0.2 API and log format (see [Upgrading](../operations/upgrading.md)).

**Operations** (see [backup and restore](../operations/backup-restore.md), [upgrading](../operations/upgrading.md), [performance](../operations/performance.md), [deployment](../operations/deployment.md))

- `aaa store verify <store> [--blobs DIR] [--deep]` / `aaa.events.verify.verify_store`: a read-only consistency check of every run's log and blob references, for after a restore. Verified on a restored Postgres dump.
- `scripts/bench_cluster.py` and measured Redis/Postgres multi-worker throughput (with the connection-budget finding).
- `Dockerfile`, `compose.yaml` and `deploy/app.py`: the server in a non-root container with Postgres; smoke-tested end to end. No Helm chart.
- `RedisEventStore.create(max_connections=...)`: the pool now scales with the partition count; the redis-py 8 default of 100 starved a 16-partition worker (`Too many connections`).
- **Fix (`aaa-server`)**: an `async` factory given to `aaa-server serve` was run in a throw-away event loop, so a connection pool it opened failed with "Event loop is closed" on the first request. The configuration is now built inside the server's own loop.

**Evals** (see [the guide](../guides/evals.md))

- Scores now reach the backends: `LangfuseSink`, `PhoenixSink`, `LangSmithSink` and `OTelSink` implement `score()` (`OnlineEvaluator` passes the run's span). Verified against real Langfuse 2.x and Phoenix servers; LangSmith by mock only.
- `Dataset.from_traces` (scrubbed by default), `aaa.eval.redteam` (attack corpus, `evaluate_guard`, `attack_suite`, `NoLeak`, `Resisted`), and `SimulatedUser` + `SessionEvalRunner` for multi-turn evaluation. `CaseResult.turns`.

**New algorithms** (see [the guide](../guides/algorithms/reasoning.md))

- `Reflexion` (fresh attempts informed by written lessons; validator or LLM critic; lessons returned and re-injectable), `Debate` (parallel agents, rounds, judge or vote; at most 7 agents and 4 rounds), `Swarm` with `Agent` (handoffs as tool calls: an agent can only use its own tools, one transfer per turn, capped transfers) and `TreeOfThoughts` (beam search with a hard call bound).
- `LLMLoop.request_for(state, messages)` hook (ordinary loops are unchanged); `LoopState` gains `agent`/`handoffs`, omitted from its serialized form when unused so existing logs and snapshots fold identically. New events `aaa.swarm.handoff` and `aaa.reflexion.feedback`.

**MCP server and A2A** (see [serving tools over MCP](../guides/mcp.md#serving-your-tools-over-mcp) and the [server guide](../guides/server.md))

- `aaa.mcp.MCPToolServer` / `serve_stdio` / `aaa mcp serve`: expose a `ToolRegistry` to any MCP client (SDK 1.x and 2.x). Approval-gated tools are hidden unless you provide an `approval=` callback; failures never leak internals; tenant and user are fixed by the host; `run_agent` runs algorithms under a server-side budget and is opt-in.
- `aaa-server`: `POST /mcp` (stateless Streamable HTTP subset, behind the server's auth and limits, Origin-checked) and **experimental** A2A 1.0 (`A2AConfig`: agent card, `SendMessage`, `SendStreamingMessage`, `SubscribeToTask`, `GetTask`, `ListTasks`, `CancelTask`), both verified against the official `mcp` and `a2a-sdk` clients.

**HTTP server** (new package `aaa-server`; see [the guide](../guides/server.md) and [threat model](../security/server.md))

- Starlette/ASGI app with SSE: runs (idempotent start, cancel, resumable event stream via `Last-Event-ID`), tool approvals over HTTP (`approve` scope, pending calls only), chat sessions with streaming replies. Tenant, user and metadata come only from the authenticated principal; foreign ids are indistinguishable from missing ones; the event stream is a filtered projection; budgets are a server-side ceiling.
- Authentication: `ApiKeyAuthenticator` (hashed, constant-time) and `JWTAuthenticator` with `JWKSKeys` (algorithm allow-list, required `exp`/`iss`/`aud`/`sub`, rate-limited key refresh). Admission: `InMemoryRateLimiter`, `RedisRateLimiter` (atomic Lua script), event-sourced `QuotaStore` with race-free reservation, open-run and stream caps.
- `aaa-server serve module:factory` (loopback by default). The core library gains a read-only `aaa` command: `aaa store check|runs|events`.
- Core is now `0.3.0.dev0`; `aaa-server` is versioned and released separately (`server-v*` tags).

**Built-in tools and a code sandbox** (see [the guide](../guides/builtin-tools.md) and [threat model](../security/tools.md))

- `aaa.tools.builtin`: `http_fetch_tool` (SSRF-hardened: every resolved address must be public, the connection is pinned to the validated address, every redirect re-checked, bodies capped while streaming), `read_file_tool` / `list_files_tool` (confined to a root, secrets and hidden files refused), `web_search_tool` with `SearXNGSearch` (verified against a real SearXNG) and `BraveSearch`.
- `aaa.sandbox`: `DockerSandbox` (no network, read-only filesystem, no capabilities, resource and output limits, nothing mounted from the host, a new container per run), the dev-only `SubprocessSandbox(unsafe=True)` and `code_exec_tool`, which **requires human approval by default**.

**Ingestion** (see [the guide](../guides/ingestion.md))

- `aaa.ingest`: `DirectoryLoader` / `FileLoader` / `PDFLoader` (extra `pdf`), `RecursiveChunker` (characters or tokens) and `MarkdownChunker`, and `Pipeline.sync()`: incremental (unchanged sources skipped, only changed chunks re-embedded, stale chunks and removed sources deleted), batched and retried embedding, a manifest kept in the store itself, per-tenant, resumable. Run against pgvector, sqlite-vec, Qdrant, LanceDB, Chroma and Pinecone Local in CI.
- `VoyageReranker` (Voyage AI's rerank endpoint). A Jina reranker is not included: Jina's documentation does not specify its response format.

**Hybrid search**

- `HybridSearch.query_hybrid` is now implemented by `InMemoryVectorStore` (BM25), `PgVectorStore` (full-text `tsv` column, added automatically) and `SqliteVecStore` (FTS5 via triggers), fusing keyword relevance with similarity by reciprocal rank. `EmbeddingIndex(..., hybrid=True|"auto")` / `search(..., hybrid=)`; `Hit.metric == "rrf"`. New `HybridUnavailable`.

**Native providers** (see [LLM gateways](../guides/llm-gateways.md))

- `provider="anthropic"` (`AnthropicClient`: Messages API, tool use, forced-tool structured output, images and PDFs, prompt caching, streaming) and `provider="gemini"` (`GeminiClient`: Generative Language API, function calling, JSON-schema output, images/audio/files, streaming, embeddings). Plain httpx, no vendor SDK; shapes validated against the vendors' SDK types in CI.

**Multimodal messages** (see [the guide](../guides/multimodal.md) and [threat model](../security/multimodal.md))

- `Message.parts`: images, audio and files after the text (`aaa.llm.parts`: `ImagePart`, `AudioPart`, `FilePart`), validated at creation (type vs kind, image signatures, https-only URLs, size). Text-only messages serialize exactly as before.
- The engine moves attachment bytes into the blob store before an event is appended (salted per run, so erasing a run deletes exactly its media); the log keeps references. Without a blob store, attachments over 256 KiB are refused. `engine.start` now stores `input` in its JSON form.
- Clients declare `modalities`; an unsupported attachment fails the call with `UnsupportedModality` instead of being dropped. OpenAI chat-completions and Responses clients send images, audio (chat) and files.
- `Session.send(..., attachments=)`, `{"attachments": [...]}` as an algorithm input, `GuardContext.has_media`, attachment-aware token counting, the `ClearMedia` context strategy (in the default policy), and trace/redactor handling that never exposes or corrupts attachment bytes.

**Event-format safety net** (see [the guide](../guides/evolving-events.md))

- Frozen fixtures of real v0.2.0 logs (every built-in event type, 24 scenarios) and the Projection state they fold to, replayed by the current code on every CI run (`tests/unit/test_event_compat.py`); the trace and CloudEvents exporters are checked against them too.
- The SQLite, PostgreSQL and Redis stores record a layout version (`await store.schema_version()`): a store written by a newer release is refused (`SchemaTooNew`), 0.1/0.2 stores are adopted as version 1, and a migration framework (ordered, transactional, run once under the setup lock) is in place for the first layout change. Stores now also close their connection pool / thread when opening fails.

**Policy and hygiene**

- [Stability and deprecation policy](../policy/stability.md): versioning while 0.x, stable vs provisional vs experimental modules, an additive-only rule for the event log, and a two-minor-release deprecation window (`aaa._compat.deprecated`).
- `SECURITY.md`, a root `CONTRIBUTING.md`, Dependabot for Actions and Python dependencies; Python 3.13 is now tested in CI.
- Docs: deployment guide no longer says OpenTelemetry is missing, and states that schema upcasting happens on write, never on read.

## 0.2.0 (2026-10-15)

Observability, evals, guardrails and persistent vector search, each with adapters for the common libraries and services. New optional extras: `otel`, `langfuse`, `langsmith`, `phoenix`, `autoevals`, `pgvector`, `qdrant`, `sqlitevec`, `lancedb`, `chroma`, `pinecone`, `bedrock`. Additive: new `OutcomeKind.BLOCKED`, `Hit.metric` / `Hit.higher_is_better`, `LLMHandler(guards=)` and `ToolHandler(guards=)`.

**Observability** (see [the guide](../guides/observability.md))

- `aaa.observe.Observer`: derives traces from the event log (`aaa.trace.spans_from_events`) and exports them whole, parents first, once per cluster, without ever failing a run. Sinks: `InMemorySink`, `ConsoleSink`, `CallableSink`, `OTelSink` and presets `LangfuseSink`, `LangSmithSink`, `PhoenixSink` (extras `otel`, `langfuse`, `langsmith`, `phoenix`). Content capture is opt-in; sampling, tenant filter, incremental export of open sessions, `backfill`.

**More guard adapters**

- `BedrockGuard` (`ApplyGuardrail`), `AzurePromptShieldGuard`, `LakeraGuard`, `GuardrailsAIGuard` and `NeMoGuard`, with the same stage and action semantics as the built-in guards. Extra `bedrock`.

**More vector stores**

- `SqliteVecStore` (one file, exact, filters in SQL), `LanceDBStore` (embedded; exact filters evaluated client-side in growing batches) and `ChromaStore` (embedded or server; refuses what Chroma cannot express), `PineconeStore` (own namespaces, waits for eventual consistency), all through the same contract suite. Extras `sqlitevec`, `lancedb`, `chroma`, `pinecone`.

**Vector stores** (see [the guide](../guides/vector-stores.md))

- `aaa.vector`: `VectorStore` protocol (vectors + payload, namespaces, dimension checks), `EmbeddingIndex(store, embedder)` implementing the existing `Index` (so `RAGAgent`, `search_tool` and `NativeMemory` run on a persistent store), a small filter language with identical semantics in every store (`UnsupportedFilter` instead of silent post-filtering), `TenantVectorStore` (namespace wall plus an outer tenant filter), `RerankingIndex`.
- Stores: `InMemoryVectorStore`, `PgVectorStore` (pgvector over asyncpg, HNSW, jsonb filters) and `QdrantStore` (server or in-process), all run through one shared contract suite. `Hit` gains `metric` and `higher_is_better`. CI's PostgreSQL image is now `pgvector/pgvector:pg16`.

**Guardrails** (see [the guide](../guides/guardrails.md))

- `aaa.guard`: `GuardPolicy` + `LLMHandler(guards=)` / `ToolHandler(guards=)` check `input`, `output`, `tool_call` and `tool_result`. Actions allow / flag / redact / approve / block; a blocked run ends as the new `OutcomeKind.BLOCKED`, a blocked tool call is a normal tool failure, a poisoned tool result is replaced by a placeholder, `approve` uses the existing approval flow.
- Every check leaves an `aaa.guard.verdict` event (content hash, never content; counted in spend; shown as a guardrail span). The stored verdict wins, so a classifier is not called twice for the same content, including after a crash; a guard failure follows its `on_error` policy and is never reused. `GuardWatcher` runs guards in observe mode.
- Guards: regex, deny-list, max length, tool allow-list / escalation, `LLMJudgeGuard` (any model, Llama Guard categories), callable; adapters `OpenAIModerationGuard`, `PresidioGuard` and `PresidioRedactor`.

**Evals** (see [the guide](../guides/evals.md))

- `aaa.eval`: `Dataset`/`Case` (list, JSONL, CSV), `EvalSuite` (real runs; concurrency, repeats, per-case budget; trajectory, steps, tokens and cost read back from the log), built-in metrics (text, outcome, tool use, budgets, semantic similarity, `LLMJudge`, callables), `EvalReport` (summaries, baseline regression, `assert_ok` for CI, Markdown/JUnit), and `OnlineEvaluator` (sampled, cost-capped scoring of live runs, recorded as `aaa.eval.score` events and sent to score-capable sinks).

- Adapters for autoevals, DeepEval, Ragas (metrics judged through your own `LLMClient`) and Inspect AI (`aaa_solver`).

- Foundations: `aaa.score.Score`, `aaa.safe.safe_subscribe` (isolated, bounded observer workers).

## 0.1.1 (2026-10-15)

**Chat sessions and provider-side state** (see [the guide](../guides/sessions.md))

- `aaa.session.SessionManager` / `Session`: a durable multi-turn conversation. A session is a long-lived run; each user message starts a child run of your algorithm; the reply is recorded by whichever worker finishes the turn, so a turn survives the process that sent it. One turn at a time, per-tenant namespacing, whole-conversation budget, streaming, `context=` for long histories (summaries persisted and reused), and erasure/retention as for any run.
- Provider-held conversation state: `OpenAIResponsesClient` (`provider="openai_responses"`) plus a small `ServerStateClient` protocol. The `LLMHandler` continues from the provider's handle and sends only new messages (within a run and across session turns), falls back to a full request when the state is gone or the context policy rewrote the conversation, and `engine.erase` deletes provider-held state (`ErasureReport.provider_state`).
- `Usage.cached_tokens` and `price_cached_in_per_mtok`; `Engine.await_event`; `FakeStatefulClient` for tests.

**Release process**

- Releases publish to PyPI from GitHub through trusted publishing (no stored token); the release workflow skips a version that is already on PyPI.

## 0.1.0 (2026-10-15)

**Context management** (see [the guide](../guides/context.md))

- `aaa.context.ContextPolicy` + `LLMHandler(client, context=policy)`: each request is fitted to the model's window as a send-time *view*; the logical conversation, the delta requests and the log are untouched.
- Strategies: `TruncateToolResults`, `ClearToolResults`, `Summarize` (incremental; persisted as `aaa.context.summary` events, reused by content hash across restarts and workers, billed to the run), `DropOldest`. Tool calls stay with their results; the head and recent turns are never touched.
- Calibrated token estimates (`CalibratedCounter` learns from `usage.prompt_tokens`; optional `TiktokenCounter`); trigger/target hysteresis; one deeper retry when the provider still reports an overflow.

**MCP client** (see [the guide](../guides/mcp.md))

- `aaa.mcp.MCPClient`: connect to any MCP server over stdio, Streamable HTTP, or your own open `Client`/`ClientSession`, and use its tools as AAA tools (`SchemaTool`: argument validation against the server's JSON Schema, timeouts, approval, execution guard). Works with both the 1.x and 2.x `mcp` SDK.
- Safety policy from MCP annotations: read-only/idempotent tools are `at_least_once`, everything else `at_most_once`; `approval="destructive"` follows the spec's "no annotations means destructive"; `trust_annotations=False`, `include`/`exclude`, result truncation.
- `mcp_memory(client, add=..., search=...)`: any MCP memory server as an AAA `Memory`. Resources can be listed and read.
- Tested against real SDK servers in-process, over stdio and over HTTP with credentials, on both SDK versions in CI.

**Long-term memory** (see [the guide](../guides/memory.md))

- `aaa.memory`: a small `Memory` protocol (`add`, `search`) with optional capabilities (`Updatable`, `Deletable`, `Forgettable`, `Profiled`, `Listable`) and a tenant/user/agent `Scope`.
- Backends: `NativeMemory`, `Mem0Memory` (open source and hosted), `GraphitiMemory` (temporal, point-in-time recall), `SupermemoryMemory` (HTTP), `LangGraphStoreMemory` (any LangGraph store, LangMem records read natively), `CallableMemory` for any other library, `FakeMemory` for tests.
- Event-native: `MemoryHandler` (`aaa.memory.requested` → `completed`/`failed`, replayable, redacted, crash-safe writes), `ToolLoop(recall=True)` automatic recall, `WriteBack`, `memory_tools`.
- Erasure: `engine.forget_memory(scope)`, `TenantScope.forget_memory(user=...)`, and `erase_all()` reaches registered memory backends.
- Also: run metadata (`start(meta=...)`, `TenantScope.start(user=...)`), `ToolContext.tenant`/`meta`, `LLMEmbedder`, optional extras `mem0`, `graphiti`, `langgraph`, `langmem`.
- CI: an `adapters` job replays each adapter's calls against the real libraries' signatures and runs weekly against their newest releases.

**Privacy: tenancy, redaction, retention** (see [the guide](../guides/privacy.md))

- `Event.tenant`, inherited by every derived event; `engine.tenant(name)` scopes that namespace run ids and check ownership on every lookup; `Engine(require_tenant=True)`.
- `Engine(redactor=...)` with `PatternRedactor` (e-mail, Luhn-checked cards, SSN, API keys, JWT, bearer), `KeyRedactor` and `standard_redactor()`: applied before persistence and delivery.
- `engine.erase` (scrub or delete), `engine.purge` and `Engine(retention=[RetentionPolicy(...)])` applied by `sweep()`. `EventStore` gained `scrub` and `delete_stream` (all four backends), which also remove
  snapshots and the bus outbox copies. Claim-checked blobs are now namespaced per stream so erasing one run cannot affect another.

**Quality gates**

- CI: ruff, `pyright` (0 errors over the library, examples and scripts) and the full suite on Python 3.11 and 3.12 against real Postgres and Redis.
- Type-correctness fixes found by pyright: the `EventBus` and `EventStore` protocols now declare what the engine and users rely on (`aclose`, `drain`, `abort`, `execution_guard`);
  `Algorithm.on_event` and `LLMClient.complete/stream` take positional-only parameters so any implementation naming (`e`, `request`, ...) type-checks (previously every built-in
  algorithm was rejected when passed to `Engine.run`); `Outcome.value` no longer forces narrowing before indexing.
- Python 3.11: long-lived loops could outlive a cancellation (the standard library's `asyncio.timeout`, used by redis-py, can swallow a cancel that races its deadline), which made
  `abort()`/shutdown hang with Redis. Loops now re-raise a swallowed cancel, and `aaa.core.wait_for`/`wait_event` replace `asyncio.wait_for` where tasks are shut down by cancellation.

**Resilience**

- Store outages no longer fail runs or lose reactions: bounded retry with backoff (`Engine(store_retry=...)`), `Redeliver` so the bus delivers an event again, a tool
  result retried until recorded (`ToolHandler(result_retry=...)`), and distributed-bus consumers that survive backend errors. Fixed a race where a tool task cancelled before its first step
  left its bus message unacknowledged. Tested with real dropped Postgres and Redis connections.

First public version. Alpha: see [Status and limitations](limitations.md).

**Runtime and data**

- Event-sourced engine: append-only log, deterministic ids, idempotent appends, budgets, deadlines, cancellation, child runs with spend roll-up, signals.
- Projections with versioned snapshots; incremental state for every built-in algorithm.
- Event stores: in-memory, SQLite, PostgreSQL, Redis, all passing one contract suite. CloudEvents 1.0 conversion. Versioned payload schemas with upcasters.
- Claim-check blob storage (in-memory, file) with integrity verification. In-memory hybrid (BM25 + vector) index.
- `Item`/`Codec` data layer with capability preflight.

**Delivery and clustering**

- In-process bus and a partitioned `DistributedBus` over Postgres (transactional outbox, `LISTEN/NOTIFY`, advisory-lock leases) and Redis (Streams + Lua), with consumer groups,
  cursors, fair-share rebalancing, durable acknowledgement and `ClientBus` for submit-only processes.
- Single-process crash recovery (`recover()`), cluster `sweep()`, adoption of any run by any worker.

**LLM layer**

- OpenAI-compatible client with streaming, tool calls and structured output; presets for LiteLLM proxy, Portkey, Cloudflare, Vercel, Bifrost, Kong, MLflow; LiteLLM SDK adapter;
  retry and circuit-breaker wrappers; typed errors; fake and replay clients; delta requests so long conversations stay linear.

**Algorithms**

- `ToolLoop`, `ReAct`, `PlanAndExecute`, `Router`, `Supervisor`, `GenerateTestRepair`, `GeneratorCritic`, `WorkflowGraph`, `FSM`, `SelfConsistency`, `RAGAgent`.

**Tools**

- Typed `@tool` with schema generation, pools, timeouts, human approval, and the **execution guard** (durable acknowledgement, execution lease, fencing attempts,
  `at_least_once`/`at_most_once`, idempotency keys), fully configurable globally and per tool.

**Docs and tooling**

- This documentation site, with every example executed by the test suite. License: MIT.
