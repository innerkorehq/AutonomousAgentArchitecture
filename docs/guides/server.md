# The HTTP server (`aaa-server`)

!!! info "Part of `aaa-server`"
    This page documents **`aaa-server`**, the HTTP server package ([`pip install aaa-server`](https://pypi.org/project/aaa-server/)). Start at the [server overview](../server/index.md); the library docs begin at the [AAA home](../index.md).

`aaa-server` puts an engine behind a network port: bearer authentication, tenant isolation, rate limits, quotas, server-side budgets, human approvals over HTTP, and live event streams (SSE). It is
a **separate package** (`pip install aaa-server`, extras `jwt` and `redis`) so the library stays free of web dependencies; it releases on its own schedule. The design is argued, threat by threat, on
the [server threat-model page](../security/server.md).

```python
--8<-- "examples/server_quickstart.py"
```

```text
no credentials : 401
started        : 202 01M50H4W...
waiting for    : wire_money {'amount': 250}
other tenant   : 404 not_found
approved       : 202
event stream   : ['run.started', 'tool.requested', 'approval.requested', 'tool.completed', 'run.completed', 'end']
outcome        : "Done: sent $250"
```

## Running it

You describe the deployment in a function that returns a `ServerConfig`; the command only adds the listener.

```python
# myapp/server.py
from aaa_server import ServerConfig, JWTAuthenticator, JWKSKeys

def build() -> ServerConfig:
    engine = ...        # your Engine: durable store, LLM handler, tool handler
    return ServerConfig(
        engine=engine,
        authenticator=JWTAuthenticator(issuer="https://id.example", audience="aaa",
                                       keys=JWKSKeys("https://id.example/.well-known/jwks.json")),
        algorithms={"agent": ...},
    )
```

```bash
aaa-server serve myapp.server:build            # listens on 127.0.0.1:8000
```

It binds to loopback by default. Exposing it (`--host 0.0.0.0`) is a flag you type, and it belongs **behind a reverse proxy** that terminates TLS and applies header/body timeouts; the server does
neither. `create_app(cfg)` returns a plain ASGI app if you would rather run it with your own server; then you own the engine's lifetime (`async with engine`), or pass `manage_engine=True`.

## Authentication and who you are

Every route except `/healthz` needs `Authorization: Bearer <credential>`. The credential resolves to a `Principal(tenant, subject, scopes)`, and **that is the only source of tenant and user**: a
request body that names `tenant`, `user`, `meta` or `run_id` is rejected with `422`, and headers like `X-Tenant` are not read. The run's trusted `meta["user"]` is the principal's subject.

| Authenticator | Notes |
|---|---|
| `ApiKeyAuthenticator` | Static keys held as SHA-256 digests (`ApiKey(key_hash=...)`), compared in constant time with no early exit. Keys are tenant- and scope-bound. |
| `JWTAuthenticator` | `exp`, `iss`, `aud`, `sub` required; the algorithm allow-list is yours (asymmetric by default, never `none`, never HS* with a JWKS), `kid` selects a key from your fixed JWKS URL (`jku`/`x5u` are ignored; unknown-`kid` refreshes are rate-limited); the tenant comes from a claim you name. Shared-secret HS256 is available for small setups (secret ≥ 32 bytes). |

Write your own by implementing `Authenticator.authenticate(headers) -> Principal`. Failures are always the same `401` with `WWW-Authenticate: Bearer`; the reason goes to the server log only.

Scopes: `run` (start, read, cancel), `approve` (resolve tool approvals), `answer` (answer the questions an `AskHuman` run is waiting on), `session`, `admin` (read usage).

## Endpoints

| | |
|---|---|
| `POST /v1/runs` | `{"algo": "agent", "input": {...}, "budget": {...}}` → `202` with the run summary. `Idempotency-Key` makes it safe to retry (same key and request → `200` with the same run; same key, different request → `409`). |
| `GET /v1/runs`, `GET /v1/runs/{id}` | The tenant's runs; one run's status, outcome and usage. |
| `GET /v1/runs/{id}/events` | Server-sent events. Each carries `id: <sequence>`; reconnect with `Last-Event-ID` (or `?after=`) to resume exactly after it. A finished run replays and ends. |
| `POST /v1/runs/{id}/cancel` | |
| `POST /v1/runs/{id}/approvals` | `{"call_id": "...", "decision": "grant"\|"deny", "reason": "..."}`. Needs the `approve` scope, and only a call whose approval is **currently pending** is accepted (`409` otherwise): a call id cannot be pre-signed. The approver is recorded in the log. |
| `POST /v1/sessions`, `GET /v1/sessions`, `GET`/`DELETE /v1/sessions/{id}` | Chat sessions ([guide](sessions.md)); `agent` names which configured `SessionManager`. |
| `POST /v1/sessions/{id}/messages` | `{"text": "...", "attachments": [...], "stream": false}`; with `"stream": true` the reply arrives as SSE `delta` events then a `session.reply`. Attachments are the [multimodal parts](multimodal.md), inline or by `https` URL. |
| `GET /v1/usage` | Today's usage and limits for the tenant (`admin`). |
| `GET /healthz` | `{"ok": true}`, unauthenticated, nothing else. |

Run and session ids in the API are the tenant's own (`order-1`); the `tenant/` namespace is internal. A foreign id and a nonexistent id give the **same `404`**, so ids cannot be probed.
Errors look like `{"error": {"code", "message", "request_id"}}` with an `X-Request-Id` header and never a stack trace.

### What the event stream shows

The stream is a **projection** of the log, not the log: `run.started`, `delta` (model text), `tool.requested|completed|failed` (names and call ids, not results), `approval.requested` (with the
arguments, clipped to 4 KiB, so a human can decide), `session.reply`, and `run.completed` (kind and reason; the value is on `GET /v1/runs/{id}`). System prompts, memory, guard internals and raw
model requests never leave. To show more or less, pass `event_view=` (a function from `Event` to a dict or `None`).

## MCP and A2A

- **MCP**: `ServerConfig(mcp=MCPToolServer(registry))` serves the tools at `POST /mcp` (see [serving your tools over MCP](mcp.md#serving-your-tools-over-mcp)). It is behind the same bearer authentication, rate
  limit and body limit as everything else; the tool's tenant and user are the principal's. A browser `Origin` header is refused unless listed in `mcp_origins` (the MCP spec's defence against
  DNS rebinding); non-browser clients send none. `GET`/`DELETE` are `405`.
- **A2A (experimental)**: `ServerConfig(a2a=A2AConfig(algorithm="agent", url="https://agents.example.com", name=..., description=...))` serves one algorithm as an
  [Agent2Agent](https://a2a-protocol.org) agent: the card at `/.well-known/agent-card.json` (public) and the JSON-RPC binding at `POST /a2a` (authenticated), targeting **A2A 1.0**:
  `SendMessage` (blocks until the task finishes, or returns the working task with `returnImmediately`), `SendStreamingMessage`, `SubscribeToTask`, `GetTask`, `ListTasks`, `CancelTask`.
  Tasks are runs: same admission, quota, budget and tenant scope as `POST /v1/runs`. A task that parks on a human approval is `TASK_STATE_INPUT_REQUIRED`; it is approved through the REST
  endpoint by a principal with the `approve` scope. A task whose algorithm is [`AskHuman`](algorithms/multi-agent-human.md) and is waiting for a person is `TASK_STATE_INPUT_REQUIRED` too, with the question in the status message; the client answers by sending a message with that `taskId`, which needs the `answer` scope (a task id that is not waiting for an answer is `UnsupportedOperation`, a foreign or unknown one `TaskNotFound`). Text parts only; push notifications, the extended card, other multi-turn tasks and the gRPC/REST bindings are not implemented. It is marked
  experimental in the [stability policy](../policy/stability.md): the protocol is young, and this surface may change in a minor release.

## Limits

- **Budgets are server-side.** `ServerConfig.budget` is a ceiling; a request may lower a limit (or set one the server left open) but never raise it.
- **Rate limit** per principal: `InMemoryRateLimiter(per_second, burst)` for one process, `RedisRateLimiter(redis, ...)` for several (one atomic Lua script, time taken from Redis). `429` with `Retry-After`.
- **Quota** per tenant per UTC day, event-sourced: `QuotaStore(store, Quota(max_runs=..., max_tokens=..., max_cost=...), overrides={...})`. A run is reserved against the quota by an optimistic-concurrency append before it starts (so racing requests cannot both take the last unit) and its tokens and cost are recorded when it finishes. Because spend is recorded at the end, concurrent runs can overshoot a *cost* quota by what they are still spending; per-run budgets are the hard limit.
- `max_open_runs` (per tenant, counted from the log), `max_streams` (open SSE streams per tenant **per process**), `max_stream_s` (a stream ends after this long; the client reconnects with `Last-Event-ID`, which re-authenticates), `max_body_bytes` (1 MiB).
- `cors_origins` is empty (CORS off) unless you list origins; a wildcard is refused. There are no cookies, so there is no CSRF surface.

## Verified, and not

Covered by the test-suite (ASGI, in-process): the 401 behaviour, scope checks, every endpoint against a foreign id, identity-field spoofing, idempotency, SSE ordering and resume, the approval flow
including pre-signing, budgets, rate limits (fake clock), quota races (12 concurrent requests for 3 units admit exactly 3), JWT pitfalls (`none`, key confusion, missing `exp`, wrong `aud`/`iss`,
key-id floods, `jku`), and the Redis limiter against a real Redis. MCP over HTTP and A2A are checked against the official `mcp` SDK client and the official `a2a-sdk` client over real sockets (the A2A agent card is parsed with the protocol's own generated `AgentCard` type). **Not covered**: a real TLS/proxy deployment, very large numbers of concurrent SSE clients, and a real identity provider (JWKS is
served by a mock transport with keys generated in the test).
