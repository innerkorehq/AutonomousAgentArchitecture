# Threat model: the HTTP server (`aaa-server`)

!!! info "Part of `aaa-server`"
    This page documents **`aaa-server`**, the HTTP server package ([`pip install aaa-server`](https://pypi.org/project/aaa-server/)). Start at the [server overview](../server/index.md); the library docs begin at the [AAA home](../index.md).

Written before the code (milestone M6). The server puts an engine behind a network port. The engine already isolates tenants (`TenantScope`), validates tool arguments, enforces budgets
and asks for approval; the server's job is to make sure **only the right caller reaches those mechanisms, as the right tenant, and cannot use the port to hurt the host or other tenants.**

## Trust boundaries

| Comes from | Trusted? |
|---|---|
| The authenticated **principal** (tenant, subject, scopes) | Yes: it is the only source of tenant and user |
| Request body, query string and headers | **No**: validated, size-limited, and never allowed to name a tenant, user or metadata |
| The model's output and tool results | No (as everywhere in AAA) |
| The event log | Yes, but only a filtered view of it leaves the process |

## Threats and decisions

| # | Threat | Decision |
|---|---|---|
| 1 | **Authentication bypass or timing leaks.** | API keys are stored as SHA-256 digests and compared in constant time; a missing, malformed or unknown credential gives the same `401` with a `WWW-Authenticate: Bearer` header and no hint of which part failed. Keys and `Authorization` headers are never logged. |
| 2 | **Cross-tenant access (IDOR).** Guessing or reusing another tenant's run or session id. | Every operation goes through `TenantScope` / `TenantSessions` bound to the principal's tenant. A foreign id and a nonexistent id both return the **same `404`**, so existence does not leak. A test sends every endpoint a foreign id. |
| 3 | **Tenant or user spoofing** through the body (`{"tenant": "acme"}`), a header (`X-Tenant`) or metadata. | Request models reject unknown fields (`422`). Tenant, user and the run's trusted `meta` come only from the principal. Tests post spoofed fields to every endpoint. |
| 4 | **JWT pitfalls**: `alg: none`, HS/RS confusion, missing `exp`, key-id spam, a token naming its own key URL. | The accepted algorithms are an explicit allow-list set by the operator, never read from the token; `exp`, `iss` and `aud` are required; `jku`/`x5u` headers are ignored (the JWKS URL is fixed by configuration); an unknown `kid` triggers at most one JWKS refresh per interval. |
| 5 | **Resource exhaustion**: huge bodies, floods, endless runs, many open streams. | A body size limit (1 MiB default) enforced before parsing; a token-bucket rate limit per principal; a cap on concurrent runs and open streams per tenant; **budgets are server-side**: the server sets a default `Budget` per tenant and a request may only lower it, never raise it. Slow-client protection (header/body timeouts, TLS) belongs to the reverse proxy in front: the docs say so. |
| 6 | **Approval abuse.** Approving a call that was never requested (pre-signing), or approving without the right. | The approvals endpoint needs the `approve` scope and only accepts a `call_id` for which an approval request is **currently pending** in that tenant's run. |
| 6a | **Answer abuse (`AskHuman`).** Answering a question that was never asked, answering twice, answering as the wrong tenant, or a client with only `run` or `approve` answering. | The answers endpoint needs its own `answer` scope (not implied by `approve`), is tenant-scoped like every run endpoint (a foreign run is the same 404 as a missing one), and accepts a `question_id` only while that question is **outstanding**: `409` otherwise. The answerer's subject is recorded in the log, the answer is capped at 2000 characters, and it reaches the model only as quoted data. The event stream shows the question, never the answer. The A2A endpoint offers the same thing (a message naming a waiting task) under the same rules: the `answer` scope, the task's own tenant, one outstanding question, once. |
| 7 | **Information disclosure through events.** The raw log holds the operator's system prompt, memory contents and guard internals. | The stream is a **projection**, not the log: by default only run lifecycle, token deltas, tool names/outcomes, approval requests and session replies leave, with payloads trimmed. Showing more is an explicit operator setting. Errors are generic with a request id; no stack traces. |
| 8 | **Idempotency-Key abuse.** Replaying a key with a different request, or using keys to collide with others. | Keys are scoped to the tenant (run ids are namespaced), limited in length and charset, and a repeat with a different request body is `409`, not a silent return of the old run. |
| 9 | **CSRF and CORS.** | There are no cookies or ambient credentials; only `Authorization: Bearer`. CORS is off unless the operator lists origins explicitly (never a wildcard together with credentials). |
| 10 | **Long-lived streams outliving their credentials.** | Authentication is checked when a stream opens; a stream ends after a maximum duration (default 15 minutes) and the client reconnects with `Last-Event-ID`, which re-authenticates. A client disconnect stops the stream. |
| 11 | **Quota bypass by racing** the check against the start. | Documented: quotas are checked before a run starts and recorded when it completes, so concurrent starts can overshoot by at most the concurrency cap. Per-run budgets are the hard limit. |
| 12 | **Information at rest in the process**: unauthenticated endpoints. | Only `/healthz`, which returns `{"ok": true}` and no version or configuration. |
| 13 | **Binding to the world by accident.** | `aaa-server serve` listens on `127.0.0.1` by default; exposing it is a flag you type. TLS terminates at a reverse proxy (documented), not in the library. |

## MCP and A2A endpoints (added in M7)

| # | Threat | Decision |
|---|---|---|
| 14 | **An MCP/A2A client choosing its own tenant or user** (tool arguments, task metadata). | Both endpoints take the tenant and user from the authenticated principal; tool arguments are validated with `extra=forbid`, so `{"tenant": ...}` is an error. Tests call both endpoints as two tenants. |
| 15 | **Reaching an approval-gated tool through MCP**, which has no approval step. | Such tools are not listed and a call is answered like an unknown tool. Exposing one needs an explicit `approval=` callback (fails closed) or `expose_approval_tools=True`. Nothing is auto-approved. |
| 16 | **Tool failures leaking internals** (connection strings, stack traces) to a remote caller. | Only a tool's own `ToolError` text is returned; other exceptions are logged server-side and reported as "`<tool>` failed". |
| 17 | **DNS rebinding against `/mcp`.** | A request carrying an `Origin` header is refused unless the origin is listed; bearer auth is required regardless. |
| 18 | **`run_agent` / A2A as a cost amplifier.** | Both go through the same admission path as `POST /v1/runs`: rate limit, open-run cap, quota, and a server-side budget the caller cannot raise. |
| 19 | **Probing task ids across tenants through A2A.** | A foreign task and a missing task give the same `TaskNotFound` error. |
| 20 | **Trusting a remote agent card.** | Out of scope here: this server only *publishes* a card (public, containing no secrets). Consuming other agents' cards is not implemented. |

## Out of scope

Authenticating end users (OIDC login flows, sessions in browsers), TLS termination, DDoS protection, WAF rules, and what the algorithms and tools themselves do once a run starts
(see the tool, guardrail and multimodal threat models).
