# aaa-server

**An HTTP/SSE server for AAA agents**: bearer authentication, tenants taken only from the credential, rate limits and quotas, server-side budgets, human approvals over HTTP, resumable event streams, MCP and (experimental) A2A endpoints.

`aaa-server` is a **separate package** from the [AAA library](../index.md) (Autonomous Agent Architecture): the library has no web dependencies, and the server releases on its own schedule.

[![PyPI](https://img.shields.io/pypi/v/aaa-server)](https://pypi.org/project/aaa-server/) &nbsp; Library: [`autonomous-agent-architecture`](https://pypi.org/project/autonomous-agent-architecture/) · [docs home](../index.md)

## Install

```bash
pip install aaa-server                  # the server (pulls in autonomous-agent-architecture>=0.3.0)
pip install "aaa-server[jwt,redis]"    # + JWT/JWKS authentication, + the shared Redis rate limiter
```

| `aaa-server` | needs `autonomous-agent-architecture` | Python |
|---|---|---|
| 0.1.x | `>=0.3.0` | 3.11, 3.12, 3.13 |

## A server in thirty lines

```python
# myapp/server.py
from aaa.algos import ToolLoop
from aaa.llm import LLMHandler, build_client
from aaa.runtime import Budget, Engine
from aaa.tools import ToolHandler, ToolRegistry
from aaa_server import ApiKeyAuthenticator, InMemoryRateLimiter, ServerConfig


async def build() -> ServerConfig:
    engine = Engine()                                   # use PostgresEventStore + DistributedBus in production
    registry = ToolRegistry([])                         # your tools
    LLMHandler(build_client(...)).attach(engine)        # your model
    ToolHandler(registry).attach(engine)
    return ServerConfig(
        engine=engine,
        authenticator=ApiKeyAuthenticator.from_keys({"s3cret-key": ("acme", "ann")}),   # key -> (tenant, subject)
        algorithms={"agent": ToolLoop(registry, system="Be brief.")},
        budget=Budget(max_steps=20, deadline_s=60),     # a ceiling a request may lower but never raise
        rate_limiter=InMemoryRateLimiter(per_second=10, burst=50),
    )
```

```bash
aaa-server serve myapp.server:build                   # listens on 127.0.0.1:8000
curl -s -X POST localhost:8000/v1/runs -H "Authorization: Bearer s3cret-key" \
     -d '{"algo": "agent", "input": {"task": "hello"}}'
curl -N localhost:8000/v1/runs/<run_id>/events -H "Authorization: Bearer s3cret-key"
```

A complete, runnable, offline version (including an approval over HTTP and a second tenant that cannot see any of it) is the [server guide's example](../guides/server.md).

## What you get

| | |
|---|---|
| **Tenant isolation** | Tenant and user come only from the authenticated principal. A foreign run or session id is indistinguishable from a missing one (`404`); identity fields in a body are rejected. |
| **Authentication** | Hashed API keys (constant-time), or JWT with a JWKS URL: fixed algorithm allow-list, required `exp`/`iss`/`aud`, rate-limited key refresh. |
| **Admission control** | Token-bucket rate limits (in-memory or Redis), event-sourced daily quotas with race-free reservation, open-run and stream caps, a body-size limit. |
| **Runs and streams** | Idempotent `POST /v1/runs`, cancel, and server-sent events that resume with `Last-Event-ID`. The stream is a filtered projection of the log, never the raw log. |
| **Human approval** | `POST /v1/runs/{id}/approvals` with its own `approve` scope; only a call that is currently pending can be approved. |
| **Human answers** | `POST /v1/runs/{id}/answers` with its own `answer` scope, for [`AskHuman`](../guides/algorithms/multi-agent-human.md) runs; only the outstanding question, once, is accepted. |
| **Sessions** | Create, send (blocking or streamed), list, close; attachments are the library's multimodal parts. |
| **MCP and A2A** | `POST /mcp` serves your tools to any MCP client under the same auth; `A2AConfig` serves an algorithm as an A2A 1.0 agent (experimental). |

## Where to read next

| | |
|---|---|
| [**Server guide**](../guides/server.md) | Running it, authentication, every endpoint, limits, MCP and A2A, what is verified. |
| [**Threat model**](../security/server.md) | Why each default is what it is, threat by threat. Read before exposing it. |
| [**API reference**](../reference/api/server.md) | `ServerConfig`, authenticators, rate limiters, quotas. |
| [Deployment checklist](../operations/deployment.md) | The container image, `compose.yaml`, TLS, reverse proxy. |
| [Back to the AAA docs](../index.md) | The library: algorithms, tools, memory, evals, guardrails. |

## What it is not

It is not an identity provider, a TLS terminator or a WAF: put a reverse proxy in front for TLS and header/body timeouts, and an identity provider behind the JWT authenticator. Rate limits are per principal, not per IP. `aaa-server` is **0.1, alpha**: the HTTP surface may change in minor releases, and the A2A endpoint is experimental
(see the [stability policy](../policy/stability.md)). Source and issues: [github.com/innerkorehq/AutonomousAgentArchitecture](https://github.com/innerkorehq/AutonomousAgentArchitecture/tree/master/packages/aaa-server).
