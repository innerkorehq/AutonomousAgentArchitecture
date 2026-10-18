# Generating an MCP server

Instead of writing a server in Python, describe the one you want. `aaa.mcp` compiles a **configuration** (a file, flags, environment variables or the Python API) into a running MCP server: it chooses which tools to expose, how they are named and described, which hints clients get, and whether results are structured. The same description serves over stdio (`aaa mcp serve`) and over HTTP behind [`aaa-server`](../server/index.md).

This page covers: AAA objects as the source, tool selection and metadata, structured output, algorithms exposed as tools, Python modules, OpenAPI documents and other MCP servers as tools, resources and prompts, the Python builder and the command line. The [threat model](../security/mcp-generator.md) covers every source.

```python
--8<-- "examples/mcp_generated.py"
```

```text
offers: desk_forecast | Weather forecast | {'readOnlyHint': True, 'openWorldHint': True}
hidden (need approval): ['desk_erase_history']
structured result: {"city": "Oslo", "celsius": 21.5}
```

## The configuration

TOML and JSON need nothing extra; YAML needs `pip install "autonomous-agent-architecture[mcp-config]"` (read with `safe_load` only). Unknown keys are errors that name the field (`tools.exclud: Extra inputs are not permitted`).

```toml
version = 1

[server]
name = "weather-desk"            # also: title, version, instructions (shown to the client)

[limits]
max_result_chars = 100_000       # text results are cut here, with a marker
max_tools = 200                  # a source that offers more is an error, not silently truncated
structured_output = true         # send structuredContent for tools whose return type is an object

[tenancy]                        # who calls are for on a stdio server (aaa-server uses the authenticated principal instead)
tenant = "mcp"
user = "dev"

[[sources]]
type = "objects"                 # package.module:attribute, see below
ref = "myapp.tools:registry"
tags = ["core"]                  # labels `tools.tags` can select on
prefix = ""                      # added to every tool name from this source

[tools]
include = ["*"]                  # globs on the name the source gave
exclude = ["admin_*"]
tags = []                        # when set, only tools carrying one of these tags
rename = { search_docs = "search" }
prefix = "desk_"                 # added after renaming
expose_approval_tools = false

[tools.overrides.desk_search]    # keyed by the FINAL exposed name
title = "Search the docs"
description = "Full-text search over the documentation."
annotations = { read_only = true, open_world = false }
requires_approval = false        # explicitly lift an approval requirement
```

**Sources.** `type = "objects"` imports `package.module:attribute`; the attribute may be a `ToolRegistry`, an `MCPToolServer`, a `Tool`, a list of tools, or a function (sync or async) returning one of those. From Python you can also pass objects directly with `type = "live"` and `compile_spec(spec, live={"key": registry})`.

**Selection and naming.** `include`/`exclude`/`tags` match the name the source gave (after the source's own `prefix`); `rename` keys are those names; the global `prefix` comes after renaming; `overrides` use the final name. Two tools ending up with the same name is an error that names both origins. A typo in `rename` or `overrides` is an error, not a silent no-op.

**Approval.** A tool that `requires_approval` is **hidden** (and answers exactly like an unknown tool) unless you set `expose_approval_tools`, give it an override `requires_approval = false`, or pass an `approval=` callback to `compile_spec`. Nothing is ever approved for you. `aaa mcp check` and `inspect` list what is hidden.

**Hints.** Without overrides, clients get what the tool already says: an approval-gated tool is `destructiveHint`, a tool with `delivery="at_least_once"` is `idempotentHint`; nothing else is claimed. `annotations` in an override *replace* those. Hints are advice to clients; the server enforces none of them.

**Structured output.** A tool whose return type is a pydantic model, `TypedDict` or dataclass advertises an `outputSchema`, and a successful result carries `structuredContent` (validated against the schema; a result that does not conform is reported as an error, not sent) plus the same JSON as text. A structure that would be cut short by `max_result_chars` is not sent at all, only the truncated text. Other return types (strings, lists, numbers) have no output schema. `SchemaTool` accepts an explicit `output_schema=`, and an override can set one.

## Other MCP servers as a source

Fold the tools of existing MCP servers into the one you serve, so a client connects once and sees everything.

```toml
[[sources]]
type = "mcp"
name = "notes"                 # a label for reports and errors
transport = "stdio"
command = "uvx"
args = ["mcp-server-notes"]    # an argument vector: there is no shell
env = { NOTES_REGION = "eu" }  # literal, non-secret values
env_from = ["NOTES_TOKEN"]     # variables passed through BY NAME; the child inherits nothing else
prefix = "notes_"
include = ["search_*", "get_*"]
approval = "destructive"       # destructive | always | none
trust_annotations = false      # when true, the upstream's read-only hints decide what needs approval

[[sources]]
type = "mcp"
name = "crm"
transport = "http"
url = "https://mcp.crm.example.com/mcp"
allowed_hosts = ["mcp.crm.example.com"]     # mandatory
headers = { Authorization = "Bearer ${CRM_TOKEN}" }   # an environment reference, never the value
optional = true                # if it is down at startup: say so and carry on without its tools
```

- **The upstream is untrusted input.** By default none of its tools are offered: without `trust_annotations` every tool is treated as potentially destructive and requires approval (so it stays hidden). Set `trust_annotations = true` to offer the tools the upstream marks read-only, or `approval = "none"` to offer everything (you are then trusting the upstream completely). Descriptions are capped and can be replaced (`description_overrides`), results are capped (`max_result_chars`), every call has a timeout (`call_timeout_s`), and `aaa mcp inspect` shows what the model will see.
- **A subprocess gets a minimal environment**: only `env` and the variables named in `env_from` (plus the small safe set the MCP SDK always adds). A secret in this process's environment is not inherited by accident. `AAA_MCP_ALLOW_SUBPROCESS=0` / `--no-subprocess-sources` turns every stdio upstream into a startup error before anything is spawned, and a configuration that launches a process is refused when it is writable by everyone.
- **An HTTP upstream goes through the same protection as an OpenAPI call**: https only (plain http solely to loopback with `allow_insecure_loopback`), the host must be in `allowed_hosts`, the host is resolved once, every address must be public (`allow_private_networks` to opt out, reported), the connection is pinned to the checked address, and redirects are never followed. This holds on both SDK generations (the 1.x SDK uses `httpx`, the 2.x SDK its own HTTP stack).
- **Lifecycle.** Connections are opened at startup (a failure is an error, or a note when `optional`) and the tool list is read once: restart to pick up upstream changes. A dead upstream gives a short generic tool error while the rest of the server keeps serving. `await core.aclose()` (or `async with core:`) disconnects and stops launched processes; `aaa mcp serve` does it for you, and a failed compile closes whatever it had already opened.
- **One connection serves every caller, so an upstream is not tenant-isolated**, and tenant and user are never forwarded to it. Use an upstream only if it is safe to share.

### Its resources and prompts

Tools are the default. Add `resources = true` and/or `prompts = true` to the source to re-expose those too:

```toml
[[sources]]
type = "mcp"
name = "notes"
transport = "stdio"
command = "uvx"
args = ["mcp-server-notes"]
resources = true                  # text resources, listed once at startup
resource_exclude = ["*secret*"]   # globs on a resource's name or URI
max_resource_bytes = 200000
prompts = true                    # prefix and include/exclude apply to prompt names too
prefix = "notes_"
```

- A resource appears as `aaa://upstream/notes/<token>`. A read is accepted **only for a URI the upstream listed**: a caller cannot ask for any other URI, and the upstream's resource *templates* are not passed on. Only text is passed (binary resources are left out), cut to `max_resource_bytes`; any failure is the uniform `resource not found`.
- A prompt keeps only its text messages (user and assistant), at most 20 of at most 20 000 characters each; only the arguments the upstream declared are accepted (values cut at 2 000 characters); names and descriptions are capped (`description_overrides` can replace a description). If the upstream cannot fill it, the caller gets a generic message. Two sources offering the same prompt name, or a prompt that clashes with one in your configuration, is a startup error.
- Both are the upstream's words and reach your callers' models: untrusted content, like tool results. The lists are read once at startup. Over HTTP, `aaa-server` serves them through `resources/*` and `prompts/*`.

## OpenAPI documents as tools

```toml
[[sources]]
type = "openapi"
document = "openapi.json"            # a file path, or an https:// URL
allowed_hosts = ["api.example.com"]  # mandatory: the only host(s) this server may call; the API's host must be in it
# base_url = "https://api.example.com/v1"   # overrides servers[0] (needed when the server URL has variables)
no_approval = ["getOrder", "list*"]  # operationId globs that run without approval
operations = ["*"]                   # which operationIds to turn into tools ...
exclude_operations = ["admin*"]      # ... and which never to
max_response_bytes = 1000000
timeout_s = 20
use_response_schemas = false         # advertise the 2xx JSON schema as outputSchema and validate results against it

[sources.auth]                       # optional; the secret is a reference, never the value
scheme = "bearer"                    # bearer | api_key | basic
secret = "env:ORDERS_API_TOKEN"
```

```python
--8<-- "examples/mcp_openapi.py"
```

```text
offered: getOrder | Fetch one order | read-only: True
hidden (state-changing, need approval): ['cancelOrder', 'createOrder']
note: 1 operation(s) skipped as unsupported: uploadReceipt (request body of type ['multipart/form-data'] (only JSON bodies are supported))
```

- **One tool per operation.** The name is the `operationId` (or `method_path`); path, query and header parameters and a JSON request body are merged into one flat input schema (a small object body is flattened; a name that clashes becomes `path_id`, `query_id`, `header_id`). Responses: JSON is returned parsed, text as text, a non-2xx status as a tool error with the status and the first 300 characters of the body.
- **State changes need approval.** GET and HEAD are offered (hinted read-only and idempotent). POST, PUT, PATCH and DELETE require approval by default, and so stay hidden, until you list them in `no_approval`; every such operation exposed without approval is named in a WARNING in `check`/`inspect`.
- **Only the API's host is ever called.** Requests go through the SSRF-safe fetcher: public addresses only (an API on a private address needs `allow_private_networks = true`, which is reported), the connection pinned to the checked address, plain `http` only if the base URL is `http` (reported), and the port taken from the base URL. A redirect to another host is refused, only a GET or HEAD is redirected at all (a POST body is never re-sent), and a caller's argument cannot reach another host: path values are percent-encoded, so `../admin` stays one segment.
- **Credentials.** The secret is read from the environment at startup (missing = error, before serving) and again for each call, sent only to the API's own host, never after a redirect, and never present in a schema, report, error or `inspect` output. `Authorization`, `Cookie`, `Host` and framing headers cannot be set from tool arguments (an operation declaring such a header parameter is skipped).
- **Not supported, so skipped with a note instead of guessed:** cookie parameters, object-valued or non-default-style parameters (`deepObject`, `matrix`, non-exploded arrays), form/multipart/XML bodies, remote `$ref`s, Swagger 2.0. Local `$ref`s are resolved (cycles are cut at a fixed depth), `allOf` is merged one level, and OpenAPI 3.0's `nullable` and boolean `exclusiveMinimum` are converted to JSON Schema.
- **A spec you do not control puts its words in your model's context.** Operation summaries and descriptions become tool descriptions; they are length-capped (`description_max_chars`) and can be replaced (`description_overrides`), and `aaa mcp inspect` prints exactly what the model will see. Review it before connecting a third party's document.

## Python modules as tools

```toml
[[sources]]
type = "module"
module = "myapp.text_tools"
path = "/srv/myapp"            # optional: a directory added to sys.path (it must exist and not be world-writable)
allow = ["word_count", "reverse"]   # optional: otherwise the module's __all__ is used; with neither, it is an error
no_approval = ["word_count"]   # run without approval; everything else in the module still requires it
read_only = ["reverse"]        # also tells clients it only reads (and needs no approval)
strict = true                  # every parameter needs a type hint
```

```python
--8<-- "examples/mcp_modules.py"
```

```text
tool: word_count | Count the words in a text. | ['text']
tool: reverse | Reverse a text. | ['text']
description of 'text': the text to count
hidden (need approval): ['delete_file']
word_count -> 3
```

- **Exposure is an explicit list.** `allow`, or the module's own `__all__`: there is no "everything public" default. Only functions defined in that module are exposed (a re-export in `__all__` is skipped with a note; list it in `allow` to expose it). `_private` names, classes, lambdas, generators and `*args/**kwargs` are refused with a reason.
- **The type hints are the argument schema**, so callers are validated against them (unknown arguments are rejected). `strict = true` (the default) refuses unannotated parameters; `strict = false` types them as `Any` and says so in `check`. A pydantic-model, `TypedDict` or dataclass return type becomes the output schema; a parameter named `ctx: ToolContext` receives the caller's tenant and user.
- **Descriptions come from the docstring**: the first paragraph, and per-parameter text from a Google-style `Args:` section (other docstring styles keep the summary only).
- **Approval is the default.** Module functions are hidden until you say which are safe (`no_approval`, `read_only`) or turn the default off (`default_approval = false`) for a module you fully trust. Naming heuristics (`delete_*`) are deliberately not used: the allow-list and the approval default are the controls.
- **Importing runs the module's code.** Introspection reads signatures and docstrings and never calls a function, but the import itself executes top-level code. That is why `AAA_MCP_ALLOW_IMPORT=0` / `--no-import-sources` stops every module source before anything is imported, and why a configuration naming modules is refused when it is writable by everyone.

## Algorithms as tools

Each configured agent becomes its own tool, with its own name, input schema, budget and hints, instead of the generic `run_agent`.

```toml
[runtime]
engine = "myapp.agents:engine"        # an Engine (or a function returning one) wired with your model and handlers

[limits]
max_concurrent_agents = 4             # agent runs in flight across all agent tools

[[agents]]
name = "ask_support"
algorithm = "myapp.agents:support_agent"     # module:attribute, or a live key from Python
description = "Ask the support agent a customer question."
output = "text"                       # the answer's text; "json" returns the whole outcome
input_template = { task = "Customer {customer}: {question}" }
annotations = { open_world = true }

[agents.budget]
deadline_s = 30                       # mandatory
max_steps = 10
max_cost = 0.25

[agents.input_schema]
type = "object"
required = ["question"]
additionalProperties = false
properties = { question = { type = "string" }, customer = { type = "string" } }
```

- **A deadline is mandatory.** MCP has no way to approve a tool call, so a run that parks on a human approval would otherwise wait forever; the engine ends it at the deadline. Steps, tokens and cost are optional extra limits.
- **The template can only insert validated arguments.** Placeholders are plain names (`{question}`): no attributes, indexes, conversions or format specs, and every placeholder must be a property of the input schema. A value that is exactly one placeholder keeps the argument's own type (a list stays a list).
- **The tenant and user come from the host**, never from the arguments (there is no `tenant` field to fill in): the stdio configuration, or the authenticated principal behind `aaa-server`.
- **Concurrency is capped.** Beyond `max_concurrent_agents` runs in flight, a call waits a few seconds and is then refused with a clear message instead of queuing model spend.
- A failed agent (budget exceeded, cancelled, failed) is an MCP tool error carrying the outcome kind and reason.
- With `aaa mcp serve`, the engine named in `runtime.engine` is started and stopped with the server. Behind `aaa-server`, pass the server's engine to `acompile_spec(spec, engine=...)` and `runtime` is ignored.

## Resources

Read-only context a client can pull in. Four providers:

| Type | Serves | URI |
|---|---|---|
| `static` | inline text | `aaa://static/<name>` |
| `files` | a directory | `aaa://files/<name>/<path>` |
| `index` | search over an `Index` | template `aaa://index/<name>/search?q={query}` (JSON hits) |
| `memory` | search over a memory backend | template `aaa://memory/<name>/search?q={query}` (JSON items) |

```toml
[[resources]]
type = "files"
name = "handbook"
root = "/srv/handbooks"        # one subdirectory per tenant: a caller sees only root/<tenant>/...
pattern = "**/*.md"            # `*` crosses directories; a leading **/ also matches top-level files
# shared = true                # serve root itself to every tenant; must be said explicitly

[[resources]]
type = "index"
name = "faq"
ref = "myapp.search:index"     # module:attribute (or a live key from Python)
tenant_key = "tenant"          # documents carry meta["tenant"]; or `shared = true` for a public index

[limits]
max_resource_bytes = 262144    # one read is cut here, with a marker
max_resources_listed = 500
```

```python
--8<-- "examples/mcp_resources.py"
```

```text
acme sees: ['aaa://files/handbook/handbook.md']
globex sees: ['aaa://files/handbook/handbook.md']
acme reads: Acme handbook
acme reading globex's file: resource not found
globex searches the faq: ['r2']
prompt: Summarize the handbook for a new hire in 50 words.
```

- **Tenant scoping is the default and must be opted out of.** File roots are per tenant; an index needs `tenant_key` (the document metadata field holding the tenant) or an explicit `shared = true`; memory is scoped by the caller's own tenant and user. The tenant is the host's (stdio configuration, or the authenticated principal behind `aaa-server`), never part of a URI.
- **One error for everything unreadable.** An unknown name, a malformed URI, a path outside the root (`..`, encoded or not, symlinks, absolute paths), a hidden or key-like file (`.env`, `id_rsa`), another tenant's document, and a provider that failed all answer the same `resource not found`: a client cannot tell forbidden from missing, and a provider's internal error text never leaves.
- Text only (a binary file is refused), size-capped with an explicit truncation marker, listings capped and scanned within a fixed budget, index/memory queries limited to 500 characters and one `q` parameter.
- Listing returns concrete resources (static and files); index and memory are *templates*, because their content is a query.

## Prompts

```toml
[[prompts]]
name = "triage"
title = "Triage a ticket"
description = "Classify a support ticket."

[[prompts.arguments]]
name = "ticket"
description = "The ticket text"

[[prompts.arguments]]
name = "priority"
type = "enum"                  # string (default), integer, boolean or enum
choices = ["low", "high"]
required = false

[[prompts.messages]]
role = "user"
text = "Triage this ticket (priority {priority}): {ticket}. Use {{braces}} literally."
```

Placeholders are plain argument names, checked when the configuration loads (every one must be a declared argument). Arguments are validated when the prompt is fetched: unknown or missing ones, a bad integer, an enum value outside `choices` are errors that say which. What the client sends is only ever *inserted* into the text, never interpreted: a ticket containing `{priority}` or `{ticket.__class__}` arrives literally.

## The Python builder

The same configuration as typed, chainable Python. Every method edits an `MCPSpec`, so what you build can be written to a file.

```python
--8<-- "examples/mcp_builder.py"
```

```text
tool: ask_support | None | {'openWorldHint': True}
tool: lookup_order | Look up an order | {'readOnlyHint': True}
answer: Order A1 shipped yesterday.
```

The builder also has `static_resource`, `files`, `index`, `memory` and `prompt`. `builder.spec()` returns the validated `MCPSpec`; `builder.dump("server.toml")` writes it (JSON, TOML or YAML by extension, defaults left out) unless it holds live Python objects (`add_tools(registry)`, an algorithm object), which a file cannot carry; `await builder.acompile()` / `builder.compile()` build the server. `dump_spec`/`load_spec` round-trip: a spec written and read back is equal.

## Generating a project

`aaa mcp init` writes a project around a configuration, so a generated server is something you can version, test and deploy. It is **packaging, not logic**: `server.py` is a fixed file that loads the configuration and compiles it, so what the project does is decided by the configuration alone.

```bash
aaa mcp init my-server                                # interactive: asks for the sources, transport, Dockerfile, format
aaa mcp init my-server --yes --source openapi=api.json --allowed-host api.example.com --bearer-env ORDERS_API_TOKEN --transport http
aaa mcp init my-server --yes --source module=myapp.tools:word_count,reverse --source objects=myapp.tools:registry
aaa mcp export my-server --config server.toml         # a project around a configuration you already have (copied as written)
aaa mcp export my-server --check                      # have generated files drifted from their templates? (exit 1 if so)
aaa mcp export my-server --update                     # regenerate the generated files you have not edited
```

| File | What it is |
|---|---|
| `server.toml` (or `.json`/`.yaml`) | The configuration. **Yours**: edit it; `--update` never touches it. |
| `server.py` | Stdio: loads the configuration and serves. HTTP: `async def build() -> ServerConfig` for `aaa-server serve server:build`, with API-key authentication from `MCP_API_KEYS` (the key decides tenant and user). |
| `requirements.txt` | The library (and `aaa-server` for HTTP) pinned to the compatible range. |
| `Dockerfile`, `.dockerignore` | A slim non-root image; stdio runs with `docker run -i`, HTTP listens on 8000 behind your reverse proxy. `--no-docker` skips them. |
| `.env.example` | The **names** of the environment variables the configuration refers to (and `MCP_API_KEYS`), never values. |
| `tests/test_server.py` | A smoke test: the configuration is valid and compiles. |
| `README.md`, `.gitignore`, `.aaa-scaffold.json` | How to run it; ignores `.env`; the options and a hash of each generated file (what `--check`/`--update` compare). |

```python
--8<-- "examples/mcp_init.py"
```

```text
project files: .dockerignore, .env.example, .gitignore, Dockerfile, README.md, requirements.txt, server.py, server.toml, tests/test_server.py
secret names (never values): ['ORDERS_API_TOKEN']
drift right after generating: none
after editing the Dockerfile: ['Dockerfile: differs from its template (and was edited by hand)']
recorded: aaa with 8 hashed files
```

- **No user text reaches Python source.** The generated `.py` files are identical whatever the configuration says; the configuration travels only in the data file, written from the validated model. Project names are restricted to `[a-z][a-z0-9_-]*`, and a generated path can never leave the project directory.
- **No secret value is written anywhere**, only `env:NAME` references and the names in `.env.example`. A test plants a canary secret and searches the whole generated tree for it.
- **Relative paths in a configuration are relative to the configuration file** (the OpenAPI `document`, a module `path`, a files-resource `root`), so a project works from any directory. `--vendor-spec` copies OpenAPI documents into `openapi/` (an https document is fetched through the same safe fetcher) for reproducible builds.
- `init` refuses to write into a non-empty directory without `--force`. With HTTP, put a TLS-terminating reverse proxy in front (see [deployment](../operations/deployment.md)); if the configuration has `agents`, wire your model and handlers onto the engine first.

## Limits and visibility per tool

`[[tools.policies]]` sets limits for every tool whose final name matches a glob; a tool's own `[tools.overrides.NAME]` takes the same fields and beats the policies. For each setting the first matching policy wins.

```toml
[[tools.policies]]
match = ["orders_*"]
timeout_s = 10
rate_limit = { per_minute = 30, burst = 5 }   # per caller and tool
max_concurrency = 2                           # the next call is refused, not queued

[[tools.policies]]
match = ["orders_refund"]
scopes = ["admin"]                            # visible and callable only to callers holding one of these
tenants = ["acme"]
```

* A caller who may not see a tool gets exactly the answer for a tool that does not exist, in `tools/list` and in `tools/call`.
* Behind `aaa-server`, the caller's scopes are those of the API key. Over stdio the one caller holds `tenancy.scopes` (default: unrestricted).
* Rate-limit and concurrency refusals are tool errors with a fixed message (`... retry shortly`) that does not say how the limits are set. Rate limits are in memory, per process; use the server's admission control for a cluster-wide limit.
* A `match` that selects no tool is an error, so a typo cannot silently leave a tool unlimited. `aaa mcp inspect` lists the rules.

## Split files and per-environment overlays

A large configuration can be split with `include`, and a base configuration can be adapted per environment with overlays.

```toml
# server.toml
include = ["tools/orders.toml", "tools/search.toml"]
[server]
name = "everything"
```

* Each included path is relative to the file that names it and must stay inside the top file's directory (symlinks are resolved first; absolute paths and `..` escapes are refused). Nesting is limited to 5 levels, a cycle is an error naming the files, and a file is limited to 1 MB.
* Paths inside a file (`document`, module `path`, files `root`) are relative to that file, wherever it is included from.
* Merging: `sources`, `agents`, `resources`, `prompts` and `tools.policies` are **concatenated** in order (included files first, then the including file); dictionaries merge key by key; scalars and every other list are replaced by the later file.
* Overlays are extra files layered on top: `--overlay prod.toml` (repeatable) or `AAA_MCP_OVERLAY` (paths separated by `:`). They are chosen by the operator, so they may live anywhere, and use the same merge rules.
* `aaa mcp flatten --config server.toml --overlay prod.toml -o flat.toml` writes the composed result as one file (paths made relative to the output) for review or shipping. Without `-o` it prints JSON.
* If the composed configuration imports code or launches processes, **every** contributing file must not be world-writable.

## Connecting a client

`aaa mcp client-config` prints the entry an MCP client needs for this one server:

```bash
aaa mcp client-config --config server.toml --client claude-desktop
aaa mcp client-config --config server.toml --client cursor --transport http --url https://mcp.example.com/mcp
```

* `--client` is `claude-desktop`, `cursor` or `generic`; `--transport` is `stdio` (the client launches `aaa mcp serve --config ...`, with your `--overlay`s) or `http` (a server behind aaa-server, `--url` required, https unless localhost).
* The configuration is validated first; the output is `{"mcpServers": {name: entry}}`, ready to merge into the client's own file.
* The entry lists the **names** of the environment variables the server reads (API credentials, upstream headers, pass-through variables) as `${NAME}` placeholders (`${env:NAME}` for Cursor). A value is never printed. HTTP entries send `Authorization: Bearer ${MCP_API_KEY}`.
* Claude Desktop starts local commands, so its HTTP entry goes through the `mcp-remote` bridge via `npx`.

One configuration for everything: an included file, a module, an OpenAPI document, another MCP server, a resource, a prompt, a policy and an overlay.

```python
--8<-- "examples/mcp_everything.py"
```

```text
title: Everything (production)
tools: ['getOrder', 'up_shout', 'word_count']
resources: ['readme']
prompts: ['summarize']
word_count: 4
up_shout: HELLO
policies: ['up_shout', 'word_count']
```

## Precedence

Lowest to highest: the file with its includes, then overlay files, then `AAA_MCP__*` environment variables, then command-line flags. Levels are separated by double underscores (`AAA_MCP__SERVER__NAME=desk`, `AAA_MCP__TOOLS__EXCLUDE='["a","b"]'`: lists, numbers and booleans are JSON). Dictionaries merge key by key; lists and scalars are **replaced**, so `--exclude x` means exactly `x`.

Secrets never go in the file: a field that needs one takes an `env:NAME` or `${NAME}` reference (`Secret.parse`), and the value is read when needed and never shown in a `repr`.

## The command line

```bash
aaa mcp serve --config server.toml                        # serve over stdio
aaa mcp serve --source myapp.tools:registry --exclude 'admin_*' --tenant acme
aaa mcp check --config server.toml                        # validate, compile, report what would be exposed
aaa mcp inspect --config server.toml                      # print exactly what a client (and its model) will see
aaa mcp serve myapp.tools:build                           # the older form: a factory returning a registry or MCPToolServer
```

Flags: `--source`, `--include`, `--exclude`, `--tag`, `--prefix`, `--name`, `--title`, `--instructions`, `--tenant`, `--user`, `--expose-approval-tools`, `--max-result-chars`, `--no-import-sources`. `serve` writes its messages to stderr, because stdout carries the protocol.

## Over HTTP, behind aaa-server

```python
from aaa.mcp.config import load_spec
from aaa.mcp.compile import acompile_spec

async def build() -> ServerConfig:
    core = await acompile_spec(load_spec("server.toml"), engine=engine)
    return ServerConfig(engine=engine, authenticator=..., mcp=core)
```

`aaa-server` serves the compiled core at `POST /mcp` with the same authentication, rate limits and body limit as everything else; tenant and user come from the principal, not from the file. Titles, annotations, output schemas, structured results, instructions and the server version appear on the wire exactly as over stdio: both are built from one set of wire shapes (`aaa.mcp.wire`).

## Safety

- **Importing runs code.** A source of type `objects` imports your module. Set `AAA_MCP_ALLOW_IMPORT=0` (or `--no-import-sources`) to make any such source an error before anything is imported, and a configuration that imports code is refused if it is writable by everyone.
- Generated servers keep the defaults of [serving tools over MCP](mcp.md#serving-your-tools-over-mcp): failures never leak internals, arguments are validated by the tool's schema, results are capped.
- The full list of threats, including those of the sources that are not built yet, is the [threat model](../security/mcp-generator.md).

## Verified, and not

Tested: configuration loading and precedence (JSON, TOML, YAML, environment, flags), selection/naming/override rules and their error messages, the kill-switch, structured output including the non-conforming and oversized cases, and that the official SDK server, the stdio subprocess and the HTTP handler all describe tools identically. The suite passes on the `mcp` SDK 1.21.2, 1.30 and 2.x. **Not verified:** real MCP hosts (Claude Desktop, IDEs); large public OpenAPI documents (hundreds of operations); OpenAPI 3.1-only constructs beyond the JSON Schema they share with 3.0; APIs that require OAuth. **The `mcp` extra now requires 1.21.2 or newer**: older 1.x releases no longer import with current pydantic.
