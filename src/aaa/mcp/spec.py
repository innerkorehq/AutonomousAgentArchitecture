"""`MCPSpec`: the one description of a generated MCP server.

A YAML/JSON/TOML file, the Python builder, command-line flags and environment variables all produce an `MCPSpec`; the compiler
(`aaa.mcp.compile`) is the only thing that reads one. Unknown keys are errors, so a typo cannot silently do nothing."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TOOL_NAME = r"^[A-Za-z0-9_-]{1,64}$"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnnotationsSpec(_M):
    """MCP tool hints: for clients to decide how cautious to be; the server enforces none of them."""

    title: str | None = Field(default=None, max_length=200)
    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None


class ServerSpec(_M):
    name: str = Field(default="aaa", pattern=TOOL_NAME)
    title: str | None = Field(default=None, max_length=200)
    version: str = Field(default="0.1", max_length=40)
    instructions: str | None = Field(default=None, max_length=4000)


class TransportSpec(_M):
    """What `aaa mcp serve` does. Behind `aaa-server` the HTTP listener is the server's own, and these are ignored."""

    kind: Literal["stdio"] = "stdio"


class LimitsSpec(_M):
    max_result_chars: int = Field(default=100_000, ge=1_000, le=5_000_000)
    max_tools: int = Field(default=200, ge=1, le=2_000)
    max_concurrent_agents: int = Field(
        default=4, ge=1, le=64
    )  # agent runs in flight across all agent tools
    max_resource_bytes: int = Field(
        default=262_144, ge=1_000, le=10_000_000
    )  # one resource read is cut here
    max_resources_listed: int = Field(default=500, ge=1, le=10_000)
    structured_output: bool = (
        True  # send structuredContent for tools whose return type is an object type
    )


class TenancySpec(_M):
    """Who the calls are for on a stdio server. Behind `aaa-server` the authenticated principal replaces both."""

    tenant: str = Field(default="mcp", min_length=1, max_length=100, pattern=r"^[^/\s]+$")
    user: str | None = Field(default=None, max_length=200)
    scopes: list[str] | None = (
        None  # scopes the single stdio caller holds; None = all (a tool restricted by `scopes` is then visible)
    )


class RateLimit(_M):
    """A token bucket per caller and tool: `per_minute` calls on average, bursts up to `burst` (default: `per_minute`, at most 10)."""

    per_minute: int = Field(ge=1, le=1_000_000)
    burst: int | None = Field(default=None, ge=1, le=1_000_000)


class ToolLimits(_M):
    """Per-tool limits and visibility. A caller who may not see a tool gets exactly the answer for a tool that does not exist."""

    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    rate_limit: RateLimit | None = None
    max_concurrency: int | None = Field(
        default=None, ge=1, le=256
    )  # calls in flight; the next is refused, not queued
    scopes: list[str] | None = Field(
        default=None, min_length=1
    )  # visible to callers holding at least one of these
    tenants: list[str] | None = Field(default=None, min_length=1)  # visible to these tenants only


class ToolPolicy(ToolLimits):
    """Limits for every tool whose final name matches one of the globs. For each setting the first matching policy wins; a tool's own
    `overrides` entry beats every policy. A glob that matches no tool is an error."""

    match: list[str] = Field(min_length=1)


class ToolOverride(ToolLimits):
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    annotations: AnnotationsSpec | None = None
    requires_approval: bool | None = None
    output_schema: dict[str, Any] | None = None


class ToolsSpec(_M):
    """Which tools are exposed and how they look.

    `include`/`exclude`/`tags` match the name a source gave the tool (after the source's own `prefix`); `rename` keys are those names;
    the global `prefix` is added after renaming; `overrides` are keyed by the name the tool is finally exposed under.
    A tool that requires approval is hidden unless `expose_approval_tools` is true or its override sets `requires_approval: false`."""

    include: list[str] = Field(default_factory=lambda: ["*"])
    exclude: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    rename: dict[str, str] = Field(default_factory=dict)
    prefix: str = Field(default="", max_length=40, pattern=r"^[A-Za-z0-9_-]*$")
    overrides: dict[str, ToolOverride] = Field(default_factory=dict)
    policies: list[ToolPolicy] = Field(default_factory=list)
    expose_approval_tools: bool = False

    @field_validator("rename")
    @classmethod
    def _names(cls, v: dict[str, str]) -> dict[str, str]:
        import re

        for new in v.values():
            if not re.match(TOOL_NAME, new):
                raise ValueError(
                    f"{new!r} is not a valid tool name (letters, digits, '_' and '-', at most 64)"
                )
        return v


class BudgetSpec(_M):
    """What one agent run may spend. `deadline_s` is mandatory: a run that parks on a human approval (which MCP cannot give)
    would otherwise wait forever."""

    deadline_s: float = Field(gt=0, le=3600)
    max_steps: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost: float | None = Field(default=None, gt=0)


DEFAULT_AGENT_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"type": "string", "description": "What to ask the agent."}},
    "required": ["input"],
    "additionalProperties": False,
}


class AgentToolSpec(_M):
    """Expose one algorithm as its own tool. `algorithm` is a `module:attribute` reference (an `Algorithm`) or the key of an object
    passed to `compile_spec(..., live={...})`. `input_template` builds the algorithm's input from the validated arguments:
    string values may contain `{argument}` placeholders (plain names only; no attribute access, no format specs) and a value that is
    exactly one placeholder keeps the argument's own type."""

    name: str = Field(pattern=TOOL_NAME)
    algorithm: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    budget: BudgetSpec
    input_schema: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_AGENT_INPUT))
    input_template: dict[str, Any] = Field(default_factory=lambda: {"task": "{input}"})
    output: Literal["text", "json"] = "text"  # the answer's text, or the whole outcome as JSON
    annotations: AnnotationsSpec | None = None
    tags: list[str] = Field(default_factory=list)


class RuntimeSpec(_M):
    """The engine agent tools run on, for a server that builds its own (`aaa mcp serve`). `engine` is `module:attribute`: an
    `Engine` or a function (sync or async) returning one, already wired with your model and tool handlers. Behind
    `aaa-server` the server's engine is used and this is ignored."""

    engine: str | None = Field(default=None, pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


class StaticResource(_M):
    type: Literal["static"]
    name: str = Field(pattern=TOOL_NAME)
    text: str = Field(max_length=200_000)
    mime_type: str = "text/plain"
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class FilesResource(_M):
    """Files under `root` as resources. Per-tenant by default: a caller sees only `root/<tenant>/...`; `shared = true` serves `root`
    itself to every tenant and must be said explicitly. Hidden files and key-like names are never served."""

    type: Literal["files"]
    name: str = Field(pattern=TOOL_NAME)
    root: str = Field(min_length=1)
    pattern: str = "**/*"
    shared: bool = False
    allow_hidden: bool = False
    mime_type: str | None = None
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class IndexResource(_M):
    """Search over an `Index` (`ref` = `module:attribute`) as `aaa://index/<name>/search?q=...`. Documents are tenant-scoped through
    their `meta[tenant_key]`; an index holding one tenant's or public data says `shared = true` instead."""

    type: Literal["index"]
    name: str = Field(pattern=TOOL_NAME)
    ref: str = Field(min_length=1, max_length=200)  # module:attribute, or the key of a live object
    k: int = Field(default=5, ge=1, le=20)
    max_chars: int = Field(default=800, ge=50, le=20_000)
    tenant_key: str | None = None
    shared: bool = False
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _scoped(self) -> IndexResource:
        if not self.shared and not self.tenant_key:
            raise ValueError(
                "set tenant_key (the document meta field holding the tenant) or shared = true"
            )
        return self


class MemoryResource(_M):
    """Search over a memory backend (`ref` = `module:attribute`), scoped by the caller's tenant and user."""

    type: Literal["memory"]
    name: str = Field(pattern=TOOL_NAME)
    ref: str = Field(min_length=1, max_length=200)  # module:attribute, or the key of a live object
    k: int = Field(default=5, ge=1, le=20)
    max_chars: int = Field(default=800, ge=50, le=20_000)
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


ResourceSpec = Annotated[
    StaticResource | FilesResource | IndexResource | MemoryResource, Field(discriminator="type")
]


class PromptArg(_M):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=64)
    description: str | None = Field(default=None, max_length=500)
    required: bool = True
    type: Literal["string", "integer", "boolean", "enum"] = "string"
    choices: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _choices(self) -> PromptArg:
        if (self.type == "enum") != bool(self.choices):
            raise ValueError("choices are required for, and only for, type = 'enum'")
        return self


class PromptMessage(_M):
    role: Literal["user", "assistant"] = "user"
    text: str = Field(min_length=1, max_length=20_000)


class PromptSpec(_M):
    """A prompt template. `{argument}` placeholders in message texts are filled from the (validated) arguments; write a literal brace
    as `{{` or `}}`."""

    name: str = Field(pattern=TOOL_NAME)
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    arguments: list[PromptArg] = Field(default_factory=list)
    messages: list[PromptMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def _placeholders(self) -> PromptSpec:
        from aaa.mcp.config import SpecError
        from aaa.mcp.templating import template_fields

        names = [a.name for a in self.arguments]
        if len(set(names)) != len(names):
            raise ValueError("duplicate argument names")
        try:
            used = set().union(
                *(template_fields(m.text, where=f"prompts.{self.name}") for m in self.messages)
            )
        except SpecError as e:
            raise ValueError(str(e)) from None
        if missing := sorted(used - set(names)):
            raise ValueError(f"message placeholders {missing} are not declared as arguments")
        return self


class _Source(_M):
    tags: list[str] = Field(default_factory=list)
    prefix: str = Field(default="", max_length=40, pattern=r"^[A-Za-z0-9_-]*$")


class ObjectsSource(_Source):
    """Tools from AAA objects defined in your code: `ref` is `package.module:attribute`, and the attribute is a `ToolRegistry`, an
    `MCPToolServer`, a `Tool`, a list of tools, or a function (sync or async) returning one of those."""

    type: Literal["objects"]
    ref: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


class AuthSpec(_M):
    """Credentials for the upstream API. The secret is NEVER written here: `secret` is an `env:NAME` reference, read from the
    environment when the server starts (a missing variable is a startup error) and again for each call. The credential is sent only
    to the API's own host, never after a redirect, and never exposed to the model."""

    scheme: Literal["bearer", "api_key", "basic"]
    secret: str = Field(min_length=1, max_length=200)
    username: str | None = Field(default=None, max_length=200)  # basic auth only
    location: Literal["header", "query"] = "header"  # api_key only
    name: str = Field(
        default="X-API-Key", pattern=r"^[A-Za-z0-9_-]{1,64}$"
    )  # api_key header or query-parameter name

    @model_validator(mode="after")
    def _check(self) -> AuthSpec:
        from aaa.mcp.config import Secret, SpecError

        try:
            Secret.parse(self.secret, field="auth.secret")
        except SpecError as e:
            raise ValueError(str(e)) from None
        if self.scheme == "basic" and not self.username:
            raise ValueError("basic auth needs a username")
        if self.scheme != "basic" and self.username:
            raise ValueError("username applies to basic auth only")
        return self


class OpenAPISource(_Source):
    """One tool per operation of an OpenAPI 3.0/3.1 document.

    `allowed_hosts` is mandatory: the API's own host must be in it, and every request goes through the SSRF-safe fetcher (public
    addresses only, connection pinned, redirects to other hosts refused). Operations that change state (`methods_requiring_approval`,
    by default POST, PUT, PATCH and DELETE) require approval, and so stay hidden, unless matched by `no_approval`. Features this
    generator does not support (cookie parameters, form bodies, non-default parameter styles ...) skip that operation with a note,
    never silently mis-call it."""

    type: Literal["openapi"]
    document: str = Field(min_length=1, max_length=500)  # a file path or an https:// URL
    base_url: str | None = Field(
        default=None, max_length=500
    )  # overrides the document's servers[0]
    allowed_hosts: list[str] = Field(min_length=1)
    allow_private_networks: bool = False  # explicit opt-in for an API on a private address
    auth: AuthSpec | None = None
    operations: list[str] = Field(default_factory=lambda: ["*"])  # globs on operationId
    exclude_operations: list[str] = Field(default_factory=list)
    methods_requiring_approval: list[Literal["get", "head", "put", "post", "patch", "delete"]] = (
        Field(default_factory=lambda: ["post", "put", "patch", "delete"])
    )
    no_approval: list[str] = Field(
        default_factory=list
    )  # operationId globs that run without approval
    description_overrides: dict[str, str] = Field(default_factory=dict)
    description_max_chars: int = Field(default=1000, ge=50, le=5000)
    max_response_bytes: int = Field(default=1_000_000, ge=1_000, le=20_000_000)
    max_request_bytes: int = Field(default=100_000, ge=100, le=5_000_000)
    timeout_s: float = Field(default=20.0, gt=0, le=300)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    use_response_schemas: bool = (
        False  # advertise the 2xx JSON schema as outputSchema and validate results against it
    )


_HEADER_REF = r"^(?:env:[A-Za-z_][A-Za-z0-9_]*|[A-Za-z0-9 _.-]{0,40}\$\{[A-Za-z_][A-Za-z0-9_]*\})$"


class McpSource(_Source):
    """The tools of ANOTHER MCP server, folded into this one.

    `transport = "stdio"` launches `command` (an argument vector, never a shell) with a minimal environment: only `env` (literal,
    non-secret values) and the variables named in `env_from`. `transport = "http"` connects to an https `url` through a transport that
    resolves the host once, requires a public address and `allowed_hosts`, pins the connection and follows no redirects.

    The upstream is untrusted input: its tools require approval unless it marks them read-only (`approval = "destructive"`), its
    annotations are not trusted unless `trust_annotations`, its descriptions are capped, and its results are capped. One connection
    serves every caller, so it is NOT tenant-isolated; tenant and user are never forwarded to it."""

    type: Literal["mcp"]
    name: str = Field(pattern=TOOL_NAME)  # a label for reports and errors
    transport: Literal["stdio", "http"]
    command: str | None = Field(default=None, max_length=500)
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)  # literal, NON-secret values
    env_from: list[str] = Field(
        default_factory=list
    )  # variables passed through from this process by name
    url: str | None = Field(default=None, max_length=500)
    headers: dict[str, str] = Field(default_factory=dict)  # values: "env:NAME" or "Bearer ${NAME}"
    allowed_hosts: list[str] = Field(default_factory=list)
    allow_private_networks: bool = False
    allow_insecure_loopback: bool = False  # plain http, to 127.0.0.1/localhost only
    include: list[str] = Field(default_factory=lambda: ["*"])
    exclude: list[str] = Field(default_factory=list)
    approval: Literal["destructive", "always", "none"] = "destructive"
    trust_annotations: bool = False
    description_overrides: dict[str, str] = Field(default_factory=dict)
    description_max_chars: int = Field(default=1000, ge=50, le=5000)
    connect_timeout_s: float = Field(default=20.0, gt=0, le=300)
    call_timeout_s: float = Field(default=60.0, gt=0, le=3600)
    max_result_chars: int = Field(default=100_000, ge=1_000, le=5_000_000)
    optional: bool = (
        False  # if the upstream is down at startup: report it and carry on without its tools
    )
    resources: bool = (
        False  # also re-expose the upstream's (text) resources, under aaa://upstream/<name>/...
    )
    prompts: bool = False  # also re-expose the upstream's prompts (text messages only)
    resource_exclude: list[str] = Field(default_factory=list)  # globs on a resource's name or URI
    max_resource_bytes: int = Field(default=200_000, ge=1_000, le=2_000_000)
    max_listed: int = Field(default=100, ge=1, le=500)  # resources taken from the upstream

    @model_validator(mode="after")
    def _check(self) -> McpSource:
        import re

        if self.transport == "stdio":
            if not self.command:
                raise ValueError("a stdio upstream needs a command")
            if self.url or self.headers or self.allowed_hosts:
                raise ValueError("url, headers and allowed_hosts apply to http upstreams only")
        else:
            if not self.url:
                raise ValueError("an http upstream needs a url")
            if self.command or self.args or self.env or self.env_from or self.cwd:
                raise ValueError(
                    "command, args, cwd, env and env_from apply to stdio upstreams only"
                )
            if not self.allowed_hosts:
                raise ValueError(
                    "an http upstream needs allowed_hosts: the only host(s) this server may call"
                )
        for k in [*self.env, *self.env_from]:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
                raise ValueError(f"{k!r} is not a valid environment variable name")
        for k, v in self.headers.items():
            if not re.match(r"^[A-Za-z0-9-]{1,64}$", k) or k.lower() in (
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
            ):
                raise ValueError(f"header {k!r} cannot be set")
            if not re.match(_HEADER_REF, v):
                raise ValueError(
                    f"header {k!r}: give an environment reference ('env:NAME' or 'Bearer ${{NAME}}'), never the value itself"
                )
        if any(len(v) > 500 for v in self.env.values()):
            raise ValueError("env values are limited to 500 characters; secrets belong in env_from")
        return self


class ModuleSource(_Source):
    """Public functions of a Python module as tools: schema from type hints, description from the docstring.

    Importing a module runs its top-level code, so this is as trusted as `objects`, and the list of what is exposed is explicit:
    `allow` names functions, or the module's `__all__` is used; a module with neither is an error. Only functions DEFINED in the
    module are exposed (re-exports need `allow`), never `_private` names, classes, generators, lambdas or `*args/**kwargs`.
    Nothing is ever called while the module is inspected.

    Module tools REQUIRE APPROVAL by default (so they stay hidden): list the harmless ones in `no_approval`, or in `read_only`
    (which also tells clients they only read)."""

    type: Literal["module"]
    module: str = Field(pattern=r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)*$", max_length=200)
    allow: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    path: str | None = (
        None  # a directory to add to sys.path so the module can be found (explicit, never guessed)
    )
    strict: bool = True  # every parameter needs a type hint; false types the unannotated ones as Any, with a warning
    default_approval: bool = True
    no_approval: list[str] = Field(default_factory=list)
    read_only: list[str] = Field(default_factory=list)


class LiveSource(_Source):
    """Tools passed to `compile_spec(..., live={key: ...})` from Python. Not loadable from a file (there is nothing to load)."""

    type: Literal["live"]
    key: str = Field(min_length=1, max_length=80)


Source = Annotated[
    ObjectsSource | ModuleSource | OpenAPISource | McpSource | LiveSource,
    Field(discriminator="type"),
]


class MCPSpec(_M):
    version: Literal[1] = 1
    server: ServerSpec = Field(default_factory=ServerSpec)
    transport: TransportSpec = Field(default_factory=TransportSpec)
    limits: LimitsSpec = Field(default_factory=LimitsSpec)
    tenancy: TenancySpec = Field(default_factory=TenancySpec)
    sources: list[Source] = Field(default_factory=list)
    tools: ToolsSpec = Field(default_factory=ToolsSpec)
    agents: list[AgentToolSpec] = Field(default_factory=list)
    resources: list[ResourceSpec] = Field(default_factory=list)
    prompts: list[PromptSpec] = Field(default_factory=list)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)

    def launches_processes(self) -> bool:
        """Whether serving this spec starts a subprocess (a stdio upstream): see `AAA_MCP_ALLOW_SUBPROCESS`."""
        return any(isinstance(s, McpSource) and s.transport == "stdio" for s in self.sources)

    def imports_code(self) -> bool:
        """Whether compiling this spec imports modules (and so runs code): see the kill-switch in `aaa.mcp.compile`."""
        return (
            any(s.type in ("objects", "module") for s in self.sources)
            or self.runtime.engine is not None
            or any(":" in a.algorithm for a in self.agents)
            or any(
                isinstance(r, IndexResource | MemoryResource) and ":" in r.ref
                for r in self.resources
            )
        )


__all__ = [
    "AgentToolSpec",
    "AnnotationsSpec",
    "AuthSpec",
    "BudgetSpec",
    "FilesResource",
    "IndexResource",
    "LimitsSpec",
    "LiveSource",
    "MCPSpec",
    "McpSource",
    "MemoryResource",
    "ModuleSource",
    "ObjectsSource",
    "OpenAPISource",
    "PromptArg",
    "PromptMessage",
    "PromptSpec",
    "RateLimit",
    "ResourceSpec",
    "RuntimeSpec",
    "ServerSpec",
    "Source",
    "StaticResource",
    "TenancySpec",
    "ToolLimits",
    "ToolOverride",
    "ToolPolicy",
    "ToolsSpec",
    "TransportSpec",
]
