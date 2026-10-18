"""Tools from an OpenAPI 3.0/3.1 document: one tool per operation, called through the SSRF-safe fetcher.

What is supported, so nothing is called in a way you did not expect: path, query and header parameters in their DEFAULT serialization
(simple/form, arrays and primitives), a JSON request body, JSON or text responses, local `$ref`s (cycles cut at a fixed depth),
`allOf` merged shallowly. An operation using anything else (cookie parameters, form/multipart/XML bodies, `deepObject`/`matrix` styles,
callbacks, remote `$ref`s ...) is SKIPPED with a note, never approximated. Credentials come from `env:` references, go to the API's own
host only, and are never visible to the model."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

import httpx

from aaa.mcp.config import Secret, SpecError
from aaa.mcp.model import MCPAnnotations
from aaa.mcp.sources.mcp_oauth import OAuthError, OAuthTokens
from aaa.mcp.spec import OpenAPISource
from aaa.tools.builtin.net import FetchBlocked, FetchPolicy, SafeFetcher
from aaa.tools.tool import SchemaTool, ToolError

METHODS = ("get", "head", "put", "post", "patch", "delete")
FORBIDDEN_HEADERS = {
    "authorization",
    "cookie",
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "proxy-authorization",
    "set-cookie",
}
REF_DEPTH = 8
MAX_FLATTEN = 8
_STRIP = {
    "example",
    "examples",
    "xml",
    "externalDocs",
    "discriminator",
    "deprecated",
    "readOnly",
    "writeOnly",
}
JSON_TYPES = ("application/json",)


class Unsupported(Exception):
    """An operation uses something the generator will not guess at."""


@dataclass
class Loaded:
    tools: list[SchemaTool] = field(default_factory=list)
    annotations: dict[str, MCPAnnotations] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# -- the document ----------------------------------------------------------------------------------
async def read_document(src: OpenAPISource, fetcher: SafeFetcher | None = None) -> dict[str, Any]:
    ref = src.document
    if ref.startswith(("http://", "https://")):
        u = httpx.URL(ref)
        if u.scheme != "https":
            raise SpecError(f"openapi: the document URL must be https ({ref})")
        f = fetcher or SafeFetcher(
            FetchPolicy.strict(
                [u.host], max_bytes=5_000_000, content_types=("application/", "text/")
            )
        )
        try:
            res = await f.request("GET", ref, raw=True)
        except FetchBlocked as e:
            raise SpecError(f"openapi: cannot fetch the document ({e})") from None
        if res.status >= 400:
            raise SpecError(f"openapi: fetching the document gave HTTP {res.status}")
        raw = res.text
        name = u.path
    else:
        p = Path(ref)
        try:
            raw = p.read_text()
        except OSError as e:
            raise SpecError(f"openapi: cannot read {ref!r} ({e.strerror})") from None
        name = p.name
    try:
        if name.lower().endswith((".yaml", ".yml")) or not raw.lstrip().startswith("{"):
            try:
                import yaml  # pyright: ignore[reportMissingModuleSource]
            except ImportError:
                raise SpecError(
                    "openapi: this document is YAML; install PyYAML: pip install 'autonomous-agent-architecture[mcp-config]'"
                ) from None
            doc = yaml.safe_load(raw)
        else:
            doc = json.loads(raw)
    except SpecError:
        raise
    except Exception as e:  # noqa: BLE001
        raise SpecError(f"openapi: {ref}: not valid JSON/YAML ({str(e)[:150]})") from None
    if not isinstance(doc, dict):
        raise SpecError("openapi: the document must be an object")
    version = str(doc.get("openapi", ""))
    if "swagger" in doc:
        raise SpecError(
            "openapi: this is a Swagger 2.0 document; only OpenAPI 3.0 and 3.1 are supported (convert it first)"
        )
    if not version.startswith(("3.0", "3.1")):
        raise SpecError(f"openapi: unsupported version {version!r}; expected 3.0.x or 3.1.x")
    return doc


# -- schemas ---------------------------------------------------------------------------------------
class Resolver:
    def __init__(self, doc: dict[str, Any]) -> None:
        self.doc = doc

    def ref(self, ref: str) -> Any:
        if not ref.startswith("#/"):
            raise Unsupported(f"remote $ref {ref!r}")
        node: Any = self.doc
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                raise Unsupported(f"unresolvable $ref {ref!r}")
            node = node[part]
        return node

    def resolve(self, node: Any, depth: int = 0, seen: tuple[str, ...] = ()) -> Any:
        """The node with local `$ref`s inlined, OpenAPI-only keywords removed and 3.0 idioms turned into JSON Schema."""
        if isinstance(node, list):
            return [self.resolve(x, depth, seen) for x in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str):
                raise Unsupported("a $ref that is not a string")
            if ref in seen or depth >= REF_DEPTH:
                return {}  # a cycle, or deeper than we go: "anything"
            return self.resolve(self.ref(ref), depth + 1, (*seen, ref))
        out = {k: self.resolve(v, depth, seen) for k, v in node.items() if k not in _STRIP}
        if "allOf" in out and isinstance(out["allOf"], list):
            merged: dict[str, Any] = {k: v for k, v in out.items() if k != "allOf"}
            for part in out["allOf"]:
                if not isinstance(part, dict):
                    continue
                for k, v in part.items():
                    if k == "properties":
                        merged.setdefault("properties", {}).update(v)
                    elif k == "required":
                        merged["required"] = sorted({*merged.get("required", []), *v})
                    else:
                        merged.setdefault(k, v)
            out = merged
        if out.pop("nullable", False) is True and "type" in out:
            t = out["type"]
            out["type"] = [*t, "null"] if isinstance(t, list) else [t, "null"]
        for ex, bound in (("exclusiveMinimum", "minimum"), ("exclusiveMaximum", "maximum")):
            if (
                out.get(ex) is True and bound in out
            ):  # OpenAPI 3.0 boolean form -> JSON Schema number form
                out[ex] = out.pop(bound)
            elif out.get(ex) is False:
                del out[ex]
        return out


# -- one operation -----------------------------------------------------------------------------------
@dataclass
class Param:
    arg: str  # the tool-argument name
    name: str  # the wire name
    where: str  # path | query | header
    required: bool
    schema: dict[str, Any]
    description: str = ""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:64] or "operation"


def _is_array(schema: dict[str, Any]) -> bool:
    t = schema.get("type")
    return t == "array" or (isinstance(t, list) and "array" in t)


def _is_object(schema: dict[str, Any]) -> bool:
    t = schema.get("type")
    return t == "object" or (isinstance(t, list) and "object" in t) or "properties" in schema


class Operation:
    def __init__(
        self,
        method: str,
        path: str,
        op: dict[str, Any],
        shared: list[Any],
        rs: Resolver,
        src: OpenAPISource,
    ) -> None:
        self.method, self.path = method, path
        self.op_id = str(op.get("operationId") or f"{method}_{_slug(path)}")
        resolved = [rs.resolve(p) for p in (*shared, *op.get("parameters", []))]
        by_key = {
            (p.get("in"), p.get("name")): p for p in resolved if isinstance(p, dict)
        }  # op level overrides path level
        resolved_params = list(by_key.values())
        self.params: list[Param] = []
        for p in resolved_params:
            loc, name = p.get("in"), str(p.get("name", ""))
            if loc not in ("path", "query", "header"):
                raise Unsupported(f"{loc} parameter {name!r}")
            if loc == "header" and name.lower() in FORBIDDEN_HEADERS:
                raise Unsupported(
                    f"header parameter {name!r} (credentials and framing headers are never set by a caller)"
                )
            if (
                p.get("style") not in (None, "simple" if loc in ("path", "header") else "form")
                or p.get("explode") is False
                or "content" in p
            ):
                raise Unsupported(f"non-default serialization of parameter {name!r}")
            schema = p.get("schema") or {}
            if _is_object(schema):
                raise Unsupported(f"object-valued parameter {name!r}")
            if loc == "path" and _is_array(schema):
                raise Unsupported(f"array path parameter {name!r}")
            self.params.append(
                Param(
                    name,
                    name,
                    loc,
                    bool(p.get("required")) or loc == "path",
                    schema,
                    str(p.get("description") or ""),
                )
            )
        self.body_schema: dict[str, Any] | None = None
        self.body_required = False
        body = rs.resolve(op.get("requestBody")) if op.get("requestBody") else None
        if body:
            content = body.get("content") or {}
            jtype = next(
                (
                    t
                    for t in content
                    if t.split(";")[0].strip().lower() in JSON_TYPES
                    or t.split(";")[0].strip().lower().endswith("+json")
                ),
                None,
            )
            if jtype is None:
                raise Unsupported(
                    f"request body of type {sorted(content)} (only JSON bodies are supported)"
                )
            self.body_schema = content[jtype].get("schema") or {}
            self.body_required = bool(body.get("required"))
        names = [p.arg for p in self.params]
        dupes = {n for n in names if names.count(n) > 1}
        for p in self.params:
            if p.arg in dupes:
                p.arg = f"{p.where}_{p.name}"
        self.flatten = False
        if self.body_schema is not None and _is_object(self.body_schema):
            props = self.body_schema.get("properties", {})
            if (
                0 < len(props) <= MAX_FLATTEN
                and not (set(props) & {p.arg for p in self.params})
                and not self.body_schema.get("additionalProperties") is True
            ):
                self.flatten = True
        resp = None
        for code in sorted(op.get("responses", {})):
            if str(code).startswith("2"):
                resp = rs.resolve(op["responses"][code])
                break
        self.response_schema: dict[str, Any] | None = None
        if resp:
            for t, v in (resp.get("content") or {}).items():
                if t.split(";")[0].strip().lower() in JSON_TYPES or t.lower().endswith("+json"):
                    sch = (v or {}).get("schema")
                    if isinstance(sch, dict) and _is_object(sch):
                        self.response_schema = sch
                    break
        text = " ".join(
            x for x in (str(op.get("summary") or ""), str(op.get("description") or "")) if x
        ).strip()
        self.description = (
            src.description_overrides.get(self.op_id) or text or f"{method.upper()} {path}"
        )[: src.description_max_chars]

    # the tool's argument schema -------------------------------------------------------------------
    def input_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in self.params:
            sch = dict(p.schema) or {"type": "string"}
            if p.description:
                sch.setdefault("description", p.description)
            props[p.arg] = sch
            if p.required:
                required.append(p.arg)
        if self.body_schema is not None:
            if self.flatten:
                for k, v in self.body_schema.get("properties", {}).items():
                    props[k] = v
                if self.body_required:
                    required += [r for r in self.body_schema.get("required", []) if r in props]
            else:
                props["body"] = self.body_schema
                if self.body_required:
                    required.append("body")
        out: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
        if required:
            out["required"] = sorted(set(required))
        return out

    # the call --------------------------------------------------------------------------------------
    def build(
        self, args: dict[str, Any], base: str, src: OpenAPISource
    ) -> tuple[str, bytes | None, dict[str, str]]:
        path = self.path
        query: list[tuple[str, str]] = []
        headers: dict[str, str] = {}
        for p in self.params:
            if p.arg not in args:
                continue
            v = args[p.arg]
            if p.where == "path":
                path = path.replace("{" + p.name + "}", quote(_scalar(v), safe=""))
            elif p.where == "query":
                query += [(p.name, _scalar(x)) for x in (v if isinstance(v, list) else [v])]
            else:
                val = ",".join(_scalar(x) for x in v) if isinstance(v, list) else _scalar(v)
                if re.search(r"[\r\n\x00]", val):
                    raise ToolError(f"header {p.name} contains a control character")
                headers[p.name] = val
        data: bytes | None = None
        if self.body_schema is not None:
            if self.flatten:
                payload: Any = {
                    k: args[k] for k in self.body_schema.get("properties", {}) if k in args
                } or None
            else:
                payload = args.get("body")
            if payload is not None:
                data = json.dumps(payload, separators=(",", ":")).encode()
                if len(data) > src.max_request_bytes:
                    raise ToolError("the request body is too large")
                headers["content-type"] = "application/json"
        url = base.rstrip("/") + path
        if query:
            url += "?" + urlencode(query)
        return url, data, headers


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str | int | float):
        return str(v)
    raise ToolError("a parameter value must be a string, number or boolean")


# -- the source ---------------------------------------------------------------------------------------
def _base_url(src: OpenAPISource, doc: dict[str, Any]) -> str:
    base = src.base_url
    if base is None:
        servers = doc.get("servers") or []
        if not servers or not isinstance(servers[0], dict) or not servers[0].get("url"):
            raise SpecError("openapi: the document has no servers entry; set base_url")
        base = str(servers[0]["url"])
        if "{" in base:
            raise SpecError("openapi: the server URL uses variables; set base_url explicitly")
    u = urlsplit(base)
    if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password:
        raise SpecError(
            f"openapi: base URL {base!r} must be http(s) with a host and no credentials"
        )
    return base


def _policy(src: OpenAPISource, base: str) -> FetchPolicy:
    u = httpx.URL(base)
    host = (u.host or "").lower()
    if not any(
        fnmatch.fnmatchcase(host, h.lower()) or (h.startswith("*.") and host == h[2:].lower())
        for h in src.allowed_hosts
    ):
        raise SpecError(
            f"openapi: the API host {host!r} is not in allowed_hosts {src.allowed_hosts}"
        )
    port = u.port or (443 if u.scheme == "https" else 80)
    return FetchPolicy(
        allowed_hosts=tuple(src.allowed_hosts),
        allowed_ports=(port,),
        allow_http=u.scheme == "http",
        max_redirects=3,
        max_bytes=src.max_response_bytes,
        timeout_s=src.timeout_s,
        content_types=(
            "application/json",
            "application/problem+json",
            "application/vnd.",
            "application/xml",
            "text/",
        ),
        user_agent="aaa-mcp-openapi/1.0",
        allow_private_networks=src.allow_private_networks,
    )


class _Caller:
    def __init__(
        self,
        op: Operation,
        src: OpenAPISource,
        base: str,
        fetcher: SafeFetcher,
        gate: asyncio.Semaphore,
        tokens: OAuthTokens | None = None,
    ) -> None:
        self.op, self.src, self.base, self.fetcher, self.gate = op, src, base, fetcher, gate
        self.tokens = tokens
        self.host = (httpx.URL(base).host or "").lower()
        self.secret = Secret.parse(src.auth.secret, field="auth.secret") if src.auth else None

    def _auth(self) -> tuple[dict[str, str], list[tuple[str, str]]]:
        a = self.src.auth
        if a is None or self.secret is None:
            return {}, []
        value = self.secret.value()
        if a.scheme == "bearer":
            return {"authorization": f"Bearer {value}"}, []
        if a.scheme == "basic":
            import base64

            return {
                "authorization": "Basic "
                + base64.b64encode(f"{a.username}:{value}".encode()).decode()
            }, []
        return ({a.name: value}, []) if a.location == "header" else ({}, [(a.name, value)])

    async def __call__(self, args: dict[str, Any]) -> Any:
        url, data, headers = self.op.build(args, self.base, self.src)
        auth_headers, auth_query = self._auth()
        if auth_query:
            url += ("&" if "?" in url else "?") + urlencode(auth_query)
        async with self.gate:
            token: str | None = None
            for attempt in range(2 if self.tokens is not None else 1):
                if self.tokens is not None:
                    try:
                        token = await self.tokens.get(force=attempt > 0, stale=token)
                    except OAuthError:
                        raise ToolError(
                            "the API call was not made: no access token could be obtained"
                        ) from None
                    auth_headers = {"authorization": f"Bearer {token}"}
                try:
                    res = await self.fetcher.request(
                        self.op.method.upper(),
                        url,
                        headers={**headers, "accept": "application/json, text/plain;q=0.5"},
                        body=data,
                        auth=auth_headers or None,
                        auth_host=self.host,
                        same_host_redirects_only=True,
                        raw=True,
                    )
                except FetchBlocked as e:
                    raise ToolError(f"the API call was not made or failed: {e}") from None
                if res.status != 401 or self.tokens is None:
                    break  # a 401 with a token: the token was probably revoked or expired; get a new one and try once more
        if res.status >= 400:
            raise ToolError(f"the API answered HTTP {res.status}: {res.text[:300]}")
        if self.op.method == "head":
            return {"status": res.status}
        if res.content_type.endswith("json") and not res.truncated:
            try:
                return json.loads(res.text)
            except ValueError:
                return res.text
        return res.text + ("\n[truncated]" if res.truncated else "")


async def load_openapi(src: OpenAPISource, *, fetcher: SafeFetcher | None = None) -> Loaded:
    doc = await read_document(src)
    rs = Resolver(doc)
    base = _base_url(src, doc)
    policy = _policy(src, base)
    shared_fetcher = fetcher or SafeFetcher(policy)
    if src.auth is not None:
        Secret.parse(
            src.auth.secret, field="auth.secret"
        ).value()  # a missing variable is a startup error, not a runtime one
    tokens: OAuthTokens | None = None
    if src.oauth is not None:
        tokens = OAuthTokens(
            "openapi",
            src.oauth,
            allow_private_networks=src.allow_private_networks,
            allow_insecure_loopback=src.allow_private_networks,
        )
        try:
            await tokens.get()  # an unusable authorization server is a startup error
        except OAuthError as e:
            raise SpecError(f"openapi {src.document}: {e}") from None
    gate = asyncio.Semaphore(src.max_concurrency)
    out = Loaded()
    skipped: list[str] = []
    seen: dict[str, str] = {}
    for path, item in sorted((doc.get("paths") or {}).items()):
        if not isinstance(item, dict):
            continue
        try:
            shared = [rs.resolve(p) for p in item.get("parameters", [])]
        except Unsupported:
            shared = []
        for method in METHODS:
            raw = item.get(method)
            if not isinstance(raw, dict):
                continue
            op_id = str(raw.get("operationId") or f"{method}_{_slug(path)}")
            if not any(fnmatch.fnmatchcase(op_id, g) for g in src.operations) or any(
                fnmatch.fnmatchcase(op_id, g) for g in src.exclude_operations
            ):
                continue
            try:
                op = Operation(method, path, raw, shared, rs, src)
            except Unsupported as e:
                skipped.append(f"{op_id} ({e})")
                continue
            name = _slug(op.op_id)
            if name in seen:
                raise SpecError(
                    f"openapi: operations {seen[name]!r} and {op.op_id!r} both map to the tool name {name!r}; exclude one or give them distinct operationIds"
                )
            seen[name] = op.op_id
            approval = method in src.methods_requiring_approval and not any(
                fnmatch.fnmatchcase(op.op_id, g) for g in src.no_approval
            )
            tool = SchemaTool(
                _Caller(op, src, base, shared_fetcher, gate, tokens),
                name=name,
                description=op.description,
                parameters=op.input_schema(),
                output_schema=op.response_schema if src.use_response_schemas else None,
                requires_approval=approval,
                delivery="at_most_once" if method in ("post", "patch") else "at_least_once",
                timeout_s=src.timeout_s + 5,
            )
            out.tools.append(tool)
            out.annotations[name] = MCPAnnotations(
                read_only=method in ("get", "head"),
                destructive=None if method in ("get", "head") else method == "delete" or None,
                idempotent=method in ("get", "head", "put", "delete") or None,
                open_world=True,
            )
            if not approval and method not in ("get", "head"):
                out.notes.append(
                    f"WARNING: {method.upper()} {path} ({op.op_id}) changes state and is exposed WITHOUT approval"
                )
    if skipped:
        out.notes.append(
            f"{len(skipped)} operation(s) skipped as unsupported: "
            + "; ".join(skipped[:12])
            + ("; ..." if len(skipped) > 12 else "")
        )
    if urlsplit(base).scheme == "http":
        out.notes.append(f"WARNING: {urlsplit(base).hostname} is called over plain http")
    if src.allow_private_networks:
        out.notes.append(
            "WARNING: allow_private_networks is on: the API may be on a private address"
        )
    return out
