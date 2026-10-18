"""The OpenAPI source, against a real HTTP backend on a local port: mapping, calling, credentials, redirects, SSRF and limits."""

import asyncio
import json
import socket

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route

from aaa.mcp.compile import acompile_spec
from aaa.mcp.config import SpecError
from aaa.mcp.spec import MCPSpec


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Backend:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str, dict[str, str], bytes]] = []
        self.other_seen: list[tuple[str, dict[str, str]]] = []
        self.port, self.other_port = free_port(), free_port()
        self.base = f"http://127.0.0.1:{self.port}"

    def record(self, request: Request, body: bytes = b"") -> None:
        self.seen.append(
            (
                request.method,
                request.scope["raw_path"].decode()
                + (
                    "?" + request.scope["query_string"].decode()
                    if request.scope["query_string"]
                    else ""
                ),
                dict(request.headers),
                body,
            )
        )

    def app(self) -> Starlette:
        async def pets(request: Request) -> Response:
            self.record(request)
            if request.method == "POST":
                body = await request.body()
                self.seen[-1] = (*self.seen[-1][:3], body)
                return JSONResponse({"id": 7, **json.loads(body)}, status_code=201)
            return JSONResponse(
                [{"id": 1, "name": "rex", "tags": request.query_params.getlist("tag")}]
            )

        async def pet(request: Request) -> Response:
            body = await request.body()
            self.record(request, body)
            pid = request.path_params["petId"]
            if request.method == "DELETE":
                return Response(status_code=204)
            if request.method == "PUT":
                return JSONResponse({"id": pid, **json.loads(body)})
            if pid == "missing":
                return JSONResponse({"error": "no such pet"}, status_code=404)
            return JSONResponse({"id": pid, "name": "rex", "score": 1.5})

        async def whoami(request: Request) -> Response:
            self.record(request)
            return JSONResponse(
                {
                    "authorization": request.headers.get("authorization"),
                    "x-api-key": request.headers.get("x-api-key"),
                    "query": dict(request.query_params),
                }
            )

        async def trace(request: Request) -> Response:
            self.record(request)
            return JSONResponse({"x-trace": request.headers.get("x-trace")})

        async def big(request: Request) -> Response:
            self.record(request)
            return PlainTextResponse("x" * 3_000_000)

        async def slow(request: Request) -> Response:
            self.record(request)
            await asyncio.sleep(3)
            return PlainTextResponse("late")

        async def boom(request: Request) -> Response:
            self.record(request)
            return PlainTextResponse("internal secret detail " * 50, status_code=500)

        async def same(request: Request) -> Response:
            self.record(request)
            return RedirectResponse("/pets/1", status_code=302)

        async def cross(request: Request) -> Response:
            self.record(request)
            return RedirectResponse(f"http://localhost:{self.other_port}/leak", status_code=302)

        async def post_redirect(request: Request) -> Response:
            self.record(request, await request.body())
            return RedirectResponse("/pets", status_code=307)

        async def binary(request: Request) -> Response:
            self.record(request)
            return Response(b"\x00\x01\x02", media_type="application/octet-stream")

        async def text(request: Request) -> Response:
            self.record(request)
            return PlainTextResponse("hello text")

        return Starlette(
            routes=[
                Route("/pets", pets, methods=["GET", "POST"]),
                Route("/pets/{petId}", pet, methods=["GET", "PUT", "DELETE"]),
                Route("/whoami", whoami),
                Route("/trace", trace),
                Route("/big", big),
                Route("/slow", slow),
                Route("/boom", boom),
                Route("/same", same),
                Route("/cross", cross),
                Route("/post-redirect", post_redirect, methods=["POST"]),
                Route("/binary", binary),
                Route("/text", text),
            ]
        )

    def other_app(self) -> Starlette:
        async def leak(request: Request) -> Response:
            self.other_seen.append((request.method, dict(request.headers)))
            return JSONResponse({"leaked": True})

        return Starlette(routes=[Route("/leak", leak)])


@pytest.fixture
async def backend():
    b = Backend()
    servers = [
        uvicorn.Server(uvicorn.Config(b.app(), host="127.0.0.1", port=b.port, log_level="error")),
        uvicorn.Server(
            uvicorn.Config(b.other_app(), host="127.0.0.1", port=b.other_port, log_level="error")
        ),
    ]
    tasks = [asyncio.create_task(s.serve()) for s in servers]
    while not all(s.started for s in servers):
        await asyncio.sleep(0.02)
    yield b
    for s in servers:
        s.should_exit = True
    await asyncio.gather(*tasks)


def schema(base: str) -> dict:
    pet = {"type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}}}
    return {
        "openapi": "3.0.3",
        "info": {"title": "pets", "version": "1"},
        "servers": [{"url": base}],
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "summary": "List pets",
                    "description": "Returns pets, optionally filtered.",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {"type": "integer", "minimum": 1},
                        },
                        {
                            "name": "tag",
                            "in": "query",
                            "schema": {"type": "array", "items": {"type": "string"}},
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "ok",
                            "content": {
                                "application/json": {"schema": {"type": "array", "items": pet}}
                            },
                        }
                    },
                },
                "post": {
                    "operationId": "createPet",
                    "summary": "Create a pet",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["name"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "tag": {"type": "string", "nullable": True},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {
                            "description": "created",
                            "content": {"application/json": {"schema": pet}},
                        }
                    },
                },
            },
            "/pets/{petId}": {
                "parameters": [
                    {"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "get": {
                    "operationId": "getPet",
                    "summary": "Get a pet",
                    "responses": {
                        "200": {
                            "description": "ok",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "name": {"type": "string"},
                                            "score": {"type": "number"},
                                        },
                                        "required": ["id", "name"],
                                    }
                                }
                            },
                        }
                    },
                },
                "put": {
                    "operationId": "replacePet",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "ok"}},
                },
                "delete": {
                    "operationId": "deletePet",
                    "responses": {"204": {"description": "gone"}},
                },
            },
            "/whoami": {
                "get": {"operationId": "whoami", "responses": {"200": {"description": "ok"}}}
            },
            "/trace": {
                "get": {
                    "operationId": "trace",
                    "parameters": [
                        {"name": "X-Trace", "in": "header", "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            },
            "/big": {"get": {"operationId": "big", "responses": {"200": {"description": "ok"}}}},
            "/slow": {"get": {"operationId": "slow", "responses": {"200": {"description": "ok"}}}},
            "/boom": {"get": {"operationId": "boom", "responses": {"200": {"description": "ok"}}}},
            "/same": {
                "get": {"operationId": "sameRedirect", "responses": {"200": {"description": "ok"}}}
            },
            "/cross": {
                "get": {"operationId": "crossRedirect", "responses": {"200": {"description": "ok"}}}
            },
            "/post-redirect": {
                "post": {
                    "operationId": "postRedirect",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"a": {"type": "string"}},
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "ok"}},
                }
            },
            "/binary": {
                "get": {"operationId": "binary", "responses": {"200": {"description": "ok"}}}
            },
            "/text": {"get": {"operationId": "text", "responses": {"200": {"description": "ok"}}}},
        },
    }


async def build(tmp_path, backend, *, doc=None, **over):
    path = tmp_path / "openapi.json"
    path.write_text(json.dumps(doc or schema(backend.base)))
    src = {
        "type": "openapi",
        "document": str(path),
        "allowed_hosts": ["127.0.0.1"],
        "allow_private_networks": True,
        **over,
    }
    return await acompile_spec(
        MCPSpec.model_validate({"sources": [src], "tools": {"expose_approval_tools": True}})
    )


def text_of(r):
    return r.text


# ===================================================================== mapping
async def test_operations_become_tools_with_schemas_hints_and_default_approval(tmp_path, backend):
    spec = MCPSpec.model_validate(
        {
            "sources": [
                {
                    "type": "openapi",
                    "document": str(_write(tmp_path, schema(backend.base))),
                    "allowed_hosts": ["127.0.0.1"],
                    "allow_private_networks": True,
                }
            ]
        }
    )
    core = await acompile_spec(spec)
    offered = {t.name: t for t in core.list_tools()}
    assert (
        "listPets" in offered and "getPet" in offered and "createPet" not in offered
    )  # state-changing operations are hidden
    assert {"createPet", "replacePet", "deletePet", "postRedirect"} <= set(core.hidden)
    lp = offered["listPets"]
    assert lp.description == "List pets Returns pets, optionally filtered."
    assert (
        lp.input_schema["properties"]["tag"]["type"] == "array"
        and lp.input_schema["additionalProperties"] is False
    )
    assert (
        lp.annotations
        and lp.annotations.read_only is True
        and lp.annotations.idempotent is True
        and lp.annotations.open_world is True
    )
    assert offered["getPet"].input_schema["required"] == ["petId"]

    full = await build(tmp_path, backend)
    by = {t.name: t for t in full.list_tools()}
    cp = by["createPet"]
    assert cp.input_schema["required"] == ["name"] and set(cp.input_schema["properties"]) == {
        "name",
        "tag",
    }  # a small body is flattened
    assert cp.input_schema["properties"]["tag"]["type"] == [
        "string",
        "null",
    ]  # OpenAPI 3.0 nullable became JSON Schema
    assert (
        cp.annotations and cp.annotations.read_only is False and cp.annotations.idempotent is None
    )
    assert (
        by["deletePet"].annotations
        and by["deletePet"].annotations.destructive is True
        and by["deletePet"].annotations.idempotent is True
    )
    assert by["replacePet"].annotations and by["replacePet"].annotations.idempotent is True


def _write(tmp_path, doc):
    p = tmp_path / "doc.json"
    p.write_text(json.dumps(doc))
    return p


async def test_no_approval_exposes_an_operation_and_warns_about_state_changes(tmp_path, backend):
    core = await build(
        tmp_path,
        backend,
        no_approval=["createPet"],
        methods_requiring_approval=["put", "delete", "post"],
    )
    assert any(
        "WARNING: POST /pets (createPet) changes state and is exposed WITHOUT approval" in n
        for n in core.report
    )
    core2 = await acompile_spec(
        MCPSpec.model_validate(
            {
                "sources": [
                    {
                        "type": "openapi",
                        "document": str(_write(tmp_path, schema(backend.base))),
                        "allowed_hosts": ["127.0.0.1"],
                        "allow_private_networks": True,
                        "no_approval": ["createPet"],
                    }
                ]
            }
        )
    )
    assert "createPet" in {t.name for t in core2.list_tools()}
    assert any("allow_private_networks is on" in n for n in core2.report) and any(
        "plain http" in n for n in core2.report
    )


async def test_operation_filters_and_description_controls(tmp_path, backend):
    core = await build(
        tmp_path,
        backend,
        operations=["*Pet*", "list*"],
        exclude_operations=["delete*"],
        description_overrides={"getPet": "Fetch one pet by id."},
        description_max_chars=60,
    )
    names = {t.name for t in core.list_tools()}
    assert "deletePet" not in names and "whoami" not in names and "listPets" in names
    d = {t.name: t.description for t in core.list_tools()}
    assert d["getPet"] == "Fetch one pet by id." and len(d["listPets"]) <= 60


# ===================================================================== calling
async def test_calls_are_built_from_the_arguments(tmp_path, backend):
    core = await build(tmp_path, backend)
    r = await core.call("listPets", {"limit": 5, "tag": ["a", "b c"]})
    assert json.loads(r.text) == [{"id": 1, "name": "rex", "tags": ["a", "b c"]}]
    method, path, _, _ = backend.seen[-1]
    assert (method, path) == ("GET", "/pets?limit=5&tag=a&tag=b+c")

    assert json.loads((await core.call("getPet", {"petId": "a b?c"})).text)["id"] == "a b?c"
    assert backend.seen[-1][1] == "/pets/a%20b%3Fc"  # a path value cannot add a query string...
    caller = core.registry.get("getPet").fn  # type: ignore[union-attr]
    url, _, _ = caller.op.build({"petId": "../admin/x"}, backend.base, caller.src)
    assert url == f"{backend.base}/pets/..%2Fadmin%2Fx"  # ...or path segments

    made = json.loads((await core.call("createPet", {"name": "fido", "tag": None})).text)
    assert made == {"id": 7, "name": "fido", "tag": None}
    assert (
        json.loads(backend.seen[-1][3]) == {"name": "fido", "tag": None}
        and backend.seen[-1][2]["content-type"] == "application/json"
    )
    assert (await core.call("deletePet", {"petId": "1"})).text == ""
    assert json.loads((await core.call("replacePet", {"petId": "9", "name": "max"})).text) == {
        "id": "9",
        "name": "max",
    }
    assert (await core.call("text")).text == "hello text"


async def test_arguments_are_validated_before_any_request(tmp_path, backend):
    core = await build(tmp_path, backend)
    before = len(backend.seen)
    for name, args in (
        ("listPets", {"limit": 0}),
        ("listPets", {"limit": "five"}),
        ("getPet", {}),
        ("createPet", {"tag": "x"}),
        ("listPets", {"nope": 1}),
    ):
        assert (await core.call(name, args)).is_error, (name, args)
    assert len(backend.seen) == before


async def test_api_errors_are_reported_with_a_short_body_and_no_headers(tmp_path, backend):
    core = await build(tmp_path, backend)
    r = await core.call("getPet", {"petId": "missing"})
    assert r.is_error and r.text.startswith("the API answered HTTP 404") and "no such pet" in r.text
    r = await core.call("boom")
    assert r.is_error and "HTTP 500" in r.text and len(r.text) < 400  # the body is cut


async def test_binary_and_oversized_responses_are_not_passed_on(tmp_path, backend):
    core = await build(tmp_path, backend, max_response_bytes=10_000)
    assert (await core.call("binary")).is_error
    r = await core.call("big")
    assert not r.is_error and len(r.text) < 11_000 and r.text.endswith("[truncated]")


async def test_a_slow_api_times_out_and_oversized_requests_are_refused(tmp_path, backend):
    core = await build(tmp_path, backend, timeout_s=0.5)
    r = await core.call("slow")
    assert r.is_error and "timed out" in r.text
    core2 = await build(tmp_path, backend, max_request_bytes=100)
    before = len(backend.seen)
    r = await core2.call("createPet", {"name": "x" * 500})
    assert r.is_error and "too large" in r.text and len(backend.seen) == before


async def test_header_parameters_are_sent_and_cannot_smuggle_headers(tmp_path, backend):
    core = await build(tmp_path, backend)
    assert json.loads((await core.call("trace", {"X-Trace": "abc"})).text) == {"x-trace": "abc"}
    r = await core.call("trace", {"X-Trace": "a\r\nX-Evil: 1"})
    assert r.is_error and "control character" in r.text


# ===================================================================== credentials
@pytest.mark.parametrize(
    ("auth", "env", "check"),
    [
        (
            {"scheme": "bearer", "secret": "env:API_TOKEN"},
            {"API_TOKEN": "tok-123"},
            lambda j: j["authorization"] == "Bearer tok-123",
        ),
        (
            {"scheme": "api_key", "secret": "${API_TOKEN}", "name": "X-API-Key"},
            {"API_TOKEN": "tok-123"},
            lambda j: j["x-api-key"] == "tok-123",
        ),
        (
            {
                "scheme": "api_key",
                "secret": "env:API_TOKEN",
                "location": "query",
                "name": "api_key",
            },
            {"API_TOKEN": "tok-123"},
            lambda j: j["query"] == {"api_key": "tok-123"},
        ),
        (
            {"scheme": "basic", "secret": "env:API_TOKEN", "username": "ann"},
            {"API_TOKEN": "pw"},
            lambda j: j["authorization"] == "Basic YW5uOnB3",
        ),
    ],
)
async def test_credentials_come_from_the_environment_and_reach_the_api(
    tmp_path, backend, monkeypatch, auth, env, check
):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    core = await build(tmp_path, backend, auth=auth)
    assert check(json.loads((await core.call("whoami")).text))
    dumped = (
        json.dumps([t.input_schema for t in core.list_tools()])
        + json.dumps(core.report)
        + json.dumps(core.spec.model_dump(mode="json"))
    )
    assert (
        "tok-123" not in dumped and "YW5uOnB3" not in dumped
    )  # never in schemas, reports or the stored configuration


async def test_a_literal_secret_is_refused_and_a_missing_variable_fails_at_startup(
    tmp_path, backend, monkeypatch
):
    with pytest.raises(Exception, match="never the value"):
        MCPSpec.model_validate(
            {
                "sources": [
                    {
                        "type": "openapi",
                        "document": "x.json",
                        "allowed_hosts": ["h"],
                        "auth": {"scheme": "bearer", "secret": "sk-live-abc"},
                    }
                ]
            }
        )
    with pytest.raises(Exception, match="username"):
        MCPSpec.model_validate(
            {
                "sources": [
                    {
                        "type": "openapi",
                        "document": "x.json",
                        "allowed_hosts": ["h"],
                        "auth": {"scheme": "basic", "secret": "env:X"},
                    }
                ]
            }
        )
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(SpecError, match="NOT_SET_ANYWHERE is not set"):
        await build(tmp_path, backend, auth={"scheme": "bearer", "secret": "env:NOT_SET_ANYWHERE"})


# ===================================================================== redirects and SSRF
async def test_redirects_stay_on_the_apis_host_and_never_carry_credentials_away(
    tmp_path, backend, monkeypatch
):
    monkeypatch.setenv("API_TOKEN", "tok-123")
    core = await build(tmp_path, backend, auth={"scheme": "bearer", "secret": "env:API_TOKEN"})
    ok = await core.call("sameRedirect")
    assert (
        not ok.is_error and json.loads(ok.text)["id"] == "1"
    )  # a same-host GET redirect is followed
    r = await core.call("crossRedirect")
    assert r.is_error and "another host was refused" in r.text
    assert (
        backend.other_seen == []
    )  # the second host never received a request, let alone the credential
    assert not any("localhost" in h.get("host", "") for _, _, h, _ in backend.seen)


async def test_a_cross_host_redirect_is_refused_even_without_credentials_and_the_allow_list_still_applies(
    tmp_path, backend
):
    core = await build(tmp_path, backend, allowed_hosts=["127.0.0.1", "localhost"])
    r = await core.call("crossRedirect")
    assert r.is_error and backend.other_seen == []


async def test_a_post_is_never_redirected(tmp_path, backend):
    core = await build(tmp_path, backend)
    r = await core.call("postRedirect", {"a": "1"})
    assert r.is_error and "was not followed" in r.text
    assert [m for m, *_ in backend.seen].count(
        "POST"
    ) == 1  # the body was sent once, to the original URL only


async def test_the_apis_host_must_be_allowed_and_private_addresses_need_an_explicit_opt_in(
    tmp_path, backend
):
    with pytest.raises(SpecError, match="not in allowed_hosts"):
        await build(tmp_path, backend, allowed_hosts=["api.example.com"])
    path = _write(tmp_path, schema(backend.base))
    strict = await acompile_spec(
        MCPSpec.model_validate(
            {
                "sources": [
                    {"type": "openapi", "document": str(path), "allowed_hosts": ["127.0.0.1"]}
                ]
            }
        )
    )
    before = len(backend.seen)
    r = await strict.call("listPets")
    assert (
        r.is_error and "non-public address" in r.text and len(backend.seen) == before
    )  # no request was made


async def test_the_document_url_must_be_https(tmp_path):
    spec = MCPSpec.model_validate(
        {
            "sources": [
                {
                    "type": "openapi",
                    "document": "http://example.com/openapi.json",
                    "allowed_hosts": ["example.com"],
                }
            ]
        }
    )
    with pytest.raises(SpecError, match="must be https"):
        await acompile_spec(spec)


# ===================================================================== what is not supported
def doc_with(path_item: dict, base="http://127.0.0.1:1") -> dict:
    return {
        "openapi": "3.1.0",
        "info": {"title": "t", "version": "1"},
        "servers": [{"url": base}],
        "paths": {"/x": path_item},
    }


@pytest.mark.parametrize(
    ("op", "why"),
    [
        (
            {"parameters": [{"name": "s", "in": "cookie", "schema": {"type": "string"}}]},
            "cookie parameter",
        ),
        (
            {
                "parameters": [
                    {
                        "name": "f",
                        "in": "query",
                        "style": "deepObject",
                        "schema": {"type": "object"},
                    }
                ]
            },
            "non-default serialization",
        ),
        (
            {
                "parameters": [
                    {
                        "name": "f",
                        "in": "query",
                        "explode": False,
                        "schema": {"type": "array", "items": {"type": "string"}},
                    }
                ]
            },
            "non-default serialization",
        ),
        (
            {
                "parameters": [
                    {"name": "Authorization", "in": "header", "schema": {"type": "string"}}
                ]
            },
            "never set by a caller",
        ),
        (
            {"parameters": [{"name": "o", "in": "query", "schema": {"type": "object"}}]},
            "object-valued",
        ),
        (
            {"requestBody": {"content": {"multipart/form-data": {"schema": {"type": "object"}}}}},
            "only JSON bodies",
        ),
        (
            {
                "requestBody": {
                    "content": {"application/x-www-form-urlencoded": {"schema": {"type": "object"}}}
                }
            },
            "only JSON bodies",
        ),
        ({"parameters": [{"$ref": "https://evil.example/p.json"}]}, "remote"),
    ],
)
async def test_unsupported_features_skip_the_operation_with_a_reason(tmp_path, backend, op, why):
    doc = doc_with(
        {
            "post": {"operationId": "weird", "responses": {"200": {"description": "ok"}}, **op},
            "get": {"operationId": "fine", "responses": {"200": {"description": "ok"}}},
        },
        backend.base,
    )
    core = await build(tmp_path, backend, doc=doc)
    assert [t.name for t in core.list_tools()] == ["fine"]
    note = next(n for n in core.report if "skipped as unsupported" in n)
    assert "weird" in note and why in note


async def test_only_openapi_3_documents_are_accepted(tmp_path, backend):
    for doc, msg in (
        ({"swagger": "2.0", "paths": {}}, "Swagger 2.0"),
        ({"openapi": "4.0.0", "paths": {}}, "unsupported version"),
        ({"paths": {}}, "unsupported version"),
    ):
        with pytest.raises(SpecError, match=msg):
            await build(tmp_path, backend, doc=doc)
    with pytest.raises(SpecError, match="no servers entry"):
        await build(tmp_path, backend, doc={"openapi": "3.0.0", "paths": {}})
    with pytest.raises(SpecError, match="variables"):
        await build(
            tmp_path,
            backend,
            doc={"openapi": "3.0.0", "servers": [{"url": "https://{host}/v1"}], "paths": {}},
        )


async def test_duplicate_operation_ids_and_clashing_parameter_names(tmp_path, backend):
    dup = doc_with(
        {
            "get": {"operationId": "same", "responses": {}},
            "post": {"operationId": "same", "responses": {}},
        },
        backend.base,
    )
    with pytest.raises(SpecError, match="both map to the tool name"):
        await build(tmp_path, backend, doc=dup)
    clash = doc_with(
        {
            "get": {
                "operationId": "clash",
                "parameters": [
                    {"name": "id", "in": "query", "schema": {"type": "string"}},
                    {"name": "id", "in": "header", "schema": {"type": "string"}},
                ],
                "responses": {},
            }
        },
        backend.base,
    )
    core = await build(tmp_path, backend, doc=clash)
    assert set(core.list_tools()[0].input_schema["properties"]) == {"query_id", "header_id"}


async def test_refs_cycles_allof_and_3_0_numeric_bounds_are_resolved_safely(tmp_path, backend):
    doc = {
        "openapi": "3.0.0",
        "servers": [{"url": backend.base}],
        "components": {
            "schemas": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "next": {"$ref": "#/components/schemas/Node"},
                        "v": {"type": "integer", "minimum": 1, "exclusiveMinimum": True},
                    },
                },
                "Base": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
                "Child": {
                    "allOf": [
                        {"$ref": "#/components/schemas/Base"},
                        {
                            "type": "object",
                            "properties": {"extra": {"type": "string"}},
                            "required": ["extra"],
                        },
                    ]
                },
            }
        },
        "paths": {
            "/graph": {
                "post": {
                    "operationId": "graph",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Node"}}
                        },
                    },
                    "responses": {},
                }
            },
            "/child": {
                "post": {
                    "operationId": "child",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Child"}}
                        },
                    },
                    "responses": {},
                }
            },
        },
    }
    core = await build(tmp_path, backend, doc=doc)
    by = {t.name: t for t in core.list_tools()}
    props = by["graph"].input_schema["properties"]
    assert props["v"] == {
        "type": "integer",
        "exclusiveMinimum": 1,
    }  # 3.0's boolean form became JSON Schema's number form
    assert by["child"].input_schema["required"] == ["extra", "id"] and set(
        by["child"].input_schema["properties"]
    ) == {"id", "extra"}


async def test_response_schemas_are_only_advertised_when_asked_for(tmp_path, backend):
    plain = await build(tmp_path, backend)
    assert all(t.output_schema is None for t in plain.list_tools())
    typed = await build(tmp_path, backend, use_response_schemas=True)
    by = {t.name: t for t in typed.list_tools()}
    assert by["getPet"].output_schema is not None
    r = await typed.call("getPet", {"petId": "1"})
    assert r.structured == {"id": "1", "name": "rex", "score": 1.5}
    # an API that answers with something other than its own schema is reported, not passed through as structured data
    bad_doc = schema(backend.base)
    bad_doc["paths"]["/pets/{petId}"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["required"] = ["nonexistent_field"]
    bad = await build(tmp_path, backend, doc=bad_doc, use_response_schemas=True)
    assert (await bad.call("getPet", {"petId": "1"})).is_error


async def test_the_concurrency_cap_serialises_calls(tmp_path, backend):
    core = await build(tmp_path, backend, max_concurrency=1, timeout_s=10)
    t0 = asyncio.get_running_loop().time()
    await asyncio.gather(core.call("slow"), core.call("slow"))
    assert asyncio.get_running_loop().time() - t0 >= 5.5  # two 3 s calls, one at a time
