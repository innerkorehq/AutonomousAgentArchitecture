"""OAuth client credentials for an HTTP upstream: the token is fetched safely, cached, refreshed, sent only to the upstream, and nothing leaks."""

import asyncio
import base64
import json
import socket
from urllib.parse import parse_qs

import pytest

from aaa.mcp.compile import acompile_spec
from aaa.mcp.config import SpecError
from aaa.mcp.sources import mcp_oauth
from aaa.mcp.spec import MCPSpec

SECRET = "s3cr3t-canary-value"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AuthServer:
    """A token endpoint that issues tok-1, tok-2, ... and remembers what it was asked."""

    def __init__(self):
        self.requests: list[dict] = []
        self.valid: set[str] = set()
        self.expires_in: object = 3600
        self.mode = "ok"  # ok | status500 | not_json | wrong_type | huge | redirect

    async def app(self, scope, receive, send):
        if scope["type"] != "http":
            return
        body = b""
        while True:
            m = await receive()
            body += m.get("body", b"")
            if not m.get("more_body"):
                break
        headers = {k.decode(): v.decode() for k, v in scope["headers"]}
        self.requests.append(
            {
                "headers": headers,
                "form": {k: v[0] for k, v in parse_qs(body.decode()).items()},
                "path": scope["path"],
            }
        )

        async def reply(status, payload, ctype="application/json", extra=()):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [(b"content-type", ctype.encode()), *extra],
                }
            )
            await send({"type": "http.response.body", "body": data})

        n = len(self.requests)
        if self.mode == "status500":
            return await reply(500, {"error": "server_error", "detail": SECRET})
        if self.mode == "not_json":
            return await reply(200, b"<html>no</html>", "text/html")
        if self.mode == "redirect":
            return await reply(302, b"", extra=[(b"location", b"http://127.0.0.1:1/steal")])
        token = "x" * 9000 if self.mode == "huge" else f"tok-{n}"
        self.valid = {token}
        await reply(
            200,
            {
                "access_token": token,
                "token_type": "bearer" if self.mode != "wrong_type" else "mac",
                "expires_in": self.expires_in,
            },
        )


@pytest.fixture
async def stack():
    import uvicorn

    from helpers.mcp_server import build as build_notes

    auth = AuthServer()
    notes, _ = build_notes()
    inner = notes.streamable_http_app()
    seen: list[dict[str, str]] = []

    async def gated(scope, receive, send):
        if scope["type"] == "lifespan":
            return await inner(scope, receive, send)
        headers = {k.decode(): v.decode() for k, v in scope["headers"]}
        seen.append(headers)
        bearer = headers.get("authorization", "")
        if not (bearer.startswith("Bearer ") and bearer[7:] in auth.valid):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
            return
        await inner(scope, receive, send)

    async def token_app(scope, receive, send):
        await auth.app(scope, receive, send)

    ports = free_port(), free_port()
    servers = [
        uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=p, log_level="error"))
        for app, p in ((gated, ports[0]), (token_app, ports[1]))
    ]
    tasks = [asyncio.create_task(s.serve()) for s in servers]
    while not all(s.started for s in servers):
        await asyncio.sleep(0.02)
    yield auth, seen, ports
    for s in servers:
        s.should_exit = True
    await asyncio.gather(*tasks)


def source(ports, **over):
    oauth = {
        "token_url": f"http://127.0.0.1:{ports[1]}/token",
        "token_hosts": ["127.0.0.1"],
        "client_id": "agent-1",
        "client_secret": "env:OAUTH_SECRET",
        "scope": "notes.read",
        **over.pop("oauth", {}),
    }
    return {
        "type": "mcp",
        "name": "web",
        "transport": "http",
        "url": f"http://127.0.0.1:{ports[0]}/mcp",
        "allowed_hosts": ["127.0.0.1"],
        "allow_insecure_loopback": True,
        "approval": "none",
        "connect_timeout_s": 10,
        "oauth": oauth,
        **over,
    }


async def build(src):
    return await acompile_spec(MCPSpec.model_validate({"sources": [src]}))


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv("OAUTH_SECRET", SECRET)


# ================================================================ the happy path
async def test_a_token_is_fetched_with_the_client_credentials_grant_and_sent_to_the_upstream_only(
    stack,
):
    auth, seen, ports = stack
    core = await build(source(ports))
    try:
        r = await core.call("search_notes", {"query": "x", "limit": 1})
        assert not r.is_error and "note about x" in r.text
        token_request = auth.requests[0]
        assert token_request["form"] == {"grant_type": "client_credentials", "scope": "notes.read"}
        basic = base64.b64decode(
            token_request["headers"]["authorization"].removeprefix("Basic ")
        ).decode()
        assert basic == f"agent-1:{SECRET}"  # the secret went to the token endpoint, in the header
        assert any(h.get("authorization") == "Bearer tok-1" for h in seen)
        assert not any(
            "basic" in h.get("authorization", "").lower() for h in seen
        )  # never to the upstream
        assert not any(
            "authorization" in str(rq["headers"]).lower()
            and "Bearer" in rq["headers"].get("authorization", "")
            for rq in auth.requests
        )
        dumped = (
            json.dumps(core.report)
            + json.dumps(core.spec.model_dump(mode="json"), default=str)
            + repr(core._upstreams)
        )
        assert SECRET not in dumped and "tok-1" not in dumped
    finally:
        await core.aclose()


async def test_post_style_client_authentication_puts_the_credentials_in_the_form(stack):
    auth, _seen, ports = stack
    core = await build(
        source(ports, oauth={"client_auth": "post", "audience": "https://api.example"})
    )
    try:
        form = auth.requests[0]["form"]
        assert (
            form["client_id"] == "agent-1"
            and form["client_secret"] == SECRET
            and form["audience"] == "https://api.example"
        )
        assert "authorization" not in auth.requests[0]["headers"]
    finally:
        await core.aclose()


async def test_the_token_is_cached_shared_and_refreshed_before_it_expires(stack, monkeypatch):
    auth, seen, ports = stack
    core = await build(source(ports))
    try:
        await asyncio.gather(
            *[core.call("search_notes", {"query": f"q{i}", "limit": 1}) for i in range(6)]
        )
        assert len(auth.requests) == 1  # one token served six concurrent calls
        clock = {"t": 0.0}
        real = mcp_oauth.time.monotonic
        monkeypatch.setattr(mcp_oauth.time, "monotonic", lambda: real() + clock["t"])
        clock["t"] = 3600 - 10  # inside the refresh margin
        r = await core.call("search_notes", {"query": "later", "limit": 1})
        assert (
            not r.is_error
            and len(auth.requests) == 2
            and seen[-1]["authorization"] == "Bearer tok-2"
        )
    finally:
        await core.aclose()


async def test_a_rejected_token_is_replaced_once_and_the_call_goes_through(stack):
    auth, seen, ports = stack
    core = await build(source(ports))
    try:
        auth.valid = set()  # the upstream stops accepting tok-1 (revoked, rotated)
        r = await core.call("search_notes", {"query": "again", "limit": 1})
        assert (
            not r.is_error
            and len(auth.requests) == 2
            and seen[-1]["authorization"] == "Bearer tok-2"
        )
    finally:
        await core.aclose()


# ================================================================ a misbehaving authorization server
@pytest.mark.parametrize("mode", ["status500", "not_json", "wrong_type", "huge", "redirect"])
async def test_an_unusable_answer_from_the_authorization_server_fails_generically(stack, mode):
    auth, seen, ports = stack
    auth.mode = mode
    with pytest.raises(SpecError) as e:
        await build(source(ports))
    assert SECRET not in str(e.value) and "could not obtain an access token" in str(e.value)
    assert seen == []  # the upstream was never contacted without a token
    core = await build(source(ports, optional=True))
    try:
        assert core.list_tools() == [] and any("not available" in n for n in core.report)
    finally:
        await core.aclose()


async def test_a_token_expiry_is_clamped(stack):
    auth, _seen, ports = stack
    for odd in (-5, 0, "soon", None, 10**12, True):
        auth.expires_in = odd
        tokens = mcp_oauth.OAuthTokens(
            MCPSpec.model_validate({"sources": [source(ports)]}).sources[0],
            MCPSpec.model_validate({"sources": [source(ports)]}).sources[0].oauth,
        )
        token, expires = await tokens._fetch()
        assert (
            token.startswith("tok-") and 29.9 <= expires - mcp_oauth.time.monotonic() <= 86_401
        ), odd


# ================================================================ configuration mistakes
async def test_mistakes_are_refused_before_any_request(stack, monkeypatch):
    auth, _seen, ports = stack
    cases = [
        ({"oauth": {"token_hosts": ["auth.example.com"]}}, "not in oauth.token_hosts"),
        (
            {"oauth": {"token_url": "http://127.0.0.1:1/t"}, "allow_insecure_loopback": False},
            "must be https",
        ),
        ({"oauth": {"token_url": "https://user:pw@127.0.0.1/t"}}, "no credentials"),
    ]
    for over, why in cases:
        with pytest.raises(SpecError, match=why):
            await build(source(ports, **over))
    monkeypatch.delenv("OAUTH_SECRET")
    with pytest.raises(SpecError, match="OAUTH_SECRET"):
        await build(source(ports))
    assert auth.requests == []
    for bad in (
        source(ports, oauth={"client_secret": SECRET}),  # the value itself
        source(ports, headers={"Authorization": "Bearer ${X}"}),  # two sources of the same header
        {
            "type": "mcp",
            "name": "s",
            "transport": "stdio",
            "command": "x",
            "oauth": source(ports)["oauth"],
        },
        source(ports, oauth={"token_hosts": []}),
        source(ports, oauth={"client_auth": "digest"}),
        source(ports, oauth={"surprise": 1}),
    ):
        with pytest.raises(ValueError):
            MCPSpec.model_validate({"sources": [bad]})
    assert "tok" not in repr(mcp_oauth.OAuthError("x"))
