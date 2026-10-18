"""OAuth 2.0 client credentials for an HTTP upstream: a bearer token fetched from an authorization server and refreshed before it expires.

Only the **client-credentials grant** (machine to machine) is supported: a headless server has nobody to send through a browser. What is enforced:

* `client_secret` (and, if you like, `client_id`) are `env:` references; the secret is sent only to the token endpoint, in the `Authorization: Basic` header (or the form
  body, `client_auth = "post"`, for servers that insist), never anywhere else;
* the token endpoint must be https (plain http only to loopback with `allow_insecure_loopback`), its host must be in `token_hosts`, and it is fetched through the
  SSRF-safe fetcher (public addresses only, connection pinned, no redirects);
* the token is sent **only** to the upstream's own host (every request of the upstream client goes to that host and nowhere else), held in memory, refreshed `SKEW_S`
  seconds before it expires, fetched by one caller at a time, and replaced once after a `401`;
* a token that is not a plain bearer token, or an authorization server that answers anything unexpected, is a generic failure with no detail from the server;
* the token and secret appear in no log line, error, report or `repr`."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any
from urllib.parse import urlencode, urlsplit

from aaa.mcp.config import Secret, SpecError
from aaa.mcp.spec import OAuthSpec
from aaa.tools.builtin.net import FetchBlocked, FetchPolicy, SafeFetcher

log = logging.getLogger("aaa.mcp.oauth")
SKEW_S = 30.0
MIN_TTL_S, MAX_TTL_S, DEFAULT_TTL_S = 30, 86_400, 300
LOOPBACK = ("127.0.0.1", "localhost", "::1")
_TOKEN = re.compile(r"^[A-Za-z0-9._~+/=-]{1,8192}$")


class OAuthError(Exception):
    """No token could be obtained. The message is generic on purpose: nothing from the authorization server's answer is repeated."""


def _secret(ref: str, what: str) -> Secret:
    return Secret.parse(ref, field=what)


class OAuthTokens:
    def __init__(
        self,
        name: str,
        spec: OAuthSpec,
        *,
        allow_private_networks: bool = False,
        allow_insecure_loopback: bool = False,
    ) -> None:
        self.name, self.spec = name, spec
        u = urlsplit(spec.token_url)
        host = (u.hostname or "").lower()
        if u.scheme not in ("https", "http") or not host or u.username or u.password:
            raise SpecError(
                f"mcp {self.name}: oauth.token_url must be an http(s) URL with a host and no credentials"
            )
        if u.scheme == "http" and not (allow_insecure_loopback and host in LOOPBACK):
            raise SpecError(
                f"mcp {self.name}: oauth.token_url must be https (plain http only to loopback with allow_insecure_loopback)"
            )
        if not any(
            host == h.lower() or (h.startswith("*.") and host.endswith(h[1:].lower()))
            for h in spec.token_hosts
        ):
            raise SpecError(
                f"mcp {self.name}: the token endpoint host {host!r} is not in oauth.token_hosts"
            )
        self._host = host
        self._id = (
            _secret(spec.client_id, f"mcp {self.name} oauth.client_id")
            if spec.client_id.startswith(("env:", "${"))
            else None
        )
        self._client_id = spec.client_id
        self._secret = _secret(spec.client_secret, f"mcp {self.name} oauth.client_secret")
        self._id_value()  # a missing variable is a startup error
        self._secret.value()
        self._fetcher = SafeFetcher(
            FetchPolicy(
                allowed_hosts=tuple(spec.token_hosts),
                allowed_ports=(u.port or (443 if u.scheme == "https" else 80),),
                allow_http=u.scheme == "http",
                max_redirects=0,
                max_bytes=64_000,
                timeout_s=spec.timeout_s,
                connect_timeout_s=min(spec.timeout_s, 10.0),
                content_types=("application/json",),
                allow_private_networks=allow_private_networks
                or (u.scheme == "http" and host in LOOPBACK),
            )
        )
        self._lock = asyncio.Lock()
        self._token: str | None = None
        self._expires = 0.0
        self.fetches = 0

    def __repr__(self) -> str:
        return f"OAuthTokens({self.name!r}, token={'held' if self._token else 'none'})"

    def _id_value(self) -> str:
        return self._id.value() if self._id is not None else self._client_id

    async def get(self, *, force: bool = False, stale: str | None = None) -> str:
        """A bearer token, fetching or refreshing when needed. With `force`, a fresh one, unless another caller already replaced `stale`."""
        async with self._lock:
            now = time.monotonic()
            if self._token is not None and not force and now < self._expires - SKEW_S:
                return self._token
            if force and stale is not None and self._token is not None and self._token != stale:
                return self._token  # somebody else refreshed while we waited for the lock
            self._token, self._expires = await self._fetch()
            return self._token

    async def _fetch(self) -> tuple[str, float]:
        form = {"grant_type": "client_credentials"}
        if self.spec.scope:
            form["scope"] = self.spec.scope
        if self.spec.audience:
            form["audience"] = self.spec.audience
        auth: dict[str, str] | None = None
        cid, secret = self._id_value(), self._secret.value()
        if self.spec.client_auth == "basic":
            raw = f"{cid}:{secret}".encode()
            auth = {"Authorization": "Basic " + base64.b64encode(raw).decode()}
        else:
            form |= {"client_id": cid, "client_secret": secret}
        self.fetches += 1
        try:
            r = await self._fetcher.request(
                "POST",
                self.spec.token_url,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                body=urlencode(form).encode(),
                auth=auth,
                auth_host=self._host if auth else None,
                same_host_redirects_only=True,
                raw=True,
            )
            if r.status != 200:
                raise OAuthError
            data = json.loads(r.text)
            token, kind = data.get("access_token"), str(data.get("token_type", "")).lower()
            if not isinstance(token, str) or not _TOKEN.match(token) or kind != "bearer":
                raise OAuthError
            ttl = data.get("expires_in", DEFAULT_TTL_S)
            ttl = (
                int(ttl)
                if isinstance(ttl, (int, float)) and not isinstance(ttl, bool)
                else DEFAULT_TTL_S
            )
        except OAuthError:
            raise OAuthError(
                "could not obtain an access token from the authorization server"
            ) from None
        except (FetchBlocked, ValueError, TypeError, AttributeError):
            log.warning("oauth %s: the token request failed", self.name)
            raise OAuthError(
                "could not obtain an access token from the authorization server"
            ) from None
        return token, time.monotonic() + max(MIN_TTL_S, min(MAX_TTL_S, ttl))


def bearer_auth(hx: Any, tokens: OAuthTokens) -> Any:
    """An httpx `Auth` of the given flavour (`httpx` or `httpx2`): attach the token; after a 401 get a new one once and try again."""

    class OAuthBearer(hx.Auth):
        requires_response_body = False

        async def async_auth_flow(self, request: Any) -> Any:
            try:
                token = await tokens.get()
            except OAuthError as e:
                raise hx.ConnectError(str(e)) from None
            request.headers["Authorization"] = f"Bearer {token}"
            response = yield request
            if response.status_code == 401:
                try:
                    token = await tokens.get(force=True, stale=token)
                except OAuthError as e:
                    raise hx.ConnectError(str(e)) from None
                request.headers["Authorization"] = f"Bearer {token}"
                yield request

    return OAuthBearer()
