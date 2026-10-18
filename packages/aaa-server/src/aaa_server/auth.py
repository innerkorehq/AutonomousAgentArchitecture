"""Authentication: who is calling, and which tenant they belong to.

The tenant, user and scopes of a request come ONLY from the `Principal` an `Authenticator` returns. Nothing in
a request body, query string or header other than the credential is ever read as an identity."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from aaa.runtime.tenancy import valid_tenant

SCOPE_RUN = "run"  # start runs, read events, cancel
SCOPE_APPROVE = "approve"  # resolve tool approvals (a separate, rarer privilege)
SCOPE_ANSWER = "answer"  # answer the questions an `AskHuman` run is waiting on (separate from approving tool calls)
SCOPE_SESSION = "session"  # sessions and their messages
SCOPE_ADMIN = "admin"  # read the tenant's usage and quota
ALL_SCOPES = frozenset({SCOPE_RUN, SCOPE_APPROVE, SCOPE_ANSWER, SCOPE_SESSION, SCOPE_ADMIN})


class AuthError(Exception):
    """The credential is missing, malformed, unknown, expired or otherwise not acceptable.

    The message is for server-side logs; the HTTP response is always the same generic 401."""


@dataclass(frozen=True)
class Principal:
    tenant: str
    subject: str
    scopes: frozenset[str] = frozenset({SCOPE_RUN, SCOPE_SESSION})

    def __post_init__(self) -> None:
        valid_tenant(self.tenant)
        if not self.subject:
            raise ValueError("a principal needs a subject")

    def can(self, scope: str) -> bool:
        return scope in self.scopes


@runtime_checkable
class Authenticator(Protocol):
    async def authenticate(self, headers: Mapping[str, str]) -> Principal:
        """Return the caller or raise `AuthError`. `headers` keys are lower-case."""
        ...


def bearer_token(headers: Mapping[str, str]) -> str:
    raw = headers.get("authorization", "")
    scheme, _, token = raw.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("missing bearer token")
    return token.strip()


def hash_api_key(key: str) -> str:
    """SHA-256 hex of an API key. Keys are long random strings, so a fast hash is the right tool (there is
    nothing to brute-force); store this, never the key."""
    return hashlib.sha256(key.encode()).hexdigest()


PROFILE_SCOPE = re.compile(r"^profile:[a-z][a-z0-9_-]{0,31}$")


def _known(scopes: frozenset[str]) -> frozenset[str]:
    """The scopes of a token that this server understands: its own, and `profile:<name>` (which audience of an MCP server's tools the caller gets)."""
    return frozenset(s for s in scopes if s in ALL_SCOPES or PROFILE_SCOPE.match(s))


@dataclass(frozen=True)
class ApiKey:
    key_hash: str
    tenant: str
    subject: str
    scopes: frozenset[str] = frozenset({SCOPE_RUN, SCOPE_SESSION})


class ApiKeyAuthenticator:
    """Static API keys, held as hashes. Every candidate is compared in constant time, so the response time does
    not reveal how much of a key was right or which keys exist."""

    def __init__(self, keys: Sequence[ApiKey]) -> None:
        self._keys = tuple(keys)

    @classmethod
    def from_keys(
        cls,
        mapping: Mapping[str, tuple[str, str] | tuple[str, str, Sequence[str]]],
    ) -> ApiKeyAuthenticator:
        """`{"<the key>": ("tenant", "subject"[, scopes])}` for tests and small deployments. Prefer
        `ApiKey(key_hash=...)` entries loaded from a secret store."""
        out = []
        for key, spec in mapping.items():
            tenant, subject = spec[0], spec[1]
            scopes = frozenset(spec[2]) if len(spec) > 2 else frozenset({SCOPE_RUN, SCOPE_SESSION})  # type: ignore[misc]
            out.append(ApiKey(hash_api_key(key), tenant, subject, scopes))
        return cls(out)

    async def authenticate(self, headers: Mapping[str, str]) -> Principal:
        digest = hash_api_key(bearer_token(headers))
        found: ApiKey | None = None
        for k in self._keys:  # no early exit: every entry is compared
            if hmac.compare_digest(k.key_hash, digest):
                found = k
        if found is None:
            raise AuthError("unknown api key")
        return Principal(found.tenant, found.subject, found.scopes)


class JWKSKeys:
    """Signing keys fetched from a JWKS URL and cached; an unknown `kid` triggers one refetch (rotation),
    rate-limited so a stream of bogus tokens cannot make the server hammer the identity provider."""

    def __init__(
        self,
        url: str,
        *,
        client: httpx.AsyncClient | None = None,
        ttl_s: float = 600.0,
        min_refetch_s: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        import time

        self.url, self._client, self.ttl_s, self.min_refetch_s = url, client, ttl_s, min_refetch_s
        self._clock = clock or time.monotonic
        self._keys: dict[str, Any] = {}
        self._fetched = float("-inf")

    async def _fetch(self) -> None:
        from jwt import PyJWK

        client = self._client or httpx.AsyncClient(timeout=10.0, trust_env=False)
        try:
            r = await client.get(self.url)
            r.raise_for_status()
            doc = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise AuthError(f"jwks fetch failed: {type(e).__name__}") from None
        finally:
            if self._client is None:
                await client.aclose()
        keys: dict[str, Any] = {}
        for jwk in doc.get("keys", []):
            if jwk.get("use") not in (None, "sig") or "kid" not in jwk:
                continue
            try:
                keys[jwk["kid"]] = PyJWK(jwk).key
            except Exception:  # noqa: BLE001, S112 - an unusable key in the set must not break the others
                continue
        self._keys, self._fetched = keys, self._clock()

    async def key(self, kid: str) -> Any:
        now = self._clock()
        if (kid not in self._keys and now - self._fetched >= self.min_refetch_s) or (
            now - self._fetched >= self.ttl_s
        ):
            await self._fetch()
        try:
            return self._keys[kid]
        except KeyError:
            raise AuthError("unknown signing key") from None


@dataclass
class JWTAuthenticator:
    """Verifies a bearer JWT. Safe by construction:

    - the algorithm allow-list is fixed by YOU (asymmetric by default); the token's own `alg` header is never trusted,
      and `none` and the HS* family are refused unless you configure a shared `secret` (and then only HS*);
    - `exp`, `iss` and `aud` are required and checked;
    - the tenant, subject and scopes come from claims you name, and a token without a tenant claim is rejected."""

    issuer: str
    audience: str
    keys: JWKSKeys | None = None
    secret: str | None = None
    algorithms: tuple[str, ...] = ("RS256", "ES256", "EdDSA")
    tenant_claim: str = "tenant"
    scope_claim: str = "scope"
    leeway_s: float = 30.0
    default_scopes: frozenset[str] = field(
        default_factory=lambda: frozenset({SCOPE_RUN, SCOPE_SESSION})
    )

    def __post_init__(self) -> None:
        if (self.keys is None) == (self.secret is None):
            raise ValueError("give exactly one of `keys` (JWKS) or `secret` (HS256)")
        if self.secret is not None:
            self.algorithms = ("HS256",)
            if len(self.secret.encode()) < 32:
                raise ValueError("a shared JWT secret must be at least 32 bytes")
        elif any(a.lower() == "none" or a.upper().startswith("HS") for a in self.algorithms):
            raise ValueError("a JWKS-verified token must use an asymmetric algorithm")

    async def authenticate(self, headers: Mapping[str, str]) -> Principal:
        import jwt

        token = bearer_token(headers)
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") not in self.algorithms:
                raise AuthError("algorithm not allowed")
            if self.keys is not None:
                kid = header.get("kid")
                if not isinstance(kid, str):
                    raise AuthError("token has no kid")
                key: Any = await self.keys.key(kid)
            else:
                key = self.secret
            claims = jwt.decode(
                token,
                key,
                algorithms=list(self.algorithms),
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway_s,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except AuthError:
            raise
        except jwt.PyJWTError as e:
            raise AuthError(f"invalid token: {type(e).__name__}") from None
        tenant = claims.get(self.tenant_claim)
        if not isinstance(tenant, str):
            raise AuthError("token has no tenant")
        try:
            valid_tenant(tenant)
        except ValueError:
            raise AuthError("token tenant is invalid") from None
        raw = claims.get(self.scope_claim)
        if raw is None:
            scopes = self.default_scopes
        elif isinstance(raw, str):
            scopes = _known(frozenset(raw.split()))
        elif isinstance(raw, list):
            scopes = _known(frozenset(str(s) for s in raw))
        else:
            raise AuthError("token scope is malformed")
        return Principal(tenant, str(claims["sub"]), scopes)


__all__ = [
    "ALL_SCOPES",
    "SCOPE_ADMIN",
    "SCOPE_ANSWER",
    "SCOPE_APPROVE",
    "SCOPE_RUN",
    "SCOPE_SESSION",
    "ApiKey",
    "ApiKeyAuthenticator",
    "AuthError",
    "Authenticator",
    "JWKSKeys",
    "JWTAuthenticator",
    "Principal",
    "bearer_token",
    "hash_api_key",
]
