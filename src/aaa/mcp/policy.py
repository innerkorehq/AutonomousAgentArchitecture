"""Per-tool rules a server enforces at call time: visibility (scopes, tenants), a timeout, a rate limit and a concurrency cap.

Every refusal is deliberately plain. A tool the caller may not see answers exactly like a tool that does not exist; rate-limit and
concurrency refusals say only that the caller should retry, never how the limits are set."""

from __future__ import annotations

import fnmatch
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRule:
    timeout_s: float | None = None
    rate_per_minute: int | None = None
    burst: int | None = None
    max_concurrency: int | None = None
    scopes: frozenset[str] | None = None
    tenants: frozenset[str] | None = None

    def visible_to(self, tenant: str, scopes: frozenset[str] | None) -> bool:
        """`scopes=None` is a caller the host did not restrict. Otherwise the caller needs one of the tool's scopes."""
        if self.tenants is not None and tenant not in self.tenants:
            return False
        return not (self.scopes is not None and scopes is not None and not (self.scopes & scopes))


class Limiter:
    """Token buckets keyed by (caller, tool), bounded in memory (the least recently used keys are forgotten) and with an injectable clock."""

    def __init__(self, clock: Callable[[], float] | None = None, max_keys: int = 10_000) -> None:
        self.clock = clock or time.monotonic
        self.max_keys = max_keys
        self._buckets: OrderedDict[str, tuple[float, float]] = (
            OrderedDict()
        )  # key -> (tokens, last)

    def allow(self, key: str, per_minute: int, burst: int | None) -> bool:
        capacity = float(burst or min(per_minute, 10))
        rate = per_minute / 60.0
        now = self.clock()
        tokens, last = self._buckets.pop(key, (capacity, now))
        tokens = min(capacity, tokens + max(0.0, now - last) * rate)
        ok = tokens >= 1.0
        self._buckets[key] = (tokens - 1.0 if ok else tokens, now)
        while len(self._buckets) > self.max_keys:
            self._buckets.popitem(last=False)
        return ok


class Gates:
    """A cap on calls in flight per tool; `enter` refuses at once when the tool is at its cap instead of queueing."""

    def __init__(self) -> None:
        self._active: dict[str, int] = {}

    def enter(self, name: str, limit: int) -> bool:
        if self._active.get(name, 0) >= limit:
            return False
        self._active[name] = self._active.get(name, 0) + 1
        return True

    def leave(self, name: str) -> None:
        self._active[name] = max(0, self._active.get(name, 0) - 1)


PROFILE_PREFIX = "profile:"


@dataclass(frozen=True)
class Profile:
    tools: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()


class ProfileSet:
    """Which tools, resources and prompts a caller may see, by audience.

    The active profiles come only from what the host authenticated: the caller's `profile:<name>` scopes, or, for a caller the host did not restrict (stdio), `default`.
    Nothing a request carries can choose one. With no active profile the caller is unrestricted, unless `required`, in which case it sees nothing. A profile that lists no
    globs for a kind sees none of that kind."""

    def __init__(
        self, profiles: dict[str, Profile], *, default: str | None = None, required: bool = False
    ) -> None:
        self.profiles, self.default, self.required = dict(profiles), default, required

    def __bool__(self) -> bool:
        return bool(self.profiles)

    def active(self, scopes: frozenset[str] | None) -> frozenset[str] | None:
        """The profiles in force: None when the caller is unrestricted, an empty set when it sees nothing.

        A caller that holds `profile:` scopes is restricted to the ones that exist; if none of them do (a misspelt scope in a key's configuration) it sees nothing,
        rather than everything."""
        if not self.profiles:
            return None
        if scopes is None:
            wanted = {self.default} if self.default else set()
        else:
            wanted = {s[len(PROFILE_PREFIX) :] for s in scopes if s.startswith(PROFILE_PREFIX)}
        if wanted:
            return frozenset(wanted & set(self.profiles))
        return frozenset() if self.required else None

    def allows(self, kind: str, scopes: frozenset[str] | None, *candidates: str) -> bool:
        """Whether any of `candidates` (a tool name; a resource's URI and name) is allowed to this caller under `kind` ('tools', 'resources', 'prompts')."""
        active = self.active(scopes)
        if active is None:
            return True
        globs = [g for n in active for g in getattr(self.profiles[n], kind)]
        return any(fnmatch.fnmatchcase(c, g) for g in globs for c in candidates)
