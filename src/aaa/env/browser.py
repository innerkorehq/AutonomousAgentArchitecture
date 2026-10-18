"""A browser as an `Environment`, driven with Playwright (`pip install "autonomous-agent-architecture[browser]"`, then `playwright install chromium`).

The policy lives in code, not in the prompt:

* **Hosts.** Every request the page makes (the page itself, every redirect hop, scripts, images, frames) is checked against `allowed_hosts` and **aborted** if the host is not
  listed; only `http` and `https` (and `about:blank`) are allowed. Unless `allow_private_networks`, a host that resolves to a private, loopback or link-local address is
  refused as well. (Checking at request time still leaves a DNS-rebinding gap between this check and the browser's own lookup; do not allowlist hosts you do not control.)
* **A clean browser.** A fresh context for every environment: no cookies, no stored logins, downloads off, service workers blocked, extra windows closed, dialogs dismissed.
* **Actions.** `goto`, `scroll`, `back` and `read` have no side effects; `click` and `type` do, and run only through the approval tool (see `aaa.env.env_tools`).
  Typing into anything that looks like a **credential field** (a password input, or a field whose name, id, label or autocomplete suggests a password, token, code or card
  number) is **refused outright**, whatever the model says and whoever approves.
* **What it shows is data.** The observation is the page's text and a numbered list of its controls. The model is told it is untrusted."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import re
import socket
from collections.abc import Sequence
from typing import Any, Self
from urllib.parse import urljoin, urlparse

from aaa.env import ActionRefused, ActionSpec, Observation

MAX_HOPS = 6
_SENSITIVE = re.compile(
    r"pass(word|wd)?|pwd|secret|token|otp|one-time|verification|cvv|cvc|card|ssn|iban|pin\b",
    re.IGNORECASE,
)
_ELEMENTS_JS = """
() => {
  const sel = 'a[href], button, input, select, textarea, [role=button]';
  const out = []; let n = 0;
  document.querySelectorAll(sel).forEach(el => {
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    if (r.width === 0 || r.height === 0 || st.visibility === 'hidden' || st.display === 'none') return;
    n += 1; el.setAttribute('data-aaa-id', String(n));
    const label = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || '').trim().slice(0, 80);
    out.push({id: n, tag: el.tagName.toLowerCase(), type: el.getAttribute('type') || '', text: label, href: el.tagName === 'A' ? el.href : ''});
  });
  return out.slice(0, MAX);
}
"""


def host_allowed(host: str, patterns: Sequence[str]) -> bool:
    """`example.com` matches only that host; `*.example.com` matches its subdomains (not the bare domain)."""
    h = host.lower().rstrip(".")
    for p in patterns:
        p = p.lower()
        if p == h or (p.startswith("*.") and h.endswith(p[1:]) and len(h) > len(p) - 1):
            return True
    return False


async def _is_public(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
        return ip.is_global
    except ValueError:
        pass
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def looks_sensitive(attrs: dict[str, Any]) -> bool:
    """Whether a form field looks like it takes a credential (checked on the field's type and its name, id, label, placeholder and autocomplete)."""
    if str(attrs.get("type", "")).lower() == "password":
        return True
    return any(
        _SENSITIVE.search(str(attrs.get(k, "")))
        for k in ("name", "id", "autocomplete", "ariaLabel", "placeholder", "label")
    )


class BrowserEnvironment:
    """One browser page, restricted to `allowed_hosts`. See the module docstring for the guarantees.

    `max_elements` caps the controls listed to the model; `timeout_ms` bounds every page operation. Use `async with` or call `aclose()`."""

    def __init__(
        self,
        allowed_hosts: Sequence[str],
        start_url: str,
        *,
        headless: bool = True,
        allow_private_networks: bool = False,
        max_elements: int = 40,
        max_text_chars: int = 4000,
        timeout_ms: int = 10_000,
        executable_path: str | None = None,
    ) -> None:
        if not allowed_hosts:
            raise ValueError("allowed_hosts is required: name the hosts the browser may reach")
        self.allowed_hosts = [h.lower() for h in allowed_hosts]
        self._check_url(start_url)
        self.start_url, self.headless = start_url, headless
        self.allow_private_networks, self.max_elements = allow_private_networks, max_elements
        self.max_text, self.timeout_ms = max_text_chars, timeout_ms
        self.executable_path = (
            executable_path  # a specific Chromium/Chrome binary instead of Playwright's own
        )
        self.blocked: list[
            str
        ] = []  # hosts the page tried to reach and was refused, for the operator
        self._pw: Any = None
        self._browser: Any = None
        self._ctx: Any = None
        self._page: Any = None
        self.actions = {
            "goto": ActionSpec(
                description="Open a web address.",
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                side_effect=False,
            ),
            "scroll": ActionSpec(
                description="Scroll the page.",
                parameters={
                    "type": "object",
                    "properties": {"direction": {"type": "string", "enum": ["up", "down"]}},
                    "required": ["direction"],
                    "additionalProperties": False,
                },
                side_effect=False,
            ),
            "back": ActionSpec(description="Go back one page.", side_effect=False),
            "read": ActionSpec(description="Read the page again.", side_effect=False),
            "click": ActionSpec(
                description="Click a control by its number.",
                parameters={
                    "type": "object",
                    "properties": {"element": {"type": "integer"}},
                    "required": ["element"],
                    "additionalProperties": False,
                },
            ),
            "type": ActionSpec(
                description="Type text into a control by its number (never into password or other credential fields). Set submit to press Enter afterwards.",
                parameters={
                    "type": "object",
                    "properties": {
                        "element": {"type": "integer"},
                        "text": {"type": "string"},
                        "submit": {"type": "boolean"},
                    },
                    "required": ["element", "text"],
                    "additionalProperties": False,
                },
            ),
        }

    # -- policy ------------------------------------------------------------------------------
    def _check_url(self, url: str) -> None:
        p = urlparse(url)
        if (
            p.scheme not in ("http", "https")
            or not p.hostname
            or not host_allowed(p.hostname, self.allowed_hosts)
        ):
            raise ActionRefused(f"{p.hostname or url!r} is not an allowed address")

    async def _permitted(self, url: str) -> bool:
        p = urlparse(url)
        if url == "about:blank" or p.scheme in ("data", "blob"):
            return True
        if (
            p.scheme in ("http", "https")
            and p.hostname
            and host_allowed(p.hostname, self.allowed_hosts)
        ):
            return self.allow_private_networks or await _is_public(p.hostname)
        return False

    async def _refuse(self, route: Any, url: str) -> None:
        self.blocked.append(urlparse(url).hostname or urlparse(url).scheme)
        await route.abort("blockedbyclient")

    async def _route(self, route: Any) -> None:
        """Check a request, and every redirect hop of it, before anything is sent to that host.

        A request that is simply continued would be redirected inside the browser without coming back here, so requests are made here without following redirects:
        a navigation that is redirected is answered with a tiny page that moves the browser on to the (checked) target, which then arrives here as a new request; the
        redirect of any other request (an image, a script) is followed here, hop by hop, with each hop checked."""
        url = route.request.url
        if not await self._permitted(url):
            return await self._refuse(route, url)
        if urlparse(url).scheme not in ("http", "https"):
            return await route.continue_()
        response = await route.fetch(max_redirects=0)
        for _ in range(MAX_HOPS):
            location = response.headers.get("location")
            if not (300 <= response.status < 400 and location):
                return await route.fulfill(response=response)
            target = urljoin(url, location)
            if not await self._permitted(target):
                return await self._refuse(route, target)
            if route.request.is_navigation_request():
                page = f"<!doctype html><script>location.replace({json.dumps(target).replace('<', chr(92) + 'u003c')})</script>"
                return await route.fulfill(status=200, content_type="text/html", body=page)
            url = target
            response = await route.fetch(url=target, max_redirects=0)
        await self._refuse(route, url)  # too many hops

    # -- lifecycle ---------------------------------------------------------------------------
    async def _start(self) -> Any:
        if self._page is not None:
            return self._page
        try:
            from playwright.async_api import async_playwright  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise RuntimeError(
                'BrowserEnvironment needs Playwright: pip install "autonomous-agent-architecture[browser]" && playwright install chromium'
            ) from None
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self.headless, executable_path=self.executable_path
        )
        self._ctx = await self._browser.new_context(
            accept_downloads=False, service_workers="block", viewport={"width": 1100, "height": 800}
        )
        self._ctx.set_default_timeout(self.timeout_ms)
        await self._ctx.route("**/*", self._route)
        self._page = await self._ctx.new_page()
        self._page.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))
        self._ctx.on(
            "page", lambda pg: asyncio.ensure_future(pg.close()) if pg is not self._page else None
        )
        return self._page

    async def aclose(self) -> None:
        for closer in (self._ctx, self._browser):
            if closer is not None:
                with contextlib.suppress(Exception):  # closing a browser that already died is fine
                    await closer.close()
        if self._pw is not None:
            await self._pw.stop()
        self._pw = self._browser = self._ctx = self._page = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- observing and acting ----------------------------------------------------------------
    async def _look(self) -> Observation:
        page = await self._start()
        with contextlib.suppress(Exception):  # a slow page is still worth looking at
            await page.wait_for_load_state("domcontentloaded", timeout=self.timeout_ms)
        text = (await page.evaluate("() => document.body ? document.body.innerText : ''"))[
            : self.max_text
        ]
        els = await page.evaluate(_ELEMENTS_JS.replace("MAX", str(self.max_elements)))
        lines = []
        for el in els:
            kind = el["tag"] if not el["type"] else f"{el['tag']}({el['type']})"
            link = f" -> {el['href']}" if el["href"] else ""
            lines.append(f'[{el["id"]}] {kind} "{el["text"]}"{link}')
        title = await page.title()
        body = f"Title: {title}\nURL: {page.url}\n\nPage text:\n{text}\n\nControls:\n" + (
            "\n".join(lines) or "(none)"
        )
        return Observation(text=body, url=page.url, title=title)

    async def reset(self) -> Observation:
        page = await self._start()
        await page.goto(self.start_url, timeout=self.timeout_ms)
        return await self._look()

    async def observe(self) -> Observation:
        await self._start()
        return await self._look()

    async def _element(self, number: int) -> Any:
        page = await self._start()
        loc = page.locator(f'[data-aaa-id="{int(number)}"]')
        if await loc.count() != 1:
            raise ActionRefused(f"there is no control number {number} on this page")
        return loc

    async def act(self, action: str, arguments: dict[str, Any]) -> Observation:
        page = await self._start()
        if action == "goto":
            self._check_url(str(arguments["url"]))
            await page.goto(str(arguments["url"]), timeout=self.timeout_ms)
        elif action == "scroll":
            await page.mouse.wheel(0, -600 if arguments["direction"] == "up" else 600)
        elif action == "back":
            await page.go_back(timeout=self.timeout_ms)
        elif action == "read":
            pass
        elif action == "click":
            await (await self._element(arguments["element"])).click(timeout=self.timeout_ms)
        elif action == "type":
            loc = await self._element(arguments["element"])
            attrs = await loc.evaluate(
                "el => ({type: el.getAttribute('type') || '', name: el.getAttribute('name') || '', id: el.id || '', autocomplete: el.getAttribute('autocomplete') || '',"
                " ariaLabel: el.getAttribute('aria-label') || '', placeholder: el.getAttribute('placeholder') || '',"
                " label: (el.labels && el.labels[0] ? el.labels[0].innerText : '')})"
            )
            if looks_sensitive(attrs):
                raise ActionRefused(
                    "typing into a password or other credential field is not allowed"
                )
            await loc.fill(str(arguments["text"])[:2000], timeout=self.timeout_ms)
            if arguments.get("submit"):
                await loc.press("Enter", timeout=self.timeout_ms)
        else:
            raise ActionRefused(f"unknown action {action!r}")
        return await self._look()


__all__ = ["BrowserEnvironment", "host_allowed", "looks_sensitive"]
