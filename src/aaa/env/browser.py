"""A browser as an `Environment`, driven with Playwright (`pip install "autonomous-agent-architecture[browser]"`, then `playwright install chromium`).

The policy lives in code, not in the prompt:

* **Hosts and addresses.** All browser traffic goes through a local egress proxy (`aaa.env.egress`). For every connection it checks the host against `allowed_hosts`, resolves it **once**,
  requires every address to be public (unless `allow_private_networks`), and connects to that same address, so DNS rebinding cannot swap in a private one. Redirects, scripts,
  frames and WebSockets are held to the same rules because they are all connections through the proxy; only ports in `allowed_ports` (80 and 443, plus the start URL's) are reachable.
  `pin_dns=False` drops the proxy and falls back to checking each request (and redirect hop) in the page: weaker, with a rebinding gap, for setups that cannot use a local proxy.
* **A clean browser.** A fresh context for every environment: no cookies, no stored logins, downloads off, service workers blocked, extra windows closed, dialogs dismissed.
* **Actions.** `goto`, `scroll`, `back` and `read` have no side effects; `click` and `type` do, and run only through the approval tool (see `aaa.env.env_tools`).
  Typing into anything that looks like a **credential field** (a password input, or a field whose name, id, label or autocomplete suggests a password, token, code or card
  number) is **refused outright**, whatever the model says and whoever approves.
* **What it shows is data.** The observation is the page's text and a numbered list of its controls, and with `screenshots=True` a picture of the page (animations off, caret hidden, so an unchanged page gives identical bytes). The model is told it is untrusted."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
from collections.abc import Sequence
from typing import Any, Self
from urllib.parse import urljoin, urlparse

from aaa.env import ActionRefused, ActionSpec, Observation
from aaa.env.egress import EgressProxy, host_allowed, is_public, system_resolver

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
        pin_dns: bool = True,
        allowed_ports: Sequence[int] = (80, 443),
        screenshots: bool = False,
        screenshot_quality: int = 60,
        max_image_bytes: int = 400_000,
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
        self.pin_dns, self.screenshots = pin_dns, screenshots
        self.screenshot_quality, self.max_image_bytes = screenshot_quality, max_image_bytes
        ports = {int(x) for x in allowed_ports}
        if (start_port := urlparse(start_url).port) is not None:
            ports.add(start_port)
        self.allowed_ports: list[int] = sorted(ports)
        self._proxy: EgressProxy | None = None
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
            if self.pin_dns or self.allow_private_networks:
                return True  # with the egress proxy, the address check happens where the connection is made
            try:
                addresses = await system_resolver(p.hostname, p.port or 443)
            except OSError:
                return False
            return bool(addresses) and all(is_public(a) for a in addresses)
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
        if self.pin_dns or urlparse(url).scheme not in ("http", "https"):
            return await route.continue_()  # redirects, scripts and frames all go through the egress proxy, which holds them to the same rules
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
        launch: dict[str, Any] = {
            "headless": self.headless,
            "executable_path": self.executable_path,
        }
        if self.pin_dns:
            self._proxy = EgressProxy(
                self.allowed_hosts,
                allow_private_networks=self.allow_private_networks,
                allowed_ports=self.allowed_ports,
            )
            self._proxy.blocked = self.blocked
            await self._proxy.start()
            launch |= {
                "proxy": {"server": self._proxy.url},
                "args": ["--proxy-bypass-list=<-loopback>"],
            }
        self._browser = await self._pw.chromium.launch(**launch)
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
        if self._proxy is not None:
            await self._proxy.aclose()
        if self._pw is not None:
            await self._pw.stop()
        self._proxy = None
        self._pw = self._browser = self._ctx = self._page = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- observing and acting ----------------------------------------------------------------
    async def _read_page(self, page: Any) -> tuple[str, list[dict[str, Any]], str]:
        text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        els = await page.evaluate(_ELEMENTS_JS.replace("MAX", str(self.max_elements)))
        return text[: self.max_text], els, await page.title()

    async def _look(self) -> Observation:
        page = await self._start()
        for attempt in range(4):
            with contextlib.suppress(Exception):  # a slow page is still worth looking at
                await page.wait_for_load_state("domcontentloaded", timeout=self.timeout_ms)
            try:
                text, els, title = await self._read_page(page)
                break
            except Exception as e:
                if (
                    attempt == 3
                    or "navigation" not in str(e).lower()
                    and "context was destroyed" not in str(e).lower()
                ):
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
        lines = []
        for el in els:
            kind = el["tag"] if not el["type"] else f"{el['tag']}({el['type']})"
            link = f" -> {el['href']}" if el["href"] else ""
            lines.append(f'[{el["id"]}] {kind} "{el["text"]}"{link}')
        body = f"Title: {title}\nURL: {page.url}\n\nPage text:\n{text}\n\nControls:\n" + (
            "\n".join(lines) or "(none)"
        )
        image = None
        if self.screenshots:
            with contextlib.suppress(Exception):  # no picture is better than no observation
                raw = await page.screenshot(
                    type="jpeg",
                    quality=self.screenshot_quality,
                    animations="disabled",
                    caret="hide",
                )
                if len(raw) <= self.max_image_bytes:
                    image = base64.b64encode(raw).decode()
        return Observation(text=body, url=page.url, title=title, image=image)

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
