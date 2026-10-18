"""The browser environment: its policy without a browser, and (where Playwright and Chromium are installed) against a real page on a local server."""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aaa.env import ActionRefused
from aaa.env.browser import BrowserEnvironment, host_allowed, looks_sensitive

PAGE = """<!doctype html><html><head><title>Shop</title></head><body>
<h1>Welcome</h1><p>Hello from the shop.</p>
<a href="/about">About us</a>
<a href="/redirect">Careful link</a>
<a href="http://localhost:{port}/secret">Other host</a>
<form onsubmit="document.title='submitted';return false">
  <input type="text" name="q" placeholder="Search">
  <input type="password" name="pw" placeholder="Password">
  <input type="text" name="card_number" placeholder="Number">
  <input type="text" name="field" autocomplete="one-time-code">
  <button type="submit">Go</button>
</form></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def do_GET(self):
        port = self.server.server_address[1]
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{port}/secret")
            self.end_headers()
            return
        body = {
            "/": PAGE.format(port=port),
            "/about": "<title>About</title><p>About page</p>",
            "/secret": "<title>S</title><p>SECRET</p>",
        }.get(self.path, "<p>404</p>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())


@pytest.fixture(scope="module")
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


# ===================================================================== policy, no browser needed
def test_host_patterns():
    assert host_allowed("docs.example.com", ["docs.example.com"]) and host_allowed(
        "DOCS.example.com.", ["docs.example.com"]
    )
    assert not host_allowed("evil.com", ["example.com"]) and not host_allowed(
        "example.com.evil.com", ["example.com"]
    )
    assert host_allowed("a.example.com", ["*.example.com"]) and not host_allowed(
        "example.com", ["*.example.com"]
    )
    assert not host_allowed("notexample.com", ["*.example.com"])


def test_credential_fields_are_recognised_by_type_name_label_and_autocomplete():
    assert looks_sensitive({"type": "password"})
    for key in ("name", "id", "autocomplete", "ariaLabel", "placeholder", "label"):
        assert (
            looks_sensitive({key: "Card number"})
            or looks_sensitive({key: "one-time-code"})
            or looks_sensitive({key: "password"})
        )
    assert (
        looks_sensitive({"name": "user_pwd"})
        and looks_sensitive({"autocomplete": "one-time-code"})
        and looks_sensitive({"label": "CVV"})
    )
    assert not looks_sensitive({"type": "text", "name": "q", "placeholder": "Search"})


def test_construction_requires_allowed_hosts_and_an_allowed_start_url():
    with pytest.raises(ValueError, match="allowed_hosts is required"):
        BrowserEnvironment([], "https://example.com/")
    with pytest.raises(ActionRefused):
        BrowserEnvironment(["example.com"], "https://other.com/")
    with pytest.raises(ActionRefused):
        BrowserEnvironment(["example.com"], "file:///etc/passwd")
    env = BrowserEnvironment(["example.com"], "https://example.com/")
    assert {n for n, s in env.actions.items() if s.side_effect} == {"click", "type"}


class FakeResponse:
    def __init__(self, status=200, location=None):
        self.status, self.headers = status, ({"location": location} if location else {})


class FakeRoute:
    """What the handler needs of a Playwright route; `chain` maps a url to its response."""

    def __init__(self, url, chain=None, navigation=True):
        self.url, self.chain, self.navigation = url, chain or {}, navigation
        self.request = self
        self.done = []
        self.fetched = []

    def is_navigation_request(self):
        return self.navigation

    async def fetch(self, url=None, max_redirects=20):
        assert max_redirects == 0  # a redirect must never be followed by the fetch itself
        self.fetched.append(url or self.url)
        return self.chain.get(url or self.url, FakeResponse())

    async def continue_(self):
        self.done.append("continue")

    async def abort(self, reason):
        self.done.append("abort")

    async def fulfill(self, response=None, **kw):
        self.done.append(
            ("fulfill", kw.get("status", getattr(response, "status", None)), kw.get("body", ""))
        )


async def test_every_request_is_checked_against_the_allow_list():
    env = BrowserEnvironment(
        ["example.com"], "https://example.com/", allow_private_networks=True, pin_dns=False
    )
    results = {}
    for url in (
        "https://example.com/a",
        "https://evil.com/x",
        "ftp://example.com/f",
        "file:///etc/passwd",
        "about:blank",
        "https://example.com.evil.com/",
    ):
        route = FakeRoute(url)
        await env._route(route)
        results[url] = route.done[0] if route.done else None
    assert results["https://example.com/a"][0] == "fulfill" and results["about:blank"] == "continue"
    assert [
        results[u]
        for u in (
            "https://evil.com/x",
            "ftp://example.com/f",
            "file:///etc/passwd",
            "https://example.com.evil.com/",
        )
    ] == ["abort"] * 4
    assert "evil.com" in env.blocked


async def test_a_redirect_to_a_host_that_is_not_allowed_is_never_requested():
    env = BrowserEnvironment(
        ["example.com"], "https://example.com/", allow_private_networks=True, pin_dns=False
    )
    start = "https://example.com/go"
    for navigation in (True, False):
        route = FakeRoute(start, {start: FakeResponse(302, "https://evil.com/steal")}, navigation)
        await env._route(route)
        assert route.done == ["abort"] and route.fetched == [start]  # evil.com was not fetched
    assert env.blocked.count("evil.com") == 2


async def test_an_allowed_redirect_is_followed_without_skipping_the_check():
    env = BrowserEnvironment(
        ["example.com"], "https://example.com/", allow_private_networks=True, pin_dns=False
    )
    start, middle = "https://example.com/a", "https://example.com/b"
    nav = FakeRoute(start, {start: FakeResponse(301, "/b")}, navigation=True)
    await env._route(nav)
    status, body = nav.done[0][1], nav.done[0][2]
    assert (
        status == 200 and "location.replace" in body and "https://example.com/b" in body
    )  # the browser comes back through the handler for /b
    sub = FakeRoute(
        start, {start: FakeResponse(302, "/b"), middle: FakeResponse(200)}, navigation=False
    )
    await env._route(sub)
    assert (
        sub.fetched == [start, middle] and sub.done[0][0] == "fulfill" and sub.done[0][1] == 200
    )  # a subresource redirect is followed here, hop by hop
    loop = FakeRoute(start, {start: FakeResponse(302, "/a")}, navigation=False)
    await env._route(loop)
    assert loop.done == ["abort"]  # a redirect loop is cut off
    evil = FakeRoute(
        start,
        {start: FakeResponse(302, "https://example.com/</script><script>alert(1)")},
        navigation=True,
    )
    await env._route(evil)
    assert (
        "</script><script>alert" not in evil.done[0][2]
    )  # the target cannot break out of the redirect page


# ===================================================================== a real browser
async def browser(site, **kw):
    pytest.importorskip("playwright")
    env = BrowserEnvironment(
        ["127.0.0.1"],
        site + "/",
        allow_private_networks=True,
        timeout_ms=5000,
        executable_path=os.environ.get("AAA_TEST_CHROMIUM"),
        **kw,
    )
    try:
        await env.reset()
    except Exception as e:
        await env.aclose()
        if os.environ.get("AAA_REQUIRE_BROWSER"):
            raise AssertionError(
                "a browser is required in this job but could not be started"
            ) from e
        pytest.skip(f"no usable browser: {type(e).__name__}: {str(e)[:300]}")
    return env


async def test_the_page_is_shown_as_text_and_numbered_controls(site):
    env = await browser(site)
    try:
        obs = await env.observe()
        assert "Hello from the shop." in obs.text and "Title: Shop" in obs.text
        assert (
            '[1] a "About us"' in obs.text
            and "input(password)" in obs.text
            and obs.url == site + "/"
        )
    finally:
        await env.aclose()


async def test_clicking_navigates_and_a_link_to_another_host_is_blocked(site):
    env = await browser(site)
    try:
        about = await env.act("click", {"element": 1})
        assert (
            "About page" in about.text
            and about.fingerprint != (await env.act("back", {})).fingerprint
        )
        await env.act("click", {"element": 3})  # a link to localhost, which is not allowed
        assert "SECRET" not in (await env.observe()).text and "localhost" in env.blocked
    finally:
        await env.aclose()


async def test_a_redirect_to_a_host_that_is_not_allowed_is_stopped(site):
    env = await browser(site)
    try:
        await env.act("click", {"element": 2})
        assert "SECRET" not in (await env.observe()).text and "localhost" in env.blocked
        with pytest.raises(ActionRefused):
            await env.act("goto", {"url": "https://example.org/"})
    finally:
        await env.aclose()


async def test_text_goes_into_ordinary_fields_and_never_into_credential_fields(site):
    env = await browser(site)
    try:
        obs = await env.observe()
        ids = {
            int(line.split("]")[0][1:]): line
            for line in obs.text.splitlines()
            if line.startswith("[")
        }

        def number(needle):
            return next(k for k, v in ids.items() if needle in v)

        done = await env.act(
            "type", {"element": number('"Search"'), "text": "shoes", "submit": True}
        )
        assert "Title: submitted" in done.text  # Enter was pressed in an ordinary field
        for needle in ("input(password)", '"Number"'):
            with pytest.raises(ActionRefused, match="credential"):
                await env.act("type", {"element": number(needle), "text": "4111111111111111"})
        with pytest.raises(ActionRefused, match="no control number"):
            await env.act("click", {"element": 999})
    finally:
        await env.aclose()


async def test_with_the_egress_proxy_the_route_hook_only_screens_and_the_proxy_holds_redirects():
    env = BrowserEnvironment(["example.com"], "https://example.com/")
    assert env.pin_dns and env.allowed_ports == [80, 443]
    ok, bad = FakeRoute("https://example.com/a"), FakeRoute("https://evil.com/x")
    await env._route(ok)
    await env._route(bad)
    assert (
        ok.done == ["continue"] and ok.fetched == []
    )  # nothing is fetched here: the proxy makes the connection, to the address it checked
    assert bad.done == ["abort"]
    assert BrowserEnvironment(["example.com"], "http://example.com:8080/").allowed_ports == [
        80,
        443,
        8080,
    ]


async def test_screenshots_are_taken_diffed_and_off_by_default(site):
    env = await browser(site, screenshots=True)
    try:
        first = await env.observe()
        again = await env.observe()
        assert (
            first.image and first.fingerprint == again.fingerprint
        )  # an unchanged page looks unchanged
        moved = await env.act("click", {"element": 1})
        assert moved.fingerprint != first.fingerprint
    finally:
        await env.aclose()
    plain = await browser(site)
    try:
        assert (await plain.observe()).image is None
    finally:
        await plain.aclose()
