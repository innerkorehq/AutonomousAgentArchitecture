"""The egress proxy: allow-list, one resolution per connection to the address that was checked, private addresses refused, redirects held to the same rules."""

import asyncio

import httpx
import pytest

from aaa.env.egress import EgressProxy, host_allowed, is_public


class Origin:
    """A tiny HTTP origin on 127.0.0.1 that records what it was asked."""

    def __init__(self):
        self.seen = []

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def handle(self, reader, writer):
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        first, *headers = head.split("\r\n")
        host = next(
            (h.split(":", 1)[1].strip() for h in headers if h.lower().startswith("host:")), ""
        )
        self.seen.append((first, host))
        if "/redirect" in first:
            body = b""
            writer.write(
                f"HTTP/1.1 302 Found\r\nLocation: http://evil.test:{self.port}/loot\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode()
            )
        else:
            body = b"hello"
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\n" + body
            )
        await writer.drain()
        writer.close()

    async def close(self):
        self.server.close()
        await self.server.wait_closed()


class Resolver:
    def __init__(self, table):
        self.table, self.calls = table, []

    async def __call__(self, host, port):
        self.calls.append(host)
        return self.table.get(host, [])


@pytest.fixture
async def origin():
    o = await Origin().start()
    yield o
    await o.close()


def client(proxy):
    return httpx.AsyncClient(proxy=proxy.url, timeout=5)


def test_host_patterns_and_public_addresses():
    assert host_allowed("a.example.com", ["*.example.com"]) and not host_allowed(
        "example.com", ["*.example.com"]
    )
    assert not host_allowed("example.com.evil.com", ["example.com"])
    assert is_public("93.184.216.34") and not any(
        is_public(a)
        for a in (
            "127.0.0.1",
            "10.1.2.3",
            "192.168.0.9",
            "169.254.169.254",
            "::1",
            "fe80::1",
            "not an ip",
        )
    )


async def test_an_allowed_host_is_reached_at_the_address_that_was_resolved_once(origin):
    resolver = Resolver({"shop.test": ["127.0.0.1"]})
    async with (
        EgressProxy(
            ["shop.test"],
            allow_private_networks=True,
            allowed_ports=[origin.port],
            resolve=resolver,
        ) as proxy,
        client(proxy) as c,
    ):
        r = await c.get(f"http://shop.test:{origin.port}/page?x=1")
    assert r.status_code == 200 and r.text == "hello"
    assert resolver.calls == ["shop.test"]  # looked up once, and the connection went to that answer
    assert origin.seen == [("GET /page?x=1 HTTP/1.1", f"shop.test:{origin.port}")]


async def test_https_style_tunnels_are_checked_the_same_way(origin):
    resolver = Resolver({"shop.test": ["127.0.0.1"], "evil.test": ["127.0.0.1"]})
    async with EgressProxy(
        ["shop.test"], allow_private_networks=True, allowed_ports=[origin.port], resolve=resolver
    ) as proxy:

        async def connect(host):
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
            writer.write(
                f"CONNECT {host}:{origin.port} HTTP/1.1\r\nHost: {host}:{origin.port}\r\n\r\n".encode()
            )
            await writer.drain()
            status = (await reader.readline()).decode()
            return status, reader, writer

        status, reader, writer = await connect("shop.test")
        assert " 200 " in status
        await reader.readline()  # the blank line that ends the proxy's reply
        writer.write(b"GET /t HTTP/1.1\r\nHost: shop.test\r\n\r\n")
        await writer.drain()
        assert b"hello" in await asyncio.wait_for(reader.read(), 5)
        writer.close()
        status, _, w2 = await connect("evil.test")
        assert " 403 " in status
        w2.close()
    assert proxy.blocked == ["evil.test"]


async def test_hosts_ports_and_private_addresses_that_are_not_allowed_are_refused(origin):
    resolver = Resolver(
        {
            "shop.test": ["127.0.0.1"],
            "mixed.test": ["93.184.216.34", "10.0.0.5"],
            "rebind.test": ["169.254.169.254"],
            "ok.test": ["93.184.216.34"],
        }
    )
    async with (
        EgressProxy(
            ["shop.test", "mixed.test", "rebind.test", "ok.test"],
            allowed_ports=[origin.port],
            resolve=resolver,
        ) as proxy,
        client(proxy) as c,
    ):
        for url in (
            f"http://shop.test:{origin.port}/",  # resolves to loopback and private addresses are not allowed here
            f"http://mixed.test:{origin.port}/",  # ONE private answer among public ones is enough to refuse
            f"http://rebind.test:{origin.port}/",  # a name that answers with the metadata address
            f"http://not-listed.test:{origin.port}/",
            f"http://ok.test:{origin.port + 1}/",  # a port that is not allowed
        ):
            assert (await c.get(url)).status_code == 403, url
    assert origin.seen == []  # nothing reached the origin
    assert (
        "not-listed.test" not in resolver.calls
    )  # a host that is not listed is not even looked up


async def test_a_redirect_to_another_host_is_stopped_at_the_proxy(origin):
    resolver = Resolver({"shop.test": ["127.0.0.1"], "evil.test": ["127.0.0.1"]})
    async with (
        EgressProxy(
            ["shop.test"],
            allow_private_networks=True,
            allowed_ports=[origin.port],
            resolve=resolver,
        ) as proxy,
        client(proxy) as c,
    ):
        r = await c.get(f"http://shop.test:{origin.port}/redirect", follow_redirects=True)
    assert r.status_code == 403 and proxy.blocked == ["evil.test"]
    assert [first for first, _ in origin.seen] == [
        "GET /redirect HTTP/1.1"
    ]  # /loot was never requested


async def test_garbage_oversized_and_non_http_requests_do_not_break_it(origin):
    async with EgressProxy(
        ["shop.test"],
        allow_private_networks=True,
        allowed_ports=[origin.port],
        resolve=Resolver({"shop.test": ["127.0.0.1"]}),
        head_timeout_s=0.3,
    ) as proxy:

        async def raw(data, wait=True):
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
            writer.write(data)
            await writer.drain()
            out = await asyncio.wait_for(reader.read(200), 5) if wait else b""
            writer.close()
            return out

        assert b" 400 " in await raw(b"garbage\r\n\r\n")
        assert b" 403 " in await raw(b"GET ftp://shop.test/x HTTP/1.1\r\n\r\n")
        assert b" 403 " in await raw(b"GET /relative HTTP/1.1\r\n\r\n")
        assert b" 400 " in await raw(b"CONNECT shop.test:notaport HTTP/1.1\r\n\r\n")
        assert b" 400 " in await raw(
            b"GET http://shop.test/ HTTP/1.1\r\n"
        )  # never finishes its head: timed out
        await raw(b"X" * 40_000, wait=False)
        async with client(proxy) as c:
            assert (
                await c.get(f"http://shop.test:{origin.port}/")
            ).status_code == 200  # still serving


def test_construction_needs_hosts():
    with pytest.raises(ValueError):
        EgressProxy([])
