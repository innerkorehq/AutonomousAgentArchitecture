"""A small forward proxy that only lets a browser reach the hosts you allowed, and connects to the address it checked.

Checking a host name and then letting the browser look it up again leaves a gap: a name can answer with a public address for the check and a private one for the
connection (DNS rebinding). This proxy closes it by doing both steps itself, once: for every connection it resolves the name, requires **every** address to be
public (unless `allow_private_networks`), and connects to **that** address; the browser never connects anywhere else. Because a browser sends every request and every
redirect hop through it, the allow-list also holds for redirects, scripts, frames and WebSockets, which a page-level hook cannot promise.

It speaks just enough HTTP: `CONNECT host:port` (HTTPS is tunnelled, so the browser verifies certificates itself) and absolute-URI `http://` requests (one request per
connection). Only `allowed_ports` (default 80 and 443) are reachable. Refusals are `403`, unreachable hosts `502`; both are recorded in `blocked`."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Sequence
from typing import Self
from urllib.parse import urlsplit

MAX_HEAD = 16 * 1024
_HOP = {
    "proxy-connection",
    "proxy-authorization",
    "connection",
    "keep-alive",
    "te",
    "trailer",
    "upgrade",
}

Resolver = Callable[[str, int], Awaitable[list[str]]]


def host_allowed(host: str, patterns: Sequence[str]) -> bool:
    """`example.com` matches only that host; `*.example.com` matches its subdomains (not the bare domain)."""
    h = host.lower().rstrip(".")
    for p in patterns:
        p = p.lower()
        if p == h or (p.startswith("*.") and h.endswith(p[1:]) and len(h) > len(p) - 1):
            return True
    return False


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(i[4][0]) for i in infos]


def is_public(address: str) -> bool:
    try:
        return ipaddress.ip_address(address.split("%")[0]).is_global
    except ValueError:
        return False


class EgressProxy:
    """Start it, point the browser at `url`, close it when done (`async with` works)."""

    def __init__(
        self,
        allowed_hosts: Sequence[str],
        *,
        allow_private_networks: bool = False,
        allowed_ports: Sequence[int] = (80, 443),
        resolve: Resolver = system_resolver,
        connect_timeout_s: float = 10.0,
        head_timeout_s: float = 10.0,
        idle_timeout_s: float = 120.0,
        max_connections: int = 128,
    ) -> None:
        if not allowed_hosts:
            raise ValueError("allowed_hosts is required")
        self.allowed_hosts = [h.lower() for h in allowed_hosts]
        self.allow_private_networks, self.allowed_ports = allow_private_networks, set(allowed_ports)
        self.resolve, self.connect_timeout_s = resolve, connect_timeout_s
        self.head_timeout_s, self.idle_timeout_s = head_timeout_s, idle_timeout_s
        self.blocked: list[str] = []
        self._sem = asyncio.Semaphore(max_connections)
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self.port = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> Self:
        self._server = await asyncio.start_server(self._client, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def aclose(self) -> None:
        if self._server is None:
            return
        self._server.close()
        for t in list(
            self._tasks
        ):  # connections first: since Python 3.12 wait_closed() waits for them
            t.cancel()
        for t in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._server.wait_closed(), 5)
        self._server = None

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- one client connection ---------------------------------------------------------------
    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        try:
            async with self._sem:
                await self._serve(reader, writer)
        except (
            asyncio.CancelledError,
            ConnectionError,
            OSError,
            asyncio.IncompleteReadError,
            TimeoutError,
        ):
            pass
        finally:
            self._tasks.discard(task)
            with contextlib.suppress(Exception):
                writer.close()

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, status: int, reason: str) -> None:
        body = f"{status} {reason}\n".encode()
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        with contextlib.suppress(Exception):
            await writer.drain()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), self.head_timeout_s)
        except (asyncio.LimitOverrunError, asyncio.IncompleteReadError, TimeoutError):
            return await self._reply(writer, 400, "Bad Request")
        if len(head) > MAX_HEAD:
            return await self._reply(writer, 431, "Request Header Fields Too Large")
        lines = head.decode("latin-1").split("\r\n")
        try:
            method, target, _version = lines[0].split(" ", 2)
        except ValueError:
            return await self._reply(writer, 400, "Bad Request")
        if method.upper() == "CONNECT":
            host, _, port_s = target.rpartition(":")
            host = host.strip("[]")
            tunnel = True
            rewritten = b""
        else:
            parts = urlsplit(target)
            if parts.scheme != "http" or not parts.hostname:
                return await self._reply(writer, 403, "Forbidden")
            host, port_s, tunnel = parts.hostname, str(parts.port or 80), False
            path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
            kept = [
                ln for ln in lines[1:] if ln and ln.split(":", 1)[0].strip().lower() not in _HOP
            ]
            rewritten = (
                f"{method} {path} HTTP/1.1\r\n"
                + "\r\n".join(kept)
                + "\r\nConnection: close\r\n\r\n"
            ).encode("latin-1")
        try:
            port = int(port_s)
        except ValueError:
            return await self._reply(writer, 400, "Bad Request")
        address = await self._check(host, port)
        if address is None:
            self.blocked.append(host)
            return await self._reply(writer, 403, "Forbidden")
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(address, port), self.connect_timeout_s
            )
        except (OSError, TimeoutError):
            return await self._reply(writer, 502, "Bad Gateway")
        try:
            if tunnel:
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                up_writer.write(rewritten)
                await up_writer.drain()
            await asyncio.gather(self._pipe(reader, up_writer), self._pipe(up_reader, writer))
        finally:
            with contextlib.suppress(Exception):
                up_writer.close()

    async def _check(self, host: str, port: int) -> str | None:
        """The address to connect to, or None. Resolved once; the same address is used for the connection."""
        if port not in self.allowed_ports or not host_allowed(host, self.allowed_hosts):
            return None
        try:
            addresses = await asyncio.wait_for(self.resolve(host, port), self.connect_timeout_s)
        except (OSError, TimeoutError):
            return None
        if not addresses or (
            not self.allow_private_networks and not all(is_public(a) for a in addresses)
        ):
            return None
        return addresses[0]

    async def _pipe(self, src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
        try:
            while True:
                chunk = await asyncio.wait_for(src.read(65536), self.idle_timeout_s)
                if not chunk:
                    break
                dst.write(chunk)
                await dst.drain()
        except (ConnectionError, OSError, TimeoutError, asyncio.CancelledError):
            pass
        finally:
            with contextlib.suppress(Exception):
                if dst.can_write_eof():
                    dst.write_eof()


__all__ = ["EgressProxy", "host_allowed", "is_public", "system_resolver"]
