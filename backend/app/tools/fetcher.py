"""SSRF-hardened page fetching.

Defences, in order:
  1. Every URL (and every redirect hop) passes validate_public_url().
  2. DNS is resolved AGAIN at connect time inside SafeNetworkBackend, every resolved address must be
     public, and the socket connects to that validated IP. This closes the DNS-rebinding window
     between "check" and "connect". TLS still verifies the certificate for the original hostname.
  3. Redirects are followed manually (bounded), proxies from the environment are ignored
     (trust_env=False) because a proxy would bypass IP validation.
  4. Response size is capped on the DECODED stream (gzip bombs), content types are allow-listed,
     and total wall-clock time is bounded.

Not implemented: robots.txt, cookies, JavaScript rendering.
"""

import ipaddress
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin

import httpcore
import httpx

from app.security.url_validation import (
    UnsafeURLError,
    _default_resolver,
    _is_public,
    validate_public_url,
)

ALLOWED_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
REDIRECT_CODES = {301, 302, 303, 307, 308}
USER_AGENT = "ResearchPilotBot/0.3 (research assistant; bounded fetches)"
Resolver = Callable[[str], list[str]]


class FetchError(Exception):
    """code: blocked | timeout | network | http_error | too_many_redirects | unsupported_type"""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    text: str
    truncated: bool
    retrieved_at: datetime
    provider: str = "http"
    # Set when the text is already cleaned (e.g. returned by the MCP fetch_webpage tool):
    extracted: bool = False
    title: str = ""
    published_at: datetime | None = None


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchResult: ...


class BlockedAddressError(httpcore.ConnectError):
    """Raised at connect time when a host resolves to a non-public address."""


class SafeNetworkBackend(httpcore.NetworkBackend):
    def __init__(
        self, resolver: Resolver = _default_resolver, inner: httpcore.NetworkBackend | None = None
    ) -> None:
        self._resolver = resolver
        self._inner = inner or httpcore.SyncBackend()

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        try:
            ipaddress.ip_address(host)
            addresses = [host]
        except ValueError:
            try:
                addresses = self._resolver(host)
            except OSError as exc:
                raise httpcore.ConnectError("host could not be resolved") from exc
        if not addresses:
            raise httpcore.ConnectError("host could not be resolved")
        for addr in addresses:  # ALL must be public, so a mixed answer cannot smuggle a private IP
            ip = ipaddress.ip_address(addr)
            if not _is_public(ip):
                raise BlockedAddressError("blocked: host resolves to a non-public address")
        last: Exception | None = None
        for addr in addresses:
            try:
                return self._inner.connect_tcp(
                    addr,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except httpcore.ConnectError as exc:
                last = exc
        raise last or httpcore.ConnectError("connection failed")

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise BlockedAddressError("blocked: unix sockets are not allowed")

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


def make_safe_client(timeout: float, resolver: Resolver = _default_resolver) -> httpx.Client:
    transport = httpx.HTTPTransport(retries=0)
    old = transport._pool
    transport._pool = httpcore.ConnectionPool(
        ssl_context=httpx.create_ssl_context(),
        max_connections=10,
        max_keepalive_connections=5,
        keepalive_expiry=5.0,
        http1=True,
        http2=False,
        network_backend=SafeNetworkBackend(resolver),
    )
    old.close()
    return httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(timeout),
        follow_redirects=False,
        trust_env=False,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
        },
    )


def _decode(raw: bytes, charset: str | None) -> str:
    try:
        return raw.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


class SafeFetcher:
    def __init__(
        self,
        *,
        max_bytes: int = 2_000_000,
        timeout: float = 10.0,
        max_redirects: int = 4,
        client: httpx.Client | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.deadline_seconds = timeout * 3
        self._resolver = resolver or _default_resolver
        self._client = client or make_safe_client(timeout, self._resolver)

    def fetch(self, url: str) -> FetchResult:
        current = url
        started = time.monotonic()
        for _ in range(self.max_redirects + 1):
            try:
                current = validate_public_url(current, resolver=self._resolver)
            except UnsafeURLError as exc:
                raise FetchError("blocked", str(exc)) from exc
            try:
                with self._client.stream("GET", current) as resp:
                    code = resp.status_code
                    if code in REDIRECT_CODES:
                        loc = resp.headers.get("location")
                        if not loc:
                            raise FetchError("http_error", f"HTTP {code} without Location")
                        current = urljoin(current, loc)
                        continue
                    if code >= 400:
                        raise FetchError(
                            "http_error",
                            f"HTTP {code}",
                            retryable=code in (429, 502, 503, 504),
                        )
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype not in ALLOWED_CONTENT_TYPES:
                        raise FetchError("unsupported_type", f"unsupported content type '{ctype}'")
                    buf = bytearray()
                    truncated = False
                    for chunk in resp.iter_bytes():  # decoded bytes: caps gzip bombs too
                        buf.extend(chunk)
                        if len(buf) > self.max_bytes:
                            del buf[self.max_bytes :]
                            truncated = True
                            break
                        if time.monotonic() - started > self.deadline_seconds:
                            raise FetchError(
                                "timeout", "fetch exceeded total time limit", retryable=True
                            )
                    return FetchResult(
                        url=url,
                        final_url=current,
                        status_code=code,
                        content_type=ctype,
                        text=_decode(bytes(buf), resp.charset_encoding),
                        truncated=truncated,
                        retrieved_at=datetime.now(UTC),
                    )
            except FetchError:
                raise
            except httpx.TimeoutException as exc:
                raise FetchError("timeout", "request timed out", retryable=True) from exc
            except httpx.HTTPError as exc:
                if isinstance(exc.__cause__, BlockedAddressError):
                    raise FetchError("blocked", str(exc.__cause__)) from exc
                raise FetchError("network", "network error", retryable=True) from exc
        raise FetchError("too_many_redirects", f"more than {self.max_redirects} redirects")


class FixtureFetcher:
    """OFFLINE fetcher serving pages from a fixture file. NOT live web access."""

    def __init__(self, data: dict) -> None:
        self._pages: dict = data.get("pages", {})

    @classmethod
    def from_path(cls, path: str | Path) -> "FixtureFetcher":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def fetch(self, url: str) -> FetchResult:
        from app.tools.urls import normalize_url

        page = self._pages.get(url) or self._pages.get(normalize_url(url))
        if page is None:
            raise FetchError("http_error", "HTTP 404")
        if "error" in page:
            raise FetchError(page["error"], f"simulated fetch failure: {page['error']}")
        ctype = page.get("content_type", "text/html")
        if ctype not in ALLOWED_CONTENT_TYPES:
            raise FetchError("unsupported_type", f"unsupported content type '{ctype}'")
        return FetchResult(
            url=url,
            final_url=url,
            status_code=200,
            content_type=ctype,
            text=page["html"],
            truncated=False,
            retrieved_at=datetime.now(UTC),
            provider="fixture",
        )
