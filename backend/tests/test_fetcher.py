import gzip

import httpcore
import httpx
import pytest

from app.tools.fetcher import (
    FetchError,
    FixtureFetcher,
    SafeFetcher,
    SafeNetworkBackend,
)
from tests.helpers import FIXTURE

PUBLIC = lambda h: ["93.184.216.34"]  # noqa: E731


def fetcher(handler, **kw):
    kw.setdefault("resolver", PUBLIC)
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    return SafeFetcher(client=client, **kw)


def ok(body="<p>hello world</p>", ctype="text/html; charset=utf-8", status=200, headers=None):
    return httpx.Response(
        status, content=body.encode(), headers={"content-type": ctype, **(headers or {})}
    )


def test_success_returns_decoded_text_and_metadata():
    r = fetcher(lambda req: ok("<p>héllo</p>")).fetch("https://a.test/x")
    assert r.text == "<p>héllo</p>" and r.status_code == 200 and not r.truncated
    assert r.content_type == "text/html" and r.final_url == "https://a.test/x"


def test_redirect_followed_and_each_hop_validated():
    def handler(req):
        if req.url.path == "/a":
            return httpx.Response(302, headers={"location": "/b"})
        return ok("<p>final</p>")

    assert fetcher(handler).fetch("https://a.test/a").final_url == "https://a.test/b"

    def to_private(req):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    with pytest.raises(FetchError) as ei:
        fetcher(to_private).fetch("https://a.test/a")
    assert ei.value.code == "blocked"

    def to_internal_name(req):
        return httpx.Response(302, headers={"location": "https://internal.test/"})

    res = lambda h: ["10.0.0.9"] if h == "internal.test" else ["93.184.216.34"]  # noqa: E731
    with pytest.raises(FetchError) as ei:
        fetcher(to_internal_name, resolver=res).fetch("https://a.test/a")
    assert ei.value.code == "blocked"


def test_redirect_loop_is_bounded():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(302, headers={"location": "/again"})

    with pytest.raises(FetchError) as ei:
        fetcher(handler, max_redirects=3).fetch("https://a.test/x")
    assert ei.value.code == "too_many_redirects" and len(calls) == 4


@pytest.mark.parametrize("status,retryable", [(404, False), (403, False), (429, True), (503, True)])
def test_http_errors_classified(status, retryable):
    with pytest.raises(FetchError) as ei:
        fetcher(lambda r: httpx.Response(status)).fetch("https://a.test/x")
    assert ei.value.code == "http_error" and ei.value.retryable is retryable


def test_unsupported_content_type_rejected():
    with pytest.raises(FetchError) as ei:
        fetcher(lambda r: ok("%PDF-1.4", "application/pdf")).fetch("https://a.test/x")
    assert ei.value.code == "unsupported_type"


def test_size_cap_truncates():
    r = fetcher(lambda req: ok("x" * 10_000), max_bytes=1_000).fetch("https://a.test/x")
    assert r.truncated and len(r.text) == 1_000


def test_gzip_bomb_is_capped_on_decoded_stream():
    bomb = gzip.compress(b"a" * 50_000_000)  # ~50 KB on the wire, 50 MB decoded
    assert len(bomb) < 200_000
    h = lambda req: httpx.Response(  # noqa: E731
        200, content=bomb, headers={"content-type": "text/html", "content-encoding": "gzip"}
    )
    r = fetcher(h, max_bytes=100_000).fetch("https://a.test/x")
    assert r.truncated and len(r.text) == 100_000


def test_timeout_and_network_errors():
    def slow(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(FetchError) as ei:
        fetcher(slow).fetch("https://a.test/x")
    assert ei.value.code == "timeout" and ei.value.retryable

    def down(req):
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(FetchError) as ei:
        fetcher(down).fetch("https://a.test/x")
    assert ei.value.code == "network"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/",
        "http://127.0.0.1/",
        "http://u:p@a.test/",
        "https://a.test:8443/",
    ],
)
def test_unsafe_urls_never_reach_the_network(url):
    called = []
    with pytest.raises(FetchError) as ei:
        fetcher(lambda r: called.append(1) or ok()).fetch(url)
    assert ei.value.code == "blocked" and not called


# ---- connect-time pinning (DNS rebinding defence) ----
class RecordingBackend(httpcore.NetworkBackend):
    def __init__(self):
        self.connected = []

    def connect_tcp(self, host, port, **kw):
        self.connected.append((host, port))
        return object()

    def connect_unix_socket(self, path, **kw):  # pragma: no cover
        raise AssertionError

    def sleep(self, s):  # pragma: no cover
        pass


def test_backend_connects_to_validated_ip_not_hostname():
    inner = RecordingBackend()
    SafeNetworkBackend(PUBLIC, inner).connect_tcp("example.test", 443)
    assert inner.connected == [("93.184.216.34", 443)]


@pytest.mark.parametrize(
    "answers", [["127.0.0.1"], ["10.1.1.1"], ["93.184.216.34", "192.168.0.5"], ["::1"]]
)
def test_backend_blocks_any_non_public_answer(answers):
    inner = RecordingBackend()
    with pytest.raises(httpcore.ConnectError, match="blocked"):
        SafeNetworkBackend(lambda h: answers, inner).connect_tcp("evil.test", 80)
    assert inner.connected == []


def test_backend_blocks_unix_sockets():
    with pytest.raises(httpcore.ConnectError, match="blocked"):
        SafeNetworkBackend(PUBLIC, RecordingBackend()).connect_unix_socket("/var/run/docker.sock")


def test_dns_rebinding_blocked_with_real_transport():
    answers = iter(
        [["93.184.216.34"], ["127.0.0.1"]]
    )  # pre-check sees public, connect sees loopback
    f = SafeFetcher(resolver=lambda h: next(answers), timeout=2)
    with pytest.raises(FetchError) as ei:
        f.fetch("https://rebind.test/")
    assert ei.value.code == "blocked"


def test_real_client_ignores_proxy_env(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    answers = iter([["93.184.216.34"], ["127.0.0.1"]])
    f = SafeFetcher(resolver=lambda h: next(answers), timeout=2)
    with pytest.raises(FetchError) as ei:  # would be 'network' if it tried the proxy
        f.fetch("https://rebind.test/")
    assert ei.value.code == "blocked"


def test_fixture_fetcher():
    f = FixtureFetcher.from_path(FIXTURE)
    page = f.fetch("https://example.test/agentic-overview")
    assert page.provider == "fixture" and "Orchestra-X" in page.text
    for url, code in [
        ("https://example.test/slow", "timeout"),
        ("https://example.test/paper.pdf", "unsupported_type"),
        ("https://example.test/missing", "http_error"),
    ]:
        with pytest.raises(FetchError) as ei:
            f.fetch(url)
        assert ei.value.code == code
