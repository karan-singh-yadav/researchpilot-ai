import pytest

from app.config import get_settings
from app.security.rate_limit import SlidingWindowLimiter
from app.security.url_validation import UnsafeURLError, validate_public_url

PUBLIC = lambda host: ["93.184.216.34"]  # noqa: E731


def test_accepts_public_https():
    assert validate_public_url("https://example.com/a?b=1", resolver=PUBLIC).startswith("https://")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "http://user:pw@example.com/",
        "http://example.com:8080/",
        "http://localhost/",
        "http://foo.localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
        "",
    ],
)
def test_blocks_unsafe_urls(url):
    with pytest.raises(UnsafeURLError):
        validate_public_url(url, resolver=PUBLIC)


def test_blocks_hostname_resolving_to_private():
    with pytest.raises(UnsafeURLError):
        validate_public_url("https://evil.example/", resolver=lambda h: ["10.1.2.3"])


def test_blocks_if_any_resolved_address_is_private():
    with pytest.raises(UnsafeURLError):
        validate_public_url(
            "https://rebind.example/", resolver=lambda h: ["93.184.216.34", "127.0.0.1"]
        )


def test_rate_limiter_window():
    lim = SlidingWindowLimiter(window_seconds=10)
    assert all(lim.allow("ip", 3, now=t) for t in (0, 1, 2))
    assert not lim.allow("ip", 3, now=3)
    assert lim.allow("ip", 3, now=11)  # oldest hit expired


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-one,secret-two")
    get_settings.cache_clear()
    assert client.get("/api/research").status_code == 401
    assert client.get("/api/research", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/research", headers={"X-API-Key": "secret-two"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays public


def test_rate_limit_http(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    get_settings.cache_clear()
    assert client.get("/api/research").status_code == 200
    assert client.get("/api/research").status_code == 200
    r = client.get("/api/research")
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
