import re
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_|igshid$|ref_src$)", re.I)


def normalize_url(url: str) -> str:
    """Canonical form used for display/storage: no fragment, no credentials, no tracking params."""
    p = urlsplit(url.strip())
    host = (p.hostname or "").lower()
    port = p.port
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    query = urlencode(
        [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING.match(k)]
    )
    path = p.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((p.scheme.lower(), netloc, path, query, ""))


def url_key(url: str) -> str:
    """Dedup key: like normalize_url but scheme-insensitive and ignoring a leading www."""
    p = urlsplit(normalize_url(url))
    host = p.netloc.removeprefix("www.")
    return f"{host}{p.path}" + (f"?{p.query}" if p.query else "")


_FORMATS = ("%Y/%m/%d", "%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%b %d, %Y")


def parse_datetime(value: object, now: datetime | None = None) -> datetime | None:
    """Parse an absolute date/time string to aware UTC; None if missing/invalid/implausible."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    dt: datetime | None = None
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        for fmt in _FORMATS:
            try:
                dt = datetime.strptime(v, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    now = now or datetime.now(UTC)
    if dt.year < 1990 or dt > now + timedelta(days=1):
        return None
    return dt
