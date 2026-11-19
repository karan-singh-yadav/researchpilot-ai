"""SSRF-safe URL validation used by every outbound fetch (Reader agent / MCP fetch_webpage).

Blocks non-http(s) schemes, credentials in URLs, non-standard ports, localhost, and any host that
resolves to a private/loopback/link-local/reserved address. Callers must ALSO re-validate every
redirect hop and should connect to the validated IP to avoid DNS-rebinding (done in Phase 3).
"""

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}
BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}


class UnsafeURLError(ValueError):
    pass


def _is_public(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _default_resolver(host: str) -> list[str]:
    return list({info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)})


def validate_public_url(url: str, resolver: Callable[[str], list[str]] = _default_resolver) -> str:
    """Return the normalized URL if safe, else raise UnsafeURLError."""
    if not url or len(url) > 2048:
        raise UnsafeURLError("URL is empty or too long")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError("Only http and https URLs are allowed")
    if parsed.username or parsed.password:
        raise UnsafeURLError("Credentials in URLs are not allowed")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeURLError("URL has no host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURLError("Invalid port") from exc
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError("Non-standard ports are not allowed")
    if host in BLOCKED_HOSTNAMES or host.endswith((".localhost", ".internal", ".local")):
        raise UnsafeURLError("Host is not allowed")
    try:
        addresses = [host] if _is_ip_literal(host) else resolver(host)
    except OSError as exc:
        raise UnsafeURLError("Host could not be resolved") from exc
    if not addresses:
        raise UnsafeURLError("Host could not be resolved")
    for addr in addresses:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError as exc:
            raise UnsafeURLError("Resolved to an invalid address") from exc
        if not _is_public(ip):
            raise UnsafeURLError("Host resolves to a non-public address")
    return parsed.geturl()


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False
