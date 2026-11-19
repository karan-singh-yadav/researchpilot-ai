"""Machine-readable tool errors over MCP.

MCP tool failures are plain text. We prefix a stable code so clients can react without parsing
prose:   "[timeout:retryable] fetch_webpage exceeded 30s"   ->   code=timeout, retryable=True
Messages are written to be safe to show to a model/user: no paths, keys or stack traces.
"""

import re
from dataclasses import dataclass

_PARSE = re.compile(r"^\[([a-z_]+)(:retryable)?\]\s*(.*)$", re.S)
_SDK_PREFIX = re.compile(r"^Error executing tool \S+?:\s*")

KNOWN_CODES = {
    "blocked", "timeout", "network", "http_error", "too_many_redirects", "unsupported_type",
    "no_text", "search_failed", "not_configured", "not_available", "not_found", "too_large",
    "rate_limited", "invalid_arguments", "tool_disabled", "internal", "unavailable",
    "invalid_response", "permission_denied",
}  # fmt: skip


class ToolFailure(Exception):
    """An anticipated, safe-to-report tool failure (raised inside tool handlers)."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


def format_error(code: str, message: str, retryable: bool = False) -> str:
    return f"[{code}{':retryable' if retryable else ''}] {message}"


@dataclass(frozen=True)
class ParsedError:
    code: str
    message: str
    retryable: bool


def parse_error_text(text: str) -> ParsedError:
    text = _SDK_PREFIX.sub("", text.strip())  # the SDK wraps tool messages with this prefix
    m = _PARSE.match(text)
    if m:
        return ParsedError(m.group(1), m.group(3).strip(), bool(m.group(2)))
    low = text.lower()
    if "validation error" in low or "rejected arguments" in low or "unknown argument" in low:
        return ParsedError("invalid_arguments", text.strip()[:400], False)
    if "unknown tool" in low:
        return ParsedError("tool_disabled", text.strip()[:200], False)
    return ParsedError("internal", text.strip()[:200] or "tool failed", False)
