"""Resource restrictions for tool execution: allow-list, rate limit, concurrency cap, timeouts."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from mcp.server.mcpserver.exceptions import ToolError

from app.mcp.errors import ToolFailure, format_error
from app.security.rate_limit import SlidingWindowLimiter


@dataclass
class ToolLimits:
    timeouts: dict[str, float] = field(default_factory=dict)
    default_timeout: float = 30.0
    max_concurrency: int = 8
    calls_per_minute: int = 240
    max_text_chars: int = 50_000
    max_export_bytes: int = 2_000_000

    def timeout_for(self, tool: str) -> float:
        return self.timeouts.get(tool, self.default_timeout)


class ToolGuard:
    """Wraps every tool call. Sync handlers run in worker threads so a slow tool never blocks the
    event loop; the wait is bounded by a per-tool timeout (the worker thread itself is bounded by
    the handler's own network timeouts)."""

    def __init__(self, limits: ToolLimits) -> None:
        self.limits = limits
        self._sem = asyncio.Semaphore(limits.max_concurrency)
        self._rate = SlidingWindowLimiter(60.0)

    async def run(self, tool: str, fn: Callable, *args):
        if not self._rate.allow("calls", self.limits.calls_per_minute):
            raise ToolError(format_error("rate_limited", "too many tool calls; slow down", True))
        timeout = self.limits.timeout_for(tool)
        async with self._sem:
            try:
                return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout)
            except TimeoutError:
                raise ToolError(
                    format_error("timeout", f"{tool} exceeded its {timeout:g}s time limit", True)
                ) from None
            except ToolFailure as f:
                raise ToolError(format_error(f.code, f.message, f.retryable)) from None
