"""Simple in-process sliding-window rate limiter keyed by client IP.

Limitation: state is per-process. Use a shared store (e.g. Redis) for multi-replica deployments.
"""

import time
from collections import defaultdict, deque

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.errors import AppError


class SlidingWindowLimiter:
    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        q = self._hits[key]
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True

    def reset(self) -> None:
        self._hits.clear()


limiter = SlidingWindowLimiter()


def rate_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client, settings.rate_limit_per_minute):
        raise AppError(429, "rate_limited", "Too many requests. Please slow down.")
