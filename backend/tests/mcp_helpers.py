import asyncio
import threading
import time
from datetime import UTC, datetime

from mcp import Client

from app.mcp.guard import ToolLimits
from app.mcp.handlers import ToolBackends
from app.mcp.server import build_server
from app.tools.fetcher import FetchResult, FixtureFetcher
from app.tools.search import FixtureSearch
from tests.helpers import FIXTURE


def fixture_backends(**kw) -> ToolBackends:
    limits = kw.pop("limits", ToolLimits())
    return ToolBackends(
        search=kw.pop("search", FixtureSearch.from_path(FIXTURE)),
        fetcher=kw.pop("fetcher", FixtureFetcher.from_path(FIXTURE)),
        limits=limits,
        **kw,
    )


def run(coro):
    return asyncio.run(coro)


def with_client(backends, coro_fn, enabled=None):
    """Run `coro_fn(client)` against a server over the in-memory MCP transport."""

    async def go():
        async with Client(build_server(backends, enabled)) as c:
            return await coro_fn(c)

    return run(go())


def err_text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


class SlowSearch:
    name = "slow"

    def __init__(self, delay=0.0, boom=None):
        self.delay, self.boom = delay, boom
        self.active = self.peak = 0
        self._lock = threading.Lock()

    def search(self, query, n):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.boom:
                raise self.boom
            time.sleep(self.delay)
            from app.tools.search import SearchHit

            return [
                SearchHit(
                    title="T", url="https://s.test/1", snippet="s", provider="slow", rank=1,
                    query=query, retrieved_at=datetime.now(UTC),
                )
            ]  # fmt: skip
        finally:
            with self._lock:
                self.active -= 1


class StubFetch:
    def __init__(self, html="", error=None, ctype="text/html", truncated=False):
        self.html, self.error, self.ctype, self.truncated = html, error, ctype, truncated

    def fetch(self, url):
        if self.error:
            raise self.error
        return FetchResult(
            url, url, 200, self.ctype, self.html, self.truncated, datetime.now(UTC), provider="stub"
        )
