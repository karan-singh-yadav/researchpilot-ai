"""Web-search providers (Brave, Tavily, offline fixture) behind one interface.

API usage verified against vendor docs (Brave: GET /res/v1/web/search + X-Subscription-Token;
Tavily: POST /search + Bearer token). Errors never include keys or response bodies.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel

from app.tools.urls import parse_datetime


class SearchError(Exception):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolNotConfiguredError(ValueError):
    pass


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str = ""
    published_at: datetime | None = None
    provider: str
    rank: int  # 1-based position in the provider's result list
    query: str
    retrieved_at: datetime


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[SearchHit]: ...


_TAG = re.compile(r"<[^>]+>")


def _clean(text: object, limit: int) -> str:
    return " ".join(_TAG.sub("", str(text or "")).split())[:limit]


def _check_status(r: httpx.Response) -> None:
    code = r.status_code
    if code == 429 or code >= 500:
        raise SearchError(f"search provider returned HTTP {code}", retryable=True)
    if code in (401, 403):
        raise SearchError("search provider authentication failed")
    if code >= 400:
        raise SearchError(f"search provider rejected the request (HTTP {code})")


def _send(client: httpx.Client, method: str, url: str, **kw) -> httpx.Response:
    try:
        r = client.request(method, url, **kw)
    except httpx.TimeoutException as exc:
        raise SearchError("search request timed out", retryable=True) from exc
    except httpx.HTTPError as exc:
        raise SearchError("search request failed (network error)", retryable=True) from exc
    _check_status(r)
    return r


class BraveSearch:
    name = "brave"
    URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, timeout: float = 15.0, client: httpx.Client | None = None):
        if not api_key:
            raise ToolNotConfiguredError("SEARCH_API_KEY must be set for the brave provider")
        self._key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        r = _send(
            self._client,
            "GET",
            self.URL,
            params={"q": query, "count": min(max_results, 20)},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._key,
                "Cache-Control": "no-cache",
            },
        )
        try:
            results = (r.json().get("web") or {}).get("results") or []
            now = datetime.now(UTC)
            hits = []
            for i, item in enumerate(results[:max_results], start=1):
                if not item.get("url"):
                    continue
                hits.append(
                    SearchHit(
                        title=_clean(item.get("title"), 300),
                        url=str(item["url"]),
                        snippet=_clean(item.get("description"), 1000),
                        published_at=parse_datetime(item.get("page_age")),
                        provider=self.name,
                        rank=i,
                        query=query,
                        retrieved_at=now,
                    )
                )
            return hits
        except (ValueError, AttributeError, TypeError) as exc:
            raise SearchError("search provider returned a malformed response") from exc


class TavilySearch:
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float = 20.0, client: httpx.Client | None = None):
        if not api_key:
            raise ToolNotConfiguredError("SEARCH_API_KEY must be set for the tavily provider")
        self._key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        r = _send(
            self._client,
            "POST",
            self.URL,
            json={
                "query": query,
                "max_results": min(max_results, 20),
                "search_depth": "basic",  # explicit: 'advanced' costs more credits
                "include_answer": False,
                "include_raw_content": False,
            },
            headers={"Authorization": f"Bearer {self._key}"},
        )
        try:
            results = r.json().get("results") or []
            now = datetime.now(UTC)
            hits = []
            for i, item in enumerate(results[:max_results], start=1):
                if not item.get("url"):
                    continue
                hits.append(
                    SearchHit(
                        title=_clean(item.get("title"), 300),
                        url=str(item["url"]),
                        snippet=_clean(item.get("content"), 1000),
                        published_at=parse_datetime(item.get("published_date")),
                        provider=self.name,
                        rank=i,
                        query=query,
                        retrieved_at=now,
                    )
                )
            return hits
        except (ValueError, AttributeError, TypeError) as exc:
            raise SearchError("search provider returned a malformed response") from exc


class FixtureSearch:
    """OFFLINE provider backed by a JSON file of synthetic results. NOT live web search."""

    name = "fixture"

    def __init__(self, data: dict) -> None:
        self._queries: dict = data.get("search", {})

    @classmethod
    def from_path(cls, path: str | Path) -> "FixtureSearch":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        q = query.lower()
        rows = next(
            (v for k, v in self._queries.items() if k != "default" and k.lower() in q), None
        )
        rows = rows if rows is not None else self._queries.get("default", [])
        now = datetime.now(UTC)
        return [
            SearchHit(
                title=row["title"],
                url=row["url"],
                snippet=row.get("snippet", ""),
                published_at=parse_datetime(row.get("published_at")),
                provider=self.name,
                rank=i,
                query=query,
                retrieved_at=now,
            )
            for i, row in enumerate(rows[:max_results], start=1)
        ]
