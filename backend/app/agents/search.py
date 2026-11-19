"""Search agent: runs queries, retries transient failures, dedupes, ranks, caps to budget.

Ranking is a transparent HEURISTIC (provider rank + recency + query/snippet overlap). It does not
score domain reputation and is not evidence of correctness.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.tools.search import SearchError, SearchHit, SearchProvider
from app.tools.urls import url_key

_WORD = re.compile(r"[a-z0-9]{3,}")


@dataclass
class SearchOutcome:
    hits: list[SearchHit] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    queries_run: int = 0

    @property
    def failed_all(self) -> bool:
        return self.queries_run > 0 and not self.hits and len(self.errors) >= self.queries_run


def score_hit(hit: SearchHit, now: datetime) -> float:
    score = 1.0 / (1 + hit.rank)
    if hit.published_at:
        age_days = max((now - hit.published_at).days, 0)
        score += 0.5 * max(0.0, 1.0 - age_days / 730)  # linear decay over two years
    q = set(_WORD.findall(hit.query.lower()))
    if q:
        text = set(_WORD.findall(f"{hit.title} {hit.snippet}".lower()))
        score += 0.3 * len(q & text) / len(q)
    return score


class SearchAgent:
    def __init__(
        self,
        provider: SearchProvider,
        *,
        results_per_query: int = 5,
        retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.results_per_query = results_per_query
        self.retries = retries
        self._sleep = sleep

    def _search_one(self, query: str) -> list[SearchHit]:
        for attempt in range(self.retries + 1):
            try:
                return self.provider.search(query, self.results_per_query)
            except SearchError as exc:
                if not exc.retryable or attempt == self.retries:
                    raise
                self._sleep(min(1.0 * 2**attempt, 8.0))
        return []  # pragma: no cover

    def run(
        self,
        queries: list[str],
        *,
        max_sources: int,
        exclude_keys: frozenset[str] = frozenset(),
        now: datetime | None = None,
    ) -> SearchOutcome:
        now = now or datetime.now(UTC)
        out = SearchOutcome()
        best: dict[str, SearchHit] = {}
        for q in queries:
            out.queries_run += 1
            try:
                hits = self._search_one(q)
            except SearchError as exc:  # isolate: one bad query must not sink the rest
                out.errors.append(f"{q!r}: {exc}")
                continue
            for h in hits:
                if not h.url.lower().startswith(("http://", "https://")) or len(h.url) > 2048:
                    continue
                try:
                    key = url_key(h.url)
                except ValueError:
                    continue
                if key in exclude_keys:
                    continue
                if key not in best or score_hit(h, now) > score_hit(best[key], now):
                    best[key] = h
        ranked = sorted(best.values(), key=lambda h: -score_hit(h, now))
        out.hits = ranked[: max(0, max_sources)]
        return out
