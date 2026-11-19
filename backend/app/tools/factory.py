from app.config import Settings
from app.tools.fetcher import FixtureFetcher, PageFetcher, SafeFetcher
from app.tools.search import (
    BraveSearch,
    FixtureSearch,
    SearchProvider,
    TavilySearch,
    ToolNotConfiguredError,
)


def build_search_provider(settings: Settings) -> SearchProvider:
    p = settings.search_provider.lower()
    if p == "brave":
        return BraveSearch(settings.search_api_key)
    if p == "tavily":
        return TavilySearch(settings.search_api_key)
    if p == "fixture":
        if not settings.search_fixture_path:
            raise ToolNotConfiguredError("SEARCH_FIXTURE_PATH must be set for the fixture provider")
        return FixtureSearch.from_path(settings.search_fixture_path)
    raise ToolNotConfiguredError(
        "No search provider configured. Set SEARCH_PROVIDER to brave, tavily or fixture."
    )


def build_fetcher(settings: Settings) -> PageFetcher:
    if settings.search_provider.lower() == "fixture":
        if not settings.search_fixture_path:
            raise ToolNotConfiguredError("SEARCH_FIXTURE_PATH must be set for the fixture provider")
        return FixtureFetcher.from_path(settings.search_fixture_path)
    return SafeFetcher(
        max_bytes=settings.max_fetch_bytes,
        timeout=settings.fetch_timeout_seconds,
        max_redirects=settings.fetch_max_redirects,
    )
