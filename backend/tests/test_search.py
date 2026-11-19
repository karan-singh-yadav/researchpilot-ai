import json

import httpx
import pytest

from app.config import Settings
from app.tools.factory import build_fetcher, build_search_provider
from app.tools.fetcher import FixtureFetcher
from app.tools.search import (
    BraveSearch,
    FixtureSearch,
    SearchError,
    TavilySearch,
    ToolNotConfiguredError,
)
from tests.helpers import FIXTURE

SECRET = "key-123-secret"


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_brave_request_shape_and_parsing():
    seen = {}

    def handler(req):
        seen.update(
            url=str(req.url),
            token=req.headers["x-subscription-token"],
            cc=req.headers["cache-control"],
        )
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "<b>A</b> title",
                            "url": "https://a.test/1",
                            "description": "x <strong>y</strong>",
                            "page_age": "2026-02-01T00:00:00",
                        },
                        {"title": "no url"},
                        {
                            "title": "B",
                            "url": "https://b.test/2",
                            "description": "d",
                            "age": "2 days ago",
                        },
                    ]
                }
            },
        )

    hits = BraveSearch(SECRET, client=client(handler)).search("agentic ai", 5)
    assert seen["token"] == SECRET and "q=agentic+ai" in seen["url"] and "count=5" in seen["url"]
    assert [h.url for h in hits] == ["https://a.test/1", "https://b.test/2"]
    assert (
        hits[0].title == "A title"
        and hits[0].snippet == "x y"
        and hits[0].published_at.year == 2026
    )
    assert hits[1].published_at is None and hits[0].provider == "brave" and hits[0].rank == 1


def test_tavily_request_shape_and_parsing():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T",
                        "url": "https://t.test/",
                        "content": "c",
                        "published_date": "2026-03-01",
                    }
                ]
            },
        )

    hits = TavilySearch(SECRET, client=client(handler)).search("q", 50)
    assert seen["auth"] == f"Bearer {SECRET}"
    assert seen["body"]["max_results"] == 20 and seen["body"]["search_depth"] == "basic"
    assert hits[0].published_at.month == 3 and hits[0].snippet == "c"


@pytest.mark.parametrize("cls", [BraveSearch, TavilySearch])
@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (401, False), (422, False)])
def test_error_classification_and_no_secret_leak(cls, status, retryable):
    p = cls(SECRET, client=client(lambda r: httpx.Response(status, text=f"echo {SECRET}")))
    with pytest.raises(SearchError) as ei:
        p.search("q", 3)
    assert ei.value.retryable is retryable and SECRET not in str(ei.value)


@pytest.mark.parametrize("cls", [BraveSearch, TavilySearch])
def test_malformed_and_timeout(cls):
    with pytest.raises(SearchError, match="malformed"):
        cls("k", client=client(lambda r: httpx.Response(200, json=["not", "a", "dict"]))).search(
            "q", 3
        )

    def boom(req):
        raise httpx.ConnectTimeout("t", request=req)

    with pytest.raises(SearchError) as ei:
        cls("k", client=client(boom)).search("q", 3)
    assert ei.value.retryable


def test_fixture_search_matching_and_default():
    fs = FixtureSearch(
        {
            "search": {
                "special topic": [{"title": "S", "url": "https://s.test/"}],
                "default": [{"title": "D", "url": "https://d.test/"}],
            }
        }
    )
    assert fs.search("About SPECIAL TOPIC today", 5)[0].url == "https://s.test/"
    assert fs.search("something else", 5)[0].url == "https://d.test/"
    assert fs.search("x", 5)[0].provider == "fixture"


def test_factory():
    with pytest.raises(ToolNotConfiguredError):
        build_search_provider(Settings(_env_file=None, search_provider="none"))
    with pytest.raises(ToolNotConfiguredError):
        build_search_provider(Settings(_env_file=None, search_provider="brave", search_api_key=""))
    with pytest.raises(ToolNotConfiguredError):
        build_search_provider(Settings(_env_file=None, search_provider="fixture"))
    s = Settings(_env_file=None, search_provider="fixture", search_fixture_path=str(FIXTURE))
    assert build_search_provider(s).name == "fixture" and isinstance(
        build_fetcher(s), FixtureFetcher
    )
    assert (
        build_search_provider(
            Settings(_env_file=None, search_provider="tavily", search_api_key="k")
        ).name
        == "tavily"
    )
