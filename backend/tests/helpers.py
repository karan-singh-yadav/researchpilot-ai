import json

from app.services.llm.base import LLMError, LLMResponse, LLMUsage


class ScriptedLLM:
    """Returns queued replies in order (str -> reply text, Exception -> raised). Records calls."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self.replies:
            raise LLMError("script exhausted")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(r, LLMUsage(10, 5), self.model, self.name)


def task(
    id="t1", agent="search", queries=("agentic ai overview",), deps=(), desc="Survey the field"
):
    return {
        "id": id,
        "description": desc,
        "agent": agent,
        "search_queries": list(queries),
        "depends_on": list(deps),
    }


def plan_json(subtasks=None, **extra):
    body = {
        "objectives": ["Understand the topic"],
        "subtasks": subtasks if subtasks is not None else [task()],
        "report_outline": ["Executive summary", "Findings"],
    }
    body.update(extra)
    return json.dumps(body)


# ---------- Phase 3 helpers ----------
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from app.tools.fetcher import FetchError, FetchResult  # noqa: E402
from app.tools.search import SearchHit  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "evaluation/datasets/fixtures/demo_web.json"
NOW = datetime(2026, 10, 31, tzinfo=UTC)


def make_hit(url="https://a.test/x", title="T", rank=1, query="agentic ai", pub=None, snippet=""):
    return SearchHit(
        title=title, url=url, snippet=snippet, published_at=pub, provider="stub",
        rank=rank, query=query, retrieved_at=NOW,
    )  # fmt: skip


def html_page(*paragraphs, title="Page"):
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f"<html><head><title>{title}</title></head><body><article>{body}</article></body></html>"


class StubFetcher:
    """url -> html str | FetchError | Exception. Records calls."""

    def __init__(self, pages):
        self.pages = pages
        self.calls: list[str] = []

    def fetch(self, url):
        self.calls.append(url)
        v = self.pages[url]
        if isinstance(v, Exception):
            raise v
        return FetchResult(url, url, 200, "text/html", v, False, NOW, provider="stub")


def extraction_json(*passages, relevance=0.9):
    return json.dumps(
        {
            "items": [
                {"passage": p, "claim": p[:80], "kind": "fact", "relevance": relevance}
                for p in passages
            ]
        }
    )


__all__ = ["FetchError"]
