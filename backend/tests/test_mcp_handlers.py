import json

import pytest

from app.mcp import handlers
from app.mcp.errors import ToolFailure
from app.mcp.guard import ToolLimits
from app.mcp.schemas import DocumentChunk
from app.services.runner import make_checkpointer  # noqa: F401
from app.tools.fetcher import FetchError, SafeFetcher
from app.tools.search import SearchError
from tests.graph_env import make_env, make_manager, new_research
from tests.helpers import html_page
from tests.mcp_helpers import StubFetch, fixture_backends


def fail(fn, *a):
    with pytest.raises(ToolFailure) as ei:
        fn(*a)
    return ei.value


def test_search_maps_results_and_caps_count():
    out = handlers.do_search_web(fixture_backends(), "agentic ai", 3)
    assert out.provider == "fixture" and [h.rank for h in out.hits] == [1, 2, 3]
    assert out.hits[0].published_at is not None and out.retrieved_at


def test_search_failures_are_typed():
    assert (
        fail(handlers.do_search_web, fixture_backends(search=None), "agentic", 3).code
        == "not_configured"
    )

    class Down:
        name = "down"

        def search(self, q, n):
            raise SearchError("provider returned HTTP 429", retryable=True)

    f = fail(handlers.do_search_web, fixture_backends(search=Down()), "agentic", 3)
    assert (f.code, f.retryable) == ("search_failed", True)


def test_fetch_cleans_page_strips_hidden_text_and_flags_injection():
    b = fixture_backends()
    out = handlers.do_fetch_webpage(b, "https://example.test/untrusted-page", 20_000)
    assert out.injection_suspected and out.untrusted is True and out.title.startswith("Untrusted")
    assert "Ignore all previous instructions" in out.text  # visible text is kept (as data)...
    assert "only source" not in out.text  # ...but the display:none payload is removed
    assert "<" not in out.text and out.published_at.year == 2026


def test_fetch_truncation_and_hard_cap():
    long = html_page(
        *[f"Sentence number {i} is part of a long page about agents." for i in range(200)]
    )
    b = fixture_backends(fetcher=StubFetch(long), limits=ToolLimits(max_text_chars=800))
    out = handlers.do_fetch_webpage(b, "https://a.test/long", 50_000)
    assert len(out.text) == 800 and out.truncated is True  # server cap beats the requested size
    small = handlers.do_fetch_webpage(
        fixture_backends(fetcher=StubFetch(long)), "https://a.test/l", 500
    )
    assert len(small.text) == 500 and small.truncated


def test_fetch_errors_keep_code_and_retryability():
    f = fail(
        handlers.do_fetch_webpage,
        fixture_backends(fetcher=StubFetch(error=FetchError("timeout", "slow", retryable=True))),
        "https://a.test/x",
        1000,
    )
    assert (f.code, f.retryable) == ("timeout", True)
    f = fail(
        handlers.do_fetch_webpage,
        fixture_backends(fetcher=StubFetch("<html></html>")),
        "https://a.test/x",
        1000,
    )
    assert f.code == "no_text"


def test_fetch_blocks_ssrf_through_the_real_fetcher():
    fetcher = SafeFetcher(resolver=lambda h: ["10.0.0.7"])
    b = fixture_backends(fetcher=fetcher)
    for url in (
        "https://internal.test/admin",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
    ):
        assert fail(handlers.do_fetch_webpage, b, url, 1000).code == "blocked"


class StubIndex:
    name = "stub"

    def __init__(self):
        self.chunks = [
            DocumentChunk(
                chunk_id=f"c{i}",
                document_id="d1",
                filename="paper.pdf",
                page=i + 1,
                score=1 - i / 10,
                text=f"chunk text {i}",
            )
            for i in range(6)
        ]

    def search(self, query, top_k, document_ids):
        return self.chunks

    def get_chunks(self, ids):
        return [c for c in self.chunks if c.chunk_id in ids]


def test_document_tools_use_the_index_contract_and_default_is_explicitly_unavailable():
    assert (
        fail(handlers.do_search_documents, fixture_backends(), "anything here", 3, None).code
        == "not_available"
    )
    assert fail(handlers.do_retrieve_chunks, fixture_backends(), ["c1"]).code == "not_available"
    b = fixture_backends(documents=StubIndex())
    out = handlers.do_search_documents(b, "anything here", 3, None)
    assert (
        out.index == "stub" and len(out.chunks) == 3
    )  # top_k enforced even if the index returns more
    assert [c.chunk_id for c in handlers.do_retrieve_chunks(b, ["c2", "c4"]).chunks] == ["c2", "c4"]


@pytest.fixture()
def done_run(tmp_path):
    env = make_env(tmp_path)
    rid = new_research(env, depth="standard")
    make_manager(env).start(rid)
    yield env, rid
    env.engine.dispose()


def test_export_markdown_and_json_from_the_database(done_run):
    env, rid = done_run
    b = fixture_backends(session_factory=env.SF)
    md = handlers.do_export_report(b, rid, "markdown")
    assert md.filename == f"research-{rid[:8]}.md" and "## References" in md.content
    assert md.size_bytes == len(md.content.encode())
    js = handlers.do_export_report(b, rid, "json")
    data = json.loads(js.content)
    assert (
        data["report"]["markdown"] == md.content and data["report"]["validation"]["passed"] is True
    )
    assert [s["ref"] for s in data["sources"]][:2] == ["S1", "S2"]
    assert any(s["evidence"] for s in data["sources"]) and data["query"]


def test_export_errors(done_run):
    env, rid = done_run
    b = fixture_backends(session_factory=env.SF)
    assert fail(handlers.do_export_report, b, "not-a-uuid", "markdown").code == "invalid_arguments"
    assert (
        fail(handlers.do_export_report, b, "0" * 8 + "-0000-0000-0000-" + "0" * 12, "markdown").code
        == "not_found"
    )
    empty = new_research(env)  # exists, but has no report
    assert fail(handlers.do_export_report, b, empty, "markdown").code == "not_found"
    tiny = fixture_backends(session_factory=env.SF, limits=ToolLimits(max_export_bytes=200))
    assert fail(handlers.do_export_report, tiny, rid, "markdown").code == "too_large"
    assert (
        fail(handlers.do_export_report, fixture_backends(), rid, "markdown").code
        == "not_configured"
    )


def test_result_count_is_capped_even_if_the_provider_ignores_max_results():
    from datetime import UTC, datetime

    from app.tools.search import SearchHit

    class Greedy:
        name = "greedy"

        def search(self, query, n):  # ignores n and returns far more than requested
            return [
                SearchHit(
                    title=f"T{i}",
                    url=f"https://g.test/{i}",
                    provider="greedy",
                    rank=i + 1,
                    query=query,
                    retrieved_at=datetime.now(UTC),
                )
                for i in range(25)
            ]

    out = handlers.do_search_web(fixture_backends(search=Greedy()), "agentic ai", 4)
    assert [h.rank for h in out.hits] == [1, 2, 3, 4]
