"""End-to-end tests of the reference pipeline using the synthetic offline fixture."""

import json
import re

import pytest

from app.agents.analyst import AnalysisError
from app.config import Settings
from app.services.llm import build_llm_provider
from app.services.pipeline import run_pipeline
from app.tools.factory import build_fetcher, build_search_provider
from app.tools.fetcher import FixtureFetcher
from app.tools.search import FixtureSearch, SearchError
from tests.helpers import FIXTURE

Q = "Research the latest developments in agentic AI. Compare frameworks and limitations."


@pytest.fixture()
def tools():
    s = Settings(
        _env_file=None,
        llm_provider="fake",
        search_provider="fixture",
        search_fixture_path=str(FIXTURE),
    )
    return build_llm_provider(s), build_search_provider(s), build_fetcher(s)


def run(tools, **kw):
    llm, search, fetcher = tools
    return run_pipeline(Q, llm=llm, search=search, fetcher=fetcher, sleep=lambda s: None, **kw)


def test_end_to_end_offline_report_is_grounded(tools):
    res = run(tools, depth="standard")
    fixture = json.loads(FIXTURE.read_text())
    md = res.report.markdown
    # every in-text citation resolves to a real source, and every source URL is from the fixture
    cited = set(re.findall(r"\[\[(S\d+)\]\]", md))
    assert cited and cited <= set(res.sources)
    for ref in res.report.cited:
        assert res.sources[ref].url in fixture["pages"]
    assert "OFFLINE FIXTURE DATA" in md and "not a language model" in md
    # every evidence passage is verbatim text from the page it is attributed to
    from app.agents.reader import normalize_text
    from app.tools.html_extract import extract_page

    assert res.evidence
    for e in res.evidence:
        page = fixture["pages"][res.sources[e.source_ref].url]
        text = normalize_text(extract_page("text/html", page["html"]).text)
        assert normalize_text(e.passage) in text
    assert res.usage.total > 0 and set(res.timings_ms) == {"plan", "research", "analysis", "write"}


def test_failures_dedupe_and_injection_are_handled(tools):
    res = run(tools, depth="standard", results_per_query=7)
    by_url = {s.url: s for s in res.sources.values()}
    assert by_url["https://example.test/slow"].status == "failed"
    assert by_url["https://example.test/paper.pdf"].status == "failed"
    assert "unsupported" in by_url["https://example.test/paper.pdf"].error
    # tracking-param duplicate collapsed into a single source
    assert sum("agentic-overview" in u for u in by_url) == 1
    # injection page flagged; its hidden/instruction text never became evidence
    assert by_url["https://example.test/untrusted-page"].injection_suspected
    assert not any(
        "ignore" in e.passage.lower() and "instructions" in e.passage.lower() for e in res.evidence
    )
    assert not any("only source" in e.passage.lower() for e in res.evidence)
    assert any("prompt injection" in w for w in res.warnings)
    ref = by_url["https://example.test/untrusted-page"].ref
    if ref in res.report.cited:  # if the flagged page is cited, the reader must be warned
        assert "page contained instruction-like text" in res.report.markdown


def test_source_budget_is_respected(tools):
    res = run(tools, depth="quick")
    assert len(res.sources) <= res.plan.budget.max_sources == 5


class OnePageSearch:
    name = "stub"

    def __init__(self):
        self.inner = FixtureSearch.from_path(FIXTURE)
        self.queries = []

    def search(self, query, n):
        self.queries.append(query)
        return self.inner.search(query, n)[:1]  # only ever one result -> thin evidence


def test_follow_up_loop_is_bounded_by_budget(tools):
    llm, _, fetcher = tools
    search = OnePageSearch()
    res = run_pipeline(
        Q, llm=llm, search=search, fetcher=fetcher, depth="quick", sleep=lambda s: None
    )
    assert res.iterations <= res.plan.budget.max_iterations == 1
    assert res.analysis.needs_more_research
    assert any("insufficient" in w or "no new evidence" in w for w in res.warnings)
    assert "Evidence was limited" in res.report.markdown
    none = run_pipeline(
        Q,
        llm=llm,
        search=OnePageSearch(),
        fetcher=fetcher,
        depth="quick",
        max_iterations=0,
        sleep=lambda s: None,
    )
    assert none.iterations == 0


class AllFailSearch:
    name = "stub"

    def search(self, query, n):
        raise SearchError("provider down", retryable=True)


def test_total_search_failure_raises_clean_error_without_report(tools):
    llm, _, fetcher = tools
    with pytest.raises(AnalysisError, match="No evidence"):
        run_pipeline(Q, llm=llm, search=AllFailSearch(), fetcher=fetcher, sleep=lambda s: None)


def test_all_pages_unreadable_raises_clean_error(tools):
    llm, search, _ = tools
    empty = FixtureFetcher({"pages": {}})
    with pytest.raises(AnalysisError):
        run_pipeline(Q, llm=llm, search=search, fetcher=empty, sleep=lambda s: None)


def test_demo_script_runs_offline():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, str(root / "scripts" / "demo_pipeline.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "## References" in out.stdout and "OFFLINE FIXTURE DATA" in out.stdout
