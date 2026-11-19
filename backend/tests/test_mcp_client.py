import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest
from mcp import StdioServerParameters
from mcp.server.mcpserver import MCPServer

from app.config import Settings
from app.mcp.client import (
    AGENT_PERMISSIONS,
    AgentToolbox,
    McpFetcher,
    McpSearchProvider,
    McpToolClient,
    McpToolError,
    ToolPermissionError,
    stdio_params,
)
from app.mcp.schemas import ALL_TOOLS
from app.mcp.server import build_server
from app.tools.fetcher import FetchError
from app.tools.search import SearchError
from tests.helpers import FIXTURE
from tests.mcp_helpers import SlowSearch, StubFetch, fixture_backends


def inproc(backends=None, **kw):
    """A client whose server lives in this process (new server object per (re)connect)."""
    holder = backends or fixture_backends()
    return McpToolClient(lambda: build_server(holder), connect_timeout=10, **kw)


# --------------------------------------------------------------------- bridge behaviour
def test_calls_work_and_results_are_validated_models():
    with inproc() as c:
        out = c.call("search_web", {"query": "agentic ai", "max_results": 2})
        assert out.provider == "fixture" and len(out.hits) == 2
        page = c.call("fetch_webpage", {"url": "https://example.test/agentic-overview"})
        assert page.untrusted and page.title
        assert {t.name for t in c.list_tools()} == set(ALL_TOOLS)


def test_many_threads_share_one_session_and_run_concurrently():
    search = SlowSearch(delay=0.4)
    with inproc(fixture_backends(search=search)) as c:
        results, errors = [], []

        def worker(i):
            try:
                results.append(c.call("search_web", {"query": f"agentic {i}"}))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        t0 = time.monotonic()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        [t.start() for t in threads]
        [t.join(20) for t in threads]
        elapsed = time.monotonic() - t0
    assert not errors and len(results) == 8
    assert search.peak >= 4 and elapsed < 8 * 0.4 * 0.7  # genuinely overlapped, not serialised


def test_tool_errors_become_typed_exceptions():
    with inproc(
        fixture_backends(fetcher=StubFetch(error=FetchError("timeout", "slow", retryable=True)))
    ) as c:
        with pytest.raises(McpToolError) as ei:
            c.call("fetch_webpage", {"url": "https://a.test/x"})
        assert (ei.value.code, ei.value.retryable, ei.value.tool) == (
            "timeout",
            True,
            "fetch_webpage",
        )
        with pytest.raises(McpToolError) as ei:
            c.call("search_web", {"query": "a"})
        assert ei.value.code == "invalid_arguments"
        with pytest.raises(McpToolError) as ei:
            c.call("search_documents", {"query": "something"})
        assert ei.value.code == "not_available"


def test_client_side_timeout_is_a_retryable_error_not_a_hang():
    with inproc(fixture_backends(search=SlowSearch(delay=3.0))) as c:
        t0 = time.monotonic()
        with pytest.raises(McpToolError) as ei:
            c.call("search_web", {"query": "agentic ai"}, timeout=0.4)
        assert ei.value.code == "timeout" and ei.value.retryable and time.monotonic() - t0 < 3


def test_malformed_server_responses_are_rejected_by_output_validation():
    evil = MCPServer("evil")

    @evil.tool(name="search_web", description="returns a shape that violates the contract")
    def search_web(query: str) -> dict:
        return {"hits": "not-a-list", "provider": 5}

    with McpToolClient(lambda: evil, connect_timeout=10) as c:
        with pytest.raises(McpToolError) as ei:
            c.call("search_web", {"query": "agentic ai"})
        assert ei.value.code == "invalid_response"


def test_transport_failure_restarts_the_session_and_retries_once():
    with inproc() as c:
        assert c.call("search_web", {"query": "agentic ai"}).hits
        dead = c._session
        c._stop_session(dead)  # simulates the server going away underneath us
        out = c.call("search_web", {"query": "agentic ai again"})
        assert out.hits and c.restarts == 1 and c._session is not dead


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses /proc to find the child")
def test_real_subprocess_killed_with_sigkill_is_recovered():
    s = Settings(_env_file=None, search_provider="fixture", search_fixture_path=str(FIXTURE))
    with McpToolClient(stdio_params(s), call_timeout=20) as c:
        c.call("search_web", {"query": "agentic ai"})
        me = str(os.getpid())
        kids = [
            int(p)
            for p in os.listdir("/proc")
            if p.isdigit() and Path(f"/proc/{p}/stat").read_text().split(")")[-1].split()[1] == me
        ]
        assert kids
        for k in kids:
            os.kill(k, signal.SIGKILL)
        time.sleep(0.3)
        assert (
            c.call("search_web", {"query": "after sigkill", "max_results": 1}).hits
            and c.restarts == 1
        )


def test_startup_failures_are_reported_not_hung():
    bad = StdioServerParameters(command=sys.executable, args=["-m", "definitely_not_a_module_xyz"])
    c = McpToolClient(bad, connect_timeout=15)
    t0 = time.monotonic()
    with pytest.raises(McpToolError) as ei:
        c.call("search_web", {"query": "agentic ai"})
    assert ei.value.code == "unavailable" and time.monotonic() - t0 < 15
    c.close()
    missing = McpToolClient(StdioServerParameters(command="/no/such/binary"), connect_timeout=10)
    with pytest.raises(McpToolError) as ei:
        missing.start()
    assert ei.value.code == "unavailable"
    missing.close()


def test_close_is_idempotent_and_blocks_further_calls():
    c = inproc()
    c.start()
    c.close()
    c.close()
    with pytest.raises(McpToolError) as ei:
        c.call("search_web", {"query": "agentic ai"})
    assert ei.value.code == "unavailable"


# --------------------------------------------------------------------- permissions
class Recorder:
    def __init__(self):
        self.calls = []

    def call(self, tool, arguments=None, **kw):
        self.calls.append(tool)
        return object()


def test_agents_can_only_call_their_own_tools_and_denied_calls_never_reach_the_server():
    allowed = {
        "search": "search_web",
        "reader": "fetch_webpage",
        "document": "search_documents",
        "exporter": "export_research_report",
    }
    for agent, own in allowed.items():
        rec = Recorder()
        box = AgentToolbox(rec, agent)
        box.call(own)
        assert rec.calls == [own]
        for other in ALL_TOOLS:
            if other not in AGENT_PERMISSIONS[agent]:
                with pytest.raises(ToolPermissionError) as ei:
                    box.call(other)
                assert ei.value.code == "permission_denied"
        assert rec.calls == [own]  # nothing else was sent
    with pytest.raises(ValueError):
        AgentToolbox(Recorder(), "hacker")
    # no tool is granted to every agent, and the Writer/Analyst/Critic have no tools at all
    all_granted = [t for t in ALL_TOOLS if all(t in p for p in AGENT_PERMISSIONS.values())]
    assert all_granted == [] and not {"writer", "analyst", "critic"} & set(AGENT_PERMISSIONS)


# --------------------------------------------------------------------- adapters
def test_adapters_translate_tool_errors_to_the_agents_native_exceptions():
    with inproc(
        fixture_backends(fetcher=StubFetch(error=FetchError("blocked", "Host is not allowed")))
    ) as c:
        with pytest.raises(FetchError) as ei:
            McpFetcher(AgentToolbox(c, "reader")).fetch("https://internal.test/x")
        assert ei.value.code == "blocked" and not ei.value.retryable
        with pytest.raises(SearchError) as ei:
            McpSearchProvider(AgentToolbox(c, "search")).search("a", 3)  # fails argument validation
        assert not ei.value.retryable
        hits = McpSearchProvider(AgentToolbox(c, "search")).search("agentic ai", 3)
        assert (
            [h.rank for h in hits] == [1, 2, 3]
            and hits[0].provider == "fixture"
            and hits[0].query == "agentic ai"
        )
    with inproc(fixture_backends(search=SlowSearch(delay=0))) as c:
        assert (
            McpSearchProvider(AgentToolbox(c, "search")).search("agentic ai", 1)[0].url
            == "https://s.test/1"
        )


def test_mcp_fetcher_returns_pre_extracted_pages():
    with inproc() as c:
        page = McpFetcher(AgentToolbox(c, "reader")).fetch("https://example.test/agentic-overview")
    assert page.extracted and page.title.startswith("Agentic AI") and page.published_at.year == 2026
    assert page.provider == "fixture" and "<" not in page.text and page.content_type == "text/plain"


# --------------------------------------------------------------------- launch parameters
def test_stdio_params_apply_least_privilege_and_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(
        _env_file=None,
        llm_api_key="SECRET-LLM-KEY",
        llm_provider="openai",
        search_provider="fixture",
        search_fixture_path="rel/fixture.json",
        search_api_key="search-key",
        database_url="sqlite:///./data/x.db",
    )
    p = stdio_params(s)
    env = p.env
    assert (
        "LLM_API_KEY" not in env and "SECRET-LLM-KEY" not in env.values()
    )  # tool server never gets it
    assert env["SEARCH_API_KEY"] == "search-key" and env["MCP_IGNORE_DOTENV"] == "1"
    assert Path(env["SEARCH_FIXTURE_PATH"]).is_absolute() and env["SEARCH_FIXTURE_PATH"].startswith(
        str(tmp_path)
    )
    assert env["DATABASE_URL"] == f"sqlite:///{tmp_path / 'data' / 'x.db'}"
    assert p.args == ["-m", "app.mcp.server"] and p.command == sys.executable
    assert (Path(p.cwd) / "app" / "mcp" / "server.py").exists()
