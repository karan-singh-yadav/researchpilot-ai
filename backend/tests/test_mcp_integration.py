"""The whole workflow through a real MCP subprocess, compared with the in-process path."""

import pytest
from sqlalchemy import select

from app.config import Settings
from app.errors import AppError
from app.mcp.client import (
    AgentToolbox,
    McpFetcher,
    McpSearchProvider,
    McpToolClient,
    McpToolError,
    ToolPermissionError,
    stdio_params,
)
from app.models import Evidence, Report, Source
from app.services.events import EventBus
from app.services.runner import RunManager, Tools, ToolServerUnavailable, make_checkpointer
from app.services.tooling import ToolHub
from app.tools.search import ToolNotConfiguredError
from tests.graph_env import CountingLLM, make_env, make_manager, new_research, row
from tests.helpers import FIXTURE


def mcp_settings(**kw) -> Settings:
    base = dict(
        _env_file=None,
        llm_provider="fake",
        search_provider="fixture",
        search_fixture_path=str(FIXTURE),
        tool_transport="mcp",
        search_max_results_per_query=7,
        llm_max_retries=0,
        run_executor="inline",
    )
    base.update(kw)
    return Settings(**base)


@pytest.fixture(scope="module")
def mcp_client():
    c = McpToolClient(stdio_params(mcp_settings()), call_timeout=30, connect_timeout=30)
    c.start()
    yield c
    c.close()


def snapshot(env, rid):
    db = env.SF()
    try:
        srcs = db.scalars(select(Source).where(Source.session_id == rid)).all()
        srcs = sorted(srcs, key=lambda s: int(s.source_ref[1:]))
        ev = sorted(
            (e.evidence_ref, e.passage)
            for s in srcs
            for e in db.scalars(select(Evidence).where(Evidence.source_id == s.id))
        )
        rep = db.scalars(select(Report).where(Report.session_id == rid)).one()
        return {
            "sources": [
                (s.source_ref, s.url, s.status, s.injection_suspected, s.provider, s.title)
                for s in srcs
            ],
            "evidence": ev,
            "validation_passed": rep.validation["passed"],
            "cited": sorted(rep.markdown.count(f"#ref-S{i}") > 0 for i in range(1, 5)),
        }
    finally:
        db.close()


def test_workflow_over_mcp_is_equivalent_to_the_direct_path(tmp_path, mcp_client):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    direct_env, mcp_env = make_env(tmp_path / "a"), make_env(tmp_path / "b")
    d_rid = new_research(direct_env, depth="standard")
    m_rid = new_research(mcp_env, depth="standard")
    make_manager(direct_env).start(d_rid)

    llm = CountingLLM()
    tools = Tools(
        llm,
        McpSearchProvider(AgentToolbox(mcp_client, "search")),
        McpFetcher(AgentToolbox(mcp_client, "reader")),
    )
    s = mcp_settings()
    RunManager(
        session_factory=mcp_env.SF, settings=s, tools_factory=lambda: tools, bus=EventBus(),
        checkpointer=make_checkpointer(":memory:"), executor="inline", sleep=lambda x: None,
    ).start(m_rid)  # fmt: skip

    assert row(direct_env, d_rid).status == row(mcp_env, m_rid).status == "completed"
    a, b = snapshot(direct_env, d_rid), snapshot(mcp_env, m_rid)
    assert b["sources"] == a["sources"]  # same S# ids, urls, failures, injection flags, titles
    assert b["evidence"] == a["evidence"]  # same verbatim evidence extracted through the tools
    assert b["validation_passed"] is a["validation_passed"] is True
    statuses = {src[1].rsplit("/", 1)[-1]: src[2] for src in b["sources"]}
    assert statuses["slow"] == "failed" and statuses["paper.pdf"] == "failed"
    assert any(src[3] for src in b["sources"])  # prompt-injection page was flagged via MCP too
    for env in (direct_env, mcp_env):
        env.engine.dispose()


def test_tool_calls_really_cross_the_mcp_boundary(mcp_client):
    names = {t.name for t in mcp_client.list_tools()}
    assert {"search_web", "fetch_webpage"} <= names
    hits = McpSearchProvider(AgentToolbox(mcp_client, "search")).search("agentic ai", 2)
    assert [h.provider for h in hits] == ["fixture", "fixture"]
    with pytest.raises(ToolPermissionError):  # the Search agent cannot fetch pages
        AgentToolbox(mcp_client, "search").call("fetch_webpage", url="https://example.test/x")


def test_export_through_mcp_matches_the_database(tmp_path):
    env = make_env(tmp_path)
    rid = new_research(env, depth="standard")
    make_manager(env).start(rid)  # produces a report in the shared database file
    s = mcp_settings(database_url=f"sqlite:///{env.path}")
    with McpToolClient(stdio_params(s), call_timeout=30) as c:
        md = AgentToolbox(c, "exporter").call("export_research_report", research_id=rid)
        js = AgentToolbox(c, "exporter").call(
            "export_research_report", research_id=rid, format="json"
        )
        with pytest.raises(McpToolError) as ei:
            AgentToolbox(c, "exporter").call(
                "export_research_report", research_id="0" * 8 + "-0000-0000-0000-" + "0" * 12
            )
    db = env.SF()
    stored = db.scalars(select(Report).where(Report.session_id == rid)).one()
    db.close()
    assert (
        md.content == stored.markdown and md.filename.endswith(".md") and '"sources"' in js.content
    )
    assert ei.value.code == "not_found"
    env.engine.dispose()


# ------------------------------------------------------------------------- ToolHub
def test_hub_selects_transport_reuses_one_client_and_closes_cleanly():
    direct = ToolHub(mcp_settings(tool_transport="direct"))
    assert type(direct.factory().search).__name__ == "FixtureSearch"
    direct.close()
    hub = ToolHub(mcp_settings())
    try:
        t1, t2 = hub.factory(), hub.factory()
        assert isinstance(t1.search, McpSearchProvider) and isinstance(t1.fetcher, McpFetcher)
        assert t1.search.toolbox.client is t2.search.toolbox.client  # ONE shared server process
        assert t1.search.toolbox.agent == "search" and t1.fetcher.toolbox.agent == "reader"
        assert t1.search.search("agentic ai", 1)
    finally:
        hub.close()
        hub.close()  # idempotent


def test_hub_validates_configuration_before_spawning_anything():
    with pytest.raises(ToolNotConfiguredError):
        ToolHub(mcp_settings(search_provider="none")).factory()
    with pytest.raises(ValueError):
        ToolHub(mcp_settings(llm_provider="openai", llm_model="m", llm_api_key="")).factory()


def test_unreachable_tool_server_gives_a_503_and_creates_no_session(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from mcp import StdioServerParameters

    import app.services.tooling as tooling
    from app.api.deps import get_bus, get_run_manager
    from app.database import get_db
    from app.main import create_app

    monkeypatch.setattr(
        tooling, "stdio_params", lambda s: StdioServerParameters(command="/no/such/server-binary")
    )
    env = make_env(tmp_path)
    hub = ToolHub(mcp_settings(mcp_connect_timeout_seconds=10))
    mgr = RunManager(
        session_factory=env.SF, settings=mcp_settings(), tools_factory=hub.factory, bus=EventBus(),
        checkpointer=make_checkpointer(":memory:"), executor="inline",
    )  # fmt: skip
    app = create_app()

    def _db():
        db = env.SF()
        try:
            yield db
            db.commit()
        finally:
            db.close()

    app.dependency_overrides.update(
        {get_db: _db, get_run_manager: lambda: mgr, get_bus: lambda: mgr.bus}
    )
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/api/research", json={"query": "Research agentic AI frameworks please."})
    assert r.status_code == 503 and r.json()["error"]["code"] == "tool_server_unavailable"
    assert c.get("/api/research").json()["total"] == 0
    with pytest.raises(ToolServerUnavailable):
        hub.factory()
    with pytest.raises(AppError):
        mgr.preflight()
    hub.close()
    env.engine.dispose()


def test_full_api_run_over_mcp_and_http_export(tmp_path):
    from fastapi.testclient import TestClient

    from app.api.deps import get_bus, get_run_manager
    from app.database import get_db
    from app.main import create_app

    env = make_env(tmp_path)
    s = mcp_settings(database_url=f"sqlite:///{env.path}")
    hub = ToolHub(s)
    mgr = RunManager(
        session_factory=env.SF, settings=s, tools_factory=hub.factory, bus=EventBus(),
        checkpointer=make_checkpointer(":memory:"), executor="inline", sleep=lambda x: None,
    )  # fmt: skip
    app = create_app()

    def _db():
        db = env.SF()
        try:
            yield db
            db.commit()
        finally:
            db.close()

    app.dependency_overrides.update(
        {get_db: _db, get_run_manager: lambda: mgr, get_bus: lambda: mgr.bus}
    )
    c = TestClient(app, raise_server_exceptions=False)
    try:
        r = c.post(
            "/api/research",
            json={"query": "Research agentic AI frameworks and limitations.", "depth": "standard"},
        )
        assert r.status_code == 201, r.text
        rid = r.json()["id"]
        assert c.get(f"/api/research/{rid}/status").json()["status"] == "completed"
        trace = c.get(f"/api/research/{rid}/trace?limit=500").json()["items"]
        assert any("search[" in e["summary"] and e["payload"]["provider"] == "mcp" for e in trace)
        md = c.get(f"/api/research/{rid}/export")
        assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown")
        assert md.headers["content-disposition"] == f'attachment; filename="research-{rid[:8]}.md"'
        assert md.text == c.get(f"/api/research/{rid}/report").json()["markdown"]
        js = c.get(f"/api/research/{rid}/export?format=json")
        assert js.status_code == 200 and js.json()["report"]["validation"]["passed"] is True
        assert c.get(f"/api/research/{rid}/export?format=pdf").status_code == 422
        assert c.get("/api/research/nope/export").status_code == 404
        empty = c.post(
            "/api/research", json={"query": "A session with no report yet.", "start": False}
        ).json()["id"]
        assert c.get(f"/api/research/{empty}/export").status_code == 404
    finally:
        hub.close()
        env.engine.dispose()


def test_cli_tools_work_offline_over_stdio():
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]

    def run(*args):
        return subprocess.run(
            [sys.executable, str(root / "scripts" / args[0]), *args[1:]],
            capture_output=True, text=True, encoding="utf-8", timeout=90,
        )  # fmt: skip

    listed = run("mcp_call.py", "list")
    assert listed.returncode == 0 and "search_web(query, max_results)" in listed.stdout
    out = run("mcp_call.py", "call", "search_web", "query=agentic ai", "max_results=2")
    assert out.returncode == 0 and len(json.loads(out.stdout)["hits"]) == 2
    bad = run("mcp_call.py", "call", "search_web", "query=ab")
    assert bad.returncode == 1 and "invalid_arguments" in bad.stdout
    demo = run("demo_pipeline.py", "--mcp", "--depth", "quick")
    assert demo.returncode == 0, demo.stderr[-400:]
    assert "## References" in demo.stdout and "OFFLINE FIXTURE DATA" in demo.stdout
