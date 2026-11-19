"""API tests for the workflow endpoints (real graph + real manager, offline fixtures)."""

import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_bus, get_run_manager
from app.config import get_settings
from app.database import get_db
from app.main import create_app
from app.security.rate_limit import limiter
from app.tools.search import ToolNotConfiguredError
from tests.graph_env import CountingLLM, InstrumentedSearch, make_env, make_manager
from tests.test_graph_workflow import fail_once

Q = "Research the latest developments in agentic AI frameworks."


def build(tmp_path, **mgr_kwargs):
    env = make_env(tmp_path)
    mgr = make_manager(env, **mgr_kwargs)
    app = create_app()

    def _db():
        db = env.SF()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_run_manager] = lambda: mgr
    app.dependency_overrides[get_bus] = lambda: mgr.bus
    limiter.reset()
    get_settings.cache_clear()
    return SimpleNamespace(
        c=TestClient(app, raise_server_exceptions=False), env=env, mgr=mgr, app=app
    )


@pytest.fixture()
def api(tmp_path):
    a = build(tmp_path)
    yield a
    a.mgr.shutdown()
    a.env.engine.dispose()
    get_settings.cache_clear()


def create(api, **kw):
    r = api.c.post("/api/research", json={"query": Q, "depth": "quick", **kw})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def sse(api, rid):
    r = api.c.get(f"/api/research/{rid}/events")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in r.text.split("\n\n"):
        if block.startswith("id:") or "\nevent:" in block:
            data = [ln[6:] for ln in block.splitlines() if ln.startswith("data: ")]
            if data:
                events.append(json.loads(data[0]))
    return r.text, events


def test_create_starts_workflow_and_all_result_endpoints_work(api):
    rid = create(api)
    st = api.c.get(f"/api/research/{rid}/status").json()
    assert st["status"] == "completed"

    rep = api.c.get(f"/api/research/{rid}/report").json()
    assert "## References" in rep["markdown"] and rep["validation"]["passed"] is True
    assert rep["version"] == 1 and "disclaimer" in rep["validation"]

    srcs = api.c.get(f"/api/research/{rid}/sources").json()
    assert [s["source_ref"] for s in srcs][:2] == ["S1", "S2"]
    good = next(s for s in srcs if s["status"] == "read" and s["evidence"])
    assert good["evidence"][0]["evidence_ref"].startswith("E") and good["url"].startswith(
        "https://"
    )

    trace = api.c.get(f"/api/research/{rid}/trace?limit=500").json()
    assert trace["total"] > 8 and trace["items"][-1]["event_type"] == "run_end"
    assert any(e["agent"] == "critic" for e in trace["items"])

    one = api.c.get(f"/api/research/{rid}").json()
    assert (
        one["token_usage"] > 0
        and one["status"] == "completed"
        and isinstance(one["warnings"], list)
    )


def test_sse_replays_finished_run_and_closes(api):
    rid = create(api)
    text, events = sse(api, rid)
    assert events[-1]["event_type"] == "run_end" and "event: run_end" in text
    assert [e["event_type"] for e in events][0] == "run_start" or events[0]["agent"] == "system"
    assert {"node_start", "node_end", "tool_call"} <= {e["event_type"] for e in events}


def test_sse_streams_live_events_from_a_background_run(tmp_path):
    gate, started = threading.Event(), threading.Event()
    api = build(tmp_path, executor="thread", search=InstrumentedSearch(gate=gate, started=started))
    try:
        rid = create(api)  # returns immediately; the run is blocked inside search
        assert started.wait(5)
        timer = threading.Timer(0.3, gate.set)
        timer.start()
        with api.c.stream("GET", f"/api/research/{rid}/events") as r:
            text = "".join(r.iter_text())
        events = [json.loads(ln[6:]) for ln in text.splitlines() if ln.startswith("data: ")]
        types = [e["event_type"] for e in events]
        assert types[-1] == "run_end" and "node_end" in types
        assert len({e["id"] for e in events}) == len(events)  # replay + live never duplicates
        assert any(e["agent"] == "analyst" for e in events)  # events produced AFTER we connected
    finally:
        gate.set()
        api.mgr.wait(rid) if "rid" in locals() else None
        api.mgr.shutdown()


def test_state_conflicts_and_not_found(api):
    rid = create(api)
    r = api.c.post(f"/api/research/{rid}/resume")
    assert r.status_code == 409 and r.json()["error"]["code"] == "invalid_state"
    r = api.c.post(f"/api/research/{rid}/cancel")
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_running"
    for path in ("resume", "cancel"):
        assert api.c.post(f"/api/research/nope/{path}").status_code == 404
    for path in ("report", "sources", "events"):
        assert api.c.get(f"/api/research/nope/{path}").status_code == 404
    rid2 = api.c.post("/api/research", json={"query": Q, "start": False}).json()["id"]
    assert api.c.get(f"/api/research/{rid2}/report").json()["error"]["code"] == "report_not_found"


def test_unconfigured_provider_fails_fast_without_creating_a_session(api):
    def broken():
        raise ToolNotConfiguredError("No search provider configured. Set SEARCH_PROVIDER.")

    api.mgr.tools_factory = broken
    r = api.c.post("/api/research", json={"query": Q})
    assert r.status_code == 503 and r.json()["error"]["code"] == "search_not_configured"
    assert api.c.get("/api/research").json()["total"] == 0
    ok = api.c.post("/api/research", json={"query": Q, "start": False})  # no run needed => allowed
    assert ok.status_code == 201


def test_plan_approval_flow_over_http(api):
    rid = create(api, require_plan_approval=True)
    assert api.c.get(f"/api/research/{rid}/status").json()["status"] == "awaiting_approval"
    assert api.c.get(f"/api/research/{rid}/report").status_code == 404  # nothing researched yet
    r = api.c.post(f"/api/research/{rid}/resume")
    assert r.status_code == 409 and r.json()["error"]["code"] == "plan_not_approved"
    plan = api.c.get(f"/api/research/{rid}/plan").json()
    assert plan["plan_approved"] is False and plan["plan"]["subtasks"]
    assert api.c.post(f"/api/research/{rid}/plan/approve").status_code == 200
    r = api.c.post(f"/api/research/{rid}/resume")
    assert r.status_code == 202 and r.json()["status"] == "completed"
    assert api.c.get(f"/api/research/{rid}/report").status_code == 200


def test_failed_run_can_be_resumed_over_http(tmp_path):
    api = build(tmp_path, llm=CountingLLM({"analysis": fail_once}))
    try:
        rid = create(api)
        st = api.c.get(f"/api/research/{rid}/status").json()
        assert st["status"] == "failed" and "error" in st["error"].lower()
        assert "provider rejected" not in json.dumps(st)  # internals are not leaked to clients
        r = api.c.post(f"/api/research/{rid}/resume")
        assert r.status_code == 202 and r.json()["status"] == "completed"
        assert api.c.get(f"/api/research/{rid}/report").status_code == 200
    finally:
        api.mgr.shutdown()


def test_delete_removes_results_and_is_refused_while_running(tmp_path):
    gate, started = threading.Event(), threading.Event()
    api = build(tmp_path, executor="thread", search=InstrumentedSearch(gate=gate, started=started))
    try:
        rid = create(api)
        assert started.wait(5)
        r = api.c.delete(f"/api/research/{rid}")
        assert r.status_code == 409 and r.json()["error"]["code"] == "run_in_progress"
        assert api.c.post(f"/api/research/{rid}/cancel").status_code == 202
        gate.set()
        api.mgr.wait(rid)
        assert api.c.get(f"/api/research/{rid}/status").json()["status"] == "cancelled"
        assert api.c.delete(f"/api/research/{rid}").status_code == 204
        assert api.c.get(f"/api/research/{rid}/sources").status_code == 404
    finally:
        gate.set()
        api.mgr.shutdown()


def test_graph_endpoint_returns_real_topology(api):
    m = api.c.get("/api/graph").json()["mermaid"]
    assert "research_task" in m and "approval_gate" in m and "critic" in m


def test_new_endpoints_require_api_key(api, monkeypatch):
    rid = create(api)
    monkeypatch.setenv("API_KEYS", "k1")
    get_settings.cache_clear()
    for method, path in [
        ("get", f"/api/research/{rid}/report"),
        ("get", f"/api/research/{rid}/sources"),
        ("get", f"/api/research/{rid}/events"),
        ("post", f"/api/research/{rid}/resume"),
        ("post", f"/api/research/{rid}/cancel"),
        ("get", "/api/graph"),
    ]:
        assert getattr(api.c, method)(path).status_code == 401, path
    assert api.c.get(f"/api/research/{rid}/report", headers={"X-API-Key": "k1"}).status_code == 200


def test_sse_dedupes_events_present_in_both_replay_and_live_queue(api):
    """Simulates the subscribe-then-replay race: stored events also arrive on the live queue."""
    from app.services.events import record_event

    rid = api.c.post("/api/research", json={"query": Q, "start": False}).json()["id"]
    stored = [
        record_event(api.env.SF, None, rid, "system", "node_start", f"stored {i}") for i in range(3)
    ]
    live_only = {"id": "live-1", "event_type": "node_end", "agent": "writer", "summary": "x"}
    terminal = {"id": "live-2", "event_type": "run_end", "agent": "system", "summary": "end"}
    prefill = [stored[0], stored[1], live_only, stored[2], terminal]  # duplicates + new + terminal

    original = api.mgr.bus.subscribe

    def subscribe(research_id):
        q = original(research_id)
        for ev in prefill:
            q.put_nowait(ev)
        return q

    api.mgr.bus.subscribe = subscribe
    _, events = sse(api, rid)
    ids = [e["id"] for e in events]
    assert ids == [s["id"] for s in stored] + ["live-1", "live-2"]  # each exactly once, in order
