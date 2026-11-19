import pytest

from app.api.deps import get_llm
from app.models import AgentEvent, ResearchSession, ResearchTask
from app.services.llm import LLMError
from app.services.llm.fake import FakeLLM
from tests.helpers import ScriptedLLM, plan_json, task

Q = "Research the latest developments in agentic AI frameworks."


@pytest.fixture()
def use_llm(client):
    def _set(llm):
        client.app.dependency_overrides[get_llm] = lambda: llm

    _set(FakeLLM())
    return _set


def mk(client, **kw):
    return client.post(
        "/api/research", json={"start": False, "query": Q, "depth": "quick", **kw}
    ).json()["id"]


def test_plan_without_approval_is_planned_and_persisted(client, use_llm, db_session):
    rid = mk(client)
    r = client.post(f"/api/research/{rid}/plan")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "planned" and body["plan_approved"] is True
    assert body["plan"]["generated_by"].startswith("fake:")
    assert body["plan"]["budget"]["max_subtasks"] == 3

    assert client.get(f"/api/research/{rid}/plan").json()["plan"]["mode"] == "web"
    tasks = db_session.query(ResearchTask).filter_by(session_id=rid).all()
    assert len(tasks) == len(body["plan"]["subtasks"]) and all(
        t.assigned_agent == "search" for t in tasks
    )
    s = client.get(f"/api/research/{rid}").json()
    assert s["token_usage"] > 0 and s["estimated_cost_usd"] == 0.0  # no prices configured


def test_trace_records_planning_events(client, use_llm):
    rid = mk(client)
    client.post(f"/api/research/{rid}/plan")
    t = client.get(f"/api/research/{rid}/trace").json()
    assert [e["event_type"] for e in t["items"]] == ["node_start", "node_end"]
    assert t["items"][1]["agent"] == "supervisor" and t["items"][1]["duration_ms"] is not None
    assert "Plan created" in t["items"][1]["summary"]


def test_approval_flow(client, use_llm):
    rid = mk(client, require_plan_approval=True)
    body = client.post(f"/api/research/{rid}/plan").json()
    assert body["status"] == "awaiting_approval" and body["plan_approved"] is False
    ok = client.post(f"/api/research/{rid}/plan/approve").json()
    assert ok["status"] == "planned" and ok["plan_approved"] is True
    again = client.post(f"/api/research/{rid}/plan/approve")
    assert again.status_code == 409 and again.json()["error"]["code"] == "invalid_state"


def test_edit_plan_valid_and_resets_approval(client, use_llm):
    rid = mk(client, require_plan_approval=True)
    client.post(f"/api/research/{rid}/plan/approve")  # 409 before plan exists; ignored
    client.post(f"/api/research/{rid}/plan")
    client.post(f"/api/research/{rid}/plan/approve")
    edit = {
        "objectives": ["Only look at frameworks"],
        "subtasks": [task("t1", queries=["langgraph vs crewai"])],
        "report_outline": ["Summary", "Comparison"],
        "budget": {"max_subtasks": 500},  # must be ignored
    }
    r = client.put(f"/api/research/{rid}/plan", json=edit)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "awaiting_approval" and body["plan_approved"] is False
    assert (
        body["plan"]["generated_by"] == "human-edited"
        and body["plan"]["budget"]["max_subtasks"] == 3
    )


def test_edit_plan_rejects_rule_violations(client, use_llm):
    rid = mk(client)
    client.post(f"/api/research/{rid}/plan")
    bad = {
        "objectives": ["x" * 5],
        "subtasks": [task("t1", agent="document")],
        "report_outline": ["Summary", "Body"],
    }
    r = client.put(f"/api/research/{rid}/plan", json=bad)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_plan"
    cyc = {**bad, "subtasks": [task("t1", deps=["t1"])]}
    assert client.put(f"/api/research/{rid}/plan", json=cyc).status_code == 422


def test_llm_failure_marks_session_failed_and_is_retryable(client, use_llm):
    use_llm(ScriptedLLM([LLMError("bad key")]))
    rid = mk(client)
    r = client.post(f"/api/research/{rid}/plan")
    assert r.status_code == 502 and r.json()["error"]["code"] == "planning_failed"
    s = client.get(f"/api/research/{rid}").json()
    assert s["status"] == "failed" and s["error"]
    assert client.get(f"/api/research/{rid}/trace").json()["items"][-1]["event_type"] == "error"
    use_llm(ScriptedLLM([plan_json()]))  # failed sessions may be re-planned
    assert client.post(f"/api/research/{rid}/plan").status_code == 200


def test_unusable_llm_output_returns_502_and_counts_tokens(client, use_llm):
    use_llm(ScriptedLLM(["junk"] * 3))
    rid = mk(client)
    assert client.post(f"/api/research/{rid}/plan").status_code == 502
    s = client.get(f"/api/research/{rid}").json()
    assert s["status"] == "failed" and s["token_usage"] == 45


def test_state_guards_and_404s(client, use_llm, db_session):
    assert client.post("/api/research/nope/plan").status_code == 404
    rid = mk(client)
    assert client.get(f"/api/research/{rid}/plan").json()["error"]["code"] == "plan_not_found"
    db_session.get(ResearchSession, rid).status = "running"
    db_session.commit()
    assert client.post(f"/api/research/{rid}/plan").status_code == 409
    assert client.put(f"/api/research/{rid}/plan", json={}).status_code in (409, 422)


def test_unconfigured_provider_gives_503(client):
    client.app.dependency_overrides.pop(get_llm, None)
    import os

    from app.config import get_settings

    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_API_KEY"] = ""
    os.environ["LLM_MODEL"] = "m"
    get_settings.cache_clear()
    try:
        rid = mk(client)
        r = client.post(f"/api/research/{rid}/plan")
        assert r.status_code == 503 and r.json()["error"]["code"] == "llm_not_configured"
    finally:
        for k in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL"):
            os.environ.pop(k, None)
        get_settings.cache_clear()


def test_replan_replaces_tasks(client, use_llm, db_session):
    rid = mk(client)
    client.post(f"/api/research/{rid}/plan")
    client.post(f"/api/research/{rid}/plan")
    n = db_session.query(ResearchTask).filter_by(session_id=rid).count()
    assert n == len(client.get(f"/api/research/{rid}/plan").json()["plan"]["subtasks"])
    assert db_session.query(AgentEvent).filter_by(session_id=rid).count() == 4
