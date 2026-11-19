from app.api.deps import get_llm
from app.services.llm.fake import FakeLLM
from tests.test_run_api import build, create


def test_hitl_patch_reject_and_approve_with_auto_resume(tmp_path):
    api = build(tmp_path)
    try:
        rid = create(api, require_plan_approval=True)
        assert api.c.get(f"/api/research/{rid}/status").json()["status"] == "awaiting_approval"

        # 1. Partial PATCH update (objectives only; subtasks preserved)
        patched = api.c.patch(
            f"/api/research/{rid}/plan",
            json={"objectives": ["Focus strictly on LangGraph vs AutoGen"]},
        )
        assert patched.status_code == 200
        assert patched.json()["plan"]["objectives"] == ["Focus strictly on LangGraph vs AutoGen"]
        assert patched.json()["plan"]["generated_by"] == "human-edited"

        # 2. Reject with feedback -> regenerates plan and logs plan_rejected event
        api.app.dependency_overrides[get_llm] = lambda: FakeLLM()
        rejected = api.c.post(
            f"/api/research/{rid}/plan/reject",
            json={"feedback": "Include enterprise security and cost trade-offs"},
        )
        assert rejected.status_code == 200
        assert rejected.json()["status"] == "awaiting_approval"
        trace_types = [
            e["event_type"] for e in api.c.get(f"/api/research/{rid}/trace").json()["items"]
        ]
        assert "plan_rejected" in trace_types

        # 3. Approve with audit note + auto_resume=true completes the workflow in one call
        approved = api.c.post(
            f"/api/research/{rid}/plan/approve?auto_resume=true",
            json={"note": "Looks solid, proceed"},
        )
        assert approved.status_code == 200
        assert api.c.get(f"/api/research/{rid}/status").json()["status"] == "completed"
    finally:
        api.mgr.shutdown()
        api.env.engine.dispose()


def test_citations_endpoint_resolves_inline_spans_and_evidence(tmp_path):
    api = build(tmp_path)
    try:
        rid = create(api)
        r = api.c.get(f"/api/research/{rid}/citations")
        assert r.status_code == 200
        body = r.json()
        assert body["total_citations"] >= 1
        assert body["unique_sources"] >= 1
        first = body["occurrences"][0]
        assert first["resolved"] is True
        assert first["source_ref"].startswith("S")
        assert isinstance(first["evidence"], list) and len(first["evidence"]) >= 1
        assert body["sources"][0]["citation_count"] >= 1

        reports = api.c.get(f"/api/research/{rid}/reports")
        assert reports.status_code == 200 and len(reports.json()) >= 1
    finally:
        api.mgr.shutdown()
        api.env.engine.dispose()
