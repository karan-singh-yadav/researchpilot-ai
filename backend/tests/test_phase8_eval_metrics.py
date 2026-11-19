from pathlib import Path

from app.config import get_settings
from tests.test_run_api import build, create


def test_metrics_and_evaluation_compare_endpoints(tmp_path, monkeypatch):
    fixture = Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "fixtures" / "demo_web.json"
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture))
    get_settings.cache_clear()

    api = build(tmp_path)
    try:
        # Create one completed session so /api/metrics has non-zero session telemetry
        create(api)

        m = api.c.get("/api/metrics")
        assert m.status_code == 200
        mdata = m.json()
        assert mdata["sessions"]["total"] >= 1
        assert mdata["usage"]["total_tokens"] > 0

        # Run baseline and improved evaluation suites
        r_base = api.c.post("/api/evaluation/runs", json={"variant": "baseline", "dataset": "demo-web"})
        assert r_base.status_code == 201
        assert r_base.json()["workflow_variant"] == "baseline"

        r_imp = api.c.post("/api/evaluation/runs", json={"variant": "improved", "dataset": "demo-web"})
        assert r_imp.status_code == 201
        assert r_imp.json()["workflow_variant"] == "improved"

        runs = api.c.get("/api/evaluation/runs")
        assert runs.status_code == 200 and len(runs.json()) == 2

        cmp = api.c.get("/api/evaluation/compare")
        assert cmp.status_code == 200
        cdata = cmp.json()
        assert "completion_rate" in cdata["deltas"]
        assert "critic_pass_rate" in cdata["deltas"]
    finally:
        api.mgr.shutdown()
        api.env.engine.dispose()
        get_settings.cache_clear()
