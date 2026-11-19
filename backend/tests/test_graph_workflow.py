"""End-to-end tests of the LangGraph workflow (real graph, real agents, offline fixtures)."""

import json
import re
import threading
from collections import Counter
from pathlib import Path

import pytest
from sqlalchemy import select

from app.errors import AppError
from app.graph.builder import build_graph, render_graph_doc
from app.models import Evidence, Report, ResearchTask, Source
from app.schemas.plan import PlanDraft
from app.services import planning
from app.services.events import EventBus, event_to_dict
from app.services.llm import LLMError
from app.services.runner import make_checkpointer
from app.tools.fetcher import FixtureFetcher
from tests.graph_env import (
    CountingLLM,
    InstrumentedSearch,
    events,
    make_env,
    make_manager,
    new_research,
    row,
)
from tests.helpers import task


@pytest.fixture()
def env(tmp_path):
    e = make_env(tmp_path)
    yield e
    e.engine.dispose()


def unsupported_first_claim(n, kw):
    """Critic override: claim 1 is always 'unsupported', everything else supported."""
    ids = [int(x) for x in re.findall(r"^\[C(\d+)\]", kw["user"], re.M)]
    v = [
        {"claim_id": i, "verdict": "unsupported" if i == 1 else "supported", "reason": "test"}
        for i in ids
    ]
    return json.dumps({"verdicts": v})


# ------------------------------------------------------------------ happy path
def test_full_run_completes_and_persists_everything(env):
    rid = new_research(env, depth="standard")
    mgr = make_manager(env, search=InstrumentedSearch())
    mgr.start(rid)
    r = row(env, rid)
    assert r.status == "completed" and r.completed_at and r.token_usage > 0
    db = env.SF()
    rep = db.scalars(select(Report).where(Report.session_id == rid)).one()
    assert rep.validation["passed"] is True and "OFFLINE FIXTURE DATA" in rep.markdown
    srcs = db.scalars(select(Source).where(Source.session_id == rid)).all()
    assert {s.source_ref for s in srcs} >= {"S1", "S2"} and any(s.status == "failed" for s in srcs)
    ev = db.scalars(select(Evidence)).all()
    assert ev and all(e.evidence_ref.startswith("E") and e.passage for e in ev)
    tasks = db.scalars(select(ResearchTask).where(ResearchTask.session_id == rid)).all()
    assert tasks and all(t.status == "completed" for t in tasks)
    db.close()
    assert any("prompt injection" in w for w in r.warnings)
    kinds = [(e.agent, e.event_type) for e in events(env, rid)]
    for agent in ("system", "supervisor", "search", "reader", "analyst", "writer", "critic"):
        assert any(a == agent for a, _ in kinds), agent
    assert kinds[-1] == ("system", "run_end")
    last = events(env, rid)[-1]
    assert last.payload["status"] == "completed"


def test_parallel_workers_and_dependency_waves(env):
    rid = new_research(env, depth="standard")
    search = InstrumentedSearch(delay=0.3)
    make_manager(env, search=search).start(rid)
    by = {q.split(" ", 5)[-1]: (s, e) for q, s, e in search.log}
    find = lambda kw: next(v for k, v in by.items() if kw in k)  # noqa: E731
    t1, t2, t3 = find("overview"), find("comparison"), find("limitations risks")
    # wave 1 (t1,t2,t3 independent) overlaps in time => genuinely parallel
    assert t2[0] < t1[1] and t3[0] < t1[1] and t3[0] < t2[1]
    # wave 2 (t4,t5 depend on t1) starts only after wave 1 finished
    t4 = find("real-world applications")
    assert t4[0] >= max(t1[1], t2[1], t3[1]) - 0.01


def test_source_budget_respected(env):
    rid = new_research(env, depth="quick")
    make_manager(env).start(rid)
    db = env.SF()
    assert db.query(Source).filter_by(session_id=rid).count() <= 5
    db.close()


def test_events_stream_to_bus_matches_stored_events(env):
    rid = new_research(env)
    bus = EventBus()
    q = bus.subscribe(rid)
    make_manager(env, bus=bus).start(rid)
    live = []
    while not q.empty():
        live.append(q.get_nowait())
    stored = [event_to_dict(e) for e in events(env, rid)]
    # parallel workers may publish in a different order than created_at; clients sort by created_at
    assert {e["id"] for e in live} == {e["id"] for e in stored}
    assert [e["id"] for e in sorted(live, key=lambda e: (e["created_at"], e["id"]))] == [
        e["id"] for e in stored
    ]
    assert live[-1]["event_type"] == "run_end" and stored[-1]["event_type"] == "run_end"


# --------------------------------------------------------------- conditional routing
def staged_hits(env_search):
    """First search returns one page; follow-up queries return two NEW pages."""
    inner = env_search.inner

    def hits_for(query):
        allh = inner.search(query, 10)
        if query.endswith("independent analysis"):
            return [h for h in allh if "comparison" in h.url or "limitations" in h.url]
        return [h for h in allh if "agentic-overview" in h.url][:1]

    return hits_for


def test_followup_research_runs_then_reanalyzes_when_new_evidence_found(env):
    search = InstrumentedSearch()
    search.hits_for = staged_hits(search)
    llm = CountingLLM()
    rid = new_research(env, depth="standard")
    make_manager(env, llm=llm, search=search).start(rid)
    followups = [q for q in search.queries if q.endswith("independent analysis")]
    assert len(followups) == 1  # one bounded follow-up round
    assert llm.calls["analysis"] == 2  # re-analysed because new evidence arrived
    assert row(env, rid).status == "completed"


def test_followup_without_new_evidence_skips_reanalysis_and_is_bounded(env):
    search = InstrumentedSearch()
    search.hits_for = lambda q: search.inner.search(q, 10)[:1]  # always the same single page
    llm = CountingLLM()
    rid = new_research(env, depth="quick")  # max_iterations == 1
    make_manager(env, llm=llm, search=search).start(rid)
    assert len([q for q in search.queries if q.endswith("independent analysis")]) == 1
    assert llm.calls["analysis"] == 1  # nothing new => no wasted second analysis
    r = row(env, rid)
    assert r.status == "completed" and any("thin" in w or "insufficient" in w for w in r.warnings)


@pytest.mark.parametrize("depth,max_rev", [("quick", 1), ("standard", 2)])
def test_revision_loop_is_bounded(env, depth, max_rev):
    llm = CountingLLM({"claim_review": unsupported_first_claim})
    rid = new_research(env, depth=depth)
    make_manager(env, llm=llm).start(rid)
    r = row(env, rid)
    assert llm.calls["report_writing"] == 1 + max_rev and llm.calls["claim_review"] == 1 + max_rev
    assert r.status == "completed" and r.revision_count == max_rev
    db = env.SF()
    rep = db.scalars(select(Report).where(Report.session_id == rid)).one()
    db.close()
    assert rep.validation["passed"] is False
    assert any("did not pass validation" in w for w in r.warnings)


def test_revision_fixes_unsupported_claim(env):
    flagged = {}

    def first_call_only(n, kw):
        if n > 1:
            return None  # delegate to the real keyword reviewer afterwards
        flagged["claim"] = re.search(r"^\[C1\] (.+)$", kw["user"], re.M).group(1)
        return unsupported_first_claim(n, kw)

    llm = CountingLLM({"claim_review": first_call_only})
    rid = new_research(env, depth="standard")
    make_manager(env, llm=llm).start(rid)
    r = row(env, rid)
    db = env.SF()
    rep = db.scalars(select(Report).where(Report.session_id == rid)).one()
    db.close()
    assert r.revision_count == 1 and rep.validation["passed"] is True
    plain = re.sub(r"\[S\d+\]", "", flagged["claim"]).strip()
    assert plain not in re.sub(r"\[\[S\d+\]\]\(#ref-S\d+\)", "", rep.markdown)  # claim was removed
    assert llm.calls["report_writing"] == 2 and llm.calls["claim_review"] == 2


def test_critic_llm_failure_degrades_to_deterministic_checks(env):
    def boom(n, kw):
        raise LLMError("reviewer down")

    rid = new_research(env)
    make_manager(env, llm=CountingLLM({"claim_review": boom})).start(rid)
    db = env.SF()
    rep = db.scalars(select(Report).where(Report.session_id == rid)).one()
    db.close()
    codes = [i["code"] for i in rep.validation["issues"]]
    assert row(env, rid).status == "completed" and "llm_validation_unavailable" in codes
    assert rep.validation["llm_judged"] is False


# ------------------------------------------------------------------- human in the loop
def test_hitl_pause_edit_approve_resume(env):
    rid = new_research(env, approval=True)
    search = InstrumentedSearch()
    mgr = make_manager(env, search=search)
    mgr.start(rid)
    assert row(env, rid).status == "awaiting_approval" and search.log == []  # paused, no research
    assert events(env, rid)[-1].payload["status"] == "awaiting_approval"
    with pytest.raises(AppError) as ei:
        mgr.start(rid)  # not approved yet
    assert ei.value.code == "plan_not_approved"

    db = env.SF()
    from app.models import ResearchSession

    research = db.get(ResearchSession, rid)
    draft = PlanDraft(
        objectives=["Only frameworks"],
        subtasks=[task("t1", queries=["orchestra-x overview focus"])],
        report_outline=["Summary", "Findings"],
    )
    planning.edit_plan(db, research, draft)
    planning.approve_plan(db, research)
    db.commit()
    db.close()

    mgr.start(rid)
    assert row(env, rid).status == "completed"
    assert search.queries == ["orchestra-x overview focus"]  # the EDITED plan was executed


# ----------------------------------------------------------- failure recovery / resume
def fail_once(n, kw):
    if n == 1:
        raise LLMError("provider rejected the request")
    return None


def test_failure_then_resume_does_not_repeat_completed_work(env):
    llm = CountingLLM({"analysis": fail_once})
    search = InstrumentedSearch()
    rid = new_research(env)
    mgr = make_manager(env, llm=llm, search=search)
    mgr.start(rid)
    r = row(env, rid)
    assert r.status == "failed" and "error" in r.error.lower()
    task_searches = list(search.queries)
    mgr.start(rid)  # resume
    assert row(env, rid).status == "completed"
    # the original task searches are not repeated; only a NEW follow-up round may be added
    assert max(Counter(search.queries).values()) == 1
    assert all(q in search.queries for q in task_searches)
    assert llm.calls["research_plan"] == 1
    assert llm.calls["analysis"] >= 2  # 1 failed + 1 ok (+1 if the follow-up found new evidence)
    types = [e.event_type for e in events(env, rid)]
    assert "run_resume" in types and types.count("run_start") == 1


def test_resume_survives_process_restart_via_sqlite_checkpoints(env, tmp_path):
    cp_path = str(tmp_path / "checkpoints.sqlite")
    llm1, search1 = CountingLLM({"analysis": fail_once}), InstrumentedSearch()
    rid = new_research(env)
    make_manager(env, llm=llm1, search=search1, checkpointer=make_checkpointer(cp_path)).start(rid)
    assert row(env, rid).status == "failed"
    # brand-new manager, new SQLite connection, new provider instances == process restart
    llm2, search2 = CountingLLM(), InstrumentedSearch()
    make_manager(env, llm=llm2, search=search2, checkpointer=make_checkpointer(cp_path)).start(rid)
    assert row(env, rid).status == "completed"
    # nothing redone: no planning, and no task query searched again (only a follow-up may run)
    assert all(q.endswith("independent analysis") for q in search2.queries)
    assert llm2.calls["research_plan"] == 0
    assert llm2.calls["analysis"] >= 1


def test_checkpointed_state_is_pure_json(env, tmp_path):
    cp = make_checkpointer(str(tmp_path / "cp.sqlite"))
    rid = new_research(env)
    mgr = make_manager(env, checkpointer=cp)
    mgr.start(rid)
    graph = build_graph(None, cp)
    values = graph.get_state({"configurable": {"thread_id": rid}}).values
    json.dumps(values)  # raises if any live object/dataclass leaked into state
    assert values["status"] == "completed" and values["sources"] and values["evidence"]


def test_no_evidence_fails_cleanly_then_restarts_fresh_on_resume(env):
    llm = CountingLLM()
    rid = new_research(env)
    mgr = make_manager(env, llm=llm, fetcher=FixtureFetcher({"pages": {}}))
    mgr.start(rid)
    r = row(env, rid)
    assert r.status == "failed" and "No evidence" in r.error
    db = env.SF()
    assert db.query(Report).filter_by(session_id=rid).count() == 0
    db.close()
    # fix the environment and resume: the finished thread restarts cleanly and re-uses the plan
    mgr2 = make_manager(env, llm=llm, checkpointer=mgr.checkpointer)
    mgr2.start(rid)
    assert row(env, rid).status == "completed" and llm.calls["research_plan"] == 1


def test_document_modes_fail_fast_with_clear_reasons_when_documents_are_unusable(env):
    rid = new_research(env, mode="document", document_ids=["no-such-doc"])
    make_manager(env).start(rid)  # harness has no document retriever configured
    r = row(env, rid)
    assert r.status == "failed" and "not configured" in r.error
    rid2 = new_research(env, mode="hybrid", document_ids=[])
    make_manager(env).start(rid2)
    assert "at least one" in row(env, rid2).error


# ------------------------------------------------------- cancellation / timeout / threads
def test_cancel_then_resume(env):
    gate, started = threading.Event(), threading.Event()
    search = InstrumentedSearch(gate=gate, started=started)
    rid = new_research(env)
    mgr = make_manager(env, search=search, executor="thread")
    mgr.start(rid)
    assert started.wait(5) and mgr.is_active(rid)
    with pytest.raises(AppError) as ei:
        mgr.start(rid)
    assert ei.value.code == "already_running"
    assert mgr.cancel(rid) is True
    gate.set()
    mgr.wait(rid)
    r = row(env, rid)
    assert r.status == "cancelled" and not mgr.is_active(rid)
    assert events(env, rid)[-1].payload["status"] == "cancelled"
    assert mgr.cancel(rid) is False
    mgr.start(rid)
    mgr.wait(rid)
    assert row(env, rid).status == "completed"
    mgr.shutdown()


def test_timeout_mid_run_then_resume_does_not_repeat_searches(env):
    cp = make_checkpointer(":memory:")
    baseline = InstrumentedSearch()
    make_manager(env, search=baseline).start(new_research(env))
    expected_searches = len(baseline.log)

    rid = new_research(env)
    slow = InstrumentedSearch(delay=0.4)
    make_manager(env, search=slow, timeout_override=0.2, checkpointer=cp).start(rid)
    r = row(env, rid)
    assert r.status == "failed" and "time budget" in r.error
    done_before = len(slow.log)
    assert 0 < done_before <= expected_searches  # finished work was kept, not discarded
    make_manager(env, search=slow, checkpointer=cp).start(rid)
    assert row(env, rid).status == "completed"
    assert max(Counter(slow.queries).values()) == 1  # no query was ever searched twice
    assert len(slow.log) == expected_searches


# ------------------------------------------------------------------ graph visualisation
def test_graph_doc_in_sync_and_has_all_nodes():
    doc = render_graph_doc()
    for n in ("validate_input", "plan", "approval_gate", "research_task", "ingest", "analyze",
              "write", "critic", "revise", "finalize"):  # fmt: skip
        assert n in doc
    assert "interrupt = before" in doc
    on_disk = (Path(__file__).resolve().parents[2] / "docs" / "graph.md").read_text(
        encoding="utf-8"
    )
    assert on_disk == doc, "docs/graph.md is stale: run python scripts/export_graph.py"


def test_restart_recovery_marks_stuck_runs_interrupted_and_they_can_resume(env):
    stuck = new_research(env)
    planning_stuck = new_research(env)
    done = new_research(env)
    from app.services import persistence

    persistence.set_status(env.SF, stuck, "running")
    persistence.set_status(env.SF, planning_stuck, "planning")
    persistence.set_status(env.SF, done, "completed")
    mgr = make_manager(env)
    assert mgr.recover_interrupted() == 2
    for rid in (stuck, planning_stuck):
        r = row(env, rid)
        assert r.status == "interrupted" and "restarted" in r.error
    assert row(env, done).status == "completed"  # finished runs are untouched
    mgr.start(stuck)  # an interrupted run is resumable
    assert row(env, stuck).status == "completed"


# ---------------------------------------------- guards that need an endless supply of pages
class EndlessWeb:
    """Search returns N brand-new URLs per call; fetcher serves a relevant page for any URL."""

    def __init__(self, per_call):
        self.per_call, self.n = per_call, 0
        self._lock = threading.Lock()

    def hits_for(self, query):
        from tests.helpers import make_hit

        with self._lock:
            hits = []
            for _ in range(self.per_call):
                self.n += 1
                hits.append(
                    make_hit(
                        f"https://many.test/p{self.n}",
                        title=f"Page {self.n}",
                        rank=len(hits) + 1,
                        query=query,
                    )
                )
            return hits

    def fetch(self, url):
        from app.tools.fetcher import FetchResult
        from tests.helpers import NOW

        body = (
            f"<p>Agentic AI frameworks developments overview comparison limitations risks research "
            f"evidence gaps are discussed in detail on page {url[-4:]} of this synthetic site.</p>"
        )
        return FetchResult(
            url,
            url,
            200,
            "text/html",
            f"<html><body>{body}</body></html>",
            False,
            NOW,
            provider="fixture",
        )


def test_source_budget_is_a_hard_cap_at_ingest(env):
    web = EndlessWeb(per_call=12)
    search = InstrumentedSearch(hits_for=web.hits_for)
    rid = new_research(env, depth="quick")  # budget: 5 sources
    make_manager(env, search=search, fetcher=web).start(rid)
    db = env.SF()
    n = db.query(Source).filter_by(session_id=rid).count()
    db.close()
    assert n == 5 and row(env, rid).status == "completed"
    assert any("source budget" in w for w in row(env, rid).warnings)


def always_wants_more(n, kw):
    first = re.search(r"^\[(E\d+)\]", kw["user"], re.M).group(1)
    return json.dumps(
        {
            "findings": [
                {
                    "statement": "Evidence is still thin on this topic.",
                    "evidence_ids": [first],
                    "type": "fact",
                    "confidence": "low",
                }
            ],
            "gaps": [
                {
                    "description": "need more evidence",
                    "follow_up_queries": [f"more evidence round {n}"],
                }
            ],
            "needs_more_research": True,
        }
    )


@pytest.mark.parametrize("depth,max_iter", [("quick", 1), ("standard", 2)])
def test_follow_up_iterations_are_capped_by_budget_not_by_luck(env, depth, max_iter):
    web = EndlessWeb(per_call=1)  # never runs out of new pages; source cap is far away
    search = InstrumentedSearch(hits_for=web.hits_for)
    llm = CountingLLM({"analysis": always_wants_more})
    rid = new_research(env, depth=depth)
    make_manager(env, llm=llm, search=search, fetcher=web).start(rid)
    followups = [q for q in search.queries if q.startswith("more evidence round")]
    assert len(followups) == max_iter and llm.calls["analysis"] == max_iter + 1
    assert row(env, rid).status == "completed"
    db = env.SF()
    assert (
        db.query(Source).filter_by(session_id=rid).count() < 12
    )  # the source cap was not what stopped it
    db.close()
