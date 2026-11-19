"""Graph nodes. Each node is a closure over a RunContext; all state in/out is JSON-safe.

Every node is wrapped by `traced`, which checks cancellation/deadline, emits start/end/error
events (stored + streamed), and times the node.
"""

import re
import time
from collections.abc import Callable
from functools import wraps

from langgraph.types import Send

from app.agents.analyst import AnalysisError, AnalystAgent
from app.agents.critic import CriticAgent
from app.agents.document import DocumentAgent
from app.agents.reader import ReaderAgent
from app.agents.search import SearchAgent
from app.agents.writer import WriterAgent, WriterError
from app.errors import AppError
from app.graph import serde
from app.graph.context import RunCancelled, RunContext, WorkflowTimeout
from app.graph.state import ResearchState
from app.models import ResearchSession
from app.schemas.evidence import ReportDraft
from app.services import persistence, planning
from app.services.registry import SourceRegistry
from app.tools.urls import url_key

END = "__end__"


def _task_order(task_id: str) -> int:
    return int(re.sub(r"\D", "", task_id) or 0)


def _notes(ctx: RunContext, sources: dict, analysis: dict) -> list[str]:
    notes = []
    if any(s["provider"] == "fixture" for s in sources.values()):
        notes.append(
            "OFFLINE FIXTURE DATA: sources below are synthetic test data, not live web results."
        )
    if ctx.llm.name == "fake":
        notes.append("Written by the offline templated provider, not a language model.")
    if analysis.get("needs_more_research"):
        notes.append("Evidence was limited; treat conclusions with caution.")
    return notes


def make_nodes(ctx: RunContext | None) -> dict[str, Callable]:
    """Build node functions. `ctx` may be None when only the graph structure is needed."""

    def traced(name: str, agent: str):
        def deco(fn):
            @wraps(fn)
            def wrapper(state):
                ctx.check()
                ctx.emit(agent, "node_start", f"{name} started")
                t0 = time.perf_counter()
                try:
                    update, summary = fn(state)
                except (RunCancelled, WorkflowTimeout):
                    raise
                except Exception as exc:
                    ms = int((time.perf_counter() - t0) * 1000)
                    ctx.emit(agent, "error", f"{name} failed: {type(exc).__name__}", None, ms)
                    raise
                ms = int((time.perf_counter() - t0) * 1000)
                ctx.emit(agent, "node_end", summary, None, ms)
                return update

            return wrapper

        return deco

    def _usage(usage) -> None:
        s = ctx.settings
        persistence.add_usage(
            ctx.session_factory,
            ctx.research_id,
            usage,
            s.llm_input_cost_per_mtok,
            s.llm_output_cost_per_mtok,
        )

    def _budget(state) -> dict:
        return state["plan"]["budget"]

    def _web_count(sources: dict) -> int:
        return sum(1 for s in sources.values() if s.get("kind", "web") == "web")

    # ------------------------------------------------------------------ nodes
    @traced("validate_input", "system")
    def validate_input(state: ResearchState):
        if not state.get("query", "").strip():
            return {
                "status": "failed",
                "error": "Research query is empty.",
            }, "rejected: empty query"
        if state["mode"] in ("document", "hybrid"):
            ids = state.get("document_ids") or []
            problem = None
            if not ids:
                problem = "Select at least one uploaded document for document research."
            elif ctx.documents is None:
                problem = "Document retrieval is not configured on this server."
            else:
                missing, not_ready = persistence.check_documents(ctx.session_factory, ids)
                if missing:
                    problem = f"Unknown document(s): {missing}"
                elif not_ready:
                    problem = f"Document(s) are not ready (still processing or failed): {not_ready}"
            if problem:
                return {"status": "failed", "error": problem}, f"rejected: {problem}"
        persistence.set_status(ctx.session_factory, ctx.research_id, "running")
        return {"status": "running", "warnings": []}, f"input valid (mode={state['mode']})"

    @traced("plan", "supervisor")
    def plan(state: ResearchState):
        info = persistence.load_session(ctx.session_factory, ctx.research_id)
        if info and info["plan"]:
            plan_d, approved, how = info["plan"], info["plan_approved"], "reused existing plan"
        else:
            db = ctx.session_factory()
            try:
                research = db.get(ResearchSession, ctx.research_id)
                try:
                    planning.generate_plan(
                        db, research, ctx.llm, ctx.settings, check_state=False, emit_events=False
                    )
                    db.commit()
                except AppError as exc:  # planning service already persisted the failure
                    return {
                        "status": "failed",
                        "error": exc.message,
                    }, f"planning failed: {exc.message}"
            finally:
                db.close()
            info = persistence.load_session(ctx.session_factory, ctx.research_id)
            plan_d, approved, how = info["plan"], info["plan_approved"], "plan generated"
        needs_gate = state["require_approval"] and not approved
        if not needs_gate:
            persistence.set_status(ctx.session_factory, ctx.research_id, "running")
        n = len(plan_d["subtasks"])
        return (
            {"plan": plan_d, "plan_approved": approved},
            f"{how}: {n} subtasks" + ("; awaiting human approval" if needs_gate else ""),
        )

    @traced("approval_gate", "supervisor")
    def approval_gate(state: ResearchState):
        persistence.set_status(ctx.session_factory, ctx.research_id, "running")
        return {"plan_approved": True}, "plan approved by user; research starting"

    @traced("dispatch", "supervisor")
    def dispatch(state: ResearchState):
        n = len(state["plan"]["subtasks"])
        return {"read_batches": {"__reset__": True}}, f"scheduling {n} subtasks in dependency waves"

    def next_wave(state: ResearchState):
        """Path function: next wave of ready subtasks as parallel Sends, else move on."""
        done = set(state.get("done_task_ids", []))
        pending = [t for t in state["plan"]["subtasks"] if t["id"] not in done]
        if not pending:
            if state.get("followup_active") and state.get("new_evidence", 0) == 0:
                return "write"  # follow-up found nothing new: don't re-analyse the same evidence
            return "analyze"
        ready = [t for t in pending if set(t["depends_on"]) <= done] or pending
        remaining = max(1, _budget(state)["max_sources"] - _web_count(state.get("sources", {})))
        # Parallel workers cannot see each other's results, so they often pick the same URLs.
        # Oversample 2x per worker; the hard source cap is enforced deterministically at ingest.
        per = min(remaining, 2 * -(-remaining // len(ready)))
        keys = sorted(url_key(s["url"]) for s in state.get("sources", {}).values())
        return [
            Send(
                "research_task",
                {
                    "task": t,
                    "max_sources": per,
                    "kind": "task",
                    "exclude_keys": keys,
                    "document_ids": state.get("document_ids", []),
                },
            )
            for t in ready
        ]

    @traced("research_task", "search")
    def research_task(payload: dict):
        task = payload["task"]
        s = ctx.settings
        searcher = SearchAgent(
            ctx.search, results_per_query=s.search_max_results_per_query, sleep=ctx.sleep
        )
        warnings: list[str] = []
        if task["agent"] == "document":
            if ctx.documents is None:
                msg = f"skipped {task['id']}: document retrieval is not configured"
                batch = {
                    "kind": payload["kind"],
                    "task_id": task["id"],
                    "results": [],
                    "warnings": [msg],
                }
                return {"read_batches": [batch], "warnings": [msg]}, msg
            agent = DocumentAgent(
                ctx.documents,
                ctx.llm,
                max_chars=s.reader_max_chars,
                transport_retries=s.llm_max_retries,
            )
            focus = f"{task['description']} (queries: {'; '.join(task['search_queries'])})"
            got = agent.gather(task["search_queries"], payload.get("document_ids", []), focus)
            _usage(got.usage)
            ctx.emit(
                "document",
                "tool_call",
                f"documents[{task['id']}]: {len(got.results)} document(s) yielded "
                f"{sum(len(r['items']) for r in got.results)} verified evidence items with pages",
            )
            batch = {
                "kind": payload["kind"],
                "task_id": task["id"],
                "results": got.results,
                "warnings": got.warnings,
            }
            return {"read_batches": [batch], "warnings": got.warnings}, (
                f"task {task['id']}: {len(got.results)} document(s) searched"
            )
        out = searcher.run(
            task["search_queries"],
            max_sources=payload["max_sources"],
            exclude_keys=frozenset(payload["exclude_keys"]),
        )
        ctx.emit(
            "search",
            "tool_call",
            f"search[{task['id']}]: {out.queries_run} queries -> {len(out.hits)} new results"
            + (f", {len(out.errors)} errors" if out.errors else ""),
            {"provider": ctx.search.name, "queries": task["search_queries"]},
        )
        warnings += [f"search error: {e}" for e in out.errors]
        ctx.check_cancelled()
        reader = ReaderAgent(
            ctx.fetcher,
            ctx.llm,
            max_chars=s.reader_max_chars,
            max_workers=s.reader_max_workers,
            transport_retries=s.llm_max_retries,
        )
        focus = f"{task['description']} (queries: {'; '.join(task['search_queries'])})"
        results = reader.read_many(out.hits, focus)
        ok = sum(1 for r in results if r.status == "read")
        for r in results:
            _usage(r.usage)
            if r.status == "failed":
                warnings.append(f"could not read {r.hit.url}: {r.error}")
            if r.injection_suspected:
                warnings.append(f"possible prompt injection in {r.hit.url} (treated as data)")
        ctx.emit(
            "reader",
            "tool_call",
            f"read[{task['id']}]: {ok}/{len(results)} pages read, "
            f"{sum(len(r.items) for r in results)} verified evidence items",
        )
        batch = {
            "kind": payload["kind"],
            "task_id": task["id"],
            "results": [serde.read_result_to_dict(r) for r in results],
            "warnings": warnings,
        }
        return {"read_batches": [batch]}, f"task {task['id']}: {ok}/{len(results)} pages read"

    @traced("ingest", "supervisor")
    def ingest(state: ResearchState):
        reg = SourceRegistry.restore(state.get("sources", {}), state.get("evidence", []))
        max_sources = _budget(state)["max_sources"]
        before = len(reg.evidence)
        done_new: list[str] = []
        warns: list[str] = []
        batches = sorted(
            state.get("read_batches", []), key=lambda b: (b["kind"], _task_order(b["task_id"]))
        )
        dropped = 0
        for b in batches:
            warns += b["warnings"]
            if b["kind"] == "task":
                done_new.append(b["task_id"])
            for r in sorted(b["results"], key=lambda x: (x["rank"], x["url"])):
                if reg.has_url(r["url"]):
                    continue
                is_doc = r.get("kind") == "document"
                if (
                    not is_doc
                    and _web_count({k: v.model_dump() for k, v in reg.sources.items()})
                    >= max_sources
                ):
                    dropped += 1
                    continue
                src = reg.add_source(
                    kind=r.get("kind", "web"),
                    title=r["title"],
                    url=r["url"],
                    published_at=serde.from_iso(r["published_at"]),
                    retrieved_at=serde.from_iso(r["retrieved_at"]),
                    provider=r["provider"],
                    status=r["status"],
                    error=r["error"],
                    injection_suspected=r["injection_suspected"],
                )
                for it in serde.items_from_dict(r):
                    reg.add_evidence(
                        src.ref,
                        passage=it.passage,
                        claim=it.claim,
                        kind=it.kind,
                        relevance=it.relevance,
                        page_number=it.page_number,
                    )
        if dropped:
            warns.append(f"source budget reached: {dropped} additional result(s) were not used")
        done = set(state.get("done_task_ids", [])) | set(done_new)
        if _web_count({k: v.model_dump() for k, v in reg.sources.items()}) >= max_sources:
            leftover = [t["id"] for t in state["plan"]["subtasks"] if t["id"] not in done]
            if leftover:
                warns.append(f"source budget exhausted; skipped subtasks {leftover}")
                done |= set(leftover)
        new_evidence = len(reg.evidence) - before
        persistence.persist_registry(
            ctx.session_factory, ctx.research_id, reg.sources, reg.evidence
        )
        persistence.mark_tasks(ctx.session_factory, ctx.research_id, done)
        update = {
            "sources": {ref: s.model_dump(mode="json") for ref, s in reg.sources.items()},
            "evidence": [e.model_dump() for e in reg.evidence],
            "new_evidence": new_evidence,
            "done_task_ids": sorted(done, key=_task_order),
            "round": state.get("round", 0) + 1,
            "read_batches": {"__reset__": True},
            "warnings": warns,
        }
        return update, (
            f"ingested {len(batches)} batch(es): {len(reg.sources)} sources, "
            f"{len(reg.evidence)} evidence items (+{new_evidence})"
        )

    @traced("analyze", "analyst")
    def analyze(state: ResearchState):
        evidence = serde.evidence_from_state(state.get("evidence", []))
        sources = serde.sources_from_state(state.get("sources", {}))
        if not evidence:
            return (
                {"status": "failed", "error": "No evidence could be extracted from any source."},
                "no evidence to analyze",
            )
        analyst = AnalystAgent(ctx.llm, transport_retries=ctx.settings.llm_max_retries)
        try:
            res = analyst.analyze(state["query"], state["plan"]["objectives"], evidence, sources)
        except AnalysisError as exc:
            _usage(exc.usage)
            return (
                {"status": "failed", "error": "The analyst could not produce a valid analysis."},
                "analysis failed",
            )
        _usage(res.usage)
        b = _budget(state)
        want = bool(
            res.needs_more_research
            and res.follow_up_queries
            and state.get("iterations", 0) < b["max_iterations"]
            and _web_count(state.get("sources", {})) < b["max_sources"]
        )
        update = {
            "analysis": serde.analysis_to_dict(res),
            "followup_active": want,
            "followup_queries": res.follow_up_queries if want else [],
            "iterations": state.get("iterations", 0) + (1 if want else 0),
            "warnings": res.warnings,
        }
        return update, (
            f"{len(res.findings)} findings, {len(res.contradictions)} contradictions, "
            f"{len(res.gaps)} gaps; "
            + ("follow-up research scheduled" if want else "evidence sufficient or budget reached")
        )

    def after_analysis(state: ResearchState):
        if state.get("status") == "failed":
            return END
        if state.get("followup_queries"):
            remaining = max(1, _budget(state)["max_sources"] - _web_count(state.get("sources", {})))
            keys = sorted(url_key(s["url"]) for s in state.get("sources", {}).values())
            n = state.get("iterations", 1)
            base = {
                "id": f"f{n}",
                "description": "Follow-up research on evidence gaps",
                "search_queries": state["followup_queries"],
                "depends_on": [],
            }
            mode = state["mode"]
            tasks = []
            if mode in ("web", "hybrid"):
                tasks.append({**base, "agent": "search"})
            if mode in ("document", "hybrid"):  # never reach for the web in document-only mode
                tasks.append({**base, "id": f"f{n}d", "agent": "document"})
            return [
                Send(
                    "research_task",
                    {
                        "task": t,
                        "max_sources": remaining,
                        "kind": "followup",
                        "exclude_keys": keys,
                        "document_ids": state.get("document_ids", []),
                    },
                )
                for t in tasks
            ]
        return "write"

    def _write(state: ResearchState, feedback: list[str] | None):
        evidence = serde.evidence_from_state(state["evidence"])
        sources = serde.sources_from_state(state["sources"])
        analysis = serde.analysis_from_dict(state["analysis"])
        writer = WriterAgent(ctx.llm, transport_retries=ctx.settings.llm_max_retries)
        res = writer.write(
            state["query"],
            state["plan"]["report_outline"],
            analysis,
            evidence,
            sources,
            notes=_notes(ctx, state["sources"], state["analysis"]),
            feedback=feedback,
        )
        _usage(res.usage)
        return {
            "draft": res.draft.model_dump(),
            "markdown": res.markdown,
            "cited": res.cited,
            "attempts": res.attempts,
        }

    @traced("write", "writer")
    def write(state: ResearchState):
        try:
            report = _write(state, None)
        except WriterError as exc:
            _usage(exc.usage)
            return (
                {
                    "status": "failed",
                    "error": "The writer could not produce a report that passes citation checks.",
                },
                "writing failed",
            )
        return {"report": report, "feedback": []}, (
            f"draft written: {len(report['draft']['sections'])} sections, "
            f"{len(report['cited'])} sources cited (attempts={report['attempts']})"
        )

    def after_write(state: ResearchState):
        return END if state.get("status") == "failed" else "critic"

    @traced("critic", "critic")
    def critic(state: ResearchState):
        agent = CriticAgent(ctx.llm, transport_retries=ctx.settings.llm_max_retries)
        res = agent.review(
            draft=ReportDraft(**state["report"]["draft"]),
            analysis=serde.analysis_from_dict(state["analysis"]),
            evidence=serde.evidence_from_state(state["evidence"]),
            sources=serde.sources_from_state(state["sources"]),
            depth=state["depth"],
        )
        _usage(res.usage)
        rep = res.report
        update = {"validation": rep.model_dump(), "feedback": res.feedback}
        return update, (
            f"validation {'passed' if rep.passed else 'FAILED'}: {len(rep.errors)} errors, "
            f"{len(rep.warnings)} warnings, {rep.claims_checked} claims reviewed"
        )

    def after_critic(state: ResearchState):
        if state["validation"]["passed"]:
            return "finalize"
        if state.get("revisions", 0) < _budget(state)["max_revisions"] and state.get("feedback"):
            return "revise"
        return "finalize"

    @traced("revise", "writer")
    def revise(state: ResearchState):
        n = state.get("revisions", 0) + 1
        try:
            report = _write(state, state.get("feedback"))
        except WriterError as exc:
            _usage(exc.usage)
            return (
                {
                    "revisions": n,
                    "revise_failed": True,
                    "warnings": ["revision attempt failed; keeping the previous draft"],
                },
                f"revision {n} failed; keeping previous draft",
            )
        return {"report": report, "revisions": n, "revise_failed": False}, (
            f"revision {n} written ({len(state.get('feedback') or [])} feedback items addressed)"
        )

    def after_revise(state: ResearchState):
        return "finalize" if state.get("revise_failed") else "critic"

    @traced("finalize", "supervisor")
    def finalize(state: ResearchState):
        warns = list(dict.fromkeys(state.get("warnings", [])))
        v = state["validation"]
        if not v["passed"]:
            warns.append("report did not pass validation within the revision budget; see issues")
        persistence.save_report(
            ctx.session_factory,
            ctx.research_id,
            state["report"]["markdown"],
            v,
            state.get("revisions", 0),
            warns,
        )
        return {"status": "completed"}, (
            f"report saved (validation {'passed' if v['passed'] else 'failed'}, "
            f"{state.get('revisions', 0)} revision(s))"
        )

    return {
        "validate_input": validate_input,
        "plan": plan,
        "approval_gate": approval_gate,
        "dispatch": dispatch,
        "research_task": research_task,
        "ingest": ingest,
        "analyze": analyze,
        "write": write,
        "critic": critic,
        "revise": revise,
        "finalize": finalize,
        "next_wave": next_wave,
        "after_analysis": after_analysis,
        "after_write": after_write,
        "after_critic": after_critic,
        "after_revise": after_revise,
    }
