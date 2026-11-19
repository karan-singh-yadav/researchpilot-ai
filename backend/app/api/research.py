"""Research session endpoints (persistence, lifecycle, HITL plan review, citations, SSE)."""

import json
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_bus, get_llm, get_run_manager
from app.config import Settings, get_settings
from app.database import get_db
from app.errors import AppError
from app.mcp.errors import ToolFailure
from app.mcp.handlers import build_export
from app.models import AgentEvent, Report, ResearchSession, Source
from app.schemas.plan import (
    PlanApproveRequest,
    PlanDraft,
    PlanOut,
    PlanPatch,
    PlanRejectRequest,
    ResearchPlan,
)
from app.schemas.research import (
    AgentEventOut,
    Page,
    ResearchCreate,
    ResearchOut,
    ResearchStatusOut,
)
from app.schemas.results import ReportCitationsOut, ReportOut, SourceOut
from app.security.auth import require_api_key
from app.security.rate_limit import rate_limit
from app.services import planning
from app.services.citations import build_citation_index
from app.services.events import TERMINAL, EventBus, event_to_dict
from app.services.llm import LLMProvider
from app.services.runner import RunManager

router = APIRouter(
    prefix="/api/research",
    tags=["research"],
    dependencies=[Depends(rate_limit), Depends(require_api_key)],
)


def _get_or_404(db: Session, research_id: str) -> ResearchSession:
    obj = db.get(ResearchSession, research_id)
    if obj is None:
        raise AppError(404, "not_found", "Research session not found.")
    return obj


@router.post("", response_model=ResearchOut, status_code=201)
def create_research(
    payload: ResearchCreate,
    db: Session = Depends(get_db),
    mgr: RunManager = Depends(get_run_manager),
) -> ResearchSession:
    if payload.mode.value in ("document", "hybrid") and not payload.document_ids:
        raise AppError(
            422, "documents_required", "document and hybrid modes need at least one document_id."
        )
    if payload.document_ids:
        from app.services.persistence import check_documents

        missing, not_ready = check_documents(lambda: db, payload.document_ids)
        if missing:
            raise AppError(422, "unknown_document", f"Unknown document id(s): {missing}")
        if not_ready:
            raise AppError(
                422, "document_not_ready", f"Document(s) still processing or failed: {not_ready}"
            )
    if payload.start:
        mgr.preflight()  # fail with 503 before creating anything if providers are not configured
    session = ResearchSession(
        query=payload.query,
        mode=payload.mode.value,
        depth=payload.depth.value,
        status="pending",
        require_plan_approval=payload.require_plan_approval,
        document_ids=payload.document_ids,
    )
    db.add(session)
    db.flush()
    if payload.start:
        db.commit()  # the background run reads this row from its own connection
        mgr.start(session.id)
        db.refresh(session)
    return session


@router.get("", response_model=Page[ResearchOut])
def list_research(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None, max_length=200, description="Substring match on the query"),
    db: Session = Depends(get_db),
) -> dict:
    stmt = select(ResearchSession)
    count_stmt = select(func.count()).select_from(ResearchSession)
    if q:
        like = f"%{q.replace('%', '').replace('_', '')}%"
        stmt = stmt.where(ResearchSession.query.ilike(like))
        count_stmt = count_stmt.where(ResearchSession.query.ilike(like))
    items = db.scalars(
        stmt.order_by(ResearchSession.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return {"items": items, "total": db.scalar(count_stmt) or 0, "limit": limit, "offset": offset}


@router.get("/{research_id}", response_model=ResearchOut)
def get_research(research_id: str, db: Session = Depends(get_db)) -> ResearchSession:
    return _get_or_404(db, research_id)


@router.get("/{research_id}/status", response_model=ResearchStatusOut)
def get_status(research_id: str, db: Session = Depends(get_db)) -> ResearchSession:
    return _get_or_404(db, research_id)


@router.delete("/{research_id}", status_code=204)
def delete_research(
    research_id: str,
    db: Session = Depends(get_db),
    mgr: RunManager = Depends(get_run_manager),
) -> Response:
    research = _get_or_404(db, research_id)
    if mgr.is_active(research_id):
        raise AppError(409, "run_in_progress", "Cancel the running research before deleting it.")
    db.delete(research)
    mgr.delete_thread(research_id)
    return Response(status_code=204)


def _plan_out(r: ResearchSession) -> PlanOut:
    return PlanOut(
        research_id=r.id,
        status=r.status,
        requires_approval=r.require_plan_approval,
        plan_approved=r.plan_approved,
        plan=ResearchPlan.model_validate(r.plan) if r.plan else None,
    )


@router.post("/{research_id}/plan", response_model=PlanOut)
def create_plan(
    research_id: str,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> PlanOut:
    """Run the Supervisor agent to (re)generate the research plan."""
    research = _get_or_404(db, research_id)
    planning.generate_plan(db, research, llm, settings)
    return _plan_out(research)


@router.get("/{research_id}/plan", response_model=PlanOut)
def get_plan(research_id: str, db: Session = Depends(get_db)) -> PlanOut:
    research = _get_or_404(db, research_id)
    if not research.plan:
        raise AppError(404, "plan_not_found", "No plan has been generated yet.")
    return _plan_out(research)


@router.put("/{research_id}/plan", response_model=PlanOut)
def edit_plan(research_id: str, draft: PlanDraft, db: Session = Depends(get_db)) -> PlanOut:
    """Human-in-the-loop: replace the plan. Budgets and mode rules are re-enforced server-side."""
    research = _get_or_404(db, research_id)
    planning.edit_plan(db, research, draft)
    return _plan_out(research)


@router.patch("/{research_id}/plan", response_model=PlanOut)
def patch_plan(research_id: str, patch: PlanPatch, db: Session = Depends(get_db)) -> PlanOut:
    """Human-in-the-loop: partially update objectives, subtasks, or report outline."""
    research = _get_or_404(db, research_id)
    planning.patch_plan(db, research, patch)
    return _plan_out(research)


@router.post("/{research_id}/plan/reject", response_model=PlanOut)
def reject_plan(
    research_id: str,
    body: PlanRejectRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> PlanOut:
    """Human-in-the-loop: reject the current plan with feedback and regenerate it."""
    research = _get_or_404(db, research_id)
    planning.reject_plan(db, research, body.feedback, llm, settings)
    return _plan_out(research)


@router.post("/{research_id}/plan/approve", response_model=PlanOut)
def approve_plan(
    research_id: str,
    body: PlanApproveRequest | None = None,
    auto_resume: bool = Query(False, description="Immediately resume execution after approval"),
    db: Session = Depends(get_db),
    mgr: RunManager = Depends(get_run_manager),
) -> PlanOut:
    research = _get_or_404(db, research_id)
    planning.approve_plan(db, research, note=body.note if body else None)
    if auto_resume:
        db.commit()
        mgr.start(research_id)
        db.refresh(research)
    return _plan_out(research)


@router.get("/{research_id}/trace", response_model=Page[AgentEventOut])
def get_trace(
    research_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    _get_or_404(db, research_id)
    base = AgentEvent.session_id == research_id
    items = db.scalars(
        select(AgentEvent)
        .where(base)
        .order_by(AgentEvent.created_at, AgentEvent.id)
        .limit(limit)
        .offset(offset)
    ).all()
    total = db.scalar(select(func.count()).select_from(AgentEvent).where(base)) or 0
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.post("/{research_id}/resume", response_model=ResearchStatusOut, status_code=202)
def resume_research(
    research_id: str,
    db: Session = Depends(get_db),
    mgr: RunManager = Depends(get_run_manager),
) -> ResearchSession:
    """Start the workflow, or continue it from its last checkpoint (after failure, cancel,
    restart, or human plan approval)."""
    research = _get_or_404(db, research_id)
    db.commit()
    mgr.start(research_id)
    db.refresh(research)
    return research


@router.post("/{research_id}/cancel", status_code=202)
def cancel_research(
    research_id: str,
    db: Session = Depends(get_db),
    mgr: RunManager = Depends(get_run_manager),
) -> dict:
    _get_or_404(db, research_id)
    if not mgr.cancel(research_id):
        raise AppError(409, "not_running", "There is no active run to cancel.")
    return {"id": research_id, "status": "cancelling"}


@router.get("/{research_id}/report", response_model=ReportOut)
def get_report(research_id: str, db: Session = Depends(get_db)) -> Report:
    _get_or_404(db, research_id)
    rep = db.scalars(
        select(Report).where(Report.session_id == research_id).order_by(Report.version.desc())
    ).first()
    if rep is None:
        raise AppError(404, "report_not_found", "No report has been generated yet.")
    return rep


@router.get("/{research_id}/reports", response_model=list[ReportOut])
def list_reports(research_id: str, db: Session = Depends(get_db)) -> list[Report]:
    _get_or_404(db, research_id)
    return list(
        db.scalars(
            select(Report).where(Report.session_id == research_id).order_by(Report.version.desc())
        ).all()
    )


@router.get("/{research_id}/citations", response_model=ReportCitationsOut)
def get_report_citations(research_id: str, db: Session = Depends(get_db)) -> ReportCitationsOut:
    """Structured inline citation spans mapped to sources, page numbers, and verbatim evidence."""
    _get_or_404(db, research_id)
    rep = db.scalars(
        select(Report).where(Report.session_id == research_id).order_by(Report.version.desc())
    ).first()
    if rep is None:
        raise AppError(404, "report_not_found", "No report has been generated yet.")
    sources = list(db.scalars(select(Source).where(Source.session_id == research_id)).all())
    index = build_citation_index(rep.markdown, sources)
    return ReportCitationsOut(research_id=research_id, report_version=rep.version, **index)


@router.get("/{research_id}/sources", response_model=list[SourceOut])
def get_sources(research_id: str, db: Session = Depends(get_db)) -> list[SourceOut]:
    _get_or_404(db, research_id)
    rows = db.scalars(select(Source).where(Source.session_id == research_id)).all()
    rows = sorted(rows, key=lambda r: int(r.source_ref[1:]))
    return [SourceOut.model_validate({**_src(r), "evidence": list(r.evidence)}) for r in rows]


def _src(r: Source) -> dict:
    return {
        k: getattr(r, k)
        for k in (
            "source_ref", "kind", "title", "url", "document_id", "published_at",
            "retrieved_at", "provider", "status", "error", "injection_suspected",
        )
    }  # fmt: skip


@router.get("/{research_id}/events")
def stream_events(
    research_id: str,
    db: Session = Depends(get_db),
    bus: EventBus = Depends(get_bus),
    mgr: RunManager = Depends(get_run_manager),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Server-Sent Events: replays stored events, then streams live ones until the run ends.

    Note: browsers' EventSource cannot send the X-API-Key header; the frontend reads this stream
    with fetch() instead.
    """
    _get_or_404(db, research_id)
    q = bus.subscribe(research_id)  # subscribe BEFORE replay so nothing is missed
    stored = db.scalars(
        select(AgentEvent)
        .where(AgentEvent.session_id == research_id)
        .order_by(AgentEvent.created_at, AgentEvent.id)
    ).all()
    replay = [event_to_dict(e) for e in stored]
    keepalive = settings.sse_keepalive_seconds

    def frame(ev: dict) -> str:
        return f"id: {ev['id']}\nevent: {ev['event_type']}\ndata: {json.dumps(ev)}\n\n"

    def gen():
        import queue as _q

        seen: set[str] = set()
        try:
            for ev in replay:
                seen.add(ev["id"])
                yield frame(ev)
            if replay and replay[-1]["event_type"] == TERMINAL and not mgr.is_active(research_id):
                return
            while True:
                try:
                    ev = q.get(timeout=keepalive)
                except _q.Empty:
                    if not mgr.is_active(research_id):
                        yield 'event: stream_end\ndata: {"reason": "no active run"}\n\n'
                        return
                    yield ": keepalive\n\n"
                    continue
                if ev["id"] in seen:
                    continue
                seen.add(ev["id"])
                yield frame(ev)
                if ev["event_type"] == TERMINAL:
                    return
        finally:
            bus.unsubscribe(research_id, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{research_id}/export")
def export_report(
    research_id: str,
    format: Literal["markdown", "json"] = Query("markdown"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Download the report as a Markdown or JSON file (same export code the MCP tool uses)."""
    _get_or_404(db, research_id)
    try:
        out = build_export(db, research_id, format, settings.mcp_max_export_bytes)
    except ToolFailure as exc:
        status = {"not_found": 404, "too_large": 413}.get(exc.code, 422)
        raise AppError(status, exc.code, exc.message) from None
    media = "text/markdown; charset=utf-8" if format == "markdown" else "application/json"
    return Response(
        out.content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{out.filename}"'},
    )
