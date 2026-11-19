"""Plan lifecycle: generate (Supervisor), human edit/patch, reject/replan, approve.

Persists tasks + trace events for complete HITL auditability.
"""

import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.budgets import budget_for
from app.agents.supervisor import (
    SupervisorAgent,
    SupervisorError,
    check_plan_against_context,
)
from app.config import Settings
from app.errors import AppError
from app.logging_config import get_logger
from app.models import AgentEvent, ResearchSession, ResearchTask, UploadedDocument
from app.schemas.plan import PlanDraft, PlanPatch, ResearchPlan
from app.services.llm import LLMError, LLMProvider, LLMUsage
from app.services.llm.pricing import estimate_cost_usd

log = get_logger(__name__)

PLANNABLE = {"pending", "failed", "planned", "awaiting_approval"}
EDITABLE = {"planned", "awaiting_approval"}


def _names(db: Session, ids: list[str]) -> list[str]:
    if not ids:
        return []
    rows = {
        d.id: d.filename
        for d in db.scalars(select(UploadedDocument).where(UploadedDocument.id.in_(ids)))
    }
    return [rows.get(i, i) for i in ids]


def _event(
    db: Session,
    research: ResearchSession,
    event_type: str,
    summary: str,
    payload: dict | None = None,
    duration_ms: int | None = None,
) -> None:
    db.add(
        AgentEvent(
            session_id=research.id,
            agent="supervisor",
            event_type=event_type,
            summary=summary,  # concise operational summary; never chain-of-thought
            payload=payload,
            duration_ms=duration_ms,
        )
    )


def _account(research: ResearchSession, usage: LLMUsage, settings: Settings) -> None:
    research.token_usage += usage.total
    research.estimated_cost_usd += estimate_cost_usd(
        usage, settings.llm_input_cost_per_mtok, settings.llm_output_cost_per_mtok
    )


def _store_plan(db: Session, research: ResearchSession, plan: ResearchPlan) -> None:
    research.plan = plan.model_dump(mode="json")
    research.tasks.clear()
    db.flush()
    for st in plan.subtasks:
        research.tasks.append(
            ResearchTask(
                description=st.description,
                assigned_agent=st.agent,
                status="pending",
                result={
                    "plan_task_id": st.id,
                    "search_queries": st.search_queries,
                    "depends_on": st.depends_on,
                },
            )
        )
    research.plan_approved = not research.require_plan_approval
    research.status = "awaiting_approval" if research.require_plan_approval else "planned"
    research.error = None


def generate_plan(
    db: Session,
    research: ResearchSession,
    llm: LLMProvider,
    settings: Settings,
    *,
    check_state: bool = True,
    emit_events: bool = True,
    feedback: str | None = None,
) -> ResearchPlan:
    if check_state and research.status not in PLANNABLE:
        raise AppError(409, "invalid_state", f"Cannot plan a session in state '{research.status}'.")
    research.status = "planning"
    db.flush()
    if emit_events:
        _event(
            db,
            research,
            "node_start",
            "Planning started" + (" (with human feedback)" if feedback else ""),
            {"mode": research.mode, "has_feedback": bool(feedback)},
        )
    started = time.perf_counter()
    agent = SupervisorAgent(llm, transport_retries=settings.llm_max_retries)
    effective_query = (
        f"{research.query}\n\nHuman guidance for plan revision: {feedback.strip()}"
        if feedback and feedback.strip()
        else research.query
    )
    try:
        result = agent.plan(
            effective_query,
            research.mode,
            research.depth,
            documents=_names(db, research.document_ids or []),
        )
    except (SupervisorError, LLMError) as exc:
        ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(exc, "usage", None)
        if usage:
            _account(research, usage, settings)
        log.warning(
            "planning_failed research=%s error=%s detail=%s",
            research.id,
            type(exc).__name__,
            getattr(exc, "last_error", "") or str(exc),
        )
        research.status = "failed"
        research.error = (
            str(exc) if isinstance(exc, SupervisorError) else "The LLM provider request failed."
        )
        if emit_events:
            _event(db, research, "error", f"Planning failed: {research.error}", None, ms)
        db.commit()  # persist the failure before the request errors out
        raise AppError(502, "planning_failed", research.error) from exc

    ms = int((time.perf_counter() - started) * 1000)
    _account(research, result.usage, settings)
    _store_plan(db, research, result.plan)
    if emit_events:
        _event(
            db,
            research,
            "node_end",
            f"Plan created: {len(result.plan.subtasks)} subtasks, "
            f"{len(result.plan.objectives)} objectives "
            f"(attempts={result.attempts}, tokens={result.usage.total}, "
            f"generated_by={result.plan.generated_by})",
            {"attempts": result.attempts, "tokens": result.usage.total},
            ms,
        )
    return result.plan


def edit_plan(db: Session, research: ResearchSession, draft: PlanDraft) -> ResearchPlan:
    if research.status not in EDITABLE:
        raise AppError(409, "invalid_state", f"Cannot edit a plan in state '{research.status}'.")
    budget = budget_for(research.depth)
    try:
        check_plan_against_context(draft, research.mode, budget, bool(research.document_ids))
    except ValueError as exc:
        raise AppError(422, "invalid_plan", str(exc)) from exc
    plan = ResearchPlan(
        **draft.model_dump(),
        budget=budget,
        mode=research.mode,
        depth=research.depth,
        generated_by="human-edited",
    )
    _store_plan(db, research, plan)
    _event(db, research, "plan_edited", f"Plan edited by user: {len(plan.subtasks)} subtasks")
    return plan


def patch_plan(db: Session, research: ResearchSession, patch: PlanPatch) -> ResearchPlan:
    """Apply partial updates to an existing plan without resending unchanged fields."""
    if research.status not in EDITABLE:
        raise AppError(409, "invalid_state", f"Cannot edit a plan in state '{research.status}'.")
    if not research.plan:
        raise AppError(404, "plan_not_found", "No plan has been generated yet.")
    current = ResearchPlan.model_validate(research.plan)
    merged_data = {
        "objectives": patch.objectives if patch.objectives is not None else current.objectives,
        "subtasks": (
            [t.model_dump() for t in patch.subtasks]
            if patch.subtasks is not None
            else [t.model_dump() for t in current.subtasks]
        ),
        "report_outline": (
            patch.report_outline if patch.report_outline is not None else current.report_outline
        ),
    }
    try:
        draft = PlanDraft.model_validate(merged_data)
    except ValueError as exc:
        raise AppError(422, "invalid_plan", str(exc)) from exc
    return edit_plan(db, research, draft)


def reject_plan(
    db: Session,
    research: ResearchSession,
    feedback: str,
    llm: LLMProvider,
    settings: Settings,
) -> ResearchPlan:
    """Record human rejection feedback and regenerate the plan with that guidance."""
    if research.status not in EDITABLE:
        raise AppError(409, "invalid_state", f"Cannot reject a plan in state '{research.status}'.")
    _event(
        db,
        research,
        "plan_rejected",
        f"Plan rejected by user: {feedback.strip()[:200]}",
        {"feedback": feedback.strip()},
    )
    return generate_plan(
        db, research, llm, settings, check_state=False, emit_events=True, feedback=feedback
    )


def approve_plan(db: Session, research: ResearchSession, note: str | None = None) -> None:
    if research.status != "awaiting_approval":
        raise AppError(409, "invalid_state", "No plan is awaiting approval.")
    research.plan_approved = True
    research.status = "planned"
    clean_note = note.strip() if note and note.strip() else None
    summary = "Plan approved by user" + (f": {clean_note}" if clean_note else "")
    _event(
        db,
        research,
        "plan_approved",
        summary,
        {"note": clean_note} if clean_note else None,
    )
