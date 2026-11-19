"""Observability & telemetry aggregation over persisted sessions, traces, and reports."""

from collections import Counter

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import AgentEvent, Report, ResearchSession, Source, UploadedDocument


def compute_system_metrics(db: Session) -> dict:
    """Aggregate run counts, token/cost totals, latency percentiles, and validation pass rates."""
    sessions = list(db.scalars(select(ResearchSession)).all())
    by_status = Counter(s.status for s in sessions)
    by_mode = Counter(s.mode for s in sessions)
    by_depth = Counter(s.depth for s in sessions)

    total_tokens = sum(s.token_usage or 0 for s in sessions)
    total_cost_usd = round(sum(s.estimated_cost_usd or 0.0 for s in sessions), 6)

    durations_ms: list[float] = []
    for s in sessions:
        if s.completed_at and s.created_at:
            diff = (s.completed_at - s.created_at).total_seconds() * 1000.0
            if diff >= 0:
                durations_ms.append(diff)
    durations_ms.sort()

    def _pct(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        idx = min(len(vals) - 1, int(round((len(vals) - 1) * q)))
        return round(vals[idx], 1)

    reports = list(db.scalars(select(Report)).all())
    passed_reports = sum(1 for r in reports if (r.validation or {}).get("passed") is True)
    validation_pass_rate = round(passed_reports / len(reports), 4) if reports else None

    # Agent node latency breakdown from AgentEvent
    events = list(
        db.scalars(
            select(AgentEvent).where(
                AgentEvent.event_type == "node_end", AgentEvent.duration_ms.is_not(None)
            )
        ).all()
    )
    by_agent: dict[str, list[int]] = {}
    for ev in events:
        if ev.duration_ms is not None:
            by_agent.setdefault(ev.agent, []).append(ev.duration_ms)

    agent_latency_ms = {
        agent: {
            "calls": len(vals),
            "avg_ms": round(sum(vals) / len(vals), 1),
            "max_ms": max(vals),
        }
        for agent, vals in sorted(by_agent.items())
    }

    total_sources = db.scalar(select(func.count()).select_from(Source)) or 0
    flagged_sources = (
        db.scalar(select(func.count()).select_from(Source).where(Source.injection_suspected.is_(True)))
        or 0
    )
    total_documents = db.scalar(select(func.count()).select_from(UploadedDocument)) or 0

    return {
        "sessions": {
            "total": len(sessions),
            "by_status": dict(by_status),
            "by_mode": dict(by_mode),
            "by_depth": dict(by_depth),
        },
        "usage": {
            "total_tokens": total_tokens,
            "total_estimated_cost_usd": total_cost_usd,
            "avg_tokens_per_session": round(total_tokens / len(sessions), 1) if sessions else 0.0,
        },
        "latency_ms": {
            "completed_runs": len(durations_ms),
            "p50": _pct(durations_ms, 0.50),
            "p95": _pct(durations_ms, 0.95),
            "by_agent": agent_latency_ms,
        },
        "quality": {
            "reports_generated": len(reports),
            "validation_passed": passed_reports,
            "validation_pass_rate": validation_pass_rate,
            "total_sources": total_sources,
            "injection_flagged_sources": flagged_sources,
            "uploaded_documents": total_documents,
        },
    }
