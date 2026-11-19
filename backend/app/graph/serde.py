"""JSON-safe (de)serialisation for graph state, so checkpoints never hold live objects."""

from datetime import datetime

from app.agents.analyst import AnalysisResult, CalibratedFinding
from app.schemas.evidence import (
    Contradiction,
    EvidenceItem,
    ExtractedItem,
    Finding,
    Gap,
    SourceRecord,
)
from app.services.llm import LLMUsage


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def from_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def analysis_to_dict(a: AnalysisResult) -> dict:
    return {
        "findings": [
            {
                "finding": cf.finding.model_dump(),
                "corroborating_sources": cf.corroborating_sources,
                "downgraded": cf.downgraded,
            }
            for cf in a.findings
        ],
        "contradictions": [c.model_dump() for c in a.contradictions],
        "gaps": [g.model_dump() for g in a.gaps],
        "follow_up_queries": list(a.follow_up_queries),
        "needs_more_research": a.needs_more_research,
        "warnings": list(a.warnings),
        "attempts": a.attempts,
    }


def analysis_from_dict(d: dict) -> AnalysisResult:
    return AnalysisResult(
        findings=[
            CalibratedFinding(
                Finding(**f["finding"]), f["corroborating_sources"], downgraded=f["downgraded"]
            )
            for f in d["findings"]
        ],
        contradictions=[Contradiction(**c) for c in d["contradictions"]],
        gaps=[Gap(**g) for g in d["gaps"]],
        follow_up_queries=list(d["follow_up_queries"]),
        needs_more_research=d["needs_more_research"],
        usage=LLMUsage(),
        attempts=d["attempts"],
        warnings=list(d["warnings"]),
    )


def sources_from_state(raw: dict[str, dict]) -> dict[str, SourceRecord]:
    return {ref: SourceRecord(**s) for ref, s in raw.items()}


def evidence_from_state(raw: list[dict]) -> list[EvidenceItem]:
    return [EvidenceItem(**e) for e in raw]


def read_result_to_dict(r) -> dict:
    return {
        "url": r.final_url,
        "hit_url": r.hit.url,
        "rank": r.hit.rank,
        "title": r.title,
        "published_at": iso(r.published_at),
        "retrieved_at": iso(r.retrieved_at),
        "provider": r.provider,
        "status": r.status,
        "error": r.error,
        "injection_suspected": r.injection_suspected,
        "items": [i.model_dump() for i in r.items],
    }


def items_from_dict(d: dict) -> list[ExtractedItem]:
    return [ExtractedItem(**i) for i in d["items"]]
