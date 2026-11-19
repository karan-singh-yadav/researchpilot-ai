"""Short-lived-session helpers used by graph nodes (each call commits independently)."""

from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import Evidence, Report, ResearchSession, ResearchTask, Source, UploadedDocument
from app.schemas.evidence import EvidenceItem, SourceRecord
from app.services.llm import LLMUsage
from app.services.llm.pricing import estimate_cost_usd

SF = Callable[[], Session]


def _with(sf: SF, fn):
    db = sf()
    try:
        out = fn(db)
        db.commit()
        return out
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def load_session(sf: SF, research_id: str) -> dict | None:
    def f(db: Session):
        r = db.get(ResearchSession, research_id)
        if r is None:
            return None
        return {
            "query": r.query,
            "mode": r.mode,
            "depth": r.depth,
            "status": r.status,
            "plan": r.plan,
            "plan_approved": r.plan_approved,
            "require_plan_approval": r.require_plan_approval,
            "document_ids": list(r.document_ids or []),
        }

    return _with(sf, f)


def set_status(sf: SF, research_id: str, status: str, error: str | None = None, **fields) -> None:
    def f(db: Session):
        r = db.get(ResearchSession, research_id)
        if r is None:
            return
        r.status = status
        r.error = error
        for k, v in fields.items():
            setattr(r, k, v)

    _with(sf, f)


def add_usage(sf: SF, research_id: str, usage: LLMUsage, in_price: float, out_price: float) -> None:
    if not usage.total:
        return

    def f(db: Session):
        r = db.get(ResearchSession, research_id)
        if r:
            r.token_usage += usage.total
            r.estimated_cost_usd += estimate_cost_usd(usage, in_price, out_price)

    _with(sf, f)


def persist_registry(
    sf: SF, research_id: str, sources: dict[str, SourceRecord], evidence: list[EvidenceItem]
) -> None:
    """Idempotent snapshot write: the registry is append-only, so rewriting it is safe."""

    def f(db: Session):
        db.execute(delete(Source).where(Source.session_id == research_id))  # cascades to evidence
        db.flush()
        rows: dict[str, Source] = {}
        for ref, s in sources.items():
            row = Source(
                session_id=research_id,
                source_ref=ref,
                kind=s.kind,
                title=s.title,
                url=s.url,
                published_at=s.published_at,
                retrieved_at=s.retrieved_at,
                document_id=s.url.removeprefix("document://") if s.kind == "document" else None,
                provider=s.provider,
                status=s.status,
                error=s.error,
                injection_suspected=s.injection_suspected,
            )
            db.add(row)
            rows[ref] = row
        db.flush()
        for e in evidence:
            db.add(
                Evidence(
                    source_id=rows[e.source_ref].id,
                    evidence_ref=e.id,
                    claim=e.claim,
                    kind=e.kind,
                    passage=e.passage,
                    page_number=e.page_number,
                    relevance_score=e.relevance,
                )
            )

    _with(sf, f)


def mark_tasks(
    sf: SF, research_id: str, done_plan_ids: set[str], status: str = "completed"
) -> None:
    def f(db: Session):
        for t in db.scalars(select(ResearchTask).where(ResearchTask.session_id == research_id)):
            if (t.result or {}).get("plan_task_id") in done_plan_ids and t.status != status:
                t.status = status
                t.attempts += 1

    _with(sf, f)


def save_report(
    sf: SF, research_id: str, markdown: str, validation: dict, revisions: int, warnings: list[str]
) -> str:
    def f(db: Session):
        version = (
            len(db.scalars(select(Report.id).where(Report.session_id == research_id)).all()) + 1
        )
        rep = Report(
            session_id=research_id, version=version, markdown=markdown, validation=validation
        )
        db.add(rep)
        r = db.get(ResearchSession, research_id)
        r.status = "completed"
        r.error = None
        r.revision_count = revisions
        r.warnings = warnings[:200]
        r.completed_at = datetime.now(UTC)
        db.flush()
        return rep.id

    return _with(sf, f)


def check_documents(sf: SF, ids: list[str]) -> tuple[list[str], list[str]]:
    """Returns (missing_ids, not_ready_ids) for a list of document ids."""

    def f(db: Session):
        rows = {
            d.id: d.status
            for d in db.scalars(select(UploadedDocument).where(UploadedDocument.id.in_(ids)))
        }
        return [i for i in ids if i not in rows], [
            i for i in ids if i in rows and rows[i] != "ready"
        ]

    return _with(sf, f)


def document_names(sf: SF, ids: list[str]) -> list[str]:
    def f(db: Session):
        names = {
            d.id: d.filename
            for d in db.scalars(select(UploadedDocument).where(UploadedDocument.id.in_(ids)))
        }
        return [names.get(i, i) for i in ids]

    return _with(sf, f)
