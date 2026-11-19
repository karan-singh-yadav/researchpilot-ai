"""Tool implementations: plain synchronous functions over the project's existing tool layer.

They hold no MCP/protocol code, so they can be unit-tested directly and reused elsewhere.
All failures are raised as ToolFailure with a stable code and a message that is safe to expose.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.mcp.documents import DocumentIndex, UnavailableDocumentIndex
from app.mcp.errors import ToolFailure
from app.mcp.guard import ToolLimits
from app.mcp.schemas import (
    DocumentChunk,
    DocumentChunksOut,
    ExportOut,
    FetchWebpageOut,
    SearchWebOut,
    WebHit,
)
from app.models import Report, ResearchSession, Source
from app.security.injection import looks_like_injection
from app.tools.fetcher import FetchError, PageFetcher
from app.tools.html_extract import extract_page
from app.tools.search import SearchError, SearchProvider

_UUID = re.compile(r"^[0-9a-fA-F-]{36}$")


@dataclass
class ToolBackends:
    search: SearchProvider | None
    fetcher: PageFetcher
    limits: ToolLimits
    documents: DocumentIndex = None  # type: ignore[assignment]
    session_factory: Callable[[], Session] | None = None

    def __post_init__(self) -> None:
        if self.documents is None:
            self.documents = UnavailableDocumentIndex()


def do_search_web(b: ToolBackends, query: str, max_results: int) -> SearchWebOut:
    if b.search is None:
        raise ToolFailure("not_configured", "No search provider is configured on the tool server.")
    try:
        hits = b.search.search(query, max_results)
    except SearchError as exc:
        raise ToolFailure("search_failed", str(exc), retryable=exc.retryable) from None
    return SearchWebOut(
        provider=b.search.name,
        query=query,
        hits=[
            WebHit(
                rank=h.rank,
                title=h.title,
                url=h.url,
                snippet=h.snippet,
                published_at=h.published_at,
            )
            for h in hits[:max_results]
        ],
        retrieved_at=datetime.now(UTC),
    )


def do_fetch_webpage(b: ToolBackends, url: str, max_chars: int) -> FetchWebpageOut:
    max_chars = min(max_chars, b.limits.max_text_chars)
    try:
        res = b.fetcher.fetch(url)
    except FetchError as exc:
        raise ToolFailure(exc.code, str(exc), retryable=exc.retryable) from None
    page = extract_page(res.content_type, res.text)
    if not page.text.strip():
        raise ToolFailure("no_text", "The page had no extractable text.")
    text = page.text[:max_chars]
    return FetchWebpageOut(
        url=url,
        final_url=res.final_url,
        title=page.title,
        text=text,
        published_at=page.published_at,
        content_type=res.content_type,
        truncated=res.truncated or len(page.text) > max_chars,
        retrieved_at=res.retrieved_at,
        provider=res.provider,
        injection_suspected=looks_like_injection(text),
    )


def do_search_documents(
    b: ToolBackends, query: str, top_k: int, document_ids: list[str] | None
) -> DocumentChunksOut:
    chunks = b.documents.search(query, top_k, document_ids)
    return DocumentChunksOut(index=b.documents.name, chunks=_clip(chunks[:top_k], b.limits))


def do_retrieve_chunks(b: ToolBackends, chunk_ids: list[str]) -> DocumentChunksOut:
    chunks = b.documents.get_chunks(chunk_ids)
    return DocumentChunksOut(index=b.documents.name, chunks=_clip(chunks, b.limits))


def _clip(chunks: list[DocumentChunk], limits: ToolLimits) -> list[DocumentChunk]:
    return [c.model_copy(update={"text": c.text[:5000]}) for c in chunks]


def do_export_report(b: ToolBackends, research_id: str, fmt: str) -> ExportOut:
    if b.session_factory is None:
        raise ToolFailure("not_configured", "The tool server has no database configured.")
    db = b.session_factory()
    try:
        return build_export(db, research_id, fmt, b.limits.max_export_bytes)
    finally:
        db.close()


def build_export(db: Session, research_id: str, fmt: str, max_bytes: int) -> ExportOut:
    """Render the latest report of a research session as Markdown or JSON (shared by MCP + HTTP)."""
    if not _UUID.match(research_id):
        raise ToolFailure("invalid_arguments", "research_id must be a UUID.")
    research = db.get(ResearchSession, research_id)
    report = db.scalars(
        select(Report).where(Report.session_id == research_id).order_by(Report.version.desc())
    ).first()
    if research is None or report is None:
        raise ToolFailure("not_found", "No report exists for that research id.")
    if fmt == "markdown":
        content, ext = report.markdown, "md"
    else:
        sources = db.scalars(select(Source).where(Source.session_id == research_id)).all()
        content = json.dumps(
            {
                "research_id": research_id,
                "query": research.query,
                "status": research.status,
                "report": {
                    "version": report.version,
                    "markdown": report.markdown,
                    "validation": report.validation,
                },
                "sources": [
                    {
                        "ref": s.source_ref,
                        "title": s.title,
                        "url": s.url,
                        "published_at": s.published_at.isoformat() if s.published_at else None,
                        "retrieved_at": s.retrieved_at.isoformat(),
                        "provider": s.provider,
                        "status": s.status,
                        "injection_suspected": s.injection_suspected,
                        "evidence": [
                            {"ref": e.evidence_ref, "claim": e.claim, "passage": e.passage}
                            for e in s.evidence
                        ],
                    }
                    for s in sorted(sources, key=lambda s: int(s.source_ref[1:]))
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        ext = "json"
    size = len(content.encode("utf-8"))
    if size > max_bytes:
        raise ToolFailure("too_large", f"The export is {size} bytes; the limit is {max_bytes}.")
    return ExportOut(
        research_id=research_id,
        format=fmt,  # type: ignore[arg-type]
        filename=f"research-{research_id[:8]}.{ext}",
        content=content,
        size_bytes=size,
    )
