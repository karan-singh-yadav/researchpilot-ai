from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.document import DocumentAgent
from app.api.deps import get_documents, get_llm, get_retriever
from app.config import Settings, get_settings
from app.database import get_db
from app.errors import AppError
from app.models import DocumentChunk, ResearchSession, UploadedDocument
from app.rag.retriever import RetrievalError, Retriever
from app.rag.service import DocumentError, DocumentService, doc_to_dict
from app.schemas.documents import (
    AskOut,
    AskRequest,
    CitationOut,
    DocumentOut,
    SearchOut,
    SearchRequest,
    StoredChunk,
    UploadOut,
)
from app.schemas.research import Page
from app.security.auth import require_api_key
from app.security.rate_limit import rate_limit
from app.services.llm import LLMError, LLMProvider

router = APIRouter(
    prefix="/api/documents",
    tags=["documents"],
    dependencies=[Depends(rate_limit), Depends(require_api_key)],
)


def _get_or_404(db: Session, doc_id: str) -> UploadedDocument:
    doc = db.get(UploadedDocument, doc_id)
    if doc is None:
        raise AppError(404, "not_found", "Document not found.")
    return doc


def _err(exc: DocumentError) -> AppError:
    return AppError(exc.status, exc.code, exc.message)


@router.post("", response_model=UploadOut, status_code=202)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    svc: DocumentService = Depends(get_documents),
    settings: Settings = Depends(get_settings),
) -> UploadOut:
    """Upload a PDF. Returns 202 while it is processed (poll GET /api/documents/{id}), or 200 with
    duplicate=true if identical bytes were already uploaded."""
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)  # never buffers more than limit+1 bytes
    try:
        doc, duplicate = await run_in_threadpool(svc.upload, file.filename or "document.pdf", data)
    except DocumentError as exc:
        raise _err(exc) from None
    if duplicate:
        response.status_code = 200
    return UploadOut(document=DocumentOut(**doc), duplicate=duplicate)


@router.get("", response_model=Page[DocumentOut])
def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, pattern="^(processing|ready|failed)$"),
    q: str | None = Query(None, max_length=100),
    db: Session = Depends(get_db),
) -> dict:
    stmt, cnt = select(UploadedDocument), select(func.count()).select_from(UploadedDocument)
    if status:
        stmt, cnt = (
            stmt.where(UploadedDocument.status == status),
            cnt.where(UploadedDocument.status == status),
        )
    if q:
        like = f"%{q.replace('%', '').replace('_', '')}%"
        stmt, cnt = (
            stmt.where(UploadedDocument.filename.ilike(like)),
            cnt.where(UploadedDocument.filename.ilike(like)),
        )
    rows = db.scalars(
        stmt.order_by(UploadedDocument.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return {
        "items": [DocumentOut(**doc_to_dict(d)) for d in rows],
        "total": db.scalar(cnt) or 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str, db: Session = Depends(get_db)) -> DocumentOut:
    return DocumentOut(**doc_to_dict(_get_or_404(db, doc_id)))


@router.get("/{doc_id}/chunks", response_model=list[StoredChunk])
def get_chunks(
    doc_id: str,
    page: int | None = Query(None, ge=1),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[StoredChunk]:
    _get_or_404(db, doc_id)
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    if page:
        stmt = stmt.where(DocumentChunk.page == page)
    rows = db.scalars(stmt.order_by(DocumentChunk.chunk_index).limit(limit).offset(offset)).all()
    return [
        StoredChunk(chunk_id=r.id, chunk_index=r.chunk_index, page=r.page, text=r.text)
        for r in rows
    ]


@router.post("/{doc_id}/reindex", response_model=DocumentOut, status_code=202)
def reindex_document(doc_id: str, svc: DocumentService = Depends(get_documents)) -> DocumentOut:
    """Re-embed with the current embedding model (or retry a failed document)."""
    try:
        return DocumentOut(**svc.reindex(doc_id))
    except DocumentError as exc:
        raise _err(exc) from None


@router.delete("/{doc_id}", status_code=204)
def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    svc: DocumentService = Depends(get_documents),
) -> Response:
    _get_or_404(db, doc_id)
    busy = db.scalars(
        select(ResearchSession).where(ResearchSession.status.in_(("running", "planning")))
    ).all()
    if any(doc_id in (r.document_ids or []) for r in busy):
        raise AppError(409, "document_in_use", "A running research session is using this document.")
    db.close()  # release the read transaction before the service writes
    svc.delete(doc_id)
    return Response(status_code=204)


@router.post("/search", response_model=SearchOut)
def search_documents(
    body: SearchRequest, retriever: Retriever = Depends(get_retriever)
) -> SearchOut:
    """Inspect retrieval: scored chunks with document name and page."""
    try:
        chunks = retriever.search(
            body.query, body.top_k, body.document_ids, body.page_min, body.page_max
        )
    except RetrievalError as exc:
        raise AppError(502, "retrieval_failed", str(exc)) from None
    return SearchOut(chunks=chunks)


@router.post("/ask", response_model=AskOut)
def ask_documents(
    body: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    llm: LLMProvider = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> AskOut:
    """Answer from the documents; each citation is a verified verbatim quote with its page."""
    agent = DocumentAgent(retriever, llm, transport_retries=settings.llm_max_retries)
    try:
        res = agent.answer(body.question, body.document_ids, body.top_k)
    except RetrievalError as exc:
        raise AppError(502, "retrieval_failed", str(exc)) from None
    except LLMError:
        raise AppError(502, "llm_failed", "The language model request failed.") from None
    return AskOut(
        answered=res.answered,
        answer=res.answer,
        citations=[CitationOut(**c.model_dump()) for c in res.citations],
        retrieved=res.retrieved,
        tokens=res.usage.total,
        llm=f"{llm.name}:{llm.model}",
    )
