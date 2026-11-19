"""Document lifecycle: validated upload -> background processing -> searchable; reindex; delete."""

import hashlib
import os
import re
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.logging_config import get_logger
from app.models import DocumentChunk, UploadedDocument
from app.models.base import utcnow
from app.rag.chunking import chunk_pages
from app.rag.embeddings import Embedder, EmbeddingError
from app.rag.pdf import PdfError, extract_pdf, validate_pdf_header

log = get_logger(__name__)
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


class DocumentError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def sanitize_filename(name: str) -> str:
    """Display name only. Files are stored under a generated id, never under this name."""
    name = _CTRL.sub("", name or "").replace("\\", "/").split("/")[-1].strip().strip(".")
    name = name or "document.pdf"
    if len(name) > 200:  # truncate the stem, never the extension
        stem, dot, ext = name.rpartition(".")
        name = f"{stem[: 200 - len(ext) - 1]}.{ext}" if dot and len(ext) <= 10 else name[:200]
    return name


def doc_to_dict(d: UploadedDocument) -> dict:
    return {
        "id": d.id,
        "filename": d.filename,
        "size_bytes": d.size_bytes,
        "page_count": d.page_count,
        "chunk_count": d.chunk_count,
        "status": d.status,
        "error": d.error,
        "content_sha256": d.content_sha256,
        "embedding_model": d.embedding_model,
        "doc_metadata": d.doc_metadata or {},
        "created_at": d.created_at,
        "updated_at": d.updated_at,
    }


class DocumentService:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        settings: Settings,
        embedder: Embedder,
        index,
        *,
        executor: str | None = None,
    ) -> None:
        self.sf, self.settings, self.embedder, self.index = (
            session_factory,
            settings,
            embedder,
            index,
        )
        self.upload_dir = Path(settings.upload_dir)
        self.mode = executor or settings.run_executor
        self._pool = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="docs")
            if self.mode == "thread"
            else None
        )
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- upload
    def upload(self, filename: str, data: bytes) -> tuple[dict, bool]:
        """Validate, store and schedule processing. Returns (document, is_duplicate)."""
        s = self.settings
        if not data:
            raise DocumentError("empty_file", "The uploaded file is empty.")
        if len(data) > s.max_upload_mb * 1024 * 1024:
            raise DocumentError("file_too_large", f"The file exceeds {s.max_upload_mb} MB.", 413)
        name = sanitize_filename(filename)
        if not name.lower().endswith(".pdf"):
            raise DocumentError("invalid_file_type", "Only .pdf files are accepted.", 415)
        try:
            validate_pdf_header(data)
        except PdfError as exc:
            raise DocumentError(exc.code, exc.message, 415) from None
        digest = hashlib.sha256(data).hexdigest()
        db = self.sf()
        try:
            existing = db.scalars(
                select(UploadedDocument).where(UploadedDocument.content_sha256 == digest)
            ).first()
            if existing is not None and existing.status != "failed":
                return doc_to_dict(existing), True
            if existing is not None:  # same bytes failed before: retry using the same record
                doc = existing
                doc.status, doc.error, doc.filename = "processing", None, name
            else:
                doc = UploadedDocument(
                    filename=name, content_sha256=digest, size_bytes=len(data), status="processing"
                )
                db.add(doc)
                db.flush()
                doc.storage_name = f"{doc.id}.pdf"
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.upload_dir / f"{doc.id}.{uuid.uuid4().hex}.part"
            tmp.write_bytes(data)
            os.replace(tmp, self.upload_dir / doc.storage_name)
            db.commit()
            out = doc_to_dict(doc)
        finally:
            db.close()
        self._submit(out["id"], self.process)
        return self._snapshot(out["id"]) or out, False

    def _snapshot(self, doc_id: str) -> dict | None:
        db = self.sf()
        try:
            d = db.get(UploadedDocument, doc_id)
            return doc_to_dict(d) if d else None
        finally:
            db.close()

    def _submit(self, doc_id: str, fn: Callable[[str], None]) -> None:
        if self._pool is None:
            fn(doc_id)
            return
        fut = self._pool.submit(fn, doc_id)
        with self._lock:
            self._futures[doc_id] = fut

    def wait(self, doc_id: str, timeout: float = 60.0) -> None:
        with self._lock:
            fut = self._futures.get(doc_id)
        if fut:
            fut.result(timeout=timeout)

    def shutdown(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------- processing
    def _fail(self, doc_id: str, message: str) -> None:
        db = self.sf()
        try:
            d = db.get(UploadedDocument, doc_id)
            if d is not None:
                d.status, d.error = "failed", message[:500]
                db.commit()
        finally:
            db.close()

    def process(self, doc_id: str) -> None:
        s = self.settings
        db = self.sf()
        try:
            doc = db.get(UploadedDocument, doc_id)
            if doc is None:
                return
            path = self.upload_dir / (doc.storage_name or "")
            data = path.read_bytes()
        except OSError:
            self._fail(doc_id, "The stored file could not be read.")
            return
        finally:
            db.close()
        try:
            content = extract_pdf(data, max_pages=s.max_pdf_pages, max_chars=s.max_pdf_chars)
            chunks = chunk_pages(
                content.pages, s.chunk_size, min(s.chunk_overlap, s.chunk_size // 2)
            )
            matrix = self.embedder.embed([c.text for c in chunks])
            self._store(doc_id, chunks, matrix, content.page_count, content.metadata)
        except PdfError as exc:
            self._fail(doc_id, exc.message)
        except EmbeddingError as exc:
            self._fail(doc_id, f"Embedding failed: {exc}. Use reindex to retry.")
        except Exception as exc:
            log.exception("document processing failed id=%s", doc_id)
            self._fail(doc_id, f"Processing failed ({type(exc).__name__}). Use reindex to retry.")

    def _store(self, doc_id, chunks, matrix: np.ndarray, page_count: int, metadata: dict) -> None:
        ids = [f"{doc_id}:{c.index}" for c in chunks]
        pages = [c.page for c in chunks]
        db = self.sf()
        try:
            doc = db.get(UploadedDocument, doc_id)
            if doc is None:  # deleted while processing
                return
            for old in db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc_id)):
                db.delete(old)
            db.flush()
            for c, cid, vec in zip(chunks, ids, matrix, strict=True):
                db.add(
                    DocumentChunk(
                        id=cid,
                        document_id=doc_id,
                        chunk_index=c.index,
                        page=c.page,
                        text=c.text,
                        embedding=np.asarray(vec, dtype=np.float32).tobytes(),
                    )
                )
            doc.page_count, doc.chunk_count = page_count, len(chunks)
            doc.doc_metadata = metadata
            doc.embedding_model, doc.embedding_dim = self.embedder.name, int(matrix.shape[1])
            db.commit()
            self.index.on_document_deleted(doc_id)  # drop any stale derived entries first
            self.index.on_chunks_added(doc_id, ids, pages, matrix)
            doc.status, doc.error, doc.updated_at = "ready", None, utcnow()  # ready == searchable
            db.commit()
        finally:
            db.close()

    # ---------------------------------------------------------------- reindex
    def reindex(self, doc_id: str) -> dict:
        db = self.sf()
        try:
            doc = db.get(UploadedDocument, doc_id)
            if doc is None:
                raise DocumentError("not_found", "Document not found.", 404)
            if doc.status == "processing":
                raise DocumentError(
                    "already_processing", "The document is still being processed.", 409
                )
            has_chunks = db.scalar(
                select(DocumentChunk.id).where(DocumentChunk.document_id == doc_id).limit(1)
            )
            doc.status, doc.error = "processing", None
            db.commit()
            out = doc_to_dict(doc)
        finally:
            db.close()
        self._submit(doc_id, self._reembed if has_chunks else self.process)
        return self._snapshot(doc_id) or out

    def _reembed(self, doc_id: str) -> None:
        db = self.sf()
        try:
            rows = db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == doc_id)
                .order_by(DocumentChunk.chunk_index)
            ).all()
            texts = [r.text for r in rows]
            ids, pages = [r.id for r in rows], [r.page for r in rows]
            page_count = db.get(UploadedDocument, doc_id).page_count
            meta = db.get(UploadedDocument, doc_id).doc_metadata or {}
        finally:
            db.close()
        try:
            matrix = self.embedder.embed(texts)
            from app.rag.chunking import Chunk

            self._store(
                doc_id,
                [Chunk(i, p, t) for i, (p, t) in enumerate(zip(pages, texts, strict=True))],
                matrix,
                page_count,
                meta,
            )
        except EmbeddingError as exc:
            self._fail(doc_id, f"Embedding failed: {exc}. Use reindex to retry.")
        except Exception as exc:
            log.exception("reindex failed id=%s", doc_id)
            self._fail(doc_id, f"Reindex failed ({type(exc).__name__}).")
        _ = ids

    # ----------------------------------------------------------------- delete
    def delete(self, doc_id: str) -> bool:
        db = self.sf()
        try:
            doc = db.get(UploadedDocument, doc_id)
            if doc is None:
                return False
            storage = doc.storage_name
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc_id))
            db.delete(doc)
            db.commit()
        finally:
            db.close()
        if storage:
            try:
                (self.upload_dir / storage).unlink(missing_ok=True)
            except OSError:
                log.warning("could not remove stored file for %s", doc_id)
        self.index.on_document_deleted(doc_id)
        return True
