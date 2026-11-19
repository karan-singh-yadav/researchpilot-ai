"""Vector indexes over document chunks.

The database (document_chunks.embedding) is always the source of truth. SqlIndex searches it
directly (brute-force cosine with numpy, cached per process and invalidated automatically when
the set of ready documents changes), which is safe across processes - the MCP tool server relies
on this. ChromaIndex is an optional derived ANN index for the backend process only: a Chroma
index written by one process is NOT reliably queryable from another (verified), so never share it.
"""

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.logging_config import get_logger
from app.models import DocumentChunk, UploadedDocument

log = get_logger(__name__)


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    document_id: str
    page: int
    score: float  # cosine similarity


class SqlIndex:
    name = "sql"

    def __init__(self, session_factory: Callable[[], Session], embedding_model: str) -> None:
        self.sf = session_factory
        self.model = embedding_model
        self._lock = threading.Lock()
        self._stamp: tuple | None = None
        self._ids: list[str] = []
        self._docs: np.ndarray = np.array([], dtype=object)
        self._pages: np.ndarray = np.array([], dtype=np.int32)
        self._mat: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def _current_stamp(self, db: Session) -> tuple:
        row = db.execute(
            select(func.count(UploadedDocument.id), func.max(UploadedDocument.updated_at)).where(
                UploadedDocument.status == "ready", UploadedDocument.embedding_model == self.model
            )
        ).one()
        return (row[0], str(row[1]))

    def _ensure_loaded(self) -> None:
        db = self.sf()
        try:
            stamp = self._current_stamp(db)
            if stamp == self._stamp:
                return
            with self._lock:
                rows = db.execute(
                    select(
                        DocumentChunk.id,
                        DocumentChunk.document_id,
                        DocumentChunk.page,
                        DocumentChunk.embedding,
                    )
                    .join(UploadedDocument, UploadedDocument.id == DocumentChunk.document_id)
                    .where(
                        UploadedDocument.status == "ready",
                        UploadedDocument.embedding_model == self.model,
                        DocumentChunk.embedding.is_not(None),
                    )
                    .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
                ).all()
                vecs = [np.frombuffer(r[3], dtype=np.float32) for r in rows]
                dim = max((len(v) for v in vecs), default=0)
                keep = [i for i, v in enumerate(vecs) if len(v) == dim]
                if len(keep) != len(rows):
                    log.warning(
                        "ignoring %d chunk(s) with a different embedding size",
                        len(rows) - len(keep),
                    )
                self._ids = [rows[i][0] for i in keep]
                self._docs = np.array([rows[i][1] for i in keep], dtype=object)
                self._pages = np.array([rows[i][2] for i in keep], dtype=np.int32)
                self._mat = (
                    np.stack([vecs[i] for i in keep]) if keep else np.zeros((0, dim), np.float32)
                )
                self._stamp = stamp
        finally:
            db.close()

    def query(
        self,
        vector: np.ndarray,
        top_k: int,
        document_ids: list[str] | None = None,
        page_min: int | None = None,
        page_max: int | None = None,
    ) -> list[Hit]:
        self._ensure_loaded()
        with self._lock:
            ids, docs, pages, mat = self._ids, self._docs, self._pages, self._mat
        if mat.shape[0] == 0 or mat.shape[1] != len(vector):
            return []
        mask = np.ones(len(ids), dtype=bool)
        if document_ids is not None:
            mask &= np.isin(docs, list(document_ids))
        if page_min is not None:
            mask &= pages >= page_min
        if page_max is not None:
            mask &= pages <= page_max
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        scores = mat[idx] @ np.asarray(vector, dtype=np.float32)
        k = min(top_k, idx.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(ids[idx[i]], docs[idx[i]], int(pages[idx[i]]), float(scores[i])) for i in top]

    # The database already holds everything; these only matter for derived indexes.
    def on_chunks_added(self, document_id: str, ids, pages, matrix) -> None:
        self._stamp = None

    def on_document_deleted(self, document_id: str) -> None:
        self._stamp = None

    def rebuild(self) -> int:
        self._stamp = None
        self._ensure_loaded()
        return len(self._ids)


class ChromaIndex:
    name = "chroma"

    def __init__(
        self, path: str, embedding_model: str, session_factory: Callable[[], Session]
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise RuntimeError(
                "VECTOR_STORE=chroma needs chromadb (pip install chromadb)"
            ) from None
        self.sf, self.model = session_factory, embedding_model
        self._client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", embedding_model).strip("-.")[:48] or "default"
        self._name = f"chunks_{slug}"
        self._col = self._open()

    def _open(self):
        return self._client.get_or_create_collection(
            self._name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )

    @staticmethod
    def _where(document_ids, page_min, page_max):
        conds = []
        if document_ids is not None:
            conds.append({"document_id": {"$in": list(document_ids)}})
        if page_min is not None:
            conds.append({"page": {"$gte": page_min}})
        if page_max is not None:
            conds.append({"page": {"$lte": page_max}})
        return None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}

    def query(self, vector, top_k, document_ids=None, page_min=None, page_max=None) -> list[Hit]:
        n = self._col.count()
        if n == 0 or (document_ids is not None and not document_ids):
            return []
        res = self._col.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=min(top_k, n),
            where=self._where(document_ids, page_min, page_max),
            include=["metadatas", "distances"],
        )
        return [
            Hit(cid, meta["document_id"], int(meta["page"]), 1.0 - float(dist))
            for cid, meta, dist in zip(
                res["ids"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
        ]

    def on_chunks_added(self, document_id: str, ids, pages, matrix) -> None:
        for i in range(0, len(ids), 500):
            self._col.upsert(
                ids=list(ids[i : i + 500]),
                embeddings=np.asarray(matrix[i : i + 500], dtype=np.float32).tolist(),
                metadatas=[
                    {"document_id": document_id, "page": int(p)} for p in pages[i : i + 500]
                ],
            )

    def on_document_deleted(self, document_id: str) -> None:
        self._col.delete(where={"document_id": document_id})

    def rebuild(self) -> int:
        """Recreate the collection from the database (the source of truth)."""
        self._client.delete_collection(self._name)
        self._col = self._open()
        db = self.sf()
        try:
            docs = db.scalars(
                select(UploadedDocument).where(
                    UploadedDocument.status == "ready",
                    UploadedDocument.embedding_model == self.model,
                )
            ).all()
            total = 0
            for d in docs:
                rows = db.execute(
                    select(DocumentChunk.id, DocumentChunk.page, DocumentChunk.embedding)
                    .where(DocumentChunk.document_id == d.id, DocumentChunk.embedding.is_not(None))
                    .order_by(DocumentChunk.chunk_index)
                ).all()
                if rows:
                    mat = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
                    self.on_chunks_added(d.id, [r[0] for r in rows], [r[1] for r in rows], mat)
                    total += len(rows)
            return total
        finally:
            db.close()

    def ensure_synced(self) -> bool:
        """Rebuild if the derived index disagrees with the database. Returns True if rebuilt."""
        db = self.sf()
        try:
            expected = db.scalar(
                select(func.count(DocumentChunk.id))
                .join(UploadedDocument, UploadedDocument.id == DocumentChunk.document_id)
                .where(
                    UploadedDocument.status == "ready",
                    UploadedDocument.embedding_model == self.model,
                )
            )
        finally:
            db.close()
        if self._col.count() != expected:
            log.warning(
                "chroma index out of sync (%s vs %s chunks): rebuilding",
                self._col.count(),
                expected,
            )
            self.rebuild()
            return True
        return False
