"""Semantic retrieval with metadata filtering and optional reranking."""

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DocumentChunk, UploadedDocument
from app.rag.embeddings import Embedder, EmbeddingError
from app.rag.types import DocumentChunk as ChunkOut

_TOKEN = re.compile(r"[a-z0-9]+")


class RetrievalError(Exception):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LexicalReranker:
    """BM25 over the candidate set, blended with the vector score (both min-max normalised).

    A cheap second opinion that rewards exact term matches (names, numbers, acronyms) which dense
    or hashed embeddings can blur. It is NOT a cross-encoder."""

    name = "lexical"

    def __init__(self, weight: float = 0.5, k1: float = 1.5, b: float = 0.75) -> None:
        self.w, self.k1, self.b = weight, k1, b

    def rerank(self, query: str, chunks: list[ChunkOut]) -> list[ChunkOut]:
        if len(chunks) < 2:
            return chunks
        q = [t for t in _TOKEN.findall(query.lower()) if len(t) > 1]
        docs = [Counter(_TOKEN.findall(c.text.lower())) for c in chunks]
        lens = [sum(d.values()) or 1 for d in docs]
        avg = sum(lens) / len(lens)
        n = len(chunks)
        bm = []
        for d, ln in zip(docs, lens, strict=True):
            s = 0.0
            for t in set(q):
                df = sum(1 for x in docs if t in x)
                if not df:
                    continue
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                f = d.get(t, 0)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / avg))
            bm.append(s)

        def norm(xs: list[float]) -> list[float]:
            lo, hi = min(xs), max(xs)
            return [0.5] * len(xs) if hi == lo else [(x - lo) / (hi - lo) for x in xs]

        vec = norm([c.score or 0.0 for c in chunks])
        lex = norm(bm)
        scored = [
            c.model_copy(update={"score": (1 - self.w) * v + self.w * l_})
            for c, v, l_ in zip(chunks, vec, lex, strict=True)
        ]
        return sorted(scored, key=lambda c: -(c.score or 0.0))


@dataclass
class Retriever:
    index: object
    embedder: Embedder
    session_factory: Callable[[], Session]
    reranker: LexicalReranker | None = None
    overfetch: int = 4

    def _hydrate(self, hits) -> list[ChunkOut]:
        db = self.session_factory()
        try:
            ids = [h.chunk_id for h in hits]
            rows = {
                c.id: (c, fn)
                for c, fn in db.execute(
                    select(DocumentChunk, UploadedDocument.filename)
                    .join(UploadedDocument, UploadedDocument.id == DocumentChunk.document_id)
                    .where(DocumentChunk.id.in_(ids), UploadedDocument.status == "ready")
                ).all()
            }
        finally:
            db.close()
        out = []
        for h in hits:
            if h.chunk_id in rows:  # a document deleted meanwhile simply disappears
                c, fn = rows[h.chunk_id]
                out.append(
                    ChunkOut(
                        chunk_id=c.id,
                        document_id=c.document_id,
                        filename=fn,
                        page=c.page,
                        score=h.score,
                        text=c.text[:5000],
                    )
                )
        return out

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        page_min: int | None = None,
        page_max: int | None = None,
    ) -> list[ChunkOut]:
        if document_ids is not None and not document_ids:
            return []
        try:
            vec = self.embedder.embed([query])[0]
        except EmbeddingError as exc:
            raise RetrievalError(str(exc), retryable=exc.retryable) from None
        want = top_k * self.overfetch if self.reranker else top_k
        hits = self.index.query(vec, want, document_ids, page_min, page_max)
        chunks = self._hydrate(hits)
        if self.reranker:
            chunks = self.reranker.rerank(query, chunks)
        return chunks[:top_k]

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkOut]:
        from app.rag.index import Hit

        by_id = {c.chunk_id: c for c in self._hydrate([Hit(i, "", 0, 0.0) for i in chunk_ids])}
        return [by_id[i].model_copy(update={"score": None}) for i in chunk_ids if i in by_id]
