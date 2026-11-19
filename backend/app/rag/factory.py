from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import Settings
from app.logging_config import get_logger
from app.rag.embeddings import Embedder, build_embedder
from app.rag.index import ChromaIndex, SqlIndex
from app.rag.retriever import LexicalReranker, Retriever

log = get_logger(__name__)


@dataclass
class RagComponents:
    embedder: Embedder
    index: object
    retriever: Retriever


def build_rag(
    settings: Settings, session_factory: Callable[[], Session], *, force_sql: bool = False
) -> RagComponents:
    """force_sql=True is used by the MCP tool server: a Chroma index is not safe to share across
    processes, whereas the SQL index (embeddings stored in the database) is."""
    embedder = build_embedder(settings)
    if settings.vector_store.lower() == "chroma" and not force_sql:
        index = ChromaIndex(settings.chroma_path, embedder.name, session_factory)
        index.ensure_synced()
    else:
        index = SqlIndex(session_factory, embedder.name)
    reranker = LexicalReranker() if settings.reranker.lower() == "lexical" else None
    retriever = Retriever(
        index=index,
        embedder=embedder,
        session_factory=session_factory,
        reranker=reranker,
        overfetch=settings.retrieval_overfetch,
    )
    return RagComponents(embedder, index, retriever)
