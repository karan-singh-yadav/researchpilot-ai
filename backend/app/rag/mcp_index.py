"""DocumentIndex for the MCP tools, built on the Retriever (always SQL-backed: process-safe)."""

from app.mcp.errors import ToolFailure
from app.rag.retriever import RetrievalError, Retriever
from app.rag.types import DocumentChunk


class RagDocumentIndex:
    name = "rag"

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def search(self, query: str, top_k: int, document_ids: list[str] | None) -> list[DocumentChunk]:
        try:
            return self.retriever.search(query, top_k, document_ids)
        except RetrievalError as exc:
            raise ToolFailure("search_failed", str(exc), retryable=exc.retryable) from None

    def get_chunks(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        return self.retriever.get_chunks(chunk_ids)
