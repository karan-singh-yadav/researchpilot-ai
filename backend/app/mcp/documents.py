"""Document index contract used by the document tools.

The real implementation (PDF ingestion, embeddings, vector store) arrives with RAG in Phase 6.
Until then the server runs with `UnavailableDocumentIndex`, which fails with a clear, typed error,
so the tool contracts, permissions and tests exist now and Phase 6 only has to plug in an index.
"""

from typing import Protocol

from app.mcp.errors import ToolFailure
from app.mcp.schemas import DocumentChunk


class DocumentIndex(Protocol):
    name: str

    def search(
        self, query: str, top_k: int, document_ids: list[str] | None
    ) -> list[DocumentChunk]: ...

    def get_chunks(self, chunk_ids: list[str]) -> list[DocumentChunk]: ...


class UnavailableDocumentIndex:
    name = "unavailable"

    def search(self, query, top_k, document_ids):
        raise ToolFailure("not_available", "Document search is not available yet.")

    def get_chunks(self, chunk_ids):
        raise ToolFailure("not_available", "Document retrieval is not available yet.")
