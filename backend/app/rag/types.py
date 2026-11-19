from typing import Protocol

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A retrieved passage of an uploaded document (shared by the RAG layer and the MCP tools)."""

    chunk_id: str
    document_id: str
    filename: str
    page: int | None = None
    score: float | None = None
    text: str = Field(max_length=5000)


class ChunkSearcher(Protocol):
    """What agents need from document retrieval (in-process Retriever or the MCP adapter)."""

    def search(
        self, query: str, top_k: int = 5, document_ids: list[str] | None = None
    ) -> list[DocumentChunk]: ...
