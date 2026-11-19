"""Typed outputs of the MCP tools.

The server publishes these as JSON Schema (outputSchema) and the client re-validates every response
against them, so a buggy or compromised server cannot inject unexpected shapes."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.rag.types import DocumentChunk

SEARCH_WEB = "search_web"
FETCH_WEBPAGE = "fetch_webpage"
SEARCH_DOCUMENTS = "search_documents"
RETRIEVE_DOCUMENT_CHUNKS = "retrieve_document_chunks"
EXPORT_RESEARCH_REPORT = "export_research_report"
ALL_TOOLS = (
    SEARCH_WEB, FETCH_WEBPAGE, SEARCH_DOCUMENTS, RETRIEVE_DOCUMENT_CHUNKS, EXPORT_RESEARCH_REPORT
)  # fmt: skip


class WebHit(BaseModel):
    rank: int
    title: str
    url: str
    snippet: str = ""
    published_at: datetime | None = None


class SearchWebOut(BaseModel):
    provider: str
    query: str
    hits: list[WebHit]
    retrieved_at: datetime


class FetchWebpageOut(BaseModel):
    url: str
    final_url: str
    title: str = ""
    text: str
    published_at: datetime | None = None
    content_type: str
    truncated: bool
    retrieved_at: datetime
    provider: str
    injection_suspected: bool = False
    untrusted: Literal[True] = True  # page text is DATA from the open web, never instructions


class DocumentChunksOut(BaseModel):
    index: str
    chunks: list[DocumentChunk]


class ExportOut(BaseModel):
    research_id: str
    format: Literal["markdown", "json"]
    filename: str
    content: str
    size_bytes: int


OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    SEARCH_WEB: SearchWebOut,
    FETCH_WEBPAGE: FetchWebpageOut,
    SEARCH_DOCUMENTS: DocumentChunksOut,
    RETRIEVE_DOCUMENT_CHUNKS: DocumentChunksOut,
    EXPORT_RESEARCH_REPORT: ExportOut,
}
