from datetime import datetime

from pydantic import BaseModel, Field

from app.rag.types import DocumentChunk


class DocumentOut(BaseModel):
    id: str
    filename: str
    size_bytes: int
    page_count: int
    chunk_count: int
    status: str  # processing | ready | failed
    error: str | None = None
    content_sha256: str
    embedding_model: str = ""
    doc_metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class UploadOut(BaseModel):
    document: DocumentOut
    duplicate: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = Field(default=None, max_length=20)
    page_min: int | None = Field(default=None, ge=1)
    page_max: int | None = Field(default=None, ge=1)


class SearchOut(BaseModel):
    chunks: list[DocumentChunk]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    document_ids: list[str] | None = Field(default=None, max_length=20)
    top_k: int = Field(default=5, ge=1, le=10)


class CitationOut(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    quote: str


class AskOut(BaseModel):
    answered: bool
    answer: str
    citations: list[CitationOut]
    retrieved: list[DocumentChunk]
    tokens: int
    llm: str


class StoredChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    page: int
    text: str
