from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    evidence_ref: str
    claim: str
    kind: str
    passage: str
    page_number: int | None = None
    relevance_score: float | None = None


class SourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    source_ref: str
    kind: str
    title: str
    url: str | None
    document_id: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime
    provider: str
    status: str
    error: str | None = None
    injection_suspected: bool = False
    evidence: list[EvidenceOut] = []


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str
    version: int
    markdown: str
    validation: dict | None = None
    created_at: datetime


class CitationOccurrenceOut(BaseModel):
    id: str
    raw: str
    start: int
    end: int
    section: str
    source_ref: str
    page: int | None = None
    resolved: bool
    source_title: str | None = None
    source_kind: str | None = None
    source_url: str | None = None
    document_id: str | None = None
    injection_suspected: bool = False
    evidence: list[EvidenceOut] = []


class CitedSourceSummaryOut(BaseModel):
    source_ref: str
    title: str
    kind: str
    url: str | None = None
    document_id: str | None = None
    cited_pages: list[int] = []
    citation_count: int
    evidence_count: int
    injection_suspected: bool = False


class ReportCitationsOut(BaseModel):
    research_id: str
    report_version: int
    total_citations: int
    unique_sources: int
    occurrences: list[CitationOccurrenceOut] = []
    sources: list[CitedSourceSummaryOut] = []
