from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractedItem(BaseModel):
    """One piece of evidence proposed by the LLM. `passage` MUST be verbatim from the source."""

    model_config = ConfigDict(extra="ignore")
    passage: str = Field(min_length=20, max_length=600)
    claim: str = Field(min_length=5, max_length=300)
    kind: Literal["fact", "figure", "claim", "definition"] = "fact"
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)  # model-estimated, used for ranking only
    page_number: int | None = Field(default=None, ge=1)  # set by code for document evidence


class ExtractionDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")
    items: list[ExtractedItem] = Field(default_factory=list, max_length=10)


class SourceRecord(BaseModel):
    ref: str  # stable citation id, e.g. "S1"
    kind: Literal["web", "document"] = "web"
    title: str
    url: str
    published_at: datetime | None = None
    retrieved_at: datetime
    provider: str = "http"  # http | fixture
    status: Literal["read", "failed"] = "read"
    error: str | None = None
    injection_suspected: bool = False


class EvidenceItem(BaseModel):
    id: str  # "E1"
    source_ref: str
    passage: str
    claim: str
    kind: str = "fact"
    relevance: float = 0.5
    page_number: int | None = None


class Finding(BaseModel):
    model_config = ConfigDict(extra="ignore")
    statement: str = Field(min_length=10, max_length=800)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)
    type: Literal["fact", "interpretation", "assumption"] = "fact"
    confidence: Literal["low", "medium", "high"] = "low"


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    topic: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=600)
    evidence_ids: list[str] = Field(min_length=2, max_length=10)


class Gap(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str = Field(min_length=5, max_length=400)
    follow_up_queries: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("follow_up_queries")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        out = [" ".join(q.split()) for q in v]
        if any(not 3 <= len(q) <= 200 for q in out):
            raise ValueError("follow-up queries must be 3-200 characters")
        return out


class AnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")
    findings: list[Finding] = Field(min_length=1, max_length=25)
    contradictions: list[Contradiction] = Field(default_factory=list, max_length=8)
    gaps: list[Gap] = Field(default_factory=list, max_length=8)
    needs_more_research: bool = False


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    heading: str = Field(min_length=2, max_length=120)
    body: str = Field(min_length=20, max_length=8000)


class ComparisonTable(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = Field(min_length=2, max_length=160)
    columns: list[str] = Field(min_length=2, max_length=6)
    rows: list[list[str]] = Field(min_length=1, max_length=12)

    @field_validator("rows")
    @classmethod
    def _rect(cls, rows: list[list[str]], info) -> list[list[str]]:
        cols = info.data.get("columns")
        if cols and any(len(r) != len(cols) for r in rows):
            raise ValueError("every table row must have one cell per column")
        return rows


class ReportDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = Field(min_length=5, max_length=200)
    executive_summary: str = Field(min_length=20, max_length=4000)
    sections: list[ReportSection] = Field(min_length=1, max_length=12)
    comparison_tables: list[ComparisonTable] = Field(default_factory=list, max_length=4)
    limitations: str = Field(default="", max_length=3000)
