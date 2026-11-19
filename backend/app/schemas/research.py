from datetime import datetime
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResearchMode(StrEnum):
    web = "web"
    document = "document"
    hybrid = "hybrid"


class ResearchDepth(StrEnum):
    quick = "quick"
    standard = "standard"
    deep = "deep"


class ResearchStatus(StrEnum):
    pending = "pending"
    planning = "planning"
    planned = "planned"
    awaiting_approval = "awaiting_approval"
    running = "running"
    completed = "completed"
    failed = "failed"
    interrupted = "interrupted"
    cancelled = "cancelled"


class ResearchCreate(BaseModel):
    query: str = Field(min_length=10, max_length=4000)
    mode: ResearchMode = ResearchMode.web
    depth: ResearchDepth = ResearchDepth.standard
    require_plan_approval: bool = False
    document_ids: list[str] = Field(default_factory=list, max_length=20)
    start: bool = True  # begin the workflow immediately (otherwise call /resume later)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("query must contain at least 10 non-whitespace characters")
        return v


class ResearchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    query: str
    mode: ResearchMode
    depth: ResearchDepth
    status: ResearchStatus
    revision_count: int
    token_usage: int
    estimated_cost_usd: float
    require_plan_approval: bool = False
    plan_approved: bool = False
    document_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    @field_validator("document_ids", "warnings", mode="before")
    @classmethod
    def _none_to_list(cls, v):
        return [] if v is None else v


class AgentEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    agent: str
    event_type: str
    summary: str
    payload: dict | None = None
    duration_ms: int | None = None
    created_at: datetime


class ResearchStatusOut(BaseModel):
    id: str
    status: ResearchStatus
    revision_count: int
    error: str | None = None


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
