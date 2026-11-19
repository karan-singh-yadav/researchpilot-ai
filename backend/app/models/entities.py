"""ORM models. String UUID primary keys keep the schema portable across PostgreSQL and SQLite."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    false,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin, new_id, utcnow


class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120), default="")
    api_key_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class ResearchSession(Base, TimestampMixin):
    __tablename__ = "research_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    query: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(16), default="web")  # web|document|hybrid
    depth: Mapped[str] = mapped_column(String(16), default="standard")  # quick|standard|deep
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    plan_approved: Mapped[bool] = mapped_column(default=False)
    require_plan_approval: Mapped[bool] = mapped_column(default=False, server_default=false())
    document_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    revision_count: Mapped[int] = mapped_column(Integer, default=0)
    token_usage: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)

    tasks = relationship("ResearchTask", cascade="all, delete-orphan", back_populates="session")
    events = relationship("AgentEvent", cascade="all, delete-orphan", back_populates="session")
    sources = relationship("Source", cascade="all, delete-orphan", back_populates="session")
    reports = relationship("Report", cascade="all, delete-orphan", back_populates="session")

    __table_args__ = (Index("ix_research_sessions_user_created", "user_id", "created_at"),)


class ResearchTask(Base, TimestampMixin):
    __tablename__ = "research_tasks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    description: Mapped[str] = mapped_column(Text)
    assigned_agent: Mapped[str] = mapped_column(String(48))
    status: Mapped[str] = mapped_column(String(24), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    session = relationship("ResearchSession", back_populates="tasks")


class AgentEvent(Base):
    """Append-only execution trace entry (concise summaries only; never raw chain-of-thought)."""

    __tablename__ = "agent_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    agent: Mapped[str] = mapped_column(String(48))
    event_type: Mapped[str] = mapped_column(String(32))  # node_start|node_end|tool_call|error|retry
    summary: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    session = relationship("ResearchSession", back_populates="events")


class UploadedDocument(Base, TimestampMixin):
    __tablename__ = "uploaded_documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    filename: Mapped[str] = mapped_column(String(255))
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)  # duplicate detection
    size_bytes: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24), default="processing")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    storage_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(160), default="", server_default="")
    embedding_dim: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    chunks = relationship(
        "DocumentChunk",
        cascade="all, delete-orphan",
        back_populates="document",
        passive_deletes=True,
    )


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    source_ref: Mapped[str] = mapped_column(String(16))  # stable citation ID e.g. "S1"
    kind: Mapped[str] = mapped_column(String(16), default="web")  # web|document
    title: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("uploaded_documents.id", ondelete="SET NULL"), nullable=True
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    provider: Mapped[str] = mapped_column(String(16), default="http", server_default="http")
    status: Mapped[str] = mapped_column(String(16), default="read", server_default="read")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    injection_suspected: Mapped[bool] = mapped_column(default=False, server_default=false())
    session = relationship("ResearchSession", back_populates="sources")
    evidence = relationship("Evidence", cascade="all, delete-orphan", back_populates="source")
    __table_args__ = (Index("uq_source_ref_per_session", "session_id", "source_ref", unique=True),)


class Evidence(Base):
    __tablename__ = "evidence"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    evidence_ref: Mapped[str] = mapped_column(String(16), default="", server_default="")
    claim: Mapped[str] = mapped_column(Text, default="", server_default="")
    kind: Mapped[str] = mapped_column(String(16), default="fact", server_default="fact")
    passage: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source = relationship("Source", back_populates="evidence")


class Report(Base, TimestampMixin):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    markdown: Mapped[str] = mapped_column(Text)
    validation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    session = relationship("ResearchSession", back_populates="reports")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    workflow_variant: Mapped[str] = mapped_column(String(48))  # baseline|improved
    dataset: Mapped[str] = mapped_column(String(120))
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DocumentChunk(Base):
    """Source of truth for retrieval: text, page number and the embedding (float32 bytes)."""

    __tablename__ = "document_chunks"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)  # "<document_id>:<index>"
    document_id: Mapped[str] = mapped_column(
        ForeignKey("uploaded_documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    document = relationship("UploadedDocument", back_populates="chunks")
    __table_args__ = (Index("ix_chunks_doc_index", "document_id", "chunk_index", unique=True),)
