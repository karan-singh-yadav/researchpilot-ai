from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExecutionBudget(BaseModel):
    """Set by the server from the research depth. The LLM can never choose its own budget."""

    max_subtasks: int
    max_sources: int
    max_iterations: int  # additional-research loops
    max_revisions: int  # writer/critic revision loops
    timeout_seconds: int


class SubTask(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(pattern=r"^t\d{1,2}$")
    description: str = Field(min_length=5, max_length=400)
    agent: Literal["search", "document"]
    search_queries: list[str] = Field(min_length=1, max_length=5)
    depends_on: list[str] = Field(default_factory=list, max_length=11)

    @field_validator("description")
    @classmethod
    def _strip_desc(cls, v: str) -> str:
        return v.strip()

    @field_validator("search_queries")
    @classmethod
    def _clean_queries(cls, v: list[str]) -> list[str]:
        out = []
        for q in v:
            q = " ".join(q.split())
            if not 3 <= len(q) <= 200:
                raise ValueError("each search query must be 3-200 characters")
            out.append(q)
        return out


class PlanDraft(BaseModel):
    """What the LLM (or a human editor) supplies. Budget/mode/depth are added by the server."""

    model_config = ConfigDict(extra="ignore")
    objectives: list[str] = Field(min_length=1, max_length=6)
    subtasks: list[SubTask] = Field(min_length=1, max_length=12)
    report_outline: list[str] = Field(min_length=2, max_length=12)

    @field_validator("objectives", "report_outline")
    @classmethod
    def _clean_strings(cls, v: list[str]) -> list[str]:
        out = [" ".join(s.split()) for s in v]
        if any(not 2 <= len(s) <= 300 for s in out):
            raise ValueError("entries must be 2-300 characters")
        return out

    @model_validator(mode="after")
    def _check_graph(self) -> "PlanDraft":
        ids = [t.id for t in self.subtasks]
        if len(set(ids)) != len(ids):
            raise ValueError("subtask ids must be unique")
        known = set(ids)
        for t in self.subtasks:
            if t.id in t.depends_on:
                raise ValueError(f"subtask {t.id} depends on itself")
            missing = [d for d in t.depends_on if d not in known]
            if missing:
                raise ValueError(f"subtask {t.id} depends on unknown ids {missing}")
        # cycle detection (Kahn)
        indeg = {t.id: len(set(t.depends_on)) for t in self.subtasks}
        ready = [i for i, d in indeg.items() if d == 0]
        seen = 0
        while ready:
            cur = ready.pop()
            seen += 1
            for t in self.subtasks:
                if cur in t.depends_on:
                    indeg[t.id] -= 1
                    if indeg[t.id] == 0:
                        ready.append(t.id)
        if seen != len(self.subtasks):
            raise ValueError("subtask dependencies contain a cycle")
        return self


class PlanPatch(BaseModel):
    """Partial HITL update to an existing plan (objectives, subtasks, or outline)."""

    model_config = ConfigDict(extra="ignore")
    objectives: list[str] | None = Field(default=None, min_length=1, max_length=6)
    subtasks: list[SubTask] | None = Field(default=None, min_length=1, max_length=12)
    report_outline: list[str] | None = Field(default=None, min_length=2, max_length=12)


class PlanApproveRequest(BaseModel):
    """Optional human note recorded in the audit trace when approving a plan."""

    note: str | None = Field(default=None, max_length=500)


class PlanRejectRequest(BaseModel):
    """Human feedback when rejecting/requesting replanning during HITL review."""

    feedback: str = Field(min_length=3, max_length=1000)


class ResearchPlan(PlanDraft):
    budget: ExecutionBudget
    mode: str
    depth: str
    generated_by: str  # e.g. "fake:fake-deterministic-v1" so fake plans are never mistaken for LLM


class PlanOut(BaseModel):
    research_id: str
    status: str
    requires_approval: bool
    plan_approved: bool
    plan: ResearchPlan | None
