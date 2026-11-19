"""Typed shared state for the research graph (JSON-serialisable, so it can be checkpointed)."""

import operator
from typing import Annotated, Any, TypedDict


def batches_reducer(old: list | None, new: Any) -> list:
    """Append worker batches from parallel branches; a {"__reset__": True} update clears them."""
    if isinstance(new, dict) and new.get("__reset__"):
        return []
    return (old or []) + (new if isinstance(new, list) else [new])


def concat(old: list | None, new: list | None) -> list:
    return (old or []) + (new or [])


class ResearchState(TypedDict, total=False):
    # inputs
    research_id: str
    query: str
    mode: str
    depth: str
    require_approval: bool
    document_ids: list[str]
    # control
    status: str  # running | failed | completed
    error: str | None
    plan: dict
    plan_approved: bool
    round: int
    done_task_ids: list[str]
    iterations: int
    revisions: int
    followup_active: bool
    followup_queries: list[str]
    revise_failed: bool
    # research data
    read_batches: Annotated[list[dict], batches_reducer]
    sources: dict[str, dict]
    evidence: list[dict]
    new_evidence: int
    # outputs
    analysis: dict
    report: dict
    validation: dict
    feedback: list[str]
    warnings: Annotated[list[str], concat]


__all__ = ["ResearchState", "batches_reducer", "concat", "operator"]
