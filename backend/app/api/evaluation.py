"""Observability metrics & evaluation benchmark endpoints."""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.errors import AppError
from app.evaluation.runner import compare_evaluation_runs, run_evaluation_suite
from app.models import EvaluationRun
from app.observability.metrics import compute_system_metrics
from app.security.auth import require_api_key
from app.security.rate_limit import rate_limit

router = APIRouter(
    prefix="/api",
    tags=["evaluation", "observability"],
    dependencies=[Depends(rate_limit), Depends(require_api_key)],
)


class EvalRunRequest(BaseModel):
    variant: Literal["baseline", "improved"] = "improved"
    dataset: str = Field(default="default-offline", min_length=1, max_length=120)


class EvalRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    workflow_variant: str
    dataset: str
    metrics: dict
    created_at: datetime


@router.get("/metrics")
def get_metrics(db: Session = Depends(get_db)) -> dict:
    """System-wide observability metrics: session counts, token/cost totals, latency & quality."""
    return compute_system_metrics(db)


@router.post("/evaluation/runs", response_model=EvalRunOut, status_code=201)
def trigger_evaluation(
    body: EvalRunRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> EvaluationRun:
    """Run the benchmark suite for `baseline` or `improved` and store the metrics."""
    return run_evaluation_suite(
        db, settings, variant=body.variant, dataset_name=body.dataset
    )


@router.get("/evaluation/runs", response_model=list[EvalRunOut])
def list_evaluation_runs(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[EvaluationRun]:
    return list(
        db.scalars(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
        ).all()
    )


@router.get("/evaluation/compare")
def compare_evaluations(
    baseline_id: str | None = Query(None),
    improved_id: str | None = Query(None),
    db: Session = Depends(get_db),
) -> dict:
    """Compare a baseline evaluation run against an improved run (defaults to latest of each)."""
    base = (
        db.get(EvaluationRun, baseline_id)
        if baseline_id
        else db.scalars(
            select(EvaluationRun)
            .where(EvaluationRun.workflow_variant == "baseline")
            .order_by(EvaluationRun.created_at.desc())
        ).first()
    )
    imp = (
        db.get(EvaluationRun, improved_id)
        if improved_id
        else db.scalars(
            select(EvaluationRun)
            .where(EvaluationRun.workflow_variant == "improved")
            .order_by(EvaluationRun.created_at.desc())
        ).first()
    )
    if base is None or imp is None:
        raise AppError(
            404,
            "evaluation_not_found",
            "Both a baseline and an improved evaluation run are required for comparison.",
        )
    return compare_evaluation_runs(base, imp)
