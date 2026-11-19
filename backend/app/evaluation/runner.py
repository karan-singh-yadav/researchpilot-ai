"""Automated offline/live evaluation harness comparing `baseline` vs `improved` workflows.

- `baseline`: sequential reference pipeline (`services/pipeline.py`, no critic/revision loop)
- `improved`: full LangGraph workflow (`graph/builder.py`, parallel waves, critic, revision loop)

Measures task completion, citation validity, source coverage, validation pass rate, token usage,
and wall-clock latency, storing structured results in `evaluation_runs` for regression tracking.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.critic import CriticAgent
from app.agents.writer import check_draft_detailed
from app.config import Settings
from app.database import Base, enable_sqlite_fks
from app.models import EvaluationRun, Report, ResearchSession, Source
from app.services.events import EventBus
from app.services.llm import LLMProvider, build_llm_provider
from app.services.pipeline import run_pipeline
from app.services.runner import RunManager, Tools, make_checkpointer
from app.tools.factory import build_fetcher, build_search_provider

DEFAULT_CASES = [
    {
        "id": "agentic-overview",
        "query": (
            "Research the latest developments in agentic AI frameworks and compare architectures."
        ),
        "depth": "quick",
        "expected_terms": ["agentic", "framework"],
    },
    {
        "id": "agentic-risks",
        "query": (
            "Analyze limitations, prompt injection risks, and operational safeguards in agentic AI."
        ),
        "depth": "quick",
        "expected_terms": ["risk", "limitation"],
    },
]


@dataclass
class CaseMetric:
    case_id: str
    query: str
    completed: bool
    citation_count: int
    deterministic_errors: int
    critic_passed: bool
    tokens: int
    latency_ms: int
    term_recall: float


def load_benchmark_cases(dataset_path: str | Path | None = None) -> list[dict]:
    if dataset_path:
        p = Path(dataset_path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "cases" in data:
                return data["cases"]
            if isinstance(data, list):
                return data
    return list(DEFAULT_CASES)


def _term_recall(markdown: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    low = markdown.lower()
    hits = sum(1 for t in expected_terms if t.lower() in low)
    return round(hits / len(expected_terms), 4)


def run_evaluation_suite(
    db: Session,
    settings: Settings,
    *,
    variant: Literal["baseline", "improved"] = "improved",
    dataset_name: str = "default-offline",
    cases: list[dict] | None = None,
    llm_override: LLMProvider | None = None,
) -> EvaluationRun:
    """Execute benchmark cases under `baseline` or `improved` and save an EvaluationRun."""
    bench = cases or load_benchmark_cases()
    llm = llm_override or build_llm_provider(settings)
    search = build_search_provider(settings)
    fetcher = build_fetcher(settings)

    results: list[CaseMetric] = []
    for case in bench:
        cid = case.get("id", "case")
        query = case["query"]
        depth = case.get("depth", "quick")
        expected = case.get("expected_terms", [])
        t0 = time.perf_counter()

        if variant == "baseline":
            try:
                res = run_pipeline(
                    query,
                    llm=llm,
                    search=search,
                    fetcher=fetcher,
                    depth=depth,
                    results_per_query=settings.search_max_results_per_query,
                )
                ms = int((time.perf_counter() - t0) * 1000)
                allowed = {r for r, s in res.sources.items() if s.status == "read"}
                det_errs = check_draft_detailed(res.report.draft, allowed, res.evidence)
                critic_res = CriticAgent(llm, transport_retries=0).review(
                    draft=res.report.draft,
                    analysis=res.analysis,
                    evidence=res.evidence,
                    sources=res.sources,
                    depth=depth,
                )
                results.append(
                    CaseMetric(
                        case_id=cid,
                        query=query,
                        completed=True,
                        citation_count=len(res.report.cited),
                        deterministic_errors=len(det_errs),
                        critic_passed=critic_res.report.passed,
                        tokens=res.usage.total + critic_res.usage.total,
                        latency_ms=ms,
                        term_recall=_term_recall(res.report.markdown, expected),
                    )
                )
            except Exception:
                ms = int((time.perf_counter() - t0) * 1000)
                results.append(
                    CaseMetric(cid, query, False, 0, 1, False, 0, ms, 0.0)
                )
        else:
            # Run isolated in-memory LangGraph workflow so benchmarks never pollute user sessions
            eng = create_engine(
                "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
            )
            enable_sqlite_fks(eng)
            Base.metadata.create_all(eng)
            sf = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False)
            sess_db = sf()
            rs = ResearchSession(query=query, mode="web", depth=depth, status="pending")
            sess_db.add(rs)
            sess_db.commit()
            rid = rs.id
            sess_db.close()

            mgr = RunManager(
                session_factory=sf,
                settings=settings,
                tools_factory=lambda: Tools(llm, search, fetcher),
                bus=EventBus(),
                checkpointer=make_checkpointer(":memory:"),
                executor="inline",
            )
            mgr.start(rid)
            ms = int((time.perf_counter() - t0) * 1000)

            check_db = sf()
            fin = check_db.get(ResearchSession, rid)
            rep = check_db.scalars(
                select(Report).where(Report.session_id == rid).order_by(Report.version.desc())
            ).first()
            srcs = list(check_db.scalars(select(Source).where(Source.session_id == rid)).all())
            val = (rep.validation or {}) if rep else {}
            det_err_cnt = sum(
                1
                for i in val.get("issues", [])
                if i.get("severity") == "error" and i.get("method") == "deterministic"
            )
            results.append(
                CaseMetric(
                    case_id=cid,
                    query=query,
                    completed=bool(fin and fin.status == "completed" and rep is not None),
                    citation_count=len(srcs),
                    deterministic_errors=det_err_cnt,
                    critic_passed=bool(val.get("passed", False)),
                    tokens=fin.token_usage if fin else 0,
                    latency_ms=ms,
                    term_recall=_term_recall(rep.markdown if rep else "", expected),
                )
            )
            check_db.close()
            eng.dispose()

    n = max(1, len(results))
    summary = {
        "cases_total": len(results),
        "completion_rate": round(sum(1 for r in results if r.completed) / n, 4),
        "critic_pass_rate": round(sum(1 for r in results if r.critic_passed) / n, 4),
        "avg_deterministic_errors": round(sum(r.deterministic_errors for r in results) / n, 4),
        "avg_citations": round(sum(r.citation_count for r in results) / n, 2),
        "avg_term_recall": round(sum(r.term_recall for r in results) / n, 4),
        "avg_tokens": round(sum(r.tokens for r in results) / n, 1),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
        "cases": [
            {
                "case_id": r.case_id,
                "query": r.query,
                "completed": r.completed,
                "citation_count": r.citation_count,
                "deterministic_errors": r.deterministic_errors,
                "critic_passed": r.critic_passed,
                "tokens": r.tokens,
                "latency_ms": r.latency_ms,
                "term_recall": r.term_recall,
            }
            for r in results
        ],
    }

    row = EvaluationRun(workflow_variant=variant, dataset=dataset_name, metrics=summary)
    db.add(row)
    db.flush()
    return row


def compare_evaluation_runs(baseline: EvaluationRun, improved: EvaluationRun) -> dict:
    """Compute metric deltas between a baseline run and an improved run."""
    bm, im = baseline.metrics, improved.metrics
    keys = [
        "completion_rate",
        "critic_pass_rate",
        "avg_deterministic_errors",
        "avg_citations",
        "avg_term_recall",
        "avg_tokens",
        "avg_latency_ms",
    ]
    deltas = {}
    for k in keys:
        bv = float(bm.get(k, 0.0))
        iv = float(im.get(k, 0.0))
        deltas[k] = {
            "baseline": bv,
            "improved": iv,
            "delta": round(iv - bv, 4),
        }
    return {
        "baseline_id": baseline.id,
        "improved_id": improved.id,
        "dataset": improved.dataset,
        "deltas": deltas,
    }
