"""Sequential reference pipeline: plan -> search -> read -> analyze -> (bounded follow-up) -> write.

Real agents, no persistence. It is the BASELINE workflow for Phase 8 evaluation and the offline
demo; Phase 4 replaces orchestration with a LangGraph graph (adds critic, checkpointing, SSE).
Only web-mode research is supported here; document tasks arrive with RAG in Phase 6.
"""

import time
from dataclasses import dataclass, field

from app.agents.analyst import AnalysisResult, AnalystAgent
from app.agents.reader import ReaderAgent, ReadResult
from app.agents.search import SearchAgent
from app.agents.supervisor import SupervisorAgent
from app.agents.writer import ReportResult, WriterAgent
from app.schemas.evidence import EvidenceItem, SourceRecord
from app.schemas.plan import ResearchPlan
from app.services.llm import LLMProvider, LLMUsage
from app.services.registry import SourceRegistry
from app.tools.fetcher import PageFetcher
from app.tools.search import SearchProvider


@dataclass
class PipelineResult:
    plan: ResearchPlan
    sources: dict[str, SourceRecord]
    evidence: list[EvidenceItem]
    analysis: AnalysisResult
    report: ReportResult
    usage: LLMUsage
    iterations: int
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)


def _ingest(registry: SourceRegistry, results: list[ReadResult]) -> int:
    added = 0
    for r in results:
        src = registry.add_source(
            title=r.title,
            url=r.final_url,
            published_at=r.published_at,
            retrieved_at=r.retrieved_at,
            provider=r.provider,
            status=r.status,
            error=r.error,
            injection_suspected=r.injection_suspected,
        )
        for it in r.items:
            registry.add_evidence(
                src.ref, passage=it.passage, claim=it.claim, kind=it.kind, relevance=it.relevance
            )
            added += 1
    return added


def run_pipeline(
    query: str,
    *,
    llm: LLMProvider,
    search: SearchProvider,
    fetcher: PageFetcher,
    depth: str = "quick",
    max_iterations: int | None = None,
    results_per_query: int = 5,
    sleep=time.sleep,
) -> PipelineResult:
    t: dict[str, int] = {}
    usage = LLMUsage()
    warnings: list[str] = []

    def _tick(name: str, start: float) -> None:
        t[name] = int((time.perf_counter() - start) * 1000)

    s = time.perf_counter()
    sup = SupervisorAgent(llm).plan(query, "web", depth)
    plan, usage = sup.plan, usage + sup.usage
    _tick("plan", s)

    registry = SourceRegistry()
    searcher = SearchAgent(search, results_per_query=results_per_query, sleep=sleep)
    reader = ReaderAgent(fetcher, llm)

    def gather(queries: list[str], focus: str) -> None:
        nonlocal usage
        remaining = plan.budget.max_sources - len(registry.sources)
        if remaining <= 0:
            warnings.append("source budget exhausted")
            return
        out = searcher.run(queries, max_sources=remaining, exclude_keys=registry.keys())
        warnings.extend(f"search error: {e}" for e in out.errors)
        results = reader.read_many(out.hits, focus)
        for r in results:
            usage = usage + r.usage
            if r.status == "failed":
                warnings.append(f"could not read {r.hit.url}: {r.error}")
            if r.injection_suspected:
                warnings.append(f"possible prompt injection in {r.hit.url} (treated as data)")
        _ingest(registry, results)

    s = time.perf_counter()
    done: set[str] = set()
    pending = list(plan.subtasks)
    while pending:  # topological order over depends_on
        ready = [x for x in pending if set(x.depends_on) <= done]
        for st in ready or pending:
            if st.agent == "document":
                warnings.append(f"skipped {st.id}: document research is not available yet")
            else:
                gather(
                    st.search_queries, f"{st.description} (queries: {'; '.join(st.search_queries)})"
                )
            done.add(st.id)
            pending.remove(st)
    _tick("research", s)

    analyst = AnalystAgent(llm)
    s = time.perf_counter()
    evidence = registry.evidence
    if not evidence:
        from app.agents.analyst import AnalysisError

        raise AnalysisError("No evidence could be extracted from any source", usage)
    analysis = analyst.analyze(query, plan.objectives, evidence, registry.sources)
    usage = usage + analysis.usage
    iterations = 0
    limit = plan.budget.max_iterations if max_iterations is None else max_iterations
    while analysis.needs_more_research and analysis.follow_up_queries and iterations < limit:
        iterations += 1
        before = len(registry.evidence)
        gather(analysis.follow_up_queries, "; ".join(plan.objectives))
        if len(registry.evidence) == before:
            warnings.append("follow-up research found no new evidence")
            break
        evidence = registry.evidence
        analysis = analyst.analyze(query, plan.objectives, evidence, registry.sources)
        usage = usage + analysis.usage
    _tick("analysis", s)
    if analysis.needs_more_research:
        warnings.append("evidence remained insufficient within the research budget")
    warnings.extend(analysis.warnings)

    notes = []
    providers = {src.provider for src in registry.sources.values()}
    if "fixture" in providers:
        notes.append(
            "OFFLINE FIXTURE DATA: sources below are synthetic test data, not live web results."
        )
    if llm.name == "fake":
        notes.append("Written by the offline templated provider, not a language model.")
    if analysis.needs_more_research:
        notes.append("Evidence was limited; treat conclusions with caution.")
    s = time.perf_counter()
    report = WriterAgent(llm).write(
        query, plan.report_outline, analysis, evidence, registry.sources, notes=notes
    )
    usage = usage + report.usage
    _tick("write", s)
    return PipelineResult(
        plan, registry.sources, evidence, analysis, report, usage, iterations, warnings, t
    )
