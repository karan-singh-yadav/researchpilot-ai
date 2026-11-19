"""Harness for workflow tests: temp SQLite DB, real graph, fixture tools, instrumented providers."""

import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.database import Base, enable_sqlite_fks
from app.models import AgentEvent, ResearchSession
from app.services.events import EventBus
from app.services.llm import LLMResponse, LLMUsage
from app.services.llm.fake import FakeLLM
from app.services.runner import RunManager, Tools, make_checkpointer
from app.tools.fetcher import FixtureFetcher
from app.tools.search import FixtureSearch
from tests.helpers import FIXTURE

Q = "Research the latest developments in agentic AI. Compare frameworks and limitations."


@dataclass
class Env:
    engine: object
    SF: object
    path: Path


def make_env(tmp_path: Path) -> Env:
    import app.models  # noqa: F401

    path = tmp_path / "t.db"
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30}
    )
    enable_sqlite_fks(engine)
    Base.metadata.create_all(engine)
    return Env(engine, sessionmaker(bind=engine, autoflush=False, expire_on_commit=False), path)


def new_research(env, *, query=Q, mode="web", depth="quick", approval=False, document_ids=None):
    db = env.SF()
    r = ResearchSession(
        query=query,
        mode=mode,
        depth=depth,
        status="pending",
        require_plan_approval=approval,
        document_ids=document_ids,
    )
    db.add(r)
    db.commit()
    rid = r.id
    db.close()
    return rid


def row(env, rid) -> ResearchSession:
    db = env.SF()
    try:
        return db.get(ResearchSession, rid)
    finally:
        db.close()


def events(env, rid):
    db = env.SF()
    try:
        return list(
            db.query(AgentEvent)
            .filter_by(session_id=rid)
            .order_by(AgentEvent.created_at, AgentEvent.id)
            .all()
        )
    finally:
        db.close()


class CountingLLM:
    """FakeLLM wrapper: counts calls per task and lets tests override individual calls.

    overrides[task] = fn(call_no, kwargs) -> str reply | None (delegate) | raises
    """

    name = "fake"
    model = "fake-deterministic-v1"

    def __init__(self, overrides=None):
        self.inner = FakeLLM()
        self.calls: Counter = Counter()
        self.overrides = overrides or {}
        self._lock = threading.Lock()

    def complete(self, **kw):
        task = kw["task"]
        with self._lock:
            self.calls[task] += 1
            n = self.calls[task]
        fn = self.overrides.get(task)
        if fn:
            out = fn(n, kw)
            if out is not None:
                return LLMResponse(out, LLMUsage(5, 5), self.model, self.name)
        return self.inner.complete(**kw)


class InstrumentedSearch:
    """Wraps FixtureSearch: records (query, start, end), optional delay and blocking gate."""

    name = "fixture"

    def __init__(self, delay=0.0, gate=None, started=None, hits_for=None):
        self.inner = FixtureSearch.from_path(FIXTURE)
        self.delay, self.gate, self.started = delay, gate, started
        self.hits_for = hits_for  # optional fn(query) -> list[SearchHit]
        self.log: list[tuple[str, float, float]] = []
        self._lock = threading.Lock()

    @property
    def queries(self):
        return [q for q, _, _ in self.log]

    def search(self, query, n):
        t0 = time.monotonic()
        if self.started:
            self.started.set()
        if self.gate:
            self.gate.wait(10)
        if self.delay:
            time.sleep(self.delay)
        hits = self.hits_for(query) if self.hits_for else self.inner.search(query, n)
        with self._lock:
            self.log.append((query, t0, time.monotonic()))
        return hits[:n]


def make_manager(
    env,
    *,
    llm=None,
    search=None,
    fetcher=None,
    executor="inline",
    timeout_override=None,
    checkpointer=None,
    bus=None,
    max_concurrent_runs=2,
    documents=None,
    settings_overrides=None,
):
    settings = Settings(
        _env_file=None,
        llm_provider="fake",
        search_provider="fixture",
        search_fixture_path=str(FIXTURE),
        run_executor=executor,
        llm_max_retries=0,
        search_max_results_per_query=7,
        max_concurrent_runs=max_concurrent_runs,
        **(settings_overrides or {}),
    )
    tools = Tools(
        llm or CountingLLM(),
        search or InstrumentedSearch(),
        fetcher or FixtureFetcher.from_path(FIXTURE),
        documents=documents,
    )
    return RunManager(
        session_factory=env.SF,
        settings=settings,
        tools_factory=lambda: tools,
        bus=bus or EventBus(),
        checkpointer=checkpointer or make_checkpointer(":memory:"),
        executor=executor,
        timeout_override=timeout_override,
        sleep=lambda s: None,
    )
