"""Runs the research graph in the background; supports cancel, resume and crash recovery.

Resumption is real: graph state is checkpointed after every step (SQLite file by default), so a
failed, cancelled, timed-out or restarted run continues from its last completed step instead of
repeating work. Cancellation/cancel flags are in-process (single-instance deployment).
"""

import os
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.budgets import budget_for
from app.config import Settings
from app.errors import AppError
from app.graph.builder import build_graph
from app.graph.context import RunCancelled, RunContext, WorkflowTimeout
from app.logging_config import get_logger
from app.models import ResearchSession
from app.services import persistence
from app.services.events import TERMINAL, EventBus, record_event
from app.services.llm import LLMProvider, build_llm_provider
from app.tools.factory import build_fetcher, build_search_provider
from app.tools.fetcher import PageFetcher
from app.tools.search import SearchProvider, ToolNotConfiguredError

log = get_logger(__name__)
RESUMABLE = {
    "pending",
    "planned",
    "awaiting_approval",
    "failed",
    "cancelled",
    "interrupted",
    "running",
}


class ToolServerUnavailable(RuntimeError):
    """The MCP tool server could not be started or reached."""


@dataclass
class Tools:
    llm: LLMProvider
    search: SearchProvider
    fetcher: PageFetcher
    documents: object | None = None  # ChunkSearcher: in-process Retriever or MCP adapter


def default_tools_factory(settings: Settings) -> Callable[[], Tools]:
    def factory() -> Tools:
        return Tools(
            build_llm_provider(settings), build_search_provider(settings), build_fetcher(settings)
        )

    return factory


def make_checkpointer(path: str):
    if path == ":memory:":
        return InMemorySaver()
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


class RunManager:
    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        settings: Settings,
        tools_factory: Callable[[], Tools],
        bus: EventBus,
        checkpointer,
        executor: str | None = None,
        timeout_override: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.tools_factory = tools_factory
        self.bus = bus
        self.checkpointer = checkpointer
        self.mode = executor or settings.run_executor
        self.timeout_override = timeout_override
        self.sleep = sleep
        self._pool = (
            ThreadPoolExecutor(max_workers=settings.max_concurrent_runs, thread_name_prefix="run")
            if self.mode == "thread"
            else None
        )
        self._lock = threading.Lock()
        self._active: dict[str, tuple[threading.Event, Future | None]] = {}

    # ------------------------------------------------------------------ public
    def preflight(self) -> Tools:
        """Fail fast (503) when providers are not configured."""
        try:
            return self.tools_factory()
        except ToolNotConfiguredError as exc:
            raise AppError(503, "search_not_configured", str(exc)) from exc
        except ToolServerUnavailable as exc:
            raise AppError(503, "tool_server_unavailable", str(exc)) from exc
        except ValueError as exc:
            raise AppError(503, "llm_not_configured", str(exc)) from exc

    def is_active(self, research_id: str) -> bool:
        with self._lock:
            return research_id in self._active

    def start(self, research_id: str) -> None:
        """Start a new run or resume from the last checkpoint."""
        info = persistence.load_session(self.session_factory, research_id)
        if info is None:
            raise AppError(404, "not_found", "Research session not found.")
        if info["status"] not in RESUMABLE:
            raise AppError(
                409, "invalid_state", f"Cannot run a session in state '{info['status']}'."
            )
        if (
            info["status"] == "awaiting_approval"
            and info["require_plan_approval"]
            and not info["plan_approved"]
        ):
            raise AppError(409, "plan_not_approved", "Approve the research plan before resuming.")
        tools = self.preflight()
        cancel = threading.Event()
        with self._lock:
            if research_id in self._active:
                raise AppError(409, "already_running", "A run is already in progress.")
            self._active[research_id] = (cancel, None)
        persistence.set_status(self.session_factory, research_id, "running")
        if self._pool is None:
            self._execute(research_id, tools, cancel)
            return
        fut = self._pool.submit(self._execute, research_id, tools, cancel)
        with self._lock:
            if research_id in self._active:
                self._active[research_id] = (cancel, fut)

    def cancel(self, research_id: str) -> bool:
        with self._lock:
            entry = self._active.get(research_id)
        if not entry:
            return False
        entry[0].set()
        return True

    def wait(self, research_id: str, timeout: float = 30.0) -> None:
        """Block until the run finishes (used by tests and graceful shutdown)."""
        with self._lock:
            entry = self._active.get(research_id)
        if entry and entry[1] is not None:
            entry[1].result(timeout=timeout)

    def delete_thread(self, research_id: str) -> None:
        try:
            self.checkpointer.delete_thread(research_id)
        except Exception:  # checkpoint cleanup must never block deletion
            log.warning("could not delete checkpoints for %s", research_id)

    def recover_interrupted(self) -> int:
        """After a crash/restart, runs left 'running' cannot still be executing."""
        db = self.session_factory()
        try:
            rows = db.scalars(
                select(ResearchSession).where(ResearchSession.status.in_(("running", "planning")))
            ).all()
            for r in rows:
                r.status = "interrupted"
                r.error = "The server restarted during this run. Resume to continue."
            db.commit()
            return len(rows)
        finally:
            db.close()

    def shutdown(self) -> None:
        for cancel, _ in list(self._active.values()):
            cancel.set()
        if self._pool:
            self._pool.shutdown(wait=True, cancel_futures=True)

    # ----------------------------------------------------------------- internal
    def _emit(self, research_id: str, event_type: str, summary: str, payload: dict | None = None):
        record_event(
            self.session_factory, self.bus, research_id, "system", event_type, summary, payload
        )

    def _execute(self, research_id: str, tools: Tools, cancel: threading.Event) -> None:
        sf = self.session_factory
        end_status, end_error = "failed", None
        try:
            info = persistence.load_session(sf, research_id)
            timeout = self.timeout_override or budget_for(info["depth"]).timeout_seconds
            ctx = RunContext(
                research_id=research_id,
                session_factory=sf,
                llm=tools.llm,
                search=tools.search,
                fetcher=tools.fetcher,
                documents=tools.documents,
                settings=self.settings,
                bus=self.bus,
                cancel_event=cancel,
                deadline=time.monotonic() + timeout,
                sleep=self.sleep,
            )
            graph = build_graph(ctx, self.checkpointer)
            config = {"configurable": {"thread_id": research_id}, "recursion_limit": 60}
            snap = graph.get_state(config)
            finished = bool(snap.values) and not snap.next
            if finished:  # previous run reached END (e.g. logical failure): restart cleanly
                self.checkpointer.delete_thread(research_id)
                snap = graph.get_state(config)
            if not snap.values:
                self._emit(research_id, "run_start", f"run started (timeout {int(timeout)}s)")
                graph_input = {
                    "research_id": research_id,
                    "query": info["query"],
                    "mode": info["mode"],
                    "depth": info["depth"],
                    "require_approval": info["require_plan_approval"],
                    "document_ids": info["document_ids"],
                    "status": "running",
                    "iterations": 0,
                    "revisions": 0,
                    "round": 0,
                    "done_task_ids": [],
                    "sources": {},
                    "evidence": [],
                    "new_evidence": 0,
                    "warnings": [],
                }
            else:
                self._emit(research_id, "run_resume", "run resumed from checkpoint")
                if snap.next and "approval_gate" in snap.next and info["plan"]:
                    graph.update_state(
                        config, {"plan": info["plan"], "plan_approved": info["plan_approved"]}
                    )
                graph_input = None
            graph.invoke(graph_input, config)
            final = graph.get_state(config)
            if final.next:  # paused before approval_gate
                end_status = "awaiting_approval"
                persistence.set_status(sf, research_id, "awaiting_approval")
            elif final.values.get("status") == "failed":
                end_status, end_error = "failed", final.values.get("error") or "The run failed."
                persistence.set_status(sf, research_id, "failed", end_error)
            else:
                end_status = "completed"
        except RunCancelled:
            end_status, end_error = "cancelled", "Run cancelled by user. Resume to continue."
            persistence.set_status(sf, research_id, "cancelled", end_error)
        except WorkflowTimeout:
            end_status = "failed"
            end_error = "The run exceeded its time budget. Resume to continue from the last step."
            persistence.set_status(sf, research_id, "failed", end_error)
        except Exception as exc:
            log.exception("run failed research=%s", research_id)
            end_status = "failed"
            end_error = (
                f"The workflow stopped because of an error ({type(exc).__name__}). Resume to retry."
            )
            try:
                persistence.set_status(sf, research_id, "failed", end_error)
            except Exception:
                log.exception("could not record failure for %s", research_id)
        finally:
            with self._lock:
                self._active.pop(research_id, None)
            try:
                self._emit(
                    research_id,
                    TERMINAL,
                    f"run ended: {end_status}",
                    {"status": end_status, "error": end_error},
                )
            except Exception:
                log.exception("could not emit run_end for %s", research_id)
