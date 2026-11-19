"""Per-run context shared by graph nodes: providers, event emission, cancellation, deadline."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.config import Settings
from app.services.events import EventBus, record_event
from app.services.llm import LLMProvider
from app.tools.fetcher import PageFetcher
from app.tools.search import SearchProvider


class RunCancelled(Exception):
    pass


class WorkflowTimeout(Exception):
    pass


@dataclass
class RunContext:
    research_id: str
    session_factory: Callable[[], Session]
    llm: LLMProvider
    search: SearchProvider
    fetcher: PageFetcher
    settings: Settings
    documents: object | None = None
    bus: EventBus | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    deadline: float | None = None  # time.monotonic() value
    sleep: Callable[[float], None] = time.sleep

    def check_cancelled(self) -> None:
        """Mid-task check: only user cancellation interrupts work that is already in flight."""
        if self.cancel_event.is_set():
            raise RunCancelled()

    def check(self) -> None:
        """Node-boundary check: cancellation AND the time budget. The budget is soft inside a
        node so finished work is checkpointed (not discarded) and a resume does not repeat it."""
        self.check_cancelled()
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise WorkflowTimeout()

    def emit(
        self,
        agent: str,
        event_type: str,
        summary: str,
        payload: dict | None = None,
        duration_ms: int | None = None,
    ) -> dict:
        return record_event(
            self.session_factory,
            self.bus,
            self.research_id,
            agent,
            event_type,
            summary,
            payload,
            duration_ms,
        )
