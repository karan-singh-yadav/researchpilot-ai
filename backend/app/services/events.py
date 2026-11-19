"""Execution trace: every event is stored (agent_events table) AND fanned out to live subscribers.

Events carry concise operational summaries only: no prompts, no chain-of-thought, no secrets.
"""

import queue
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime

from sqlalchemy.orm import Session

from app.logging_config import get_logger
from app.models import AgentEvent

log = get_logger(__name__)
TERMINAL = "run_end"


def event_to_dict(e: AgentEvent) -> dict:
    created: datetime = e.created_at
    return {
        "id": e.id,
        "research_id": e.session_id,
        "agent": e.agent,
        "event_type": e.event_type,
        "summary": e.summary,
        "payload": e.payload,
        "duration_ms": e.duration_ms,
        "created_at": created.isoformat(),
    }


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[queue.Queue]] = defaultdict(set)
        self._lock = threading.Lock()

    def subscribe(self, research_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs[research_id].add(q)
        return q

    def unsubscribe(self, research_id: str, q: queue.Queue) -> None:
        with self._lock:
            self._subs[research_id].discard(q)
            if not self._subs[research_id]:
                del self._subs[research_id]

    def publish(self, research_id: str, event: dict) -> None:
        with self._lock:
            targets = list(self._subs.get(research_id, ()))
        for q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:  # a stalled client must never block the workflow
                log.warning("dropping event for slow subscriber research=%s", research_id)


def record_event(
    session_factory: Callable[[], Session],
    bus: EventBus | None,
    research_id: str,
    agent: str,
    event_type: str,
    summary: str,
    payload: dict | None = None,
    duration_ms: int | None = None,
) -> dict:
    db = session_factory()
    try:
        row = AgentEvent(
            session_id=research_id,
            agent=agent,
            event_type=event_type,
            summary=summary[:2000],
            payload=payload,
            duration_ms=duration_ms,
        )
        db.add(row)
        db.commit()
        data = event_to_dict(row)
    finally:
        db.close()
    if bus is not None:
        bus.publish(research_id, data)
    return data
