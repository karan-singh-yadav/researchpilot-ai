"""MCP client used by the agents: a synchronous bridge over the async MCP SDK.

One persistent session (a stdio subprocess running `python -m app.mcp.server`, or an in-process
server object in tests) lives on a dedicated event-loop thread; any number of worker threads can
call tools concurrently (JSON-RPC multiplexing). If the transport dies the session is restarted once
and the call retried (all tools are read-only/idempotent). Every response is re-validated against
the published output models.
"""

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.logging_config import get_logger
from app.mcp.errors import parse_error_text
from app.mcp.schemas import (
    EXPORT_RESEARCH_REPORT,
    FETCH_WEBPAGE,
    OUTPUT_MODELS,
    RETRIEVE_DOCUMENT_CHUNKS,
    SEARCH_DOCUMENTS,
    SEARCH_WEB,
)
from app.rag.retriever import RetrievalError
from app.rag.types import DocumentChunk
from app.tools.fetcher import FetchError, FetchResult
from app.tools.search import SearchError, SearchHit

log = get_logger(__name__)

# Least privilege: which agent may call which tool. Enforced client-side before any request is sent
# (a stdio server has a single client and cannot tell agents apart; the server-side allow-list
# MCP_ENABLED_TOOLS is the second, independent layer).
AGENT_PERMISSIONS: dict[str, frozenset[str]] = {
    "search": frozenset({SEARCH_WEB}),
    "reader": frozenset({FETCH_WEBPAGE}),
    "document": frozenset({SEARCH_DOCUMENTS, RETRIEVE_DOCUMENT_CHUNKS}),
    "exporter": frozenset({EXPORT_RESEARCH_REPORT}),
}


class McpToolError(Exception):
    def __init__(
        self, code: str, message: str, *, retryable: bool = False, tool: str | None = None
    ) -> None:
        super().__init__(message)
        self.code, self.message, self.retryable, self.tool = code, message, retryable, tool


class ToolPermissionError(McpToolError):
    def __init__(self, agent: str, tool: str) -> None:
        super().__init__("permission_denied", f"agent '{agent}' may not call '{tool}'", tool=tool)


class _Transport(Exception):
    """Internal: the session/transport failed (as opposed to the tool reporting an error)."""


@dataclass
class _Session:
    ready: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    client: Any = None
    error: BaseException | None = None
    stop: asyncio.Event | None = None


class McpToolClient:
    def __init__(
        self,
        target: StdioServerParameters | Any | Callable[[], Any],
        *,
        call_timeout: float = 60.0,
        connect_timeout: float = 30.0,
        max_restarts: int = 1,
    ) -> None:
        self._target = target
        self.call_timeout = call_timeout
        self.connect_timeout = connect_timeout
        self.max_restarts = max_restarts
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: _Session | None = None
        self._closed = False
        self.restarts = 0

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise McpToolError("unavailable", "MCP client is closed")
            if self._loop is None:
                ready = threading.Event()

                def run() -> None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self._loop = loop
                    ready.set()
                    loop.run_forever()
                    loop.close()

                self._thread = threading.Thread(target=run, name="mcp-client", daemon=True)
                self._thread.start()
                ready.wait(10)
            if self._session is None or self._session.done.is_set():
                if self._session is not None:  # the previous session died: this is a reconnect
                    self.restarts += 1
                    log.warning("MCP session was closed; reconnecting")
                self._session = self._open_session()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sess, loop, thread = self._session, self._loop, self._thread
            self._session = None
        if sess is not None:
            self._stop_session(sess)
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> "McpToolClient":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _target_obj(self):
        return (
            self._target()
            if callable(self._target) and not hasattr(self._target, "run")
            else self._target
        )

    def _open_session(self) -> _Session:
        sess = _Session()

        async def runner() -> None:
            try:
                sess.stop = asyncio.Event()
                async with Client(self._target_obj(), read_timeout_seconds=self.call_timeout) as c:
                    sess.client = c
                    sess.ready.set()
                    await sess.stop.wait()
            except BaseException as exc:  # noqa: BLE001 - reported via sess.error
                sess.error = exc
            finally:
                sess.client = None
                sess.ready.set()
                sess.done.set()

        asyncio.run_coroutine_threadsafe(runner(), self._loop)
        if not sess.ready.wait(self.connect_timeout):
            self._stop_session(sess)
            raise McpToolError("unavailable", "MCP server did not start in time", retryable=True)
        if sess.error is not None or sess.client is None:
            log.warning("MCP server failed to start: %s", type(sess.error).__name__)
            kind = type(sess.error).__name__ if sess.error else "unknown"
            raise McpToolError(
                "unavailable", f"MCP server failed to start ({kind})", retryable=True
            )
        return sess

    def _stop_session(self, sess: _Session) -> None:
        if self._loop is not None and sess.stop is not None:
            self._loop.call_soon_threadsafe(sess.stop.set)
        sess.done.wait(10)

    def _restart(self, failed: _Session) -> None:
        with self._lock:
            if self._closed:
                raise McpToolError("unavailable", "MCP client is closed")
            if self._session is not failed:  # another thread already restarted it
                return
            log.warning("restarting MCP session after a transport failure")
            self._stop_session(failed)
            self.restarts += 1
            self._session = self._open_session()

    # --------------------------------------------------------------------- calls
    def _call_once(self, sess: _Session, tool: str, arguments: dict, timeout: float):
        if sess.client is None or sess.done.is_set():
            raise _Transport()
        fut = asyncio.run_coroutine_threadsafe(
            sess.client.call_tool(tool, arguments, read_timeout_seconds=timeout), self._loop
        )
        try:
            return fut.result(timeout=timeout + 5)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise McpToolError(
                "timeout", f"{tool} timed out after {timeout:g}s", retryable=True, tool=tool
            ) from None
        except MCPError as exc:
            if exc.code == REQUEST_TIMEOUT:
                raise McpToolError(
                    "timeout", f"{tool} timed out after {timeout:g}s", retryable=True, tool=tool
                ) from None
            if exc.code == CONNECTION_CLOSED:  # the server process died or closed its pipe
                raise _Transport() from exc
            raise McpToolError(
                "internal", f"protocol error: {exc.message[:200]}", tool=tool
            ) from None
        except Exception as exc:  # closed pipe / dead subprocess / cancelled stream
            log.warning("MCP transport error on %s: %s", tool, type(exc).__name__)
            raise _Transport() from exc

    def call(
        self, tool: str, arguments: dict | None = None, *, timeout: float | None = None
    ) -> BaseModel:
        """Call a tool; returns the validated output model or raises McpToolError."""
        timeout = timeout or self.call_timeout
        self.start()
        for attempt in range(self.max_restarts + 1):
            sess = self._session
            try:
                result = self._call_once(sess, tool, arguments or {}, timeout)
                break
            except _Transport:
                if attempt == self.max_restarts or self._closed:
                    raise McpToolError(
                        "unavailable", "MCP server connection lost", retryable=True, tool=tool
                    ) from None
                self._restart(sess)
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            p = parse_error_text(text)
            raise McpToolError(p.code, p.message, retryable=p.retryable, tool=tool)
        model = OUTPUT_MODELS.get(tool)
        if model is None or result.structured_content is None:
            raise McpToolError(
                "invalid_response", f"{tool} returned no structured output", tool=tool
            )
        try:
            return model.model_validate(result.structured_content)
        except ValidationError:
            raise McpToolError(
                "invalid_response",
                f"{tool} returned data that does not match its schema",
                tool=tool,
            ) from None

    def list_tools(self):
        self.start()
        sess = self._session
        fut = asyncio.run_coroutine_threadsafe(sess.client.list_tools(), self._loop)
        return fut.result(timeout=self.call_timeout).tools


class AgentToolbox:
    """A permission-scoped view of the client for one agent."""

    def __init__(self, client: McpToolClient, agent: str) -> None:
        if agent not in AGENT_PERMISSIONS:
            raise ValueError(f"unknown agent '{agent}'")
        self.client, self.agent = client, agent
        self.allowed = AGENT_PERMISSIONS[agent]

    def call(self, tool: str, **arguments) -> BaseModel:
        if tool not in self.allowed:
            raise ToolPermissionError(self.agent, tool)
        return self.client.call(tool, arguments)


# ------------------------------------------------------------------- adapters
class McpSearchProvider:
    """SearchProvider backed by the MCP `search_web` tool (drop-in for the Search agent)."""

    name = "mcp"

    def __init__(self, toolbox: AgentToolbox) -> None:
        self.toolbox = toolbox

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        try:
            out = self.toolbox.call(SEARCH_WEB, query=query, max_results=min(max_results, 10))
        except McpToolError as exc:
            raise SearchError(exc.message, retryable=exc.retryable) from None
        return [
            SearchHit(
                title=h.title,
                url=h.url,
                snippet=h.snippet,
                published_at=h.published_at,
                provider=out.provider,
                rank=h.rank,
                query=query,
                retrieved_at=out.retrieved_at,
            )
            for h in out.hits
        ]


class McpFetcher:
    """PageFetcher backed by the MCP `fetch_webpage` tool. Pages arrive already cleaned."""

    def __init__(self, toolbox: AgentToolbox, max_chars: int = 20_000) -> None:
        self.toolbox = toolbox
        self.max_chars = max(500, min(max_chars, 50_000))

    def fetch(self, url: str) -> FetchResult:
        try:
            out = self.toolbox.call(FETCH_WEBPAGE, url=url, max_chars=self.max_chars)
        except McpToolError as exc:
            raise FetchError(exc.code, exc.message, retryable=exc.retryable) from None
        return FetchResult(
            url=out.url,
            final_url=out.final_url,
            status_code=200,
            content_type="text/plain",
            text=out.text,
            truncated=out.truncated,
            retrieved_at=out.retrieved_at,
            provider=out.provider,
            extracted=True,
            title=out.title,
            published_at=out.published_at,
        )


class McpDocumentSearcher:
    """ChunkSearcher backed by the MCP `search_documents` tool (for TOOL_TRANSPORT=mcp)."""

    def __init__(self, toolbox: AgentToolbox) -> None:
        self.toolbox = toolbox

    def search(
        self, query: str, top_k: int = 5, document_ids: list[str] | None = None
    ) -> list[DocumentChunk]:
        try:
            out = self.toolbox.call(
                SEARCH_DOCUMENTS, query=query, top_k=min(top_k, 20), document_ids=document_ids
            )
        except McpToolError as exc:
            raise RetrievalError(exc.message, retryable=exc.retryable) from None
        return out.chunks


# --------------------------------------------------------------- server launch
_PASS = (
    "search_provider", "search_api_key", "search_fixture_path", "search_max_results_per_query",
    "max_fetch_bytes", "fetch_timeout_seconds", "fetch_max_redirects", "database_url",
    "mcp_enabled_tools", "mcp_tool_timeout_seconds", "mcp_max_concurrency", "mcp_calls_per_minute",
    "mcp_max_text_chars", "mcp_max_export_bytes", "embedding_provider", "embedding_model",
    "embedding_api_key", "embedding_base_url", "embedding_dim", "embedding_batch_size",
    "reranker", "retrieval_overfetch",
)  # fmt: skip


def _absolute_sqlite(url: str) -> str:
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////") and ":memory:" not in url:
        rel = urlsplit(url).path.lstrip("/")
        return "sqlite:///" + str(Path(rel).resolve())
    return url


def stdio_params(settings: Settings) -> StdioServerParameters:
    """Launch parameters for the tool server. Only the settings the tools need are passed; in
    particular the LLM API key is NOT given to the tool server (least privilege)."""
    import sys

    env = {
        **get_default_environment(),
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "MCP_IGNORE_DOTENV": "1",
    }
    for name in _PASS:
        value = getattr(settings, name)
        if value in ("", None):
            continue
        env[name.upper()] = str(value)
    if env.get("SEARCH_FIXTURE_PATH"):
        env["SEARCH_FIXTURE_PATH"] = str(Path(env["SEARCH_FIXTURE_PATH"]).resolve())
    if "DATABASE_URL" in env:
        env["DATABASE_URL"] = _absolute_sqlite(env["DATABASE_URL"])
    backend_dir = Path(__file__).resolve().parents[2]
    return StdioServerParameters(
        command=sys.executable, args=["-m", "app.mcp.server"], env=env, cwd=backend_dir
    )
