"""Chooses how agents reach their tools: in-process ("direct") or through MCP ("mcp")."""

import threading

from app.config import Settings
from app.mcp.client import (
    AgentToolbox,
    McpDocumentSearcher,
    McpFetcher,
    McpSearchProvider,
    McpToolClient,
    McpToolError,
    stdio_params,
)
from app.services.llm import build_llm_provider
from app.services.runner import Tools, ToolServerUnavailable, default_tools_factory
from app.tools.factory import build_search_provider


class ToolHub:
    """Owns the (single, shared) MCP client when TOOL_TRANSPORT=mcp."""

    def __init__(self, settings: Settings, retriever=None) -> None:
        self.settings = settings
        self.retriever = retriever
        self._lock = threading.Lock()
        self._client: McpToolClient | None = None
        self._direct = default_tools_factory(settings)

    def factory(self) -> Tools:
        s = self.settings
        if s.tool_transport.lower() != "mcp":
            tools = self._direct()
            tools.documents = self.retriever
            return tools
        llm = build_llm_provider(s)
        build_search_provider(s)  # validate configuration now (raises ToolNotConfiguredError)
        client = self._ensure_client()
        return Tools(
            llm,
            McpSearchProvider(AgentToolbox(client, "search")),
            McpFetcher(AgentToolbox(client, "reader"), max_chars=s.reader_max_chars),
            documents=McpDocumentSearcher(AgentToolbox(client, "document")),
        )

    def _ensure_client(self) -> McpToolClient:
        with self._lock:
            if self._client is None:
                self._client = McpToolClient(
                    stdio_params(self.settings),
                    call_timeout=self.settings.mcp_tool_timeout_seconds + 10,
                    connect_timeout=self.settings.mcp_connect_timeout_seconds,
                )
            try:
                self._client.start()
            except McpToolError as exc:
                raise ToolServerUnavailable(
                    f"The MCP tool server is unavailable: {exc.message}"
                ) from exc
            return self._client

    def close(self) -> None:
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()
