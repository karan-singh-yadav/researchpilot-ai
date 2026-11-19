"""Central application settings, loaded from environment / .env (never hardcode secrets)."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str = "sqlite:///./data/researchpilot.db"
    api_keys: str = ""
    cors_origins: str = "http://localhost:5173"
    rate_limit_per_minute: int = Field(default=60, ge=1)
    max_upload_mb: int = Field(default=25, ge=1)
    max_fetch_bytes: int = Field(default=2_000_000, ge=1024)
    fetch_timeout_seconds: float = Field(default=10.0, gt=0)
    llm_provider: str = "fake"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"  # any OpenAI-compatible endpoint
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)  # transport retries (429/5xx/timeouts)
    # Prices are NOT built in (they change). Set them to get a cost estimate; 0 => cost not tracked.
    llm_input_cost_per_mtok: float = Field(default=0.0, ge=0)
    llm_output_cost_per_mtok: float = Field(default=0.0, ge=0)
    search_provider: str = "none"
    search_api_key: str = ""
    search_fixture_path: str = ""  # used when SEARCH_PROVIDER=fixture (offline demo/test data)
    search_max_results_per_query: int = Field(default=5, ge=1, le=20)
    fetch_max_redirects: int = Field(default=4, ge=0, le=10)
    reader_max_chars: int = Field(default=12_000, ge=1000)
    # Tool transport: "direct" = in-process calls; "mcp" = agents call tools through an MCP server
    tool_transport: str = "direct"
    mcp_enabled_tools: str = ""  # comma-separated allow-list; empty = all tools
    mcp_tool_timeout_seconds: float = Field(default=30.0, gt=0)
    mcp_max_concurrency: int = Field(default=8, ge=1, le=64)
    mcp_calls_per_minute: int = Field(default=240, ge=1)
    mcp_max_text_chars: int = Field(default=50_000, ge=1000)
    mcp_max_export_bytes: int = Field(default=2_000_000, ge=1024)
    mcp_connect_timeout_seconds: float = Field(default=30.0, gt=0)
    # Documents / RAG (Phase 6)
    upload_dir: str = "./data/uploads"
    max_pdf_pages: int = Field(default=500, ge=1)
    max_pdf_chars: int = Field(default=5_000_000, ge=1000)
    chunk_size: int = Field(default=1000, ge=200, le=8000)
    chunk_overlap: int = Field(default=150, ge=0)
    embedding_provider: str = "hash"  # hash | openai | sentence-transformers
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_dim: int = Field(default=512, ge=32, le=4096)  # hash provider only
    embedding_batch_size: int = Field(default=64, ge=1, le=512)
    vector_store: str = "sql"  # sql (default, process-safe) | chroma (optional, backend-only)
    chroma_path: str = "./data/chroma"
    reranker: str = "none"  # none | lexical
    retrieval_overfetch: int = Field(default=4, ge=1, le=20)
    checkpoint_path: str = "./data/checkpoints.sqlite"  # ":memory:" disables persistence
    max_concurrent_runs: int = Field(default=2, ge=1, le=16)
    sse_keepalive_seconds: float = Field(default=15.0, gt=0)
    run_executor: str = "thread"  # thread | inline (inline = synchronous, used by tests)
    reader_max_workers: int = Field(default=4, ge=1, le=16)

    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
