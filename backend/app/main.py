from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents as documents_api
from app.api import evaluation as evaluation_api
from app.api import graph as graph_api
from app.api import health, research
from app.config import get_settings
from app.database import SessionLocal
from app.errors import register_error_handlers
from app.logging_config import configure_logging, get_logger
from app.rag.factory import build_rag
from app.rag.service import DocumentService
from app.services.events import EventBus
from app.services.runner import RunManager, make_checkpointer
from app.services.tooling import ToolHub

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    s = get_settings()
    if not s.api_key_set:
        log.warning("API_KEYS is empty: authentication is DISABLED (development only).")
    bus = EventBus()
    app.state.documents = app.state.retriever = None
    rag = None
    try:
        rag = build_rag(s, SessionLocal)
        app.state.documents = DocumentService(SessionLocal, s, rag.embedder, rag.index)
        app.state.retriever = rag.retriever
    except Exception as exc:  # misconfigured embeddings must not stop the rest of the API
        log.warning("document features disabled: %s", exc)
    hub = ToolHub(s, retriever=rag.retriever if rag else None)
    manager = RunManager(
        session_factory=SessionLocal,
        settings=s,
        tools_factory=hub.factory,
        bus=bus,
        checkpointer=make_checkpointer(s.checkpoint_path),
    )
    app.state.bus, app.state.run_manager, app.state.tool_hub = bus, manager, hub
    try:
        recovered = manager.recover_interrupted()
        if recovered:
            log.warning("marked %d interrupted run(s); they can be resumed", recovered)
    except Exception:  # e.g. migrations not applied yet; /health reports that separately
        log.warning("could not check for interrupted runs (is the schema migrated?)")
    log.info("ResearchPilot AI started env=%s", s.app_env)
    yield
    manager.shutdown()
    hub.close()
    if app.state.documents is not None:
        app.state.documents.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ResearchPilot AI",
        version="1.0.0",
        description="Multi-agent research & intelligence system.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(research.router)
    app.include_router(graph_api.router)
    app.include_router(documents_api.router)
    app.include_router(evaluation_api.router)
    return app


app = create_app()
