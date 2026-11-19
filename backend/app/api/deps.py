from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.errors import AppError
from app.services.events import EventBus
from app.services.llm import LLMProvider, build_llm_provider
from app.services.runner import RunManager


def get_llm(settings: Settings = Depends(get_settings)) -> LLMProvider:
    try:
        return build_llm_provider(settings)
    except ValueError as exc:  # misconfiguration (missing key/model, unknown provider)
        raise AppError(503, "llm_not_configured", str(exc)) from exc


def get_run_manager(request: Request) -> RunManager:
    return request.app.state.run_manager


def get_bus(request: Request) -> EventBus:
    return request.app.state.bus


def get_documents(request: Request):
    svc = getattr(request.app.state, "documents", None)
    if svc is None:
        raise AppError(
            503,
            "documents_not_configured",
            "Document processing is not configured (check EMBEDDING_* settings).",
        )
    return svc


def get_retriever(request: Request):
    r = getattr(request.app.state, "retriever", None)
    if r is None:
        raise AppError(
            503,
            "documents_not_configured",
            "Document retrieval is not configured (check EMBEDDING_* settings).",
        )
    return r
