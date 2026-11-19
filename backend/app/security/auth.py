"""API-key authentication. Keys come from settings only; comparison is constant-time."""

import hmac

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.errors import AppError


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> str:
    keys = settings.api_key_set
    if not keys:
        if settings.is_production:
            raise AppError(503, "auth_not_configured", "Server has no API keys configured.")
        return "dev-anonymous"  # development convenience only
    if x_api_key and any(hmac.compare_digest(x_api_key, k) for k in keys):
        return "api-key-user"
    raise AppError(401, "unauthorized", "Missing or invalid API key.")
