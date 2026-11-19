from functools import lru_cache

from app.config import Settings
from app.services.llm.anthropic_provider import AnthropicProvider
from app.services.llm.base import LLMProvider
from app.services.llm.fake import FakeLLM
from app.services.llm.openai_compat import OpenAICompatProvider

DEFAULT_OPENAI_URL = "https://api.openai.com/v1"


@lru_cache(maxsize=8)
def _build(provider: str, model: str, api_key: str, base_url: str, timeout: float) -> LLMProvider:
    if provider == "fake":
        return FakeLLM()
    if provider == "openai":
        if not api_key and base_url.rstrip("/") == DEFAULT_OPENAI_URL:
            raise ValueError("LLM_API_KEY must be set for the openai provider")
        return OpenAICompatProvider(
            model=model, api_key=api_key, base_url=base_url, timeout=timeout
        )
    if provider == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key, timeout=timeout)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (expected fake, openai or anthropic)")


def build_llm_provider(settings: Settings) -> LLMProvider:
    return _build(
        settings.llm_provider.lower(),
        settings.llm_model,
        settings.llm_api_key,
        settings.llm_base_url,
        settings.llm_timeout_seconds,
    )
