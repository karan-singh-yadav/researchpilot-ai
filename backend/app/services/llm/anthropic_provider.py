"""Anthropic Messages API adapter (plain HTTP via httpx; no SDK dependency)."""

import httpx

from app.services.llm.base import LLMError, LLMResponse, LLMUsage

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout: float = 60.0,
        base_url: str = API_URL,
        client: httpx.Client | None = None,
    ) -> None:
        if not model:
            raise ValueError("LLM_MODEL must be set for the anthropic provider")
        if not api_key:
            raise ValueError("LLM_API_KEY must be set for the anthropic provider")
        self.model = model
        self._key = api_key
        self._url = base_url
        self._client = client or httpx.Client(timeout=timeout)

    def complete(
        self,
        *,
        system: str,
        user: str,
        task: str = "generic",
        max_tokens: int = 2000,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        # No native JSON mode here: callers must instruct JSON in the prompt (the structured
        # layer validates and repairs the output regardless).
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self._key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        try:
            r = self._client.post(self._url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMError("LLM request timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise LLMError("LLM request failed (network error)", retryable=True) from exc
        if r.status_code in (429, 529) or r.status_code >= 500:
            raise LLMError(
                f"LLM provider returned HTTP {r.status_code}",
                retryable=True,
                status_code=r.status_code,
            )
        if r.status_code >= 400:
            raise LLMError(
                f"LLM provider rejected the request (HTTP {r.status_code})",
                status_code=r.status_code,
            )
        try:
            data = r.json()
            text = "".join(b["text"] for b in data["content"] if b.get("type") == "text")
            usage = data.get("usage") or {}
            u = LLMUsage(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("LLM provider returned a malformed response") from exc
        return LLMResponse(text=text, usage=u, model=self.model, provider=self.name)
