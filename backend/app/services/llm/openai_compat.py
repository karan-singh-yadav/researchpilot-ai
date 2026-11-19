"""OpenAI-compatible chat-completions adapter (OpenAI, vLLM, Ollama, OpenRouter, ...).

Error messages never include request headers, bodies or the API key.
"""

import httpx

from app.services.llm.base import LLMError, LLMResponse, LLMUsage


class OpenAICompatProvider:
    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not model:
            raise ValueError("LLM_MODEL must be set for the openai provider")
        self.model = model
        self._key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = client or httpx.Client(timeout=timeout)

    def _post(self, payload: dict, headers: dict) -> httpx.Response:
        try:
            return self._client.post(self._url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMError("LLM request timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise LLMError("LLM request failed (network error)", retryable=True) from exc

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
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if "groq.com" in self._url and "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        r = self._post(payload, headers)
        if r.status_code == 400 and json_mode and "json_validate_failed" in r.text:
            # Some providers (e.g. Groq) reject JSON mode when the model's JSON is imperfect.
            # Retry once without it; the caller's own JSON extraction/repair loop still applies.
            payload.pop("response_format", None)
            r = self._post(payload, headers)
        if r.status_code == 429 or r.status_code >= 500:
            raise LLMError(
                f"LLM provider returned HTTP {r.status_code}",
                retryable=True,
                status_code=r.status_code,
            )
        if r.status_code >= 400:
            detail = r.text[:300]
            if self._key:
                detail = detail.replace(self._key, "[redacted]")
            raise LLMError(
                f"LLM provider rejected the request (HTTP {r.status_code}): {detail}",
                status_code=r.status_code,
            )
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise TypeError("content is not a string")
            usage = data.get("usage") or {}
            u = LLMUsage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM provider returned a malformed response") from exc
        return LLMResponse(text=text, usage=u, model=self.model, provider=self.name)