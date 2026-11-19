"""Structured output: JSON extraction + Pydantic validation + bounded repair loop.

Two independent, bounded loops:
  * transport retries for transient provider errors (exponential backoff), and
  * repair attempts when the reply is not valid JSON / fails schema or business validation.
Total provider calls <= (max_repairs + 1) * (transport_retries + 1).
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from app.services.llm.base import LLMError, LLMProvider, LLMResponse, LLMUsage

T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


class StructuredOutputError(Exception):
    def __init__(self, message: str, usage: LLMUsage, last_error: str = "") -> None:
        super().__init__(message)
        self.usage = usage  # tokens already spent, so callers can still account for them
        self.last_error = last_error


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    value: T
    usage: LLMUsage
    attempts: int
    model: str
    provider: str


def extract_json(text: str) -> str:
    t = text.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end < start:
        raise ValueError("reply contains no JSON object")
    return t[start : end + 1]


def _summarize_error(exc: Exception) -> str:
    return " ".join(str(exc).split())[:800]


def _call_with_retries(
    provider: LLMProvider, kwargs: dict, retries: int, sleep: Callable[[float], None]
) -> LLMResponse:
    for attempt in range(retries + 1):
        try:
            return provider.complete(**kwargs)
        except LLMError as exc:
            if not exc.retryable or attempt == retries:
                raise
            if exc.status_code == 429:
                sleep(min(5 * 2**attempt, 30.0))
            else:
                sleep(min(0.5 * 2**attempt, 8.0))
    raise AssertionError("unreachable")  # pragma: no cover


def generate_structured(
    provider: LLMProvider,
    schema: type[T],
    *,
    system: str,
    user: str,
    task: str,
    max_repairs: int = 2,
    transport_retries: int = 2,
    max_tokens: int = 2000,
    temperature: float = 0.2,
    validate: Callable[[T], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> StructuredResult[T]:
    usage = LLMUsage()
    prompt = user
    last_error = ""
    for attempt in range(1, max_repairs + 2):
        resp = _call_with_retries(
            provider,
            {
                "system": system,
                "user": prompt,
                "task": task,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "json_mode": True,
            },
            transport_retries,
            sleep,
        )
        usage = usage + resp.usage
        try:
            value = schema.model_validate_json(extract_json(resp.text))
            if validate is not None:
                validate(value)
        except ValueError as exc:  # includes pydantic.ValidationError and json errors
            last_error = _summarize_error(exc)
            prompt = (
                f"{user}\n\nYour previous reply was rejected: {last_error}\n"
                "Return ONLY one corrected JSON object that fixes every problem listed."
            )
            continue
        return StructuredResult(value, usage, attempt, resp.model, resp.provider)
    raise StructuredOutputError(
        f"No valid structured output after {max_repairs + 1} attempts", usage, last_error
    )
