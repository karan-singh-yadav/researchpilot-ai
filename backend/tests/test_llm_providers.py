import json

import httpx
import pytest

from app.config import Settings
from app.services.llm import LLMError, build_llm_provider
from app.services.llm.anthropic_provider import AnthropicProvider
from app.services.llm.base import LLMUsage
from app.services.llm.fake import FakeLLM
from app.services.llm.openai_compat import OpenAICompatProvider
from app.services.llm.pricing import estimate_cost_usd

SECRET = "sk-super-secret"


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_success_request_shape_and_usage():
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers["authorization"]
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    p = OpenAICompatProvider(
        model="m", api_key=SECRET, base_url="https://x.test/v1/", client=client_for(handler)
    )
    r = p.complete(system="sys", user="usr", json_mode=True)
    assert r.text == '{"ok": true}' and r.usage == LLMUsage(12, 4)
    assert seen["auth"] == f"Bearer {SECRET}"
    assert seen["url"] == "https://x.test/v1/chat/completions"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


@pytest.mark.parametrize(
    "status,retryable", [(429, True), (500, True), (503, True), (400, False), (401, False)]
)
def test_openai_error_classification_and_no_secret_leak(status, retryable):
    p = OpenAICompatProvider(
        model="m",
        api_key=SECRET,
        base_url="https://x.test/v1",
        client=client_for(lambda r: httpx.Response(status, text=f"echo {SECRET}")),
    )
    with pytest.raises(LLMError) as ei:
        p.complete(system="s", user="u")
    assert ei.value.retryable is retryable
    assert SECRET not in str(ei.value)


def test_openai_malformed_and_timeout():
    p = OpenAICompatProvider(
        model="m",
        api_key="k",
        base_url="https://x.test/v1",
        client=client_for(lambda r: httpx.Response(200, json={"nope": 1})),
    )
    with pytest.raises(LLMError, match="malformed"):
        p.complete(system="s", user="u")

    def boom(req):
        raise httpx.ReadTimeout("slow", request=req)

    p2 = OpenAICompatProvider(
        model="m", api_key="k", base_url="https://x.test/v1", client=client_for(boom)
    )
    with pytest.raises(LLMError) as ei:
        p2.complete(system="s", user="u")
    assert ei.value.retryable


def test_anthropic_request_shape_and_text_blocks():
    seen = {}

    def handler(req: httpx.Request):
        seen["key"] = req.headers["x-api-key"]
        seen["ver"] = req.headers["anthropic-version"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '{"a":'}, {"type": "text", "text": " 1}"}],
                "usage": {"input_tokens": 9, "output_tokens": 3},
            },
        )

    p = AnthropicProvider(model="m", api_key=SECRET, client=client_for(handler))
    r = p.complete(system="sys", user="usr", max_tokens=50)
    assert r.text == '{"a": 1}' and r.usage == LLMUsage(9, 3)
    assert seen["key"] == SECRET and seen["ver"] == "2023-06-01"
    assert seen["body"]["system"] == "sys" and seen["body"]["max_tokens"] == 50


def test_anthropic_529_is_retryable():
    p = AnthropicProvider(
        model="m", api_key="k", client=client_for(lambda r: httpx.Response(529, text="overloaded"))
    )
    with pytest.raises(LLMError) as ei:
        p.complete(system="s", user="u")
    assert ei.value.retryable


def test_factory():
    assert isinstance(build_llm_provider(Settings(llm_provider="fake")), FakeLLM)
    with pytest.raises(ValueError):
        build_llm_provider(Settings(llm_provider="openai", llm_model="m", llm_api_key=""))
    with pytest.raises(ValueError):
        build_llm_provider(Settings(llm_provider="openai", llm_model="", llm_api_key="k"))
    with pytest.raises(ValueError):
        build_llm_provider(Settings(llm_provider="nope"))
    local = build_llm_provider(
        Settings(llm_provider="openai", llm_model="m", llm_base_url="http://localhost:11434/v1")
    )
    assert local.name == "openai"


def test_fake_unknown_task_and_cost_estimate():
    with pytest.raises(LLMError):
        FakeLLM().complete(system="s", user="u", task="writing")
    assert estimate_cost_usd(LLMUsage(1_000_000, 500_000), 2.0, 4.0) == 4.0
    assert estimate_cost_usd(LLMUsage(1000, 1000), 0.0, 0.0) == 0.0
