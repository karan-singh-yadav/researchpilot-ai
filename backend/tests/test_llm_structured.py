import pytest
from pydantic import BaseModel

from app.services.llm import LLMError, LLMUsage, StructuredOutputError, generate_structured
from app.services.llm.structured import extract_json
from tests.helpers import ScriptedLLM


class Thing(BaseModel):
    n: int


def run(llm, **kw):
    sleeps: list[float] = []
    res = generate_structured(llm, Thing, system="s", user="u", task="t", sleep=sleeps.append, **kw)
    return res, sleeps


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"n": 1}\n```') == '{"n": 1}'
    assert extract_json('Sure! {"n": 2} hope that helps') == '{"n": 2}'
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_valid_first_try():
    res, _ = run(ScriptedLLM(['{"n": 3}']))
    assert res.value.n == 3 and res.attempts == 1 and res.usage == LLMUsage(10, 5)


def test_repair_loop_recovers_and_sums_usage():
    llm = ScriptedLLM(["not json", '{"n": "x"}', '{"n": 7}'])
    res, _ = run(llm)
    assert res.value.n == 7 and res.attempts == 3 and res.usage == LLMUsage(30, 15)
    assert "rejected" in llm.calls[1]["user"]  # error feedback is sent back


def test_gives_up_after_max_repairs_and_reports_usage():
    llm = ScriptedLLM(["bad"] * 5)
    with pytest.raises(StructuredOutputError) as ei:
        run(llm, max_repairs=1)
    assert len(llm.calls) == 2 and ei.value.usage == LLMUsage(20, 10)


def test_business_validation_triggers_repair():
    def must_be_even(t: Thing):
        if t.n % 2:
            raise ValueError("n must be even")

    res, _ = run(ScriptedLLM(['{"n": 1}', '{"n": 2}']), validate=must_be_even)
    assert res.value.n == 2 and res.attempts == 2


def test_transport_retry_with_backoff_then_success():
    llm = ScriptedLLM(
        [LLMError("429", retryable=True), LLMError("503", retryable=True), '{"n": 1}']
    )
    res, sleeps = run(llm)
    assert res.value.n == 1 and sleeps == [0.5, 1.0]


def test_transport_retries_are_bounded():
    llm = ScriptedLLM([LLMError("x", retryable=True)] * 10)
    with pytest.raises(LLMError):
        run(llm, transport_retries=2)
    assert len(llm.calls) == 3


def test_non_retryable_error_not_retried():
    llm = ScriptedLLM([LLMError("bad key"), '{"n": 1}'])
    with pytest.raises(LLMError):
        run(llm)
    assert len(llm.calls) == 1
