"""Deterministic OFFLINE stand-in provider for demos and tests.

It is NOT a language model: it does not understand requests. Its research plans are templated
from simple string handling so the whole pipeline can run without paid credentials. Anything
produced with it is labelled `generated_by="fake:..."`. Token counts are rough estimates
(characters / 4).
"""

import json
import re
from collections.abc import Callable

from app.services.llm.base import LLMError, LLMResponse, LLMUsage
from app.services.llm.fake_tasks import (
    analysis_handler,
    claim_review_handler,
    document_answer_handler,
    evidence_handler,
    writer_handler,
)

Handler = Callable[[str, str], str]  # (system, user) -> reply text


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _constraint(user: str, key: str, default: str = "") -> str:
    m = re.search(rf"^{key}:[ \t]*(.*)$", user, re.M)
    return m.group(1).strip() if m else default


def _plan_handler(system: str, user: str) -> str:
    m = re.search(r"<user_request>\n(.*?)\n</user_request>", user, re.S)
    request = (m.group(1) if m else "the topic").strip()
    topic = re.split(r"[.?!\n]", request, maxsplit=1)[0].strip()[:100] or "the topic"
    mode = _constraint(user, "mode", "web")
    depth = _constraint(user, "depth", "standard")
    max_sub = int(_constraint(user, "max_subtasks", "3"))
    aspects = [
        ("Survey background and recent developments", f"{topic} overview"),
        ("Compare major approaches, frameworks and architectures", f"{topic} comparison"),
        ("Identify advantages, limitations and risks", f"{topic} limitations risks"),
        ("Collect real-world applications and case studies", f"{topic} real-world applications"),
        ("Assess future opportunities and open problems", f"{topic} future outlook"),
        ("Verify key claims against primary sources", f"{topic} primary sources evidence"),
    ]
    n = min(len(aspects), max_sub, {"quick": 3, "standard": 5, "deep": 6}.get(depth, 5))
    subtasks = []
    for i, (desc, query) in enumerate(aspects[:n]):
        if mode == "document":
            agent = "document"
        elif mode == "hybrid":
            agent = "document" if i % 2 == 0 else "search"
        else:
            agent = "search"
        subtasks.append(
            {
                "id": f"t{i + 1}",
                "description": desc,
                "agent": agent,
                "search_queries": [query],
                "depends_on": ["t1"] if i >= 3 else [],
            }
        )
    return json.dumps(
        {
            "objectives": [
                f"Understand: {topic}",
                "Compare evidence across multiple credible sources",
                "Identify limitations, contradictions and gaps",
            ],
            "subtasks": subtasks,
            "report_outline": [
                "Executive summary",
                "Background",
                "Findings",
                "Comparison",
                "Limitations and risks",
                "Future outlook",
                "References",
            ],
        }
    )


class FakeLLM:
    name = "fake"
    model = "fake-deterministic-v1"

    def __init__(self, handlers: dict[str, Handler] | None = None) -> None:
        # Later phases register more task handlers here (analysis, writing, critique, ...).
        self.handlers: dict[str, Handler] = {
            "research_plan": _plan_handler,
            "evidence_extraction": evidence_handler,
            "analysis": analysis_handler,
            "report_writing": writer_handler,
            "claim_review": claim_review_handler,
            "document_answer": document_answer_handler,
            **(handlers or {}),
        }

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
        handler = self.handlers.get(task)
        if handler is None:
            raise LLMError(f"fake provider has no handler for task '{task}'")
        text = handler(system, user)
        usage = LLMUsage(_estimate_tokens(system + user), _estimate_tokens(text))
        return LLMResponse(text=text, usage=usage, model=self.model, provider=self.name)
