"""Prompt-injection helpers for UNTRUSTED text (web pages, documents, evidence).

These are defence-in-depth, not a guarantee. The primary defences are structural: untrusted text
is delimited and escaped, model output is schema-validated, and agents have no side-effecting
tools that untrusted text could trigger.
"""

import re

_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)",
    r"forget\s+(everything|all)\s+(you|above|previous)",
    r"you\s+are\s+now\s+(a|an|the)\b",
    r"\bnew\s+instructions?\s*:",
    r"(reveal|print|show|repeat)\s+(your|the)\s+(system\s+)?(prompt|instructions)",
    r"<\s*/?\s*(system|assistant|developer)\s*>",
    r"\b(system|developer)\s+(prompt|message)\s*:",
    r"do\s+not\s+(tell|inform)\s+the\s+user",
]
_RX = re.compile("|".join(_PATTERNS), re.I)


def looks_like_injection(text: str) -> bool:
    return bool(_RX.search(text or ""))


def escape_delimiters(text: str, tags: tuple[str, ...]) -> str:
    """Neutralise any opening/closing delimiter tags inside untrusted text."""
    for tag in tags:
        text = re.sub(rf"<\s*(/?)\s*{re.escape(tag)}\b", rf"[\1{tag}", text, flags=re.I)
    return text
