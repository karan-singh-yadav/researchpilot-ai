"""Deterministic Markdown rendering. Only sources actually cited appear in References."""

import re
from datetime import datetime

from app.schemas.evidence import ReportDraft, SourceRecord

# [S1] for web sources; [S2, p.4] for uploaded documents (page is mandatory for documents)
CITE = re.compile(r"\[(S\d+)(?:,\s*p\.\s*\d+)?\]")
CITE_PAGE = re.compile(r"\[(S\d+),\s*p\.\s*(\d+)\]")


def cited_refs(draft: ReportDraft) -> list[str]:
    texts = [draft.executive_summary, draft.limitations]
    texts += [s.body for s in draft.sections]
    for t in draft.comparison_tables:
        texts += [c for row in t.rows for c in row]
    seen: list[str] = []
    for t in texts:
        for ref in CITE.findall(t):
            if ref not in seen:
                seen.append(ref)
    return seen


def _all_text(draft: ReportDraft) -> list[str]:
    texts = [draft.executive_summary, draft.limitations] + [s.body for s in draft.sections]
    for t in draft.comparison_tables:
        texts += [c for row in t.rows for c in row]
    return texts


def cited_pages(draft: ReportDraft) -> dict[str, list[int]]:
    pages: dict[str, set[int]] = {}
    for t in _all_text(draft):
        for ref, page in CITE_PAGE.findall(t):
            pages.setdefault(ref, set()).add(int(page))
    return {r: sorted(p) for r, p in pages.items()}


def _link(text: str) -> str:
    return CITE.sub(lambda m: f"[{m.group(0)}](#ref-{m.group(1)})", text)


def _cell(text: str) -> str:
    return _link(text.replace("|", "\\|").replace("\n", " "))


def render_markdown(
    draft: ReportDraft,
    sources: dict[str, SourceRecord],
    *,
    notes: list[str],
    generated_at: datetime,
) -> str:
    out = [f"# {draft.title}", "", f"_Generated {generated_at:%Y-%m-%d %H:%M} UTC_", ""]
    for n in notes:
        out.append(f"> **Note:** {n}")
    if notes:
        out.append("")
    out += ["## Executive Summary", "", _link(draft.executive_summary), ""]
    for s in draft.sections:
        out += [f"## {s.heading}", "", _link(s.body), ""]
    for t in draft.comparison_tables:
        out += [f"### {t.title}", ""]
        out.append("| " + " | ".join(_cell(c) for c in t.columns) + " |")
        out.append("|" + "---|" * len(t.columns))
        out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in t.rows]
        out.append("")
    if draft.limitations.strip():
        out += ["## Limitations", "", _link(draft.limitations), ""]
    out += ["## References", ""]
    pages = cited_pages(draft)
    for ref in cited_refs(draft):
        s = sources[ref]
        if s.kind == "document":
            where = f", pp. {', '.join(map(str, pages[ref]))}" if ref in pages else ""
            warn = "; ⚠ document contained instruction-like text" if s.injection_suspected else ""
            out.append(
                f'<a id="ref-{ref}"></a>**[{ref}]** {s.title} (uploaded document{where}){warn}'
            )
            out.append("")
            continue
        pub = (
            f", published {s.published_at:%Y-%m-%d}"
            if s.published_at
            else ", publication date unknown"
        )
        out.append(
            f'<a id="ref-{ref}"></a>**[{ref}]** [{s.title or s.url}]({s.url})'
            f"{pub}; retrieved {s.retrieved_at:%Y-%m-%d}"
            + ("; ⚠ page contained instruction-like text" if s.injection_suspected else "")
        )
        out.append("")
    return "\n".join(out).rstrip() + "\n"
