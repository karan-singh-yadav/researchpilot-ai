"""Structured inline citation parsing & verification metadata for the UI.

Parses `[S1]` and `[S2, p.4]` spans out of report Markdown so the frontend can render
interactive citation badges, jump directly to the supporting Source / Evidence passage / PDF page,
and display verification status without re-implementing regexes in TypeScript.
"""

import re
from dataclasses import dataclass

from app.models import Evidence, Source

CITE_SPAN = re.compile(r"\[(S\d+(?:,\s*p\.\s*\d+)?)\]")
SINGLE_CITE = re.compile(r"^(S\d+)(?:,\s*p\.\s*(\d+))?$")


@dataclass(frozen=True)
class ParsedCitation:
    raw: str
    start: int
    end: int
    source_ref: str
    page: int | None
    section: str


def extract_citation_spans(markdown: str) -> list[ParsedCitation]:
    """Return every inline citation span with character offsets and its Markdown heading."""
    out: list[ParsedCitation] = []
    current_section = "Header"
    offset = 0
    for line in markdown.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#"):
            current_section = stripped.lstrip("#").strip()
        if current_section.lower() == "references":
            offset += len(line)
            continue
        for m in CITE_SPAN.finditer(line):
            inner = m.group(1)
            sm = SINGLE_CITE.match(inner)
            if not sm:
                continue
            ref = sm.group(1)
            page = int(sm.group(2)) if sm.group(2) else None
            out.append(
                ParsedCitation(
                    raw=m.group(0),
                    start=offset + m.start(),
                    end=offset + m.end(),
                    source_ref=ref,
                    page=page,
                    section=current_section,
                )
            )
        offset += len(line)
    return out


def build_citation_index(markdown: str, sources: list[Source]) -> dict:
    """Resolve inline citations against persisted Sources and their Evidence items."""
    by_ref: dict[str, Source] = {s.source_ref: s for s in sources}
    spans = extract_citation_spans(markdown)

    occurrences = []
    used_refs: list[str] = []
    pages_by_ref: dict[str, set[int]] = {}

    for idx, sp in enumerate(spans, 1):
        if sp.source_ref not in used_refs:
            used_refs.append(sp.source_ref)
        if sp.page is not None:
            pages_by_ref.setdefault(sp.source_ref, set()).add(sp.page)

        src = by_ref.get(sp.source_ref)
        matching_evidence: list[Evidence] = []
        if src is not None:
            evs = list(src.evidence)
            if sp.page is not None:
                matching_evidence = [e for e in evs if e.page_number == sp.page]
            if not matching_evidence:
                matching_evidence = evs

        occurrences.append(
            {
                "id": f"c{idx}",
                "raw": sp.raw,
                "start": sp.start,
                "end": sp.end,
                "section": sp.section,
                "source_ref": sp.source_ref,
                "page": sp.page,
                "resolved": src is not None,
                "source_title": src.title if src else None,
                "source_kind": src.kind if src else None,
                "source_url": src.url if src else None,
                "document_id": src.document_id if src else None,
                "injection_suspected": bool(src.injection_suspected) if src else False,
                "evidence": [
                    {
                        "evidence_ref": e.evidence_ref,
                        "claim": e.claim,
                        "kind": e.kind,
                        "passage": e.passage,
                        "page_number": e.page_number,
                        "relevance_score": e.relevance_score,
                    }
                    for e in sorted(
                        matching_evidence,
                        key=lambda x: (-(x.relevance_score or 0.0), x.evidence_ref),
                    )
                ],
            }
        )

    source_summary = []
    for ref in used_refs:
        src = by_ref.get(ref)
        source_summary.append(
            {
                "source_ref": ref,
                "title": src.title if src else ref,
                "kind": src.kind if src else "unknown",
                "url": src.url if src else None,
                "document_id": src.document_id if src else None,
                "cited_pages": sorted(pages_by_ref.get(ref, set())),
                "citation_count": sum(1 for sp in spans if sp.source_ref == ref),
                "evidence_count": len(src.evidence) if src else 0,
                "injection_suspected": bool(src.injection_suspected) if src else False,
            }
        )

    return {
        "total_citations": len(occurrences),
        "unique_sources": len(used_refs),
        "occurrences": occurrences,
        "sources": source_summary,
    }
