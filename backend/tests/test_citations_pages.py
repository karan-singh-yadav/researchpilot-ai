from datetime import UTC, datetime

from app.agents.critic import _claims_prompt, extract_claims
from app.agents.writer import build_user_prompt, check_draft_detailed
from app.schemas.evidence import EvidenceItem, ReportDraft, SourceRecord
from app.services.llm.fake_tasks import _cite
from app.services.report_render import CITE, CITE_PAGE, cited_pages, cited_refs, render_markdown
from tests.test_critic import analysis

NOW = datetime(2026, 10, 31, tzinfo=UTC)


def draft(summary, body="Beta uses a graph approach [S1].", limitations="", tables=None):
    return ReportDraft.model_validate(
        {
            "title": "A valid report title",
            "executive_summary": summary,
            "sections": [{"heading": "Findings", "body": body}],
            "comparison_tables": tables or [],
            "limitations": limitations,
        }
    )


EV = [
    EvidenceItem(
        id="E1", source_ref="S1", passage="Web evidence passage about graphs and agents.", claim="c"
    ),
    EvidenceItem(
        id="E2",
        source_ref="S2",
        passage="Doc evidence from page four about checkpointing.",
        claim="c",
        page_number=4,
    ),
    EvidenceItem(
        id="E3",
        source_ref="S2",
        passage="Doc evidence from page seven about budgets.",
        claim="c",
        page_number=7,
    ),
]


def problems(d):
    return check_draft_detailed(d, {"S1", "S2"}, EV)


def codes(d):
    return sorted(c for c, _ in problems(d))


def test_citation_syntax_accepts_pages_and_extracts_refs_and_pages():
    text = "A [S1]. B [S2, p.4] and [S2,p. 7]; C [S3]. Not cites: [S], [s1], [S1, page 3], (S2)."
    assert CITE.findall(text) == ["S1", "S2", "S2", "S3"]
    assert CITE_PAGE.findall(text) == [("S2", "4"), ("S2", "7")]
    d = draft("The summary text is [S1] and [S2, p.4].", body="Body [S2, p.7] and [S2, p.4].")
    assert cited_refs(d) == ["S1", "S2"] and cited_pages(d) == {"S2": [4, 7]}


def test_valid_page_citations_pass():
    assert (
        problems(
            draft("The summary text is [S1] and [S2, p.4].", body="Body [S2, p.7]. More [S1].")
        )
        == []
    )


def test_document_citations_must_carry_a_page_that_exists_in_the_evidence():
    assert "missing_page" in codes(
        draft("The summary text is [S2].", body="The body text cites [S2, p.4].")
    )
    bad = problems(draft("The summary text is [S2, p.9].", body="The body text cites [S2, p.4]."))
    assert any("pages [4, 7]" in m for _, m in bad) and codes(
        draft("The summary text is [S2, p.9].", body="The body text cites [S2, p.4].")
    ) == ["invalid_citation"]


def test_web_sources_must_not_be_cited_with_pages():
    d = draft("The summary text is [S1, p.3].", body="The body text cites [S2, p.4].")
    assert any("not a document" in m for _, m in problems(d))


def test_pages_are_checked_inside_tables_and_limitations_too():
    t = [{"title": "Cmp", "columns": ["A", "B"], "rows": [["x", "see [S2, p.1]"]]}]
    assert "invalid_citation" in codes(draft("The summary text is [S1].", tables=t))
    assert "missing_page" in codes(
        draft("The summary text is [S1].", limitations="Caveat from [S2].")
    )


def test_claim_extraction_handles_page_qualified_citations_in_all_layouts():
    d = draft(
        "Alpha is documented. [S2, p.4] Beta is described elsewhere. [S1]",
        body="[S2, p.7] Prefix style claim here.\n- Bullet claim text here [S2, p.4]",
    )
    got = [(c.text, c.refs) for c in extract_claims(d, 30)[0]]
    assert ("Alpha is documented. [S2, p.4]", ["S2"]) in got and (
        "Beta is described elsewhere. [S1]",
        ["S1"],
    ) in got
    assert ("[S2, p.7] Prefix style claim here.", ["S2"]) in got and (
        "Bullet claim text here [S2, p.4]",
        ["S2"],
    ) in got
    assert all(len(t) > 12 for t, _ in got)


def test_critic_shows_evidence_from_the_cited_page_first():
    d = draft("The summary text is [S1].", body="Claim about budgets [S2, p.7].")
    claims, _ = extract_claims(d, 30)
    c = next(c for c in claims if c.section == "Findings")
    assert c.pages == {"S2": {7}}
    prompt = _claims_prompt([c], {"S2": [e for e in EV if e.source_ref == "S2"]}, set())
    assert prompt.index("page seven") < prompt.index("page four")


def test_writer_prompt_lists_pages_for_documents_and_a_sources_legend():
    a = analysis(("E2", "E3", "E1"))
    ev_map = {e.id: e for e in EV}
    src = {
        "S1": SourceRecord(ref="S1", title="Web Page", url="https://a.test/", retrieved_at=NOW),
        "S2": SourceRecord(
            ref="S2", kind="document", title="paper.pdf", url="document://x", retrieved_at=NOW
        ),
    }
    p = build_user_prompt("q", ["Findings"], a, ev_map, None, src)
    assert "[S1,S2 p.4,S2 p.7]" in p
    assert 'S2 = "paper.pdf" (uploaded document; cite as [S2, p.N], pages [4, 7])' in p
    assert 'S1 = "Web Page" (web page; cite as [S1])' in p
    assert _cite("S2 p.4") == "[S2, p.4]" and _cite("S1") == "[S1]"


def test_report_links_keep_pages_and_references_list_pages_for_documents():
    d = draft("The summary text is [S1] and [S2, p.4].", body="Body [S2, p.7] and [S2, p.4].")
    src = {
        "S1": SourceRecord(ref="S1", title="Web Page", url="https://a.test/", retrieved_at=NOW),
        "S2": SourceRecord(
            ref="S2", kind="document", title="paper.pdf", url="document://x", retrieved_at=NOW
        ),
    }
    md = render_markdown(d, src, notes=[], generated_at=NOW)
    assert "[[S2, p.4]](#ref-S2)" in md and "[[S1]](#ref-S1)" in md
    assert (
        "**[S2]** paper.pdf (uploaded document, pp. 4, 7)" in md
        and "[Web Page](https://a.test/)" in md
    )
    assert "document://" not in md  # internal identifiers are never rendered as links
