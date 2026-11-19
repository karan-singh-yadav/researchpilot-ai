"""Dependency-free HTML -> clean text + metadata extraction.

Drops scripts/styles/nav/footer/forms and any element hidden from human readers (a common
carrier for hidden prompt-injection text). Output is plain text, one paragraph per line.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser

from app.tools.urls import parse_datetime

SKIP_TAGS = {
    "script", "style", "noscript", "template", "svg", "iframe", "canvas", "nav", "footer",
    "aside", "form", "header", "button", "select", "dialog", "head",
}  # fmt: skip
BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "ul", "ol", "br", "tr", "table", "blockquote",
    "pre", "main", "figcaption", "dd", "dt", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
}  # fmt: skip
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
    "source", "track", "wbr",
}  # fmt: skip
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?![.\d])|opacity\s*:\s*0(?![.\d])",
    re.I,
)
_DATE_META = {
    "article:published_time", "og:article:published_time", "datepublished", "date", "pubdate",
    "publish-date", "dc.date", "dc.date.issued", "citation_publication_date", "citation_date",
}  # fmt: skip
_JSONLD_DATE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractedPage:
    title: str
    text: str
    published_at: datetime | None


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.skip = 0
        self.lines: list[tuple[str, bool]] = []  # (text, is_heading)
        self.cur: list[str] = []
        self.cur_heading = False
        self.title = ""
        self.og_title = ""
        self.in_title = False
        self.meta_date: str | None = None
        self.time_date: str | None = None

    def _flush(self) -> None:
        text = _WS.sub(" ", " ".join(self.cur)).strip()
        if text:
            self.lines.append((text, self.cur_heading))
        self.cur, self.cur_heading = [], False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "title":
            self.in_title = True
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            content = a.get("content", "")
            if key == "og:title" and content:
                self.og_title = content
            if key in _DATE_META and content and not self.meta_date:
                self.meta_date = content
        if tag == "time" and a.get("datetime") and not self.time_date:
            self.time_date = a["datetime"]
        if tag in VOID:
            if tag in BLOCK_TAGS and not self.skip:
                self._flush()
            return
        hidden = (
            "hidden" in a
            or a.get("aria-hidden", "").lower() == "true"
            or bool(_HIDDEN_STYLE.search(a.get("style", "")))
        )
        is_skip = tag in SKIP_TAGS or hidden
        self.stack.append((tag, is_skip))
        if is_skip:
            self.skip += 1
        elif tag in BLOCK_TAGS:
            self._flush()
            self.cur_heading = tag in HEADINGS

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        idx = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag), None)
        if idx is None:
            return
        popped = self.stack[idx:]
        del self.stack[idx:]
        self.skip -= sum(1 for _, s in popped if s)
        if tag in BLOCK_TAGS and not self.skip:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
            return
        if not self.skip:
            self.cur.append(data)


def _clean_lines(lines: list[tuple[str, bool]]) -> str:
    out: list[str] = []
    for text, is_heading in lines:
        if not is_heading and len(text.split()) < 4 and len(text) < 25:
            continue  # menu fragments, button labels, bylines
        if out and out[-1] == text:
            continue
        out.append(text)
    return "\n".join(out)


def extract_page(content_type: str, raw: str, *, max_chars: int = 200_000) -> ExtractedPage:
    if content_type == "text/plain":
        text = "\n".join(_WS.sub(" ", ln).strip() for ln in raw.splitlines() if ln.strip())
        return ExtractedPage(title="", text=text[:max_chars], published_at=None)
    parser = _Extractor()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:  # malformed markup must never crash the pipeline
        pass
    parser._flush()
    title = _WS.sub(" ", parser.og_title or parser.title).strip()
    published = parse_datetime(parser.meta_date)
    if published is None:
        m = _JSONLD_DATE.search(raw)
        published = parse_datetime(m.group(1)) if m else None
    if published is None:
        published = parse_datetime(parser.time_date)
    return ExtractedPage(
        title=title, text=_clean_lines(parser.lines)[:max_chars], published_at=published
    )
