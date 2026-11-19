"""PDF text extraction with page numbers and hard safety limits.

Handles untrusted files: header check, encryption, page/char caps, and every parser failure mapped
to a PdfError with a message that is safe to show. Scanned PDFs (no text layer) are rejected: OCR is
not implemented.
"""

import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader


class PdfError(Exception):
    """code: not_pdf | encrypted | corrupt | too_many_pages | too_much_text | no_text"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


@dataclass(frozen=True)
class PdfPage:
    number: int  # 1-based
    text: str


@dataclass
class PdfContent:
    pages: list[PdfPage]
    page_count: int
    metadata: dict = field(default_factory=dict)


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")


def validate_pdf_header(data: bytes) -> None:
    if b"%PDF-" not in data[:1024]:
        raise PdfError("not_pdf", "The file is not a PDF (missing %PDF header).")


def _clean(text: str) -> str:
    text = _CTRL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words hyphenated across a line break
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return _MULTI_BLANK.sub("\n\n", "\n".join(lines)).strip()


def _meta(reader: PdfReader) -> dict:
    out: dict = {}
    try:
        meta = reader.metadata
        for key, attr in (("title", "title"), ("author", "author")):
            val = getattr(meta, attr, None) if meta else None
            if isinstance(val, str) and val.strip():
                out[key] = _CTRL.sub("", val).strip()[:200]
    except Exception:  # metadata is optional and often malformed
        pass
    return out


def extract_pdf(data: bytes, *, max_pages: int = 500, max_chars: int = 5_000_000) -> PdfContent:
    validate_pdf_header(data)
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")  # many PDFs are "encrypted" with an empty user password
            except Exception:
                ok = 0
            if not ok:
                raise PdfError("encrypted", "The PDF is password-protected.")
        n = len(reader.pages)
        if n > max_pages:
            raise PdfError("too_many_pages", f"The PDF has {n} pages; the limit is {max_pages}.")
        pages: list[PdfPage] = []
        total = 0
        for i, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if not text:
                continue
            total += len(text)
            if total > max_chars:
                raise PdfError("too_much_text", "The PDF contains too much text to index.")
            pages.append(PdfPage(i, text))
        meta = _meta(reader)
    except PdfError:
        raise
    except Exception as exc:  # pypdf can raise many types on hostile/corrupt input
        raise PdfError("corrupt", f"The PDF could not be read ({type(exc).__name__}).") from None
    if not pages:
        raise PdfError(
            "no_text",
            "The PDF has no extractable text (scanned PDFs need OCR, which is not supported).",
        )
    return PdfContent(pages=pages, page_count=n, metadata=meta)
