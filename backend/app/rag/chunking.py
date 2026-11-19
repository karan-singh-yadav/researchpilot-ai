"""Page-aware chunking: chunks never span pages, so every chunk has exactly one page number
(this is what makes page-level citations verifiable)."""

import re
from dataclasses import dataclass

from app.rag.pdf import PdfPage

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    index: int  # global, 0-based, in document order
    page: int
    text: str


def _tail(text: str, overlap: int) -> str:
    """Last ~overlap chars of text, starting at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    cut = text[-overlap:]
    sp = cut.find(" ")
    return cut[sp + 1 :] if sp != -1 else cut


def _hard_split(sentence: str, size: int) -> list[str]:
    out, cur = [], ""
    for word in sentence.split(" "):
        while len(word) > size:  # a single token longer than a chunk
            if cur:
                out.append(cur)
                cur = ""
            out.append(word[:size])
            word = word[size:]
        if cur and len(cur) + 1 + len(word) > size:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_pages(
    pages: list[PdfPage], chunk_size: int = 1000, overlap: int = 150, min_chunk: int = 60
) -> list[Chunk]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100")
    if not 0 <= overlap <= chunk_size // 2:
        raise ValueError("overlap must be between 0 and chunk_size/2")
    chunks: list[Chunk] = []
    for page in pages:
        pieces: list[str] = []
        for sent in (s.strip() for s in _SENT.split(page.text)):
            if not sent:
                continue
            pieces.extend(_hard_split(sent, chunk_size) if len(sent) > chunk_size else [sent])
        page_chunks: list[str] = []
        cur = ""
        for piece in pieces:
            if cur and len(cur) + 1 + len(piece) > chunk_size:
                page_chunks.append(cur)
                carry = _tail(cur, overlap)
                cur = (
                    f"{carry} {piece}".strip()
                    if carry and len(carry) + 1 + len(piece) <= chunk_size
                    else piece
                )
            else:
                cur = f"{cur} {piece}".strip()
        if cur:
            if page_chunks and len(cur) < min_chunk:
                page_chunks[-1] = f"{page_chunks[-1]} {cur}"[: chunk_size * 2]  # fold a tiny tail
            else:
                page_chunks.append(cur)
        for text in page_chunks:
            chunks.append(Chunk(len(chunks), page.number, text))
    return chunks
