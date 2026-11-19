"""Builds small but valid text PDFs (one string per page) without any third-party writer."""

import io


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str], title: str | None = None) -> bytes:
    objs: list[bytes] = []

    def add(body: str | bytes) -> int:
        objs.append(body if isinstance(body, bytes) else body.encode("latin-1"))
        return len(objs)

    add("<< /Type /Catalog /Pages 2 0 R >>")  # 1
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    add(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")  # 2
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")  # 3
    for i, text in enumerate(pages):
        lines = [_esc(ln) for ln in text.split("\n")] or [""]
        stream = "BT /F1 11 Tf 14 TL 50 740 Td " + " T* ".join(f"({ln}) Tj" for ln in lines) + " ET"
        page_no = add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_no_content(i)} 0 R >>"
        )
        assert page_no == 4 + 2 * i
        add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    if title:
        info = add(f"<< /Title ({_esc(title)}) >>")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{n} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    trailer = (
        f"<< /Size {len(objs) + 1} /Root 1 0 R" + (f" /Info {info} 0 R" if title else "") + " >>"
    )
    out.write(f"trailer\n{trailer}\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def page_no_content(i: int) -> int:
    return 5 + 2 * i


def wrap(text: str, width: int = 95) -> str:
    """Hard-wrap a paragraph into lines so it fits on the page."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


ORCHESTRA = [
    wrap(
        "Orchestra-X Whitepaper. Orchestra-X is a fictional orchestration framework used for offline tests. "
        "It organizes agents as nodes in a directed graph with typed shared state. "
        "Checkpointing after every step allows failed runs to resume without repeating completed work."
    ),
    wrap(
        "Evaluation results. In the fictional 2026 Orchestra Benchmark, the supervised graph variant "
        "completed 87% of multi-step tasks while the single-agent baseline completed 61%. "
        "Latency increased by 2.4 seconds on average because of additional coordination steps."
    ),
    wrap(
        "Limitations and risks. Graph based orchestration adds operational complexity and requires careful "
        "budget limits to prevent runaway loops. Human approval steps reduce risk but slow down delivery."
    ),
]
