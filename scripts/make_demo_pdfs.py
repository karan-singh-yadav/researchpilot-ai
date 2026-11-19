"""Generate SYNTHETIC demo PDFs (fictional content) into evaluation/datasets/pdfs/.

python scripts/make_demo_pdfs.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from tests.pdf_factory import make_pdf, wrap  # noqa: E402

DOCS = {
    "orchestra_x_whitepaper.pdf": (
        "Orchestra-X Whitepaper (synthetic)",
        [
            "Orchestra-X Whitepaper. Orchestra-X is a fictional orchestration framework used for offline "
            "tests. It organizes agents as nodes in a directed graph with typed shared state. "
            "Checkpointing after every step allows failed runs to resume without repeating completed work.",
            "Evaluation results. In the fictional 2026 Orchestra Benchmark, the supervised graph variant "
            "completed 87% of multi-step tasks while the single-agent baseline completed 61%. Latency "
            "increased by 2.4 seconds on average because of additional coordination steps.",
            "Limitations and risks. Graph based orchestration adds operational complexity and requires "
            "careful budget limits to prevent runaway loops. Human approval steps reduce risk but slow "
            "down delivery.",
        ],
    ),
    "agentic_risk_report.pdf": (
        "Agentic Risk Report (synthetic)",
        [
            "Agentic risk report. This fictional report reviews operational risks of autonomous research "
            "agents, including compounding errors, prompt injection through retrieved documents and "
            "uncontrolled cost growth.",
            "Mitigations. The report recommends strict tool permissions, bounded retries, schema "
            "validation of every model output and mandatory citation checks before a report is released. "
            "Teams that adopted these controls reported 38% fewer escalations in the fictional study.",
        ],
    ),
}

out = ROOT / "evaluation" / "datasets" / "pdfs"
out.mkdir(parents=True, exist_ok=True)
for name, (title, pages) in DOCS.items():
    (out / name).write_bytes(make_pdf([wrap(p) for p in pages], title=title))
    print("wrote", (out / name).relative_to(ROOT))
