"""Regenerate docs/graph.md from the real compiled graph:  python scripts/export_graph.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.graph.builder import render_graph_doc  # noqa: E402

(ROOT / "docs" / "graph.md").write_text(render_graph_doc(), encoding="utf-8", newline="\n")
print("wrote docs/graph.md")
