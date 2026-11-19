"""Run the reference pipeline. Default = OFFLINE fixture data + offline templated LLM.

python scripts/demo_pipeline.py                      # offline demo, prints Markdown
python scripts/demo_pipeline.py --out report.md
python scripts/demo_pipeline.py --live               # uses .env (LLM_*, SEARCH_*) for real
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.services.llm import build_llm_provider  # noqa: E402
from app.services.pipeline import run_pipeline  # noqa: E402
from app.tools.factory import build_fetcher, build_search_provider  # noqa: E402

DEFAULT_Q = "Research the latest developments in agentic AI. Compare frameworks and limitations."


def main() -> int:
    # Windows consoles/pipes default to legacy code pages; reports contain non-ASCII symbols.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_Q)
    ap.add_argument("--depth", default="quick", choices=["quick", "standard", "deep"])
    ap.add_argument("--live", action="store_true", help="use real providers from .env")
    ap.add_argument("--mcp", action="store_true", help="route search/fetch through the MCP server")
    ap.add_argument("--out", help="write Markdown report to this file")
    a = ap.parse_args()
    if a.live:
        settings = Settings(_env_file=ROOT / "backend" / ".env")
    else:
        fixture = ROOT / "evaluation" / "datasets" / "fixtures" / "demo_web.json"
        settings = Settings(
            _env_file=None,
            llm_provider="fake",
            search_provider="fixture",
            search_fixture_path=str(fixture),
        )
    client = None
    if a.mcp:
        from app.mcp.client import (
            AgentToolbox,
            McpFetcher,
            McpSearchProvider,
            McpToolClient,
            stdio_params,
        )

        build_search_provider(settings)  # validate configuration first
        client = McpToolClient(stdio_params(settings), call_timeout=60)
        search = McpSearchProvider(AgentToolbox(client, "search"))
        fetcher = McpFetcher(AgentToolbox(client, "reader"), settings.reader_max_chars)
    else:
        search, fetcher = build_search_provider(settings), build_fetcher(settings)
    try:
        res = run_pipeline(
            a.query,
            llm=build_llm_provider(settings),
            search=search,
            fetcher=fetcher,
            depth=a.depth,
            results_per_query=7,
        )
    finally:
        if client is not None:
            client.close()
    if a.out:
        Path(a.out).write_text(res.report.markdown, encoding="utf-8")
        print(f"report written to {a.out}")
    else:
        print(res.report.markdown)
    print("--- run summary ---")
    print(
        f"sources: {len(res.sources)} | evidence: {len(res.evidence)} "
        f"| iterations: {res.iterations}"
    )
    print(f"tokens: {res.usage.total} | timings_ms: {res.timings_ms}")
    for w in res.warnings:
        print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
