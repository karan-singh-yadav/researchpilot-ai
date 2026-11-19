"""Talk to the ResearchPilot MCP server from the command line (spawns it over stdio).

  python scripts/mcp_call.py list
  python scripts/mcp_call.py call search_web query="agentic ai" max_results=2
  python scripts/mcp_call.py call fetch_webpage url=https://example.test/agentic-overview
  python scripts/mcp_call.py call export_research_report research_id=<uuid> format=json

Default = OFFLINE fixtures (synthetic data). Add --live to use backend/.env (real providers).
Arguments are key=value; values are parsed as JSON when possible (numbers, lists), else strings.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.mcp.client import McpToolClient, McpToolError, stdio_params  # noqa: E402


def parse_kv(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"argument '{item}' must look like key=value")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("action", choices=["list", "call"])
    ap.add_argument("tool", nargs="?")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--live", action="store_true", help="use real providers from backend/.env")
    a = ap.parse_args()
    if a.live:
        settings = Settings(_env_file=ROOT / "backend" / ".env")
    else:
        fixture = ROOT / "evaluation" / "datasets" / "fixtures" / "demo_web.json"
        settings = Settings(
            _env_file=None, search_provider="fixture", search_fixture_path=str(fixture)
        )
    with McpToolClient(stdio_params(settings), call_timeout=60) as client:
        if a.action == "list":
            for t in client.list_tools():
                props = ", ".join((t.input_schema or {}).get("properties", {}))
                print(f"{t.name}({props})\n    {t.description}")
            return 0
        if not a.tool:
            raise SystemExit("usage: mcp_call.py call <tool> key=value ...")
        try:
            result = client.call(a.tool, parse_kv(a.args))
        except McpToolError as exc:
            print(f"error [{exc.code}{', retryable' if exc.retryable else ''}]: {exc.message}")
            return 1
        print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
