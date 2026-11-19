# ResearchPilot AI

**A multi-agent research assistant that plans, searches, reads, writes, and verifies every citation.**

[![CI](https://github.com/karan-singh-yadav/researchpilot-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/researchpilot-ai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?logo=tailwindcss&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

A multi-agent research assistant. Give it a question and it plans the research,
searches the web, reads sources, analyzes the evidence, writes a report, and
verifies every citation before releasing it.

Ask a question and ResearchPilot drafts a research plan for you to approve, runs specialist agents in parallel, and returns a report in which every claim links back to a verbatim source passage. Citations, statistics, and quotes are checked against the collected evidence before a report is released.

---

## Features

- **Human-approved plans.** You review, edit, or reject the plan before any search runs.
- **Specialist agents.** Supervisor, Search, Reader, Analyst, Writer, and Critic, orchestrated as a LangGraph workflow with parallel workers, bounded revision loops, and checkpointed resume.
- **Verified citations.** Click any `[S1]` badge to see the exact source passage. Citation IDs, statistics, and quotations are checked deterministically, and an LLM Critic reviews claim support.
- **Document RAG.** Upload PDFs and get page-level citations such as `[S2, p.4]`, or mix documents and web sources in one report.
- **Live execution trace.** Watch each agent step stream in over SSE.
- **Provider-agnostic.** Works with any OpenAI-compatible LLM endpoint (tested with Groq), plus Anthropic, with Tavily or Brave for search.
- **Offline demo mode.** Runs with no API keys using synthetic fixtures.
- **Observability and evaluation.** Token and cost metrics plus a baseline-vs-improved benchmark runner.
- **MCP tools.** The agents can reach their tools through an MCP server.
- **Production basics.** API-key auth, rate limiting, an SSRF-hardened fetcher, PostgreSQL, Docker Compose, and CI.

## Architecture

```
React UI  ──►  FastAPI  ──►  LangGraph workflow
                               ├─ Supervisor   (plan + budgets, human approval)
                               ├─ Search ► Reader   (parallel, dependency waves)
                               ├─ Analyst      (evidence-grounded findings)
                               ├─ Writer       (cited report, checks every citation)
                               └─ Critic       (deterministic checks + LLM review)
                          ├─ SQLite / PostgreSQL   (sessions, evidence, reports)
                          └─ Documents + RAG       (PDF chunks, embeddings, reranking)
```

More detail: [docs/architecture.md](docs/architecture.md) · [docs/rag.md](docs/rag.md) · [docs/mcp.md](docs/mcp.md) · [docs/security.md](docs/security.md) · [docs/graph.md](docs/graph.md)

## Quickstart

**Requirements:** Python 3.11+, Node.js 18+. Docker is optional.

### 1. Backend (Windows PowerShell)

```powershell
.\scripts\setup.ps1      # creates .venv, installs deps, writes backend/.env (SQLite), runs migrations
cd backend
.\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

API docs are at `http://localhost:8000/docs`.

If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### 3. Docker Compose (PostgreSQL + backend + frontend)

```bash
docker compose up --build
```

The UI is at `http://localhost:3000` and the API docs at `http://localhost:8000/docs`.

## Configuration

Copy `.env.example` to `backend/.env` (the setup script does this for you). Never commit `.env`.

### Offline demo (no keys)

```dotenv
LLM_PROVIDER=fake
SEARCH_PROVIDER=fixture
SEARCH_FIXTURE_PATH=../evaluation/datasets/fixtures/demo_web.json
```

The demo uses synthetic data and a templated writer, so it shows the workflow but not real answers.

### Real research (example: Groq + Tavily)

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=openai/gpt-oss-120b
LLM_API_KEY=your-groq-key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=5

SEARCH_PROVIDER=tavily
SEARCH_API_KEY=your-tavily-key
SEARCH_FIXTURE_PATH=
```

Any other OpenAI-compatible endpoint works the same way: set `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`. For Anthropic, use `LLM_PROVIDER=anthropic`.

Use the `quick` depth first, and run one research job at a time. Free API tiers rate-limit quickly.

## Try it

1. Open the app and start a new run, for example: *What is retrieval-augmented generation and when should it be used?*
2. Review the plan and click **Approve**.
3. Watch the trace, then read the report and click citation badges to inspect the evidence.
4. On the **Documents** page, upload a PDF from `evaluation/datasets/pdfs/` and ask questions about it.

## Testing

```powershell
cd backend
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\ruff check --config pyproject.toml . ..\scripts
```

```powershell
cd frontend
npm run build
```

CI runs the same checks on every push (`.github/workflows/ci.yml`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `429 Too Many Requests` | Provider rate limit. Wait a minute, use `quick` depth, run one job at a time. |
| `404 model_not_found` | The model is not available on your account. List your models at `<LLM_BASE_URL>/models`. |
| `400 json_validate_failed` | The provider rejected JSON mode. The adapter retries once without it. |
| Frontend shows network errors | The backend is not running on port 8000. |
| CORS error | Add your frontend URL to `CORS_ORIGINS` in `backend/.env`. |
| PostCSS or Tailwind errors after `npm audit fix --force` | Do not run it. Tailwind must stay on v3 (`^3.4.16`). |

## Project layout

```
backend/     FastAPI app, agents, LangGraph workflow, RAG, evaluation, tests
frontend/    React + TypeScript + Vite + Tailwind UI
evaluation/  Offline fixtures and demo PDFs
scripts/     Setup, check, and demo pipeline scripts
docs/        Architecture, RAG, MCP, security, workflow graph
```

## Status

All 11 build phases are complete: foundation, LLM and supervisor, search/reader/analyst/writer, LangGraph workflow and critic, MCP, document RAG, citation UX, evaluation, frontend, Docker/CI, and final audit.

## Notes

- Embeddings default to a lexical `hash` provider, which matches words rather than meaning. Switch `EMBEDDING_PROVIDER` for semantic retrieval.
- The Critic's claim review is performed by an LLM and can be wrong. A passing report is not a guarantee of factual correctness.
