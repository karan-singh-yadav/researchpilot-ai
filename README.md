# ResearchPilot AI — Multi-Agent Research & Intelligence System

A multi-agent research assistant. Give it a question and it plans the research,
searches the web, reads sources, analyzes the evidence, writes a report, and
verifies every citation before releasing it.

**Features**
- Human-approved research plan before any search runs
- Specialist agents: Planner, Search, Reader, Analyst, Writer, Critic
- Citation inspector: click any [S1] badge to see the verbatim source passage
- PDF upload with page-level citations (document RAG)
- Live execution trace, token/cost metrics, and an evaluation benchmark
- Works with any OpenAI-compatible LLM (tested with Groq) and Tavily/Brave search
- Offline demo mode with no API keys required

**Stack:** FastAPI, LangGraph, SQLAlchemy, React, Vite, Tailwind CSS

**Status: Phases 1–11 of 11 complete (Full-Stack Release).**

## Progress
- [x] **1 Foundation** — config, schema, Alembic, FastAPI, error envelope, API-key auth, rate limit
- [x] **2 LLM + Supervisor** — provider-agnostic LLM layer, structured output with bounded repair,
      Supervisor with server-enforced budgets, plan edit/approve API
- [x] **3 Search / Reader / Analyst / Writer** — Brave/Tavily/offline search, SSRF-hardened fetcher,
      verbatim-verified evidence, calibrated analysis, citation/statistic/quote-checked writer
- [x] **4 LangGraph workflow + Critic** — real graph (parallel research workers in dependency waves,
      conditional routing, bounded follow-up research and revision loops), human plan-approval pause,
      checkpointing + resume (survives restart), cancel, time budget, Critic agent (deterministic
      checks + LLM claim review), background runner, SSE live progress, report/sources/trace APIs
- [x] **5 MCP server + client** — official-SDK MCP server (5 typed tools, strict args, timeouts, rate/
      concurrency limits, allow-list), sync client bridge with crash-restart, per-agent permissions,
      `TOOL_TRANSPORT=mcp` routes the agents through it (proved equivalent to the direct path), HTTP export
- [x] **6 Documents + RAG** — validated PDF upload, page-aware chunking, pluggable embeddings, SQL (default) /
      Chroma (optional) index, lexical reranking, Document Research agent, page-level citations (`[S2, p.4]`) checked
      by Writer + Critic, document & hybrid research, `/api/documents/*` incl. search + ask, MCP document tools
- [x] **7 Citation UX / HITL polish** — structured inline citation span parser & resolver (`/citations`, `/reports`),
      partial plan editing (`PATCH /plan`), feedback-guided plan rejection/replanning (`POST /plan/reject`),
      one-click approve + resume (`POST /plan/approve?auto_resume=true`)
- [x] **8 Evaluation / observability** — system telemetry aggregation (`/api/metrics`), automated offline/live
      benchmark runner (`baseline` vs `improved` in `app/evaluation/runner.py` + `scripts/run_eval.py`),
      comparison API (`/api/evaluation/compare`)
- [x] **9 Frontend** — React 18 + TypeScript + Vite + Tailwind CSS workspace with live SSE trace streaming,
      interactive HITL plan editor/approver, clickable inline citation & evidence inspector, PDF RAG manager,
      and evaluation/metrics dashboard
- [x] **10 Postgres / Docker / CI** — multi-service `docker-compose.yml` (PostgreSQL 16 + FastAPI backend +
      Nginx-served React frontend) and GitHub Actions CI (`ci.yml`)
- [x] **11 Final audit** — end-to-end test suite across all 11 phases, AST/lint verification, and documentation

Workflow diagram (generated from the compiled graph): [docs/graph.md](docs/graph.md) · `GET /api/graph`.

## Quickstart (Local Development)

### 1. Backend (Windows PowerShell)
```powershell
.\scripts\setup.ps1      # creates .venv, installs deps, configures .env (SQLite), runs migrations
.\scripts\check.ps1      # runs ruff + pytest + offline demo pipeline
```
Start the API server:
```powershell
cd backend
.\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

### 2. Frontend (React + Vite)
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`.

### 3. Full-Stack Docker Compose (PostgreSQL + Backend + Frontend)
```bash
docker compose up --build
```
Open `http://localhost:3000` (Frontend) and `http://localhost:8000/docs` (FastAPI Swagger UI).
