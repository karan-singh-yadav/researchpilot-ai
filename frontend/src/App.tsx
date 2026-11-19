import React, { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Bot,
  FileText,
  KeyRound,
  Layers,
  Play,
  PlusCircle,
  RotateCcw,
  Square,
  Trash2,
} from "lucide-react";
import { PlanApprovalCard } from "./components/PlanApprovalCard";
import { ReportViewer } from "./components/ReportViewer";
import { TracePanel } from "./components/TracePanel";
import { DocumentsPage } from "./pages/DocumentsPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { api, getApiKey, setApiKey, subscribeEvents } from "./services/api";
import type {
  AgentEvent,
  PlanOut,
  ReportCitations,
  ReportOut,
  ResearchDepth,
  ResearchMode,
  ResearchSession,
  SourceItem,
  UploadedDocument,
} from "./types/api";

type Tab = "workspace" | "documents" | "evaluation" | "topology";

export default function App() {
  const [tab, setTab] = useState<Tab>("workspace");
  const [sessions, setSessions] = useState<ResearchSession[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [activeSession, setActiveSession] = useState<ResearchSession | null>(null);
  const [planOut, setPlanOut] = useState<PlanOut | null>(null);
  const [report, setReport] = useState<ReportOut | null>(null);
  const [citations, setCitations] = useState<ReportCitations | null>(null);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [streaming, setStreaming] = useState(false);

  // New Research Form State
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<ResearchMode>("web");
  const [depth, setDepth] = useState<ResearchDepth>("standard");
  const [requireApproval, setRequireApproval] = useState(false);
  const [availableDocs, setAvailableDocs] = useState<UploadedDocument[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [creating, setCreating] = useState(false);

  // Settings / Graph
  const [apiKeyInput, setApiKeyInput] = useState(getApiKey());
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [mermaidCode, setMermaidCode] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refreshSessions = async () => {
    try {
      const res = await api.listResearch();
      setSessions(res.items);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const refreshDocs = async () => {
    try {
      const res = await api.listDocuments();
      setAvailableDocs(res.items.filter((d) => d.status === "ready"));
    } catch {
      // ignore if disabled
    }
  };

  useEffect(() => {
    refreshSessions();
    refreshDocs();
  }, []);

  const loadSessionDetails = async (id: string) => {
    setError(null);
    try {
      const sess = await api.getResearch(id);
      setActiveSession(sess);

      try {
        setPlanOut(await api.getPlan(id));
      } catch {
        setPlanOut(null);
      }

      try {
        const rep = await api.getReport(id);
        setReport(rep);
        const [cits, srcs] = await Promise.all([api.getCitations(id), api.getSources(id)]);
        setCitations(cits);
        setSources(srcs);
      } catch {
        setReport(null);
        setCitations(null);
        try {
          setSources(await api.getSources(id));
        } catch {
          setSources([]);
        }
      }

      const trace = await api.getTrace(id);
      setEvents(trace.items);
    } catch (e: any) {
      setError(e.message);
    }
  };

  useEffect(() => {
    if (!activeId) return;
    loadSessionDetails(activeId);

    setStreaming(true);
    const unsub = subscribeEvents(
      activeId,
      (ev) => {
        setEvents((prev) => (prev.some((x) => x.id === ev.id) ? prev : [...prev, ev]));
      },
      () => {
        setStreaming(false);
        loadSessionDetails(activeId);
        refreshSessions();
      }
    );
    return () => {
      unsub();
      setStreaming(false);
    };
  }, [activeId]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim().length < 10) {
      setError("Please enter a research question of at least 10 characters.");
      return;
    }
    setCreating(true);
    setError(null);
    try {
      const created = await api.createResearch({
        query: query.trim(),
        mode,
        depth,
        require_plan_approval: requireApproval,
        document_ids: mode === "web" ? [] : selectedDocIds,
        start: true,
      });
      setQuery("");
      await refreshSessions();
      setActiveId(created.id);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setCreating(false);
    }
  };

  const handleCancel = async () => {
    if (!activeId) return;
    await api.cancelResearch(activeId);
    loadSessionDetails(activeId);
  };

  const handleResume = async () => {
    if (!activeId) return;
    await api.resumeResearch(activeId);
    setActiveId(null);
    setTimeout(() => setActiveId(activeSession?.id || null), 50);
  };

  const handleDelete = async (id: string) => {
    await api.deleteResearch(id);
    if (activeId === id) {
      setActiveId(null);
      setActiveSession(null);
      setReport(null);
    }
    refreshSessions();
  };

  const openTopology = async () => {
    setTab("topology");
    try {
      const g = await api.graph();
      setMermaidCode(g.mermaid);
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Top Navbar */}
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-white shadow">
              <Bot className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white">ResearchPilot AI</h1>
              <p className="text-[11px] text-slate-400">Multi-Agent Research &amp; Intelligence System</p>
            </div>
          </div>

          <nav className="flex items-center gap-1.5">
            <button
              onClick={() => setTab("workspace")}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                tab === "workspace" ? "bg-indigo-600 text-white" : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              <Layers className="h-3.5 w-3.5" /> Research Workspace
            </button>
            <button
              onClick={() => {
                setTab("documents");
                refreshDocs();
              }}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                tab === "documents" ? "bg-indigo-600 text-white" : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              <FileText className="h-3.5 w-3.5" /> Documents (RAG)
            </button>
            <button
              onClick={() => setTab("evaluation")}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                tab === "evaluation" ? "bg-indigo-600 text-white" : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              <BarChart3 className="h-3.5 w-3.5" /> Evaluation &amp; Metrics
            </button>
            <button
              onClick={openTopology}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                tab === "topology" ? "bg-indigo-600 text-white" : "text-slate-300 hover:bg-slate-800"
              }`}
            >
              <Activity className="h-3.5 w-3.5" /> LangGraph Topology
            </button>
            <button
              onClick={() => setShowKeyModal((v) => !v)}
              className="ml-2 inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
              title="Configure X-API-Key"
            >
              <KeyRound className="h-3.5 w-3.5" /> API Key
            </button>
          </nav>
        </div>

        {showKeyModal && (
          <div className="border-t border-slate-800 bg-slate-900 px-4 py-2.5">
            <div className="mx-auto flex max-w-7xl items-center justify-end gap-2">
              <span className="text-xs text-slate-400">X-API-Key (if enabled on server):</span>
              <input
                type="password"
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                placeholder="Leave blank in dev mode"
                className="w-64 rounded border border-slate-700 bg-slate-950 px-2.5 py-1 text-xs text-slate-100"
              />
              <button
                onClick={() => {
                  setApiKey(apiKeyInput);
                  setShowKeyModal(false);
                  refreshSessions();
                }}
                className="rounded bg-indigo-600 px-3 py-1 text-xs font-semibold text-white"
              >
                Save
              </button>
            </div>
          </div>
        )}
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {error && (
          <div className="mb-4 flex items-center justify-between rounded-lg border border-rose-500/40 bg-rose-950/50 px-4 py-2.5 text-xs text-rose-200">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-rose-300 hover:text-white">
              Dismiss
            </button>
          </div>
        )}

        {tab === "documents" && <DocumentsPage />}
        {tab === "evaluation" && <EvaluationPage />}
        {tab === "topology" && (
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
            <h2 className="text-lg font-bold text-white">Compiled LangGraph Workflow Topology</h2>
            <p className="mt-1 text-xs text-slate-400">
              Live Mermaid specification returned directly by <code className="text-indigo-300">GET /api/graph</code>.
            </p>
            <pre className="mt-4 overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 p-4 font-mono text-xs text-emerald-300">
              {mermaidCode || "Loading graph..."}
            </pre>
          </div>
        )}

        {tab === "workspace" && (
          <div className="grid gap-6 lg:grid-cols-12">
            {/* Left Sidebar: Session History */}
            <aside className="lg:col-span-3 space-y-3">
              <button
                onClick={() => {
                  setActiveId(null);
                  setActiveSession(null);
                  setPlanOut(null);
                  setReport(null);
                  setEvents([]);
                }}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow hover:bg-indigo-500"
              >
                <PlusCircle className="h-4 w-4" /> New Research Run
              </button>

              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                <div className="mb-2 px-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Recent Sessions ({sessions.length})
                </div>
                <div className="max-h-[600px] space-y-1.5 overflow-y-auto">
                  {sessions.map((s) => (
                    <div
                      key={s.id}
                      onClick={() => setActiveId(s.id)}
                      className={`group flex cursor-pointer items-start justify-between gap-2 rounded-lg p-2.5 text-xs transition ${
                        activeId === s.id
                          ? "bg-indigo-600/20 text-white ring-1 ring-indigo-500/50"
                          : "hover:bg-slate-800/70 text-slate-300"
                      }`}
                    >
                      <div className="min-w-0 flex-1">
                        <div className="truncate font-medium">{s.query}</div>
                        <div className="mt-1 flex items-center gap-1.5 text-[10px] text-slate-400">
                          <span className="uppercase">{s.mode}</span> · <span>{s.depth}</span> ·{" "}
                          <span
                            className={
                              s.status === "completed"
                                ? "text-emerald-400"
                                : s.status === "failed"
                                ? "text-rose-400"
                                : s.status === "awaiting_approval"
                                ? "text-amber-400"
                                : "text-sky-400"
                            }
                          >
                            {s.status}
                          </span>
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(s.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </aside>

            {/* Right Main Area */}
            <section className="lg:col-span-9 space-y-6">
              {!activeId ? (
                /* New Research Composer */
                <form
                  onSubmit={handleCreate}
                  className="rounded-xl border border-slate-800 bg-slate-900/70 p-6 shadow-lg space-y-5"
                >
                  <div>
                    <h2 className="text-lg font-bold text-white">Start a Multi-Agent Research Run</h2>
                    <p className="mt-1 text-xs text-slate-400">
                      Orchestrates Supervisor planning, parallel web/document retrieval, verbatim evidence verification,
                      analytical synthesis, and Critic claim checking.
                    </p>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-slate-300">Research Question</label>
                    <textarea
                      rows={3}
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="e.g., Compare LangGraph, AutoGen, and CrewAI architectures for production multi-agent orchestration..."
                      className="mt-1.5 w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm text-white focus:border-indigo-500 focus:outline-none"
                    />
                  </div>

                  <div className="grid gap-4 sm:grid-cols-3">
                    <div>
                      <label className="block text-xs font-medium text-slate-300">Research Mode</label>
                      <select
                        value={mode}
                        onChange={(e) => {
                          const m = e.target.value as ResearchMode;
                          setMode(m);
                          if (m !== "web") refreshDocs();
                        }}
                        className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-white"
                      >
                        <option value="web">Web Research</option>
                        <option value="document">Uploaded Documents Only (PDF RAG)</option>
                        <option value="hybrid">Hybrid (Web + Documents)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300">Depth &amp; Budget</label>
                      <select
                        value={depth}
                        onChange={(e) => setDepth(e.target.value as ResearchDepth)}
                        className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-white"
                      >
                        <option value="quick">Quick (3 subtasks · 5 sources)</option>
                        <option value="standard">Standard (6 subtasks · 12 sources)</option>
                        <option value="deep">Deep (10 subtasks · 25 sources)</option>
                      </select>
                    </div>

                    <div className="flex items-end">
                      <label className="flex w-full cursor-pointer items-center gap-2.5 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200">
                        <input
                          type="checkbox"
                          checked={requireApproval}
                          onChange={(e) => setRequireApproval(e.target.checked)}
                          className="rounded border-slate-600 bg-slate-900 text-indigo-600"
                        />
                        Human-in-the-Loop Plan Gate
                      </label>
                    </div>
                  </div>

                  {mode !== "web" && (
                    <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
                      <div className="text-xs font-semibold text-slate-300">
                        Select Ready Documents ({selectedDocIds.length} selected)
                      </div>
                      {availableDocs.length === 0 ? (
                        <p className="mt-1 text-xs text-amber-400">
                          No ready PDFs found. Switch to the Documents (RAG) tab to upload PDFs first.
                        </p>
                      ) : (
                        <div className="mt-2 grid gap-2 sm:grid-cols-2">
                          {availableDocs.map((d) => (
                            <label
                              key={d.id}
                              className="flex cursor-pointer items-center gap-2 rounded border border-slate-800 bg-slate-900 px-3 py-2 text-xs"
                            >
                              <input
                                type="checkbox"
                                checked={selectedDocIds.includes(d.id)}
                                onChange={() =>
                                  setSelectedDocIds((prev) =>
                                    prev.includes(d.id) ? prev.filter((x) => x !== d.id) : [...prev, d.id]
                                  )
                                }
                              />
                              <span className="truncate">{d.filename}</span>
                            </label>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="flex justify-end">
                    <button
                      type="submit"
                      disabled={creating}
                      className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-xs font-semibold text-white shadow hover:bg-indigo-500 disabled:opacity-50"
                    >
                      <Play className="h-4 w-4" />
                      {creating ? "Launching Agents..." : "Launch Research"}
                    </button>
                  </div>
                </form>
              ) : (
                /* Active Research Session View */
                activeSession && (
                  <div className="space-y-6">
                    {/* Session Status Header */}
                    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 shadow">
                      <div className="flex flex-wrap items-start justify-between gap-4">
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded-full bg-indigo-500/20 px-2.5 py-0.5 text-xs font-semibold uppercase text-indigo-300">
                              {activeSession.status}
                            </span>
                            <span className="text-xs text-slate-400">
                              {activeSession.mode} · {activeSession.depth} · {activeSession.token_usage.toLocaleString()}{" "}
                              tokens · {activeSession.revision_count} revision(s)
                            </span>
                          </div>
                          <h2 className="mt-2 text-base font-bold text-white">{activeSession.query}</h2>
                          {activeSession.error && (
                            <p className="mt-1.5 text-xs text-rose-400">{activeSession.error}</p>
                          )}
                        </div>

                        <div className="flex items-center gap-2">
                          {activeSession.status === "running" && (
                            <button
                              onClick={handleCancel}
                              className="inline-flex items-center gap-1.5 rounded-lg border border-rose-500/40 bg-rose-950/40 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-900/40"
                            >
                              <Square className="h-3.5 w-3.5" /> Cancel
                            </button>
                          )}
                          {["failed", "cancelled", "interrupted", "planned"].includes(activeSession.status) && (
                            <button
                              onClick={handleResume}
                              className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-500"
                            >
                              <RotateCcw className="h-3.5 w-3.5" /> Resume Run
                            </button>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* HITL Plan Review & Approval Card */}
                    {planOut && (
                      <PlanApprovalCard
                        researchId={activeSession.id}
                        planOut={planOut}
                        onUpdated={(p) => setPlanOut(p)}
                        onResumed={() => loadSessionDetails(activeSession.id)}
                      />
                    )}

                    {/* Final Report + Interactive Citation Inspector */}
                    {report && (
                      <ReportViewer
                        researchId={activeSession.id}
                        report={report}
                        citations={citations}
                        sources={sources}
                      />
                    )}

                    {/* Live SSE Execution Trace */}
                    <TracePanel events={events} live={streaming} />
                  </div>
                )
              )}
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
