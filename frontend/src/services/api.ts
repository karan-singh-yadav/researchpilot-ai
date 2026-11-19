import type {
  AgentEvent,
  DocumentAskResult,
  EvaluationRun,
  Page,
  PlanOut,
  ReportCitations,
  ReportOut,
  ResearchDepth,
  ResearchMode,
  ResearchSession,
  SourceItem,
  SubTask,
  SystemMetrics,
  UploadedDocument,
} from "../types/api";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export function getApiKey(): string {
  return localStorage.getItem("rp_api_key") || "";
}

export function setApiKey(key: string): void {
  if (key.trim()) {
    localStorage.setItem("rp_api_key", key.trim());
  } else {
    localStorage.removeItem("rp_api_key");
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers || {});
  const key = getApiKey();
  if (key) headers.set("X-API-Key", key);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const msg = data?.error?.message || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data as T;
}

export const api = {
  health: () => request<{ status: string; database: string; schema: string }>("/health"),
  graph: () => request<{ mermaid: string }>("/api/graph"),

  // Research sessions
  listResearch: (q?: string) =>
    request<Page<ResearchSession>>(`/api/research?limit=50${q ? `&q=${encodeURIComponent(q)}` : ""}`),
  getResearch: (id: string) => request<ResearchSession>(`/api/research/${id}`),
  createResearch: (payload: {
    query: string;
    mode: ResearchMode;
    depth: ResearchDepth;
    require_plan_approval: boolean;
    document_ids: string[];
    start: boolean;
  }) => request<ResearchSession>("/api/research", { method: "POST", body: JSON.stringify(payload) }),
  deleteResearch: (id: string) => request<void>(`/api/research/${id}`, { method: "DELETE" }),
  resumeResearch: (id: string) =>
    request<{ id: string; status: string }>(`/api/research/${id}/resume`, { method: "POST" }),
  cancelResearch: (id: string) =>
    request<{ id: string; status: string }>(`/api/research/${id}/cancel`, { method: "POST" }),

  // Plan & HITL
  getPlan: (id: string) => request<PlanOut>(`/api/research/${id}/plan`),
  generatePlan: (id: string) => request<PlanOut>(`/api/research/${id}/plan`, { method: "POST" }),
  patchPlan: (
    id: string,
    patch: { objectives?: string[]; subtasks?: SubTask[]; report_outline?: string[] }
  ) => request<PlanOut>(`/api/research/${id}/plan`, { method: "PATCH", body: JSON.stringify(patch) }),
  rejectPlan: (id: string, feedback: string) =>
    request<PlanOut>(`/api/research/${id}/plan/reject`, {
      method: "POST",
      body: JSON.stringify({ feedback }),
    }),
  approvePlan: (id: string, note?: string, autoResume = true) =>
    request<PlanOut>(`/api/research/${id}/plan/approve?auto_resume=${autoResume}`, {
      method: "POST",
      body: JSON.stringify({ note: note || null }),
    }),

  // Results, Citations, Trace
  getReport: (id: string) => request<ReportOut>(`/api/research/${id}/report`),
  getCitations: (id: string) => request<ReportCitations>(`/api/research/${id}/citations`),
  getSources: (id: string) => request<SourceItem[]>(`/api/research/${id}/sources`),
  getTrace: (id: string) => request<Page<AgentEvent>>(`/api/research/${id}/trace?limit=500`),
  exportReport: async (id: string, format: "markdown" | "json"): Promise<string> => {
    const headers = new Headers();
    const key = getApiKey();
    if (key) headers.set("X-API-Key", key);
    const res = await fetch(`${API_BASE}/api/research/${id}/export?format=${format}`, { headers });
    if (!res.ok) throw new Error(`Export failed (${res.status})`);
    return res.text();
  },

  // Documents / RAG
  listDocuments: () => request<Page<UploadedDocument>>("/api/documents?limit=100"),
  uploadDocument: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<{ document: UploadedDocument; duplicate: boolean }>("/api/documents", {
      method: "POST",
      body: fd,
    });
  },
  deleteDocument: (id: string) => request<void>(`/api/documents/${id}`, { method: "DELETE" }),
  askDocuments: (question: string, document_ids?: string[]) =>
    request<DocumentAskResult>("/api/documents/ask", {
      method: "POST",
      body: JSON.stringify({ question, document_ids: document_ids?.length ? document_ids : null }),
    }),

  // Observability & Evaluation
  getMetrics: () => request<SystemMetrics>("/api/metrics"),
  listEvaluations: () => request<EvaluationRun[]>("/api/evaluation/runs?limit=20"),
  runEvaluation: (variant: "baseline" | "improved") =>
    request<EvaluationRun>("/api/evaluation/runs", {
      method: "POST",
      body: JSON.stringify({ variant, dataset: "default-offline" }),
    }),
  compareEvaluations: () =>
    request<{
      baseline_id: string;
      improved_id: string;
      dataset: string;
      deltas: Record<string, { baseline: number; improved: number; delta: number }>;
    }>("/api/evaluation/compare"),
};

/**
 * Subscribe to SSE `/api/research/{id}/events` using fetch() so `X-API-Key` header is sent.
 */
export function subscribeEvents(
  researchId: string,
  onEvent: (ev: AgentEvent) => void,
  onDone?: () => void
): () => void {
  const ctrl = new AbortController();
  (async () => {
    try {
      const headers = new Headers();
      const key = getApiKey();
      if (key) headers.set("X-API-Key", key);
      const res = await fetch(`${API_BASE}/api/research/${researchId}/events`, {
        headers,
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        onDone?.();
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const blocks = buf.split("\n\n");
        buf = blocks.pop() || "";
        for (const block of blocks) {
          const dataLines = block
            .split("\n")
            .filter((ln) => ln.startsWith("data: "))
            .map((ln) => ln.slice(6));
          if (!dataLines.length) continue;
          try {
            const parsed = JSON.parse(dataLines.join("\n"));
            if (parsed.id) onEvent(parsed as AgentEvent);
            if (parsed.event_type === "run_end") {
              onDone?.();
              return;
            }
          } catch {
            // ignore keepalive / partial frames
          }
        }
      }
    } catch {
      // aborted or network closed
    } finally {
      onDone?.();
    }
  })();
  return () => ctrl.abort();
}
