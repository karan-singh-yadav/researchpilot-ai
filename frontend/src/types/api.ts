export type ResearchMode = "web" | "document" | "hybrid";
export type ResearchDepth = "quick" | "standard" | "deep";
export type ResearchStatus =
  | "pending"
  | "planning"
  | "planned"
  | "awaiting_approval"
  | "running"
  | "completed"
  | "failed"
  | "interrupted"
  | "cancelled";

export interface ExecutionBudget {
  max_subtasks: number;
  max_sources: number;
  max_iterations: number;
  max_revisions: number;
  timeout_seconds: number;
}

export interface SubTask {
  id: string;
  description: string;
  agent: "search" | "document";
  search_queries: string[];
  depends_on: string[];
}

export interface ResearchPlan {
  objectives: string[];
  subtasks: SubTask[];
  report_outline: string[];
  budget: ExecutionBudget;
  mode: string;
  depth: string;
  generated_by: string;
}

export interface PlanOut {
  research_id: string;
  status: ResearchStatus;
  requires_approval: boolean;
  plan_approved: boolean;
  plan: ResearchPlan | null;
}

export interface ResearchSession {
  id: string;
  query: string;
  mode: ResearchMode;
  depth: ResearchDepth;
  status: ResearchStatus;
  revision_count: number;
  token_usage: number;
  estimated_cost_usd: number;
  require_plan_approval: boolean;
  plan_approved: boolean;
  document_ids: string[];
  error: string | null;
  warnings: string[];
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface AgentEvent {
  id: string;
  research_id?: string;
  agent: string;
  event_type: string;
  summary: string;
  payload: Record<string, unknown> | null;
  duration_ms: number | null;
  created_at: string;
}

export interface EvidenceItem {
  evidence_ref: string;
  claim: string;
  kind: string;
  passage: string;
  page_number: number | null;
  relevance_score: number | null;
}

export interface SourceItem {
  source_ref: string;
  kind: "web" | "document";
  title: string;
  url: string | null;
  document_id: string | null;
  published_at: string | null;
  retrieved_at: string;
  provider: string;
  status: string;
  error: string | null;
  injection_suspected: boolean;
  evidence: EvidenceItem[];
}

export interface ValidationIssue {
  severity: "error" | "warning";
  code: string;
  message: string;
  section: string | null;
  method: "deterministic" | "llm";
}

export interface ValidationReport {
  passed: boolean;
  issues: ValidationIssue[];
  claims_checked: number;
  llm_judged: boolean;
  disclaimer: string;
}

export interface ReportOut {
  id: string;
  session_id: string;
  version: number;
  markdown: string;
  validation: ValidationReport | null;
  created_at: string;
}

export interface CitationOccurrence {
  id: string;
  raw: string;
  start: number;
  end: number;
  section: string;
  source_ref: string;
  page: number | null;
  resolved: boolean;
  source_title: string | null;
  source_kind: string | null;
  source_url: string | null;
  document_id: string | null;
  injection_suspected: boolean;
  evidence: EvidenceItem[];
}

export interface CitedSourceSummary {
  source_ref: string;
  title: string;
  kind: string;
  url: string | null;
  document_id: string | null;
  cited_pages: number[];
  citation_count: number;
  evidence_count: number;
  injection_suspected: boolean;
}

export interface ReportCitations {
  research_id: string;
  report_version: number;
  total_citations: number;
  unique_sources: number;
  occurrences: CitationOccurrence[];
  sources: CitedSourceSummary[];
}

export interface UploadedDocument {
  id: string;
  filename: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  status: "processing" | "ready" | "failed";
  error: string | null;
  content_sha256: string;
  embedding_model: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentAskResult {
  answered: boolean;
  answer: string;
  citations: {
    chunk_id: string;
    document_id: string;
    filename: string;
    page: number | null;
    quote: string;
  }[];
  tokens: number;
  llm: string;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface SystemMetrics {
  sessions: {
    total: number;
    by_status: Record<string, number>;
    by_mode: Record<string, number>;
    by_depth: Record<string, number>;
  };
  usage: {
    total_tokens: number;
    total_estimated_cost_usd: number;
    avg_tokens_per_session: number;
  };
  latency_ms: {
    completed_runs: number;
    p50: number | null;
    p95: number | null;
    by_agent: Record<string, { calls: number; avg_ms: number; max_ms: number }>;
  };
  quality: {
    reports_generated: number;
    validation_passed: number;
    validation_pass_rate: number | null;
    total_sources: number;
    injection_flagged_sources: number;
    uploaded_documents: number;
  };
}

export interface EvaluationRun {
  id: string;
  workflow_variant: "baseline" | "improved";
  dataset: string;
  metrics: Record<string, unknown>;
  created_at: string;
}
