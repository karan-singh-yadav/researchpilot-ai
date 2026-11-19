import React, { useEffect, useState } from "react";
import { BarChart3, Play, RefreshCw } from "lucide-react";
import type { EvaluationRun, SystemMetrics } from "../types/api";
import { api } from "../services/api";

export const EvaluationPage: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [comparison, setComparison] = useState<Record<
    string,
    { baseline: number; improved: number; delta: number }
  > | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const loadAll = async () => {
    try {
      const [m, r] = await Promise.all([api.getMetrics(), api.listEvaluations()]);
      setMetrics(m);
      setRuns(r);
      try {
        const cmp = await api.compareEvaluations();
        setComparison(cmp.deltas);
      } catch {
        setComparison(null);
      }
    } catch (e: any) {
      setErr(e.message);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  const triggerBenchmark = async (variant: "baseline" | "improved") => {
    setRunning(variant);
    setErr(null);
    try {
      await api.runEvaluation(variant);
      await loadAll();
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setRunning(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white">Evaluation &amp; System Observability</h2>
          <p className="text-xs text-slate-400">
            Compare sequential baseline vs LangGraph multi-agent workflow and monitor token, latency, and critic metrics.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => triggerBenchmark("baseline")}
            disabled={running !== null}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-medium text-slate-200 hover:bg-slate-700 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" /> {running === "baseline" ? "Running Baseline..." : "Run Baseline Eval"}
          </button>
          <button
            onClick={() => triggerBenchmark("improved")}
            disabled={running !== null}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3.5 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" /> {running === "improved" ? "Running Improved..." : "Run Improved Eval"}
          </button>
          <button
            onClick={loadAll}
            className="rounded-lg border border-slate-700 p-2 text-slate-400 hover:text-white"
            title="Refresh metrics"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {err && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-950/50 px-4 py-2.5 text-xs text-rose-200">
          {err}
        </div>
      )}

      {/* System Telemetry Cards */}
      {metrics && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400">Total Research Sessions</div>
            <div className="mt-1 text-2xl font-bold text-white">{metrics.sessions.total}</div>
            <div className="mt-1 text-[11px] text-slate-500">
              Completed: {metrics.sessions.by_status.completed || 0} · Failed: {metrics.sessions.by_status.failed || 0}
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400">Total Tokens / Est. Cost</div>
            <div className="mt-1 text-2xl font-bold text-white">
              {metrics.usage.total_tokens.toLocaleString()}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              ${metrics.usage.total_estimated_cost_usd.toFixed(4)} USD · avg {metrics.usage.avg_tokens_per_session} tok/run
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400">Critic Pass Rate</div>
            <div className="mt-1 text-2xl font-bold text-emerald-400">
              {metrics.quality.validation_pass_rate !== null
                ? `${(metrics.quality.validation_pass_rate * 100).toFixed(1)}%`
                : "N/A"}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              {metrics.quality.validation_passed}/{metrics.quality.reports_generated} reports passed
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400">End-to-End Latency (p50 / p95)</div>
            <div className="mt-1 text-2xl font-bold text-white">
              {metrics.latency_ms.p50 !== null ? `${Math.round(metrics.latency_ms.p50)}ms` : "N/A"}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              p95: {metrics.latency_ms.p95 !== null ? `${Math.round(metrics.latency_ms.p95)}ms` : "N/A"}
            </div>
          </div>
        </div>
      )}

      {/* Baseline vs Improved Comparison Table */}
      {comparison && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-white">Baseline vs Improved Workflow Comparison</h3>
          </div>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 pr-4">Metric</th>
                  <th className="py-2 pr-4">Baseline (Sequential)</th>
                  <th className="py-2 pr-4">Improved (LangGraph + Critic)</th>
                  <th className="py-2">Delta</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {Object.entries(comparison).map(([k, v]) => (
                  <tr key={k}>
                    <td className="py-2 pr-4 font-mono text-slate-200">{k}</td>
                    <td className="py-2 pr-4 text-slate-300">{v.baseline}</td>
                    <td className="py-2 pr-4 font-semibold text-white">{v.improved}</td>
                    <td
                      className={`py-2 font-mono ${
                        v.delta > 0 ? "text-emerald-400" : v.delta < 0 ? "text-amber-400" : "text-slate-400"
                      }`}
                    >
                      {v.delta > 0 ? `+${v.delta}` : v.delta}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Historical Evaluation Runs */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
        <h3 className="text-sm font-semibold text-white">Evaluation Run History ({runs.length})</h3>
        <div className="mt-3 space-y-2 text-xs">
          {runs.length === 0 ? (
            <p className="text-slate-500">No evaluation runs recorded yet. Click Run Baseline / Improved Eval above.</p>
          ) : (
            runs.map((r) => (
              <div
                key={r.id}
                className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950/70 px-3 py-2"
              >
                <div>
                  <span
                    className={`mr-2 rounded px-2 py-0.5 font-mono text-[11px] font-semibold ${
                      r.workflow_variant === "improved"
                        ? "bg-indigo-500/20 text-indigo-300"
                        : "bg-slate-800 text-slate-300"
                    }`}
                  >
                    {r.workflow_variant}
                  </span>
                  <span className="text-slate-300">{r.dataset}</span>
                </div>
                <div className="font-mono text-[11px] text-slate-400">
                  completion={String(r.metrics.completion_rate)} · critic_pass={String(r.metrics.critic_pass_rate)} ·
                  tokens={String(r.metrics.avg_tokens)}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
