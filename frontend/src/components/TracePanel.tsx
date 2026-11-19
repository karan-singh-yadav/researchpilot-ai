import React from "react";
import { Activity, AlertCircle, CheckCircle, Clock, Wrench } from "lucide-react";
import type { AgentEvent } from "../types/api";

interface Props {
  events: AgentEvent[];
  live: boolean;
}

export const TracePanel: React.FC<Props> = ({ events, live }) => {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 shadow">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-white">Execution Trace ({events.length} events)</h3>
        </div>
        {live && (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-xs font-medium text-emerald-300">
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" /> Live SSE
          </span>
        )}
      </div>

      <div className="mt-3 max-h-[420px] space-y-2 overflow-y-auto pr-1">
        {events.length === 0 ? (
          <div className="py-6 text-center text-xs text-slate-500">No trace events recorded yet.</div>
        ) : (
          events.map((ev) => {
            const isErr = ev.event_type === "error" || ev.event_type === "plan_rejected";
            const isTool = ev.event_type === "tool_call";
            return (
              <div
                key={ev.id}
                className="flex items-start justify-between gap-3 rounded-lg border border-slate-800/80 bg-slate-950/70 px-3 py-2 text-xs"
              >
                <div className="flex items-start gap-2.5">
                  {isErr ? (
                    <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-rose-400" />
                  ) : isTool ? (
                    <Wrench className="mt-0.5 h-3.5 w-3.5 shrink-0 text-sky-400" />
                  ) : (
                    <CheckCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-400" />
                  )}
                  <div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase text-indigo-300">
                        {ev.agent}
                      </span>
                      <span className="font-mono text-[10px] text-slate-500">{ev.event_type}</span>
                    </div>
                    <div className="mt-1 text-slate-200">{ev.summary}</div>
                  </div>
                </div>
                {ev.duration_ms !== null && (
                  <span className="inline-flex shrink-0 items-center gap-1 font-mono text-[11px] text-slate-400">
                    <Clock className="h-3 w-3" /> {ev.duration_ms}ms
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
