import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Download,
  ExternalLink,
  FileText,
  ShieldAlert,
} from "lucide-react";
import type { CitationOccurrence, ReportCitations, ReportOut, SourceItem } from "../types/api";
import { api } from "../services/api";

interface Props {
  researchId: string;
  report: ReportOut;
  citations: ReportCitations | null;
  sources: SourceItem[];
}

export const ReportViewer: React.FC<Props> = ({ researchId, report, citations, sources }) => {
  const [selectedRef, setSelectedRef] = useState<string | null>(null);
  const [selectedPage, setSelectedPage] = useState<number | null>(null);

  const sourceMap = useMemo(() => {
    const m = new Map<string, SourceItem>();
    for (const s of sources) m.set(s.source_ref, s);
    return m;
  }, [sources]);

  const activeSource = selectedRef ? sourceMap.get(selectedRef) || null : null;

  const handleDownload = async (fmt: "markdown" | "json") => {
    const content = await api.exportReport(researchId, fmt);
    const blob = new Blob([content], {
      type: fmt === "markdown" ? "text/markdown;charset=utf-8" : "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `research-${researchId.slice(0, 8)}.${fmt === "markdown" ? "md" : "json"}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Render markdown paragraphs with interactive clickable citation pills
  const renderLineWithCitations = (line: string, lineKey: number) => {
    // Strip markdown anchor links [[S1]](#ref-S1) or [[S2, p.4]](#ref-S2) into interactive pills
    const tokenRegex = /\[?\[(S\d+)(?:,\s*p\.\s*(\d+))?\]\]?(?:\(#ref-S\d+\))?/g;
    const parts: React.ReactNode[] = [];
    let lastIdx = 0;
    let match: RegExpExecArray | null;

    while ((match = tokenRegex.exec(line)) !== null) {
      if (match.index > lastIdx) {
        parts.push(line.slice(lastIdx, match.index));
      }
      const ref = match[1];
      const page = match[2] ? Number(match[2]) : null;
      const src = sourceMap.get(ref);
      const isSelected = selectedRef === ref && selectedPage === page;

      parts.push(
        <button
          key={`${lineKey}-${match.index}`}
          onClick={() => {
            setSelectedRef(ref);
            setSelectedPage(page);
          }}
          title={src ? `${src.title}${page ? ` (Page ${page})` : ""}` : ref}
          className={`mx-0.5 inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-mono text-[11px] font-semibold transition ${
            isSelected
              ? "bg-indigo-500 text-white ring-2 ring-indigo-300"
              : src?.injection_suspected
              ? "bg-amber-500/20 text-amber-300 hover:bg-amber-500/30"
              : "bg-indigo-500/20 text-indigo-300 hover:bg-indigo-500/30"
          }`}
        >
          [{ref}
          {page ? `, p.${page}` : ""}]
        </button>
      );
      lastIdx = match.index + match[0].length;
    }

    if (lastIdx < line.length) {
      parts.push(line.slice(lastIdx));
    }
    return parts;
  };

  const renderedBody = useMemo(() => {
    const lines = report.markdown.split("\n");
    const elements: React.ReactNode[] = [];
    let inReferences = false;

    lines.forEach((rawLine, idx) => {
      const line = rawLine.trimEnd();
      if (line.startsWith("# ")) {
        elements.push(
          <h1 key={idx} className="mb-3 text-2xl font-bold text-white">
            {line.slice(2)}
          </h1>
        );
      } else if (line.startsWith("## ")) {
        const heading = line.slice(3).trim();
        inReferences = heading.toLowerCase() === "references";
        elements.push(
          <h2 key={idx} className="mb-2 mt-6 border-b border-slate-800 pb-1.5 text-lg font-semibold text-slate-100">
            {heading}
          </h2>
        );
      } else if (line.startsWith("### ")) {
        elements.push(
          <h3 key={idx} className="mb-2 mt-4 text-base font-semibold text-slate-200">
            {line.slice(4)}
          </h3>
        );
      } else if (line.startsWith("> ")) {
        elements.push(
          <blockquote
            key={idx}
            className="my-2 border-l-2 border-amber-500/70 bg-amber-950/20 px-3 py-1.5 text-xs text-amber-200"
          >
            {line.slice(2)}
          </blockquote>
        );
      } else if (line.startsWith("|")) {
        elements.push(
          <div key={idx} className="font-mono text-xs text-slate-300 overflow-x-auto">
            {renderLineWithCitations(line, idx)}
          </div>
        );
      } else if (line.startsWith("- ") || line.startsWith("* ")) {
        elements.push(
          <li key={idx} className="ml-5 list-disc text-sm leading-relaxed text-slate-200">
            {renderLineWithCitations(line.slice(2), idx)}
          </li>
        );
      } else if (!line.trim()) {
        elements.push(<div key={idx} className="h-2" />);
      } else if (inReferences) {
        const cleanRefLine = line.replace(/<a id="[^"]+"><\/a>/g, "");
        elements.push(
          <p key={idx} className="my-1.5 text-xs text-slate-300">
            {renderLineWithCitations(cleanRefLine, idx)}
          </p>
        );
      } else {
        elements.push(
          <p key={idx} className="my-2 text-sm leading-relaxed text-slate-200">
            {renderLineWithCitations(line, idx)}
          </p>
        );
      }
    });

    return elements;
  }, [report.markdown, selectedRef, selectedPage, sourceMap]);

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      {/* Main Report Column */}
      <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/70 p-6 shadow">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2.5">
            <span className="rounded-md bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-300">
              v{report.version}
            </span>
            {report.validation && (
              <span
                className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                  report.validation.passed
                    ? "bg-emerald-500/15 text-emerald-300"
                    : "bg-amber-500/15 text-amber-300"
                }`}
              >
                {report.validation.passed ? (
                  <>
                    <CheckCircle2 className="h-3.5 w-3.5" /> Citation Checks Passed
                  </>
                ) : (
                  <>
                    <AlertTriangle className="h-3.5 w-3.5" /> Validation Warnings ({report.validation.issues.length})
                  </>
                )}
              </span>
            )}
            {citations && (
              <span className="text-xs text-slate-400">
                {citations.total_citations} citations · {citations.unique_sources} sources
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => handleDownload("markdown")}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs text-slate-200 hover:bg-slate-700"
            >
              <Download className="h-3.5 w-3.5" /> .md
            </button>
            <button
              onClick={() => handleDownload("json")}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs text-slate-200 hover:bg-slate-700"
            >
              <Download className="h-3.5 w-3.5" /> .json
            </button>
          </div>
        </div>

        <article className="max-w-none">{renderedBody}</article>

        {report.validation && (
          <div className="mt-6 rounded-lg border border-slate-800 bg-slate-950/70 p-4 text-xs">
            <div className="font-semibold text-slate-300">
              Critic Verification ({report.validation.claims_checked} claims reviewed)
            </div>
            <p className="mt-1 text-[11px] text-slate-400">{report.validation.disclaimer}</p>
            {report.validation.issues.length > 0 && (
              <ul className="mt-2 space-y-1">
                {report.validation.issues.map((iss, i) => (
                  <li
                    key={i}
                    className={iss.severity === "error" ? "text-rose-300" : "text-amber-300"}
                  >
                    [{iss.severity.toUpperCase()} · {iss.method}] {iss.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* Interactive Citation & Evidence Inspector Drawer */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 shadow">
        <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
          <BookOpen className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-white">Citation &amp; Evidence Inspector</h3>
        </div>

        <p className="mt-2 text-xs text-slate-400">
          Click any <span className="font-mono text-indigo-300">[S#]</span> or{" "}
          <span className="font-mono text-indigo-300">[S#, p.N]</span> badge in the report to inspect its verbatim
          passage.
        </p>

        {/* Source Pills */}
        <div className="mt-3 flex flex-wrap gap-1.5">
          {sources.map((s) => (
            <button
              key={s.source_ref}
              onClick={() => {
                setSelectedRef(s.source_ref);
                setSelectedPage(null);
              }}
              className={`rounded-md px-2 py-1 font-mono text-xs transition ${
                selectedRef === s.source_ref
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {s.source_ref}
            </button>
          ))}
        </div>

        {activeSource ? (
          <div className="mt-4 space-y-3">
            <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
              <div className="flex items-start justify-between gap-2">
                <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 font-mono text-xs font-semibold text-indigo-300">
                  {activeSource.source_ref} · {activeSource.kind}
                </span>
                {selectedPage && (
                  <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 font-mono text-xs text-emerald-300">
                    Page {selectedPage}
                  </span>
                )}
              </div>
              <div className="mt-2 text-sm font-medium text-white">{activeSource.title}</div>
              {activeSource.kind === "web" && activeSource.url && (
                <a
                  href={activeSource.url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1 inline-flex items-center gap-1 break-all text-xs text-indigo-400 hover:underline"
                >
                  {activeSource.url} <ExternalLink className="h-3 w-3 shrink-0" />
                </a>
              )}
              {activeSource.injection_suspected && (
                <div className="mt-2 flex items-center gap-1.5 rounded bg-amber-950/60 px-2 py-1 text-[11px] text-amber-300">
                  <ShieldAlert className="h-3.5 w-3.5 shrink-0" />
                  Prompt-injection pattern detected &amp; isolated
                </div>
              )}
            </div>

            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Verified Verbatim Evidence
              </div>
              {(selectedPage
                ? activeSource.evidence.filter((e) => e.page_number === selectedPage)
                : activeSource.evidence
              ).map((ev) => (
                <div
                  key={ev.evidence_ref}
                  className="rounded-lg border border-slate-800 bg-slate-950/80 p-3 text-xs"
                >
                  <div className="flex items-center justify-between text-[11px] text-slate-400">
                    <span className="font-mono font-semibold text-indigo-300">{ev.evidence_ref}</span>
                    <span>
                      {ev.kind}
                      {ev.page_number ? ` · p.${ev.page_number}` : ""}
                    </span>
                  </div>
                  <blockquote className="mt-1.5 border-l-2 border-indigo-500 pl-2.5 italic text-slate-200">
                    &ldquo;{ev.passage}&rdquo;
                  </blockquote>
                  <div className="mt-1.5 text-[11px] text-slate-400">Claim: {ev.claim}</div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="mt-6 flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 p-6 text-center text-xs text-slate-500">
            <FileText className="mb-2 h-6 w-6 text-slate-600" />
            Select a citation badge or source pill above to view verified evidence passages.
          </div>
        )}
      </div>
    </div>
  );
};
