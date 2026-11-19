import React, { useEffect, useState } from "react";
import { FileText, MessageSquare, Search, Trash2, Upload } from "lucide-react";
import type { DocumentAskResult, UploadedDocument } from "../types/api";
import { api } from "../services/api";

export const DocumentsPage: React.FC = () => {
  const [docs, setDocs] = useState<UploadedDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [askResult, setAskResult] = useState<DocumentAskResult | null>(null);
  const [asking, setAsking] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    try {
      const page = await api.listDocuments();
      setDocs(page.items);
    } catch (e: any) {
      setErr(e.message);
    }
  };

  useEffect(() => {
    load();
    const timer = setInterval(load, 4000);
    return () => clearInterval(timer);
  }, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setErr(null);
    try {
      await api.uploadDocument(file);
      await load();
    } catch (ex: any) {
      setErr(ex.message || "Upload failed");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.deleteDocument(id);
      setSelectedIds((prev) => prev.filter((x) => x !== id));
      await load();
    } catch (ex: any) {
      setErr(ex.message);
    }
  };

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    setAsking(true);
    setErr(null);
    try {
      const res = await api.askDocuments(question.trim(), selectedIds);
      setAskResult(res);
    } catch (ex: any) {
      setErr(ex.message);
    } finally {
      setAsking(false);
    }
  };

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white">Document Knowledge Base (PDF RAG)</h2>
          <p className="text-xs text-slate-400">
            Upload PDFs for page-aware chunking, semantic retrieval, and page-cited research ([S#, p.N]).
          </p>
        </div>

        <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow hover:bg-indigo-500">
          <Upload className="h-4 w-4" />
          {uploading ? "Uploading..." : "Upload PDF"}
          <input type="file" accept="application/pdf,.pdf" onChange={handleUpload} className="hidden" />
        </label>
      </div>

      {err && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-950/50 px-4 py-2.5 text-xs text-rose-200">
          {err}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Uploaded Documents List */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <h3 className="text-sm font-semibold text-white">Uploaded Documents ({docs.length})</h3>
          <div className="mt-3 space-y-2.5">
            {docs.length === 0 ? (
              <div className="rounded-lg border border-dashed border-slate-800 p-8 text-center text-xs text-slate-500">
                No PDFs uploaded yet. Upload a PDF to enable document &amp; hybrid research.
              </div>
            ) : (
              docs.map((d) => (
                <div
                  key={d.id}
                  className="flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-slate-950/70 p-3 text-xs"
                >
                  <label className="flex flex-1 cursor-pointer items-start gap-2.5">
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(d.id)}
                      onChange={() => toggleSelect(d.id)}
                      className="mt-0.5 rounded border-slate-700 bg-slate-900 text-indigo-600"
                    />
                    <div>
                      <div className="flex items-center gap-2 font-medium text-slate-100">
                        <FileText className="h-3.5 w-3.5 text-indigo-400" />
                        {d.filename}
                      </div>
                      <div className="mt-1 text-[11px] text-slate-400">
                        {d.page_count} pages · {d.chunk_count} chunks · {(d.size_bytes / 1024).toFixed(1)} KB
                      </div>
                    </div>
                  </label>

                  <div className="flex items-center gap-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${
                        d.status === "ready"
                          ? "bg-emerald-500/15 text-emerald-300"
                          : d.status === "failed"
                          ? "bg-rose-500/15 text-rose-300"
                          : "bg-amber-500/15 text-amber-300"
                      }`}
                    >
                      {d.status}
                    </span>
                    <button
                      onClick={() => handleDelete(d.id)}
                      className="text-slate-500 hover:text-rose-400"
                      title="Delete document"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Direct Document Ask / Retrieval Playground */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-white">Ask Documents (Verified Page Citations)</h3>
          </div>
          <p className="mt-1 text-xs text-slate-400">
            {selectedIds.length > 0
              ? `Scoped to ${selectedIds.length} selected document(s).`
              : "Searching across all ready documents."}
          </p>

          <form onSubmit={handleAsk} className="mt-3 flex gap-2">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a specific question grounded in your PDFs..."
              className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100 focus:border-indigo-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={asking || !question.trim()}
              className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3.5 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
            >
              <Search className="h-3.5 w-3.5" /> {asking ? "Asking..." : "Ask"}
            </button>
          </form>

          {askResult && (
            <div className="mt-4 space-y-3 rounded-lg border border-slate-800 bg-slate-950 p-4 text-xs">
              <div className="leading-relaxed text-slate-100">{askResult.answer}</div>
              {askResult.citations.length > 0 && (
                <div className="space-y-1.5 border-t border-slate-800 pt-2.5">
                  <div className="font-semibold text-slate-400">Page-Verified Quotes:</div>
                  {askResult.citations.map((c, i) => (
                    <div key={i} className="rounded bg-slate-900 p-2 text-[11px]">
                      <span className="font-semibold text-indigo-300">
                        {c.filename} {c.page ? `(p.${c.page})` : ""}
                      </span>
                      <p className="mt-1 italic text-slate-300">&ldquo;{c.quote}&rdquo;</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
