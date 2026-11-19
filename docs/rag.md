# Documents and RAG

Upload PDFs, search them semantically, ask questions with page-level citations, and use them as evidence in
`document` and `hybrid` research. Everything below is implemented and tested (see "Limitations").

## Pipeline

```mermaid
flowchart LR
  U[POST /api/documents] --> V[validate: size, .pdf, %PDF header, sanitised name, SHA-256 duplicate check]
  V --> S[store as data/uploads/<uuid>.pdf]
  S --> BG[background worker]
  BG --> X[pypdf extraction<br/>page numbers, page/char caps, encrypted/scanned rejected]
  X --> C[page-aware chunking<br/>size + overlap, never spans pages]
  C --> E[embeddings<br/>hash | openai-compatible | sentence-transformers]
  E --> DB[(document_chunks<br/>text + page + embedding)]
  DB --> I[SqlIndex<br/>numpy cosine, cached, auto-invalidated]
  DB -.optional.-> CH[(Chroma<br/>backend process only)]
  Q[query] --> EQ[embed] --> I --> R[optional lexical rerank] --> OUT[chunks: document, page, score]
```

## Design decisions (with the evidence behind them)

- **The database is the source of truth** (text, page and embedding bytes per chunk). Searching it is
  process-safe, so the **MCP tool server** (a separate process) can search the same documents.
- **Chroma is optional** (`VECTOR_STORE=chroma`, `pip install chromadb`). It is a derived index rebuilt from the
  database (`ensure_synced` at startup). We tested cross-process use: an already-open Chroma client in a second
  process sees the new row count but **fails to query** (`Nothing found on disk`), so a Chroma index must never
  be shared between processes. The MCP server therefore always uses the SQL index. Chroma also pulls ~45
  packages (~400 MB), which is why it is not a hard requirement.
- **Embeddings.** `hash` (default) is deterministic and offline but **lexical, not semantic**: it matches shared
  words/bigrams, not meaning. For real semantic search set `EMBEDDING_PROVIDER=openai` (any OpenAI-compatible
  `/embeddings`, including local servers) or `sentence-transformers`. The embedding model name is stored per
  document; documents embedded with another model are excluded from search until `POST .../reindex`.
- **Chunks never span pages**, so every passage has exactly one page number. Page numbers on evidence are
  assigned by code from the matching chunk, never taken from the LLM.

## Safety

- Upload: size cap (streamed, `MAX_UPLOAD_MB`), `.pdf` extension **and** `%PDF-` header, filename reduced to a
  basename (files are stored under a generated id, never the user's name), duplicate detection by SHA-256,
  encrypted PDFs rejected, `MAX_PDF_PAGES` / `MAX_PDF_CHARS` limits, scanned PDFs rejected with a clear message.
- Retrieved text is untrusted: delimited and escaped in prompts, scanned for injection patterns, and every
  evidence passage / answer quotation must be **verbatim** in the retrieved chunk. Injection-flagged documents
  are marked in the report's References.
- Deleting a document removes the row, chunks, vectors and stored file (409 while a running research uses it).
  Past reports keep their sources and evidence.

## Page-level citations

Document sources are cited as `[S2, p.4]` (web sources as `[S1]`). The Writer and Critic reject: a document
cited without a page, a page that is not among the evidence pages of that source, and a page on a web source.
References list the pages cited: `paper.pdf (uploaded document, pp. 3, 4)`.

## API

`POST /api/documents` (202, or 200 + `duplicate:true`) · `GET /api/documents` (`status`, `q`, paging) ·
`GET /api/documents/{id}` · `GET .../{id}/chunks?page=` · `POST .../{id}/reindex` · `DELETE .../{id}` ·
`POST /api/documents/search` · `POST /api/documents/ask`.
`POST /api/research` with `"mode": "document"|"hybrid"` and `"document_ids": [...]` (validated: unknown or
not-ready documents give 422). Document-only runs never use the web, including follow-up research.

## Try it offline

```powershell
python scripts\make_demo_pdfs.py                  # synthetic PDFs -> evaluation\datasets\pdfs\
# with the API running (fixture/fake settings from the README):
curl.exe -F "file=@evaluation\datasets\pdfs\orchestra_x_whitepaper.pdf" http://localhost:8000/api/documents
```

## Limitations

- No OCR (scanned PDFs are rejected); text extraction is pypdf's (tables/columns may come out in odd order).
- The default `hash` embedder is lexical. Semantic quality needs a real embedding model (not tested here: no
  model downloads or paid keys in the test environment). The OpenAI-compatible and sentence-transformers
  adapters are tested against mocks/stubs only.
- The SQL index is brute-force numpy over all ready chunks (cached): fine for thousands of chunks, not millions;
  use Chroma/pgvector beyond that (pgvector is not implemented yet).
- Reranking is lexical BM25 blending, not a cross-encoder.
- Chroma runs only in the backend process; no multi-user ownership/ACLs on documents yet.
