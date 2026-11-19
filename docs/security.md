# Security notes (implemented so far)

## Outbound fetching (SSRF)
1. `validate_public_url` on the URL and **every redirect hop** (scheme, credentials, port, hostname, DNS).
2. `SafeNetworkBackend` re-resolves DNS **at connect time**; all answers must be public; the socket
   connects to the validated IP (closes the DNS-rebinding window). Unix sockets refused. TLS still
   verifies the cert for the original hostname.
3. Manual bounded redirects; env proxies ignored; decoded-size cap (gzip bombs); content-type
   allow-list; total time limit. Not done: robots.txt, cookies, JS rendering.

## Untrusted content and prompt injection
- Web text, evidence and findings are delimited (`<untrusted_source>`, `<evidence>`, `<findings>`),
  with delimiter tags escaped (including entity-decoded text). Hidden HTML (display:none, hidden,
  aria-hidden) is removed. Instruction-like text is detected, flagged on the source, and dropped
  from evidence. These are defence in depth, not guarantees.
- Structural defences matter most: LLM output is schema-validated; agents have no side-effecting
  tools; the LLM cannot choose agents outside an enum, nor set budgets.

## Anti-fabrication guards
- Evidence passages must be verbatim in the fetched page. Writer output is rejected (bounded repair
  loop) for unknown citations, uncited sections, statistics not in the evidence, non-verbatim quotes.
- None of this proves factual correctness.

## Secrets / errors
- Keys only from environment; adapter errors never include keys, headers or response bodies;
  unhandled errors return a generic message and log details server-side.

## Workflow controls (Phase 4)
- Budgets (sources, iterations, revisions, wall-clock) come from the server by research depth; the LLM
  cannot change them. All loops are bounded and covered by tests.
- Agents have no side-effecting tools; plans can only name the `search` and `document` agents.
- Runs are cancellable and resumable; deleting a running session is refused (409).
- Events and API errors carry operational summaries only: no prompts, chain-of-thought or secrets.
- API errors for failed runs are generic ("error (TypeError)"); details stay in server logs.

## MCP tool layer (Phase 5)
- Strict, constraint-checked tool inputs (unknown arguments rejected); typed outputs re-validated by the client.
- Per-tool timeout, concurrency cap, rate limit, output-size caps, tool allow-list; read-only tools only.
- Least privilege: per-agent tool permissions (client-side), no tools for Analyst/Writer/Critic; the LLM API
  key is never passed to the tool server and `.env` is ignored by the spawned process.
- Unexpected tool exceptions are redacted at the protocol boundary; stdout is reserved for the protocol.
- See [mcp.md](mcp.md) for the full list and its limitations.
