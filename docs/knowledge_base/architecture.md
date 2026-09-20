# Application architecture

LyustrAI separates product behavior from provider SDKs.
The reusable `ai_sdk` package owns domain contracts, retrieval, conversations,
memory, tools, agents, evaluation, MCP, and observability. Provider adapters
translate those stable contracts to Anthropic, OpenAI, and Gemini APIs. The
application does not expose provider response objects to the rest of the code.

The local product offers two modes. **Single Model** uses one provider selected
by the user. **Adaptive Multi-Model** uses a deterministic capability router in
`Auto`, while `Maximum` explicitly forces the three-stage FULL workflow. Both
modes use the same RAG pipeline and therefore
share indexed knowledge. The web layer isolates history by user-created chat;
provider and mode changes inside one chat continue that same history.

`AssistantRuntimeResources` is the composition boundary for shared retrieval state.
It owns one text chunker, one hybrid retriever, and one long-term memory store.
CLI and web managers receive these resources through explicit construction.
This avoids hidden globals and prevents separate provider managers from loading
inconsistent copies of the vector index.

The web application is intentionally a local, single-user demonstration. A
FastAPI layer serves a framework-free HTML, CSS, and JavaScript interface. It
does not include authentication, billing, multi-tenancy, or a public hosted
service. Those are product concerns rather than evidence needed for this
portfolio release.

The web conversation catalog stores each chat in its own validated
`chat_<12 hex>` directory with metadata and provider-neutral messages. Empty
`New chat` screens are local UI state and are not persisted. Rename and delete
are blocked while a run is active, and deleting a chat does not touch uploaded
documents, the vector index, or long-term memory.

Provider clients are created lazily when a chat request starts. As a result,
the web interface, readiness endpoint, document catalog, and local evaluation
can run without paid provider credentials. API keys stay in `.env`; readiness
responses contain booleans and missing variable names, never secret values.
