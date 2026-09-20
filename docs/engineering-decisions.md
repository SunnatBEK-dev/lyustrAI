# Engineering decisions

## 1. Provider-neutral contracts instead of a framework runtime

The project uses small abstract Python contracts for LLM turns, tool calls,
embeddings, vector storage, memory, tracing, and MCP transports. Anthropic,
OpenAI, and Gemini adapters translate those contracts at the boundary.

**Why:** core orchestration can be tested offline, provider migrations remain
localized, and vendor response objects do not become domain state.

**Tradeoff:** the project owns more adapter and validation code than an
application built directly on LangChain or another orchestration framework.

## 2. Deterministic adaptive routing instead of an LLM router

The capability router uses bounded lexical and structural signals to select
FAST, CONTEXT, REASONING, or FULL. Its decision includes safe reason codes and
an estimated request count.

**Why:** selection adds no provider request, has negligible local latency, and
can be evaluated with exact expected labels. The 24-case regression suite is
fully offline.

**Tradeoff:** hand-written signals cannot resolve every ambiguous intent. Route
accuracy is not answer quality, and the internal dataset is deliberately
reported as an internal benchmark.

## 3. Explicit retry ownership and cooperative cancellation

Provider SDK retries are disabled. One policy retries only classified transient
failures within bounded attempts and delay. A thread-safe token is checked at
safe workflow boundaries, while ordered progress events reveal no content.

**Why:** request counts, latency, failures, and cancellation behavior stay
observable and deterministic enough to test.

**Tradeoff:** a blocking provider request cannot be forcibly terminated. The UI
therefore describes cancellation as requested and stops at the next safe
boundary instead of claiming immediate interruption.

## 4. Local JSON persistence for the portfolio release

Conversations, memories, uploads, and vector data remain local. Each web chat
has a validated directory, metadata record, and provider-neutral message file,
so mode changes do not fragment its history. JSON writes are atomic where
replacement safety matters, and user data is ignored by Git.

**Why:** the release demonstrates AI-system boundaries without hiding them
behind unrelated SaaS infrastructure.

**Tradeoff:** this is not a multi-process or multi-tenant storage model. A
hosted release would require authenticated database-backed isolation.
