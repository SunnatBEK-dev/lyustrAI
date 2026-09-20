# Capabilities and SDK reference

This document preserves the detailed technical reference behind LyustrAI. The
project favors explicit contracts and small
abstractions over framework-specific magic.

## Current status

The application architecture foundation is complete through evaluation phase
9.5, three provider integrations, and the first two-mode application workflow:

- conversation and message domain models;
- JSON repository abstraction;
- provider-neutral prompt construction;
- provider-neutral token counting with a deterministic local estimator;
- turn-aware sliding context windows with a configurable token budget;
- bounded extractive summary memory for turns outside the active window;
- explicit persistent long-term memory with lexical relevance retrieval;
- provider-neutral tool schemas with JSON Schema export;
- strict tool argument validation and allow-listed execution;
- structured tool calls and serialized success/error results;
- Claude tool-use request/response translation;
- OpenAI Responses API request/response translation;
- bounded multi-round tool execution with duplicate-call protection;
- optional tool-enabled conversation and RAG orchestration;
- provider-neutral `AgentRunner` loop policy;
- explicit agent state, iteration events, and termination reasons;
- provider-neutral LLM planner with strict bounded JSON output;
- ordered plan steps with controlled status transitions and outcomes;
- bounded reflection for finished agent states and terminal plans;
- structured reflection verdicts, strengths, and improvements;
- named agent workers with explicit task assignment;
- provider-bound worker construction for Anthropic, OpenAI, and Gemini;
- deterministic multi-agent result collection and failure isolation;
- explicit Single Model and Adaptive Multi-Model application modes;
- independently persisted web chats that can switch mode and provider;
- validated, bounded structured handoffs between named provider workers;
- deterministic dependency-graph validation and topological execution;
- failed-branch pruning with independent-stage continuation;
- explainable deterministic capability routing without a model call;
- user-selectable Auto routing or a forced Maximum FULL workflow;
- one-, two-, and three-provider Adaptive Multi-Model workflow variants;
- bilingual 24-case offline route-selection benchmark;
- route accuracy, request-count, savings, and local-latency reports;
- bounded content-free Adaptive Multi-Model route and stage runtime metrics;
- a local provider-readiness check that never prints key values;
- bounded transient-failure retries for Adaptive Multi-Model provider stages;
- ordered content-free Adaptive Multi-Model progress and cooperative cancellation;
- a composite Adaptive Multi-Model client that returns one final response;
- stateless MCP `2026-07-28` request metadata;
- provider-neutral MCP client and transport contracts;
- explicit optional server discovery and capability checks;
- ordered, cursor-based MCP tool and resource catalogs;
- MCP cache-hint preservation for list and discovery results;
- explicit MCP transport lifecycle, timeouts, and failure isolation;
- explicit MCP tool-call and resource-read contracts;
- generic MCP content plus structured tool-result preservation;
- explicitly approved MCP tool registration into the local allow-list;
- atomic rejection of MCP schemas the local validator cannot enforce;
- dependency-free stateless MCP Streamable HTTP transport;
- strict JSON-RPC response validation for JSON and request-scoped SSE;
- fresh per-request MCP authorization injection;
- safe MCP routing headers and approved `x-mcp-header` argument mirroring;
- bounded HTTP responses, disabled redirects, and contained network errors;
- manual MCP multi round-trip continuations for tools and resources;
- opaque request-state echoing with fresh JSON-RPC request identifiers;
- exact input-response matching, one-use continuations, and bounded rounds;
- provider-neutral trace and span records with validated identifiers;
- nested parent-child trace context for workflows and operations;
- bounded, thread-safe in-memory trace collection;
- opt-in tracing across LLM, retrieval, tool, agent, and MCP operations;
- safe trace metadata with sensitive-field redaction and error types only;
- explicit text eval cases and a provider-neutral evaluator contract;
- deterministic multi-evaluator runs with per-case failure isolation;
- evaluator thresholds and a configurable suite pass-rate threshold;
- aggregate pass-rate, error-count, failed-case, and mean-score reports;
- Anthropic/Claude adapter behind an LLM contract;
- OpenAI text, streaming, and function-calling adapter;
- Gemini Interactions API text, streaming, and function-calling adapter;
- environment-selected Anthropic, OpenAI, or Gemini client creation;
- provider-neutral embedding-client contract;
- lazy SentenceTransformer embedding adapter;
- Document and Chunk retrieval domain models;
- deterministic character-based TextChunker with overlap;
- dependency-free cosine similarity and deterministic top-k search;
- dependency-free BM25 lexical search;
- weighted reciprocal-rank fusion for hybrid retrieval;
- provider-neutral VectorStore with in-memory and persistent JSON adapters;
- atomic document re-indexing and document-level chunk deletion;
- SemanticRetriever orchestration for indexing and search;
- retrieval-aware PromptBuilder without domain-state mutation;
- RAGConversationManager for indexing, retrieval, generation, and persistence;
- offline retrieval evaluation with Hit Rate@k, Recall@k, and MRR;
- baseline-versus-candidate retrieval comparison with metric deltas;
- RAG-enabled CLI for indexing, listing, and removing text documents;
- structured RAG responses with deterministic local source citations;
- document catalog summaries with source paths and chunk counts;
- loader-based ingestion for text files and recursive directories;
- content-hash-based incremental directory synchronization;
- stale indexed-document cleanup when source files disappear;
- conversation orchestration with rollback behavior;
- embedding cache persistence;
- isolated unit tests and opt-in integration tests.

The project now has a complete offline-tested RAG pipeline exposed through the
CLI, deterministic retrieval-quality evaluation, restart-safe local vector
persistence, document-level index lifecycle, source-aware answers, and a
loader-based ingestion layer. Normal RAG queries combine semantic similarity
with exact lexical evidence. The evaluation layer can compare this hybrid
candidate against a semantic baseline and explicitly report improvements or
regressions. Long conversations retain their full persisted history while only
the newest complete turns within the configured token budget are sent directly
to the model. Recent excluded turns are compressed into a bounded local memory
block without another API call. Directory-backed indexes also remain
synchronized across application restarts without embedding unchanged files
again. User-approved durable facts are stored separately and only relevant
matches are added to a new prompt. The SDK also has an offline-tested tool loop
that sends registered schemas to the configured provider, validates every
requested call,
dispatches only explicitly registered Python handlers, and returns structured
results until the model produces a final answer. The loop policy lives in a
provider-neutral agent runtime while Claude, OpenAI, and Gemini only translate
one model turn at a time. The MCP foundation follows the stateless `2026-07-28`
protocol: every protocol request carries its own version, client identity, and
capabilities.
Opening a transport does not perform a hidden handshake, and
`server/discover` remains an explicit optional call. A concrete Streamable HTTP
transport now maps each operation to a separate POST, accepts JSON or
request-scoped SSE responses, validates JSON-RPC identifiers and result
framing, and obtains authorization immediately before each request.
When a tool call or resource read needs host input, the SDK returns a local
continuation instead of answering automatically. The host can inspect the
requested interaction, gather an approved response, resume it once, or cancel
it locally. Phase 9.1 adds optional tracing to the main workflow boundaries.
Components that share one `Tracer` produce a single parent-child operation
tree with bounded timing, status, counts, and exception type metadata. Raw
prompts, model responses, tool arguments, credentials, and MCP request state
are not collected by the built-in instrumentation.

Phase 9.2 adds a small offline evaluation harness beside the specialized
retrieval metrics. It runs an application target once per explicit case,
applies one or more deterministic evaluators, isolates case failures, and
produces an aggregate quality report without retaining generated outputs.
The specialized route suite evaluates Adaptive Multi-Model routing without calling a
provider or retaining benchmark prompts in its results.

Phase 9.3 adds secret-free configuration readiness reporting and bounded local
runtime metrics for Adaptive Multi-Model workflows. Runtime records contain only
route, stage, status, exception-class, and timing metadata.

Phase 9.4 adds one explicit retry owner for Adaptive Multi-Model provider stages. The
provider libraries' hidden retries are disabled, and `AgentRunner` may retry
only known transient connection, timeout, rate-limit, conflict, and server
errors within a deterministic attempt and delay limit. Permanent failures are
returned immediately.

Phase 9.5 adds ordered route, stage, and terminal progress events plus
thread-safe cooperative cancellation. Cancellation is checked before provider
attempts, retries, tool execution, and handoff stages without retaining user
content.

The CLI now exposes that foundation through two explicit product modes. Direct
Chat sends a turn to one user-selected provider. Adaptive Multi-Model deterministically
selects a one-, two-, or three-provider workflow from the request, uses
validated JSON handoffs where needed, and persists one final answer.

## Architecture

```text
app/cli.py -> AssistantMode
    |-- Single Model -> RAGConversationManager -> selected provider client
    `-- Adaptive Multi-Model -> RAGConversationManager -> AdaptiveMultiModelClient
                                            |-- progress events / cancel token
                                            `-- CapabilityRouter
                                                |-- FAST -> OpenAI synthesis
                                                |-- CONTEXT -> Gemini -> OpenAI
                                                |-- REASONING -> Claude -> OpenAI
                                                `-- FULL -> Gemini -> Claude -> OpenAI

RAGConversationManager
    |-- Conversation / Message
    |-- PromptBuilder -> SlidingContextWindow -> LLMMessage
    |                    |-- TokenCounter -> RegexTokenCounter
    |                    `-- ExtractiveConversationSummarizer
    |-- MultiAgentCoordinator -> AgentWorker(provider) -> isolated AgentRunner
    |-- SequentialHandoffCoordinator -> ordered, bounded stage outputs
    |-- DependencyHandoffCoordinator -> validated DAG -> declared inputs
    |-- AgentRunner -> RetryPolicy -> AgentState -> AgentEvent
    |                 |-- ToolRegistry -> ToolExecutor -> ToolResult
    |                 `-- BaseToolLLMClient -> provider adapter
    |-- LLMAgentPlanner -> AgentPlan -> PlanStep
    |-- LLMAgentReflector -> AgentReflection
    |-- BaseLLMClient -> provider factory
    |                     |-- AnthropicClient
    |                     |-- OpenAIClient
    |                     `-- GeminiClient
    |-- BaseEmbeddingClient -> SentenceTransformerEmbeddingClient
    |-- HybridRetriever -> semantic search + BM25 + rank fusion
    |                       `-- BaseVectorStore
    |                           |-- InMemoryVectorStore
    |                           `-- JSONVectorStore
    |-- Retrieval results -> Citation / RAGResponse
    |-- DirectorySynchronizer -> DocumentIngestor
    |                            `-- BaseDocumentLoader -> TextDocumentLoader
    |-- BaseMemoryStore -> JSONMemoryStore -> BM25 recall
    `-- ConversationRepository -> JSONConversationRepository

MCPClient -> MCPRequestContext
    `-- BaseMCPTransport -> StreamableHTTPTransport -> HTTP MCP endpoint
        |-- optional MCPDiscoveryResult
        |-- ordered MCPToolPage
        |-- ordered MCPResourcePage
        |-- MCPToolRequest -> MCPToolResult
        |-- MCPResourceReadRequest -> MCPResourceReadResult
        `-- input_required -> MCPContinuation -> continue_request

MCPToolAdapter -> approved compatible MCPTool -> ToolRegistry
    `-- ToolExecutor -> MCPClient.call_tool

Tracer -> TraceCollector -> InMemoryTraceCollector
    `-- TraceRecord (trace/span IDs, parent, timing, status, safe attributes)

EvaluationRunner -> EvalCase -> application target
    `-- Evaluator -> EvalScore -> EvalCaseResult -> EvaluationReport

RouteEvaluationRunner -> RouteEvalCase -> CapabilityRouter
    `-- RouteEvalResult -> accuracy + estimated requests + local latency
```

The domain layer does not know about JSON, filesystem paths, Anthropic,
OpenAI, Gemini, API keys, or environment variables. Provider-specific
behavior stays inside the provider adapter.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

Install local SentenceTransformer support before using `/index`:

```bash
.venv/bin/python -m pip install -e '.[embeddings]'
```

`pyproject.toml` is the dependency source of truth. `requirements.txt` and
`requirements-dev.txt` are convenience entry points.

## Configuration

Copy `.env.example` to `.env` and provide:

```text
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
OPENAI_API_KEY=...
OPENAI_MODEL=...
GEMINI_API_KEY=...
GEMINI_MODEL=...
MAX_TOKENS=1024
TIMEOUT=60
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=3
CONTEXT_TOKEN_BUDGET=3000
CONTEXT_SUMMARY_TOKEN_BUDGET=400
MEMORY_RETRIEVAL_K=3
LLM_RETRY_MAX_ATTEMPTS=3
LLM_RETRY_INITIAL_DELAY=0.25
LLM_RETRY_MAX_DELAY=2.0
```

Never commit `.env`, API keys, or real conversation data.

## LLM providers

Set `AI_PROVIDER` to `anthropic`, `openai`, or `gemini`. The CLI creates the
matching adapter while the conversation, RAG, memory, tool, agent, tracing,
and eval layers remain unchanged. Existing setups may keep using `MODEL` as a
fallback, but provider-specific model variables are clearer:

```text
AI_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

For Gemini:

```text
AI_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=...
```

`OpenAIClient` uses the
[OpenAI Responses API](https://developers.openai.com/api/docs/guides/text),
including
[streaming response events](https://developers.openai.com/api/docs/guides/streaming-responses)
and
[function calling](https://developers.openai.com/api/docs/guides/function-calling).
It reconstructs each tool turn locally, sends `store=False`, and does not rely
on a provider-side conversation ID. This integration requires an OpenAI API
key and does not reuse a consumer ChatGPT browser session. There is no
automatic provider fallback or load balancing.

`GeminiClient` uses Google's
[Interactions API](https://ai.google.dev/gemini-api/docs/text-generation) for
text and streaming, plus its
[function-calling contract](https://ai.google.dev/gemini-api/docs/function-calling).
Tool turns are reconstructed locally with every model-generated step preserved
and `store=False`, so the adapter does not rely on a provider-side conversation
ID.

## Application modes

The CLI asks for an application mode at startup:

1. **Single Model** asks the user to select Claude, OpenAI, or Gemini. Each turn
   goes only to that provider. Conversation history is isolated by provider in
   `data/conversations/single_model/`, so switching providers does not mix chat
   histories.
2. **Adaptive Multi-Model** selects one of four explicit routes. FAST uses OpenAI synthesis;
   CONTEXT uses Gemini then OpenAI; REASONING uses Claude then OpenAI; FULL uses
   Gemini context, Claude reasoning, and OpenAI synthesis. These are configurable
   workflow roles, not claims that one provider is universally best at a role.

Both modes use the same conversation, RAG, document, and long-term-memory
features. Adaptive Multi-Model stores its final conversation in
`data/conversations/adaptive_multi_model.json`.
The context and reasoning stages must return exactly four validated fields:
`summary`, `facts`, `uncertainties`, and `recommendations`. Each stage receives
only its explicitly declared, bounded dependency payloads as untrusted JSON
data. The FULL route's final stage depends on both context and reasoning, then
returns ordinary user-facing text. Intermediate payloads are not added to the
user-visible conversation history.

The local web interface adds a chat catalog above those reusable SDK modes.
Each chat is stored under
`data/conversations/chats/<conversation_id>/` with separate metadata and
messages. A chat may switch between Single Model providers and Adaptive
Multi-Model while preserving one history. Its last Auto or Maximum effort
choice is restored when reopened. `New chat` remains browser-only until
the first message is submitted; chats can then be loaded, renamed, or
permanently deleted without changing documents, the vector index, or long-term
memory.

Every Adaptive Multi-Model run also records a bounded in-memory metric containing
only the selected route, safe routing signals, executed/failed/blocked stage
IDs, completion state, exception class, and elapsed time. Prompts, provider
outputs, exception messages, credentials, and model names are not retained.
Use `/metrics` in Adaptive Multi-Model mode to see the aggregate route, stage, success, and
latency summary. These statistics reset when the application restarts.

Single Model needs only the selected provider's API key and normally makes one
model request per ordinary chat turn. Adaptive Multi-Model needs valid Gemini, Anthropic,
and OpenAI configuration at startup. Depending on the selected route, it makes
one, two, or three model requests per chat turn. If one stage fails, dependent
stages are not run and the incomplete user turn is rolled back. Adaptive Multi-Model
response streaming, automatic provider fallback, and parallel execution are
not implemented yet.

Adaptive Multi-Model provider stages default to at most three total attempts, including
the original request. Delays use bounded exponential backoff. Authentication,
permission, invalid-request, response-validation, and application errors are
never retried. A retry occurs before an `AgentState` event or tool execution is
recorded, so the same tool is not executed twice by this policy. Streaming is
not retried because a partial response cannot be replayed safely. Successful
model-turn traces expose only the numeric request-attempt count.

The CLI prints route and stage progress for Adaptive Multi-Model without showing internal
handoff content. A host application may call `AdaptiveMultiModelClient.cancel()`
from another thread or request handler; it returns `True` only when it changes
an active run to cancelled. The current provider or tool call is not forcibly
interrupted. After it returns, no retry, later tool, or handoff stage begins,
the returned partial model output is discarded, and the incomplete
conversation turn is rolled back. External side effects from a tool that had
already started cannot be undone by the conversation rollback. Cancellation
tokens are per-run, progress remains local and in memory, and the adaptive client
rejects concurrent runs. Progress events contain only sequence, route, stage
ID, status, and stage counts.

`CapabilityRouter` is local and deterministic; it does not make a routing model
request. Short ordinary prompts select FAST. Source, evidence, document, or RAG
context signals select CONTEXT. Analysis, comparison, planning, or multi-part
signals select REASONING. Requests containing both capability groups, or long
requests, select FULL. The decision exposes its route, required capabilities,
and matched signals for tests and future observability.
The web `Maximum` strategy supplies an explicit FULL override and adds a safe
`user_override` routing signal. `Auto` retains the default signal-based logic.

## Run the CLI

Before starting the CLI, check local provider configuration without making a
network request or printing any secret value:

```bash
.venv/bin/python app/check_provider_readiness.py
```

Single Model can use any provider reported as `READY`. Adaptive Multi-Model is reported as
`READY` only when all three provider API keys and model names are configured.

```bash
.venv/bin/python app/cli.py
```

RAG document commands are:

```text
/index /absolute/path/to/guide.txt
/index /absolute/path/to/knowledge-directory
/documents
/remove doc_123456789abc
```

Long-term memory commands are:

```text
/remember Preferred language is Uzbek
/memories
/forget mem_123456789abc
```

Indexing the same file again replaces its old chunks. Indexing a directory
performs an incremental synchronization: new and changed files are indexed,
unchanged files are skipped, and documents whose source files disappeared are
removed. Normal prompts combine embedding similarity and exact term matches,
then use the highest-ranked chunks. `/history`, `/save`, `/clear`, `/help`,
and `/exit` remain available. The first indexing run may download the
configured SentenceTransformer model.

`CONTEXT_TOKEN_BUDGET` limits conversation memory sent to the model. Selection
keeps complete recent user/assistant turns and always retains the newest turn,
even when that turn alone exceeds the soft budget. `/history` and JSON
persistence continue to keep the complete conversation.
`CONTEXT_SUMMARY_TOKEN_BUDGET` limits the deterministic extractive memory made
from excluded turns. This local summarizer adds no model request, token charge,
or network latency.

Long-term memory is explicit: the application never guesses which facts to
store. `/remember` persists a user-approved fact in `data/memories.json`, and
lexical retrieval adds only matching memories to later prompts. `/clear`
removes conversation history but leaves long-term memory intact; use `/forget`
to delete a stored fact.

The tool layer supports string, integer, number, and boolean parameters,
required and optional arguments, deterministic JSON Schema export, duplicate
registration protection, strict unknown-argument rejection, and contained
handler errors. It does not use dynamic imports, `eval`, or shell execution.
When a `ToolExecutor` is configured on `ConversationManager` or
`RAGConversationManager`, Claude, OpenAI, or Gemini may request one or more
registered tools in a round. Results, including contained validation and
execution errors, are sent back until the provider returns final text.
`AgentRunner` defaults to at most eight tool rounds and rejects duplicate call
IDs. Its returned `AgentState` contains
ordered `AgentEvent` records and ends with either `final_response` or
`max_tool_rounds`. Tool traces are transient when using the conversation
manager; persisted history contains the original user message and final
assistant answer. Applications that need the trace can invoke
`AgentRunner.run()` directly and optionally receive each event through a
callback. Tool-enabled streaming is intentionally unsupported for now, so use
`send_message()` for this workflow. The SDK does not register application
tools automatically—the host application owns that allow-list.

`LLMAgentPlanner` is an opt-in provider-neutral planning layer. It asks any
`BaseLLMClient` for a bounded JSON list of unique steps and converts that list
into an `AgentPlan`. Plans expose only controlled sequential transitions from
`pending` to `in_progress`, then `completed` or `failed`, with an optional
outcome stored for each completed step. Planning does not automatically execute
steps or re-plan: the host application explicitly passes each active step to
`AgentRunner`, which keeps planning policy separate from tool execution.

`LLMAgentReflector` performs one opt-in review of a finished `AgentState` or a
completed/failed `AgentPlan`. It returns a strict verdict (`passed`,
`needs_improvement`, or `failed`) plus bounded lists of strengths and
improvements. Reflection never mutates the reviewed object, executes tools,
retries work, or applies its own suggestions. Snapshot input is capped and may
be truncated; because it is sent to the configured LLM provider, host
applications should not include secrets in state, plan outcomes, or tool
results.

`MultiAgentCoordinator` is an explicit sequential coordinator. Named workers
are registered with one responsibility and one `AgentRunner`; every task names
its worker directly. The coordinator validates all assignments before any work
starts, creates a fresh `AgentState` for each task, preserves input order in
the result, and contains one worker failure so later tasks can continue.
Exception messages are not exposed. Isolation is at the agent-state level, not
an operating-system process boundary, so stateful clients and tool handlers
remain the host application's responsibility. The coordinator does not perform
automatic delegation, implicit handoffs, shared-state mutation, parallel
execution, or recursive agent calls.

`SequentialHandoffCoordinator` is an opt-in layer above that coordinator. The
host declares an ordered list of stages and the responsible worker for each
stage. A stage may return ordinary text or a strict `HandoffPayload`. Structured
payloads have bounded summaries and arrays of facts, uncertainties, and
recommendations; unknown fields, malformed JSON, invalid values, and oversized
outputs fail the stage before another provider runs. The next stage receives
the latest validated payload plus the original request. This makes the Adaptive Multi-Model
flow explicit and testable without giving agents permission to delegate or
call one another recursively.

`DependencyHandoffCoordinator` adds an explicit directed acyclic graph. Each
stage names its required stage IDs through `depends_on`; declaration order does
not control execution. Unknown dependencies and cycles are rejected before any
provider call, then ready stages run in deterministic topological order. A
failed stage blocks only its dependent descendants, while unrelated ready
stages may continue. Dependency payloads are combined into bounded untrusted
JSON; an oversized dependency input fails before the target worker runs. Ready
stages are still executed sequentially in this phase, not in parallel.

`CapabilityRouter` chooses between predeclared FAST, CONTEXT, REASONING, and
FULL dependency workflows. Its bounded English and Uzbek signal rules are
explicit and offline-testable. It cannot invent a provider, silently delegate,
or make an extra LLM request. `AdaptiveMultiModelClient` requires all four routes to
be configured and exposes the latest decision and workflow result.

`create_provider_worker()` binds one named worker to a configured provider and
creates its isolated `AgentRunner`. Each provider reads its own API key and
model variables from `.env`; each worker may also receive a different explicit
tool executor:

```python
from ai_sdk.agents import (
    AgentTask,
    MultiAgentCoordinator,
    create_provider_worker,
)

researcher = create_provider_worker(
    "researcher",
    "Collect verified facts",
    "gemini",
    research_tool_executor,
)
writer = create_provider_worker(
    "writer",
    "Write a concise final answer",
    "openai",
)

coordinator = MultiAgentCoordinator([researcher, writer])
result = coordinator.run(
    [
        AgentTask("research", "researcher", "Research Python"),
        AgentTask("draft", "writer", "Write the draft"),
    ]
)
```

Provider selection is explicit per worker. There is no automatic fallback,
load balancing, or hidden cross-provider routing.

`MCPClient` provides the local contract for MCP `2026-07-28`. It has an
explicit `new -> opening -> open -> failed/closed` transport lifecycle, sends
protocol version, client identity, and client capabilities with every protocol
request, and contains transport exception messages. `server/discover` is
optional and never runs automatically. If the host calls it, the client checks
the requested protocol version and advertised tool/resource capabilities.
`tools/list` and `resources/list` return exactly one ordered page at a time,
including the next cursor and server cache hints; the host decides whether to
request another page or cache a result. `tools/call` preserves text, non-text,
structured, and remote error results. `resources/read` preserves multiple text
or binary content items plus cache hints.

`tools/call` and `resources/read` may instead return `MCPContinuation` when the
server responds with `resultType: input_required`. The continuation exposes
the requested elicitation, sampling, or roots operations but keeps the
server's `requestState` opaque. `continue_request()` requires responses whose
keys exactly match the outstanding requests, echoes the state unchanged, and
uses a new JSON-RPC request ID. Each continuation is consumed once;
`cancel_continuation()` discards it locally. A logical request defaults to at
most ten input-required rounds, and every network retry uses the client's
normal request timeout. No server request is answered automatically.

`StreamableHTTPTransport` is the dependency-free concrete network adapter. It
sends every operation as a new POST with matching protocol metadata and
`MCP-Protocol-Version`, `Mcp-Method`, and, where required, `Mcp-Name` headers.
It supports ordinary JSON responses and request-scoped SSE, enforces matching
JSON-RPC IDs, bounds response size, rejects redirects, and converts valid
remote JSON-RPC errors into typed `MCPRemoteError` values. An optional
authorization callback is invoked separately for every request so the host can
provide a current token without storing it in the transport.

Tools may mark compatible primitive input properties with `x-mcp-header`.
Those values are mirrored as `Mcp-Param-*` headers when the tool is called;
non-ASCII values use the protocol's base64 sentinel form. Tools with invalid
or unreachable header annotations are excluded before approval and
registration.

```python
transport = StreamableHTTPTransport(
    "https://example.com/mcp",
    authorization_provider=lambda: get_current_authorization(),
)
client = MCPClient(
    transport,
    client_info=MCPImplementation("my-client", "1.0"),
)

with client:
    tools = client.list_tools()
    outcome = client.call_tool("confirmable_action", {"value": 1})
    if isinstance(outcome, MCPContinuation):
        outcome = client.continue_request(
            outcome,
            {"confirm": {"action": "accept", "content": {}}},
        )
```

`MCPToolAdapter` never discovers or registers tools automatically. The host
must pass exact approved names. Only schemas that fit the local tool layer's
string, integer, number, and boolean parameter subset are accepted; unsupported
constraints, nested inputs, incompatible names, missing approvals, and
registry collisions are rejected before any registry change. Approved remote
errors remain explicit tool errors, while transport exceptions still expose
only their exception type. The automatic local tool adapter does not collect
interactive input: it cancels an unexpected continuation and returns a safe
tool error so the host can use the manual client API instead.

This phase deliberately has no automatic OAuth discovery or token refresh,
automatic pagination/discovery, legacy MCP protocol fallback, subscriptions,
or automatic multi round-trip fulfillment. The authorization callback lets
the host own credential refresh without exposing secrets to the SDK.

## Tracing

Tracing is opt-in. Create one collector and tracer, then pass the same tracer
to the top-level manager or lower-level component that should be observed:

```python
from ai_sdk.observability import InMemoryTraceCollector, Tracer

collector = InMemoryTraceCollector(max_records=1000)
tracer = Tracer(collector)

manager = RAGConversationManager(
    conversation=conversation,
    prompt_builder=prompt_builder,
    client=llm_client,
    repository=repository,
    chunker=chunker,
    retriever=retriever,
    tracer=tracer,
)
response = manager.send_message("question")

for record in collector.records():
    print(record.to_dict())
```

The built-in instrumentation records operation names, parent-child
relationships, elapsed time, outcome status, safe counts, and exception class
names. It does not record exception messages or application content. Attribute
names associated with prompts, content, credentials, tokens, tool arguments,
and MCP state are redacted; custom instrumentation must still use deliberate,
non-sensitive attribute names. Records stay in the bounded in-memory collector
unless the host supplies another `TraceCollector` implementation. This phase
does not yet include remote trace propagation, sampling, an OpenTelemetry
exporter, metrics, logs, or persistent trace reports.

## Evaluation

The general evaluation harness is offline and provider-neutral. Each evaluator
returns a normalized score from zero to one and exposes the threshold that
turns that score into a pass or failure:

```python
from ai_sdk.evaluation import (
    EvalCase,
    EvaluationRunner,
    ExactMatchEvaluator,
)

cases = [
    EvalCase(
        id="python-version",
        input_text="Which Python version is required?",
        expected_output="Python 3.10 or newer.",
    ),
]
runner = EvaluationRunner(
    [ExactMatchEvaluator(strip=True)],
    minimum_pass_rate=1.0,
)
report = runner.evaluate(cases, answer_question)

if not report.passed:
    print(report.failed_case_ids)
```

`EvaluationRunner` calls the target once per case and runs evaluators in their
declared order. A target or evaluator exception fails only its current case;
the report stores the exception class, never its message or the generated
output. Missing scores on an errored case count as zero in aggregate means.
Custom evaluators implement `name`, `threshold`, and
`evaluate(case, actual_output)`. `ExactMatchEvaluator` is the only built-in
general text evaluator in this phase; retrieval keeps its purpose-built Hit
Rate, Recall, and MRR evaluators. LLM judges, dataset persistence, a dedicated
general eval CLI, provider cost/latency summaries, and general regression gates
are intentionally left for later phases.

The route-specific evaluation suite is fully offline and can be run directly:

```bash
.venv/bin/python app/evaluate_routing.py
```

Its built-in balanced benchmark contains 24 English and Uzbek cases across
FAST, CONTEXT, REASONING, and FULL. The report includes overall and per-route
accuracy, expected and selected model-request counts, the difference from an
always-FULL baseline, and local routing latency. At the current rules, the
benchmark selects all 24 expected routes, estimates 48 model requests instead
of the 72-request FULL baseline, and therefore estimates 24 saved requests.
These counts are workflow estimates and the latency measures only local routing;
no provider is called, so the suite does not claim answer quality, token cost,
or end-to-end provider latency. Results retain case IDs, routes, safe signals,
counts, timings, and exception types—not input text or exception messages.

The default ingestion layer supports UTF-8 `.txt`, `.md`, `.markdown`, and
`.rst` files. Directory synchronization scans recursively in deterministic
path order, skips unsupported formats, and persists content hashes plus the
owning root directory. `/documents` shows each source path and its current
chunk count.

After each RAG answer, the CLI prints numbered sources with document ID, chunk
ID, and fused retrieval score. Local source paths are mapped to citations after
the model call and are not included in the provider prompt.

## Tests

The default command runs deterministic unit tests only and never calls an
external API:

```bash
.venv/bin/python -m pytest
```

Measure unit-test coverage when the test extras are installed:

```bash
.venv/bin/python -m pytest \
  --cov=ai_sdk \
  --cov-report=term-missing
```

Run offline integration tests explicitly:

```bash
.venv/bin/python -m pytest -m integration
```

The real Anthropic smoke test is both marked `external` and guarded by an
environment flag. It may consume API tokens and must be explicitly enabled:

```bash
RUN_ANTHROPIC_INTEGRATION=1 \
  .venv/bin/python -m pytest \
  -m 'integration and external' \
  tests/integration/test_anthropic_api.py
```

The OpenAI smoke test is guarded separately and is also disabled by default:

```bash
RUN_OPENAI_INTEGRATION=1 \
  .venv/bin/python -m pytest \
  -m 'integration and external' \
  tests/integration/test_openai_api.py
```

The Gemini smoke test has its own explicit guard and is disabled by default:

```bash
RUN_GEMINI_INTEGRATION=1 \
  .venv/bin/python -m pytest \
  -m 'integration and external' \
  tests/integration/test_gemini_api.py
```

## Deliberate next steps

Selective answer verification is intentionally outside the current plan. Once
provider credentials are available, the next gated step is the existing opt-in
smoke suite followed by a small paid end-to-end benchmark of answer quality,
token cost, latency, and observed retries. Its measurements should guide any
automatic fallback policy. The local FastAPI layer now exposes separate chat
lifecycle, progress, cancellation, readiness, and document lifecycle through
stable request contracts. Multi-user storage, authentication, public deployment, parallel
ready stages, token streaming, exporters, and persistent observability remain
optional future work.
