# Career and interview kit

## CV bullets

- Built LyustrAI, an adaptive multi-model system that coordinates Gemini,
  Claude (Anthropic), and OpenAI through evidence extraction,
  contradiction-aware analysis, and final synthesis; `Maximum` forces the full
  three-stage workflow while citations keep indexed evidence inspectable.
- Designed an explainable deterministic router that matched 24/24 internal
  route labels and estimated 48 provider requests versus a 72-request
  always-FULL baseline; clearly separated routing efficiency from
  answer-quality claims.
- Engineered a provider-neutral Python RAG and agent SDK behind explicit
  contracts, with hybrid retrieval, page-aware citations, MCP, memory, tool
  calling, and isolated saved conversations.
- Established a deterministic quality system with 784 unit tests, 24 offline
  integration tests, 96.11% branch coverage on Python 3.14, and opt-in paid
  provider smoke tests.

Re-run CI before sending a CV. Do not round metrics up or call the local demo
production-proven.

## Interview story 1 — hybrid retrieval

**Problem:** semantic search can miss exact identifiers and rare technical
terms. **Decision:** add BM25 and weighted reciprocal-rank fusion behind the
same retriever contract. **Evidence:** the committed 27-case corpus keeps Hit
Rate@3 and Recall@3 at 1.000 while improving MRR from 0.944 to 0.963.
**Tradeoff:** the corpus is project-specific and does not prove general-domain
quality.

## Interview story 2 — adaptive routing

**Problem:** always running three providers wastes requests on simple prompts.
**Decision:** use explainable bounded signals instead of another LLM router.
**Evidence:** 24/24 internal decisions and 24 estimated requests saved against
an always-FULL baseline. **Tradeoff:** deterministic rules can misclassify
ambiguous prompts and must be maintained with failure examples.

## Interview story 3 — reliability

**Problem:** hidden SDK retries and unsafe cancellation make request counts and
state transitions hard to reason about. **Decision:** disable provider retries,
classify transient errors in one policy, emit content-free progress, and cancel
only at safe boundaries. **Evidence:** deterministic retry, rollback, progress,
and cancellation tests. **Tradeoff:** in-flight blocking API calls are not
forcibly interrupted.

## GitHub publishing checklist

- Use `SunnatBEK-dev/lyustrAI` as the public repository.
- Enable secret scanning, push protection, Dependabot alerts, and Actions.
- Add topics: `python`, `llm`, `rag`, `agents`, `mcp`, `fastapi`, `ai-evaluation`.
- Upload the social preview from `docs/assets/social-preview.png` in repository
  settings.
- Push only after `make release-check` passes.
- Create release `v0.1.0`, then pin the repository on the GitHub profile.
