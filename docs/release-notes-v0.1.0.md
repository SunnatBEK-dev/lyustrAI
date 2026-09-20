# LyustrAI v0.1.0

The first LyustrAI release turns the underlying provider-neutral AI SDK into a
local adaptive multi-model system. It coordinates Gemini, Claude (Anthropic),
and OpenAI through evidence extraction, contradiction-aware analysis, and final
synthesis.

## Highlights

- Coordinate Gemini, Claude (Anthropic), and OpenAI through validated,
  dependency-aware handoffs instead of relying on one model response.
- Use Adaptive `Auto` to select an efficient route or `Maximum` to force the
  complete evidence extraction → contradiction-aware analysis → final synthesis
  workflow.
- Inspect the supporting evidence through local source cards and page-aware PDF
  citations when relevant documents are indexed.
- Upload PDF, Markdown, or TXT files and retrieve relevant context through
  hybrid semantic + BM25 search.
- Inspect route and stage progress over SSE, cancel safely at workflow
  boundaries, and keep API keys out of responses and metrics.
- Reproduce a 27-case hybrid-retrieval benchmark: Hit Rate@3 1.000, Recall@3
  1.000, and MRR 0.963.
- Reproduce a 24/24 routing benchmark with 48 estimated provider requests
  versus a 72-request always-FULL baseline.
- Run 808 offline tests under a 95% coverage gate; the measured v0.1.0 branch
  coverage is 96.11% on Python 3.14.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[embeddings,documents,web]'
cp .env.example .env
lyustrai
```

This release is a local, single-user BYOK demonstration. It intentionally does
not claim hosted-production readiness, general-domain answer quality, or
multi-tenant security.
