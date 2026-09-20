# Evaluation and test strategy

Evaluation is split by responsibility. Retrieval quality uses Hit Rate@k,
Recall@k, and Mean Reciprocal Rank. A comparator runs the same labeled dataset
against semantic and hybrid retrievers and reports metric deltas. Stable
document-level labels are supported in addition to exact chunk labels so the
portfolio dataset remains valid across different clone paths.

The knowledge benchmark contains at least 20 questions about this project's
architecture, retrieval, routing, reliability, evaluation, and security. It
uses the project's own English technical documents as the corpus. The release
gate requires Hit Rate@3 of at least 0.90, MRR of at least 0.80, and no hybrid
regression against the semantic baseline.

The route evaluation suite is independent of retrieval. It checks 24 expected
decisions across FAST, CONTEXT, REASONING, and FULL, then reports estimated
provider request counts and local routing latency. No provider is called, so
the suite does not claim end-to-end answer quality, token cost, or API latency.

Unit tests are deterministic and external-call free. Offline integration tests
exercise multiple real components. Anthropic, OpenAI, and Gemini smoke tests
are marked `external` and remain disabled unless their individual environment
flags are set to `1`; this prevents an ordinary CI run from spending API credit.

The project does not use an uncalibrated LLM judge as a headline metric.
Generation correctness is a disclosed limitation until a human-labeled answer
dataset and judge calibration protocol are available.
