# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses semantic versioning.

## [0.1.0] - 2026-09-01

### Added

- Responsive FastAPI knowledge-assistant UI with Single Model and Adaptive
  Multi-Model modes.
- Separate local chats with lazy creation, restore, rename, and permanent
  delete controls.
- Adaptive effort control with efficient Auto routing and a user-forced
  Maximum Gemini → Claude → OpenAI workflow.
- SSE route, stage, answer, citation, cancellation, and safe error events.
- Text-based PDF ingestion with page-aware chunks and citations.
- A 27-case retrieval benchmark comparing semantic and hybrid search.
- Python 3.10, 3.12, and 3.14 CI, CodeQL, Dependabot, mypy, Ruff, coverage, and
  package-build gates.
- Non-root Docker image and one-command Compose configuration.
- CPU-only Docker inference dependencies to avoid unnecessary CUDA packages.
- A repeatable audit for tracked runtime data and recognizable secrets.
- English architecture, reliability, evaluation, security, demo, and career
  documentation.

### Changed

- Renamed the product to LyustrAI and the orchestrated
  workflow to Adaptive Multi-Model without legacy aliases.
- Moved the detailed SDK capability inventory out of the README.
- Isolated all runtime documents, conversations, and vector data from Git.

[0.1.0]: https://github.com/SunnatBEK-dev/lyustrAI/releases/tag/v0.1.0
