# Contributing

Thanks for considering a contribution to LyustrAI.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[embeddings,documents,web,dev]'
```

Copy `.env.example` to `.env` only when you want to run a live provider. The
default test suite does not call paid APIs.

## Quality checks

```bash
python app/audit_repository.py
ruff check .
ruff format --check .
mypy src/ai_sdk app
pytest -m 'not external' --cov=ai_sdk --cov-report=term-missing
pytest -m 'integration and not external'
python -m build
```

Keep pull requests focused, include tests for behavior changes, and avoid
committing credentials or files from `data/`.
