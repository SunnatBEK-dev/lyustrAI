.PHONY: audit build coverage eval install integration lint release-check serve test typecheck

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e '.[documents,embeddings,web,dev]'

test:
	$(PYTHON) -m pytest

integration:
	$(PYTHON) -m pytest -m 'integration and not external'

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src/ai_sdk app

coverage:
	$(PYTHON) -m pytest -m 'not external' --cov=ai_sdk --cov-report=term-missing

build:
	$(PYTHON) -m build

eval:
	$(PYTHON) app/evaluate_knowledge.py

audit:
	$(PYTHON) app/audit_repository.py

serve:
	lyustrai

release-check: audit lint typecheck coverage build
