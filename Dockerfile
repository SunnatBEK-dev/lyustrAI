FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8000 \
    HF_HOME=/home/assistant/.cache/huggingface

WORKDIR /app

RUN groupadd --gid 10001 assistant \
    && useradd --uid 10001 --gid assistant --create-home assistant

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        'torch==2.9.1+cpu' \
    && python -m pip install --no-cache-dir '.[documents,embeddings,web]'

COPY app ./app
COPY docs ./docs
COPY evals ./evals
RUN mkdir -p /app/data "${HF_HOME}" \
    && chown -R assistant:assistant /app /home/assistant

USER assistant

EXPOSE 8000
VOLUME ["/app/data", "/home/assistant/.cache/huggingface"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=3)" || exit 1

CMD ["lyustrai"]
