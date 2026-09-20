import json

from fastapi.testclient import TestClient

from ai_sdk.application.bootstrap import AssistantRuntimeResources
from ai_sdk.application.rag_response import RAGResponse
from ai_sdk.embeddings.base import BaseEmbeddingClient
from ai_sdk.memory.json_store import JSONMemoryStore
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.hybrid import HybridRetriever
from ai_sdk.retrieval.in_memory import InMemoryVectorStore
from ai_sdk.storage.json import JSONConversationRepository
from ai_sdk.web.app import _sse_events, create_app
from ai_sdk.web.service import KnowledgeAssistantService


class TinyEmbeddingClient(BaseEmbeddingClient):
    def embed(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class AnswerManager:
    client = object()

    def __init__(self, conversation_file):
        self.repository = JSONConversationRepository(conversation_file)
        self.conversation = self.repository.load()

    def send_message_with_citations(self, message):
        self.conversation.add_user(message)
        self.conversation.add_assistant(f"Answer: {message}")
        self.repository.save(self.conversation)
        return RAGResponse(f"Answer: {message}", ())


def make_client(tmp_path, monkeypatch, *, upload_limit=1024):
    monkeypatch.setenv("OPENAI_API_KEY", "never-return-this-secret")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    runtime = AssistantRuntimeResources(
        chunker=TextChunker(chunk_size=80, overlap=10),
        retriever=HybridRetriever(
            embedding_client=TinyEmbeddingClient(),
            vector_store=InMemoryVectorStore(),
        ),
        memory_store=JSONMemoryStore(tmp_path / "memory.json"),
    )
    service = KnowledgeAssistantService(
        runtime=runtime,
        upload_dir=tmp_path / "uploads",
        conversation_dir=tmp_path / "chats",
        upload_max_bytes=upload_limit,
        single_model_manager_factory=lambda _, path: AnswerManager(path),
    )
    return TestClient(create_app(service))


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ")
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        events.append((event_type, data))
    return events


def test_web_home_status_and_security_headers(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    home = client.get("/")
    status = client.get("/api/status")
    script = client.get("/static/app.js")
    stylesheet = client.get("/static/styles.css")

    assert home.status_code == 200
    assert "LyustrAI" in home.text
    assert "/static/favicon.svg?v=2" in home.text
    assert "default-src 'self'" in home.headers["content-security-policy"]
    assert status.status_code == 200
    assert status.json()["providers"][1]["ready"] is True
    assert "never-return-this-secret" not in status.text
    assert script.status_code == 200
    assert "renderMarkdown(answerBubble, data.content)" in script.text
    assert 'copyButton.textContent = "Copy"' in script.text
    assert "createThinkingIndicator()" in script.text
    assert "createConvergenceCoreIcon()" in script.text
    assert "createConvergenceCoreIcon({ waiting: true })" in script.text
    assert "startWanderingCore(core)" in script.text
    assert '"convergence-core wandering-core"' in script.text
    assert "microSaccade" in script.text
    assert "stopWanderingCore(bubble)" in script.text
    assert 'bubble.setAttribute("role", "status")' not in script.text
    assert "core-eye-blink" not in script.text
    assert "core-eye-gaze" not in script.text
    assert "createChandelierIcon" not in script.text
    assert "terminalEventReceived" in script.text
    assert "The response stream ended before a final answer arrived." in script.text
    assert "Working through the request" not in script.text
    assert 'behavior: reducedMotion ? "auto" : "smooth"' in script.text
    assert "new MutationObserver" in script.text
    assert 'request("/api/conversations")' in script.text
    assert "startNewChat" in script.text
    assert "adaptive_strategy" in script.text
    assert "brand-convergence" in home.text
    assert "chandelier-icon" not in home.text
    assert "/static/styles.css?v=15" in home.text
    assert "/static/app.js?v=15" in home.text
    assert 'data-strategy="maximum"' in home.text
    assert "Always runs Gemini → Claude → OpenAI" in script.text
    assert "/api/conversations/reset" not in script.text
    assert stylesheet.status_code == 200
    assert ".code-block" in stylesheet.text
    assert ".syntax-keyword" in stylesheet.text
    assert "@keyframes convergence-breathe" in stylesheet.text
    assert ".wandering-core" in stylesheet.text
    assert "core-eye-gaze" not in stylesheet.text
    assert "core-eye-blink" not in stylesheet.text
    assert "prefers-reduced-motion" in stylesheet.text
    assert ".pending-response .assistant-avatar { display: none; }" in stylesheet.text
    assert "@keyframes thinking-wave" not in stylesheet.text
    assert "@keyframes chandelier-glow" not in stylesheet.text
    assert "top: 20px; left: 50%" in stylesheet.text
    assert "height: calc(100dvh - 76px)" in stylesheet.text


def test_document_upload_list_delete_and_validation(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch, upload_limit=12)

    uploaded = client.post(
        "/api/documents",
        files={"file": ("Guide.md", b"RAG facts", "text/markdown")},
    )
    document = uploaded.json()["document"]

    assert uploaded.status_code == 201
    assert client.get("/api/documents").json()[0]["source"] == "Guide.md"
    assert client.delete(f"/api/documents/{document['document_id']}").status_code == 204
    assert client.delete(f"/api/documents/{document['document_id']}").status_code == 404

    unsafe = client.post(
        "/api/documents",
        files={"file": ("../private.txt", b"text", "text/plain")},
    )
    oversized = client.post(
        "/api/documents",
        files={"file": ("large.txt", b"x" * 13, "text/plain")},
    )
    assert unsafe.status_code == 400
    assert oversized.status_code == 413


def test_chat_sse_and_conversation_crud(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/chat/stream",
        json={
            "message": "Explain hybrid retrieval",
            "mode": "single",
            "provider": "openai",
        },
    )
    events = parse_sse(response.text)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-run-id"].startswith("run_")
    conversation_id = response.headers["x-conversation-id"]
    assert conversation_id.startswith("chat_")
    assert [event[0] for event in events] == ["run", "citations", "answer"]
    assert events[0][1]["conversation_id"] == conversation_id
    assert events[-1][1]["content"] == "Answer: Explain hybrid retrieval"

    listed = client.get("/api/conversations")
    detail = client.get(f"/api/conversations/{conversation_id}")
    renamed = client.patch(
        f"/api/conversations/{conversation_id}",
        json={"title": "Retrieval notes"},
    )

    assert listed.json()[0]["conversation_id"] == conversation_id
    assert listed.json()[0]["message_count"] == 2
    assert [message["role"] for message in detail.json()["messages"]] == [
        "user",
        "assistant",
    ]
    assert renamed.json()["title"] == "Retrieval notes"

    invalid = client.post(
        "/api/chat/stream",
        json={"message": "Question", "mode": "single"},
    )
    assert invalid.status_code == 422
    assert (
        client.patch(
            f"/api/conversations/{conversation_id}",
            json={"title": " "},
        ).status_code
        == 422
    )
    assert client.get("/api/conversations/chat_ffffffffffff").status_code == 404
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_sse_disconnect_closes_the_service_event_stream():
    closed = []

    def source():
        try:
            yield {"event": "run", "data": {"run_id": "run_one"}}
            yield {"event": "answer", "data": {"content": "late"}}
        finally:
            closed.append(True)

    stream = _sse_events(source())

    assert next(stream).startswith("event: run")
    stream.close()

    assert closed == [True]
