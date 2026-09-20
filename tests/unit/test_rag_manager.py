import pytest

from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.observability import (
    InMemoryTraceCollector,
    TraceCategory,
    Tracer,
)
from ai_sdk.retrieval.catalog import IndexedDocument
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.search import SearchResult
from ai_sdk.tools import ToolExecutor, ToolRegistry


class FakeClient:
    def __init__(self):
        self.received_messages = None

    def ask(self, messages):
        self.received_messages = messages
        return "Grounded answer"

    def stream(self, messages):
        self.received_messages = messages
        yield "Grounded answer"


class FakeToolClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.tool_request = None

    def ask_with_tools(
        self,
        messages,
        executor,
        *,
        max_tool_rounds,
    ):
        self.received_messages = messages
        self.tool_request = (executor, max_tool_rounds)
        return "Tool-grounded answer"


class FakeRepository:
    def __init__(self):
        self.saved = []

    def save(self, conversation):
        self.saved.append(conversation.history().copy())

    def load(self):
        return Conversation()


class FakeChunker:
    def __init__(self, chunks):
        self.chunks = chunks
        self.documents = []

    def split(self, document):
        self.documents.append(document)
        return self.chunks


class FakeRetriever:
    def __init__(
        self,
        results=None,
        error=None,
        documents=None,
    ):
        self.results = results or []
        self.error = error
        self.documents = ["doc_rag"] if documents is None else list(documents)
        self.indexed = []
        self.deleted_documents = []
        self.queries = []

    def index_document(self, document_id, chunks):
        self.indexed.append(
            (
                document_id,
                list(chunks),
            )
        )

    def delete_document(self, document_id):
        self.deleted_documents.append(document_id)
        return 2

    def list_documents(self):
        return self.documents.copy()

    def document_catalog(self):
        return [
            IndexedDocument(
                document_id=document_id,
                source=f"{document_id}.txt",
                chunk_count=1,
            )
            for document_id in self.documents
        ]

    def retrieve(self, query, k=5):
        self.queries.append((query, k))

        if self.error:
            raise self.error

        return self.results


def make_chunk() -> Chunk:
    return Chunk(
        id="chunk_context",
        document_id="doc_rag",
        content="Retrieved knowledge",
        index=0,
        metadata={"source": "guide.txt"},
    )


def create_rag_manager(
    *,
    retriever=None,
    chunker=None,
    retrieval_k=2,
    client=None,
    tool_executor=None,
    tracer=None,
):
    conversation = Conversation()
    client = client or FakeClient()
    repository = FakeRepository()
    chunker = chunker or FakeChunker([make_chunk()])
    retriever = retriever or FakeRetriever([SearchResult(make_chunk(), 0.9)])
    manager = RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=repository,
        chunker=chunker,
        retriever=retriever,
        retrieval_k=retrieval_k,
        tool_executor=tool_executor,
        tracer=tracer,
    )
    return (
        manager,
        conversation,
        client,
        repository,
        chunker,
        retriever,
    )


def test_index_document_chunks_and_indexes_content():
    chunk = make_chunk()
    chunker = FakeChunker([chunk])
    retriever = FakeRetriever()
    manager, _, _, _, _, _ = create_rag_manager(
        chunker=chunker,
        retriever=retriever,
    )
    document = Document(
        id="doc_rag",
        content="Document content",
    )

    chunks = manager.index_document(document)

    assert chunks == [chunk]
    assert chunker.documents == [document]
    assert retriever.indexed == [
        (
            document.id,
            [chunk],
        )
    ]


def test_delete_document_delegates_to_retriever():
    retriever = FakeRetriever()
    manager, _, _, _, _, _ = create_rag_manager(retriever=retriever)

    deleted_count = manager.delete_document("doc_rag")

    assert deleted_count == 2
    assert retriever.deleted_documents == ["doc_rag"]


def test_list_documents_delegates_to_retriever():
    retriever = FakeRetriever(documents=["doc_b", "doc_a"])
    manager, _, _, _, _, _ = create_rag_manager(retriever=retriever)

    assert manager.list_documents() == [
        "doc_b",
        "doc_a",
    ]
    assert manager.document_catalog() == [
        IndexedDocument(
            document_id="doc_b",
            source="doc_b.txt",
            chunk_count=1,
        ),
        IndexedDocument(
            document_id="doc_a",
            source="doc_a.txt",
            chunk_count=1,
        ),
    ]


def test_send_message_retrieves_context_before_llm_call():
    manager, conversation, client, repository, _, retriever = create_rag_manager()

    response = manager.send_message("User question")

    assert response == "Grounded answer"
    assert retriever.queries == [("User question", 2)]
    assert "Retrieved knowledge" in (client.received_messages[-1]["content"])
    assert "User question" in (client.received_messages[-1]["content"])
    assert [message.content for message in conversation.history()] == [
        "User question",
        "Grounded answer",
    ]
    assert len(repository.saved) == 1
    assert len(manager.last_citations) == 1
    assert manager.last_citations[0].position == 1
    assert manager.last_citations[0].source == ("guide.txt")
    assert manager.last_citations[0].score == pytest.approx(0.9)


def test_rag_message_traces_workflow_retrieval_and_llm_without_text():
    collector = InMemoryTraceCollector()
    tracer = Tracer(collector)
    manager, _, _, _, _, _ = create_rag_manager(tracer=tracer)

    response = manager.send_message("private user question")

    records = collector.records()
    root = next(record for record in records if record.name == "conversation.send")
    retrieval = next(
        record for record in records if record.category is TraceCategory.RETRIEVAL
    )
    llm = next(record for record in records if record.name == "llm.generate")
    assert response == "Grounded answer"
    assert retrieval.parent_span_id == root.span_id
    assert llm.parent_span_id == root.span_id
    assert retrieval.attributes == {
        "retrieval.k": 2,
        "retrieval.result_count": 1,
    }
    assert llm.attributes["llm.message_count"] > 0
    assert llm.attributes["llm.response_char_count"] == len(response)
    assert "private user question" not in str([record.to_dict() for record in records])


def test_send_message_with_citations_returns_structured_response():
    manager, _, _, _, _, _ = create_rag_manager()

    response = manager.send_message_with_citations("User question")

    assert response.content == "Grounded answer"
    assert response.citations == manager.last_citations
    assert response.citations[0].chunk_id == ("chunk_context")


def test_send_message_without_documents_skips_retrieval():
    retriever = FakeRetriever(documents=[])
    manager, _, client, _, _, _ = create_rag_manager(retriever=retriever)

    response = manager.send_message("Plain question")

    assert response == "Grounded answer"
    assert retriever.queries == []
    assert client.received_messages[-1]["content"] == ("Plain question")
    assert manager.last_citations == ()


def test_retrieval_failure_rolls_back_user_message():
    retriever = FakeRetriever(error=RuntimeError("retrieval failed"))
    manager, conversation, client, repository, _, _ = create_rag_manager(
        retriever=retriever
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        manager.send_message("Question")

    assert conversation.is_empty()
    assert client.received_messages is None
    assert repository.saved == []
    assert manager.last_citations == ()


def test_rag_manager_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="greater than zero"):
        create_rag_manager(retrieval_k=0)


def test_rag_manager_combines_retrieval_context_and_tools():
    client = FakeToolClient()
    executor = ToolExecutor(ToolRegistry())
    manager, _, _, repository, _, retriever = create_rag_manager(
        client=client,
        tool_executor=executor,
    )

    response = manager.send_message("Tool question")

    assert response == "Tool-grounded answer"
    assert retriever.queries == [("Tool question", 2)]
    assert "Retrieved knowledge" in (client.received_messages[-1]["content"])
    assert client.tool_request == (executor, 8)
    assert len(repository.saved) == 1
