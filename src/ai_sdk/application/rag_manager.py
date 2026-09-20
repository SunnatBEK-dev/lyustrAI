from ai_sdk.application.conversation_manager import (
    ConversationManager,
)
from ai_sdk.application.rag_response import (
    Citation,
    RAGResponse,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.types import LLMMessage
from ai_sdk.memory.base import BaseMemoryStore
from ai_sdk.observability import (
    TraceCategory,
    Tracer,
    trace_span,
)
from ai_sdk.retrieval.catalog import IndexedDocument
from ai_sdk.retrieval.chunk import Chunk
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.document import Document
from ai_sdk.retrieval.retriever import (
    SemanticRetriever,
)
from ai_sdk.storage.base import ConversationRepository
from ai_sdk.tools.executor import ToolExecutor


class RAGConversationManager(ConversationManager):
    """Conversation workflow augmented with document retrieval."""

    def __init__(
        self,
        conversation: Conversation,
        prompt_builder: PromptBuilder,
        client: BaseLLMClient,
        repository: ConversationRepository,
        chunker: TextChunker,
        retriever: SemanticRetriever,
        retrieval_k: int = 3,
        memory_store: BaseMemoryStore | None = None,
        memory_retrieval_k: int = 3,
        tool_executor: ToolExecutor | None = None,
        max_tool_rounds: int = 8,
        tracer: Tracer | None = None,
    ) -> None:
        if retrieval_k <= 0:
            raise ValueError("Retrieval top-k must be greater than zero.")

        super().__init__(
            conversation=conversation,
            prompt_builder=prompt_builder,
            client=client,
            repository=repository,
            memory_store=memory_store,
            memory_retrieval_k=memory_retrieval_k,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
            tracer=tracer,
        )
        self.chunker = chunker
        self.retriever = retriever
        self.retrieval_k = retrieval_k
        self._last_citations: tuple[
            Citation,
            ...,
        ] = ()

    def index_document(
        self,
        document: Document,
    ) -> list[Chunk]:
        chunks = self.chunker.split(document)
        self.retriever.index_document(
            document.id,
            chunks,
        )
        return chunks

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        return self.retriever.delete_document(document_id)

    def list_documents(self) -> list[str]:
        return self.retriever.list_documents()

    def document_catalog(
        self,
    ) -> list[IndexedDocument]:
        return self.retriever.document_catalog()

    @property
    def last_citations(self) -> tuple[Citation, ...]:
        return self._last_citations

    def send_message_with_citations(
        self,
        text: str,
    ) -> RAGResponse:
        content = self.send_message(text)
        return RAGResponse(
            content=content,
            citations=self.last_citations,
        )

    def _build_messages(
        self,
        text: str,
    ) -> list[LLMMessage]:
        self._last_citations = ()
        memory_results = self._recall_memories(text)

        if not self.list_documents():
            return self.prompt_builder.build_messages(memory_results=memory_results)

        with trace_span(
            self.tracer,
            "retrieval.search",
            TraceCategory.RETRIEVAL,
            {"retrieval.k": self.retrieval_k},
        ) as span:
            retrieval_results = self.retriever.retrieve(
                query=text,
                k=self.retrieval_k,
            )
            if span is not None:
                span.set_attribute(
                    "retrieval.result_count",
                    len(retrieval_results),
                )

        messages = self.prompt_builder.build_messages(
            retrieval_results=retrieval_results,
            memory_results=memory_results,
        )
        self._last_citations = tuple(
            Citation.from_search_result(
                position,
                result,
            )
            for position, result in enumerate(
                retrieval_results,
                start=1,
            )
        )

        return messages
