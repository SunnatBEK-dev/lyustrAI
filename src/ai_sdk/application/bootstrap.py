from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ai_sdk.agents import (
    CapabilityRouter,
    DependencyHandoffCoordinator,
    HandoffOutputFormat,
    HandoffStage,
    MultiAgentCoordinator,
    MultiModelRoute,
    WorkflowProgressEvent,
    create_provider_worker,
)
from ai_sdk.application.rag_manager import RAGConversationManager
from ai_sdk.config import (
    ADAPTIVE_CONVERSATION_FILE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTEXT_SUMMARY_TOKEN_BUDGET,
    CONTEXT_TOKEN_BUDGET,
    DEFAULT_CONVERSATION_FILE,
    EMBEDDING_MODEL,
    LLM_RETRY_INITIAL_DELAY,
    LLM_RETRY_MAX_ATTEMPTS,
    LLM_RETRY_MAX_DELAY,
    MEMORY_FILE,
    MEMORY_RETRIEVAL_K,
    RETRIEVAL_K,
    SINGLE_MODEL_CONVERSATION_DIR,
    VECTOR_STORE_FILE,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.context.summary import ExtractiveConversationSummarizer
from ai_sdk.context.window import SlidingContextWindow
from ai_sdk.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingClient,
)
from ai_sdk.llm.adaptive_metrics import InMemoryAdaptiveMetrics
from ai_sdk.llm.adaptive_multi_model import (
    AdaptiveMultiModelClient,
    MultiModelWorkflowClient,
)
from ai_sdk.llm.base import BaseLLMClient
from ai_sdk.llm.factory import create_llm_client, normalize_llm_provider
from ai_sdk.llm.retry import RetryPolicy
from ai_sdk.memory.json_store import JSONMemoryStore
from ai_sdk.retrieval.chunker import TextChunker
from ai_sdk.retrieval.hybrid import HybridRetriever
from ai_sdk.retrieval.json_store import JSONVectorStore
from ai_sdk.storage.json import JSONConversationRepository


@dataclass(frozen=True)
class AssistantRuntimeResources:
    """Shared retrieval runtime for one local application process."""

    chunker: TextChunker
    retriever: HybridRetriever
    memory_store: JSONMemoryStore


def create_assistant_runtime(
    *,
    vector_store_file: Path = VECTOR_STORE_FILE,
    memory_file: Path = MEMORY_FILE,
) -> AssistantRuntimeResources:
    embedding_client = SentenceTransformerEmbeddingClient(model_name=EMBEDDING_MODEL)
    return AssistantRuntimeResources(
        chunker=TextChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        ),
        retriever=HybridRetriever(
            embedding_client=embedding_client,
            vector_store=JSONVectorStore(vector_store_file),
        ),
        memory_store=JSONMemoryStore(memory_file),
    )


def create_rag_manager(
    *,
    provider: str | None = None,
    conversation_file: Path = DEFAULT_CONVERSATION_FILE,
    client: BaseLLMClient | None = None,
    runtime: AssistantRuntimeResources | None = None,
) -> RAGConversationManager:
    if provider is not None and client is not None:
        raise ValueError("Configure either a provider or an explicit client.")
    if client is not None and not isinstance(client, BaseLLMClient):
        raise TypeError("Explicit client must be a BaseLLMClient.")

    resolved_runtime = runtime or create_assistant_runtime()
    repository = JSONConversationRepository(conversation_file)
    conversation = repository.load()
    return RAGConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(
            conversation,
            context_window=SlidingContextWindow(max_tokens=CONTEXT_TOKEN_BUDGET),
            summary_memory=ExtractiveConversationSummarizer(
                max_tokens=CONTEXT_SUMMARY_TOKEN_BUDGET
            ),
        ),
        client=create_llm_client(provider) if client is None else client,
        repository=repository,
        chunker=resolved_runtime.chunker,
        retriever=resolved_runtime.retriever,
        retrieval_k=RETRIEVAL_K,
        memory_store=resolved_runtime.memory_store,
        memory_retrieval_k=MEMORY_RETRIEVAL_K,
    )


def create_single_model_manager(
    provider: str,
    *,
    conversation_file: Path | None = None,
    runtime: AssistantRuntimeResources | None = None,
) -> RAGConversationManager:
    normalized = normalize_llm_provider(provider)
    conversation_path = (
        conversation_file or SINGLE_MODEL_CONVERSATION_DIR / f"{normalized}.json"
    )
    return create_rag_manager(
        provider=normalized,
        conversation_file=conversation_path,
        runtime=runtime,
    )


def create_adaptive_multi_model_manager(
    *,
    conversation_file: Path = ADAPTIVE_CONVERSATION_FILE,
    progress_handler: Callable[[WorkflowProgressEvent], None] | None = None,
    runtime: AssistantRuntimeResources | None = None,
    metrics: InMemoryAdaptiveMetrics | None = None,
    route_override: MultiModelRoute | None = None,
) -> RAGConversationManager:
    retry_policy = RetryPolicy(
        max_attempts=LLM_RETRY_MAX_ATTEMPTS,
        initial_delay_seconds=LLM_RETRY_INITIAL_DELAY,
        max_delay_seconds=LLM_RETRY_MAX_DELAY,
    )
    context_worker = create_provider_worker(
        "context",
        "Extract relevant facts, constraints, and missing evidence",
        "gemini",
        retry_policy=retry_policy,
    )
    reasoning_worker = create_provider_worker(
        "reasoner",
        "Perform careful analysis using the available evidence",
        "anthropic",
        retry_policy=retry_policy,
    )
    synthesis_worker = create_provider_worker(
        "synthesizer",
        "Produce one clear final answer for the user",
        "openai",
        retry_policy=retry_policy,
    )
    coordinator = MultiAgentCoordinator(
        [context_worker, reasoning_worker, synthesis_worker]
    )

    def context_stage() -> HandoffStage:
        return HandoffStage(
            "context",
            "context",
            "Extract facts, constraints, and uncertainties. "
            "Do not invent missing evidence.",
            output_format=HandoffOutputFormat.STRUCTURED,
        )

    def reasoning_stage(*dependencies: str) -> HandoffStage:
        return HandoffStage(
            "reasoning",
            "reasoner",
            "Analyze the request and available evidence. "
            "Identify contradictions and a sound solution. "
            "Carry forward every useful verified fact.",
            output_format=HandoffOutputFormat.STRUCTURED,
            depends_on=dependencies,
        )

    def final_stage(*dependencies: str) -> HandoffStage:
        return HandoffStage(
            "final",
            "synthesizer",
            "Create the final answer from verified useful points. "
            "Do not mention internal stages unless needed.",
            depends_on=dependencies,
        )

    def create_workflow(stages: list[HandoffStage]) -> MultiModelWorkflowClient:
        return MultiModelWorkflowClient(
            DependencyHandoffCoordinator(coordinator, stages)
        )

    adaptive_client = AdaptiveMultiModelClient(
        CapabilityRouter(forced_route=route_override),
        {
            MultiModelRoute.FAST: create_workflow([final_stage()]),
            MultiModelRoute.CONTEXT: create_workflow(
                [context_stage(), final_stage("context")]
            ),
            MultiModelRoute.REASONING: create_workflow(
                [reasoning_stage(), final_stage("reasoning")]
            ),
            MultiModelRoute.FULL: create_workflow(
                [
                    context_stage(),
                    reasoning_stage("context"),
                    final_stage("context", "reasoning"),
                ]
            ),
        },
        metrics=metrics,
        progress_handler=progress_handler,
    )
    return create_rag_manager(
        conversation_file=conversation_file,
        client=adaptive_client,
        runtime=runtime,
    )
