from collections.abc import Callable, Sequence
from pathlib import Path

from ai_sdk.agents import (
    WorkflowProgressEvent,
    WorkflowProgressStatus,
)
from ai_sdk.application.bootstrap import (
    create_adaptive_multi_model_manager,
    create_single_model_manager,
)
from ai_sdk.application.modes import AssistantMode
from ai_sdk.application.rag_manager import (
    RAGConversationManager,
)
from ai_sdk.application.rag_response import Citation
from ai_sdk.core.conversation import Conversation
from ai_sdk.ingestion import (
    DirectorySynchronizer,
    DocumentIngestor,
    TextDocumentLoader,
    create_default_ingestor,
)
from ai_sdk.llm.adaptive_multi_model import (
    AdaptiveMultiModelClient,
)
from ai_sdk.retrieval.document import Document
from ai_sdk.security import redact_secrets


def print_history(conversation: Conversation) -> None:
    if conversation.is_empty():
        print("\nHistory is empty.\n")
        return

    print()

    for message in conversation.history():
        print(f"{message.role.capitalize()}: {message.content}")
        print(f"ID: {message.id}")
        print()


def print_documents(
    manager: RAGConversationManager,
) -> None:
    documents = manager.document_catalog()

    if not documents:
        print("\nDocument index is empty.\n")
        return

    print("\nIndexed documents:")

    for document in documents:
        print(
            f"- {document.document_id} | "
            f"chunks={document.chunk_count} | "
            f"source={document.source}"
        )

    print()


def print_citations(
    citations: Sequence[Citation],
) -> None:
    if not citations:
        print()
        return

    print("Sources:")

    for citation in citations:
        page = "" if citation.page is None else f" | page={citation.page}"
        print(
            f"[{citation.position}] {citation.source} | "
            f"document={citation.document_id} | "
            f"chunk={citation.chunk_id} | "
            f"score={citation.score:.3f}{page}"
        )

    print()


def print_memories(
    manager: RAGConversationManager,
) -> None:
    memories = manager.list_memories()

    if not memories:
        print("\nLong-term memory is empty.\n")
        return

    print("\nLong-term memories:")

    for memory in memories:
        print(f"- {memory.id} | {memory.content}")

    print()


def print_help() -> None:
    print(
        "Commands:\n"
        "  /index <path>       Index a file or sync a directory\n"
        "  /documents          Show the document catalog\n"
        "  /remove <document>  Remove a document from the index\n"
        "  /remember <fact>    Store a long-term memory\n"
        "  /memories           Show long-term memories\n"
        "  /forget <memory>    Remove a long-term memory\n"
        "  /history            Show conversation history\n"
        "  /metrics              Show Adaptive Multi-Model runtime metrics\n"
        "  /save               Save conversation history\n"
        "  /clear              Clear conversation history\n"
        "  /help               Show commands\n"
        "  /exit               Exit\n"
    )


def print_adaptive_metrics(
    manager: RAGConversationManager,
) -> None:
    client = getattr(manager, "client", None)
    if not isinstance(client, AdaptiveMultiModelClient):
        print("\nRuntime metrics are available in Adaptive Multi-Model mode.\n")
        return

    report = client.metrics.report()
    if report.total_runs == 0:
        print("\nNo Adaptive Multi-Model runs have been recorded yet.\n")
        return

    routes = ", ".join(
        f"{route}={count}" for route, count in sorted(report.route_counts.items())
    )
    stages = ", ".join(
        f"{stage}={count}"
        for stage, count in sorted(report.stage_execution_counts.items())
    )
    print("\nAdaptive Multi-Model runtime metrics:")
    print(
        f"- Runs: {report.total_runs} "
        f"(successful={report.successful_runs}, "
        f"failed={report.failed_runs})"
    )
    print(f"- Routes: {routes or 'none'}")
    print(
        f"- Stages: executed={report.executed_stage_count}, "
        f"failed={report.failed_stage_count}, "
        f"blocked={report.blocked_stage_count}"
    )
    print(f"- Stage executions: {stages or 'none'}")
    print(
        f"- Mean duration: {report.mean_duration_ms:.1f} ms "
        f"(max={report.max_duration_ms:.1f} ms)\n"
    )


def print_adaptive_progress(event: WorkflowProgressEvent) -> None:
    if event.status is WorkflowProgressStatus.ROUTE_SELECTED:
        print(
            f"\nAdaptive Multi-Model route: {event.route.upper()} "
            f"({event.expected_stage_count} stages)"
        )
        return
    if event.stage_id is not None:
        labels = {
            WorkflowProgressStatus.STAGE_STARTED: "started",
            WorkflowProgressStatus.STAGE_COMPLETED: "completed",
            WorkflowProgressStatus.STAGE_FAILED: "failed",
            WorkflowProgressStatus.STAGE_BLOCKED: "blocked",
        }
        print(f"- {event.stage_id}: {labels[event.status]}")
        return
    labels = {
        WorkflowProgressStatus.WORKFLOW_COMPLETED: "completed",
        WorkflowProgressStatus.WORKFLOW_FAILED: "failed",
        WorkflowProgressStatus.WORKFLOW_CANCELLED: "cancelled",
    }
    print(f"Adaptive Multi-Model workflow: {labels[event.status]}")


def load_document(file_path: str) -> Document:
    return TextDocumentLoader().load(Path(file_path))


def load_documents(
    path: str,
) -> list[Document]:
    return create_default_ingestor().ingest(path)


def select_assistant_mode(
    input_fn: Callable[[str], str] = input,
) -> AssistantMode:
    choices = {
        "1": AssistantMode.SINGLE_MODEL,
        "single": AssistantMode.SINGLE_MODEL,
        "single_model": AssistantMode.SINGLE_MODEL,
        "2": AssistantMode.ADAPTIVE_MULTI_MODEL,
        "adaptive": AssistantMode.ADAPTIVE_MULTI_MODEL,
        "adaptive_multi_model": AssistantMode.ADAPTIVE_MULTI_MODEL,
    }
    print("Choose a section:")
    print("  1. Single Model")
    print("  2. Adaptive Multi-Model")
    while True:
        selected = input_fn("Mode: ").strip().casefold()
        mode = choices.get(selected)
        if mode is not None:
            return mode
        print("Invalid mode. Choose 1 or 2.")


def select_single_model_provider(
    input_fn: Callable[[str], str] = input,
) -> str:
    choices = {
        "1": "anthropic",
        "claude": "anthropic",
        "anthropic": "anthropic",
        "2": "openai",
        "gpt": "openai",
        "openai": "openai",
        "3": "gemini",
        "gemini": "gemini",
    }
    print("Choose an AI provider:")
    print("  1. Claude (Anthropic)")
    print("  2. GPT (OpenAI)")
    print("  3. Gemini (Google)")
    while True:
        selected = input_fn("Provider: ").strip().casefold()
        provider = choices.get(selected)
        if provider is not None:
            return provider
        print("Invalid provider. Choose 1, 2, or 3.")


def run_cli(
    manager: RAGConversationManager,
    input_fn: Callable[[str], str] = input,
    ingestor: DocumentIngestor | None = None,
    *,
    title: str = "AI RAG Chat",
    stream_responses: bool = True,
) -> None:
    ingestor = ingestor or create_default_ingestor()
    print(title)
    print_help()

    while True:
        try:
            prompt = input_fn("You: ").strip()

            if not prompt:
                continue

            command, _, argument = prompt.partition(" ")
            command = command.lower()
            argument = argument.strip()

            if command == "/exit":
                break

            if command == "/help":
                print_help()
                continue

            if command == "/save":
                manager.repository.save(manager.conversation)
                print("Conversation saved.\n")
                continue

            if command == "/clear":
                manager.conversation.clear()
                manager.repository.save(manager.conversation)
                print("Conversation cleared.\n")
                continue

            if command == "/history":
                print_history(manager.conversation)
                continue

            if command == "/metrics":
                print_adaptive_metrics(manager)
                continue

            if command == "/remember":
                if not argument:
                    print("Usage: /remember <fact>\n")
                    continue

                memory = manager.remember(argument)
                print(f"Remembered {memory.id}.\n")
                continue

            if command == "/memories":
                print_memories(manager)
                continue

            if command == "/forget":
                if not argument:
                    print("Usage: /forget <memory_id>\n")
                    continue

                if manager.forget(argument):
                    print(f"Forgot {argument}.\n")
                else:
                    print(f"Memory not found: {argument}\n")

                continue

            if command == "/index":
                if not argument:
                    print("Usage: /index <path>\n")
                    continue

                index_path = Path(argument).expanduser()

                if index_path.is_dir():
                    result = DirectorySynchronizer(
                        ingestor,
                        manager,
                    ).sync(index_path)
                    print(
                        f"Synchronized {result.root}: "
                        f"indexed={len(result.indexed_documents)}, "
                        f"unchanged={len(result.unchanged_documents)}, "
                        f"removed={len(result.removed_documents)}, "
                        f"chunks={result.indexed_chunks}.\n"
                    )
                    continue

                document = ingestor.ingest(index_path)[0]
                chunks = manager.index_document(document)
                print(f"Indexed {document.id}: {len(chunks)} chunks.")

                print()
                continue

            if command == "/documents":
                print_documents(manager)
                continue

            if command == "/remove":
                if not argument:
                    print("Usage: /remove <document_id>\n")
                    continue

                deleted_count = manager.delete_document(argument)

                if deleted_count:
                    print(f"Removed {argument}: {deleted_count} chunks.\n")
                else:
                    print(f"Document not found: {argument}\n")

                continue

            if command.startswith("/"):
                print(f"Unknown command: {command}. Use /help.\n")
                continue

            if stream_responses:
                print(
                    "\nAssistant: ",
                    end="",
                    flush=True,
                )

                for chunk in manager.stream_message(prompt):
                    print(
                        chunk,
                        end="",
                        flush=True,
                    )

                print()
            else:
                response = manager.send_message(prompt)
                print(f"\nAssistant: {response}")
            print_citations(manager.last_citations)

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        except Exception as error:
            print(f"\nError: {redact_secrets(str(error))}\n")


def main() -> None:
    mode = select_assistant_mode()
    if mode is AssistantMode.SINGLE_MODEL:
        provider = select_single_model_provider()
        run_cli(
            create_single_model_manager(provider),
            title=f"Single Model ({provider})",
        )
        return

    run_cli(
        create_adaptive_multi_model_manager(
            progress_handler=print_adaptive_progress,
        ),
        title="Adaptive Multi-Model",
        stream_responses=False,
    )


if __name__ == "__main__":
    main()
