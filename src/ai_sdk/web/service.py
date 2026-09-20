from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from uuid import uuid4

from ai_sdk.agents import (
    MultiModelRoute,
    WorkflowCancelledError,
    WorkflowProgressEvent,
    WorkflowProgressStatus,
)
from ai_sdk.application.bootstrap import (
    AssistantRuntimeResources,
    create_adaptive_multi_model_manager,
    create_assistant_runtime,
    create_single_model_manager,
)
from ai_sdk.application.rag_manager import RAGConversationManager
from ai_sdk.config import (
    UPLOAD_DIR,
    UPLOAD_MAX_BYTES,
    WEB_CONVERSATION_DIR,
)
from ai_sdk.ingestion import DocumentIngestor, create_default_ingestor
from ai_sdk.llm.adaptive_metrics import InMemoryAdaptiveMetrics
from ai_sdk.llm.adaptive_multi_model import AdaptiveMultiModelClient
from ai_sdk.readiness import inspect_provider_readiness
from ai_sdk.retrieval.catalog import IndexedDocument
from ai_sdk.security import redact_secrets
from ai_sdk.web.conversations import JSONConversationCatalog
from ai_sdk.web.models import ChatStreamRequest

SUPPORTED_UPLOAD_SUFFIXES = frozenset({".pdf", ".md", ".markdown", ".txt"})
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_END_OF_STREAM = object()


class ActiveRunError(RuntimeError):
    """Raised when the single-user demo already has an active request."""


class UploadValidationError(ValueError):
    """Raised when an uploaded document violates the local policy."""


@dataclass(frozen=True)
class ChatStreamRun:
    run_id: str
    conversation_id: str
    events: Iterator[dict[str, object]]


@dataclass
class _ActiveRun:
    run_id: str
    conversation_id: str
    created_conversation: bool
    manager: RAGConversationManager | None = None
    cancellation_requested: bool = False


SingleModelManagerFactory = Callable[[str, Path], RAGConversationManager]
AdaptiveMultiModelManagerFactory = Callable[
    [Callable[[WorkflowProgressEvent], None], Path, MultiModelRoute | None],
    RAGConversationManager,
]


class KnowledgeAssistantService:
    """Single-user assistant service with shared retrieval state."""

    def __init__(
        self,
        *,
        runtime: AssistantRuntimeResources | None = None,
        ingestor: DocumentIngestor | None = None,
        upload_dir: Path = UPLOAD_DIR,
        conversation_dir: Path = WEB_CONVERSATION_DIR,
        upload_max_bytes: int = UPLOAD_MAX_BYTES,
        single_model_manager_factory: SingleModelManagerFactory | None = None,
        adaptive_multi_model_manager_factory: AdaptiveMultiModelManagerFactory
        | None = None,
    ) -> None:
        if upload_max_bytes <= 0:
            raise ValueError("Upload byte limit must be positive.")
        self.runtime = runtime or create_assistant_runtime()
        self.ingestor = ingestor or create_default_ingestor()
        self.upload_dir = Path(upload_dir)
        self.conversations = JSONConversationCatalog(conversation_dir)
        self.upload_max_bytes = upload_max_bytes
        self.metrics = InMemoryAdaptiveMetrics()
        self._single_model_manager_factory = single_model_manager_factory or (
            lambda provider, conversation_file: create_single_model_manager(
                provider,
                conversation_file=conversation_file,
                runtime=self.runtime,
            )
        )
        self._adaptive_multi_model_manager_factory = (
            adaptive_multi_model_manager_factory
            or (
                lambda progress, conversation_file, route_override: (
                    create_adaptive_multi_model_manager(
                        conversation_file=conversation_file,
                        progress_handler=progress,
                        runtime=self.runtime,
                        metrics=self.metrics,
                        route_override=route_override,
                    )
                )
            )
        )
        self._run_lock = Lock()
        self._active_run: _ActiveRun | None = None

    def status(self) -> dict[str, object]:
        readiness = inspect_provider_readiness()
        with self._run_lock:
            active_run_id = (
                None if self._active_run is None else self._active_run.run_id
            )
        return {
            "name": "LyustrAI",
            "version": "0.1.0",
            "adaptive_ready": readiness.adaptive_ready,
            "providers": [
                {
                    "provider": provider.provider,
                    "display_name": provider.display_name,
                    "ready": provider.ready,
                    "missing_variables": list(provider.missing_variables),
                }
                for provider in readiness.providers
            ],
            "document_count": len(self.documents()),
            "active_run_id": active_run_id,
            "adaptive_metrics": self.metrics.report().to_dict(),
        }

    def documents(self) -> list[dict[str, object]]:
        return [self._document_payload(item) for item in self._catalog()]

    def list_conversations(self) -> list[dict[str, object]]:
        return [item.to_dict() for item in self.conversations.list()]

    def get_conversation(self, conversation_id: str) -> dict[str, object]:
        return self.conversations.detail(conversation_id)

    def rename_conversation(
        self,
        conversation_id: str,
        title: str,
    ) -> dict[str, object]:
        self._require_idle("rename")
        return self.conversations.rename(conversation_id, title).to_dict()

    def delete_conversation(self, conversation_id: str) -> None:
        self._require_idle("delete")
        self.conversations.delete(conversation_id)

    def index_upload(
        self,
        filename: str,
        content: bytes,
    ) -> dict[str, object]:
        safe_name = self._validated_filename(filename)
        if not content:
            raise UploadValidationError("Uploaded document is empty.")
        if len(content) > self.upload_max_bytes:
            raise UploadValidationError("Uploaded document exceeds 10 MB.")

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        target = self.upload_dir / safe_name
        previous = target.read_bytes() if target.exists() else None
        target.write_bytes(content)
        try:
            document = self.ingestor.ingest(target)[0]
            self.runtime.retriever.index_document(
                document.id,
                self.runtime.chunker.split(document),
            )
        except Exception:
            if previous is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(previous)
            raise

        indexed = next(
            item for item in self._catalog() if item.document_id == document.id
        )
        return self._document_payload(indexed)

    def delete_document(self, document_id: str) -> int:
        item = next(
            (
                document
                for document in self._catalog()
                if document.document_id == document_id
            ),
            None,
        )
        deleted = self.runtime.retriever.delete_document(document_id)
        if deleted and item is not None:
            source_path = Path(item.source)
            try:
                source_path.resolve().relative_to(self.upload_dir.resolve())
            except ValueError:
                pass
            else:
                source_path.unlink(missing_ok=True)
        return deleted

    def start_chat(self, request: ChatStreamRequest) -> ChatStreamRun:
        self._validate_readiness(request)
        run_id = f"run_{uuid4().hex[:12]}"
        event_queue: Queue[object] = Queue()
        with self._run_lock:
            if self._active_run is not None:
                raise ActiveRunError(
                    "Another request is already running in this local demo."
                )
            if request.conversation_id is None:
                metadata = self.conversations.create(
                    request.message,
                    mode=request.mode,
                    provider=request.provider,
                    adaptive_strategy=request.adaptive_strategy,
                )
                created_conversation = True
            else:
                metadata = self.conversations.get(request.conversation_id)
                created_conversation = False
            active = _ActiveRun(
                run_id=run_id,
                conversation_id=metadata.conversation_id,
                created_conversation=created_conversation,
            )
            self._active_run = active

        worker = Thread(
            target=self._run_chat,
            args=(active, request, event_queue),
            daemon=True,
            name=f"adaptive-chat-{run_id}",
        )
        worker.start()
        return ChatStreamRun(
            run_id,
            active.conversation_id,
            self._event_iterator(run_id, event_queue),
        )

    def cancel(self, run_id: str) -> bool:
        with self._run_lock:
            active = self._active_run
            if active is None or active.run_id != run_id:
                return False
            active.cancellation_requested = True
            manager = active.manager

        if manager is None:
            return True
        client = manager.client
        if isinstance(client, AdaptiveMultiModelClient):
            client.cancel()
        return True

    def _run_chat(
        self,
        active: _ActiveRun,
        request: ChatStreamRequest,
        event_queue: Queue[object],
    ) -> None:
        completed = False

        def progress(event: WorkflowProgressEvent) -> None:
            event_type = (
                "route"
                if event.status is WorkflowProgressStatus.ROUTE_SELECTED
                else "stage"
            )
            event_queue.put(
                {
                    "event": event_type,
                    "data": event.to_dict(),
                }
            )

        try:
            conversation_file = self.conversations.messages_path(active.conversation_id)
            manager = (
                self._adaptive_multi_model_manager_factory(
                    progress,
                    conversation_file,
                    (
                        MultiModelRoute.FULL
                        if request.adaptive_strategy == "maximum"
                        else None
                    ),
                )
                if request.mode == "adaptive"
                else self._single_model_manager_factory(
                    str(request.provider),
                    conversation_file,
                )
            )
            with self._run_lock:
                if self._active_run is active:
                    active.manager = manager
                cancelled_before_start = active.cancellation_requested
            if cancelled_before_start:
                raise WorkflowCancelledError()

            event_queue.put(
                {
                    "event": "run",
                    "data": {
                        "run_id": active.run_id,
                        "conversation_id": active.conversation_id,
                        "adaptive_strategy": request.adaptive_strategy,
                    },
                }
            )
            response = manager.send_message_with_citations(request.message)
            with self._run_lock:
                cancelled_after_response = active.cancellation_requested
            if cancelled_after_response:
                raise WorkflowCancelledError()
            self.conversations.record_turn(
                active.conversation_id,
                mode=request.mode,
                provider=request.provider,
                adaptive_strategy=request.adaptive_strategy,
                message_count=manager.conversation.message_count(),
            )
            completed = True
            citations = [
                {
                    "position": citation.position,
                    "document_id": citation.document_id,
                    "chunk_id": citation.chunk_id,
                    "source": Path(citation.source).name,
                    "score": citation.score,
                    "page": citation.page,
                }
                for citation in response.citations
            ]
            event_queue.put({"event": "citations", "data": citations})
            event_queue.put(
                {
                    "event": "answer",
                    "data": {"content": response.content},
                }
            )
        except WorkflowCancelledError:
            event_queue.put(
                {
                    "event": "cancelled",
                    "data": {"run_id": active.run_id},
                }
            )
        except Exception as error:
            event_queue.put(
                {
                    "event": "error",
                    "data": {
                        "type": type(error).__name__,
                        "message": self._safe_error_message(error),
                    },
                }
            )
        finally:
            if not completed:
                message_count = (
                    0
                    if active.manager is None
                    else active.manager.conversation.message_count()
                )
                if message_count:
                    self.conversations.record_turn(
                        active.conversation_id,
                        mode=request.mode,
                        provider=request.provider,
                        adaptive_strategy=request.adaptive_strategy,
                        message_count=message_count,
                    )
                elif active.created_conversation:
                    self.conversations.discard_if_empty(active.conversation_id)
            with self._run_lock:
                if self._active_run is active:
                    self._active_run = None
            event_queue.put(_END_OF_STREAM)

    def _event_iterator(
        self,
        run_id: str,
        event_queue: Queue[object],
    ) -> Iterator[dict[str, object]]:
        try:
            while True:
                event = event_queue.get()
                if event is _END_OF_STREAM:
                    return
                if isinstance(event, dict):
                    yield event
        finally:
            self.cancel(run_id)

    def _require_idle(self, action: str) -> None:
        with self._run_lock:
            if self._active_run is not None:
                raise ActiveRunError(
                    f"Cannot {action} a conversation while a request is running."
                )

    def _validate_readiness(self, request: ChatStreamRequest) -> None:
        readiness = inspect_provider_readiness()
        if request.mode == "adaptive":
            if not readiness.adaptive_ready:
                raise RuntimeError(
                    "Adaptive Multi-Model requires all provider keys and models."
                )
            return
        provider = readiness.for_provider(str(request.provider))
        if not provider.ready:
            missing = ", ".join(provider.missing_variables)
            raise RuntimeError(
                f"{provider.display_name} is not configured. Missing: {missing}."
            )

    def _catalog(self) -> list[IndexedDocument]:
        return self.runtime.retriever.document_catalog()

    @staticmethod
    def _document_payload(item: IndexedDocument) -> dict[str, object]:
        suffix = item.format or Path(item.source).suffix.lower().lstrip(".") or "text"
        return {
            "document_id": item.document_id,
            "source": Path(item.source).name,
            "format": suffix,
            "chunk_count": item.chunk_count,
            "page_count": item.page_count,
        }

    @staticmethod
    def _safe_error_message(error: Exception) -> str:
        message = redact_secrets(str(error))
        safe_markers = (
            "not configured",
            "requires all provider",
            "already running",
        )
        if any(marker in message.casefold() for marker in safe_markers):
            return message
        return "The request failed. Check the local server log for details."

    @staticmethod
    def _validated_filename(filename: str) -> str:
        if not isinstance(filename, str) or not filename.strip():
            raise UploadValidationError("Upload filename is required.")
        stripped = filename.strip()
        if Path(stripped).name != stripped or "\\" in stripped:
            raise UploadValidationError("Upload filename contains a path.")
        suffix = Path(stripped).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            raise UploadValidationError(
                "Supported document formats are PDF, Markdown, and TXT."
            )
        stem = _SAFE_FILENAME.sub("-", Path(stripped).stem).strip("._-")
        if not stem:
            raise UploadValidationError("Upload filename is invalid.")
        return f"{stem[:96]}{suffix}"
