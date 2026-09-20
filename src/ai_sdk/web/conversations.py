from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ai_sdk.storage.json import JSONConversationRepository

CHAT_SCHEMA_VERSION = 1
CHAT_ID_PATTERN = re.compile(r"^chat_[0-9a-f]{12}$")
CHAT_TITLE_MAX_LENGTH = 80
AUTO_TITLE_MAX_LENGTH = 60
CHAT_MODES = frozenset({"single", "adaptive"})
CHAT_PROVIDERS = frozenset({"anthropic", "openai", "gemini"})
CHAT_ADAPTIVE_STRATEGIES = frozenset({"auto", "maximum"})


class ConversationNotFoundError(LookupError):
    """Raised when a local web conversation does not exist."""


@dataclass(frozen=True)
class ConversationMetadata:
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    last_mode: str
    last_provider: str | None
    last_adaptive_strategy: str
    message_count: int
    schema_version: int = CHAT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_mode": self.last_mode,
            "last_provider": self.last_provider,
            "last_adaptive_strategy": self.last_adaptive_strategy,
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: object) -> ConversationMetadata:
        if not isinstance(data, dict):
            raise ValueError("Conversation metadata must be an object.")

        conversation_id = data.get("conversation_id")
        title = data.get("title")
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        last_mode = data.get("last_mode")
        last_provider = data.get("last_provider")
        last_adaptive_strategy = data.get("last_adaptive_strategy", "auto")
        message_count = data.get("message_count")
        schema_version = data.get("schema_version")

        if (
            not isinstance(conversation_id, str)
            or CHAT_ID_PATTERN.fullmatch(conversation_id) is None
        ):
            raise ValueError("Conversation ID is invalid.")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Conversation title is invalid.")
        if len(title) > CHAT_TITLE_MAX_LENGTH:
            raise ValueError("Conversation title is too long.")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("Conversation creation timestamp is invalid.")
        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError("Conversation update timestamp is invalid.")
        if last_mode not in CHAT_MODES:
            raise ValueError("Conversation mode is invalid.")
        if last_provider is not None and last_provider not in CHAT_PROVIDERS:
            raise ValueError("Conversation provider is invalid.")
        if last_adaptive_strategy not in CHAT_ADAPTIVE_STRATEGIES:
            raise ValueError("Conversation adaptive strategy is invalid.")
        if (
            not isinstance(message_count, int)
            or isinstance(message_count, bool)
            or message_count < 0
        ):
            raise ValueError("Conversation message count is invalid.")
        if schema_version != CHAT_SCHEMA_VERSION:
            raise ValueError("Conversation schema version is unsupported.")

        return cls(
            conversation_id=conversation_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            last_mode=last_mode,
            last_provider=last_provider,
            last_adaptive_strategy=last_adaptive_strategy,
            message_count=message_count,
            schema_version=schema_version,
        )


class JSONConversationCatalog:
    """Store web chat metadata beside provider-neutral message histories."""

    def __init__(
        self,
        root: Path,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = Path(root)
        self._now = now or (lambda: datetime.now(timezone.utc))

    def create(
        self,
        first_message: str,
        *,
        mode: str,
        provider: str | None,
        adaptive_strategy: str | None = None,
    ) -> ConversationMetadata:
        effective_strategy = (
            "auto"
            if mode == "adaptive" and adaptive_strategy is None
            else adaptive_strategy
        )
        self._validate_selection(mode, provider, effective_strategy)
        conversation_id = self._new_id()
        timestamp = self._timestamp()
        metadata = ConversationMetadata(
            conversation_id=conversation_id,
            title=self.title_from_message(first_message),
            created_at=timestamp,
            updated_at=timestamp,
            last_mode=mode,
            last_provider=provider,
            last_adaptive_strategy=effective_strategy or "auto",
            message_count=0,
        )
        directory = self._directory(conversation_id)
        directory.mkdir(parents=True, exist_ok=False)
        self._write_metadata(metadata)
        return metadata

    def list(self) -> list[ConversationMetadata]:
        if not self.root.exists():
            return []

        conversations: list[ConversationMetadata] = []
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            try:
                metadata = self._read_metadata(directory.name)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            conversations.append(metadata)
        return sorted(
            conversations,
            key=lambda item: (item.updated_at, item.created_at),
            reverse=True,
        )

    def get(self, conversation_id: str) -> ConversationMetadata:
        try:
            return self._read_metadata(conversation_id)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ConversationNotFoundError("Conversation not found.") from error

    def detail(self, conversation_id: str) -> dict[str, object]:
        metadata = self.get(conversation_id)
        conversation = JSONConversationRepository(
            self.messages_path(conversation_id)
        ).load()
        return {
            **metadata.to_dict(),
            "messages": [message.to_dict() for message in conversation.history()],
        }

    def messages_path(self, conversation_id: str) -> Path:
        self.get(conversation_id)
        return self._directory(conversation_id) / "messages.json"

    def rename(self, conversation_id: str, title: str) -> ConversationMetadata:
        metadata = self.get(conversation_id)
        normalized = " ".join(title.split())
        if not normalized or len(normalized) > CHAT_TITLE_MAX_LENGTH:
            raise ValueError("Conversation title must contain 1 to 80 characters.")
        renamed = replace(metadata, title=normalized)
        self._write_metadata(renamed)
        return renamed

    def record_turn(
        self,
        conversation_id: str,
        *,
        mode: str,
        provider: str | None,
        adaptive_strategy: str | None = None,
        message_count: int,
    ) -> ConversationMetadata:
        effective_strategy = (
            "auto"
            if mode == "adaptive" and adaptive_strategy is None
            else adaptive_strategy
        )
        self._validate_selection(mode, provider, effective_strategy)
        if message_count < 0:
            raise ValueError("Conversation message count cannot be negative.")
        metadata = self.get(conversation_id)
        updated = replace(
            metadata,
            updated_at=self._timestamp(),
            last_mode=mode,
            last_provider=provider,
            last_adaptive_strategy=(
                effective_strategy or metadata.last_adaptive_strategy
            ),
            message_count=message_count,
        )
        self._write_metadata(updated)
        return updated

    def delete(self, conversation_id: str) -> None:
        self.get(conversation_id)
        directory = self._directory(conversation_id)
        shutil.rmtree(directory)

    def discard_if_empty(self, conversation_id: str) -> bool:
        try:
            metadata = self.get(conversation_id)
        except ConversationNotFoundError:
            return False
        conversation = JSONConversationRepository(
            self.messages_path(metadata.conversation_id)
        ).load()
        if not conversation.is_empty():
            return False
        self.delete(metadata.conversation_id)
        return True

    @staticmethod
    def title_from_message(message: str) -> str:
        normalized = " ".join(message.split())
        if not normalized:
            raise ValueError("Conversation title source cannot be blank.")
        if len(normalized) <= AUTO_TITLE_MAX_LENGTH:
            return normalized
        return f"{normalized[: AUTO_TITLE_MAX_LENGTH - 1].rstrip()}…"

    def _new_id(self) -> str:
        for _ in range(10):
            conversation_id = f"chat_{uuid4().hex[:12]}"
            if not self._directory(conversation_id).exists():
                return conversation_id
        raise RuntimeError("Could not allocate a unique conversation ID.")

    def _directory(self, conversation_id: str) -> Path:
        if (
            not isinstance(conversation_id, str)
            or CHAT_ID_PATTERN.fullmatch(conversation_id) is None
        ):
            raise ValueError("Conversation ID is invalid.")
        directory = self.root / conversation_id
        directory.resolve().relative_to(self.root.resolve())
        return directory

    def _read_metadata(self, conversation_id: str) -> ConversationMetadata:
        path = self._directory(conversation_id) / "metadata.json"
        with path.open("r", encoding="utf-8") as file:
            return ConversationMetadata.from_dict(json.load(file))

    def _write_metadata(self, metadata: ConversationMetadata) -> None:
        path = self._directory(metadata.conversation_id) / "metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(metadata.to_dict(), file, indent=2, ensure_ascii=False)
        temporary.replace(path)

    def _timestamp(self) -> str:
        return self._now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _validate_selection(
        mode: str,
        provider: str | None,
        adaptive_strategy: str | None,
    ) -> None:
        if mode not in CHAT_MODES:
            raise ValueError("Conversation mode is invalid.")
        if mode == "single" and provider not in CHAT_PROVIDERS:
            raise ValueError("Single Model conversation requires a provider.")
        if mode == "single" and adaptive_strategy is not None:
            raise ValueError("Single Model conversation rejects adaptive strategy.")
        if mode == "adaptive" and provider is not None:
            raise ValueError("Adaptive conversation does not accept a provider.")
        if mode == "adaptive" and adaptive_strategy not in CHAT_ADAPTIVE_STRATEGIES:
            raise ValueError("Adaptive conversation requires a valid strategy.")
