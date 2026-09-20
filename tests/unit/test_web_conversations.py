import json
from datetime import datetime, timedelta, timezone

import pytest

from ai_sdk.core.conversation import Conversation
from ai_sdk.storage.json import JSONConversationRepository
from ai_sdk.web.conversations import (
    ConversationNotFoundError,
    JSONConversationCatalog,
)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 2, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def save_turn(catalog, conversation_id, user="Question", assistant="Answer"):
    conversation = Conversation()
    conversation.add_user(user)
    conversation.add_assistant(assistant)
    JSONConversationRepository(catalog.messages_path(conversation_id)).save(
        conversation
    )
    catalog.record_turn(
        conversation_id,
        mode="single",
        provider="openai",
        message_count=conversation.message_count(),
    )


def test_catalog_creates_lists_and_loads_isolated_conversations(tmp_path):
    catalog = JSONConversationCatalog(tmp_path / "chats", now=Clock())
    first = catalog.create(
        "  Explain   hybrid retrieval  ",
        mode="single",
        provider="openai",
    )
    save_turn(catalog, first.conversation_id, "First question", "First answer")
    second = catalog.create(
        "A" * 80,
        mode="adaptive",
        provider=None,
        adaptive_strategy="maximum",
    )
    save_turn(catalog, second.conversation_id, "Second question", "Second answer")

    listed = catalog.list()
    detail = catalog.detail(first.conversation_id)

    assert [item.conversation_id for item in listed] == [
        second.conversation_id,
        first.conversation_id,
    ]
    assert first.title == "Explain hybrid retrieval"
    assert len(second.title) == 60
    assert second.title.endswith("…")
    assert second.last_adaptive_strategy == "maximum"
    assert [message["content"] for message in detail["messages"]] == [
        "First question",
        "First answer",
    ]


def test_catalog_renames_deletes_and_discards_only_empty_chat(tmp_path):
    catalog = JSONConversationCatalog(tmp_path / "chats", now=Clock())
    empty = catalog.create("Empty", mode="single", provider="gemini")
    saved = catalog.create("Saved", mode="single", provider="openai")
    save_turn(catalog, saved.conversation_id)

    renamed = catalog.rename(saved.conversation_id, "  Project   notes  ")

    assert renamed.title == "Project notes"
    assert catalog.discard_if_empty(empty.conversation_id) is True
    assert catalog.discard_if_empty(saved.conversation_id) is False
    catalog.delete(saved.conversation_id)
    assert catalog.list() == []
    with pytest.raises(ConversationNotFoundError):
        catalog.get(saved.conversation_id)


@pytest.mark.parametrize(
    "conversation_id",
    ["../chat_012345abcdef", "chat_012345abcdeg", "chat_short", ""],
)
def test_catalog_rejects_unsafe_or_invalid_ids(tmp_path, conversation_id):
    catalog = JSONConversationCatalog(tmp_path / "chats")

    with pytest.raises(ConversationNotFoundError):
        catalog.get(conversation_id)


def test_catalog_skips_corrupt_metadata_and_validates_titles(tmp_path):
    root = tmp_path / "chats"
    catalog = JSONConversationCatalog(root, now=Clock())
    valid = catalog.create("Valid", mode="single", provider="anthropic")
    corrupt = root / "chat_111111111111"
    corrupt.mkdir()
    (corrupt / "metadata.json").write_text("{", encoding="utf-8")

    assert [item.conversation_id for item in catalog.list()] == [valid.conversation_id]
    with pytest.raises(ValueError, match="1 to 80"):
        catalog.rename(valid.conversation_id, " ")
    with pytest.raises(ValueError, match="1 to 80"):
        catalog.rename(valid.conversation_id, "x" * 81)

    metadata_path = root / valid.conversation_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    metadata.pop("last_adaptive_strategy")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert catalog.get(valid.conversation_id).last_adaptive_strategy == "auto"
