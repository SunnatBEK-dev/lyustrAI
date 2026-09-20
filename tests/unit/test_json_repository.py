import json

from ai_sdk.core.conversation import Conversation
from ai_sdk.storage.json import JSONConversationRepository


def test_load_missing_file_returns_empty_conversation(tmp_path):
    repository = JSONConversationRepository(tmp_path / "missing" / "chat.json")

    conversation = repository.load()

    assert conversation.is_empty()


def test_save_and_load_round_trip_preserves_message_identity(tmp_path):
    file_path = tmp_path / "nested" / "chat.json"
    repository = JSONConversationRepository(file_path)
    conversation = Conversation()
    user = conversation.add_user("Hello")
    assistant = conversation.add_assistant("Hi")

    repository.save(conversation)
    restored = repository.load()

    assert [message.to_dict() for message in restored.history()] == [
        user.to_dict(),
        assistant.to_dict(),
    ]


def test_load_migrates_legacy_message_id_once(tmp_path):
    file_path = tmp_path / "chat.json"
    file_path.write_text(
        json.dumps([{"role": "user", "content": "Legacy"}]),
        encoding="utf-8",
    )
    repository = JSONConversationRepository(file_path)

    first_load = repository.load()
    second_load = repository.load()

    first_id = first_load.last_message().id
    assert first_id.startswith("msg_")
    assert second_load.last_message().id == first_id
    assert json.loads(file_path.read_text(encoding="utf-8"))[0]["id"] == first_id


def test_load_skips_invalid_records(tmp_path):
    file_path = tmp_path / "chat.json"
    file_path.write_text(
        json.dumps(
            [
                {"id": "msg_valid", "role": "user", "content": "Keep"},
                "not-a-message",
                {"role": "assistant"},
            ]
        ),
        encoding="utf-8",
    )

    conversation = JSONConversationRepository(file_path).load()

    assert [message.id for message in conversation.history()] == ["msg_valid"]


def test_load_invalid_json_returns_empty_conversation(tmp_path):
    file_path = tmp_path / "chat.json"
    file_path.write_text("{invalid", encoding="utf-8")

    conversation = JSONConversationRepository(file_path).load()

    assert conversation.is_empty()


def test_load_non_list_payload_returns_empty_conversation(tmp_path):
    file_path = tmp_path / "chat.json"
    file_path.write_text('{"messages": []}', encoding="utf-8")

    conversation = JSONConversationRepository(file_path).load()

    assert conversation.is_empty()
