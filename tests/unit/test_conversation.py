from ai_sdk.core.conversation import Conversation
from ai_sdk.core.message import Message


def test_new_conversation_is_empty():
    conversation = Conversation()

    assert conversation.is_empty()
    assert conversation.history() == []
    assert conversation.last_message() is None
    assert conversation.message_count() == 0


def test_add_messages_preserves_order_and_types():
    conversation = Conversation()

    user = conversation.add_user("Hello")
    assistant = conversation.add_assistant("Hi")

    assert isinstance(user, Message)
    assert isinstance(assistant, Message)
    assert conversation.history() == [user, assistant]
    assert conversation.last_message() is assistant
    assert conversation.message_count() == 2


def test_edit_message_updates_only_matching_message():
    conversation = Conversation()
    first = conversation.add_user("Before")
    second = conversation.add_assistant("Unchanged")

    assert conversation.edit_message(first.id, "After") is True
    assert first.content == "After"
    assert second.content == "Unchanged"
    assert conversation.edit_message("msg_missing", "No") is False


def test_delete_message_removes_only_matching_message():
    conversation = Conversation()
    first = conversation.add_user("Remove")
    second = conversation.add_assistant("Keep")

    assert conversation.delete_message(first.id) is True
    assert conversation.history() == [second]
    assert conversation.delete_message(first.id) is False


def test_recent_messages_honors_limit():
    conversation = Conversation()
    conversation.add_user("one")
    second = conversation.add_assistant("two")
    third = conversation.add_user("three")

    assert conversation.recent_messages(2) == [second, third]
    assert conversation.recent_messages(0) == []
    assert conversation.recent_messages(-1) == []


def test_clear_removes_all_messages():
    conversation = Conversation()
    conversation.add_user("Hello")

    conversation.clear()

    assert conversation.is_empty()
