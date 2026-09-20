import pytest

from ai_sdk.core.message import Message


def test_create_builds_message_with_unique_stable_identity():
    first = Message.create("user", "Hello")
    second = Message.create("user", "Hello")

    assert first.role == "user"
    assert first.content == "Hello"
    assert first.id.startswith("msg_")
    assert first.id != second.id


def test_message_round_trip_preserves_all_fields():
    original = Message(
        id="msg_existing",
        role="assistant",
        content="Welcome",
    )

    restored = Message.from_dict(original.to_dict())

    assert restored == original


def test_from_dict_generates_identity_for_legacy_message():
    message = Message.from_dict(
        {
            "role": "user",
            "content": "Legacy",
        }
    )

    assert message.id.startswith("msg_")
    assert message.role == "user"
    assert message.content == "Legacy"


def test_from_dict_requires_role_and_content():
    with pytest.raises(KeyError):
        Message.from_dict({"role": "user"})
