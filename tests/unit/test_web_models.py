import pytest
from pydantic import ValidationError

from ai_sdk.web.models import ChatStreamRequest, ConversationRenameRequest


def test_single_model_chat_requires_provider_and_strips_message():
    request = ChatStreamRequest(
        message="  Explain retrieval  ",
        mode="single",
        provider="openai",
    )

    assert request.message == "Explain retrieval"
    assert request.provider == "openai"

    with pytest.raises(ValidationError, match="requires a provider"):
        ChatStreamRequest(message="question", mode="single")


def test_adaptive_request_rejects_explicit_provider_and_blank_message():
    automatic = ChatStreamRequest(message="question", mode="adaptive")
    maximum = ChatStreamRequest(
        message="question",
        mode="adaptive",
        adaptive_strategy="maximum",
    )

    assert automatic.adaptive_strategy == "auto"
    assert maximum.adaptive_strategy == "maximum"

    with pytest.raises(ValidationError, match="automatically"):
        ChatStreamRequest(
            message="question",
            mode="adaptive",
            provider="openai",
        )

    with pytest.raises(ValidationError, match="blank"):
        ChatStreamRequest(message="   ", mode="adaptive")
    with pytest.raises(ValidationError, match="does not accept"):
        ChatStreamRequest(
            message="question",
            mode="single",
            provider="openai",
            adaptive_strategy="maximum",
        )


def test_conversation_id_and_rename_validation():
    request = ChatStreamRequest(
        message="continue",
        mode="single",
        provider="gemini",
        conversation_id="chat_012345abcdef",
    )
    rename = ConversationRenameRequest(title="  Retrieval   notes  ")

    assert request.conversation_id == "chat_012345abcdef"
    assert rename.title == "Retrieval notes"

    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        ChatStreamRequest(
            message="continue",
            mode="single",
            provider="gemini",
            conversation_id="../other",
        )
    with pytest.raises(ValidationError, match="blank"):
        ConversationRenameRequest(title="   ")
