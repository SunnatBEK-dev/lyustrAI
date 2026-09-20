import pytest

from ai_sdk.application.conversation_manager import ConversationManager
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.storage.json import JSONConversationRepository

pytestmark = pytest.mark.integration


class DeterministicClient:
    def ask(self, messages):
        return f"Echo: {messages[-1]['content']}"

    def stream(self, messages):
        yield "Echo: "
        yield messages[-1]["content"]


def test_streamed_conversation_survives_repository_round_trip(tmp_path):
    repository = JSONConversationRepository(tmp_path / "chat.json")
    conversation = repository.load()
    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=DeterministicClient(),
        repository=repository,
    )

    chunks = list(manager.stream_message("Hello"))
    restored = repository.load()

    assert chunks == ["Echo: ", "Hello"]
    assert [message.content for message in restored.history()] == [
        "Hello",
        "Echo: Hello",
    ]
    assert [message.id for message in restored.history()] == [
        message.id for message in conversation.history()
    ]
