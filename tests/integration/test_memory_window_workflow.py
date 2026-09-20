import pytest

from ai_sdk.application.conversation_manager import (
    ConversationManager,
)
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.context.summary import (
    ExtractiveConversationSummarizer,
)
from ai_sdk.context.window import SlidingContextWindow
from ai_sdk.core.conversation import Conversation
from ai_sdk.storage.json import JSONConversationRepository

pytestmark = pytest.mark.integration


class RecordingLLMClient:
    def __init__(self):
        self.received_messages = None

    def ask(self, messages):
        self.received_messages = messages
        return "Latest answer"

    def stream(self, messages):
        raise NotImplementedError


def test_context_window_limits_prompt_but_preserves_full_history(
    tmp_path,
):
    conversation = Conversation()
    conversation.add_user("Old question")
    conversation.add_assistant("Old answer")
    conversation.add_user("Recent question")
    conversation.add_assistant("Recent answer")
    repository = JSONConversationRepository(tmp_path / "chat.json")
    client = RecordingLLMClient()
    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(
            conversation,
            context_window=SlidingContextWindow(
                max_tokens=2,
                message_overhead=0,
            ),
            summary_memory=ExtractiveConversationSummarizer(max_tokens=8),
        ),
        client=client,
        repository=repository,
    )

    response = manager.send_message("Latest question")
    restored = repository.load()

    assert response == "Latest answer"
    assert len(client.received_messages) == 1
    prompt = client.received_messages[0]["content"]
    assert "User: Recent question" in prompt
    assert "Assistant: Recent answer" in prompt
    assert "Latest question" in prompt
    assert "Old question" not in prompt
    assert [message.content for message in restored.history()] == [
        "Old question",
        "Old answer",
        "Recent question",
        "Recent answer",
        "Latest question",
        "Latest answer",
    ]
