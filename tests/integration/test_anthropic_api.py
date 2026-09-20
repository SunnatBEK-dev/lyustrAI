import os

import pytest

from ai_sdk.llm.anthropic import AnthropicClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external,
]


@pytest.mark.skipif(
    os.getenv("RUN_ANTHROPIC_INTEGRATION") != "1",
    reason=("Set RUN_ANTHROPIC_INTEGRATION=1 to allow a real Anthropic request."),
)
def test_real_claude_request_returns_text():
    client = AnthropicClient(max_tokens=16)

    response = client.ask(
        [
            {
                "role": "user",
                "content": "Reply with the single word OK.",
            }
        ]
    )

    assert response.strip()
