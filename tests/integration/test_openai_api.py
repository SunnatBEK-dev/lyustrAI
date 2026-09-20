import os

import pytest

from ai_sdk.llm.openai import OpenAIClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external,
]


@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INTEGRATION") != "1",
    reason=("Set RUN_OPENAI_INTEGRATION=1 to allow a real OpenAI request."),
)
def test_real_openai_request_returns_text():
    client = OpenAIClient(max_output_tokens=16)

    response = client.ask(
        [
            {
                "role": "user",
                "content": "Reply with the single word OK.",
            }
        ]
    )

    assert response.strip()
