import os

import pytest

from ai_sdk.llm.gemini import GeminiClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external,
]


@pytest.mark.skipif(
    os.getenv("RUN_GEMINI_INTEGRATION") != "1",
    reason=("Set RUN_GEMINI_INTEGRATION=1 to allow a real Gemini request."),
)
def test_real_gemini_request_returns_text():
    client = GeminiClient(max_output_tokens=16)

    response = client.ask(
        [
            {
                "role": "user",
                "content": "Reply with the single word OK.",
            }
        ]
    )

    assert response.strip()
