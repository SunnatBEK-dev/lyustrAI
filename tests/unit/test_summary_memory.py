import pytest

from ai_sdk.context.summary import (
    ExtractiveConversationSummarizer,
)
from ai_sdk.context.window import RegexTokenCounter


def test_summary_keeps_recent_excluded_dialogue_in_order():
    messages = [
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
    ]
    summarizer = ExtractiveConversationSummarizer(max_tokens=8)

    summary = summarizer.summarize(messages)

    assert summary == ("User: Recent question\nAssistant: Recent answer")


def test_summary_normalizes_whitespace_and_honors_budget():
    counter = RegexTokenCounter()
    summarizer = ExtractiveConversationSummarizer(
        max_tokens=4,
        token_counter=counter,
    )

    summary = summarizer.summarize(
        [
            {
                "role": "user",
                "content": "Very   long\n historical detail",
            }
        ]
    )

    assert summary.startswith("User:")
    assert "  " not in summary
    assert counter.count(summary) <= 4


def test_summary_skips_blank_messages():
    summarizer = ExtractiveConversationSummarizer(max_tokens=10)

    assert summarizer.summarize([{"role": "user", "content": "   "}]) == ""


def test_summary_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="greater than zero"):
        ExtractiveConversationSummarizer(max_tokens=0)
