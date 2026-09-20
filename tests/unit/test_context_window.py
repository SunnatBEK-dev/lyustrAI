import pytest

from ai_sdk.context.window import (
    RegexTokenCounter,
    SlidingContextWindow,
)


def test_regex_token_counter_counts_words_and_punctuation():
    counter = RegexTokenCounter()

    assert counter.count("Hello, world!") == 4
    assert counter.count("   ") == 0


def test_window_keeps_newest_complete_turns_in_order():
    messages = [
        {"role": "user", "content": "one two"},
        {"role": "assistant", "content": "three four"},
        {"role": "user", "content": "five"},
        {"role": "assistant", "content": "six"},
        {"role": "user", "content": "seven"},
    ]
    window = SlidingContextWindow(
        max_tokens=3,
        message_overhead=0,
    )

    selected = window.select(messages)

    assert selected == [
        {"role": "user", "content": "five"},
        {"role": "assistant", "content": "six"},
        {"role": "user", "content": "seven"},
    ]


def test_window_partition_reports_excluded_messages():
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest"},
    ]
    selection = SlidingContextWindow(
        max_tokens=1,
        message_overhead=0,
    ).partition(messages)

    assert selection.included == [messages[-1]]
    assert selection.excluded == messages[:2]


def test_window_always_keeps_latest_turn_when_it_exceeds_budget():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {
            "role": "user",
            "content": "oversized current question",
        },
    ]
    window = SlidingContextWindow(
        max_tokens=2,
        message_overhead=0,
    )

    assert window.select(messages) == [messages[-1]]


def test_window_returns_copies_without_mutating_input():
    messages = [{"role": "user", "content": "Question"}]
    selected = SlidingContextWindow(
        max_tokens=10,
        message_overhead=0,
    ).select(messages)

    selected[0]["content"] = "Changed"

    assert messages[0]["content"] == "Question"


def test_window_returns_empty_for_empty_history():
    assert SlidingContextWindow(10).select([]) == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_tokens": 0}, "greater than zero"),
        (
            {"max_tokens": 1, "message_overhead": -1},
            "overhead",
        ),
    ],
)
def test_window_rejects_invalid_configuration(
    options,
    message,
):
    with pytest.raises(ValueError, match=message):
        SlidingContextWindow(**options)
