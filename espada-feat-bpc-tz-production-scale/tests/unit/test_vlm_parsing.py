"""Unit tests for VLM chat-completion response parsing."""

from __future__ import annotations

import json

import pytest

from src.vlm.parsing import (
    ChatCompletionParseError,
    parse_chat_completion,
    parse_json_strict,
)

pytestmark = pytest.mark.unit


def _completion(content="hello", **extra):
    obj = {
        "id": "cmpl-1",
        "model": "test-model",
        "created": 123,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    obj.update(extra)
    return obj


def test_parse_from_dict():
    parsed = parse_chat_completion(_completion("hi there"))
    assert parsed.content == "hi there"
    assert parsed.id == "cmpl-1"
    assert parsed.model == "test-model"
    assert parsed.created == 123
    assert parsed.role == "assistant"
    assert parsed.finish_reason == "stop"
    assert parsed.usage == {"prompt_tokens": 5, "completion_tokens": 2}


def test_parse_from_str():
    parsed = parse_chat_completion(json.dumps(_completion("via string")))
    assert parsed.content == "via string"


def test_parse_from_bytes():
    parsed = parse_chat_completion(json.dumps(_completion("via bytes")).encode())
    assert parsed.content == "via bytes"


def test_content_list_parts_concatenated():
    obj = _completion()
    obj["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "part1 "},
        {"type": "image", "url": "ignored"},
        {"type": "text", "text": "part2"},
    ]
    parsed = parse_chat_completion(obj)
    assert parsed.content == "part1 part2"


def test_content_none_becomes_empty_string():
    obj = _completion()
    obj["choices"][0]["message"]["content"] = None
    assert parse_chat_completion(obj).content == ""


def test_missing_choices_raises():
    with pytest.raises(ChatCompletionParseError, match="choices"):
        parse_chat_completion({"id": "x"})


def test_empty_choices_raises():
    with pytest.raises(ChatCompletionParseError, match="choices"):
        parse_chat_completion({"choices": []})


def test_choice_not_object_raises():
    with pytest.raises(ChatCompletionParseError):
        parse_chat_completion({"choices": ["nope"]})


def test_missing_message_raises():
    with pytest.raises(ChatCompletionParseError, match="message"):
        parse_chat_completion({"choices": [{"finish_reason": "stop"}]})


def test_invalid_json_string_raises():
    with pytest.raises(ChatCompletionParseError, match="Invalid JSON"):
        parse_chat_completion("{not json")


def test_top_level_not_object_raises():
    with pytest.raises(ChatCompletionParseError):
        parse_chat_completion("[1, 2, 3]")


def test_unsupported_input_type_raises():
    with pytest.raises(ChatCompletionParseError, match="Unsupported"):
        parse_chat_completion(12345)  # type: ignore[arg-type]


def test_malformed_optional_fields_default_gracefully():
    obj = _completion()
    obj["id"] = 999  # not a str
    obj["model"] = None
    obj["created"] = "not-int"
    obj["usage"] = "not-a-dict"
    parsed = parse_chat_completion(obj)
    assert parsed.id is None
    assert parsed.model is None
    assert parsed.created is None
    assert parsed.usage == {}


# --- parse_json_strict -----------------------------------------------------


def test_parse_json_strict_ok():
    assert parse_json_strict('{"a": 1}') == {"a": 1}


def test_parse_json_strict_bad_json():
    with pytest.raises(ChatCompletionParseError, match="not valid JSON"):
        parse_json_strict("{bad")


def test_parse_json_strict_not_object():
    with pytest.raises(ChatCompletionParseError, match="object at top-level"):
        parse_json_strict("[1, 2]")
