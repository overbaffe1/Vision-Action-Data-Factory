import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatCompletionParsed:
    id: str | None
    model: str | None
    created: int | None
    content: str
    role: str | None
    finish_reason: str | None
    usage: dict[str, object]
    raw: dict[str, object]


class ChatCompletionParseError(Exception):
    pass


def parse_chat_completion(
    data: str | bytes | dict[str, object],
) -> ChatCompletionParsed:
    """Parses a chat completion response into a normalized structure."""

    obj = _coerce_json_object(data)

    choice0 = _first_choice(obj)
    message = _get_dict(choice0, "message", path="choices[0].message")

    content = _extract_message_content(message)

    raw_id = obj.get("id")
    chat_id: str | None = raw_id if isinstance(raw_id, str) else None

    raw_model = obj.get("model")
    model: str | None = raw_model if isinstance(raw_model, str) else None

    raw_created = obj.get("created")
    created: int | None = raw_created if isinstance(raw_created, int) else None

    raw_role = message.get("role")
    role: str | None = raw_role if isinstance(raw_role, str) else None

    raw_finish_reason = choice0.get("finish_reason")
    finish_reason: str | None = raw_finish_reason if isinstance(raw_finish_reason, str) else None

    raw_usage = obj.get("usage")
    usage: dict[str, object]
    if isinstance(raw_usage, dict) and all(isinstance(k, str) for k in raw_usage):
        usage = dict(raw_usage)
    else:
        usage = {}

    return ChatCompletionParsed(
        id=chat_id,
        model=model,
        created=created,
        content=content,
        role=role,
        finish_reason=finish_reason,
        usage=usage,
        raw=obj,
    )


def parse_json_strict(text: str) -> dict[str, object]:
    """Parses a JSON object strictly and raises a readable error on failures."""

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        excerpt = text[:500].replace("\n", "\\n")
        raise ChatCompletionParseError(
            f"Model output is not valid JSON: {e}. Excerpt: {excerpt}"
        ) from e

    if not isinstance(obj, dict):
        raise ChatCompletionParseError(
            f"Expected a JSON object at top-level, got {type(obj).__name__}"
        )

    return obj


def _coerce_json_object(data: str | bytes | dict[str, object]) -> dict[str, object]:
    if isinstance(data, dict):
        return data

    if isinstance(data, (str, bytes)):
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as e:
            raise ChatCompletionParseError(f"Invalid JSON: {e}") from e

        if not isinstance(obj, dict):
            raise ChatCompletionParseError("Top-level JSON must be an object.")

        return obj

    raise ChatCompletionParseError(f"Unsupported input type: {type(data)!r}")


def _first_choice(obj: dict[str, object]) -> dict[str, object]:
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ChatCompletionParseError("Missing or empty 'choices'.")
    if not isinstance(choices[0], dict):
        raise ChatCompletionParseError("'choices[0]' is not an object.")
    return choices[0]


def _get_dict(container: dict[str, object], key: str, *, path: str) -> dict[str, object]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ChatCompletionParseError(f"Missing or invalid '{path}'.")
    return value


def _extract_message_content(message: dict[str, object]) -> str:
    content = message.get("content")

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "".join(texts)

    raise ChatCompletionParseError("'choices[0].message.content' has an unsupported type.")
