"""Atomic file I/O and output directory setup for BPC pipeline."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


LLAMA_FACTORY_IMAGE_TOKEN = "<image>"


def atomic_save_json(data: dict[str, Any], dest: Path) -> None:
    """Writes JSON atomically: write to temp file, then os.replace to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def build_llama_factory_record(
    result: dict[str, Any], image_path: str | Path
) -> dict[str, Any]:
    """Converts an internal annotation to LLaMA Factory multimodal format.

    The repository uses ``question``/``answer`` internally for validation and
    Judge processing.  LLaMA Factory expects ShareGPT-style ``messages`` and a
    separate ``images`` list.  Structured answers (boxes and points) are
    serialized as JSON text because message content must be a string.
    """
    question = result.get("question")
    answer = result.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Annotation 'question' must be a non-empty string.")
    if answer is None:
        raise ValueError("Annotation 'answer' is missing.")

    if isinstance(answer, str):
        assistant_content = answer
    else:
        assistant_content = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))

    resolved_image = str(Path(image_path).expanduser().resolve())
    return {
        "messages": [
            {
                "role": "user",
                "content": f"{LLAMA_FACTORY_IMAGE_TOKEN}\n{question.strip()}",
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [resolved_image],
    }


def parse_llama_factory_record(record: dict[str, Any]) -> dict[str, Any]:
    """Converts one LLaMA Factory record back to the internal representation."""
    messages = record.get("messages")
    images = record.get("images")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("LLaMA Factory record must contain exactly two messages.")
    user, assistant = messages
    if not isinstance(user, dict) or user.get("role") != "user":
        raise ValueError("First LLaMA Factory message must have role 'user'.")
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        raise ValueError("Second LLaMA Factory message must have role 'assistant'.")

    user_content = user.get("content")
    assistant_content = assistant.get("content")
    if not isinstance(user_content, str) or not isinstance(assistant_content, str):
        raise ValueError("LLaMA Factory message content must be a string.")
    prefix = f"{LLAMA_FACTORY_IMAGE_TOKEN}\n"
    if not user_content.startswith(prefix):
        raise ValueError(f"User message must start with {LLAMA_FACTORY_IMAGE_TOKEN!r}.")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError("LLaMA Factory record must contain exactly one image path.")

    answer: Any = assistant_content
    stripped_answer = assistant_content.strip()
    if stripped_answer.startswith(("[", "{")):
        try:
            answer = json.loads(stripped_answer)
        except json.JSONDecodeError:
            # A free-form caption is allowed to begin with a bracket/brace.
            answer = assistant_content

    return {
        "images": list(images),
        "question": user_content[len(prefix) :],
        "answer": answer,
    }


def atomic_save_llama_factory_record(
    result: dict[str, Any], image_path: str | Path, dest: Path
) -> None:
    """Atomically writes one compact LLaMA Factory JSONL record."""
    record = build_llama_factory_record(result, image_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def compile_llama_factory_jsonl(source_dir: Path, dest: Path) -> int:
    """Streams per-image JSONL shards into one LLaMA Factory dataset.

    The iterator is intentionally not materialized: production runs may hold
    millions of shards, so sorting every path in memory would make final export
    needlessly expensive.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for source in source_dir.glob("*.jsonl"):
                lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
                if len(lines) != 1:
                    raise ValueError(f"Expected one JSONL record in {source}, got {len(lines)}.")
                record = json.loads(lines[0])
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object in {source}.")
                parse_llama_factory_record(record)
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                out.write("\n")
                count += 1
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return count


def finalize_llama_factory_outputs(dirs: dict[str, Path]) -> dict[str, int]:
    """Builds one ready-to-train JSONL file next to every output task directory."""
    counts: dict[str, int] = {}
    for task_name, source_dir in dirs.items():
        if task_name in {"logs", "vis_boxes", "vis_points"}:
            continue
        output_name = task_name.replace("_judged", "-judged")
        counts[output_name] = compile_llama_factory_jsonl(
            source_dir, source_dir.parent / f"{output_name}.jsonl"
        )
    return counts


def setup_output_dirs(output_root: Path, *, save_visualizations: bool = True) -> dict[str, Path]:
    """Creates all required output subdirectories and returns a path map."""
    dirs: dict[str, Path] = {
        "bboxing": output_root / "outputs" / "bboxing",
        "pointing": output_root / "outputs" / "pointing",
        "captions": output_root / "outputs" / "captions",
        "bboxing_judged": output_root / "outputs" / "bboxing-judged",
        "pointing_judged": output_root / "outputs" / "pointing-judged",
        "logs": output_root / "logs",
    }
    if save_visualizations:
        dirs["vis_boxes"] = output_root / "visualizations" / "boxes"
        dirs["vis_points"] = output_root / "visualizations" / "points"

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def load_existing_output(path: Path) -> dict[str, Any] | None:
    """Reads a LLaMA Factory shard or legacy JSON output into internal form."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        if "messages" in data:
            return parse_llama_factory_record(data)
        return data
    except (json.JSONDecodeError, OSError, ValueError):
        return None
