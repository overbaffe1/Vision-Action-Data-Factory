from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


_PLACEHOLDER_RE = re.compile(r"{{\s*([a-zA-Z_][\w\.]*)\s*}}")


def load_yaml(path: str | Path) -> dict:
    """Loads a YAML file into a nested Python dictionary."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict).")
    return data


def get_by_path(data: dict[str, object], path: str) -> object:
    """Retrieves a value from a nested dict using dot-separated path."""
    cur: object = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def render_template(template: str, context: dict[str, object]) -> str:
    """Renders {{ placeholders }} in a template string using provided context."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            value = get_by_path(context, key)
        except KeyError as e:
            raise KeyError(f"Missing template key: {key}") from e

        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return _PLACEHOLDER_RE.sub(replace, template)


def get_prompt_block(prompts: dict[str, object], block_path: str) -> dict[str, str]:
    """Returns a prompt block (system/user) from the loaded YAML by dot-path."""
    block = get_by_path(prompts, block_path)
    if not isinstance(block, dict):
        raise TypeError(f"Prompt block must be a mapping: {block_path}")

    system = block.get("system")
    user = block.get("user")
    if not isinstance(system, str) or not isinstance(user, str):
        raise TypeError(f"Prompt block must contain string 'system' and 'user': {block_path}")

    return {"system": system, "user": user}


def get_system_user_prompts(
    prompts: dict[str, object], task_type: str, meta_info=None
) -> tuple[str, str]:
    """Aggregate get_prompt_block and render_template functions."""

    assert task_type in [
        "bboxing",
        "pointing",
        "general_captioning",
    ], "Choose an apropriate type of task!"

    if task_type == "bboxing":
        if meta_info is None:
            raise ValueError("Put detections info in meta_info!")
        ctx = {"bbox_dict": meta_info}

    elif task_type == "pointing":
        if meta_info is None:
            raise ValueError("Put pointing info in meta_info!")
        ctx = {"points_dict": meta_info}

    else:
        ctx = {}

    block = get_prompt_block(prompts, f"vlm.{task_type}")
    system_prompt = render_template(block["system"], ctx)
    user_prompt = render_template(block["user"], ctx)

    return system_prompt, user_prompt
