"""Unit tests for prompt YAML loading and template rendering."""

from __future__ import annotations

import textwrap

import pytest

from src.vlm.prompts_processing import (
    get_by_path,
    get_prompt_block,
    get_system_user_prompts,
    load_yaml,
    render_template,
)

pytestmark = pytest.mark.unit


# --- load_yaml -------------------------------------------------------------


def test_load_yaml_dict(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  b: 1\n", encoding="utf-8")
    assert load_yaml(p) == {"a": {"b": 1}}


def test_load_yaml_non_mapping_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(p)


# --- get_by_path -----------------------------------------------------------


def test_get_by_path_nested():
    assert get_by_path({"a": {"b": {"c": 42}}}, "a.b.c") == 42


def test_get_by_path_missing():
    with pytest.raises(KeyError):
        get_by_path({"a": {}}, "a.b")


# --- render_template -------------------------------------------------------


def test_render_simple_substitution():
    assert render_template("Hello {{ name }}!", {"name": "world"}) == "Hello world!"


def test_render_dotted_key():
    assert render_template("{{ a.b }}", {"a": {"b": "X"}}) == "X"


def test_render_none_becomes_empty():
    assert render_template("[{{ x }}]", {"x": None}) == "[]"


def test_render_dict_serialized_as_json():
    out = render_template("{{ d }}", {"d": {"k": 1}})
    assert out == '{"k": 1}'


def test_render_list_serialized_as_json():
    out = render_template("{{ d }}", {"d": [1, 2]})
    assert out == "[1, 2]"


def test_render_missing_key_raises():
    with pytest.raises(KeyError, match="Missing template key"):
        render_template("{{ nope }}", {})


def test_render_numbers_and_bools():
    assert render_template("{{ n }}/{{ b }}", {"n": 3, "b": True}) == "3/True"


# --- get_prompt_block ------------------------------------------------------


def _prompts_yaml_dict():
    return {
        "vlm": {
            "bboxing": {"system": "sys {{ bbox_dict }}", "user": "usr"},
            "pointing": {"system": "sys", "user": "pt {{ points_dict }}"},
            "general_captioning": {"system": "cap sys", "user": "cap usr"},
        }
    }


def test_get_prompt_block_ok():
    block = get_prompt_block(_prompts_yaml_dict(), "vlm.general_captioning")
    assert block == {"system": "cap sys", "user": "cap usr"}


def test_get_prompt_block_not_mapping():
    with pytest.raises(TypeError):
        get_prompt_block({"vlm": {"x": "string"}}, "vlm.x")


def test_get_prompt_block_missing_string_fields():
    with pytest.raises(TypeError):
        get_prompt_block({"vlm": {"x": {"system": 1, "user": "u"}}}, "vlm.x")


# --- get_system_user_prompts ----------------------------------------------


def test_get_system_user_prompts_captioning():
    sys, usr = get_system_user_prompts(_prompts_yaml_dict(), "general_captioning")
    assert sys == "cap sys"
    assert usr == "cap usr"


def test_get_system_user_prompts_bboxing_renders_meta():
    sys, usr = get_system_user_prompts(_prompts_yaml_dict(), "bboxing", {1: [0, 0, 5, 5]})
    assert '"1"' in sys or "1" in sys  # bbox_dict serialized into system prompt


def test_get_system_user_prompts_bboxing_requires_meta():
    with pytest.raises(ValueError, match="detections"):
        get_system_user_prompts(_prompts_yaml_dict(), "bboxing")


def test_get_system_user_prompts_pointing_requires_meta():
    with pytest.raises(ValueError, match="pointing"):
        get_system_user_prompts(_prompts_yaml_dict(), "pointing")


def test_get_system_user_prompts_invalid_task():
    with pytest.raises(AssertionError):
        get_system_user_prompts(_prompts_yaml_dict(), "bogus")


def test_real_prompts_yaml_loads_and_renders():
    """The shipped configs/prompts.yaml must render for all three tasks."""
    from pathlib import Path

    prompts_path = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"
    prompts = load_yaml(prompts_path)

    # captioning needs no meta
    get_system_user_prompts(prompts, "general_captioning")
    # the other two need meta dicts
    get_system_user_prompts(prompts, "bboxing", {1: [0, 0, 10, 10]})
    get_system_user_prompts(prompts, "pointing", {"id": [1], "points": [[5, 5]]})
