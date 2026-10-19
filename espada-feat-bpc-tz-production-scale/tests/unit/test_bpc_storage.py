"""Tests for src/bpc/storage.py — atomic writes, dir setup, output loading."""
import json
import pytest
from pathlib import Path

from src.bpc.storage import (
    atomic_save_json,
    atomic_save_llama_factory_record,
    build_llama_factory_record,
    compile_llama_factory_jsonl,
    load_existing_output,
    parse_llama_factory_record,
    setup_output_dirs,
)


class TestAtomicSaveJson:
    def test_writes_valid_json(self, tmp_path):
        dest = tmp_path / "out.json"
        data = {"answer": [{"label": "car", "bbox_2d": [0, 0, 100, 80]}]}
        atomic_save_json(data, dest)
        assert dest.exists()
        loaded = json.loads(dest.read_text())
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "nested" / "deep" / "out.json"
        atomic_save_json({"x": 1}, dest)
        assert dest.exists()

    def test_overwrites_existing(self, tmp_path):
        dest = tmp_path / "out.json"
        atomic_save_json({"v": 1}, dest)
        atomic_save_json({"v": 2}, dest)
        assert json.loads(dest.read_text())["v"] == 2

    def test_unicode_content(self, tmp_path):
        dest = tmp_path / "out.json"
        data = {"label": "машина и велосипед"}
        atomic_save_json(data, dest)
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        assert loaded["label"] == "машина и велосипед"

    def test_no_tmp_file_left_on_success(self, tmp_path):
        dest = tmp_path / "out.json"
        atomic_save_json({"x": 1}, dest)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestSetupOutputDirs:
    def test_creates_required_dirs(self, tmp_path):
        dirs = setup_output_dirs(tmp_path, save_visualizations=True)
        assert (tmp_path / "outputs" / "bboxing").exists()
        assert (tmp_path / "outputs" / "pointing").exists()
        assert (tmp_path / "outputs" / "captions").exists()
        assert (tmp_path / "outputs" / "bboxing-judged").exists()
        assert (tmp_path / "outputs" / "pointing-judged").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "visualizations" / "boxes").exists()
        assert (tmp_path / "visualizations" / "points").exists()

    def test_no_viz_dirs_when_disabled(self, tmp_path):
        dirs = setup_output_dirs(tmp_path, save_visualizations=False)
        assert not (tmp_path / "visualizations").exists()

    def test_returns_path_dict(self, tmp_path):
        dirs = setup_output_dirs(tmp_path)
        assert "bboxing" in dirs
        assert "pointing" in dirs
        assert "captions" in dirs
        assert "logs" in dirs

    def test_idempotent(self, tmp_path):
        setup_output_dirs(tmp_path)
        setup_output_dirs(tmp_path)  # should not raise


class TestLoadExistingOutput:
    def test_returns_none_missing(self, tmp_path):
        assert load_existing_output(tmp_path / "nope.json") is None

    def test_returns_dict_valid(self, tmp_path):
        p = tmp_path / "out.json"
        p.write_text('{"answer": []}')
        assert load_existing_output(p) == {"answer": []}

    def test_returns_none_on_corrupt(self, tmp_path):
        p = tmp_path / "out.json"
        p.write_text("{bad json}")
        assert load_existing_output(p) is None

    def test_returns_none_on_empty(self, tmp_path):
        p = tmp_path / "out.json"
        p.write_text("   ")
        assert load_existing_output(p) is None


class TestLlamaFactoryStorage:
    def test_caption_record_matches_multimodal_format(self, tmp_path):
        image = tmp_path / "frame.jpg"
        result = {"question": "What is happening?", "answer": "A person is working."}

        record = build_llama_factory_record(result, image)

        assert record == {
            "messages": [
                {"role": "user", "content": "<image>\nWhat is happening?"},
                {"role": "assistant", "content": "A person is working."},
            ],
            "images": [str(image.resolve())],
        }

    def test_structured_answer_is_json_message_content(self, tmp_path):
        answer = [{"bbox_2d": [1, 2, 3, 4], "label": "чашка"}]
        record = build_llama_factory_record(
            {"question": "Locate the cup.", "answer": answer},
            tmp_path / "frame.jpg",
        )

        content = record["messages"][1]["content"]
        assert isinstance(content, str)
        assert json.loads(content) == answer
        assert parse_llama_factory_record(record)["answer"] == answer

    def test_atomic_record_is_one_jsonl_line_and_loadable(self, tmp_path):
        dest = tmp_path / "captions" / "frame.jsonl"
        result = {"question": "Question", "answer": "Answer"}
        atomic_save_llama_factory_record(result, tmp_path / "frame.jpg", dest)

        assert dest.read_text(encoding="utf-8").endswith("\n")
        assert len(dest.read_text(encoding="utf-8").splitlines()) == 1
        loaded = load_existing_output(dest)
        assert loaded is not None
        assert loaded["question"] == "Question"
        assert loaded["answer"] == "Answer"

    def test_compile_shards_and_writes_ready_dataset(self, tmp_path):
        source = tmp_path / "captions"
        atomic_save_llama_factory_record(
            {"question": "second", "answer": "b"}, tmp_path / "b.jpg", source / "b.jsonl"
        )
        atomic_save_llama_factory_record(
            {"question": "first", "answer": "a"}, tmp_path / "a.jpg", source / "a.jsonl"
        )

        dest = tmp_path / "captions.jsonl"
        assert compile_llama_factory_jsonl(source, dest) == 2
        records = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
        questions = {record["messages"][0]["content"] for record in records}
        assert questions == {"<image>\nfirst", "<image>\nsecond"}
