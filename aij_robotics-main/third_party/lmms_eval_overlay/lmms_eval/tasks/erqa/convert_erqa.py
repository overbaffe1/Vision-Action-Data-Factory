#!/usr/bin/env python3
"""
Offline converter: ERQA TFRecord -> JSONL + image files.

TensorFlow is required only for this script, not for lmms-eval runtime evaluation.

Example:
    python lmms_eval/tasks/erqa/convert_erqa.py \\
        --input /path/to/erqa.tfrecord \\
        --output_dir /path/to/erqa_converted
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "pip install tensorflow or tensorflow-cpu is required to convert ERQA TFRecord."
        ) from exc
    return tf


def _parse_example(example_proto, tf):
    feature_description = {
        "answer": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.VarLenFeature(tf.string),
        "question_type": tf.io.VarLenFeature(tf.string),
        "visual_indices": tf.io.VarLenFeature(tf.int64),
        "question": tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    parsed["visual_indices"] = tf.sparse.to_dense(parsed["visual_indices"])
    parsed["image/encoded"] = tf.sparse.to_dense(parsed["image/encoded"])
    parsed["question_type"] = tf.sparse.to_dense(parsed["question_type"])
    return parsed


def _decode_str(value: Any) -> str:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_question_type(values: Any) -> str:
    if hasattr(values, "numpy"):
        values = values.numpy()
    if values is None or len(values) == 0:
        return ""
    return _decode_str(values[0])


def _decode_visual_indices(values: Any) -> list[int]:
    if hasattr(values, "numpy"):
        values = values.numpy()
    return [int(v) for v in list(values)]


def _save_image(image_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        suffix = output_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            image.save(output_path, format="JPEG", quality=95)
        else:
            image.save(output_path, format="PNG")


def convert_tfrecord(input_path: Path, output_dir: Path, *, jsonl_name: str = "erqa.jsonl") -> int:
    tf = _require_tensorflow()

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    jsonl_path = output_dir / jsonl_name

    if not input_path.is_file():
        raise FileNotFoundError(f"TFRecord not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = tf.data.TFRecordDataset(str(input_path))
    rows: list[dict[str, Any]] = []

    for index, example_proto in enumerate(dataset):
        example = _parse_example(example_proto, tf)
        example_id = f"erqa_{index:06d}"
        question = _decode_str(example["question"])
        answer = _decode_str(example["answer"]).strip().upper()[:1]
        question_type = _decode_question_type(example["question_type"])
        visual_indices = _decode_visual_indices(example["visual_indices"])

        images_encoded = example["image/encoded"].numpy()
        image_paths: list[str] = []
        for image_index, image_bytes in enumerate(images_encoded):
            rel_path = Path("images") / f"{example_id}_{image_index:02d}.png"
            abs_path = output_dir / rel_path
            _save_image(bytes(image_bytes), abs_path)
            image_paths.append(rel_path.as_posix())

        rows.append(
            {
                "id": example_id,
                "question": question,
                "answer": answer,
                "question_type": question_type,
                "visual_indices": visual_indices,
                "image_paths": image_paths,
            }
        )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert ERQA TFRecord to JSONL + images for lmms-eval.")
    parser.add_argument("--input", required=True, type=Path, help="Path to official erqa.tfrecord")
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory (writes erqa.jsonl and images/)",
    )
    parser.add_argument(
        "--jsonl_name",
        default="erqa.jsonl",
        help="JSONL filename inside output_dir (default: erqa.jsonl)",
    )
    args = parser.parse_args(argv)

    count = convert_tfrecord(args.input, args.output_dir, jsonl_name=args.jsonl_name)
    print(f"Converted {count} examples to {args.output_dir.resolve()}")
    print(f"  JSONL: {args.output_dir / args.jsonl_name}")
    print(f"  Images: {args.output_dir / 'images'}")
    print(f"Export ERQA_DATA_DIR={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
