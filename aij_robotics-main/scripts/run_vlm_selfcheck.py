#!/usr/bin/env python3
"""Run official 8-bench lmms-eval suite on submission/smolvlm2 (participant self-check)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _pin_erqa_data_files(lmms_root: Path, erqa_dir: Path) -> None:
    """Rewrite erqa YAML data_files to an absolute jsonl path.

    Public lmms-eval resolves relative ``data_files`` against the task YAML dir
    *before* shell ``${ERQA_DATA_DIR}`` expansion, which breaks the fork template.
    """
    template = lmms_root / "lmms_eval" / "tasks" / "erqa" / "_default_template_yaml"
    if not template.is_file():
        return
    jsonl = erqa_dir.expanduser().resolve() / "erqa.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"ERQA jsonl not found: {jsonl}")
    text = template.read_text(encoding="utf-8")
    pinned = jsonl.as_posix()
    if f"train: {pinned}" in text:
        return
    if "${ERQA_DATA_DIR}/erqa.jsonl" in text:
        text = text.replace("${ERQA_DATA_DIR}/erqa.jsonl", pinned)
    else:
        # Replace whatever train path is currently set.
        lines = []
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith("train:"):
                indent = line[: len(line) - len(line.lstrip())]
                line = f"{indent}train: {pinned}\n"
            lines.append(line)
        text = "".join(lines)
    template.write_text(text, encoding="utf-8")
    print(f"[vlm] pinned ERQA data_files → {pinned}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-checkpoint", type=Path, required=True)
    parser.add_argument("--lmms-eval", type=Path, required=True, help="Path to lmms-eval repo root")
    parser.add_argument("--python", type=Path, required=True, help="Python with lmms-eval installed")
    parser.add_argument("--suite", type=Path, required=True, help="greenvla_vlm_tasks.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--model", default="smolvlm2")
    parser.add_argument("--erqa-data-dir", type=Path, default=None)
    parser.add_argument("--exclude-erqa", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    vlm = args.vlm_checkpoint.expanduser().resolve()
    lmms = args.lmms_eval.expanduser().resolve()
    suite = args.suite.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    # Never resolve() the interpreter: venv/bin/python → base CPython drops site-packages.
    py = args.python.expanduser()
    if not py.is_absolute():
        py = Path.cwd() / py
    py = Path(os.path.normpath(str(py)))
    if not py.is_file() and py.with_name("python3").is_file():
        py = py.with_name("python3")
    run_many = lmms / "scripts" / "run_many_tasks.py"

    if not vlm.is_dir():
        raise FileNotFoundError(f"VLM checkpoint not found: {vlm}")
    if not run_many.is_file():
        raise FileNotFoundError(f"run_many_tasks.py not found: {run_many}")
    if not suite.is_file():
        raise FileNotFoundError(f"Suite YAML not found: {suite}")
    if not py.is_file():
        raise FileNotFoundError(f"Python not found: {py}")

    out.mkdir(parents=True, exist_ok=True)
    model_args = f"pretrained={vlm},device_map=auto,torch_dtype=bfloat16"

    cmd = [
        str(py),
        str(run_many),
        "--suite",
        str(suite),
        "--model",
        args.model,
        "--model_args",
        model_args,
        "--output_dir",
        str(out),
        "--gpus",
        args.gpus,
        "--batch_size",
        str(args.batch_size),
        "--resume",
        "--continue_on_error",
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.exclude_erqa:
        cmd.extend(["--exclude_tasks", "erqa"])

    env = os.environ.copy()
    env.setdefault("API_TYPE", "dummy")
    env.setdefault("PYTHONNOUSERSITE", "1")
    if args.erqa_data_dir is not None:
        erqa_dir = args.erqa_data_dir.expanduser().resolve()
        env["ERQA_DATA_DIR"] = str(erqa_dir)
        _pin_erqa_data_files(lmms, erqa_dir)

    print("[vlm] cwd=", lmms)
    print("[vlm]", " ".join(cmd))
    if args.dry_run:
        print("[vlm] dry-run: not executing")
        return 0

    proc = subprocess.run(cmd, cwd=str(lmms), env=env, check=False)
    if proc.returncode != 0:
        print(f"[vlm] run_many_tasks exited with {proc.returncode}", file=sys.stderr)
        return int(proc.returncode)
    print(f"[vlm] done → {out}")
    summary = out / "summary"
    for name in ("suite_status.json", "report.md", "results_merged.csv"):
        path = summary / name
        if path.is_file():
            print(f"[vlm] {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
