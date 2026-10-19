#!/usr/bin/env python3
"""
Launch multiple existing lmms-eval tasks from a YAML task list.

This is orchestration only — it does NOT register a combined benchmark task.
Each task runs as its own `python -m lmms_eval` subprocess with separate outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Subprocess I/O (live progress in terminal + log files)
# ---------------------------------------------------------------------------


def _stream_pipe(
    pipe,
    log_file,
    console_stream,
    prefix: str,
    mirror_console: bool,
) -> None:
    try:
        for line in pipe:
            log_file.write(line)
            log_file.flush()
            if mirror_console:
                console_stream.write(f"{prefix}{line}")
                console_stream.flush()
    finally:
        pipe.close()


def run_subprocess_with_logs(
    cmd: list[str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    label: str | None = None,
    timeout_sec: int | None = None,
) -> int:
    """Run subprocess; write logs to disk and mirror stdout/stderr to the parent terminal."""
    prefix = f"[{label}] " if label else ""
    # Default ON so progress is visible under `| tee` (isatty() is False for pipes).
    # Disable with LMMS_EVAL_MIRROR_CONSOLE=0.
    mirror_env = os.environ.get("LMMS_EVAL_MIRROR_CONSOLE", "1").lower()
    mirror = mirror_env not in ("0", "false", "no")
    env = dict(env)
    env.setdefault("LMMS_EVAL_PROGRESS_INTERVAL", os.environ.get("LMMS_EVAL_PROGRESS_INTERVAL", "30"))
    # Child lmms-eval stdout/stderr are pipes → force line progress instead of broken tqdm CR bars.
    env.setdefault("LMMS_EVAL_FORCE_LOG_PROGRESS", "1")

    with (
        stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_f,
        stderr_path.open("w", encoding="utf-8", buffering=1) as stderr_f,
    ):
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threads = [
            threading.Thread(
                target=_stream_pipe,
                args=(proc.stdout, stdout_f, sys.stdout, prefix, mirror),
                daemon=True,
            ),
            threading.Thread(
                target=_stream_pipe,
                args=(proc.stderr, stderr_f, sys.stderr, prefix, mirror),
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
        try:
            return_code = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            for t in threads:
                t.join(timeout=10)
        return return_code if return_code is not None else -1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# Pointing tasks score against doc["mask"]; lmms-eval drops Image columns unless --process_with_media.
DEFAULT_TASKS_PROCESS_WITH_MEDIA = frozenset({"refspatial", "where2place"})


@dataclass
class TaskRunConfig:
    suite_name: str
    task_label: str  # name from YAML list (e.g. crosspoint)
    lmms_task: str  # registered lmms-eval task id
    batch_size: int
    num_fewshot: int
    log_samples: bool
    limit: float | int | None
    timeout_sec: int | None = None
    process_with_media: bool = False


@dataclass
class SuiteConfig:
    suite_name: str
    tasks: list[str]
    task_aliases: dict[str, str]
    defaults: dict[str, Any]
    task_overrides: dict[str, dict[str, Any]]
    env: dict[str, str]
    tasks_process_with_media: frozenset[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand_env_string(value: str) -> str:
    return os.path.expandvars(str(value).strip())


def load_suite_config(path: Path) -> SuiteConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    media_tasks = raw.get("tasks_process_with_media")
    if media_tasks is None:
        media_set = DEFAULT_TASKS_PROCESS_WITH_MEDIA
    else:
        media_set = frozenset(str(t) for t in media_tasks)
    raw_env = raw.get("env") or {}
    return SuiteConfig(
        suite_name=str(raw.get("suite_name", path.stem)),
        tasks=list(raw.get("tasks") or []),
        task_aliases=dict(raw.get("task_aliases") or {}),
        defaults=dict(raw.get("default") or raw.get("defaults") or {}),
        task_overrides=dict(raw.get("task_overrides") or {}),
        env={str(k): expand_env_string(v) for k, v in raw_env.items()},
        tasks_process_with_media=media_set,
    )


def build_subprocess_env(suite_env: dict[str, str], gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for key, value in suite_env.items():
        if value:
            env[key] = value
    return env


def validate_suite_prerequisites(task_labels: list[str], suite_env: dict[str, str]) -> None:
    if "erqa" not in task_labels:
        return
    erqa_dir = (suite_env.get("ERQA_DATA_DIR") or os.environ.get("ERQA_DATA_DIR", "")).strip()
    if not erqa_dir:
        raise SystemExit(
            "ERQA is in the task list but ERQA_DATA_DIR is not set.\n"
            "Convert data: python lmms_eval/tasks/erqa/convert_erqa.py --input /path/to/erqa.tfrecord "
            "--output_dir /path/to/erqa_converted\n"
            "Then: export ERQA_DATA_DIR=/path/to/erqa_converted\n"
            "Or set env.ERQA_DATA_DIR in the suite YAML / scripts/run_greenvla_vlm_tasks.sh"
        )
    jsonl = Path(os.path.expanduser(erqa_dir)) / "erqa.jsonl"
    if not jsonl.is_file():
        raise SystemExit(f"ERQA_DATA_DIR does not contain erqa.jsonl: {jsonl}")


def resolve_lmms_task(task_label: str, aliases: dict[str, str]) -> str:
    return aliases.get(task_label, task_label)


def merge_task_config(
    suite: SuiteConfig,
    task_label: str,
    cli_batch_size: int | None,
    cli_limit: float | int | None,
    cli_num_fewshot: int | None,
    cli_log_samples: bool | None,
) -> TaskRunConfig:
    merged: dict[str, Any] = dict(suite.defaults)
    merged.update(suite.task_overrides.get(task_label, {}))

    batch_size = cli_batch_size if cli_batch_size is not None else int(merged.get("batch_size", 1))
    num_fewshot = cli_num_fewshot if cli_num_fewshot is not None else int(merged.get("num_fewshot", 0))
    if cli_log_samples is not None:
        log_samples = cli_log_samples
    else:
        log_samples = bool(merged.get("log_samples", True))

    limit = cli_limit if cli_limit is not None else merged.get("limit")
    timeout = merged.get("timeout_sec")
    timeout_sec = int(timeout) if timeout is not None else None

    process_with_media = bool(merged.get("process_with_media", task_label in suite.tasks_process_with_media))

    return TaskRunConfig(
        suite_name=suite.suite_name,
        task_label=task_label,
        lmms_task=resolve_lmms_task(task_label, suite.task_aliases),
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        log_samples=log_samples,
        limit=limit,
        timeout_sec=timeout_sec,
        process_with_media=process_with_media,
    )


def filter_tasks(
    suite: SuiteConfig,
    only: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    tasks = list(suite.tasks)
    if only:
        unknown = set(only) - set(tasks)
        if unknown:
            raise ValueError(f"--tasks not in suite: {sorted(unknown)}")
        tasks = [t for t in only if t in set(tasks)]
    if exclude:
        tasks = [t for t in tasks if t not in exclude]
    return tasks


# ---------------------------------------------------------------------------
# Status / resume
# ---------------------------------------------------------------------------


def read_status(task_dir: Path) -> dict[str, Any] | None:
    path = task_dir / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def should_skip_task(task_dir: Path, resume: bool) -> bool:
    if not resume:
        return False
    status = read_status(task_dir)
    if status is None:
        return False
    return status.get("status") == "success"


def write_status(task_dir: Path, payload: dict[str, Any]) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# lmms-eval command
# ---------------------------------------------------------------------------


def build_lmms_eval_command(
    cfg: TaskRunConfig,
    model: str,
    model_args: str,
    output_path: Path,
    python_executable: str,
) -> list[str]:
    cmd = [
        python_executable,
        "-m",
        "lmms_eval",
        "--model",
        model,
        "--model_args",
        model_args,
        "--tasks",
        cfg.lmms_task,
        "--batch_size",
        str(cfg.batch_size),
        "--output_path",
        str(output_path),
        "--num_fewshot",
        str(cfg.num_fewshot),
    ]
    if cfg.log_samples:
        cmd.append("--log_samples")
    if cfg.limit is not None:
        cmd.extend(["--limit", str(cfg.limit)])
    if cfg.process_with_media:
        cmd.append("--process_with_media")
    return cmd


def find_latest_results_json(search_dir: Path) -> Path | None:
    candidates = sorted(search_dir.rglob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def copy_results_artifact(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Metrics parsing / merge
# ---------------------------------------------------------------------------


def flatten_task_metrics(results_data: dict[str, Any], task_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_block = (results_data.get("results") or {}).get(task_key) or {}
    for key, value in task_block.items():
        if key in ("alias",) or key.endswith("_stderr") or key.endswith("_stderr_clt") or key.endswith("_stderr_clustered"):
            continue
        if not key.endswith(",none"):
            continue
        metric = key[: -len(",none")]
        if not isinstance(value, (int, float)):
            continue
        rows.append(
            {
                "metric": metric,
                "value": float(value),
                "higher_is_better": None,
            }
        )
    return rows


def parse_result_file(result_path: Path, lmms_task: str) -> list[dict[str, Any]]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    rows = flatten_task_metrics(data, lmms_task)
    if rows:
        return rows
    # Group tasks (e.g. refspatial) may use group name as key
    for key in (results_data := data.get("results") or {}):
        rows.extend(flatten_task_metrics(data, key))
    return rows


# ---------------------------------------------------------------------------
# Single task execution
# ---------------------------------------------------------------------------


@dataclass
class TaskOutcome:
    task_label: str
    lmms_task: str
    gpu: str
    status: str
    return_code: int | None
    duration_sec: float
    command: str
    result_file: str | None
    error_summary: str | None
    metrics: list[dict[str, Any]] = field(default_factory=list)


def run_one_task(
    cfg: TaskRunConfig,
    model: str,
    model_args: str,
    output_dir: Path,
    gpu: str,
    python_executable: str,
    suite_env: dict[str, str] | None = None,
) -> TaskOutcome:
    task_dir = output_dir / "tasks" / cfg.task_label
    task_dir.mkdir(parents=True, exist_ok=True)
    lmms_output = task_dir / "lmms_eval_output"
    lmms_output.mkdir(parents=True, exist_ok=True)

    cmd = build_lmms_eval_command(cfg, model, model_args, lmms_output, python_executable)
    cmd_str = " ".join(cmd)
    env = build_subprocess_env(suite_env or {}, gpu)

    (task_dir / "command.txt").write_text(cmd_str + "\n", encoding="utf-8")

    start = time.time()
    start_iso = _utc_now_iso()
    write_status(
        task_dir,
        {
            "task": cfg.task_label,
            "lmms_task": cfg.lmms_task,
            "status": "running",
            "start_time": start_iso,
            "end_time": None,
            "duration_sec": None,
            "gpu": gpu,
            "command": cmd_str,
            "return_code": None,
            "result_file": None,
            "error_summary": None,
        },
    )

    stdout_path = task_dir / "stdout.log"
    stderr_path = task_dir / "stderr.log"
    return_code = -1
    error_summary = None
    result_file = None
    metrics: list[dict[str, Any]] = []

    try:
        return_code = run_subprocess_with_logs(
            cmd,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            label=cfg.task_label,
            timeout_sec=cfg.timeout_sec,
        )
        if return_code != 0:
            error_summary = f"lmms-eval exited with code {return_code}"
        else:
            found = find_latest_results_json(lmms_output)
            if found:
                dest = task_dir / "results.json"
                copy_results_artifact(found, dest)
                result_file = str(dest)
                try:
                    metrics = parse_result_file(dest, cfg.lmms_task)
                except (json.JSONDecodeError, OSError) as exc:
                    error_summary = f"Could not parse results: {exc}"
                    return_code = 1
            else:
                error_summary = "lmms-eval finished but no *_results.json found"
                return_code = 1
    except subprocess.TimeoutExpired:
        error_summary = f"Timeout after {cfg.timeout_sec}s"
        return_code = 124
    except OSError as exc:
        error_summary = str(exc)
        return_code = 1

    duration = time.time() - start
    status = "success" if return_code == 0 and error_summary is None else "failed"

    write_status(
        task_dir,
        {
            "task": cfg.task_label,
            "lmms_task": cfg.lmms_task,
            "status": status,
            "start_time": start_iso,
            "end_time": _utc_now_iso(),
            "duration_sec": round(duration, 3),
            "gpu": gpu,
            "command": cmd_str,
            "return_code": return_code,
            "result_file": result_file,
            "error_summary": error_summary,
        },
    )

    return TaskOutcome(
        task_label=cfg.task_label,
        lmms_task=cfg.lmms_task,
        gpu=gpu,
        status=status,
        return_code=return_code,
        duration_sec=duration,
        command=cmd_str,
        result_file=result_file,
        error_summary=error_summary,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def run_suite(
    suite_path: Path,
    model: str,
    model_args: str,
    output_dir: Path,
    gpus: list[str],
    cli_batch_size: int | None,
    cli_limit: float | int | None,
    cli_num_fewshot: int | None,
    cli_log_samples: bool | None,
    only_tasks: list[str] | None,
    exclude_tasks: list[str] | None,
    resume: bool,
    dry_run: bool,
    continue_on_error: bool,
    python_executable: str,
) -> int:
    suite = load_suite_config(suite_path)
    task_labels = filter_tasks(suite, only_tasks, exclude_tasks)
    if not task_labels:
        print("No tasks to run.", file=sys.stderr)
        return 1

    if not dry_run:
        validate_suite_prerequisites(task_labels, suite.env)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary"
    if not dry_run:
        summary_dir.mkdir(parents=True, exist_ok=True)

    run_started = _utc_now_iso()
    t0 = time.time()

    configs = [
        merge_task_config(suite, label, cli_batch_size, cli_limit, cli_num_fewshot, cli_log_samples)
        for label in task_labels
    ]

    resolved_config = {
        "suite_path": str(suite_path.resolve()),
        "suite_name": suite.suite_name,
        "model": model,
        "model_args": model_args,
        "output_dir": str(output_dir.resolve()),
        "gpus": gpus,
        "resume": resume,
        "continue_on_error": continue_on_error,
        "dry_run": dry_run,
        "tasks": [c.task_label for c in configs],
        "suite_env": suite.env,
        "tasks_process_with_media": sorted(suite.tasks_process_with_media),
        "started_at": run_started,
    }
    if not dry_run:
        (summary_dir / "run_config_resolved.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    # Build runnable queue (respect resume)
    pending: list[TaskRunConfig] = []
    skipped_outcomes: list[TaskOutcome] = []
    for cfg in configs:
        if should_skip_task(output_dir / "tasks" / cfg.task_label, resume):
            prev = read_status(output_dir / "tasks" / cfg.task_label) or {}
            skipped_outcomes.append(
                TaskOutcome(
                    task_label=cfg.task_label,
                    lmms_task=cfg.lmms_task,
                    gpu=str(prev.get("gpu", "")),
                    status="skipped",
                    return_code=prev.get("return_code"),
                    duration_sec=float(prev.get("duration_sec") or 0),
                    command=str(prev.get("command", "")),
                    result_file=prev.get("result_file"),
                    error_summary=None,
                    metrics=[],
                )
            )
            if (output_dir / "tasks" / cfg.task_label / "results.json").exists():
                try:
                    skipped_outcomes[-1].metrics = parse_result_file(
                        output_dir / "tasks" / cfg.task_label / "results.json",
                        cfg.lmms_task,
                    )
                except (json.JSONDecodeError, OSError):
                    pass
        else:
            pending.append(cfg)

    print(f"Suite: {suite.suite_name}")
    print(f"Tasks: {len(task_labels)} total, {len(pending)} to run, {len(skipped_outcomes)} skipped (resume)")
    print(f"GPUs: {','.join(gpus)} ({'sequential' if len(gpus) == 1 else f'up to {len(gpus)} parallel'})")

    if dry_run:
        print("\n=== DRY RUN (no subprocesses) ===\n")
        for cfg in pending:
            task_dir = output_dir / "tasks" / cfg.task_label
            lmms_output = task_dir / "lmms_eval_output"
            cmd = build_lmms_eval_command(cfg, model, model_args, lmms_output, python_executable)
            gpu = gpus[0] if len(gpus) == 1 else "(assigned at runtime)"
            media = " --process_with_media" if cfg.process_with_media else ""
            print(f"[{cfg.task_label}] gpu={gpu} lmms_task={cfg.lmms_task}{media}")
            print("  " + " ".join(cmd))
            print(f"  CUDA_VISIBLE_DEVICES=<gpu>")
            print()
        for outcome in skipped_outcomes:
            print(f"[{outcome.task_label}] SKIP (resume success)")
        return 0

    outcomes: list[TaskOutcome] = list(skipped_outcomes)
    failed = False

    if len(gpus) == 1:
        gpu = gpus[0]
        for cfg in pending:
            print(f"\n>>> Running {cfg.task_label} on GPU {gpu}")
            outcome = run_one_task(
                cfg, model, model_args, output_dir, gpu, python_executable, suite_env=suite.env
            )
            outcomes.append(outcome)
            print(f"    {outcome.status} ({outcome.duration_sec:.1f}s)")
            if outcome.status == "failed":
                failed = True
                if not continue_on_error:
                    break
    else:
        task_queue: queue.Queue[TaskRunConfig] = queue.Queue()
        for cfg in pending:
            task_queue.put(cfg)

        gpu_pool: queue.Queue[str] = queue.Queue()
        for g in gpus:
            gpu_pool.put(g)

        lock = threading.Lock()
        stop_event = threading.Event()

        def worker() -> None:
            while not stop_event.is_set():
                try:
                    cfg = task_queue.get_nowait()
                except queue.Empty:
                    return
                gpu = gpu_pool.get()
                try:
                    with lock:
                        print(f"\n>>> Running {cfg.task_label} on GPU {gpu}")
                    outcome = run_one_task(
                        cfg, model, model_args, output_dir, gpu, python_executable, suite_env=suite.env
                    )
                    with lock:
                        outcomes.append(outcome)
                        print(f"    {cfg.task_label}: {outcome.status} ({outcome.duration_sec:.1f}s)")
                        if outcome.status == "failed" and not continue_on_error:
                            stop_event.set()
                            while True:
                                try:
                                    task_queue.get_nowait()
                                except queue.Empty:
                                    break
                finally:
                    gpu_pool.put(gpu)
                    task_queue.task_done()

        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(worker) for _ in gpus]
            for f in futures:
                f.result()

    run_ended = _utc_now_iso()
    total_duration = time.time() - t0

    write_summary(
        output_dir=output_dir,
        suite_name=suite.suite_name,
        model=model,
        model_args=model_args,
        gpus=gpus,
        outcomes=outcomes,
        started_at=run_started,
        ended_at=run_ended,
        total_duration_sec=total_duration,
    )

    n_failed = sum(1 for o in outcomes if o.status == "failed")
    return 1 if n_failed else 0


# ---------------------------------------------------------------------------
# Summary artifacts
# ---------------------------------------------------------------------------


def write_summary(
    output_dir: Path,
    suite_name: str,
    model: str,
    model_args: str,
    gpus: list[str],
    outcomes: list[TaskOutcome],
    started_at: str,
    ended_at: str,
    total_duration_sec: float,
) -> None:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, Any]] = []
    for o in outcomes:
        base = {
            "suite_name": suite_name,
            "model": model,
            "model_args": model_args,
            "task": o.task_label,
            "lmms_task": o.lmms_task,
            "status": o.status,
            "duration_sec": round(o.duration_sec, 3),
            "gpu": o.gpu,
            "result_path": o.result_file,
            "error_summary": o.error_summary,
        }
        if o.status == "failed" and not o.metrics:
            merged_rows.append({**base, "metric": None, "value": None, "higher_is_better": None})
        elif not o.metrics:
            merged_rows.append({**base, "metric": None, "value": None, "higher_is_better": None})
        else:
            for m in o.metrics:
                merged_rows.append({**base, **m})

    (summary_dir / "results_merged.json").write_text(
        json.dumps(merged_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if merged_rows:
        fieldnames = sorted({k for row in merged_rows for k in row})
        with (summary_dir / "results_merged.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)

    failed = [o.task_label for o in outcomes if o.status == "failed"]
    (summary_dir / "failed_tasks.json").write_text(
        json.dumps({"failed": failed, "details": [o.__dict__ for o in outcomes if o.status == "failed"]}, indent=2),
        encoding="utf-8",
    )

    suite_status = {
        "suite_name": suite_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "total_duration_sec": round(total_duration_sec, 3),
        "gpus": gpus,
        "total_tasks": len(outcomes),
        "successful": sum(1 for o in outcomes if o.status == "success"),
        "failed": sum(1 for o in outcomes if o.status == "failed"),
        "skipped": sum(1 for o in outcomes if o.status == "skipped"),
    }
    (summary_dir / "suite_status.json").write_text(json.dumps(suite_status, indent=2), encoding="utf-8")

    lines = [
        f"# {suite_name} — run report",
        "",
        f"- **Model:** `{model}`",
        f"- **Model args:** `{model_args}`",
        f"- **Output dir:** `{output_dir.resolve()}`",
        f"- **GPUs:** `{','.join(gpus)}`",
        f"- **Started:** {started_at}",
        f"- **Ended:** {ended_at}",
        f"- **Duration:** {total_duration_sec:.1f}s",
        "",
        "## Task status",
        "",
        "| Task | lmms-eval task | Status | GPU | Duration (s) |",
        "|------|----------------|--------|-----|--------------|",
    ]
    for o in outcomes:
        lines.append(
            f"| {o.task_label} | {o.lmms_task} | {o.status} | {o.gpu} | {o.duration_sec:.1f} |"
        )
    lines.extend(["", "## Metrics (sample)", "", "| Task | Metric | Value |", "|------|--------|-------|"])
    for row in merged_rows[:80]:
        if row.get("metric"):
            lines.append(f"| {row['task']} | {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Logs",
            "",
            "Per task: `{output_dir}/tasks/<task>/{stdout.log,stderr.log,command.txt,status.json,results.json}`",
            "",
            f"Merged CSV: `{summary_dir / 'results_merged.csv'}`",
        ]
    )
    (summary_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_gpus(value: str) -> list[str]:
    gpus = [g.strip() for g in value.split(",") if g.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("At least one GPU id required")
    return gpus


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple existing lmms-eval tasks from a YAML task list (orchestration only).",
    )
    parser.add_argument("--suite", required=True, type=Path, help="YAML file listing tasks to run")
    parser.add_argument("--model", required=True, help="lmms-eval model name")
    parser.add_argument("--model_args", required=True, default="", help="lmms-eval model_args string")
    parser.add_argument("--output_dir", required=True, type=Path, help="Root directory for all task outputs")
    parser.add_argument("--gpus", required=True, type=parse_gpus, help="Comma-separated GPU ids, e.g. 0 or 0,1,2,3")
    parser.add_argument("--batch_size", required=True, type=int, help="Batch size for all tasks (overrides suite default)")
    parser.add_argument("--limit", default=None, type=float, help="Passed to lmms-eval --limit (smoke testing)")
    parser.add_argument("--tasks", default=None, help="Comma-separated subset of suite tasks to run")
    parser.add_argument("--exclude_tasks", default=None, help="Comma-separated tasks to skip")
    parser.add_argument("--resume", action="store_true", help="Skip tasks with status success")
    parser.add_argument("--dry_run", action="store_true", help="Print planned commands only")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue after task failures")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--log_samples", action="store_true", default=None)
    parser.add_argument("--no_log_samples", action="store_true", help="Disable --log_samples")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for lmms-eval subprocesses",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    log_samples: bool | None = None
    if args.no_log_samples:
        log_samples = False
    elif args.log_samples:
        log_samples = True

    only = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    exclude = [t.strip() for t in args.exclude_tasks.split(",") if t.strip()] if args.exclude_tasks else None

    return run_suite(
        suite_path=args.suite,
        model=args.model,
        model_args=args.model_args,
        output_dir=args.output_dir,
        gpus=args.gpus,
        cli_batch_size=args.batch_size,
        cli_limit=args.limit,
        cli_num_fewshot=args.num_fewshot,
        cli_log_samples=log_samples,
        only_tasks=only,
        exclude_tasks=exclude,
        resume=args.resume,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        python_executable=args.python,
    )


if __name__ == "__main__":
    raise SystemExit(main())
