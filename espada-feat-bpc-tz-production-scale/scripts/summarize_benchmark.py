#!/usr/bin/env python3
"""Summarizes a BPC benchmark run: throughput, per-stage timing, GPU utilization,
per-endpoint breakdown. Reads logs/state.jsonl and logs/gpu_util.csv.

Usage:
    python scripts/summarize_benchmark.py --run /path/to/output_dir [--elapsed-sec N]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_state(state_path: Path) -> list[dict]:
    records = []
    if not state_path.exists():
        return records
    with state_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def summarize_stages(records: list[dict]) -> None:
    print("--- Per-stage / task summary (from state.jsonl) ---")
    by_task: dict[str, dict[str, list]] = defaultdict(lambda: {"done": [], "retry": 0, "failed": 0})

    for r in records:
        task = r.get("task_name", "unknown")
        event = r.get("event")
        if event == "done" and r.get("duration_sec") is not None:
            by_task[task]["done"].append(r["duration_sec"])
        elif event == "retry":
            by_task[task]["retry"] += 1
        elif event == "failed":
            by_task[task]["failed"] += 1

    header = f"{'task':<20}{'done':>8}{'retries':>10}{'failed':>8}{'p50_s':>10}{'p95_s':>10}{'mean_s':>10}"
    print(header)
    print("-" * len(header))
    for task, stats in sorted(by_task.items()):
        durations = stats["done"]
        n = len(durations)
        if n:
            durations_sorted = sorted(durations)
            p50 = durations_sorted[int(n * 0.50)]
            p95 = durations_sorted[min(int(n * 0.95), n - 1)]
            mean = statistics.mean(durations)
        else:
            p50 = p95 = mean = 0.0
        print(
            f"{task:<20}{n:>8}{stats['retry']:>10}{stats['failed']:>8}"
            f"{p50:>10.2f}{p95:>10.2f}{mean:>10.2f}"
        )
    print()


def summarize_endpoints(records: list[dict]) -> None:
    print("--- Per-endpoint request counts (from state.jsonl 'done' events) ---")
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("event") == "done":
            ep = r.get("endpoint")
            if ep:
                counts[ep] += 1
    if not counts:
        print("(no endpoint data recorded)")
        print()
        return
    total = sum(counts.values())
    for ep, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / total if total else 0.0
        print(f"{ep:<45}{n:>10}  ({pct:5.1f}%)")
    print()


def summarize_throughput(records: list[dict], elapsed_sec: float | None) -> None:
    print("--- Overall throughput ---")
    done_images = {r["image_id"] for r in records if r.get("event") == "done" and r.get("task_name") == "captions"}
    n_done = len(done_images)
    failed_images = {r["image_id"] for r in records if r.get("event") == "failed"}

    print(f"Images with captions done: {n_done}")
    print(f"Images with a failed event: {len(failed_images)}")
    if elapsed_sec and elapsed_sec > 0:
        rate = n_done / elapsed_sec
        print(f"Wall-clock elapsed: {elapsed_sec:.1f}s")
        print(f"Throughput: {rate:.3f} images/sec  ({rate * 3600:.0f} images/hour)")
        if rate > 0:
            eta_12m_days = (12_000_000 / rate) / 86400
            print(f"Extrapolated to 12M images: ~{eta_12m_days:.1f} days at this rate")
    print()


def summarize_gpu(gpu_csv: Path) -> None:
    print("--- GPU utilization (from gpu_util.csv) ---")
    if not gpu_csv.exists():
        print("(no gpu_util.csv found — nvidia-smi sampling may have been unavailable)")
        print()
        return

    by_gpu: dict[str, list[float]] = defaultdict(list)
    by_gpu_mem: dict[str, list[float]] = defaultdict(list)
    with gpu_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = row["gpu_index"]
                by_gpu[idx].append(float(row["util_gpu_pct"]))
                by_gpu_mem[idx].append(float(row["mem_used_mib"]))
            except (KeyError, ValueError):
                continue

    if not by_gpu:
        print("(no samples parsed)")
        print()
        return

    header = f"{'gpu':<6}{'mean_util%':>12}{'max_util%':>12}{'mean_mem_mib':>14}{'samples':>10}"
    print(header)
    print("-" * len(header))
    for idx in sorted(by_gpu, key=lambda x: int(x) if x.isdigit() else 0):
        utils = by_gpu[idx]
        mems = by_gpu_mem[idx]
        print(
            f"{idx:<6}{statistics.mean(utils):>12.1f}{max(utils):>12.1f}"
            f"{statistics.mean(mems):>14.0f}{len(utils):>10}"
        )
    idle_gpus = [idx for idx, utils in by_gpu.items() if statistics.mean(utils) < 5.0]
    if idle_gpus:
        print(f"\nWARNING: GPU(s) {idle_gpus} averaged <5% utilization — check --gpus / device pinning.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="Output root directory of the run.")
    ap.add_argument("--elapsed-sec", type=float, default=None, help="Total wall-clock seconds (from benchmark_run.sh).")
    args = ap.parse_args()

    run_dir: Path = args.run
    state_path = run_dir / "logs" / "state.jsonl"
    gpu_csv = run_dir / "logs" / "gpu_util.csv"

    records = load_state(state_path)
    if not records:
        print(f"No state.jsonl records found at {state_path}. Nothing to summarize.")
        return

    print(f"=== Benchmark summary: {run_dir} ===\n")
    summarize_throughput(records, args.elapsed_sec)
    summarize_stages(records)
    summarize_endpoints(records)
    summarize_gpu(gpu_csv)


if __name__ == "__main__":
    main()
