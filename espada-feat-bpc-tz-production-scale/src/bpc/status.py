"""espada-bpc status command: reads state.jsonl and outputs progress summary."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _display_name(stage: str, task: str) -> str:
    """Maps (stage, task_name) from state.jsonl to the display bucket used by
    print_status. Judge events carry the SAME task_name as the underlying VLM
    task ("bboxing", "pointing") -- only ``stage`` distinguishes a judge event
    from the base VLM event -- so grouping by task_name alone (as an earlier
    version of this function did) silently merged judge events into the base
    bucket and produced permanently-zero "bboxing-judged"/"pointing-judged"
    counts, even while outputs/bboxing-judged/*.jsonl was filling up normally.
    """
    if stage == "judge":
        return f"{task}-judged"
    return task


def read_status(output_root: Path) -> dict[str, Any]:
    """Aggregates state.jsonl into a status summary dict."""
    state_path = output_root / "logs" / "state.jsonl"
    failed_path = output_root / "logs" / "failed.jsonl"

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Keyed by (image_id, stage, task_name) so a judge event for "bboxing"
    # can't be mistaken for the base VLM "bboxing" event on the same image.
    running: dict[tuple[str, str, str], bool] = {}

    if state_path.exists():
        with state_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = rec.get("stage", "")
                task = rec.get("task_name", "unknown")
                display = _display_name(stage, task)
                event = rec.get("event", "")
                image_id = rec.get("image_id", "")
                key = (image_id, stage, task)

                if event == "started":
                    running[key] = True
                elif event in ("done", "failed", "skip"):
                    running.pop(key, None)

                if event == "done":
                    counts[display]["done"] += 1
                elif event == "failed":
                    counts[display]["failed"] += 1
                elif event == "skip":
                    counts[display]["skipped"] += 1
                elif event == "retry":
                    counts[display]["retry"] += 1

    # Count still-running (started but no done/failed) -- re-derive the
    # display bucket from the same key so judge vs base VLM stays separate.
    for (image_id, stage, task), _ in running.items():
        counts[_display_name(stage, task)]["running"] += 1

    # Count failed from failed.jsonl
    failed_counts: dict[str, int] = defaultdict(int)
    if failed_path.exists():
        with failed_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    display = _display_name(rec.get("stage", ""), rec.get("task_name", "unknown"))
                    failed_counts[display] += 1
                except json.JSONDecodeError:
                    pass

    return {
        "counts": {k: dict(v) for k, v in counts.items()},
        "permanently_failed": dict(failed_counts),
    }


def print_status(output_root: Path) -> None:
    """Prints a human-readable status summary to stdout."""
    data = read_status(output_root)
    counts = data["counts"]
    failed = data["permanently_failed"]

    tasks = ["bboxing", "pointing", "captions", "bboxing-judged", "pointing-judged"]
    for task in tasks:
        c = counts.get(task, {})
        done = c.get("done", 0)
        running_n = c.get("running", 0)
        fail_n = failed.get(task, 0)
        skipped = c.get("skipped", 0)
        print(
            f"{task:<20} done={done:>8}  running={running_n:>6}  "
            f"failed={fail_n:>6}  skipped={skipped:>6}"
        )
