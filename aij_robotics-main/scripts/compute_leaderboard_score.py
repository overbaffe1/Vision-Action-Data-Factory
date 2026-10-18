#!/usr/bin/env python3
"""Compute Score from VLM CSV + LIBERO eval_info (same formula as aij-robotics-judge).

Protocol:
  VLM = mean(accuracy_i) over 8 benches, each accuracy in [0, 1]
  VLA = mean success rate over LIBERO tasks (equal task weight)
  Score = 100 * (0.4 * VLM + 0.6 * VLA)   # unrounded for ranking

Participant self-check uses libero_spatial for VLA. Official judge VLA is separate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Official scoring set (refspatial / where2place are NOT in the leaderboard mean).
VLM_PRIMARY_METRICS: dict[str, str] = {
    "erqa": "erqa_acc",
    "mmmu_val": "mmmu_acc",
    "mmmu_pro_standard": "mmmu_acc",
    "mmmu_pro_vision": "mmmu_acc",
    "blink": "blink_acc",
    "crosspoint": "crosspoint_accuracy",
    "msmu": "msmu_overall_accuracy",
    "embspatial": "embspatial_acc",
}

VLM_WEIGHT = 0.4
VLA_WEIGHT = 0.6


def _normalize_accuracy(value: float) -> float:
    """Map reported accuracy to [0, 1]. Accepts fractions or percent."""
    if value > 1.0 + 1e-9:
        value = value / 100.0
    if value < 0.0 or value > 1.0 + 1e-6:
        raise ValueError(f"accuracy out of range after normalize: {value}")
    return float(min(1.0, max(0.0, value)))


def load_vlm_accuracies(results_csv: Path) -> dict[str, float]:
    """Return task -> primary accuracy in [0, 1] from results_merged.csv."""
    if not results_csv.is_file():
        raise FileNotFoundError(f"VLM results CSV not found: {results_csv}")

    found: dict[str, float] = {}
    with results_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            task = (row.get("task") or "").strip()
            metric = (row.get("metric") or "").strip()
            if task not in VLM_PRIMARY_METRICS:
                continue
            if metric != VLM_PRIMARY_METRICS[task]:
                continue
            if row.get("status") and row["status"] != "success":
                continue
            raw = row.get("value")
            if raw is None or raw == "":
                continue
            found[task] = _normalize_accuracy(float(raw))

    missing = [t for t in VLM_PRIMARY_METRICS if t not in found]
    if missing:
        raise ValueError(f"Missing primary VLM metrics for: {missing}")
    return found


def load_vla_success(eval_info: Path) -> dict[str, Any]:
    """Equal-weight mean success rate over tasks from lerobot eval_info.json."""
    if not eval_info.is_file():
        raise FileNotFoundError(f"LIBERO eval_info not found: {eval_info}")

    data = json.loads(eval_info.read_text(encoding="utf-8"))
    per_task = data.get("per_task") or []
    if not per_task:
        raise ValueError(f"No per_task entries in {eval_info}")

    task_rates: list[float] = []
    for entry in per_task:
        metrics = entry.get("metrics") or {}
        successes = metrics.get("successes") or []
        if not successes:
            task_rates.append(0.0)
            continue
        task_rates.append(sum(1.0 for s in successes if s) / len(successes))

    vla = sum(task_rates) / len(task_rates)
    overall = (data.get("overall") or {}).get("pc_success")
    return {
        "vla": float(vla),
        "n_tasks": len(task_rates),
        "n_episodes": int(
            (data.get("overall") or {}).get("n_episodes")
            or sum(len((t.get("metrics") or {}).get("successes") or []) for t in per_task)
        ),
        "overall_pc_success": float(overall) if overall is not None else None,
        "per_task_success_rate": task_rates,
    }


def compute_score(
    *,
    vlm_csv: Path | None,
    libero_eval_info: Path | None,
) -> dict[str, Any]:
    """Compute leaderboard payload. Partial inputs → status incomplete."""
    out: dict[str, Any] = {
        "formula": "Score = 100 * (0.4 * VLM + 0.6 * VLA)",
        "vlm_weight": VLM_WEIGHT,
        "vla_weight": VLA_WEIGHT,
        "vlm_tasks": list(VLM_PRIMARY_METRICS.keys()),
        "status": "ok",
        "note": (
            "Participant self-check VLA uses libero_spatial; "
            "official judge VLA protocol is separate."
        ),
    }

    vlm: float | None = None
    vla: float | None = None
    errors: list[str] = []

    if vlm_csv is None:
        errors.append("vlm results CSV missing")
    else:
        try:
            acc = load_vlm_accuracies(vlm_csv)
            vlm = sum(acc.values()) / len(VLM_PRIMARY_METRICS)
            out["vlm_accuracies"] = acc
            out["vlm"] = vlm
        except Exception as exc:  # noqa: BLE001
            errors.append(f"vlm: {exc}")

    if libero_eval_info is None:
        errors.append("libero eval_info missing")
    else:
        try:
            vla_info = load_vla_success(libero_eval_info)
            vla = float(vla_info["vla"])
            out["vla"] = vla
            out["vla_detail"] = {
                "n_tasks": vla_info["n_tasks"],
                "n_episodes": vla_info["n_episodes"],
                "overall_pc_success": vla_info["overall_pc_success"],
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"vla: {exc}")

    if vlm is not None and vla is not None:
        out["score"] = 100.0 * (VLM_WEIGHT * vlm + VLA_WEIGHT * vla)
    else:
        out["status"] = "incomplete"
        out["score"] = None

    if errors:
        out["errors"] = errors
        if out["status"] == "ok":
            out["status"] = "incomplete"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-csv", type=Path, default=None)
    parser.add_argument("--libero-eval-info", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON here")
    args = parser.parse_args()

    payload = compute_score(vlm_csv=args.vlm_csv, libero_eval_info=args.libero_eval_info)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[score] wrote {args.output}")
        score = payload.get("score")
        txt_path = (
            args.output.with_suffix(".txt")
            if args.output.suffix == ".json"
            else args.output.parent / "leaderboard_score.txt"
        )
        txt_path.write_text("" if score is None else f"{score}\n", encoding="utf-8")
        print(f"[score] wrote {txt_path}")
    print(text, end="")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
