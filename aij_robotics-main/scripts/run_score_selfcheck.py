#!/usr/bin/env python3
"""Participant Score self-check: 8 VLM benches + libero_spatial → leaderboard_score.txt.

Same formula as aij-robotics-judge:
  Score = 100 * (0.4 * VLM + 0.6 * VLA)

VLA here is vanilla libero_spatial (official judge VLA is separate).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from compute_leaderboard_score import compute_score  # noqa: E402

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML required: pip install pyyaml") from exc


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _submission_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "smolvlm2").is_dir() and (path / "action_expert").is_dir():
        return path
    candidate = path / "submission"
    if (candidate / "smolvlm2").is_dir() and (candidate / "action_expert").is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Expected submission/ with smolvlm2/ + action_expert/: {path}")


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_path(raw: Any, *, default: Path | None = None, keep_symlink: bool = False) -> Path | None:
    if raw is None or raw == "":
        return default
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    # Do not resolve() interpreters: venv/bin/python → /usr/bin/python3 drops site-packages.
    if keep_symlink:
        return p
    return p.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Path to submission/ (or parent containing submission/).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "selfcheck_score.example.yaml",
        help="Self-check YAML (paths to lmms-eval, ERQA, VLA knobs).",
    )
    parser.add_argument("--smoke", action="store_true", help="VLM limit=5, 1 spatial episode")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-libero", action="store_true")
    parser.add_argument("--gpus", default=None, help="Override GPU list, e.g. 0 or 0,1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.config.expanduser().resolve() if args.config else None)
    paths_cfg = cfg.get("paths") or {}
    runtime = cfg.get("runtime") or {}
    vlm_cfg = cfg.get("vlm") or {}
    vla_cfg = cfg.get("vla") or {}

    submission = _submission_dir(args.submission)
    vlm_ckpt = submission / "smolvlm2"
    action_ckpt = submission / "action_expert"

    if args.output_dir is not None:
        run_dir = args.output_dir.expanduser().resolve()
    elif runtime.get("output_root"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(str(runtime["output_root"])).expanduser().resolve() / f"score_selfcheck_{stamp}"
    else:
        parent = submission.parent
        run_dir = parent / "score_selfcheck" if parent.name != "submission" else submission / "score_selfcheck"
        run_dir = run_dir.resolve()

    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_dir / "summary"
    summary.mkdir(exist_ok=True)
    vlm_out = run_dir / "vlm_benches"
    libero_out = run_dir / "libero_spatial"

    gpu_indices = [int(x) for x in runtime.get("gpu_indices", [0])]
    if args.gpus is not None:
        gpus = args.gpus
        gpu_indices = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        gpus = ",".join(str(i) for i in gpu_indices)

    lmms_eval = _resolve_path(
        paths_cfg.get("lmms_eval"),
        default=REPO_ROOT / "third_party" / "lmms-eval",
    )
    lmms_python = _resolve_path(
        paths_cfg.get("lmms_python"),
        default=REPO_ROOT / "third_party" / "lmms-eval" / ".venv" / "bin" / "python",
        keep_symlink=True,
    )
    erqa = _resolve_path(
        paths_cfg.get("erqa_data_dir"),
        default=REPO_ROOT / "data" / "erqa_converted",
    )

    suite = _resolve_path(vlm_cfg.get("suite"), default=REPO_ROOT / "configs" / "greenvla_vlm_tasks.yaml")
    assert suite is not None

    limit = vlm_cfg.get("limit")
    if args.smoke and limit is None:
        limit = 5

    episodes = int(vla_cfg.get("episodes_per_task", 1))
    if args.smoke:
        episodes = min(episodes, 1)
    task_ids = vla_cfg.get("task_ids") or list(range(10))
    if isinstance(task_ids, str):
        task_ids_str = task_ids
    else:
        task_ids_str = str([int(x) for x in task_ids])
    suite_name = str(vla_cfg.get("suite", "libero_spatial"))
    seed = int(vla_cfg.get("seed", 1000))

    report: dict[str, Any] = {
        "mode": "participant_score_selfcheck",
        "formula": "Score = 100 * (0.4 * VLM + 0.6 * VLA)",
        "vla_suite": suite_name,
        "note": "Self-check VLA uses libero_spatial; official judge VLA is separate.",
        "submission": str(submission),
        "run_dir": str(run_dir),
        "smoke": bool(args.smoke),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "vlm": {"status": "skipped", "exit_code": 0},
        "libero": {"status": "skipped", "exit_code": 0},
    }

    # --- VLM ---
    if args.skip_vlm:
        report["vlm"]["status"] = "skipped"
    else:
        if lmms_eval is None or lmms_python is None:
            raise SystemExit(
                "paths.lmms_eval / paths.lmms_python missing. "
                "Run: ./scripts/setup_score_selfcheck.sh"
            )
        if not (lmms_eval / "scripts" / "run_many_tasks.py").is_file():
            raise SystemExit(
                f"lmms-eval not found at {lmms_eval}. "
                "Run: ./scripts/setup_score_selfcheck.sh"
            )
        if not Path(str(lmms_python)).is_file():
            raise SystemExit(
                f"lmms python not found: {lmms_python}. "
                "Run: ./scripts/setup_score_selfcheck.sh"
            )
        vlm_args = [
            sys.executable,
            str(SCRIPTS / "run_vlm_selfcheck.py"),
            "--vlm-checkpoint",
            str(vlm_ckpt),
            "--lmms-eval",
            str(lmms_eval),
            "--python",
            str(lmms_python),
            "--suite",
            str(suite),
            "--output-dir",
            str(vlm_out),
            "--gpus",
            gpus,
            "--batch-size",
            str(int(vlm_cfg.get("batch_size", 1))),
            "--model",
            str(vlm_cfg.get("model", "smolvlm2")),
        ]
        if limit is not None:
            vlm_args.extend(["--limit", str(limit)])
        erqa_ok = erqa is not None and (erqa / "erqa.jsonl").is_file()
        if erqa_ok:
            vlm_args.extend(["--erqa-data-dir", str(erqa)])
        elif vlm_cfg.get("exclude_erqa"):
            pass
        elif not vlm_cfg.get("exclude_erqa"):
            raise SystemExit(
                "ERQA data missing (need data/erqa_converted/erqa.jsonl). "
                "Run: ./scripts/setup_score_selfcheck.sh\n"
                "Or set vlm.exclude_erqa: true (Score will omit ERQA)."
            )
        if vlm_cfg.get("exclude_erqa"):
            vlm_args.append("--exclude-erqa")
        print("[score]", " ".join(vlm_args))
        if args.dry_run:
            code = 0
        else:
            code = int(subprocess.run(vlm_args, check=False).returncode)
        report["vlm"] = {
            "status": "ok" if code == 0 else "failed",
            "exit_code": code,
            "output_dir": str(vlm_out),
            "results_csv": str(vlm_out / "summary" / "results_merged.csv"),
        }
        if code != 0 and not args.dry_run:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            _finalize(report, run_dir)
            return code

    # --- LIBERO spatial ---
    if args.skip_libero:
        report["libero"]["status"] = "skipped"
    else:
        selfcheck = SCRIPTS / "run_libero_selfcheck.sh"
        if not selfcheck.is_file():
            raise FileNotFoundError(selfcheck)
        env = os.environ.copy()
        env["SUITE"] = suite_name
        env["TASK_IDS"] = task_ids_str
        env["N_EPISODES"] = str(episodes)
        env["SEED"] = str(seed)
        env["GPU_INDEX"] = str(gpu_indices[0] if gpu_indices else 0)
        env["OUT_DIR"] = str(libero_out)
        env.setdefault("MUJOCO_GL", "egl")
        cmd = ["bash", str(selfcheck), str(submission)]
        print("[score]", " ".join(cmd))
        print(
            f"[score] SUITE={suite_name} TASK_IDS={task_ids_str} "
            f"N_EPISODES={episodes} OUT_DIR={libero_out}"
        )
        if args.dry_run:
            code = 0
        else:
            code = int(subprocess.run(cmd, check=False, env=env).returncode)
        eval_info = libero_out / "eval_info.json"
        report["libero"] = {
            "status": "ok" if code == 0 else "failed",
            "exit_code": code,
            "output_dir": str(libero_out),
            "eval_info": str(eval_info) if eval_info.is_file() else None,
            "suite": suite_name,
            "task_ids": task_ids_str,
            "episodes_per_task": episodes,
        }
        if code != 0 and not args.dry_run:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            _finalize(report, run_dir)
            return code

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run:
        _finalize(report, run_dir)
    else:
        _dump_json(summary / "selfcheck_report.json", report)
        print("[score] dry-run: skipped Score computation")
    return 0


def _finalize(report: dict[str, Any], run_dir: Path) -> None:
    summary = run_dir / "summary"
    vlm_csv = report.get("vlm", {}).get("results_csv")
    libero_info = report.get("libero", {}).get("eval_info")
    vlm_path = Path(vlm_csv) if vlm_csv else None
    libero_path = Path(libero_info) if libero_info else None
    if vlm_path is not None and not vlm_path.is_file():
        vlm_path = None
    if libero_path is not None and not libero_path.is_file():
        libero_path = None
    payload = compute_score(vlm_csv=vlm_path, libero_eval_info=libero_path)
    report["score"] = payload
    _dump_json(summary / "leaderboard_score.json", payload)
    score = payload.get("score")
    (summary / "leaderboard_score.txt").write_text(
        "" if score is None else f"{score}\n",
        encoding="utf-8",
    )
    _dump_json(summary / "selfcheck_report.json", report)
    if score is not None:
        print(f"[score] Score={score}  VLM={payload.get('vlm')}  VLA={payload.get('vla')}")
    else:
        print(f"[score] incomplete: {payload.get('errors')}")
    print(f"[done] {summary / 'leaderboard_score.json'}")
    print(f"[done] {summary / 'leaderboard_score.txt'}")
    print(f"[done] {summary / 'selfcheck_report.json'}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
