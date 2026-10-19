#!/usr/bin/env python3
"""Concurrency probe for VLM endpoints: how many parallel requests can each
model actually sustain, and at what latency?

Ramps through concurrency levels per model, fires real image+prompt requests
through the SAME client code path the pipeline uses (src.vlm.client.
chat_completion, thread-local sessions), and reports success rate / p50 /
p95 latency / effective RPS per level, plus a recommended
max_concurrency_per_endpoint for configs/bpc*.yaml.

Usage (on the GPU server, .env with API_TOKEN in cwd):

    python scripts/probe_endpoints.py \
        --base-url https://gateway.frontierai.ru \
        --models qwen3.5-4b-odie2,qwen3.5-4b-odie6,qwen3.5-4b-odie7,qwen3.5-4b-odie8 \
        --levels 4,8,16,32,64 --requests-per-level 2

    # Judge models (35B are slower; probe lower levels):
    python scripts/probe_endpoints.py \
        --base-url https://gateway.frontierai.ru \
        --models test-earth,qwen-3.5-30b \
        --levels 4,8,16,32,64 --requests-per-level 2

Each level launches <level> threads, each sending --requests-per-level
sequential requests, so a level costs level*requests_per_level calls.
Levels run in ascending order with a short pause between them so an
overloaded endpoint can drain before the next measurement.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.client import chat_completion  # noqa: E402
from src.vlm.prompts_processing import load_yaml, get_system_user_prompts  # noqa: E402


def _make_test_image(width: int = 768, height: int = 512):
    """Synthetic frame roughly matching real pipeline input dimensions."""
    from PIL import Image, ImageDraw
    import random

    rng = random.Random(42)
    img = Image.new("RGB", (width, height), (120, 140, 160))
    draw = ImageDraw.Draw(img)
    for _ in range(6):
        x1, y1 = rng.randint(0, width - 100), rng.randint(0, height - 100)
        x2, y2 = x1 + rng.randint(40, 100), y1 + rng.randint(40, 100)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    return img


_SYSTEM = (
    "You are a vision assistant. Reply with STRICT JSON only: "
    '{"objects": <number of colored rectangles you see>}'
)
_USER = "How many colored rectangles are in this image? JSON only."


@dataclass(frozen=True)
class ProbePayload:
    model_task: str
    system_text: str
    user_text: str
    images: list


def _resolve_api_token(*, allow_no_auth: bool) -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    api_token = os.getenv("API_TOKEN", "")
    if not api_token and not allow_no_auth:
        sys.exit(
            "API_TOKEN is not set (put it in .env or the environment), "
            "or pass --allow-no-auth for direct backends that do not require auth."
        )
    return api_token


def _draw_bbox_overlay(image):
    from PIL import ImageDraw

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    boxes = {
        1: [60, 80, 190, 210],
        2: [260, 120, 420, 280],
        3: [480, 180, 650, 360],
    }
    for idx, (x1, y1, x2, y2) in boxes.items():
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 4, max(4, y1 - 18)), str(idx), fill="red")
    return vis, boxes


def _draw_point_overlay(image):
    from PIL import ImageDraw

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    points = {
        "id": [1, 2, 3],
        "points": [[120, 150], [340, 220], [560, 300]],
    }
    for idx, (x, y) in zip(points["id"], points["points"], strict=True):
        r = 10
        draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=3)
        draw.line([x - 18, y, x + 18, y], fill="red", width=2)
        draw.line([x, y - 18, x, y + 18], fill="red", width=2)
        draw.text((x + 8, y - 20), str(idx), fill="red")
    return vis, points


def _build_probe_payload(*, mode: str, image=None) -> ProbePayload:
    base_image = image or _make_test_image()
    if mode == "light":
        return ProbePayload(
            model_task="light",
            system_text=_SYSTEM,
            user_text=_USER,
            images=[base_image],
        )

    prompts = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "prompts.yaml")

    if mode == "bboxing":
        vis, bbox_dict = _draw_bbox_overlay(base_image)
        system_text, user_text = get_system_user_prompts(prompts, "bboxing", bbox_dict)
        return ProbePayload(
            model_task="bboxing",
            system_text=system_text,
            user_text=user_text,
            images=[base_image, vis],
        )

    if mode == "pointing":
        vis, points_dict = _draw_point_overlay(base_image)
        system_text, user_text = get_system_user_prompts(prompts, "pointing", points_dict)
        return ProbePayload(
            model_task="pointing",
            system_text=system_text,
            user_text=user_text,
            images=[base_image, vis],
        )

    if mode == "captions":
        system_text, user_text = get_system_user_prompts(prompts, "general_captioning")
        return ProbePayload(
            model_task="captions",
            system_text=system_text,
            user_text=user_text,
            images=[base_image],
        )

    raise ValueError(f"Unsupported probe mode: {mode}")


def _one_request(
    base_url: str,
    api_token: str,
    model: str,
    payload: ProbePayload,
    timeout_s: int,
) -> dict:
    t0 = time.monotonic()
    try:
        chat_completion(
            base_url=base_url,
            api_token=api_token,
            model=model,
            system_text=payload.system_text,
            user_text=payload.user_text,
            images=payload.images,
            timeout_s=timeout_s,
            max_tokens=64,
        )
        return {"ok": True, "latency": time.monotonic() - t0, "error": None}
    except Exception as exc:
        msg = str(exc)
        kind = "timeout" if "timed out" in msg.lower() else "error"
        return {"ok": False, "latency": time.monotonic() - t0, "error": kind,
                "error_text": f"{type(exc).__name__}: {msg[:200]}"}


def _run_level(
    *, base_url: str, api_token: str, model: str, payload: ProbePayload,
    concurrency: int, requests_per_thread: int, timeout_s: int,
) -> dict:
    results: list[dict] = []
    lock = threading.Lock()
    barrier = threading.Barrier(concurrency)

    def worker():
        barrier.wait()  # start all threads at the same instant
        for _ in range(requests_per_thread):
            r = _one_request(base_url, api_token, model, payload, timeout_s)
            with lock:
                results.append(r)

    t_start = time.monotonic()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t_start

    ok = [r for r in results if r["ok"]]
    latencies = sorted(r["latency"] for r in ok)
    n = len(results)
    first_error = next((r["error_text"] for r in results if not r["ok"]), None)
    summary = {
        "concurrency": concurrency,
        "total": n,
        "ok": len(ok),
        "timeouts": sum(1 for r in results if r["error"] == "timeout"),
        "errors": sum(1 for r in results if r["error"] == "error"),
        "success_rate": len(ok) / n if n else 0.0,
        "p50": latencies[len(latencies) // 2] if latencies else None,
        "p95": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else None,
        "mean": statistics.mean(latencies) if latencies else None,
        "rps": len(ok) / wall if wall > 0 else 0.0,
        "first_error": first_error,
    }
    return summary


def _recommend(levels: list[dict], timeout_s: int) -> int | None:
    """Highest concurrency with >=95% success and p95 comfortably under timeout."""
    best = None
    for s in levels:
        healthy = (
            s["success_rate"] >= 0.95
            and s["p95"] is not None
            and s["p95"] <= timeout_s * 0.8
        )
        if healthy:
            best = s["concurrency"]
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--models", required=True, help="Comma-separated model names.")
    ap.add_argument("--levels", default="4,8,16,32", help="Comma-separated concurrency levels.")
    ap.add_argument("--requests-per-level", type=int, default=2,
                    help="Sequential requests per thread at each level.")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument(
        "--mode",
        default="light",
        choices=["light", "bboxing", "pointing", "captions"],
        help="Probe payload shape: synthetic light prompt or near-real BPC task.",
    )
    ap.add_argument(
        "--allow-no-auth",
        action="store_true",
        help="Allow an empty API token for direct backends that do not require auth.",
    )
    ap.add_argument("--image", default=None,
                    help="Path to a real frame to send (default: synthetic 768x512).")
    ap.add_argument("--pause", type=float, default=5.0,
                    help="Seconds to pause between levels/models so the endpoint drains.")
    args = ap.parse_args()

    api_token = _resolve_api_token(allow_no_auth=args.allow_no_auth)

    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
    else:
        image = _make_test_image()
    payload = _build_probe_payload(mode=args.mode, image=image)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    all_recommendations: dict[str, int | None] = {}

    for model in models:
        print(f"\n{'=' * 72}\nMODEL: {model}  MODE: {payload.model_task}\n{'=' * 72}")
        header = (
            f"{'conc':>5} {'total':>6} {'ok':>5} {'tmout':>6} {'err':>5} "
            f"{'succ%':>7} {'p50s':>7} {'p95s':>7} {'rps':>7}"
        )
        print(header)
        print("-" * len(header))

        model_levels: list[dict] = []
        for level in levels:
            s = _run_level(
                base_url=args.base_url, api_token=api_token, model=model,
                payload=payload, concurrency=level,
                requests_per_thread=args.requests_per_level, timeout_s=args.timeout,
            )
            model_levels.append(s)
            print(
                f"{s['concurrency']:>5} {s['total']:>6} {s['ok']:>5} "
                f"{s['timeouts']:>6} {s['errors']:>5} {s['success_rate'] * 100:>6.1f}% "
                f"{s['p50'] or 0:>7.2f} {s['p95'] or 0:>7.2f} {s['rps']:>7.2f}"
            )
            if s["first_error"]:
                print(f"      sample error: {s['first_error']}")
            # Stop the ramp only on a real collapse with a meaningful sample —
            # a single flaky request at low levels must not abort the probe
            # (transient TLS resets on the gateway were seen doing exactly that).
            if s["success_rate"] < 0.5 and s["total"] >= 8:
                print(f"  (stopping ramp: success rate collapsed at concurrency {level})")
                break
            time.sleep(args.pause)

        rec = _recommend(model_levels, args.timeout)
        all_recommendations[model] = rec
        print(
            f"--> recommended max_concurrency_per_endpoint for {model}: "
            f"{rec if rec is not None else 'NONE HEALTHY (even the lowest level failed)'}"
        )
        time.sleep(args.pause)

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for model, rec in all_recommendations.items():
        print(f"{model:<30} recommended cap: {rec}")
    healthy = [r for r in all_recommendations.values() if r]
    if healthy:
        print(
            f"\nConfig suggestion: max_concurrency_per_endpoint = {min(healthy)} "
            "(the weakest endpoint bounds the shared setting)."
        )


if __name__ == "__main__":
    main()
