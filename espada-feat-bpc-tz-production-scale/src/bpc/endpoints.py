"""Endpoint pool: weighted round-robin VLM endpoint selection with per-endpoint caps.

When Qwen (or any VLM) is served behind multiple endpoints, we must:
  - spread requests across endpoints weighted by their safe concurrency,
  - never exceed each endpoint's safe concurrency (max_concurrency_per_endpoint),
  - degrade gracefully when an endpoint is temporarily failing.

Two deployment shapes are supported:
  1. One URL per instance (e.g. "http://odie2:8000", "http://odie6:8000", ...).
  2. A single router URL with per-instance selection via the "model" field
     (e.g. gateway.frontierai.ru routing to "qwen3.5-4b-odie2", "...-odie6", ...).
     This is expressed as EndpointTarget(base_url=<same for all>, model=<varies>).

This module is thread-safe so all VLM worker threads can share one pool.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger("bpc.endpoints")


@dataclass(frozen=True)
class EndpointTarget:
    """A single routable VLM target: where to send the request and which model to ask for.

    Two targets are distinct endpoints (get separate concurrency slots) if either
    base_url or model differs — this lets a single router URL host N distinct
    model-routed backends, each with its own concurrency cap and cooldown.
    """

    base_url: str
    model: str
    max_concurrency: int | None = None

    def __str__(self) -> str:
        return f"{self.base_url}::{self.model}" if self.model else self.base_url


@dataclass
class _EndpointSlot:
    target: EndpointTarget
    semaphore: threading.Semaphore
    max_concurrency: int
    fail_count: int = 0
    cooldown_until: float = 0.0
    in_flight: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)


def _to_target(item: "str | EndpointTarget") -> EndpointTarget:
    if isinstance(item, EndpointTarget):
        return item
    return EndpointTarget(base_url=item, model="")


class EndpointPool:
    """Weighted round-robin pool of VLM endpoints with per-endpoint limits.

    ``endpoints`` accepts either plain base_url strings (classic one-URL-per-
    instance deployment) or EndpointTarget(base_url, model) pairs (router-style
    deployment where instances are selected by model name behind one URL).
    """

    def __init__(
        self,
        endpoints: "list[str] | list[EndpointTarget]",
        *,
        max_concurrency_per_endpoint: int = 2,
        cooldown_seconds: float = 30.0,
        max_fails_before_cooldown: int = 3,
    ) -> None:
        if not endpoints:
            raise ValueError("EndpointPool requires at least one endpoint.")
        self._slots = []
        for item in endpoints:
            target = _to_target(item)
            target_cap = target.max_concurrency or max_concurrency_per_endpoint
            self._slots.append(
                _EndpointSlot(
                    target=target,
                    semaphore=threading.Semaphore(target_cap),
                    max_concurrency=target_cap,
                )
            )
        weighted_indices: list[int] = []
        max_cap = max(slot.max_concurrency for slot in self._slots)
        for position in range(max_cap):
            for idx, slot in enumerate(self._slots):
                if position < slot.max_concurrency:
                    weighted_indices.append(idx)
        self._rr = itertools.cycle(weighted_indices)
        self._rr_lock = threading.Lock()
        self._cooldown_seconds = cooldown_seconds
        self._max_fails = max_fails_before_cooldown

    @property
    def total_capacity(self) -> int:
        """Maximum concurrent requests across all endpoints."""
        return sum(s.max_concurrency for s in self._slots)

    @property
    def urls(self) -> list[str]:
        """Backward-compat: distinct base_urls (deduplicated, order-preserved)."""
        seen: dict[str, None] = {}
        for s in self._slots:
            seen.setdefault(s.target.base_url, None)
        return list(seen)

    @property
    def targets(self) -> list[EndpointTarget]:
        return [s.target for s in self._slots]

    def _next_index(self) -> int:
        with self._rr_lock:
            return next(self._rr)

    def _pick_slot(self) -> _EndpointSlot:
        """Picks the next healthy endpoint via weighted round-robin."""
        now = time.monotonic()
        n = self.total_capacity
        for _ in range(n):
            idx = self._next_index()
            slot = self._slots[idx]
            if slot.cooldown_until <= now:
                return slot
        # All on cooldown — return the one with the earliest cooldown expiry.
        return min(self._slots, key=lambda s: s.cooldown_until)

    @contextmanager
    def acquire(self) -> Iterator[EndpointTarget]:
        """Acquires a slot on a healthy endpoint, yielding its EndpointTarget.

        Blocks on the endpoint's semaphore so per-endpoint concurrency is capped.
        """
        slot = self._pick_slot()
        slot.semaphore.acquire()
        with slot._lock:
            slot.in_flight += 1
        try:
            yield slot.target
        finally:
            with slot._lock:
                slot.in_flight -= 1
            slot.semaphore.release()

    def report_success(self, target: "EndpointTarget | str") -> None:
        target = _to_target(target)
        for slot in self._slots:
            if slot.target == target:
                with slot._lock:
                    slot.fail_count = 0
                return

    def report_failure(self, target: "EndpointTarget | str") -> None:
        target = _to_target(target)
        for slot in self._slots:
            if slot.target == target:
                with slot._lock:
                    slot.fail_count += 1
                    if slot.fail_count >= self._max_fails:
                        slot.cooldown_until = time.monotonic() + self._cooldown_seconds
                        logger.warning(
                            "Endpoint %s entered cooldown for %.0fs after %d fails.",
                            slot.target, self._cooldown_seconds, slot.fail_count,
                        )
                        slot.fail_count = 0
                return


def recommended_worker_count(pool: EndpointPool) -> int:
    """Suggests a VLM worker-thread count that saturates all endpoints.

    Threads should slightly exceed total capacity so a thread is always ready
    to grab a freed slot. We use capacity (1:1) which is safe and simple.
    """
    return pool.total_capacity


def build_targets(
    *,
    targets: list[dict[str, object]] | None = None,
    endpoints: list[str] | None = None,
    router_url: str | None = None,
    models: list[str] | None = None,
) -> list[EndpointTarget]:
    """Builds an EndpointTarget list from either config shape.

    Explicit ``targets`` takes priority and supports per-target concurrency
    overrides, e.g. a stronger H100 backend can get a larger cap than an
    A100 backend.

    Router shape (single URL, many model names) takes priority when both
    ``router_url`` and ``models`` are set — this is the gateway.frontierai.ru
    pattern where one URL fans out to N backends via the "model" field.
    Otherwise falls back to one target per plain URL in ``endpoints``.
    """
    if targets:
        built: list[EndpointTarget] = []
        for item in targets:
            if not isinstance(item, dict):
                raise ValueError("Each explicit target must be a mapping.")
            base_url = item.get("base_url")
            if not isinstance(base_url, str) or not base_url.strip():
                raise ValueError("Each explicit target requires a non-empty 'base_url'.")
            model = item.get("model", "")
            if model is None:
                model = ""
            if not isinstance(model, str):
                raise ValueError("Explicit target 'model' must be a string when provided.")
            max_concurrency = item.get("max_concurrency")
            if max_concurrency is not None:
                if not isinstance(max_concurrency, int) or max_concurrency < 1:
                    raise ValueError(
                        "Explicit target 'max_concurrency' must be an integer >= 1."
                    )
            built.append(
                EndpointTarget(
                    base_url=base_url,
                    model=model,
                    max_concurrency=max_concurrency,
                )
            )
        return built
    if router_url and models:
        return [EndpointTarget(base_url=router_url, model=m) for m in models]
    if not endpoints:
        raise ValueError("Provide either (router_url + models) or endpoints.")
    return [EndpointTarget(base_url=url, model="") for url in endpoints]
