"""Tests for src/bpc/endpoints.py — round-robin pool with concurrency caps."""
import threading
import time
import pytest

from src.bpc.endpoints import EndpointPool, EndpointTarget, recommended_worker_count, build_targets


class TestEndpointPoolBasics:
    def test_requires_endpoints(self):
        with pytest.raises(ValueError):
            EndpointPool([])

    def test_total_capacity(self):
        pool = EndpointPool(["a", "b", "c"], max_concurrency_per_endpoint=2)
        assert pool.total_capacity == 6

    def test_urls(self):
        pool = EndpointPool(["http://a", "http://b"])
        assert pool.urls == ["http://a", "http://b"]

    def test_recommended_worker_count(self):
        pool = EndpointPool(["a", "b"], max_concurrency_per_endpoint=4)
        assert recommended_worker_count(pool) == 8

    def test_target_specific_capacity_overrides_default(self):
        targets = [
            EndpointTarget(base_url="https://gw", model="h100-fast", max_concurrency=12),
            EndpointTarget(base_url="https://gw", model="a100-slow", max_concurrency=6),
        ]
        pool = EndpointPool(targets, max_concurrency_per_endpoint=2)
        assert pool.total_capacity == 18
        assert recommended_worker_count(pool) == 18


class TestRoundRobin:
    def test_distributes_across_endpoints(self):
        pool = EndpointPool(["a", "b", "c"], max_concurrency_per_endpoint=10)
        seen = []
        for _ in range(6):
            with pool.acquire() as target:
                seen.append(target.base_url)
        # Round-robin: each endpoint should appear twice
        assert seen.count("a") == 2
        assert seen.count("b") == 2
        assert seen.count("c") == 2

    def test_single_endpoint(self):
        pool = EndpointPool(["only"], max_concurrency_per_endpoint=5)
        with pool.acquire() as target:
            assert target.base_url == "only"

    def test_target_specific_caps_weight_distribution(self):
        targets = [
            EndpointTarget(base_url="http://h100", model="qwen", max_concurrency=16),
            EndpointTarget(base_url="http://a100", model="qwen", max_concurrency=8),
        ]
        pool = EndpointPool(targets, max_concurrency_per_endpoint=1)

        seen = []
        for _ in range(24):
            with pool.acquire() as target:
                seen.append(target.base_url)

        assert seen.count("http://h100") == 16
        assert seen.count("http://a100") == 8


class TestConcurrencyCap:
    def test_blocks_beyond_capacity(self):
        """A second acquire on a 1-concurrency single endpoint must block."""
        pool = EndpointPool(["a"], max_concurrency_per_endpoint=1)

        acquired = threading.Event()
        released = threading.Event()
        second_acquired = threading.Event()

        def holder():
            with pool.acquire():
                acquired.set()
                released.wait(timeout=2)

        def waiter():
            acquired.wait(timeout=2)
            with pool.acquire():
                second_acquired.set()

        t1 = threading.Thread(target=holder)
        t2 = threading.Thread(target=waiter)
        t1.start()
        t2.start()

        acquired.wait(timeout=2)
        # Second thread should NOT have acquired yet (capacity=1, slot held)
        time.sleep(0.1)
        assert not second_acquired.is_set()

        # Release the first; now second can proceed
        released.set()
        t1.join(timeout=2)
        t2.join(timeout=2)
        assert second_acquired.is_set()

    def test_parallel_acquire_within_capacity(self):
        pool = EndpointPool(["a", "b"], max_concurrency_per_endpoint=2)
        # capacity 4; acquire 4 concurrently should all succeed without blocking
        results = []
        barrier = threading.Barrier(4, timeout=2)

        def grab():
            with pool.acquire() as target:
                results.append(target.base_url)
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass

        threads = [threading.Thread(target=grab) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3)
        assert len(results) == 4


class TestFailureCooldown:
    def test_failure_triggers_cooldown(self):
        pool = EndpointPool(
            ["a", "b"],
            max_concurrency_per_endpoint=1,
            cooldown_seconds=10,
            max_fails_before_cooldown=2,
        )
        # Fail endpoint "a" twice -> cooldown
        pool.report_failure("a")
        pool.report_failure("a")
        # Next acquires should prefer "b" (a is on cooldown)
        urls = []
        for _ in range(4):
            with pool.acquire() as target:
                urls.append(target.base_url)
        assert all(u == "b" for u in urls)

    def test_success_resets_fail_count(self):
        pool = EndpointPool(
            ["a", "b"],
            max_concurrency_per_endpoint=1,
            cooldown_seconds=10,
            max_fails_before_cooldown=2,
        )
        pool.report_failure("a")
        pool.report_success("a")  # reset
        pool.report_failure("a")  # only 1 fail now, no cooldown
        # "a" still healthy, round-robin includes it
        urls = set()
        for _ in range(4):
            with pool.acquire() as target:
                urls.add(target.base_url)
        assert "a" in urls

    def test_all_cooldown_still_returns_endpoint(self):
        pool = EndpointPool(
            ["a", "b"],
            max_concurrency_per_endpoint=1,
            cooldown_seconds=10,
            max_fails_before_cooldown=1,
        )
        pool.report_failure("a")
        pool.report_failure("b")
        # Both on cooldown — must still return one (graceful degradation)
        with pool.acquire() as target:
            assert target.base_url in ("a", "b")


class TestRouterModeTargets:
    """Router deployment: one URL, many model names (gateway.frontierai.ru pattern)."""

    def test_build_targets_router_mode(self):
        targets = build_targets(
            router_url="https://gw", models=["m1", "m2", "m3"]
        )
        assert len(targets) == 3
        assert all(t.base_url == "https://gw" for t in targets)
        assert [t.model for t in targets] == ["m1", "m2", "m3"]

    def test_build_targets_classic_mode(self):
        targets = build_targets(endpoints=["http://a", "http://b"])
        assert len(targets) == 2
        assert [t.base_url for t in targets] == ["http://a", "http://b"]
        assert all(t.model == "" for t in targets)

    def test_build_targets_explicit_targets_take_priority(self):
        targets = build_targets(
            endpoints=["http://ignored"],
            router_url="https://ignored",
            models=["ignored"],
            targets=[
                {"base_url": "https://gw", "model": "h100-fast", "max_concurrency": 12},
                {"base_url": "https://gw", "model": "a100-slow", "max_concurrency": 6},
            ],
        )
        assert len(targets) == 2
        assert [(t.base_url, t.model, t.max_concurrency) for t in targets] == [
            ("https://gw", "h100-fast", 12),
            ("https://gw", "a100-slow", 6),
        ]

    def test_build_targets_explicit_classic_targets_allow_per_url_caps(self):
        targets = build_targets(
            targets=[
                {"base_url": "http://h100-qwen:8000", "max_concurrency": 12},
                {"base_url": "http://a100-qwen:8000", "max_concurrency": 6},
            ],
        )
        assert [(t.base_url, t.model, t.max_concurrency) for t in targets] == [
            ("http://h100-qwen:8000", "", 12),
            ("http://a100-qwen:8000", "", 6),
        ]

    def test_build_targets_router_takes_priority(self):
        targets = build_targets(
            endpoints=["http://ignored"], router_url="https://gw", models=["m1"]
        )
        assert len(targets) == 1
        assert targets[0].base_url == "https://gw"

    def test_build_targets_requires_something(self):
        with pytest.raises(ValueError):
            build_targets()

    def test_pool_distinguishes_same_url_different_model(self):
        """Same base_url, different model — must get separate concurrency slots."""
        targets = [
            EndpointTarget(base_url="https://gw", model="m1"),
            EndpointTarget(base_url="https://gw", model="m2"),
        ]
        pool = EndpointPool(targets, max_concurrency_per_endpoint=5)
        assert pool.total_capacity == 10  # 2 distinct slots, not 1
        assert len(pool.urls) == 1  # but only one distinct base_url
        assert len(pool.targets) == 2

    def test_pool_round_robins_across_models(self):
        targets = [
            EndpointTarget(base_url="https://gw", model="odie2"),
            EndpointTarget(base_url="https://gw", model="odie6"),
        ]
        pool = EndpointPool(targets, max_concurrency_per_endpoint=10)
        seen_models = []
        for _ in range(4):
            with pool.acquire() as target:
                seen_models.append(target.model)
        assert seen_models.count("odie2") == 2
        assert seen_models.count("odie6") == 2

    def test_pool_cooldown_isolates_by_model_not_just_url(self):
        targets = [
            EndpointTarget(base_url="https://gw", model="odie2"),
            EndpointTarget(base_url="https://gw", model="odie6"),
        ]
        pool = EndpointPool(
            targets, max_concurrency_per_endpoint=1,
            cooldown_seconds=10, max_fails_before_cooldown=1,
        )
        pool.report_failure(EndpointTarget(base_url="https://gw", model="odie2"))
        # odie2 on cooldown, odie6 must be the one returned
        for _ in range(3):
            with pool.acquire() as target:
                assert target.model == "odie6"

    def test_endpoint_target_str_includes_model(self):
        t = EndpointTarget(base_url="https://gw", model="odie2")
        assert "gw" in str(t)
        assert "odie2" in str(t)

    def test_endpoint_target_str_no_model_is_just_url(self):
        t = EndpointTarget(base_url="http://a", model="")
        assert str(t) == "http://a"
