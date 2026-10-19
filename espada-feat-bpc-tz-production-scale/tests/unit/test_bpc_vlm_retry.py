"""Tests for _vlm_call_with_retry — retry, endpoint pool, failure handling."""
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from pathlib import Path

from src.bpc.pipeline import _vlm_call_with_retry
from src.bpc.endpoints import EndpointPool
from src.bpc.state import StateLogger


def _logger(tmp_path):
    return StateLogger(tmp_path / "logs", "run-test")


_GOOD_JSON = '{"grounding_question": "q", "grounding_answer": [{"label":"car","bbox_2d":[0,0,10,10]}]}'


def _fake_completion(content):
    return SimpleNamespace(content=content)


class TestRetrySuccess:
    def test_succeeds_first_attempt(self, tmp_path):
        with patch("src.bpc.pipeline.chat_completion") as mock_cc:
            mock_cc.return_value = _fake_completion(_GOOD_JSON)
            result = _vlm_call_with_retry(
                task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                system_text="s", user_text="u", images=[],
                endpoints=["http://a"], api_token="", model="qwen",
                state_logger=_logger(tmp_path), max_attempts=3, backoff_seconds=0,
            )
            assert result is not None
            assert "answer" in result
            assert mock_cc.call_count == 1

    def test_emits_response_not_done_event(self, tmp_path):
        """_vlm_call_with_retry must log 'response' (model replied+parsed), NOT
        'done'. 'done' is reserved for _bpc_vlm_worker after a successful save
        — emitting it here double-counts every task and, worse, corrupts resume
        (a task whose HTTP succeeded but whose validation later failed would be
        treated as complete by load_completed() and never reprocessed)."""
        import json
        lg = _logger(tmp_path)
        with patch("src.bpc.pipeline.chat_completion") as mock_cc:
            mock_cc.return_value = _fake_completion(_GOOD_JSON)
            _vlm_call_with_retry(
                task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                system_text="s", user_text="u", images=[],
                endpoints=["http://a"], api_token="", model="qwen",
                state_logger=lg, max_attempts=3, backoff_seconds=0,
            )
        events = [
            json.loads(l)["event"]
            for l in (tmp_path / "logs" / "state.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert "response" in events
        assert "done" not in events  # the single 'done' is the worker's job

    def test_response_not_added_to_load_completed(self, tmp_path):
        """A 'response' event alone must NOT mark a task complete for resume."""
        import json
        lg = _logger(tmp_path)
        with patch("src.bpc.pipeline.chat_completion") as mock_cc:
            mock_cc.return_value = _fake_completion(_GOOD_JSON)
            _vlm_call_with_retry(
                task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                system_text="s", user_text="u", images=[],
                endpoints=["http://a"], api_token="", model="qwen",
                state_logger=lg, max_attempts=1, backoff_seconds=0,
            )
        assert ("img1", "bboxing") not in lg.load_completed()


class TestRetryFailure:
    def test_retries_then_succeeds(self, tmp_path):
        calls = {"n": 0}

        def side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("HTTP 500")
            return _fake_completion(_GOOD_JSON)

        with patch("src.bpc.pipeline.chat_completion", side_effect=side_effect):
            result = _vlm_call_with_retry(
                task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                system_text="s", user_text="u", images=[],
                endpoints=["http://a"], api_token="", model="qwen",
                state_logger=_logger(tmp_path), max_attempts=3, backoff_seconds=0,
            )
            assert result is not None
            assert calls["n"] == 2

    def test_exhausts_retries_returns_none(self, tmp_path):
        with patch("src.bpc.pipeline.chat_completion", side_effect=RuntimeError("boom")):
            lg = _logger(tmp_path)
            result = _vlm_call_with_retry(
                task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                system_text="s", user_text="u", images=[],
                endpoints=["http://a"], api_token="", model="qwen",
                state_logger=lg, max_attempts=3, backoff_seconds=0,
            )
            assert result is None
            # failed.jsonl should have an entry
            failed = (tmp_path / "logs" / "failed.jsonl")
            assert failed.exists()
            assert "boom" in failed.read_text()


class TestEndpointPoolIntegration:
    def test_uses_pool_endpoint(self, tmp_path):
        pool = EndpointPool(["http://qwen-1", "http://qwen-2"],
                            max_concurrency_per_endpoint=2)
        used_urls = []

        def side_effect(*args, **kwargs):
            used_urls.append(kwargs["base_url"])
            return _fake_completion(_GOOD_JSON)

        with patch("src.bpc.pipeline.chat_completion", side_effect=side_effect):
            for _ in range(4):
                _vlm_call_with_retry(
                    task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                    system_text="s", user_text="u", images=[],
                    endpoint_pool=pool, api_token="", model="qwen",
                    state_logger=_logger(tmp_path), max_attempts=1, backoff_seconds=0,
                )
        # Both pool endpoints should have been used (round-robin)
        assert "http://qwen-1" in used_urls
        assert "http://qwen-2" in used_urls

    def test_pool_reports_failure_on_error(self, tmp_path):
        pool = EndpointPool(["http://a"], max_concurrency_per_endpoint=1,
                            max_fails_before_cooldown=1, cooldown_seconds=99)
        with patch("src.bpc.pipeline.chat_completion", side_effect=RuntimeError("x")):
            _vlm_call_with_retry(
                task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                system_text="s", user_text="u", images=[],
                endpoint_pool=pool, api_token="", model="qwen",
                state_logger=_logger(tmp_path), max_attempts=1, backoff_seconds=0,
            )
        # Endpoint should now be on cooldown
        snap_slot = pool._slots[0]
        assert snap_slot.cooldown_until > 0

    def test_requires_endpoints_or_pool(self, tmp_path):
        with pytest.raises(ValueError, match="endpoint"):
            _vlm_call_with_retry(
                task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                system_text="s", user_text="u", images=[],
                api_token="", model="qwen",
                state_logger=_logger(tmp_path), max_attempts=1, backoff_seconds=0,
            )


class TestRetryBackoffJitter:
    """Fixed lockstep backoff means every failing worker thread retries at
    exactly the same instant, hammering an already-struggling endpoint again
    all at once. Jitter must spread retries out."""

    def test_sleep_called_with_value_at_or_above_base_backoff(self, tmp_path):
        with patch("src.bpc.pipeline.chat_completion", side_effect=RuntimeError("x")):
            with patch("src.bpc.pipeline.time.sleep") as mock_sleep:
                _vlm_call_with_retry(
                    task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                    system_text="s", user_text="u", images=[],
                    endpoints=["http://a"], api_token="", model="qwen",
                    state_logger=_logger(tmp_path), max_attempts=2, backoff_seconds=10,
                )
        assert mock_sleep.call_count == 1
        slept = mock_sleep.call_args[0][0]
        assert slept >= 10  # never sleeps LESS than the configured backoff

    def test_sleep_value_has_jitter_variance_across_calls(self, tmp_path):
        """Not a fixed constant — repeated failures should sleep different
        amounts (within [backoff, backoff*1.3)), proving jitter is applied."""
        observed = []
        for _ in range(8):
            with patch("src.bpc.pipeline.chat_completion", side_effect=RuntimeError("x")):
                with patch("src.bpc.pipeline.time.sleep") as mock_sleep:
                    _vlm_call_with_retry(
                        task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                        system_text="s", user_text="u", images=[],
                        endpoints=["http://a"], api_token="", model="qwen",
                        state_logger=_logger(tmp_path), max_attempts=2, backoff_seconds=10,
                    )
            observed.append(mock_sleep.call_args[0][0])

        assert all(10 <= v < 13 for v in observed)  # backoff + up to 30%
        assert len(set(observed)) > 1  # not all identical -> jitter is real

    def test_no_sleep_on_final_attempt(self, tmp_path):
        with patch("src.bpc.pipeline.chat_completion", side_effect=RuntimeError("x")):
            with patch("src.bpc.pipeline.time.sleep") as mock_sleep:
                _vlm_call_with_retry(
                    task_name="bboxing", image_id="img1", relative_path="img1.jpg",
                    system_text="s", user_text="u", images=[],
                    endpoints=["http://a"], api_token="", model="qwen",
                    state_logger=_logger(tmp_path), max_attempts=1, backoff_seconds=10,
                )
        mock_sleep.assert_not_called()
