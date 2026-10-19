"""Tests for src/vlm/client.py — thread-local Session reuse (connection pooling)."""
import json
import threading
from unittest.mock import patch, MagicMock

import pytest

from src.vlm.client import chat_completion, _get_session, _thread_local


def _fake_response(content: str = '{"grounding_question": "q", "grounding_answer": []}'):
    resp = MagicMock()
    resp.ok = True
    resp.text = json.dumps({"choices": [{"message": {"content": content}}]})
    return resp


@pytest.fixture(autouse=True)
def _reset_thread_local():
    """Each test starts with no cached session on the current thread."""
    if hasattr(_thread_local, "session"):
        del _thread_local.session
    yield
    if hasattr(_thread_local, "session"):
        del _thread_local.session


class TestGetSession:
    def test_creates_session_on_first_call(self):
        session = _get_session()
        assert session is not None

    def test_reuses_same_session_within_thread(self):
        s1 = _get_session()
        s2 = _get_session()
        assert s1 is s2

    def test_session_has_http_and_https_adapters_mounted(self):
        session = _get_session()
        assert "http://" in session.adapters
        assert "https://" in session.adapters

    def test_different_threads_get_different_sessions(self):
        # Threads must all be alive simultaneously when capturing sessions —
        # otherwise the OS can recycle a finished thread's identity, making
        # two "different" threads look like the same one.
        n = 4
        sessions: dict[int, object] = {}
        lock = threading.Lock()
        barrier = threading.Barrier(n)

        def capture():
            s = _get_session()
            barrier.wait(timeout=2)  # hold until all threads have their session
            with lock:
                sessions[threading.get_ident()] = s

        threads = [threading.Thread(target=capture) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3)

        assert len(sessions) == n
        assert len(set(id(s) for s in sessions.values())) == n  # all distinct objects


class TestChatCompletionUsesSession:
    def test_chat_completion_posts_via_session(self):
        with patch("src.vlm.client._get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.post.return_value = _fake_response()
            mock_get_session.return_value = mock_session

            chat_completion(
                base_url="https://gateway.example.com",
                api_token="tok",
                model="m",
                system_text="s",
                user_text="u",
                images=None,
            )

            mock_session.post.assert_called_once()
            mock_get_session.assert_called_once()
            assert mock_session.post.call_args.args[0] == (
                "https://gateway.example.com/v1/chat/completions"
            )

    def test_chat_completion_falls_back_to_plain_chat_completions_on_v1_404(self):
        with patch("src.vlm.client._get_session") as mock_get_session:
            mock_session = MagicMock()
            not_found = MagicMock()
            not_found.ok = False
            not_found.status_code = 404
            not_found.text = "not found"
            mock_session.post.side_effect = [not_found, _fake_response()]
            mock_get_session.return_value = mock_session

            chat_completion(
                base_url="https://direct-qwen.example.com",
                api_token="tok",
                model="m",
                system_text="s",
                user_text="u",
                images=None,
            )

            assert mock_session.post.call_count == 2
            first_url = mock_session.post.call_args_list[0].args[0]
            second_url = mock_session.post.call_args_list[1].args[0]
            assert first_url == "https://direct-qwen.example.com/v1/chat/completions"
            assert second_url == "https://direct-qwen.example.com/chat/completions"

    def test_chat_completion_keeps_original_error_when_fallback_also_fails(self):
        with patch("src.vlm.client._get_session") as mock_get_session:
            mock_session = MagicMock()
            v1_not_found = MagicMock()
            v1_not_found.ok = False
            v1_not_found.status_code = 404
            v1_not_found.text = "not found"
            plain_not_found = MagicMock()
            plain_not_found.ok = False
            plain_not_found.status_code = 404
            plain_not_found.text = "still not found"
            mock_session.post.side_effect = [v1_not_found, plain_not_found]
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="HTTP 404"):
                chat_completion(
                    base_url="https://direct-qwen.example.com",
                    api_token="tok",
                    model="m",
                    system_text="s",
                    user_text="u",
                    images=None,
                )

    def test_chat_completion_omits_authorization_header_when_token_is_empty(self):
        with patch("src.vlm.client._get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.post.return_value = _fake_response()
            mock_get_session.return_value = mock_session

            chat_completion(
                base_url="http://direct-qwen.example.com:7080",
                api_token="",
                model="m",
                system_text="s",
                user_text="u",
                images=None,
            )

            headers = mock_session.post.call_args.kwargs["headers"]
            assert "Authorization" not in headers

    def test_chat_completion_raises_on_http_error(self):
        with patch("src.vlm.client._get_session") as mock_get_session:
            mock_session = MagicMock()
            bad_resp = MagicMock()
            bad_resp.ok = False
            bad_resp.status_code = 500
            bad_resp.text = "internal error"
            mock_session.post.return_value = bad_resp
            mock_get_session.return_value = mock_session

            with pytest.raises(RuntimeError, match="HTTP 500"):
                chat_completion(
                    base_url="https://gateway.example.com",
                    api_token="tok",
                    model="m",
                    system_text="s",
                    user_text="u",
                    images=None,
                )

    def test_repeated_calls_from_same_thread_reuse_session_object(self):
        calls = []
        real_get_session = _get_session

        def spy_get_session():
            s = real_get_session()
            calls.append(id(s))
            return s

        with patch("src.vlm.client._get_session", side_effect=spy_get_session):
            with patch("requests.Session.post", return_value=_fake_response()):
                for _ in range(3):
                    chat_completion(
                        base_url="https://gateway.example.com",
                        api_token="tok",
                        model="m",
                        system_text="s",
                        user_text="u",
                        images=None,
                    )

        assert len(set(calls)) == 1  # same Session object across all 3 calls
