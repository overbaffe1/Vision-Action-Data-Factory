from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .images import ImageInput, build_user_content
from .parsing import ChatCompletionParsed, parse_chat_completion

# requests.Session is documented as NOT thread-safe to share across threads
# (https://github.com/psf/requests/issues/2766) — the recommended pattern is
# one Session per thread. thread-local storage gives each of the pipeline's
# many VLM worker threads its own connection pool to the VLM gateway, so
# repeated calls to the same host reuse TCP/TLS connections instead of
# paying a fresh handshake on every single request (relevant here: up to a
# few hundred worker threads issuing thousands-to-millions of HTTPS calls to
# one router endpoint over a run).
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Returns this thread's requests.Session, creating it on first use."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            # Retries here are for transport-level failures the caller's own
            # retry loop wouldn't see cleanly (connection resets, etc.);
            # the pipeline's _vlm_call_with_retry already retries at the
            # request level with backoff, so keep this small.
            max_retries=Retry(total=0),
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return session


def chat_completion(
    base_url: str,
    api_token: str,
    model: str,
    system_text: str,
    user_text: str,
    images: Sequence[ImageInput] | None = None,
    *,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    timeout_s: int = 60,
    enforce_json_object: bool = True,
) -> ChatCompletionParsed:
    """Calls a Chat Completions-compatible endpoint and returns a parsed response."""
    headers = {
        "Content-Type": "application/json",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": build_user_content(user_text, images=images)},
        ],
        "temperature": temperature,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if enforce_json_object:
        payload["response_format"] = {"type": "json_object"}

    session = _get_session()
    urls = [
        base_url.rstrip("/") + "/v1/chat/completions",
        base_url.rstrip("/") + "/chat/completions",
    ]
    last_error: RuntimeError | None = None

    for index, url in enumerate(urls):
        r = session.post(url, headers=headers, json=payload, timeout=timeout_s)
        if r.ok:
            result = parse_chat_completion(r.text)
            return result

        body = (r.text or "").strip()
        last_error = RuntimeError(f"HTTP {r.status_code}: {body}")

        # Some direct backends expose /chat/completions instead of the OpenAI
        # /v1/chat/completions path. Fall back only on 404 from the first path.
        if not (index == 0 and r.status_code == 404):
            raise last_error

    assert last_error is not None
    raise last_error


def run_chat_completions_parallel(
    requests_data: Sequence[dict],
    *,
    max_workers: int | None = None,
) -> list[ChatCompletionParsed]:
    """Runs multiple chat completion requests in parallel and returns results in input order."""

    if not requests_data:
        return []

    workers = max_workers or min(32, len(requests_data))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                chat_completion,
                item["base_url"],
                item["api_token"],
                item["model"],
                item["system_text"],
                item["user_text"],
                item.get("images"),
                temperature=item.get("temperature", 0.1),
                max_tokens=item.get("max_tokens", None),
                timeout_s=item.get("timeout_s", 60),
                enforce_json_object=item.get("enforce_json_object", True),
            )
            for item in requests_data
        ]

        return [future.result() for future in futures]
