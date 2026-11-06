"""Streaming OpenRouter calls with fallbacks and usage capture. No CLI code here."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import openai
from openai import OpenAI

from llm_starter.usage_log import UsageRecord, utc_now_iso

BASE_URL = "https://openrouter.ai/api/v1"
CANCELLED = "cancelled by user"
MAX_ERROR_CHARS = 300

_STATUS_HINTS = {
    401: "invalid API key",
    402: "insufficient credits",
    429: "rate limited",
}


class ModelListError(Exception):
    """The public model list could not be fetched."""


@dataclass(frozen=True)
class CallResult:
    text: str
    record: UsageRecord

    @property
    def ok(self) -> bool:
        return self.record.ok

    @property
    def cancelled(self) -> bool:
        return self.record.error == CANCELLED


def make_client(api_key: str, **kwargs: Any) -> OpenAI:
    """Create the SDK client.

    SDK retries are off: they would hide calls from the usage log and spend
    free-tier quota without the user knowing.
    """
    return OpenAI(base_url=BASE_URL, api_key=api_key, max_retries=0, **kwargs)


def _one_line(text: str) -> str:
    return " ".join(text.split())[:MAX_ERROR_CHARS]


def _body_message(body: object) -> str | None:
    message = body.get("message") if isinstance(body, dict) else None
    return message if isinstance(message, str) and message else None


def _format_status(status: int, message: str) -> str:
    hint = _STATUS_HINTS.get(status)
    prefix = f"HTTP {status} ({hint})" if hint else f"HTTP {status}"
    return _one_line(f"{prefix}: {message}")


def describe_error(exc: Exception) -> str:
    """Return a one-line description.

    Reads only the error's `message`: str(exc) embeds the whole response body,
    whose metadata can quote the prompt, and this text goes to the usage log.
    """
    if isinstance(exc, openai.APIStatusError):
        message = _body_message(exc.body) or "request failed"
        return _format_status(exc.status_code, message)
    if isinstance(exc, openai.APIConnectionError):
        return _one_line(f"network error: {exc.message}")
    if isinstance(exc, openai.APIError):
        # Errors sent after the stream started arrive on an HTTP 200 response,
        # with the real status in the error body's `code`.
        message = _body_message(exc.body) or "stream failed"
        code = exc.code or ""
        return _format_status(int(code), message) if code.isdigit() else _one_line(message)
    return f"{type(exc).__name__}: invalid response from the API"


def stream_chat(
    client: OpenAI,
    *,
    messages: Sequence[dict[str, str]],
    model: str,
    fallbacks: Iterable[str] = (),
    on_text: Callable[[str], None] | None = None,
) -> CallResult:
    """Run one streaming completion. API failures and Ctrl-C are returned, not raised."""
    kwargs: dict[str, Any] = {"model": model, "messages": list(messages), "stream": True}
    fallback_models = list(fallbacks)
    if fallback_models:
        kwargs["extra_body"] = {"models": fallback_models}

    parts: list[str] = []
    answered_by = finish_reason = usage = ttft_ms = error = stream = None
    ts = utc_now_iso()
    start = time.perf_counter()

    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            answered_by = chunk.model or answered_by
            usage = chunk.usage or usage
            # Text and usage are independent: the last chunk may carry usage
            # with empty `choices`, or an empty delta.
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            finish_reason = choice.finish_reason or finish_reason
            text = choice.delta.content if choice.delta else None
            if text:
                if ttft_ms is None:
                    ttft_ms = _elapsed_ms(start)
                parts.append(text)
                if on_text:
                    on_text(text)
    except KeyboardInterrupt:
        error = CANCELLED
    except (openai.OpenAIError, json.JSONDecodeError) as exc:
        error = describe_error(exc)
    finally:
        if stream is not None:
            stream.close()  # stops generation upstream when cancelled

    latency_ms = _elapsed_ms(start)
    if error is None and finish_reason == "error":
        error = "the provider ended the stream with finish_reason=error"

    record = UsageRecord(
        ts=ts,
        requested_model=model,
        model=answered_by,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        cost=getattr(usage, "cost", None),
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        ok=error is None,
        error=error,
    )
    return CallResult(text="".join(parts), record=record)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def list_models(*, free_only: bool = False, base_url: str = BASE_URL) -> list[str]:
    """Return sorted model ids from the public endpoint (no authentication)."""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=30) as response:
            payload = json.load(response)
        ids = [item["id"] for item in payload["data"]]
    except urllib.error.HTTPError as exc:
        raise ModelListError(f"HTTP {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ModelListError(f"network error: {exc}") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise ModelListError("unexpected response from the models endpoint") from exc
    return sorted(i for i in ids if i.endswith(":free") or not free_only)
