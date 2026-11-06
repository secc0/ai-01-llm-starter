"""Shared fixtures. Nothing here touches the network or needs an API key."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx2  # the HTTP library the openai SDK requires; see `pip show openai`
import openai
import pytest

from llm_starter.chat import ChatSession
from llm_starter.client import make_client
from llm_starter.usage_log import UsageLog, UsageRecord

ENV_VARS = ("OPENROUTER_API_KEY", "MODEL_PRIMARY", "MODEL_FALLBACKS", "LLM_LOG_PATH")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Run each test in an empty directory, with no settings, restoring os.environ.

    load_dotenv writes to os.environ directly, which monkeypatch can't undo.
    """
    saved = dict(os.environ)
    for name in ENV_VARS:
        os.environ.pop(name, None)
    monkeypatch.chdir(tmp_path)
    yield
    os.environ.clear()
    os.environ.update(saved)


def sse_chunk(*, model="provider/answered", content=None, finish_reason=None, usage=None):
    """One OpenRouter chunk. With no content or finish_reason, `choices` is empty."""
    choices = []
    if content is not None or finish_reason is not None:
        delta = {"role": "assistant", "content": content or ""}
        choices = [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    return {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": choices,
        **({"usage": usage} if usage else {}),
    }


USAGE = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13, "cost": 0.00042}


def sse_body(*events) -> bytes:
    lines = [": OPENROUTER PROCESSING\n\n"]  # keep-alive comment the real API sends
    lines += [f"data: {json.dumps(event)}\n\n" for event in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def sse_response(*events) -> httpx2.Response:
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=sse_body(*events)
    )


def default_events():
    return [
        sse_chunk(content="Hel"),
        sse_chunk(content="lo"),
        sse_chunk(content="", finish_reason="stop"),
        sse_chunk(usage=USAGE),
    ]


class FakeApi:
    """Real SDK client over a mock transport, so real parsing code runs.

    Each entry in `responses` is an httpx2.Response or an exception to raise.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        transport = httpx2.MockTransport(self._handle)
        self.client = make_client(
            "test-key", http_client=openai.DefaultHttpxClient(transport=transport)
        )

    def _handle(self, request):
        self.requests.append(json.loads(request.content))
        response = self.responses.pop(0) if self.responses else sse_response(*default_events())
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def make_session(tmp_path):
    def build(*responses, model="provider/primary", fallbacks=()):
        api = FakeApi(*responses)
        log = UsageLog(tmp_path / "logs" / "usage.jsonl")
        session = ChatSession(api.client, model=model, fallbacks=fallbacks, usage_log=log)
        return SimpleNamespace(session=session, api=api, log_path=log.path)

    return build


def read_log(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_record(**overrides) -> UsageRecord:
    values = {
        "ts": "2026-01-01T00:00:00.000Z",
        "requested_model": "provider/primary",
        "model": "provider/primary",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.001,
        "ttft_ms": 100.0,
        "latency_ms": 500.0,
        "finish_reason": "stop",
        "ok": True,
        "error": None,
    }
    return UsageRecord(**{**values, **overrides})
