"""Streaming assembly, usage capture, request payload and error mapping.

These run the real SDK over a mock transport, so SSE parsing and the bytes sent
to the API are exercised too.
"""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from types import SimpleNamespace

import httpx2
import pytest
from conftest import USAGE, FakeApi, sse_chunk, sse_response

from llm_starter import client as client_module
from llm_starter.client import ModelListError, list_models, make_client, stream_chat

MESSAGES = [{"role": "user", "content": "hi"}]


def call(api, **kwargs):
    return stream_chat(api.client, messages=MESSAGES, model="provider/primary", **kwargs)


def error_response(status, message="boom", **extra):
    return httpx2.Response(status, json={"error": {"code": status, "message": message, **extra}})


# Streaming assembly


def test_assembles_text_in_order_and_streams_each_delta():
    seen = []
    result = call(FakeApi(), on_text=seen.append)
    assert result.text == "Hello"
    assert seen == ["Hel", "lo"]


def test_empty_content_deltas_are_not_forwarded():
    seen = []
    events = [sse_chunk(content=""), sse_chunk(content="x"), sse_chunk(finish_reason="stop")]
    call(FakeApi(sse_response(*events)), on_text=seen.append)
    assert seen == ["x"]


def test_finish_reason_on_a_later_chunk_than_the_text():
    assert call(FakeApi()).record.finish_reason == "stop"


# Usage capture


def test_captures_usage_from_a_last_chunk_with_empty_choices():
    record = call(FakeApi()).record
    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == (10, 3, 13)
    assert record.cost == 0.00042
    assert record.ok is True and record.error is None


def test_captures_usage_from_a_last_chunk_with_an_empty_delta():
    events = [sse_chunk(content="x"), sse_chunk(content="", finish_reason="stop", usage=USAGE)]
    record = call(FakeApi(sse_response(*events))).record
    assert record.total_tokens == 13
    assert record.finish_reason == "stop"


def test_missing_cost_is_none_not_an_estimate():
    usage = {key: value for key, value in USAGE.items() if key != "cost"}
    events = [sse_chunk(content="x", finish_reason="stop"), sse_chunk(usage=usage)]
    record = call(FakeApi(sse_response(*events))).record
    assert record.cost is None
    assert record.total_tokens == 13


def test_zero_cost_is_kept():
    events = [sse_chunk(content="x", finish_reason="stop"), sse_chunk(usage={**USAGE, "cost": 0})]
    assert call(FakeApi(sse_response(*events))).record.cost == 0


def test_stream_without_usage_still_succeeds():
    events = [sse_chunk(content="x", finish_reason="stop")]
    record = call(FakeApi(sse_response(*events))).record
    assert record.ok is True
    assert record.total_tokens is None and record.cost is None


def test_logs_the_model_that_answered_not_the_requested_one():
    events = [sse_chunk(model="provider/fallback", content="x", finish_reason="stop")]
    record = call(FakeApi(sse_response(*events)), fallbacks=["provider/fallback"]).record
    assert record.requested_model == "provider/primary"
    assert record.model == "provider/fallback"


def test_ttft_comes_from_first_non_empty_delta_and_precedes_latency(monkeypatch):
    ticks = iter([0.0, 1.0, 3.0])  # start, first text, end of stream
    monkeypatch.setattr(client_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    events = [
        sse_chunk(content=""),
        sse_chunk(content="x"),
        sse_chunk(content="y", finish_reason="stop"),
    ]
    record = call(FakeApi(sse_response(*events))).record
    assert record.ttft_ms == 1000.0
    assert record.latency_ms == 3000.0


def test_ttft_is_none_when_no_text_arrives():
    record = call(FakeApi(sse_response(sse_chunk(finish_reason="stop")))).record
    assert record.ttft_ms is None


# Request payload


def test_payload_without_fallbacks():
    api = FakeApi()
    call(api)
    assert api.requests == [{"model": "provider/primary", "messages": MESSAGES, "stream": True}]


def test_payload_with_fallbacks():
    api = FakeApi()
    call(api, fallbacks=["a/b", "c/d"])
    assert api.requests[0]["models"] == ["a/b", "c/d"]
    assert api.requests[0]["model"] == "provider/primary"


def test_deprecated_usage_options_are_not_sent():
    api = FakeApi()
    call(api, fallbacks=["a/b"])
    assert "stream_options" not in api.requests[0]
    assert "usage" not in api.requests[0]


def test_client_points_at_openrouter_without_sdk_retries():
    client = make_client("k")
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert client.max_retries == 0


def test_a_429_is_sent_only_once():
    api = FakeApi(error_response(429))
    call(api)
    assert len(api.requests) == 1


# Error handling


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "HTTP 401 (invalid API key): boom"),
        (402, "HTTP 402 (insufficient credits): boom"),
        (429, "HTTP 429 (rate limited): boom"),
        (503, "HTTP 503: boom"),
    ],
)
def test_status_errors_become_a_one_line_result(status, expected):
    result = call(FakeApi(error_response(status)))
    assert result.ok is False
    assert result.record.error == expected
    assert result.record.model is None and result.record.cost is None


def test_network_error():
    result = call(FakeApi(httpx2.ConnectError("refused")))
    assert result.ok is False
    assert result.record.error.startswith("network error:")


def test_timeout_is_a_network_error():
    result = call(FakeApi(httpx2.ReadTimeout("slow")))
    assert result.record.error.startswith("network error:")


def test_error_sent_mid_stream_keeps_partial_text_and_reports_the_status():
    events = [
        sse_chunk(content="par"),
        {
            **sse_chunk(content="", finish_reason="error"),
            "error": {"code": 502, "message": "upstream died"},
        },
    ]
    result = call(FakeApi(sse_response(*events)))
    assert result.ok is False
    assert result.text == "par"
    assert result.record.error == "HTTP 502: upstream died"


def test_connection_lost_mid_stream():
    class BrokenStream(httpx2.SyncByteStream):
        def __iter__(self):
            yield f"data: {json.dumps(sse_chunk(content='par'))}\n\n".encode()
            raise httpx2.ReadError("reset")

    response = httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=BrokenStream()
    )
    result = call(FakeApi(response))
    assert result.ok is False
    assert result.text == "par"
    assert result.record.error.startswith("network error:")


def test_malformed_stream_data_is_an_error_result():
    response = httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=b"data: {not json\n\n"
    )
    result = call(FakeApi(response))
    assert result.ok is False
    assert "invalid response" in result.record.error


def test_finish_reason_error_without_an_error_body_is_not_ok():
    result = call(FakeApi(sse_response(sse_chunk(content="x", finish_reason="error"))))
    assert result.ok is False
    assert "finish_reason=error" in result.record.error


def test_multiline_error_message_stays_on_one_line():
    result = call(FakeApi(error_response(400, "line one\nline two " + "x" * 1000)))
    assert "\n" not in result.record.error
    assert len(result.record.error) <= 300


def test_error_metadata_quoting_the_prompt_is_dropped():
    response = error_response(
        403, "flagged", metadata={"flagged_input": "SECRET PROMPT", "reasons": ["x"]}
    )
    assert "SECRET PROMPT" not in call(FakeApi(response)).record.error


def test_ctrl_c_returns_partial_text_and_closes_the_stream():
    class FakeStream:
        closed = False

        def __iter__(self):
            delta = SimpleNamespace(content="par")
            choice = SimpleNamespace(delta=delta, finish_reason=None)
            yield SimpleNamespace(model="provider/primary", usage=None, choices=[choice])
            raise KeyboardInterrupt

        def close(self):
            FakeStream.closed = True

    sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream()))
    )
    result = stream_chat(sdk, messages=MESSAGES, model="provider/primary")
    assert result.cancelled is True
    assert result.ok is False
    assert result.text == "par"
    assert result.record.error == "cancelled by user"
    assert FakeStream.closed is True


def test_ctrl_c_before_the_stream_opens():
    def interrupted(**_):
        raise KeyboardInterrupt

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=interrupted)))
    assert stream_chat(sdk, messages=MESSAGES, model="m").cancelled is True


# Model list


MODELS_BODY = json.dumps({"data": [{"id": "z/b:free"}, {"id": "a/paid"}, {"id": "b/c:free"}]})


def fake_urlopen(monkeypatch, result):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        if isinstance(result, Exception):
            raise result
        return BytesIO(result.encode())

    monkeypatch.setattr(client_module.urllib.request, "urlopen", urlopen)
    return calls


def test_list_models_is_sorted_and_uses_the_base_url(monkeypatch):
    calls = fake_urlopen(monkeypatch, MODELS_BODY)
    assert list_models(base_url="http://x/v1") == ["a/paid", "b/c:free", "z/b:free"]
    assert calls == ["http://x/v1/models"]


def test_list_models_free_only(monkeypatch):
    fake_urlopen(monkeypatch, MODELS_BODY)
    assert list_models(free_only=True) == ["b/c:free", "z/b:free"]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (urllib.error.HTTPError("u", 503, "Unavailable", None, None), "HTTP 503"),
        (urllib.error.URLError("no route"), "network error"),
        (TimeoutError("timed out"), "network error"),
        ("not json", "unexpected response"),
        ('{"oops": []}', "unexpected response"),
    ],
)
def test_list_models_failures(monkeypatch, result, expected):
    fake_urlopen(monkeypatch, result)
    with pytest.raises(ModelListError, match=expected):
        list_models()
