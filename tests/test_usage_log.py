from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timedelta

from conftest import make_record, read_log

from llm_starter.usage_log import SessionStats, UsageLog, UsageRecord, utc_now_iso

SCHEMA = [
    "ts",
    "requested_model",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost",
    "ttft_ms",
    "latency_ms",
    "finish_reason",
    "ok",
    "error",
]


def stats_of(*records):
    stats = SessionStats()
    for record in records:
        stats.add(record)
    return stats.summary()


def test_record_fields_are_exactly_the_schema():
    assert [f.name for f in fields(UsageRecord)] == SCHEMA


def test_ts_is_utc_iso_8601():
    parsed = datetime.fromisoformat(utc_now_iso())
    assert parsed.utcoffset() == timedelta(0)


def test_append_creates_directories_and_writes_one_line_per_record(tmp_path):
    path = tmp_path / "a" / "b" / "usage.jsonl"
    log = UsageLog(path)
    log.append(make_record())
    log.append(make_record(ok=False, error="HTTP 429: x", cost=None))

    raw = path.read_bytes()
    assert raw.count(b"\n") == 2 and b"\r" not in raw
    first, second = read_log(path)
    assert list(first) == SCHEMA
    assert second["ok"] is False and second["cost"] is None


def test_non_ascii_is_written_as_utf8(tmp_path):
    path = tmp_path / "usage.jsonl"
    UsageLog(path).append(make_record(error="ação"))
    assert json.loads(path.read_bytes().decode("utf-8"))["error"] == "ação"


def test_summary_of_an_empty_session():
    summary = stats_of()
    assert summary.calls == 0
    assert summary.total_cost is None
    assert summary.median_latency_ms is None and summary.median_ttft_ms is None
    assert summary.models == []


def test_summary_totals_and_medians():
    summary = stats_of(
        make_record(latency_ms=100.0, ttft_ms=10.0, cost=0.001, model="a/b"),
        make_record(latency_ms=300.0, ttft_ms=30.0, cost=0.002, model="c/d"),
        make_record(latency_ms=900.0, ttft_ms=90.0, cost=0.003, model="a/b"),
    )
    assert summary.calls == 3 and summary.failed == 0
    assert (summary.prompt_tokens, summary.completion_tokens) == (30, 15)
    assert round(summary.total_cost, 6) == 0.006
    assert summary.median_latency_ms == 300.0
    assert summary.median_ttft_ms == 30.0
    assert summary.models == ["a/b", "c/d"]


def test_failed_calls_are_counted_but_stay_out_of_medians():
    failed = make_record(
        ok=False,
        error="x",
        latency_ms=1.0,
        ttft_ms=None,
        cost=None,
        model=None,
        prompt_tokens=None,
        completion_tokens=None,
    )
    summary = stats_of(make_record(latency_ms=500.0), failed)
    assert summary.calls == 2 and summary.failed == 1
    assert summary.median_latency_ms == 500.0
    assert summary.calls_without_cost == 0


def test_missing_cost_is_reported_not_treated_as_zero():
    summary = stats_of(make_record(cost=0.5), make_record(cost=None))
    assert summary.total_cost == 0.5
    assert summary.calls_without_cost == 1


def test_all_costs_missing_gives_no_total():
    summary = stats_of(make_record(cost=None))
    assert summary.total_cost is None
    assert summary.calls_without_cost == 1


def test_free_calls_total_zero_not_none():
    assert stats_of(make_record(cost=0)).total_cost == 0


def test_missing_token_counts_do_not_break_totals():
    summary = stats_of(make_record(prompt_tokens=None, completion_tokens=None))
    assert (summary.prompt_tokens, summary.completion_tokens) == (0, 0)
