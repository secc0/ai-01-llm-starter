"""JSONL usage log and in-memory session statistics.

Records carry metadata only: prompt and response text never reach this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import median


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class UsageRecord:
    """One API call. Field names and order are the log schema."""

    ts: str
    requested_model: str
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None
    ttft_ms: float | None
    latency_ms: float
    finish_reason: str | None
    ok: bool
    error: str | None


class UsageLog:
    """Appends one JSON object per line, creating parent directories as needed."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, record: UsageRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")


@dataclass(frozen=True)
class SessionSummary:
    calls: int
    failed: int
    prompt_tokens: int
    completion_tokens: int
    total_cost: float | None  # None when no call reported a cost
    calls_without_cost: int  # successful calls that reported no cost
    median_latency_ms: float | None
    median_ttft_ms: float | None
    models: list[str]


@dataclass
class SessionStats:
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    def summary(self) -> SessionSummary:
        # Failed calls are counted but kept out of the medians: a fast 429
        # would otherwise make the session look quicker than it was.
        succeeded = [r for r in self.records if r.ok]
        costs = [r.cost for r in succeeded if r.cost is not None]
        latencies = [r.latency_ms for r in succeeded]
        ttfts = [r.ttft_ms for r in succeeded if r.ttft_ms is not None]
        return SessionSummary(
            calls=len(self.records),
            failed=len(self.records) - len(succeeded),
            prompt_tokens=sum(r.prompt_tokens or 0 for r in succeeded),
            completion_tokens=sum(r.completion_tokens or 0 for r in succeeded),
            total_cost=sum(costs) if costs else None,
            calls_without_cost=len(succeeded) - len(costs),
            median_latency_ms=median(latencies) if latencies else None,
            median_ttft_ms=median(ttfts) if ttfts else None,
            models=list(dict.fromkeys(r.model for r in succeeded if r.model)),
        )
