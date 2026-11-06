"""Chat session: message history, slash commands and model switching."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from openai import OpenAI

from llm_starter.client import CallResult, stream_chat
from llm_starter.usage_log import SessionStats, SessionSummary, UsageLog

HELP = """\
/model          show the current primary model (alias: /modelo)
/model <id>     switch the primary model for this session; fallbacks stay the same
/cost           show the session summary (alias: /custo)
/clear          forget the conversation history
/help           show this help
/quit           exit (also Ctrl-D; on Windows press Enter after Ctrl-D or Ctrl-Z)
Ctrl-C          cancel the answer being streamed"""

ALIASES = {"/modelo": "/model", "/custo": "/cost"}


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _cost(summary: SessionSummary) -> str:
    if summary.total_cost is None:
        return "n/a (no cost reported)" if summary.calls_without_cost else "n/a"
    text = f"${summary.total_cost:.6f}"
    if summary.calls_without_cost:
        text += f" (excludes {summary.calls_without_cost} call(s) with no cost reported)"
    return text


def format_summary(summary: SessionSummary) -> str:
    return "\n".join(
        [
            "--- session summary ---",
            f"calls:          {summary.calls} ({summary.failed} failed)",
            f"tokens in/out:  {summary.prompt_tokens} / {summary.completion_tokens}",
            f"total cost:     {_cost(summary)}",
            f"median latency: {_ms(summary.median_latency_ms)}",
            f"median TTFT:    {_ms(summary.median_ttft_ms)}",
            f"models seen:    {', '.join(summary.models) or 'none'}",
        ]
    )


class ChatSession:
    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        fallbacks: Sequence[str] = (),
        usage_log: UsageLog,
    ) -> None:
        self.client = client
        self.model = model
        self.fallbacks = tuple(fallbacks)
        self.usage_log = usage_log
        self.history: list[dict[str, str]] = []
        self.stats = SessionStats()

    def ask(self, prompt: str) -> CallResult:
        """Stream one answer. History only grows when a complete answer arrives."""
        user_message = {"role": "user", "content": prompt}
        result = stream_chat(
            self.client,
            messages=[*self.history, user_message],
            model=self.model,
            fallbacks=self.fallbacks,
            on_text=_write,
        )
        if result.text:
            _write("\n")

        self.stats.add(result.record)
        try:
            self.usage_log.append(result.record)
        except OSError as exc:
            print(f"warning: could not write the usage log: {exc}", file=sys.stderr)

        if result.cancelled:
            print("[cancelled]", file=sys.stderr)
        elif not result.ok:
            print(f"error: {result.record.error}", file=sys.stderr)
        elif not result.text:
            reason = result.record.finish_reason
            print(f"[empty answer, finish_reason={reason}]", file=sys.stderr)
        else:
            assistant_message = {"role": "assistant", "content": result.text}
            self.history += [user_message, assistant_message]
        return result

    def handle_command(self, line: str) -> bool:
        """Run a slash command. Returns False when the session should end."""
        name, _, argument = line.strip().partition(" ")
        name = name.lower()
        name = ALIASES.get(name, name)
        argument = argument.strip()

        if name == "/quit":
            return False
        if name == "/help":
            print(HELP)
        elif name == "/clear":
            self.history.clear()
            print("History cleared.")
        elif name == "/cost":
            print(format_summary(self.stats.summary()))
        elif name == "/model":
            self._model_command(argument)
        else:
            print(f"Unknown command: {name}. Type /help.", file=sys.stderr)
        return True

    def _model_command(self, argument: str) -> None:
        if not argument:
            fallbacks = ", ".join(self.fallbacks) or "none"
            print(f"Model: {self.model} (fallbacks: {fallbacks})")
        elif len(argument.split()) > 1:
            print("Usage: /model <id>", file=sys.stderr)
        else:
            self.model = argument
            print(f"Model switched to {self.model}")
