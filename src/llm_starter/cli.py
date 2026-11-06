"""Entry point: chat REPL, one-shot prompts (-p) and the `models` command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from llm_starter.chat import ChatSession, format_summary
from llm_starter.client import ModelListError, list_models, make_client
from llm_starter.config import ConfigError, load_settings
from llm_starter.usage_log import UsageLog

CTRL_D = "\x04"  # the Windows console passes Ctrl-D through as a character


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-starter",
        description="Chat with OpenRouter models from the terminal.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        dest="prompts",
        action="append",
        metavar="TEXT",
        help="run a prompt and exit; repeat to run several in one session",
    )
    commands = parser.add_subparsers(dest="command")
    models = commands.add_parser("models", help="list model ids (no API key needed)")
    models.add_argument("--free", action="store_true", help='keep only ids ending in ":free"')
    return parser


def run_models(free_only: bool) -> int:
    try:
        ids = list_models(free_only=free_only)
    except ModelListError as exc:
        print(f"error: could not list models: {exc}", file=sys.stderr)
        return 1
    print("\n".join(ids))
    return 0


def run_prompts(session: ChatSession, prompts: Sequence[str]) -> int:
    """Run every prompt, even after a failure; exit non-zero if any failed."""
    exit_code = 0
    for prompt in prompts:
        print(f"you> {prompt}")
        if not session.ask(prompt).ok:
            exit_code = 1
    return exit_code


def run_repl(session: ChatSession) -> int:
    print(f"Model: {session.model}. Type /help for commands.")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nUse /quit or Ctrl-D to exit.")
            continue

        if line == CTRL_D:
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if not session.handle_command(line):
                return 0
        else:
            session.ask(line)


def main(argv: Sequence[str] | None = None) -> int:
    # Windows defaults to a legacy code page when output is piped, which would
    # crash on the first emoji in an answer.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "models":
        return run_models(args.free)
    if args.prompts and not all(prompt.strip() for prompt in args.prompts):
        parser.error("-p needs non-empty text")

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    session = ChatSession(
        make_client(settings.api_key),
        model=settings.model_primary,
        fallbacks=settings.model_fallbacks,
        usage_log=UsageLog(settings.log_path),
    )
    try:
        exit_code = run_prompts(session, args.prompts) if args.prompts else run_repl(session)
    except KeyboardInterrupt:
        exit_code = 130
    print(format_summary(session.stats.summary()))
    return exit_code
