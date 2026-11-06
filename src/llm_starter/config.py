"""Build Settings from the environment or a .env file in the working directory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DEFAULT_LOG_PATH = "logs/usage.jsonl"
REQUIRED_VARS = ("OPENROUTER_API_KEY", "MODEL_PRIMARY")


class ConfigError(Exception):
    """A required setting is missing."""


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model_primary: str
    model_fallbacks: tuple[str, ...] = ()
    log_path: Path = Path(DEFAULT_LOG_PATH)


def parse_fallbacks(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated list, trimming blanks and dropping empty items."""
    return tuple(item.strip() for item in (raw or "").split(",") if item.strip())


def load_settings() -> Settings:
    """Read settings; real environment variables win over .env."""
    # Guarded because load_dotenv("") would fall back to searching upward from
    # this source file, loading a .env the working directory doesn't own.
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    values = {name: os.environ.get(name, "").strip() for name in REQUIRED_VARS}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ConfigError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in the environment or in .env (see .env.example)."
        )

    return Settings(
        api_key=values["OPENROUTER_API_KEY"],
        model_primary=values["MODEL_PRIMARY"],
        model_fallbacks=parse_fallbacks(os.environ.get("MODEL_FALLBACKS")),
        log_path=Path(os.environ.get("LLM_LOG_PATH", "").strip() or DEFAULT_LOG_PATH),
    )
