from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_starter import config
from llm_starter.config import ConfigError, load_settings, parse_fallbacks


def set_required():
    os.environ["OPENROUTER_API_KEY"] = "env-key"
    os.environ["MODEL_PRIMARY"] = "env/model"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ()),
        ("", ()),
        (" , ,", ()),
        ("a/b", ("a/b",)),
        (" a/b , c/d:free ,, ", ("a/b", "c/d:free")),
    ],
)
def test_parse_fallbacks(raw, expected):
    assert parse_fallbacks(raw) == expected


def test_defaults():
    set_required()
    settings = load_settings()
    assert settings.model_primary == "env/model"
    assert settings.model_fallbacks == ()
    assert settings.log_path == Path("logs/usage.jsonl")


def test_reads_dotenv_from_working_directory(tmp_path):
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=file-key\n"
        "MODEL_PRIMARY=file/model\n"
        "MODEL_FALLBACKS=a/b, c/d\n"
        "LLM_LOG_PATH=custom/usage.jsonl\n"
    )
    settings = load_settings()
    assert settings.api_key == "file-key"
    assert settings.model_primary == "file/model"
    assert settings.model_fallbacks == ("a/b", "c/d")
    assert settings.log_path == Path("custom/usage.jsonl")


def test_real_environment_wins_over_dotenv(tmp_path):
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=file-key\nMODEL_PRIMARY=file/model\n")
    os.environ["MODEL_PRIMARY"] = "env/model"
    settings = load_settings()
    assert settings.model_primary == "env/model"
    assert settings.api_key == "file-key"


def test_dotenv_outside_the_working_directory_tree_is_ignored(monkeypatch):
    """No .env in the cwd must not fall back to searching from the source file."""
    called = []
    monkeypatch.setattr(config, "find_dotenv", lambda usecwd: "")
    monkeypatch.setattr(config, "load_dotenv", lambda *args: called.append(args))
    with pytest.raises(ConfigError):
        load_settings()
    assert called == []


@pytest.mark.parametrize(
    ("present", "missing"),
    [("MODEL_PRIMARY", "OPENROUTER_API_KEY"), ("OPENROUTER_API_KEY", "MODEL_PRIMARY")],
)
def test_missing_variable_is_named(present, missing):
    os.environ[present] = "value"
    with pytest.raises(ConfigError, match=missing) as error:
        load_settings()
    assert present not in str(error.value)


def test_blank_value_counts_as_missing():
    set_required()
    os.environ["MODEL_PRIMARY"] = "   "
    with pytest.raises(ConfigError, match="MODEL_PRIMARY"):
        load_settings()


def test_api_key_is_not_in_repr():
    set_required()
    assert "env-key" not in repr(load_settings())
