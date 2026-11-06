from __future__ import annotations

import builtins
import os

import httpx2
import pytest
from conftest import FakeApi, read_log

from llm_starter import cli
from llm_starter.client import ModelListError


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Configure the app and route its SDK client through a fake API."""
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    os.environ["MODEL_PRIMARY"] = "provider/primary"
    os.environ["MODEL_FALLBACKS"] = "x/y"
    os.environ["LLM_LOG_PATH"] = str(tmp_path / "usage.jsonl")
    fake = FakeApi()
    monkeypatch.setattr(cli, "make_client", lambda api_key: fake.client)
    return fake


def feed_input(monkeypatch, *lines):
    """Script input(): strings are returned, exceptions are raised."""
    remaining = iter(lines)

    def fake_input(prompt=""):
        item = next(remaining)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(builtins, "input", fake_input)


def test_help_needs_no_configuration(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    assert "models" in capsys.readouterr().out


def test_missing_configuration_fails_early_with_the_variable_name(capsys):
    assert cli.main(["-p", "hi"]) == 2
    err = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in err and "MODEL_PRIMARY" in err


def test_single_prompt_runs_and_prints_the_summary(api, capsys):
    assert cli.main(["-p", "hi"]) == 0
    out = capsys.readouterr().out
    assert "you> hi\nHello\n" in out
    assert "--- session summary ---" in out
    assert "calls:          1 (0 failed)" in out


def test_repeated_prompts_share_one_session_in_order(api):
    assert cli.main(["-p", "one", "-p", "two", "-p", "three"]) == 0
    assert [m["content"] for m in api.requests[-1]["messages"]] == [
        "one",
        "Hello",
        "two",
        "Hello",
        "three",
    ]


def test_prompts_starting_with_a_slash_go_to_the_model(api):
    assert cli.main(["-p", "/etc/hosts is what?"]) == 0
    assert api.requests[0]["messages"][0]["content"] == "/etc/hosts is what?"


def test_a_failed_prompt_does_not_stop_the_next_but_sets_the_exit_code(api, capsys, tmp_path):
    api.responses.append(httpx2.Response(401, json={"error": {"code": 401, "message": "bad"}}))
    assert cli.main(["-p", "one", "-p", "two"]) == 1
    assert len(api.requests) == 2
    assert "HTTP 401" in capsys.readouterr().err
    assert [r["ok"] for r in read_log(tmp_path / "usage.jsonl")] == [False, True]


def test_blank_prompt_is_rejected(api):
    with pytest.raises(SystemExit) as stop:
        cli.main(["-p", "  "])
    assert stop.value.code == 2


def test_repl_chats_then_quits_with_a_summary(api, monkeypatch, capsys):
    feed_input(monkeypatch, "", "hi", "/model new/model", "again", "/quit")
    assert cli.main([]) == 0
    assert [r["model"] for r in api.requests] == ["provider/primary", "new/model"]
    assert "--- session summary ---" in capsys.readouterr().out


@pytest.mark.parametrize("ending", [EOFError(), "\x04"], ids=["eof", "windows-ctrl-d"])
def test_repl_exits_on_ctrl_d(api, monkeypatch, capsys, ending):
    feed_input(monkeypatch, ending)
    assert cli.main([]) == 0
    assert "--- session summary ---" in capsys.readouterr().out
    assert api.requests == []


def test_ctrl_c_at_the_prompt_keeps_the_repl_alive(api, monkeypatch):
    feed_input(monkeypatch, KeyboardInterrupt(), "hi", "/quit")
    assert cli.main([]) == 0
    assert len(api.requests) == 1


def test_models_needs_no_configuration(monkeypatch, capsys):
    calls = []

    def fake_list(free_only):
        calls.append(free_only)
        return ["a/b:free", "c/d:free"]

    monkeypatch.setattr(cli, "list_models", fake_list)
    assert cli.main(["models", "--free"]) == 0
    assert capsys.readouterr().out == "a/b:free\nc/d:free\n"
    assert calls == [True]


def test_models_failure_is_one_line(monkeypatch, capsys):
    def failing(free_only):
        raise ModelListError("HTTP 503: down")

    monkeypatch.setattr(cli, "list_models", failing)
    assert cli.main(["models"]) == 1
    assert capsys.readouterr().err == "error: could not list models: HTTP 503: down\n"
