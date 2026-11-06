from __future__ import annotations

import httpx2
import pytest
from conftest import USAGE, make_record, read_log, sse_chunk, sse_response

from llm_starter.chat import format_summary
from llm_starter.usage_log import SessionStats

NO_COST_USAGE = {key: value for key, value in USAGE.items() if key != "cost"}


def error(status):
    return httpx2.Response(status, json={"error": {"code": status, "message": "boom"}})


def test_answer_is_printed_and_history_grows(make_session, capsys):
    chat = make_session()
    chat.session.ask("hi")
    assert capsys.readouterr().out == "Hello\n"
    assert chat.session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_history_is_sent_on_the_next_call(make_session):
    chat = make_session()
    chat.session.ask("one")
    chat.session.ask("two")
    assert [m["content"] for m in chat.api.requests[1]["messages"]] == ["one", "Hello", "two"]


def test_every_call_logs_one_line_without_prompt_or_answer_text(make_session):
    chat = make_session()
    chat.session.ask("my secret prompt")
    chat.session.ask("another prompt")
    assert len(read_log(chat.log_path)) == 2
    raw = chat.log_path.read_text(encoding="utf-8")
    assert "secret" not in raw and "another" not in raw and "Hello" not in raw


def test_api_error_is_one_line_logged_and_the_session_continues(make_session, capsys):
    chat = make_session(error(429))
    assert chat.session.ask("first").ok is False

    captured = capsys.readouterr()
    assert captured.err == "error: HTTP 429 (rate limited): boom\n"

    assert chat.session.ask("second").ok is True
    first, second = read_log(chat.log_path)
    assert first["ok"] is False and first["error"].startswith("HTTP 429")
    assert second["ok"] is True


def test_failed_turn_is_not_kept_in_history(make_session):
    chat = make_session(error(500))
    chat.session.ask("lost")
    chat.session.ask("kept")
    assert [m["content"] for m in chat.api.requests[1]["messages"]] == ["kept"]


def test_interrupted_answer_is_logged_and_not_kept_in_history(make_session, capsys, monkeypatch):
    chat = make_session()

    def interrupt(text):
        if text.strip():
            raise KeyboardInterrupt

    monkeypatch.setattr("llm_starter.chat._write", interrupt)
    assert chat.session.ask("hi").cancelled is True
    monkeypatch.undo()

    assert "[cancelled]" in capsys.readouterr().err
    assert chat.session.history == []
    assert read_log(chat.log_path)[0]["error"] == "cancelled by user"


def test_empty_answer_is_reported_and_not_kept(make_session, capsys):
    chat = make_session(sse_response(sse_chunk(finish_reason="length")))
    chat.session.ask("hi")
    assert "empty answer, finish_reason=length" in capsys.readouterr().err
    assert chat.session.history == []


def test_unwritable_log_warns_and_keeps_the_session_alive(make_session, capsys):
    chat = make_session()
    chat.log_path.parent.parent.joinpath("logs").write_text("a file where the directory goes")
    assert chat.session.ask("hi").ok is True
    assert "could not write the usage log" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["/model", "/modelo", "/MODEL"])
def test_model_shows_current(make_session, capsys, command):
    chat = make_session(fallbacks=("x/y",))
    assert chat.session.handle_command(command) is True
    assert capsys.readouterr().out == "Model: provider/primary (fallbacks: x/y)\n"


@pytest.mark.parametrize("command", ["/model", "/modelo"])
def test_model_switch_keeps_fallbacks_and_history(make_session, command):
    chat = make_session(fallbacks=("x/y",))
    chat.session.ask("one")
    chat.session.handle_command(f"{command} other/Model:free")
    chat.session.ask("two")

    request = chat.api.requests[1]
    assert request["model"] == "other/Model:free"
    assert request["models"] == ["x/y"]
    assert len(request["messages"]) == 3
    assert read_log(chat.log_path)[1]["requested_model"] == "other/Model:free"


def test_model_rejects_extra_words(make_session, capsys):
    chat = make_session()
    chat.session.handle_command("/model a b")
    assert chat.session.model == "provider/primary"
    assert "Usage" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["/cost", "/custo"])
def test_cost_prints_the_summary(make_session, capsys, command):
    chat = make_session()
    chat.session.ask("hi")
    capsys.readouterr()
    chat.session.handle_command(command)

    out = capsys.readouterr().out
    assert "calls:          1 (0 failed)" in out
    assert "tokens in/out:  10 / 3" in out
    assert "total cost:     $0.000420" in out
    assert "models seen:    provider/answered" in out


def test_cost_when_the_api_reports_none(make_session, capsys):
    events = [sse_chunk(content="x", finish_reason="stop"), sse_chunk(usage=NO_COST_USAGE)]
    chat = make_session(sse_response(*events))
    chat.session.ask("hi")
    capsys.readouterr()
    chat.session.handle_command("/cost")
    assert "total cost:     n/a (no cost reported)" in capsys.readouterr().out


def test_summary_flags_a_partial_cost():
    stats = SessionStats()
    stats.add(make_record(cost=0.001))
    stats.add(make_record(cost=None))
    text = format_summary(stats.summary())
    assert "$0.001000 (excludes 1 call(s) with no cost reported)" in text


def test_summary_of_an_empty_session():
    text = format_summary(SessionStats().summary())
    assert "calls:          0 (0 failed)" in text
    assert "total cost:     n/a\n" in text
    assert "models seen:    none" in text


def test_clear_empties_history_but_keeps_stats(make_session):
    chat = make_session()
    chat.session.ask("hi")
    chat.session.handle_command("/clear")
    assert chat.session.history == []
    assert chat.session.stats.summary().calls == 1


def test_help_lists_every_command(make_session, capsys):
    make_session().session.handle_command("/help")
    out = capsys.readouterr().out
    for word in ("/model", "/modelo", "/cost", "/custo", "/clear", "/help", "/quit"):
        assert word in out


def test_quit_ends_the_session(make_session):
    assert make_session().session.handle_command("/quit") is False


def test_unknown_command_never_reaches_the_api(make_session, capsys):
    chat = make_session()
    assert chat.session.handle_command("/nope") is True
    assert "Unknown command" in capsys.readouterr().err
    assert chat.api.requests == []
