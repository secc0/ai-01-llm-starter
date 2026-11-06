.PHONY: setup run test lint demo

ifeq ($(OS),Windows_NT)
PYTHON ?= python
BIN := .venv/Scripts
else
PYTHON ?= python3
BIN := .venv/bin
endif

setup:
	$(PYTHON) -m venv .venv
	$(BIN)/python -m pip install -e ".[dev]"

run:
	$(BIN)/llm-starter

test:
	$(BIN)/python -m pytest -q

lint:
	$(BIN)/python -m ruff check .
	$(BIN)/python -m ruff format --check .

demo:
	$(BIN)/llm-starter -p "Name one prime number between 50 and 60. Reply with the number only." -p "Add 1 to that number. Reply with the number only." -p "Say goodbye in three words of Portuguese."
