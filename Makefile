# Comandos del proyecto. Los objetivos data/train/serve/monitor se agregan en sus fases.
.PHONY: setup test lint format typecheck

setup:  ## Instala Python 3.11 y dependencias exactas desde uv.lock
	uv sync --locked

test:  ## Ejecuta la suite de tests
	uv run pytest

lint:  ## Lint + formato (verificación) + tipos
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:  ## Aplica formato y autocorrecciones
	uv run ruff format .
	uv run ruff check --fix .
