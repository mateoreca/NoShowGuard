# Comandos del proyecto. Los objetivos de fases futuras se agregan cuando existen.
.PHONY: setup test lint format data features train eda

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

data:  ## Verifica el CSV crudo y genera data/processed/appointments_clean.parquet
	uv run python scripts/download_data.py
	uv run python -m noshow_guard.data

features:  ## Features sin fuga + split cronológico -> data/processed/features/*.parquet
	uv run python -m noshow_guard.features

train:  ## Baselines, LightGBM, calibración y umbrales -> models/ + reports/
	uv run python -m noshow_guard.train

eda: data  ## Ejecuta el notebook de EDA y guarda sus salidas
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
