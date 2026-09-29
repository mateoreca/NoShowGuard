# noshow-guard

Sistema de predicción de inasistencia (no-show) a citas médicas: modelo calibrado, API de inferencia, monitoreo de drift y decisión de recordatorios.

> Estado: Fase 0 (setup). El README completo se construye en la Fase 8.

## Requisitos

- [uv](https://docs.astral.sh/uv/) 0.8.x (instala Python 3.11 automáticamente)
- GNU Make (en Windows: `winget install ezwinports.make`)

## Inicio rápido

```bash
make setup   # dependencias exactas desde uv.lock
make test
make lint    # ruff + mypy
```

## Datos

Dataset público [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (Kaggle, CC BY-NC-SA 4.0). El CSV va en `data/raw/data.csv` y no se versiona.

```bash
make data   # verifica hash, limpia y valida -> data/processed/appointments_clean.parquet
make eda    # ejecuta notebooks/01_eda.ipynb
```

Reglas de limpieza, fuente y limitaciones: [docs/data_quality.md](docs/data_quality.md).
