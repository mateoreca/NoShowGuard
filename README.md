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

El CSV crudo va en `data/raw/data.csv` y no se versiona. Fuente, licencia y reglas de limpieza se documentan en la Fase 1.
