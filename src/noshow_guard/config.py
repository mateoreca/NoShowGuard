"""Configuración central del proyecto: rutas, semilla y reglas de datos."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Rutas del proyecto, relativas a la raíz del repositorio."""

    root: Path = PROJECT_ROOT
    data_raw: Path = field(default=PROJECT_ROOT / "data" / "raw")
    data_processed: Path = field(default=PROJECT_ROOT / "data" / "processed")
    raw_csv: Path = field(default=PROJECT_ROOT / "data" / "raw" / "data.csv")
    clean_parquet: Path = field(
        default=PROJECT_ROOT / "data" / "processed" / "appointments_clean.parquet"
    )
    reports: Path = field(default=PROJECT_ROOT / "reports")
    cleaning_report: Path = field(default=PROJECT_ROOT / "reports" / "data_cleaning.json")
    features_dir: Path = field(default=PROJECT_ROOT / "data" / "processed" / "features")
    split_summary: Path = field(default=PROJECT_ROOT / "reports" / "split_summary.json")


@dataclass(frozen=True)
class DataRules:
    """Rangos válidos usados por la limpieza y la validación (y luego por el simulador)."""

    age_min: int = 0
    # 115 años aparece en 5 filas de 2 pacientes; se considera error de captura.
    age_max: int = 110
    handicap_max: int = 4


@dataclass(frozen=True)
class SplitConfig:
    """Cortes del split cronológico por fecha de cita (inclusive en ambos extremos)."""

    train_end: date = date(2016, 5, 20)
    val_start: date = date(2016, 5, 24)
    val_end: date = date(2016, 5, 31)
    test_start: date = date(2016, 6, 1)


SEED: int = 42
PATHS = Paths()
DATA_RULES = DataRules()
SPLIT = SplitConfig()

# Citas con antelación menor a esto quedan fuera del alcance del modelo (mismo día).
MIN_LEAD_DAYS: int = 1

# SHA-256 del CSV de Kaggle (joniarroba/noshowappointments, versión 5) usado en el proyecto.
RAW_SHA256 = "9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d"
