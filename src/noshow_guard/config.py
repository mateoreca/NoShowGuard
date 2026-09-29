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
    figures: Path = field(default=PROJECT_ROOT / "reports" / "figures")
    metrics: Path = field(default=PROJECT_ROOT / "reports" / "metrics.json")
    models: Path = field(default=PROJECT_ROOT / "models")
    model_metadata: Path = field(default=PROJECT_ROOT / "models" / "metadata.json")


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


@dataclass(frozen=True)
class CostConfig:
    """Supuestos de costo y efecto de las acciones (NO salen de los datos).

    Unidades relativas: 1 = costo de un recordatorio estándar por WhatsApp.
    Una acción conviene frente a no hacer nada si ``no_show_cost * p * effect > costo``,
    es decir, si ``p > costo / (no_show_cost * effect)``.
    """

    no_show_cost: float = 20.0  # hueco vacío en la agenda
    standard_cost: float = 1.0  # recordatorio estándar
    reinforced_cost: float = 3.0  # recordatorio + solicitud de confirmación (incluye seguimiento)
    standard_effect: float = 0.15  # reducción relativa supuesta del no-show
    reinforced_effect: float = 0.30


@dataclass(frozen=True)
class SearchConfig:
    """Búsqueda aleatoria de hiperparámetros de LightGBM."""

    n_iter: int = 30
    max_estimators: int = 2000
    early_stopping_rounds: int = 100
    calibration_folds: int = 5


SEED: int = 42
PATHS = Paths()
DATA_RULES = DataRules()
SPLIT = SplitConfig()
COSTS = CostConfig()
SEARCH = SearchConfig()

# Citas con antelación menor a esto quedan fuera del alcance del modelo (mismo día).
MIN_LEAD_DAYS: int = 1

# SHA-256 del CSV de Kaggle (joniarroba/noshowappointments, versión 5) usado en el proyecto.
RAW_SHA256 = "9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d"
