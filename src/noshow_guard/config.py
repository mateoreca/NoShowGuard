"""Configuración central del proyecto: rutas y semilla global."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Rutas del proyecto, relativas a la raíz del repositorio."""

    root: Path = PROJECT_ROOT
    data_raw: Path = field(default=PROJECT_ROOT / "data" / "raw")
    data_processed: Path = field(default=PROJECT_ROOT / "data" / "processed")
    raw_csv: Path = field(default=PROJECT_ROOT / "data" / "raw" / "data.csv")


SEED: int = 42
PATHS = Paths()
