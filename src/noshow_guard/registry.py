"""Registro SQLite de simulaciones.

Principio: lo simulado nunca se mezcla con lo real. Esta tabla vive aparte del dataset,
todas sus filas llevan ``simulated = 1`` (lo impone un CHECK) y el historial de los
pacientes se lee solo del dataset: las simulaciones nunca entran al historial.
De un paciente nuevo se guarda un hash y las features usadas, nada más.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from noshow_guard.schemas import NewPatient, SimulationRequest, SimulationResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS simulations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL,
    patient_ref      TEXT    NOT NULL,
    patient_kind     TEXT    NOT NULL CHECK (patient_kind IN ('existente', 'nuevo')),
    as_of            TEXT    NOT NULL,
    appointment_date TEXT    NOT NULL,
    appointment_time TEXT    NOT NULL,
    lead_time_days   INTEGER NOT NULL,
    status           TEXT    NOT NULL,
    probability      REAL,
    risk_level       TEXT,
    action           TEXT,
    model_version    TEXT    NOT NULL,
    features_json    TEXT,
    simulated        INTEGER NOT NULL DEFAULT 1 CHECK (simulated = 1)
);
"""

INSERT = """
INSERT INTO simulations (
    created_at, patient_ref, patient_kind, as_of, appointment_date, appointment_time,
    lead_time_days, status, probability, risk_level, action, model_version, features_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def new_patient_ref(patient: NewPatient) -> str:
    """Referencia estable y no reversible a un paciente nuevo (hash de sus datos declarados)."""
    canonical = json.dumps(patient.model_dump(mode="json"), sort_keys=True)
    return "nuevo:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Registry:
    """Acceso al archivo SQLite; crea la tabla si no existe."""

    path: Path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.executescript(SCHEMA)
            with conn:  # commit al salir, rollback si hay error
                yield conn

    def log(
        self,
        request: SimulationRequest,
        result: SimulationResult,
        created_at: datetime | None = None,
    ) -> int:
        """Inserta una simulación y devuelve su id. Consultas siempre parametrizadas."""
        if request.patient_id is not None:
            ref, kind = request.patient_id, "existente"
        else:
            assert request.new_patient is not None
            ref, kind = new_patient_ref(request.new_patient), "nuevo"
        values = (
            (created_at or datetime.now(UTC)).isoformat(timespec="seconds"),
            ref,
            kind,
            result.as_of.isoformat(),
            result.appointment_date.isoformat(),
            result.appointment_time.isoformat(timespec="minutes"),
            result.lead_time_days,
            result.status,
            result.probability_no_show,
            result.risk_level,
            result.action,
            result.model_version,
            json.dumps(result.features, ensure_ascii=False) if result.features else None,
        )
        with self._connect() as conn:
            cursor = conn.execute(INSERT, values)
            return int(cursor.lastrowid or 0)

    def read(self) -> pd.DataFrame:
        """Todas las simulaciones registradas (vacío si el archivo no existe)."""
        if not self.path.exists():
            return pd.DataFrame()
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM simulations ORDER BY id", conn)
