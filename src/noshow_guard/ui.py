"""Funciones puras de apoyo para la app visual (``app.py``).

La app es de solo lectura: usa un ``Simulator`` sin registro ni sender y llama a
``evaluate()``, que no guarda ni envía nada.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from noshow_guard.config import PATHS, Paths
from noshow_guard.data import load_clean
from noshow_guard.formatting import es_number
from noshow_guard.model import load_model
from noshow_guard.schemas import SimulationResult, WhatIfResult
from noshow_guard.simulator import Simulator

FEATURE_LABELS: dict[str, str] = {
    "lead_time_days": "Antelación (días)",
    "appointment_weekday": "Día de la semana",
    "age": "Edad",
    "is_male": "Hombre",
    "scholarship": "Beca (Bolsa Família)",
    "hypertension": "Hipertensión",
    "diabetes": "Diabetes",
    "alcoholism": "Alcoholismo",
    "handicap": "Discapacidad (conteo)",
    "neighbourhood": "Barrio",
    "has_history": "Tiene citas previas",
    "prev_appointments": "Citas previas",
    "prev_no_shows": "Inasistencias previas",
    "prev_no_show_rate": "Tasa de inasistencia previa",
}
ACTION_LABELS: dict[str, str] = {
    "sin_accion": "Sin acción",
    "recordatorio_estandar": "Recordatorio estándar",
    "recordatorio_reforzado_con_confirmacion": "Recordatorio reforzado con confirmación",
}
RISK_LABELS: dict[str, str] = {"bajo": "Bajo", "medio": "Medio", "alto": "Alto"}
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes o fin de semana")


def read_only_simulator(paths: Paths = PATHS) -> Simulator:
    """Simulador sin registro ni sender: nada de lo que se simule queda guardado."""
    model, metadata = load_model(paths.model_metadata)
    return Simulator.build(model, metadata, load_clean(paths.clean_parquet))


def percent(p: float) -> str:
    return f"{es_number(100 * p, 1)} %"


def sample_patients(records: pd.DataFrame, as_of: date, n: int = 300, seed: int = 42) -> list[str]:
    """Pacientes que ya existen en ``as_of``; la mitad con citas previas terminadas."""
    day = pd.Timestamp(as_of)
    known = pd.Index(records.loc[records["scheduled_date"] <= day, "patient_id"].unique())
    with_history = pd.Index(records.loc[records["appointment_date"] < day, "patient_id"].unique())
    # Index.difference usa hashing; np.setdiff1d sobre IDs de texto tardaba ~60 s con 58k IDs.
    others = known.difference(with_history, sort=False)
    rng = np.random.default_rng(seed)
    half = n // 2
    picks = [
        *rng.choice(with_history, size=min(half, len(with_history)), replace=False),
        *rng.choice(others, size=min(n - half, len(others)), replace=False),
    ]
    return [str(p) for p in picks]


def patient_history(sim: Simulator, patient_id: str, as_of: date) -> pd.DataFrame:
    """Citas del paciente que el modelo puede usar (terminadas antes de ``as_of``)."""
    own = sim.records[
        (sim.records["patient_id"] == patient_id)
        & (sim.records["appointment_date"] < pd.Timestamp(as_of))
    ].sort_values("appointment_date")
    return pd.DataFrame(
        {
            "Fecha de la cita": own["appointment_date"].dt.date.to_numpy(),
            "Resultado": np.where(own["no_show"] == 1, "No asistió", "Asistió"),
        }
    )


def patient_profile(sim: Simulator, patient_id: str, as_of: date) -> dict[str, object]:
    """Atributos que usaría el simulador (último registro conocido en ``as_of``).

    Lanza ``SimulationError`` si el paciente no existe o no tiene registros en ``as_of``.
    """
    cita, history, _ = sim.existing_patient(patient_id, as_of, as_of)
    completed = [h for h in history if h.appointment_date < as_of]
    return {
        "Edad": cita.age,
        "Género": "Mujer" if cita.gender == "F" else "Hombre",
        "Barrio": cita.neighbourhood,
        "Beca": "Sí" if cita.scholarship else "No",
        "Hipertensión": "Sí" if cita.hypertension else "No",
        "Diabetes": "Sí" if cita.diabetes else "No",
        "Alcoholismo": "Sí" if cita.alcoholism else "No",
        "Discapacidad (conteo)": cita.handicap,
        "Citas previas terminadas": len(completed),
        "Inasistencias previas": sum(h.no_show for h in completed),
    }


def _display_value(feature: str, value: object) -> str:
    if feature == "appointment_weekday" and isinstance(value, int):
        return WEEKDAYS[value]
    if feature in {
        "is_male",
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "has_history",
    }:
        return "Sí" if value == 1 else "No"
    if feature == "prev_no_show_rate" and isinstance(value, float):
        return percent(value)
    return str(value)


def factors_frame(result: SimulationResult) -> pd.DataFrame:
    """Los 3 factores SHAP en lenguaje simple."""
    return pd.DataFrame(
        [
            {
                "Factor": FEATURE_LABELS.get(f.feature, f.feature),
                "Valor del caso": _display_value(f.feature, f.value),
                "Efecto": "Sube el riesgo" if f.effect == "+" else "Baja el riesgo",
                "Magnitud (SHAP)": round(abs(f.shap), 3),
            }
            for f in result.top_factors
        ]
    )


def features_frame(result: SimulationResult) -> pd.DataFrame:
    """Features exactas que vio el modelo, con nombres legibles."""
    features = result.features or {}
    return pd.DataFrame(
        [
            {"Variable": FEATURE_LABELS.get(k, k), "Valor": _display_value(k, v)}
            for k, v in features.items()
        ]
    )


def what_if_frame(result: WhatIfResult) -> pd.DataFrame:
    """Tabla de "¿y si...?" lista para graficar (solo filas con probabilidad)."""
    rows = [r for r in result.rows if r.probability_no_show is not None]
    return pd.DataFrame(
        {
            "Antelación (días)": [r.lead_time_days for r in rows],
            "Probabilidad": [r.probability_no_show for r in rows],
            "Acción": [ACTION_LABELS.get(r.action or "", r.action or "-") for r in rows],
        }
    )
