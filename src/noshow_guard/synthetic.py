"""Datos sintéticos con el esquema del dataset limpio, para tests y el entrenamiento de humo.

No imitan al dataset real: solo tienen su forma y una señal conocida (antelación y edad),
para verificar el pipeline de punta a punta sin depender del CSV de Kaggle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from noshow_guard.data import CLEAN_DTYPES

NEIGHBOURHOODS: tuple[str, ...] = ("CENTRO", "JARDIM", "PRAIA", "MORRO", "ILHA")


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def synthetic_clean(n_patients: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Dataset limpio sintético con pacientes que repiten citas.

    Las fechas de cita cubren el mismo rango que el real y evitan el hueco entre train y
    validación (2016-05-21 a 2016-05-23), así el split cronológico por defecto funciona.
    """
    rng = np.random.default_rng(seed)
    gap = pd.date_range("2016-05-21", "2016-05-23")
    days = pd.date_range("2016-04-29", "2016-06-08").difference(gap)
    rows = []
    for pid in range(n_patients):
        age = int(rng.integers(0, 95))
        patient: dict[str, object] = {
            "patient_id": str(10_000 + pid),
            "gender": str(rng.choice(["F", "M"])),
            "age": age,
            "neighbourhood": str(rng.choice(NEIGHBOURHOODS)),
            **{c: int(rng.binomial(1, 0.15)) for c in ("scholarship", "hypertension")},
            **{c: int(rng.binomial(1, 0.05)) for c in ("diabetes", "alcoholism", "sms_received")},
            "handicap": int(rng.binomial(1, 0.02)),
        }
        for _ in range(int(rng.integers(1, 6))):
            appointment = days[int(rng.integers(0, len(days)))]
            lead = int(rng.integers(0, 40))
            scheduled_at = (
                appointment
                - pd.Timedelta(days=lead)
                + pd.Timedelta(hours=int(rng.integers(7, 18)), minutes=int(rng.integers(0, 60)))
            )
            p = _sigmoid(-1.3 + 0.03 * lead - 0.01 * (age - 40))
            rows.append(
                {
                    **patient,
                    "scheduled_at": scheduled_at,
                    "scheduled_date": scheduled_at.normalize(),
                    "appointment_date": appointment,
                    "no_show": int(rng.binomial(1, p)),
                }
            )
    frame = pd.DataFrame(rows)
    frame["appointment_id"] = np.arange(len(frame)) + 5_000_000
    return frame[list(CLEAN_DTYPES)].astype(CLEAN_DTYPES)
