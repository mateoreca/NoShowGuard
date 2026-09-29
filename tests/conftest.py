"""Fixtures compartidas: splits sintéticos con el esquema de features (sin el CSV real)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noshow_guard.features import FEATURE_COLUMNS, META_COLUMNS

NEIGHBOURHOODS = ["CENTRO", "JARDIM", "PRAIA", "MORRO", "ILHA"]


def synthetic_split(n: int, start: str, days: int, seed: int) -> pd.DataFrame:
    """Frame de features sintético con señal conocida (antelación, edad, historial)."""
    rng = np.random.default_rng(seed)
    lead = rng.integers(1, 60, n)
    age = rng.integers(0, 100, n)
    prev = rng.poisson(0.5, n)
    prev_ns = rng.binomial(prev, 0.3)
    logit = -1.2 + 0.03 * lead - 0.01 * (age - 40) + 0.8 * (prev_ns > 0)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    dates = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    frame = pd.DataFrame(
        {
            "appointment_id": np.arange(n) + seed * 100_000,
            "patient_id": [f"P{i}" for i in rng.integers(0, n // 2, n)],
            "appointment_date": dates,
            "as_of": dates - pd.to_timedelta(lead, unit="D"),
            "no_show": y,
            "lead_time_days": lead,
            "appointment_weekday": rng.integers(0, 5, n),
            "age": age,
            "is_male": rng.integers(0, 2, n),
            "scholarship": rng.binomial(1, 0.1, n),
            "hypertension": rng.binomial(1, 0.2, n),
            "diabetes": rng.binomial(1, 0.07, n),
            "alcoholism": rng.binomial(1, 0.03, n),
            "handicap": rng.binomial(1, 0.02, n),
            "neighbourhood": rng.choice(NEIGHBOURHOODS, n),
            "has_history": (prev > 0).astype(int),
            "prev_appointments": prev,
            "prev_no_shows": prev_ns,
            "prev_no_show_rate": np.where(prev > 0, prev_ns / np.maximum(prev, 1), 0.0),
        }
    )
    return frame[[*META_COLUMNS, *FEATURE_COLUMNS]]


@pytest.fixture(scope="session")
def synthetic_splits() -> dict[str, pd.DataFrame]:
    return {
        "train": synthetic_split(3000, "2016-04-29", 20, seed=1),
        "val": synthetic_split(1200, "2016-05-24", 7, seed=2),
        "test": synthetic_split(1200, "2016-06-01", 7, seed=3),
    }
