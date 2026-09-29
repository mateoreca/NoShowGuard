"""Paridad simulador/offline con el modelo y los datos reales.

Se omite si no existen el modelo entrenado ni los datos (por ejemplo, en CI).
"""

from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from noshow_guard.config import PATHS
from noshow_guard.features import FEATURE_COLUMNS
from noshow_guard.schemas import SimulationRequest

REQUIRED = [PATHS.model_metadata, PATHS.clean_parquet, PATHS.features_dir / "test.parquet"]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED), reason="Requiere make data, make features y make train"
)
SAMPLE = 400


@pytest.fixture(scope="module")
def parity() -> pd.DataFrame:
    from noshow_guard.model import load_model
    from noshow_guard.simulator import Simulator

    try:
        simulator = Simulator.from_disk()
    except FileNotFoundError:
        pytest.skip("El artefacto del modelo no está en models/ (ejecuta make train).")
    model, _ = load_model()
    test = pd.read_parquet(PATHS.features_dir / "test.parquet").sample(SAMPLE, random_state=42)
    offline_p = model.predict_proba(test)
    rows = []
    for (_, row), p in zip(test.iterrows(), offline_p, strict=True):
        result = simulator.simulate(
            SimulationRequest(
                patient_id=row["patient_id"],
                appointment_date=row["appointment_date"].date(),
                appointment_time=time(10, 0),
                as_of=row["as_of"].date(),
            )
        )
        assert result.features is not None
        diff = [k for k in FEATURE_COLUMNS if result.features[k] != row[k]]
        assert result.probability_no_show is not None
        # Tolerancia solo por redondeo de punto flotante entre lote y fila (~1e-17).
        rows.append({"diff": diff, "same_p": abs(result.probability_no_show - p) <= 1e-12})
    return pd.DataFrame(rows)


def test_only_age_can_differ(parity: pd.DataFrame) -> None:
    """Todas las features salvo la edad coinciden en el 100 % de la muestra.

    La edad puede diferir si el paciente tiene otro registro agendado el mismo día con otra
    edad: el simulador toma el último registro conocido en as_of (en todo el test esto pasa en
    18 de 17.344 citas).
    """
    other = parity["diff"].map(lambda d: [k for k in d if k != "age"])
    assert other.map(len).eq(0).all()
    assert parity["diff"].map(len).eq(0).mean() >= 0.99


def test_same_features_give_same_probability(parity: pd.DataFrame) -> None:
    identical = parity["diff"].map(len).eq(0)
    assert parity.loc[identical, "same_p"].all()
