"""Criterio de la Fase 5 con datos reales: 3 simulaciones -> 3 filas simuladas, dataset intacto.

Se omite si no existen el modelo entrenado y los datos (por ejemplo, en CI).
"""

from __future__ import annotations

import dataclasses
from datetime import date, time
from pathlib import Path

import pytest

from noshow_guard.config import PATHS
from noshow_guard.data import file_sha256
from noshow_guard.registry import Registry
from noshow_guard.schemas import SimulationRequest

REQUIRED = [PATHS.model_metadata, PATHS.clean_parquet, PATHS.raw_csv]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED), reason="Requiere make data y make train"
)


def test_three_simulations_leave_dataset_intact(tmp_path: Path) -> None:
    from noshow_guard.simulator import Simulator

    try:
        base = Simulator.from_disk()
    except FileNotFoundError:
        pytest.skip("El artefacto del modelo no está en models/ (ejecuta make train).")
    sim = dataclasses.replace(base, registry=Registry(tmp_path / "sims.db"))
    hashes_before = {p: file_sha256(p) for p in (PATHS.raw_csv, PATHS.clean_parquet)}

    patient_id = str(sim.records["patient_id"].iloc[0])
    for day in (date(2016, 6, 10), date(2016, 6, 13), date(2016, 6, 14)):
        sim.simulate(
            SimulationRequest(
                patient_id=patient_id,
                appointment_date=day,
                appointment_time=time(9, 0),
                as_of=date(2016, 6, 1),
            )
        )

    assert sim.registry is not None
    rows = sim.registry.read()
    assert len(rows) == 3
    assert rows["simulated"].eq(1).all()
    assert {p: file_sha256(p) for p in hashes_before} == hashes_before
