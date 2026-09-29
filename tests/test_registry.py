"""Tests del registro SQLite de simulaciones."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import SyntheticWorld
from noshow_guard import cli, simulator
from noshow_guard.registry import Registry, new_patient_ref
from noshow_guard.schemas import NewPatient, SimulationRequest
from noshow_guard.simulator import Simulator

NEW = {"age": 30, "gender": "F", "neighbourhood": "CENTRO", "diabetes": True}


def req(**overrides: object) -> SimulationRequest:
    payload: dict[str, object] = {
        "new_patient": NEW,
        "appointment_date": date(2016, 6, 10),
        "appointment_time": time(9, 30),
        "as_of": date(2016, 6, 1),
    }
    payload.update(overrides)
    return SimulationRequest.model_validate(payload)


@pytest.fixture
def sim(world: SyntheticWorld, tmp_path: Path) -> Simulator:
    """El simulador sintético con un registro propio en un directorio temporal."""
    return dataclasses.replace(world.simulator, registry=Registry(tmp_path / "sims.db"))


def patient(world: SyntheticWorld) -> str:
    return str(world.clean["patient_id"].iloc[0])


def test_three_simulations_give_three_simulated_rows(sim: Simulator, world: SyntheticWorld) -> None:
    sim.simulate(req())
    sim.simulate(req(new_patient=None, patient_id=patient(world)))
    sim.simulate(req(appointment_date=date(2016, 6, 1)))  # fuera de alcance también se registra
    rows = sim.registry.read() if sim.registry else pd.DataFrame()
    assert len(rows) == 3
    assert rows["simulated"].eq(1).all()
    assert rows["status"].tolist() == ["ok", "ok", "fuera_de_alcance"]
    assert rows["probability"].isna().tolist() == [False, False, True]
    assert rows["appointment_time"].tolist() == ["09:30", "09:30", "09:30"]
    assert set(rows["model_version"]) == {"sintetico"}


def test_no_log_does_not_write(sim: Simulator) -> None:
    sim.simulate(req(), log=False)
    assert sim.registry is not None
    assert sim.registry.read().empty


def test_new_patient_stored_as_hash_and_features_only(sim: Simulator) -> None:
    sim.simulate(req())
    assert sim.registry is not None
    row = sim.registry.read().iloc[0]
    assert row["patient_kind"] == "nuevo"
    assert row["patient_ref"] == new_patient_ref(NewPatient.model_validate(NEW))
    assert row["patient_ref"].startswith("nuevo:") and "CENTRO" not in row["patient_ref"]
    features = json.loads(row["features_json"])
    assert features["diabetes"] == 1
    assert set(features) >= {"age", "lead_time_days", "has_history"}


def test_table_rejects_non_simulated_rows(tmp_path: Path, sim: Simulator) -> None:
    sim.simulate(req())
    assert sim.registry is not None
    with sqlite3.connect(sim.registry.path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO simulations (created_at, patient_ref, patient_kind, as_of, "
            "appointment_date, appointment_time, lead_time_days, status, model_version, simulated) "
            "VALUES ('x', 'p', 'existente', 'x', 'x', 'x', 1, 'ok', 'v', 0)"
        )


def test_simulating_does_not_change_real_history(sim: Simulator, world: SyntheticWorld) -> None:
    pid = patient(world)
    before_records = pd.util.hash_pandas_object(sim.records, index=False).to_numpy()
    before_history = list(sim.history[pid])
    for day in (date(2016, 6, 2), date(2016, 6, 3), date(2016, 6, 6)):
        sim.simulate(req(new_patient=None, patient_id=pid, appointment_date=day))
    after_records = pd.util.hash_pandas_object(sim.records, index=False).to_numpy()
    assert np.array_equal(before_records, after_records)
    assert sim.history[pid] == before_history


def test_simulations_never_enter_history(sim: Simulator, world: SyntheticWorld) -> None:
    pid = patient(world)
    later = req(
        new_patient=None, patient_id=pid, as_of=date(2016, 6, 5), appointment_date=date(2016, 6, 9)
    )
    before = sim.simulate(later, log=False)
    # Una cita simulada del paciente, anterior a as_of, queda en el registro...
    sim.simulate(req(new_patient=None, patient_id=pid, appointment_date=date(2016, 6, 2)))
    after = sim.simulate(later, log=False)
    # ...pero no cambia su historial: las features y la probabilidad son idénticas.
    assert after.features == before.features
    assert after.probability_no_show == before.probability_no_show


def test_cli_has_no_simulated_history_option() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["simulate", "--age", "30", "--gender", "F", "--neighbourhood", "X",
             "--date", "2016-06-10", "--time", "10:00", "--include-simulated-history"]
        )  # fmt: skip


def test_read_missing_registry_is_empty(tmp_path: Path) -> None:
    assert Registry(tmp_path / "nada.db").read().empty


# --- CLI -------------------------------------------------------------------------


def test_cli_no_log_flag(sim: Simulator, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simulator, "_DEFAULT", sim)
    base = ["simulate", "--age", "40", "--gender", "M", "--neighbourhood", "PRAIA",
            "--date", "2016-06-10", "--time", "10:00", "--as-of", "2016-06-01"]  # fmt: skip
    assert cli.main([*base, "--no-log"]) == 0
    assert sim.registry is not None
    assert sim.registry.read().empty
    assert cli.main(base) == 0
    assert len(sim.registry.read()) == 1


def test_cli_model_info(world: SyntheticWorld, tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(world.simulator.metadata), encoding="utf-8")
    info = cli.model_info(meta_path)
    assert info["model_version"] == "sintetico"
    assert info["principal_model"] == "regresion_logistica_calibrada"
    assert info["decided_after_test"] is True
    assert info["costs_are_assumptions"] is True
    assert info["model_file_present"] is False  # el test no guarda el .joblib
    assert set(info["thresholds"]) == {"standard", "reinforced"}
