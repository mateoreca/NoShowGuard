"""Tests de la app visual (Streamlit AppTest) y de sus helpers. La app no guarda nada."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import SyntheticWorld
from noshow_guard import ui
from noshow_guard.config import PATHS, Paths
from noshow_guard.schemas import SimulationRequest
from noshow_guard.smoke import run as smoke_run

APP = str(PATHS.root / "src" / "noshow_guard" / "app.py")
AS_OF = date(2016, 6, 1)


# --- Helpers -------------------------------------------------------------------


def test_sample_patients_exist_at_as_of(world: SyntheticWorld) -> None:
    ids = ui.sample_patients(world.clean, AS_OF, n=40)
    records = world.clean[world.clean["patient_id"].isin(ids)]
    first_seen = records.groupby("patient_id")["scheduled_date"].min()
    assert len(ids) == 40 and len(set(ids)) == 40
    assert (first_seen <= pd.Timestamp(AS_OF)).all()
    with_history = records.loc[records["appointment_date"] < pd.Timestamp(AS_OF), "patient_id"]
    assert with_history.nunique() == 20  # la mitad tiene citas previas terminadas


def test_patient_history_only_before_as_of(world: SyntheticWorld) -> None:
    pid = ui.sample_patients(world.clean, AS_OF, n=2)[0]
    history = ui.patient_history(world.simulator, pid, AS_OF)
    assert (pd.to_datetime(history["Fecha de la cita"]) < pd.Timestamp(AS_OF)).all()
    assert set(history["Resultado"]) <= {"Asistió", "No asistió"}
    profile = ui.patient_profile(world.simulator, pid, AS_OF)
    assert profile["Citas previas terminadas"] == len(history)


def test_readable_frames(world: SyntheticWorld) -> None:
    request = SimulationRequest.model_validate(
        {"new_patient": {"age": 30, "gender": "F", "neighbourhood": "CENTRO"},
         "appointment_date": "2016-06-10", "appointment_time": "09:00", "as_of": "2016-06-01"}
    )  # fmt: skip
    result = world.simulator.evaluate(request)
    factors = ui.factors_frame(result)
    assert len(factors) == 3
    assert set(factors["Efecto"]) <= {"Sube el riesgo", "Baja el riesgo"}
    assert set(factors["Factor"]) <= set(ui.FEATURE_LABELS.values())
    assert len(ui.features_frame(result)) == len(ui.FEATURE_LABELS)
    table = ui.what_if_frame(world.simulator.what_if(request, [0, 1, 7]))
    assert table["Antelación (días)"].tolist() == [1, 7]  # la de antelación 0 no tiene prob.


def test_percent_format() -> None:
    assert ui.percent(0.5422) == "54,2 %"


def test_read_only_simulator_has_no_registry_or_sender(tmp_path: Path) -> None:
    smoke_run(tmp_path)  # entrena y guarda un modelo sintético en tmp_path
    sim = ui.read_only_simulator(Paths(root=tmp_path))
    assert sim.registry is None and sim.sender is None


# --- App (AppTest) -------------------------------------------------------------------


@pytest.fixture
def app(world: SyntheticWorld) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["simulator"] = world.simulator  # sin registro ni sender
    return at.run()


def click(at: AppTest, label: str) -> AppTest:
    next(b for b in at.button if b.label == label).click()
    return at.run()


def test_app_starts_in_dataset_mode(app: AppTest) -> None:
    assert not app.exception
    assert app.radio[0].value == "Paciente del dataset"
    assert app.text_input[0].value  # un PatientId por defecto
    assert app.table  # perfil del paciente


def test_app_simulates_dataset_patient(app: AppTest) -> None:
    at = click(app, "Simular cita")
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Probabilidad de no asistir" in labels
    assert at.metric[0].value.endswith("%")


def test_app_simulates_new_patient(app: AppTest) -> None:
    app.radio[0].set_value("Mis datos (paciente nuevo)").run()
    app.number_input[0].set_value(24).run()
    at = click(app, "Simular cita")
    assert not at.exception
    assert any(m.label == "Probabilidad de no asistir" for m in at.metric)


def test_app_rejects_appointment_in_the_past(app: AppTest) -> None:
    app.date_input[1].set_value(date(2016, 5, 20)).run()  # cita antes de as_of
    at = click(app, "Simular cita")
    assert any("no se agenda en el pasado" in e.value for e in at.error)


def test_app_same_day_is_out_of_scope(app: AppTest) -> None:
    app.date_input[1].set_value(AS_OF).run()
    at = click(app, "Simular cita")
    assert any("fuera del alcance" in i.value for i in at.info)
    assert not any(m.label == "Probabilidad de no asistir" for m in at.metric)


def test_app_hour_does_not_change_probability(app: AppTest) -> None:
    first = click(app, "Simular cita").metric[0].value
    app.time_input[0].set_value(time(17, 45)).run()
    second = click(app, "Simular cita").metric[0].value
    assert first == second
