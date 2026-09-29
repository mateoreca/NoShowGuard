"""Tests del simulador: paridad con el pipeline offline, validaciones y advertencias."""

from __future__ import annotations

from datetime import date, time, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from conftest import SyntheticWorld
from noshow_guard.features import FEATURE_COLUMNS, build_features
from noshow_guard.schemas import NewPatient, SimulationRequest
from noshow_guard.simulator import SimulationError, Simulator

NEW = {"age": 30, "gender": "F", "neighbourhood": "centro"}
PARITY_ATOL = 1e-12


def request(**overrides: object) -> SimulationRequest:
    payload: dict[str, object] = {
        "new_patient": NEW,
        "appointment_date": date(2016, 6, 10),
        "appointment_time": time(9, 30),
        "as_of": date(2016, 6, 1),
    }
    payload.update(overrides)
    return SimulationRequest.model_validate(payload)


def existing(world: SyntheticWorld, **overrides: object) -> SimulationRequest:
    return request(new_patient=None, patient_id=world.clean["patient_id"].iloc[0], **overrides)


# --- Paridad offline / simulador ---------------------------------------------


def test_parity_with_offline_pipeline(world: SyntheticWorld) -> None:
    """Con as_of = fecha de agendamiento, el simulador reproduce features y probabilidad."""
    test = world.splits["test"].sample(150, random_state=42)
    offline_p = world.simulator.model.predict_proba(test)
    for (_, row), expected_p in zip(test.iterrows(), offline_p, strict=True):
        result = world.simulator.simulate(
            request(
                new_patient=None,
                patient_id=row["patient_id"],
                appointment_date=row["appointment_date"].date(),
                as_of=row["as_of"].date(),
            )
        )
        assert result.features == {k: row[k] for k in FEATURE_COLUMNS}
        # Mismas features -> misma probabilidad salvo redondeo de punto flotante (~1e-17):
        # el producto matricial en lote y fila a fila acumula en distinto orden.
        assert result.probability_no_show == pytest.approx(expected_p, rel=0, abs=PARITY_ATOL)


# --- Validaciones ------------------------------------------------------------


def test_appointment_before_as_of_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no se agenda en el pasado"):
        request(appointment_date=date(2016, 5, 31))


def test_same_day_is_out_of_scope(world: SyntheticWorld) -> None:
    result = world.simulator.simulate(request(appointment_date=date(2016, 6, 1)))
    assert result.status == "fuera_de_alcance"
    assert result.probability_no_show is None
    assert result.risk_level is None
    assert result.top_factors == []
    assert "fuera del alcance" in result.notes[0]


@pytest.mark.parametrize(
    "payload",
    [
        {"new_patient": None},  # ni paciente existente ni nuevo
        {"patient_id": "123"},  # los dos a la vez
    ],
)
def test_exactly_one_patient_source(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="exactamente uno"):
        request(**payload)


@pytest.mark.parametrize(
    "patient",
    [
        {**NEW, "age": -1},
        {**NEW, "age": 111},
        {**NEW, "handicap": 5},
        {**NEW, "gender": "X"},
        {**NEW, "neighbourhood": ""},
        {**NEW, "unknown_field": 1},
    ],
)
def test_invalid_new_patient_is_rejected(patient: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        request(new_patient=patient)


def test_unknown_patient_is_error(world: SyntheticWorld) -> None:
    with pytest.raises(SimulationError, match="no existe"):
        world.simulator.simulate(request(new_patient=None, patient_id="999999999"))


def test_patient_without_records_at_as_of_is_error(world: SyntheticWorld) -> None:
    with pytest.raises(SimulationError, match="no tiene registros"):
        world.simulator.simulate(
            existing(world, as_of=date(2015, 1, 1), appointment_date=date(2015, 1, 10))
        )


# --- Paciente nuevo ------------------------------------------------------------


def test_new_patient_matches_patient_without_history(world: SyntheticWorld) -> None:
    req = request()
    result = world.simulator.simulate(req)
    assert req.new_patient is not None
    cita = Simulator.new_patient(req.new_patient, req.appointment_date)
    assert result.features == build_features(cita, [], req.as_of)
    assert result.features is not None
    assert result.features["has_history"] == 0
    assert result.features["neighbourhood"] == "CENTRO"  # normalizado como en el dataset


def test_no_history_case_exists_in_training_data(world: SyntheticWorld) -> None:
    """El paciente nuevo se trata como un caso que el modelo sí vio al entrenar."""
    train = world.splits["train"]
    no_history = train[train["has_history"] == 0]
    assert len(no_history) > 0
    assert (
        (no_history[["prev_appointments", "prev_no_shows", "prev_no_show_rate"]] == 0).all().all()
    )


# --- Invariancias y salida -----------------------------------------------------


@pytest.mark.parametrize("source", ["new", "existing"])
def test_appointment_time_does_not_change_prediction(world: SyntheticWorld, source: str) -> None:
    base = request() if source == "new" else existing(world)
    results = [
        world.simulator.simulate(base.model_copy(update={"appointment_time": t}))
        for t in (time(7, 0), time(12, 30), time(18, 45))
    ]
    assert len({r.probability_no_show for r in results}) == 1
    assert all(r.features == results[0].features for r in results)
    assert [r.appointment_time for r in results] == [time(7, 0), time(12, 30), time(18, 45)]


def test_output_has_three_factors_notes_and_risk(world: SyntheticWorld) -> None:
    result = world.simulator.simulate(request())
    assert result.status == "ok"
    assert result.probability_no_show is not None
    assert 0 <= result.probability_no_show <= 1
    assert result.risk_level in {"bajo", "medio", "alto"}
    assert len(result.top_factors) == 3
    assert any("hora" in n for n in result.notes)
    assert any("Brasil" in n for n in result.notes)
    assert result.model_version == "sintetico"


def test_lead_time_extrapolation_warning(world: SyntheticWorld) -> None:
    max_lead = world.simulator.metadata["train_ranges"]["lead_time_days_max"]
    far = date(2016, 6, 1) + timedelta(days=max_lead + 1)
    result = world.simulator.simulate(request(appointment_date=far))
    assert any("Antelación" in w and "extrapola" in w for w in result.warnings)


def test_age_extrapolation_warning(world: SyntheticWorld) -> None:
    age_max = world.simulator.metadata["train_ranges"]["age_max"]
    if age_max >= 110:
        pytest.skip("El entrenamiento sintético ya cubre la edad máxima permitida.")
    result = world.simulator.simulate(request(new_patient={**NEW, "age": age_max + 1}))
    assert any("Edad" in w and "extrapola" in w for w in result.warnings)


def test_unknown_neighbourhood_warning(world: SyntheticWorld) -> None:
    result = world.simulator.simulate(request(new_patient={**NEW, "neighbourhood": "NARNIA"}))
    assert any("NARNIA" in w for w in result.warnings)
    assert result.probability_no_show is not None


def test_stale_age_warning_for_far_as_of(world: SyntheticWorld) -> None:
    result = world.simulator.simulate(
        existing(world, as_of=date(2026, 9, 29), appointment_date=date(2026, 10, 5))
    )
    assert any("no se ajusta" in w for w in result.warnings)


def test_simulation_does_not_modify_records(world: SyntheticWorld) -> None:
    before = pd.util.hash_pandas_object(world.simulator.records, index=False).to_numpy()
    world.simulator.simulate(existing(world))
    world.simulator.simulate(request())
    after = pd.util.hash_pandas_object(world.simulator.records, index=False).to_numpy()
    assert np.array_equal(before, after)


def test_new_patient_schema_defaults() -> None:
    patient = NewPatient.model_validate(NEW)
    assert (patient.scholarship, patient.handicap) == (False, 0)
