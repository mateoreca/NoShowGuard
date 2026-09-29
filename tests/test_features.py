"""Tests de features: anti-fuga, alcance, paridad fila/lote y split cronológico."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from noshow_guard.config import SplitConfig
from noshow_guard.features import (
    FEATURE_COLUMNS,
    META_COLUMNS,
    Appointment,
    OutOfScopeError,
    PastAppointment,
    appointment_from_row,
    appointment_weekday,
    build_feature_frame,
    build_features,
    completed_history,
    history_index,
    split_summary,
    temporal_split,
)

AS_OF = date(2016, 5, 10)
CITA = Appointment(
    appointment_date=date(2016, 5, 17),
    age=30,
    gender="F",
    neighbourhood="CENTRO",
    scholarship=0,
    hypertension=0,
    diabetes=0,
    alcoholism=0,
    handicap=0,
)


def appt_row(appointment_id: int, patient_id: str, scheduled: str, appointment: str, no_show: int):
    """Fila con el esquema del dataset limpio."""
    scheduled_at = pd.Timestamp(scheduled)
    return {
        "appointment_id": appointment_id,
        "patient_id": patient_id,
        "gender": "F",
        "scheduled_at": scheduled_at,
        "scheduled_date": scheduled_at.normalize(),
        "appointment_date": pd.Timestamp(appointment),
        "age": 40,
        "neighbourhood": "CENTRO",
        "scholarship": 0,
        "hypertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 0,
        "no_show": no_show,
    }


@pytest.fixture
def patient_timeline() -> pd.DataFrame:
    """Un paciente con 5 citas en el tiempo más otro paciente distinto."""
    return pd.DataFrame(
        [
            appt_row(1, "A", "2016-04-20 09:00", "2016-04-29", 1),
            appt_row(2, "A", "2016-04-29 10:00", "2016-05-02", 0),
            appt_row(3, "A", "2016-05-02 08:00", "2016-05-02", 1),  # mismo día
            appt_row(4, "A", "2016-05-02 15:00", "2016-05-10", 1),
            appt_row(5, "A", "2016-05-05 11:00", "2016-05-20", 0),
            appt_row(6, "B", "2016-04-20 09:00", "2016-05-03", 1),
        ]
    )


# --- Anti-fuga -----------------------------------------------------------------


def test_history_is_strictly_before_as_of() -> None:
    hist = [
        PastAppointment(date(2016, 5, 9), 1),
        PastAppointment(AS_OF, 1),  # mismo día de as_of: aún no terminó
        PastAppointment(date(2016, 5, 17), 1),  # la propia cita
        PastAppointment(date(2016, 6, 1), 1),  # futura
    ]
    assert completed_history(hist, AS_OF) == [hist[0]]
    feats = build_features(CITA, hist, AS_OF)
    assert feats["prev_appointments"] == 1
    assert feats["prev_no_shows"] == 1


def test_future_history_passed_in_does_not_change_features() -> None:
    past = [PastAppointment(date(2016, 5, 2), 0)]
    future = [PastAppointment(date(2016, 5, 12), 1), PastAppointment(date(2016, 5, 17), 1)]
    assert build_features(CITA, past, AS_OF) == build_features(CITA, past + future, AS_OF)


def test_frame_history_excludes_own_and_later_appointments(patient_timeline: pd.DataFrame) -> None:
    frame = build_feature_frame(patient_timeline, patient_timeline).set_index("appointment_id")
    # Cita 5 (agendada el 05-05): solo las citas 1, 2 y 3 terminaron antes (04-29 y 05-02).
    assert frame.loc[5, "prev_appointments"] == 3
    assert frame.loc[5, "prev_no_shows"] == 2
    # Cita 4 (agendada el 05-02): la cita 3 es ese mismo día, así que no cuenta.
    assert frame.loc[4, "prev_appointments"] == 1
    # Cita 1: nada antes.
    assert frame.loc[1, "has_history"] == 0


def test_own_outcome_does_not_alter_own_features(patient_timeline: pd.DataFrame) -> None:
    original = build_feature_frame(patient_timeline, patient_timeline).set_index("appointment_id")
    for appointment_id in original.index:
        flipped = patient_timeline.copy()
        mask = flipped["appointment_id"] == appointment_id
        flipped.loc[mask, "no_show"] = 1 - flipped.loc[mask, "no_show"]
        rebuilt = build_feature_frame(flipped, flipped).set_index("appointment_id")
        pd.testing.assert_series_equal(
            rebuilt.loc[appointment_id, list(FEATURE_COLUMNS)],
            original.loc[appointment_id, list(FEATURE_COLUMNS)],
        )


def test_later_outcomes_do_not_alter_earlier_features(patient_timeline: pd.DataFrame) -> None:
    original = build_feature_frame(patient_timeline, patient_timeline).set_index("appointment_id")
    altered = patient_timeline.copy()
    later = altered["appointment_date"] >= pd.Timestamp("2016-05-10")
    altered.loc[later, "no_show"] = 1 - altered.loc[later, "no_show"]
    rebuilt = build_feature_frame(altered, altered).set_index("appointment_id")
    # Todas estas citas se agendaron antes del 05-10: sus features no pueden cambiar.
    for appointment_id in [1, 2, 4, 5, 6]:
        pd.testing.assert_series_equal(
            rebuilt.loc[appointment_id, list(FEATURE_COLUMNS)],
            original.loc[appointment_id, list(FEATURE_COLUMNS)],
        )


def test_forbidden_columns_are_not_features() -> None:
    forbidden = {"sms_received", "month", "year", "appointment_time", "scheduled_at", "no_show"}
    assert forbidden.isdisjoint(FEATURE_COLUMNS)
    assert set(build_features(CITA, [], AS_OF)) == set(FEATURE_COLUMNS)


# --- Alcance y definición de features ----------------------------------------


def test_same_day_is_out_of_scope() -> None:
    with pytest.raises(OutOfScopeError):
        build_features(CITA, [], as_of=CITA.appointment_date)


def test_appointment_before_as_of_is_error() -> None:
    with pytest.raises(ValueError, match="anterior a as_of"):
        build_features(CITA, [], as_of=CITA.appointment_date + timedelta(days=1))


def test_frame_drops_same_day_appointments(patient_timeline: pd.DataFrame) -> None:
    frame = build_feature_frame(patient_timeline, patient_timeline)
    assert 3 not in set(frame["appointment_id"])
    assert (frame["lead_time_days"] >= 1).all()


def test_lead_time_uses_calendar_days_not_hours() -> None:
    # Agendada a las 23:59 del 05-10 para el 05-11: 1 día, aunque falten minutos.
    late = pd.DataFrame([appt_row(1, "A", "2016-05-10 23:59", "2016-05-11", 0)])
    assert build_feature_frame(late, late)["lead_time_days"].tolist() == [1]


@pytest.mark.parametrize(
    ("day", "expected"),
    [(date(2016, 5, 9), 0), (date(2016, 5, 13), 4), (date(2016, 5, 14), 4), (date(2016, 5, 15), 4)],
)
def test_weekend_is_grouped_with_friday(day: date, expected: int) -> None:
    assert appointment_weekday(day) == expected


def test_no_history_has_explicit_neutral_values() -> None:
    feats = build_features(CITA, [], AS_OF)
    assert (feats["has_history"], feats["prev_appointments"], feats["prev_no_shows"]) == (0, 0, 0)
    assert feats["prev_no_show_rate"] == 0.0


def test_history_rate() -> None:
    hist = [PastAppointment(date(2016, 5, 2), 1), PastAppointment(date(2016, 5, 3), 0)]
    feats = build_features(CITA, hist, AS_OF)
    assert feats["prev_no_show_rate"] == pytest.approx(0.5)
    assert feats["has_history"] == 1


# --- Un solo pipeline: el lote es exactamente build_features fila por fila ---


def test_frame_matches_row_by_row_build_features(patient_timeline: pd.DataFrame) -> None:
    frame = build_feature_frame(patient_timeline, patient_timeline)
    index = history_index(patient_timeline)
    by_id = patient_timeline.set_index("appointment_id")
    for record in frame.to_dict("records"):
        row = by_id.loc[record["appointment_id"]]
        expected = build_features(
            appointment_from_row(row), index[row["patient_id"]], row["scheduled_date"].date()
        )
        assert {k: record[k] for k in FEATURE_COLUMNS} == expected
    assert list(frame.columns) == list(META_COLUMNS) + list(FEATURE_COLUMNS)


# --- Split cronológico ---------------------------------------------------------

CFG = SplitConfig(
    train_end=date(2016, 5, 5),
    val_start=date(2016, 5, 9),
    val_end=date(2016, 5, 12),
    test_start=date(2016, 5, 16),
)


def split_frame(days: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"P{i % 3}" for i in range(len(days))],
            "appointment_date": pd.to_datetime(days),
            "no_show": [i % 2 for i in range(len(days))],
            "has_history": [0] * len(days),
        }
    )


def test_split_is_chronological_and_disjoint() -> None:
    frame = split_frame(["2016-05-02", "2016-05-05", "2016-05-09", "2016-05-12", "2016-05-16"])
    parts = temporal_split(frame, CFG)
    assert [len(parts[n]) for n in ("train", "val", "test")] == [2, 2, 1]
    assert parts["train"]["appointment_date"].max() < parts["val"]["appointment_date"].min()
    assert parts["val"]["appointment_date"].max() < parts["test"]["appointment_date"].min()


def test_split_rejects_dates_in_gaps() -> None:
    with pytest.raises(ValueError, match="fuera de todos los cortes"):
        temporal_split(split_frame(["2016-05-06"]), CFG)


def test_split_summary_reports_patient_overlap() -> None:
    frame = split_frame(["2016-05-02", "2016-05-05", "2016-05-09", "2016-05-12", "2016-05-16"])
    summary = split_summary(temporal_split(frame, CFG))
    assert summary["train"]["rows"] == 2
    # Pacientes P0, P1 en train; val tiene P2 y P0; test tiene P1.
    assert summary["val"]["share_patient_seen_in_train"] == 0.5
    assert summary["test"]["share_patient_seen_in_train"] == 1.0
