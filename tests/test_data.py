"""Tests de las reglas de limpieza y de la validación de esquema (sin depender del CSV real)."""

from __future__ import annotations

import pandas as pd
import pytest

from noshow_guard.data import (
    CLEAN_DTYPES,
    DataValidationError,
    clean,
    schema_errors,
    standardize,
    validate,
)


def raw_row(**overrides: object) -> dict[str, object]:
    """Fila válida con el formato del CSV de Kaggle; ``overrides`` cambia columnas puntuales."""
    row: dict[str, object] = {
        "PatientId": "29872499824296",
        "AppointmentID": 5642903,
        "Gender": "F",
        "ScheduledDay": "2016-04-25T18:38:08Z",
        "AppointmentDay": "2016-04-29T00:00:00Z",
        "Age": 62,
        "Neighbourhood": "JARDIM DA PENHA",
        "Scholarship": 0,
        "Hipertension": 1,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 0,
        "No-show": "No",
    }
    row.update(overrides)
    return row


def raw_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_target_is_explicit_yes_means_no_show() -> None:
    df = standardize(raw_frame(raw_row(**{"No-show": "Yes"}), raw_row(AppointmentID=2)))
    assert df["no_show"].tolist() == [1, 0]
    assert "no_show_raw" not in df.columns


def test_unexpected_target_value_raises() -> None:
    with pytest.raises(DataValidationError, match="No-show"):
        standardize(raw_frame(raw_row(**{"No-show": "Maybe"})))


def test_missing_raw_column_raises() -> None:
    with pytest.raises(DataValidationError, match="Faltan columnas"):
        standardize(raw_frame(raw_row()).drop(columns=["Handcap"]))


def test_columns_are_renamed_and_typos_fixed() -> None:
    df, _ = clean(raw_frame(raw_row()))
    assert list(df.columns) == list(CLEAN_DTYPES)
    assert {"hypertension", "handicap"} <= set(df.columns)


def test_dates_keep_scheduled_time_but_appointment_is_date_only() -> None:
    df, _ = clean(raw_frame(raw_row()))
    assert df.loc[0, "scheduled_at"] == pd.Timestamp("2016-04-25 18:38:08")
    assert df.loc[0, "scheduled_date"] == pd.Timestamp("2016-04-25")
    assert df.loc[0, "appointment_date"] == pd.Timestamp("2016-04-29")


@pytest.mark.parametrize(
    ("override", "rule"),
    [
        ({"PatientId": "9377952927E-5"}, "patient_id_corrupto"),
        ({"Age": -1}, "edad_fuera_de_rango"),
        ({"Age": 115}, "edad_fuera_de_rango"),
        ({"AppointmentDay": "2016-04-24T00:00:00Z"}, "cita_antes_de_agendamiento"),
    ],
)
def test_each_rule_drops_only_invalid_rows(override: dict[str, object], rule: str) -> None:
    raw = raw_frame(raw_row(), raw_row(AppointmentID=2, **override))
    df, report = clean(raw)
    assert df["appointment_id"].tolist() == [5642903]
    assert report.dropped[rule] == 1
    assert report.rows_in == 2
    assert report.rows_out == 1


def test_valid_edge_cases_are_kept() -> None:
    raw = raw_frame(
        raw_row(Age=0),  # bebés: válidos
        raw_row(AppointmentID=2, Age=110),
        raw_row(AppointmentID=3, Handcap=4),  # Handcap es un conteo 0-4
        # Mismo día: se conserva aquí; se excluye del modelado en la Fase 2.
        raw_row(AppointmentID=4, ScheduledDay="2016-04-29T08:00:00Z"),
    )
    df, report = clean(raw)
    assert len(df) == 4
    assert sum(report.dropped.values()) == 0
    assert df["handicap"].max() == 4


def test_clean_output_passes_validation() -> None:
    df, _ = clean(raw_frame(raw_row(), raw_row(AppointmentID=2, Gender="M")))
    assert validate(df) is df


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("handicap", 5, "handicap fuera de rango"),
        ("diabetes", 2, "diabetes no binario"),
        ("gender", "X", "gender fuera de {F, M}"),
        ("age", 111, "edad fuera de rango"),
        ("appointment_date", pd.Timestamp("2016-04-29 10:00"), "appointment_date con hora"),
        ("appointment_date", pd.Timestamp("2016-04-20"), "cita antes del agendamiento"),
    ],
)
def test_schema_errors_detect_violations(column: str, value: object, message: str) -> None:
    df, _ = clean(raw_frame(raw_row()))
    df.loc[0, column] = value
    df = df.astype({c: t for c, t in CLEAN_DTYPES.items() if c != "gender"})
    assert any(message in err for err in schema_errors(df))
    with pytest.raises(DataValidationError):
        validate(df)


def test_duplicate_appointment_id_is_invalid() -> None:
    df, _ = clean(raw_frame(raw_row()))
    df = pd.concat([df, df], ignore_index=True)
    assert any("appointment_id duplicado" in err for err in schema_errors(df))


def test_wrong_columns_are_reported() -> None:
    df, _ = clean(raw_frame(raw_row()))
    assert schema_errors(df.drop(columns=["age"]))[0].startswith("Columnas esperadas")
