"""Carga, limpieza y validación de esquema del dataset de citas.

Uso: ``python -m noshow_guard.data`` (o ``make data``) lee ``data/raw/data.csv``,
aplica las reglas de limpieza, valida el esquema y escribe el parquet limpio más
un reporte JSON con cuántas filas eliminó cada regla.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from noshow_guard.config import DATA_RULES, PATHS, RAW_SHA256, DataRules

RAW_COLUMNS: dict[str, str] = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_at",
    "AppointmentDay": "appointment_date",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "SMS_received": "sms_received",
    "No-show": "no_show_raw",
}

BINARY_COLUMNS: tuple[str, ...] = (
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "sms_received",
    "no_show",
)

CLEAN_DTYPES: dict[str, str] = {
    "appointment_id": "int64",
    "patient_id": "object",
    "gender": "object",
    "scheduled_at": "datetime64[ns]",
    "scheduled_date": "datetime64[ns]",
    "appointment_date": "datetime64[ns]",
    "age": "int64",
    "neighbourhood": "object",
    "scholarship": "int8",
    "hypertension": "int8",
    "diabetes": "int8",
    "alcoholism": "int8",
    "handicap": "int8",
    "sms_received": "int8",
    "no_show": "int8",
}


class DataValidationError(ValueError):
    """El DataFrame no cumple el esquema limpio."""


@dataclass
class CleaningReport:
    """Conteo de filas antes y después, y filas eliminadas por cada regla."""

    rows_in: int
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)


def file_sha256(path: Path) -> str:
    """Hash SHA-256 de un archivo, leído por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Lee el CSV crudo. ``PatientId`` se lee como texto para detectar IDs corruptos."""
    return pd.read_csv(path, dtype={"PatientId": str})


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas a snake_case, tipa fechas y define el objetivo explícito.

    ``no_show = 1`` significa que el paciente NO asistió (``No-show == "Yes"``).
    ``scheduled_at`` conserva la hora real; ``appointment_date`` es solo fecha
    porque el dataset no trae la hora de la cita.
    """
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise DataValidationError(f"Faltan columnas en el CSV crudo: {sorted(missing)}")

    df = raw.rename(columns=RAW_COLUMNS)
    scheduled = pd.to_datetime(df["scheduled_at"], utc=True).dt.tz_localize(None)
    appointment = pd.to_datetime(df["appointment_date"], utc=True).dt.tz_localize(None)

    unknown_target = set(df["no_show_raw"].unique()) - {"Yes", "No"}
    if unknown_target:
        raise DataValidationError(f"Valores inesperados en No-show: {sorted(unknown_target)}")

    return df.assign(
        patient_id=df["patient_id"].astype(str).str.strip(),
        scheduled_at=scheduled,
        scheduled_date=scheduled.dt.normalize(),
        appointment_date=appointment.dt.normalize(),
        no_show=(df["no_show_raw"] == "Yes").astype("int8"),
    ).drop(columns=["no_show_raw"])


def _drop(df: pd.DataFrame, mask: pd.Series, rule: str, report: CleaningReport) -> pd.DataFrame:
    """Elimina las filas donde ``mask`` es True y registra cuántas fueron."""
    report.dropped[rule] = int(mask.sum())
    return df.loc[~mask]


def clean(raw: pd.DataFrame, rules: DataRules = DATA_RULES) -> tuple[pd.DataFrame, CleaningReport]:
    """Aplica las reglas de limpieza en orden y devuelve el DataFrame limpio y el reporte."""
    report = CleaningReport(rows_in=len(raw))
    df = standardize(raw)

    df = _drop(df, ~df["patient_id"].str.fullmatch(r"\d+"), "patient_id_corrupto", report)
    df = _drop(df, ~df["age"].between(rules.age_min, rules.age_max), "edad_fuera_de_rango", report)
    df = _drop(
        df,
        df["appointment_date"] < df["scheduled_date"],
        "cita_antes_de_agendamiento",
        report,
    )

    df = (
        df.astype({col: dtype for col, dtype in CLEAN_DTYPES.items() if col in df.columns})
        .loc[:, list(CLEAN_DTYPES)]
        .sort_values(["appointment_date", "scheduled_at", "appointment_id"])
        .reset_index(drop=True)
    )
    report.rows_out = len(df)
    return df, report


def schema_errors(df: pd.DataFrame, rules: DataRules = DATA_RULES) -> list[str]:
    """Lista de violaciones del esquema limpio (vacía si el DataFrame es válido)."""
    if list(df.columns) != list(CLEAN_DTYPES):
        return [f"Columnas esperadas {list(CLEAN_DTYPES)}, recibidas {list(df.columns)}"]

    errors = [
        f"{col}: dtype {df[col].dtype}, esperado {dtype}"
        for col, dtype in CLEAN_DTYPES.items()
        if str(df[col].dtype) != dtype
    ]
    checks: dict[str, pd.Series] = {
        "hay valores nulos": df.isna().any(axis=1),
        "appointment_id duplicado": df["appointment_id"].duplicated(),
        "patient_id no numérico": ~df["patient_id"].astype(str).str.fullmatch(r"\d+"),
        "gender fuera de {F, M}": ~df["gender"].isin(["F", "M"]),
        "edad fuera de rango": ~df["age"].between(rules.age_min, rules.age_max),
        "handicap fuera de rango": ~df["handicap"].between(0, rules.handicap_max),
        "appointment_date con hora": df["appointment_date"]
        != df["appointment_date"].dt.normalize(),
        "cita antes del agendamiento": df["appointment_date"] < df["scheduled_date"],
    }
    for col in BINARY_COLUMNS:
        checks[f"{col} no binario"] = ~df[col].isin([0, 1])

    errors += [f"{name}: {int(mask.sum())} filas" for name, mask in checks.items() if mask.any()]
    return errors


def validate(df: pd.DataFrame, rules: DataRules = DATA_RULES) -> pd.DataFrame:
    """Devuelve el DataFrame si cumple el esquema; si no, lanza ``DataValidationError``."""
    errors = schema_errors(df, rules)
    if errors:
        raise DataValidationError("Esquema inválido:\n- " + "\n- ".join(errors))
    return df


def load_clean(path: Path = PATHS.clean_parquet) -> pd.DataFrame:
    """Lee el parquet limpio y lo valida."""
    return validate(pd.read_parquet(path))


def main() -> int:
    """Pipeline completo: CSV crudo -> parquet limpio + reporte de limpieza."""
    if not PATHS.raw_csv.exists():
        print(
            f"No existe {PATHS.raw_csv}. Ejecuta: python scripts/download_data.py", file=sys.stderr
        )
        return 1

    raw_hash = file_sha256(PATHS.raw_csv)
    if raw_hash != RAW_SHA256:
        print(f"Advertencia: el hash del CSV ({raw_hash}) no coincide con el esperado.")

    df, report = clean(load_raw(PATHS.raw_csv))
    validate(df)

    PATHS.clean_parquet.parent.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PATHS.clean_parquet, index=False)
    PATHS.cleaning_report.write_text(
        json.dumps({"raw_sha256": raw_hash, **asdict(report)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    print(f"Datos limpios en {PATHS.clean_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
