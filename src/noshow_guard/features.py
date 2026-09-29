"""Feature engineering sin fuga y split cronológico.

``build_features(cita, historial, as_of)`` es la única fuente de features: la usan el
entrenamiento (vía ``build_feature_frame``) y, en la Fase 4, el simulador.

Uso: ``python -m noshow_guard.features`` (o ``make features``) escribe los parquet de
train/val/test y ``reports/split_summary.json``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from noshow_guard.config import MIN_LEAD_DAYS, PATHS, SPLIT, SplitConfig
from noshow_guard.data import load_clean

Features = dict[str, int | float | str]

FEATURE_COLUMNS: tuple[str, ...] = (
    "lead_time_days",
    "appointment_weekday",
    "age",
    "is_male",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "handicap",
    "neighbourhood",
    "has_history",
    "prev_appointments",
    "prev_no_shows",
    "prev_no_show_rate",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("neighbourhood",)
META_COLUMNS: tuple[str, ...] = (
    "appointment_id",
    "patient_id",
    "appointment_date",
    "as_of",
    "no_show",
)
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

# Sábado (5) y domingo (6) se agrupan con viernes (4): el único sábado del dataset
# es una fecha con 31 citas y como categoría propia sería un proxy de esa fecha.
_LAST_WEEKDAY = 4


class OutOfScopeError(ValueError):
    """La cita está fuera del alcance del modelo (antelación menor a ``MIN_LEAD_DAYS``)."""


@dataclass(frozen=True)
class Appointment:
    """Cita a evaluar: fecha y atributos del paciente. La hora no existe aquí a propósito."""

    appointment_date: date
    age: int
    gender: str
    neighbourhood: str
    scholarship: int
    hypertension: int
    diabetes: int
    alcoholism: int
    handicap: int


@dataclass(frozen=True)
class PastAppointment:
    """Cita del historial del paciente con su resultado."""

    appointment_date: date
    no_show: int


def lead_time_days(appointment_date: date, as_of: date) -> int:
    """Días calendario entre ``as_of`` y la cita (sin horas: la cita no las tiene)."""
    return (appointment_date - as_of).days


def appointment_weekday(appointment_date: date) -> int:
    """Día de la semana 0 (lunes) a 4 (viernes); fin de semana cuenta como viernes."""
    return min(appointment_date.weekday(), _LAST_WEEKDAY)


def completed_history(historial: Iterable[PastAppointment], as_of: date) -> list[PastAppointment]:
    """Citas terminadas antes de ``as_of``: solo de ellas se conoce el resultado.

    Es estrictamente anterior: una cita el mismo día de ``as_of`` todavía no terminó.
    """
    return [h for h in historial if h.appointment_date < as_of]


def history_features(historial: Iterable[PastAppointment], as_of: date) -> Features:
    """Conteos del historial. Sin historial: conteos y tasa en 0 y ``has_history = 0``."""
    past = completed_history(historial, as_of)
    n_prev = len(past)
    n_no_show = sum(h.no_show for h in past)
    return {
        "has_history": int(n_prev > 0),
        "prev_appointments": n_prev,
        "prev_no_shows": n_no_show,
        "prev_no_show_rate": n_no_show / n_prev if n_prev else 0.0,
    }


def build_features(
    cita: Appointment, historial: Sequence[PastAppointment], as_of: date
) -> Features:
    """Features de una cita vista en la fecha ``as_of``.

    Lanza ``ValueError`` si la cita es anterior a ``as_of`` y ``OutOfScopeError`` si la
    antelación es menor a ``MIN_LEAD_DAYS``. El historial se filtra aquí mismo, así que
    pasarle citas futuras no produce fuga.
    """
    lead = lead_time_days(cita.appointment_date, as_of)
    if lead < 0:
        raise ValueError(f"La cita ({cita.appointment_date}) es anterior a as_of ({as_of}).")
    if lead < MIN_LEAD_DAYS:
        raise OutOfScopeError(f"Antelación de {lead} días: fuera del alcance del modelo.")

    return {
        "lead_time_days": lead,
        "appointment_weekday": appointment_weekday(cita.appointment_date),
        "age": cita.age,
        "is_male": int(cita.gender == "M"),
        "scholarship": cita.scholarship,
        "hypertension": cita.hypertension,
        "diabetes": cita.diabetes,
        "alcoholism": cita.alcoholism,
        "handicap": cita.handicap,
        "neighbourhood": cita.neighbourhood,
        **history_features(historial, as_of),
    }


def appointment_from_row(row: pd.Series) -> Appointment:
    """Convierte una fila del dataset limpio en ``Appointment``."""
    return Appointment(
        appointment_date=row["appointment_date"].date(),
        age=int(row["age"]),
        gender=str(row["gender"]),
        neighbourhood=str(row["neighbourhood"]),
        scholarship=int(row["scholarship"]),
        hypertension=int(row["hypertension"]),
        diabetes=int(row["diabetes"]),
        alcoholism=int(row["alcoholism"]),
        handicap=int(row["handicap"]),
    )


def history_index(history: pd.DataFrame) -> dict[str, list[PastAppointment]]:
    """Agrupa todas las citas conocidas por paciente (el filtro por fecha lo hace cada consulta)."""
    index: dict[str, list[PastAppointment]] = {}
    for pid, day, no_show in zip(
        history["patient_id"], history["appointment_date"], history["no_show"], strict=True
    ):
        index.setdefault(str(pid), []).append(PastAppointment(day.date(), int(no_show)))
    return index


def build_feature_frame(appointments: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Aplica ``build_features`` fila por fila con ``as_of = fecha de agendamiento``.

    Las citas fuera de alcance se descartan. ``history`` debe contener todas las citas
    conocidas (incluidas las del mismo día, que sí son historial válido para citas futuras).
    """
    index = history_index(history)
    features: list[Features] = []
    meta: list[dict[str, object]] = []
    for _, row in appointments.iterrows():
        as_of = row["scheduled_date"].date()
        try:
            feats = build_features(
                appointment_from_row(row), index.get(row["patient_id"], []), as_of
            )
        except OutOfScopeError:
            continue
        features.append(feats)
        meta.append(
            {
                "appointment_id": int(row["appointment_id"]),
                "patient_id": row["patient_id"],
                "appointment_date": row["appointment_date"],
                "as_of": row["scheduled_date"],
                "no_show": int(row["no_show"]),
            }
        )
    return pd.concat(
        [
            pd.DataFrame(meta, columns=list(META_COLUMNS)),
            pd.DataFrame(features, columns=list(FEATURE_COLUMNS)),
        ],
        axis=1,
    )


def assign_split(appointment_date: pd.Series, cfg: SplitConfig = SPLIT) -> pd.Series:
    """Etiqueta train/val/test por fecha de cita; lanza error si una fecha cae entre cortes."""
    day = appointment_date.dt.date
    split = pd.Series(pd.NA, index=appointment_date.index, dtype="object")
    split[day <= cfg.train_end] = "train"
    split[(day >= cfg.val_start) & (day <= cfg.val_end)] = "val"
    split[day >= cfg.test_start] = "test"
    if split.isna().any():
        gaps = sorted(set(day[split.isna()]))
        raise ValueError(f"Fechas de cita fuera de todos los cortes: {gaps}")
    return split


def temporal_split(frame: pd.DataFrame, cfg: SplitConfig = SPLIT) -> dict[str, pd.DataFrame]:
    """Divide el frame de features en train/val/test cronológicos."""
    labels = assign_split(frame["appointment_date"], cfg)
    return {name: frame[labels == name].reset_index(drop=True) for name in SPLIT_NAMES}


def split_summary(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    """Tamaño, tasa, rango de fechas, historial y solapamiento de pacientes con train."""
    train_patients = set(splits["train"]["patient_id"])
    total = sum(len(part) for part in splits.values())
    return {
        name: {
            "rows": len(part),
            "share": round(len(part) / total, 4),
            "no_show_rate": round(float(part["no_show"].mean()), 4),
            "date_min": str(part["appointment_date"].min().date()),
            "date_max": str(part["appointment_date"].max().date()),
            "n_dates": int(part["appointment_date"].nunique()),
            "share_with_history": round(float(part["has_history"].mean()), 4),
            "share_patient_seen_in_train": round(
                float(part["patient_id"].isin(train_patients).mean()), 4
            ),
        }
        for name, part in splits.items()
    }


def main() -> int:
    """Datos limpios -> features sin fuga -> parquet por split + resumen."""
    clean = load_clean()
    splits = temporal_split(build_feature_frame(clean, history=clean))

    PATHS.features_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(PATHS.features_dir / f"{name}.parquet", index=False)

    summary = split_summary(splits)
    PATHS.split_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
