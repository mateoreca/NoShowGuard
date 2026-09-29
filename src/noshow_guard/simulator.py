"""Núcleo del simulador: una cita hipotética -> probabilidad calibrada y explicación.

No agenda nada ni escribe en el dataset. Usa ``build_features`` (el mismo pipeline del
entrenamiento) y el modelo principal indicado en ``models/metadata.json``. Cada simulación
se guarda, marcada como simulada, en el registro SQLite (salvo ``log=False``).

Paciente existente:
- Atributos: los del último registro del paciente agendado a más tardar en ``as_of`` (los
  atributos se conocen al agendar, no hace falta que la cita haya terminado).
- Historial: sus citas con fecha anterior a ``as_of`` (``build_features`` lo filtra).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from noshow_guard.config import MIN_LEAD_DAYS, PATHS, Paths
from noshow_guard.data import load_clean
from noshow_guard.evaluation import risk_level
from noshow_guard.explain import top_factors
from noshow_guard.features import (
    FEATURE_COLUMNS,
    Appointment,
    PastAppointment,
    appointment_from_row,
    build_features,
    history_index,
    lead_time_days,
)
from noshow_guard.model import NoShowModel, load_model
from noshow_guard.registry import Registry
from noshow_guard.schemas import Factor, NewPatient, SimulationRequest, SimulationResult

FIXED_NOTES: tuple[str, ...] = (
    "La hora de la cita no influye en la predicción: el dataset no registra horas de cita.",
    "Modelo entrenado con datos de Brasil (2016), no de esta clínica: probabilidad ilustrativa.",
    "Los factores (SHAP) describen al modelo, no relaciones causales.",
    "Nivel de riesgo relativo: medio desde el percentil 50 y alto desde el 90 de validación.",
)
OUT_OF_SCOPE_NOTE = (
    "Cita con antelación menor a {min_days} día(s): fuera del alcance del modelo. "
    "No se calcula probabilidad ni acción."
)
STALE_AGE_DAYS = 365


class SimulationError(ValueError):
    """La solicitud es válida en forma pero no se puede simular (p. ej. paciente inexistente)."""


@dataclass
class Simulator:
    """Estado cargado una vez: modelo, metadata, registros e índice de historial."""

    model: NoShowModel
    metadata: dict[str, Any]
    records: pd.DataFrame
    history: dict[str, list[PastAppointment]]
    known_neighbourhoods: frozenset[str]
    registry: Registry | None = None

    @classmethod
    def build(
        cls,
        model: NoShowModel,
        metadata: dict[str, Any],
        records: pd.DataFrame,
        registry: Registry | None = None,
    ) -> Simulator:
        """Prepara el simulador a partir de un modelo y del dataset limpio."""
        ordered = records.sort_values(["scheduled_at", "appointment_id"]).reset_index(drop=True)
        return cls(
            model=model,
            metadata=metadata,
            records=ordered,
            history=history_index(ordered),
            known_neighbourhoods=frozenset(model.known_neighbourhoods()),
            registry=registry,
        )

    @classmethod
    def from_disk(cls, paths: Paths = PATHS) -> Simulator:
        """Carga el modelo principal (vía metadata), el dataset limpio y el registro."""
        model, metadata = load_model(paths.model_metadata)
        return cls.build(
            model, metadata, load_clean(paths.clean_parquet), Registry(paths.registry_db)
        )

    # --- Paciente ------------------------------------------------------------

    def existing_patient(
        self, patient_id: str, as_of: date, appointment_date: date
    ) -> tuple[Appointment, list[PastAppointment], list[str]]:
        """Atributos del último registro conocido en ``as_of`` e historial del paciente."""
        own = self.records[self.records["patient_id"] == patient_id]
        if own.empty:
            raise SimulationError(f"El paciente {patient_id} no existe en el dataset.")
        known = own[own["scheduled_date"] <= pd.Timestamp(as_of)]
        if known.empty:
            first = own["scheduled_date"].min().date()
            raise SimulationError(
                f"El paciente {patient_id} no tiene registros en o antes de as_of ({as_of}); "
                f"su primer registro es del {first}."
            )
        last = known.iloc[-1]
        base = appointment_from_row(last)
        cita = Appointment(**{**base.__dict__, "appointment_date": appointment_date})

        warnings = []
        record_day = last["scheduled_date"].date()
        if (as_of - record_day).days > STALE_AGE_DAYS:
            warnings.append(
                f"La edad ({cita.age}) es la del registro del {record_day}; "
                "no se ajusta por el tiempo transcurrido."
            )
        return cita, self.history.get(patient_id, []), warnings

    @staticmethod
    def new_patient(patient: NewPatient, appointment_date: date) -> Appointment:
        """Paciente nuevo: sus datos declarados, sin historial."""
        return Appointment(
            appointment_date=appointment_date,
            age=patient.age,
            gender=patient.gender,
            neighbourhood=patient.neighbourhood,
            scholarship=int(patient.scholarship),
            hypertension=int(patient.hypertension),
            diabetes=int(patient.diabetes),
            alcoholism=int(patient.alcoholism),
            handicap=patient.handicap,
        )

    # --- Advertencias --------------------------------------------------------

    def extrapolation_warnings(self, features: dict[str, Any]) -> list[str]:
        """Avisos cuando el caso cae fuera de lo visto en entrenamiento."""
        ranges = self.metadata["train_ranges"]
        warnings = []
        if features["lead_time_days"] > ranges["lead_time_days_max"]:
            warnings.append(
                f"Antelación de {features['lead_time_days']} días, mayor al máximo visto en "
                f"entrenamiento ({ranges['lead_time_days_max']}): el modelo extrapola."
            )
        if not ranges["age_min"] <= features["age"] <= ranges["age_max"]:
            warnings.append(
                f"Edad {features['age']} fuera del rango de entrenamiento "
                f"({ranges['age_min']}-{ranges['age_max']}): el modelo extrapola."
            )
        if features["neighbourhood"] not in self.known_neighbourhoods:
            warnings.append(
                f"Barrio '{features['neighbourhood']}' no visto en entrenamiento: "
                "se usa la tasa global de no-show para esa variable."
            )
        return warnings

    # --- Simulación ----------------------------------------------------------

    def evaluate(self, request: SimulationRequest) -> SimulationResult:
        """Evalúa una cita hipotética sin agendarla ni registrarla.

        El historial sale solo del dataset real: las simulaciones registradas nunca entran.
        """
        lead = lead_time_days(request.appointment_date, request.as_of)
        common = {
            "lead_time_days": lead,
            "appointment_date": request.appointment_date,
            "appointment_time": request.appointment_time,
            "as_of": request.as_of,
            "model_version": self.metadata["model_version"],
        }
        if lead < MIN_LEAD_DAYS:
            note = OUT_OF_SCOPE_NOTE.format(min_days=MIN_LEAD_DAYS)
            return SimulationResult(status="fuera_de_alcance", notes=[note], **common)

        warnings: list[str] = []
        if request.patient_id is not None:
            cita, history, warnings = self.existing_patient(
                request.patient_id, request.as_of, request.appointment_date
            )
        else:
            assert request.new_patient is not None
            cita, history = self.new_patient(request.new_patient, request.appointment_date), []

        features = build_features(cita, history, request.as_of)
        X = pd.DataFrame([features], columns=list(FEATURE_COLUMNS))
        probability = float(self.model.predict_proba(X)[0])
        factors = top_factors(self.model.contributions(X).iloc[0], X.iloc[0])
        return SimulationResult(
            status="ok",
            probability_no_show=probability,
            risk_level=risk_level(probability, self.metadata["risk_cutoffs"]),
            top_factors=[Factor.model_validate(f) for f in factors],
            features=features,
            warnings=warnings + self.extrapolation_warnings(features),
            notes=list(FIXED_NOTES),
            **common,
        )

    def simulate(self, request: SimulationRequest, log: bool = True) -> SimulationResult:
        """Evalúa la cita y, si hay registro y ``log`` es True, la guarda como simulada."""
        result = self.evaluate(request)
        if log and self.registry is not None:
            self.registry.log(request, result)
        return result


_DEFAULT: Simulator | None = None


def get_simulator() -> Simulator:
    """Simulador por defecto, cargado desde disco una sola vez."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Simulator.from_disk()
    return _DEFAULT


def simulate(
    request: SimulationRequest, simulator: Simulator | None = None, *, log: bool = True
) -> SimulationResult:
    """Punto de entrada: ``simulate(request) -> SimulationResult``. Registra por defecto."""
    return (simulator or get_simulator()).simulate(request, log)
