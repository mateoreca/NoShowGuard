"""Núcleo del simulador: una cita hipotética -> probabilidad calibrada y explicación.

No agenda nada ni escribe en el dataset. Usa ``build_features`` (el mismo pipeline del
entrenamiento) y el modelo principal indicado en ``models/metadata.json``. Cada simulación
decide una acción, "envía" su mensaje con ``MockSender`` (sin red) y se guarda, marcada como
simulada, en el registro SQLite (salvo ``log=False``).

Paciente existente:
- Atributos: los del último registro del paciente agendado a más tardar en ``as_of`` (los
  atributos se conocen al agendar, no hace falta que la cita haya terminado).
- Historial: sus citas con fecha anterior a ``as_of`` (``build_features`` lo filtra).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from noshow_guard.actions import (
    Message,
    MessageSender,
    MockSender,
    decide_action,
    render_message,
    thresholds_from_metadata,
)
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
from noshow_guard.schemas import (
    Factor,
    NewPatient,
    SimulationRequest,
    SimulationResult,
    WhatIfResult,
    WhatIfRow,
)

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
DEFAULT_LEAD_TIMES: tuple[int, ...] = (1, 3, 7, 14)
WHAT_IF_NOTES: tuple[str, ...] = (
    "Muestra la asociación que aprendió el modelo entre antelación y no-show, NO un efecto "
    "causal: agendar con menos antelación no garantiza ese cambio en la inasistencia.",
    "Para un paciente existente, mover as_of también cambia su historial disponible.",
    "La hora de la cita no influye en la predicción.",
)


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
    sender: MessageSender | None = None

    @classmethod
    def build(
        cls,
        model: NoShowModel,
        metadata: dict[str, Any],
        records: pd.DataFrame,
        registry: Registry | None = None,
        sender: MessageSender | None = None,
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
            sender=sender,
        )

    @classmethod
    def from_disk(cls, paths: Paths = PATHS, sender: MessageSender | None = None) -> Simulator:
        """Carga el modelo principal (vía metadata), el dataset limpio y el registro."""
        model, metadata = load_model(paths.model_metadata)
        return cls.build(
            model,
            metadata,
            load_clean(paths.clean_parquet),
            Registry(paths.registry_db),
            sender or MockSender(),
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
        """Evalúa una cita hipotética sin agendarla, registrarla ni enviar mensajes.

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
        action = decide_action(probability, thresholds_from_metadata(self.metadata))
        return SimulationResult(
            status="ok",
            probability_no_show=probability,
            risk_level=risk_level(probability, self.metadata["risk_cutoffs"]),
            action=action,
            message_preview=render_message(
                action, request.appointment_date, request.appointment_time
            ),
            top_factors=[Factor.model_validate(f) for f in factors],
            features=features,
            warnings=warnings + self.extrapolation_warnings(features),
            notes=list(FIXED_NOTES),
            **common,
        )

    def simulate(self, request: SimulationRequest, log: bool = True) -> SimulationResult:
        """Evalúa la cita, "envía" el mensaje con el sender simulado y la registra.

        El registro solo ocurre si hay registro configurado y ``log`` es True.
        """
        result = self.evaluate(request)
        if self.sender is not None and result.action and result.message_preview:
            self.sender.send(
                Message(
                    action=result.action,
                    text=result.message_preview,
                    appointment_date=result.appointment_date,
                    appointment_time=result.appointment_time,
                )
            )
        if log and self.registry is not None:
            self.registry.log(request, result)
        return result

    def what_if(
        self, request: SimulationRequest, lead_times: Sequence[int] = DEFAULT_LEAD_TIMES
    ) -> WhatIfResult:
        """Repite la misma cita con otras antelaciones, moviendo ``as_of`` hacia atrás.

        No registra ni envía nada. Para un paciente existente, mover ``as_of`` también cambia
        el historial disponible (hay menos citas terminadas antes).
        """
        if any(lead < 0 for lead in lead_times):
            raise ValueError("Las antelaciones deben ser >= 0.")
        rows = []
        for lead in sorted(set(lead_times)):
            as_of = request.appointment_date - timedelta(days=lead)
            try:
                result = self.evaluate(request.model_copy(update={"as_of": as_of}))
            except SimulationError as exc:
                rows.append(
                    WhatIfRow(lead_time_days=lead, as_of=as_of, status="error", detail=str(exc))
                )
                continue
            rows.append(
                WhatIfRow(
                    lead_time_days=lead,
                    as_of=as_of,
                    status=result.status,
                    probability_no_show=result.probability_no_show,
                    risk_level=result.risk_level,
                    action=result.action,
                )
            )
        return WhatIfResult(
            appointment_date=request.appointment_date,
            appointment_time=request.appointment_time,
            rows=rows,
            notes=list(WHAT_IF_NOTES),
            model_version=self.metadata["model_version"],
        )


_DEFAULT: Simulator | None = None


def get_simulator() -> Simulator:
    """Simulador por defecto (modelo, datos y registro en disco; mensajes a stderr)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Simulator.from_disk(sender=MockSender(stream=sys.stderr))
    return _DEFAULT


def simulate(
    request: SimulationRequest, simulator: Simulator | None = None, *, log: bool = True
) -> SimulationResult:
    """Punto de entrada: ``simulate(request) -> SimulationResult``. Registra por defecto."""
    return (simulator or get_simulator()).simulate(request, log)


def what_if(
    request: SimulationRequest,
    lead_times: Sequence[int] = DEFAULT_LEAD_TIMES,
    simulator: Simulator | None = None,
) -> WhatIfResult:
    """``what_if(request, lead_times=[1, 3, 7, 14])``: antelación -> probabilidad -> acción."""
    return (simulator or get_simulator()).what_if(request, lead_times)
