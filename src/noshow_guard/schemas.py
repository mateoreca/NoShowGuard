"""Esquemas Pydantic de entrada y salida del simulador."""

from __future__ import annotations

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from noshow_guard.config import DATA_RULES
from noshow_guard.evaluation import RiskLevel

FeatureValue = int | float | str


class NewPatient(BaseModel):
    """Paciente que no está en el dataset. Solo lo necesario para las features."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=DATA_RULES.age_min, le=DATA_RULES.age_max)
    gender: Literal["F", "M"]
    neighbourhood: str = Field(min_length=1, max_length=80)
    scholarship: bool = False
    hypertension: bool = False
    diabetes: bool = False
    alcoholism: bool = False
    handicap: int = Field(default=0, ge=0, le=DATA_RULES.handicap_max)

    @field_validator("neighbourhood")
    @classmethod
    def _normalize(cls, value: str) -> str:
        """Mismo formato que el dataset: mayúsculas y sin espacios sobrantes."""
        return " ".join(value.split()).upper()


class SimulationRequest(BaseModel):
    """Una cita hipotética: paciente existente o nuevo, fecha, hora y fecha de agendamiento."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str | None = Field(default=None, pattern=r"^\d+$")
    new_patient: NewPatient | None = None
    appointment_date: date
    appointment_time: time
    as_of: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def _check(self) -> SimulationRequest:
        if (self.patient_id is None) == (self.new_patient is None):
            raise ValueError("Indica exactamente uno: patient_id o new_patient.")
        if self.appointment_date < self.as_of:
            raise ValueError(
                f"La cita ({self.appointment_date}) es anterior a as_of ({self.as_of}): "
                "no se agenda en el pasado."
            )
        return self


class Factor(BaseModel):
    """Una contribución SHAP: feature, signo del efecto sobre el riesgo y valor del caso."""

    feature: str
    effect: Literal["+", "-"]
    value: FeatureValue
    shap: float


class SimulationResult(BaseModel):
    """Salida del simulador. Sin probabilidad ni factores cuando la cita está fuera de alcance."""

    status: Literal["ok", "fuera_de_alcance"]
    probability_no_show: float | None = None
    risk_level: RiskLevel | None = None
    lead_time_days: int
    appointment_date: date
    appointment_time: time
    as_of: date
    top_factors: list[Factor] = Field(default_factory=list)
    features: dict[str, FeatureValue] | None = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    model_version: str
