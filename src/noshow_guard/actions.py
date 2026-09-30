"""Decisión de acción y mensajería simulada.

- ``decide_action``: probabilidad calibrada -> acción, con los umbrales por costo de la Fase 3
  (leídos de ``models/metadata.json``). Función pura: misma probabilidad, misma acción.
- ``MessageSender``: interfaz de envío. Su única implementación, ``MockSender``, registra (y
  opcionalmente muestra) el mensaje que se habría enviado. No hay código de red.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, time
from functools import cache
from importlib import resources
from string import Formatter
from typing import Any, TextIO

import numpy as np

from noshow_guard.evaluation import ACTIONS, Thresholds, assign_actions

NO_ACTION = ACTIONS[0]
ALLOWED_PLACEHOLDERS = frozenset({"fecha", "hora"})
_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MONTHS = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)  # fmt: skip


# --- Decisión ------------------------------------------------------------------


def thresholds_from_metadata(metadata: dict[str, Any]) -> Thresholds:
    """Umbrales guardados por el entrenamiento (no hardcodeados)."""
    th = metadata["thresholds"]
    return Thresholds(standard=float(th["standard"]), reinforced=float(th["reinforced"]))


def decide_action(probability: float, thresholds: Thresholds) -> str:
    """Acción para una probabilidad calibrada; misma regla que ``assign_actions`` en lote."""
    return ACTIONS[int(assign_actions(np.array([probability]), thresholds)[0])]


# --- Plantillas ----------------------------------------------------------------


@cache
def templates() -> dict[str, str]:
    """Plantillas de ``templates/messages.json`` (sin la clave de nota)."""
    raw = resources.files("noshow_guard").joinpath("templates/messages.json").read_text("utf-8")
    loaded: dict[str, str] = json.loads(raw)
    return {k: v for k, v in loaded.items() if not k.startswith("_")}


def placeholders(template: str) -> set[str]:
    """Campos ``{...}`` usados por una plantilla."""
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def spanish_date(day: date) -> str:
    """Fecha en español sin depender del locale del sistema: 'martes 14 de junio de 2016'."""
    return f"{_WEEKDAYS[day.weekday()]} {day.day} de {_MONTHS[day.month - 1]} de {day.year}"


def render_message(action: str, appointment_date: date, appointment_time: time) -> str | None:
    """Texto del mensaje para la acción; ``None`` si la acción es no hacer nada."""
    if action == NO_ACTION:
        return None
    template = templates()[action]
    return template.format(
        fecha=spanish_date(appointment_date), hora=appointment_time.strftime("%H:%M")
    )


# --- Envío simulado ------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    """Mensaje que se habría enviado. Sin destinatario: el simulador no maneja contactos."""

    action: str
    text: str
    appointment_date: date
    appointment_time: time


class MessageSender(ABC):
    """Interfaz de envío de mensajes."""

    @abstractmethod
    def send(self, message: Message) -> None:
        """Entrega (o simula entregar) un mensaje."""


@dataclass
class MockSender(MessageSender):
    """Registra los mensajes en ``outbox`` y, si hay ``stream``, los muestra. Nunca usa la red."""

    stream: TextIO | None = None
    outbox: list[Message] = field(default_factory=list)

    def send(self, message: Message) -> None:
        self.outbox.append(message)
        if self.stream is not None:
            print(f"[MockSender] mensaje simulado, NO enviado: {message.text}", file=self.stream)
